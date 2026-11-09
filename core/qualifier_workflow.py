"""Predict biological qualifiers after species identity selection.

This stage only changes the relationship between an SBML species and an
already-selected identifier. It never retrieves or replaces identifiers.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, Optional

import pandas as pd

from core.llm_interface import query_llm
from core.model_info import extract_model_info
from core.update_model import annotation_resource


logger = logging.getLogger(__name__)
QUALIFIERS = {"is", "isVersionOf", "hasPart"}
_DEGRADED = re.compile(r"(?:^|[^a-z0-9])(?:degraded|degradation)(?:$|[^a-z0-9])", re.I)
_STATE = re.compile(
    r"(?:^|[^a-z0-9])(?:active|inactive|activated|inactivated|phospho|"
    r"phosphorylated|dephosphorylated|ubiquitinated|acetylated|methylated|"
    r"cleaved|oxidized)(?:$|[^a-z0-9])", re.I,
)


def _text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _qualifier_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "aaaim_species_qualifiers",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "row_id": {"type": "string"},
                                "qualifier": {"type": "string", "enum": sorted(QUALIFIERS)},
                                "reason": {"type": "string"},
                            },
                            "required": ["row_id", "qualifier", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["rows"],
                "additionalProperties": False,
            },
        },
    }


def _parse_decisions(raw: str, expected: set[str]) -> Dict[str, Dict[str, str]]:
    data = json.loads(raw)
    rows = data["rows"]
    if not isinstance(rows, list):
        raise ValueError("rows must be an array")
    decisions: Dict[str, Dict[str, str]] = {}
    for item in rows:
        key = item["row_id"]
        qualifier = item["qualifier"]
        reason = item["reason"]
        if key not in expected or key in decisions or qualifier not in QUALIFIERS:
            raise ValueError("unexpected row or qualifier")
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")
        decisions[key] = {"qualifier": qualifier, "reason": reason[:300]}
    if set(decisions) != expected:
        raise ValueError("missing qualifier decisions")
    return decisions


def _relevant_reactions(species_id: str, reactions: list[str]) -> list[str]:
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(species_id) + r"(?![A-Za-z0-9_])")
    return [str(reaction) for reaction in reactions if pattern.search(str(reaction))]


def _context_supports_state(item: Dict[str, Any], model_info: Dict[str, Any]) -> bool:
    """Require a state cue near the modeled species in supplied model context."""
    context = "\n".join([str(model_info.get("model_notes", "")), *item.get("reactions", [])])
    species_id = item["species_id"]
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(species_id) + r"(?![A-Za-z0-9_])", re.I)
    return any(
        _STATE.search(context[max(0, match.start() - 40):match.end() + 40])
        for match in pattern.finditer(context)
    )


def predict_species_qualifiers(
    recommendations_df: pd.DataFrame,
    *,
    model_file: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    effort: str = "low",
    model_info: Optional[Dict[str, Any]] = None,
    max_completion_tokens: Optional[int] = None,
    usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> pd.DataFrame:
    """Classify selected species IDs as is, isVersionOf, or hasPart.

    Low effort supplies only each SBML species ID/name/type and selected
    identifier URI/label. High effort also supplies model notes and reactions
    involving that species. Existing qualifiers are retained. Rows marked
    unresolved by the identity-support stage receive no qualifier or URI in
    SBML. Supported IDs use is by default; a more specific qualifier requires
    evidence from the modeled species.
    """
    if effort not in {"low", "high"}:
        raise ValueError("effort must be 'low' or 'high'")
    if recommendations_df.empty:
        return recommendations_df.copy()
    if effort == "high" and model_info is None:
        if not model_file:
            raise ValueError("model_file or model_info is required for effort='high'")
        species_ids = [str(value) for value in recommendations_df["id"].dropna().unique()
                       if str(value) != "Reason:"]
        model_info = extract_model_info(model_file, species_ids, "auto")
    model_info = model_info or {}

    out = recommendations_df.copy().reset_index(drop=True)
    out["qualifier_rule"] = ""
    out["qualifier_reason"] = ""
    out["qualifier_status"] = "not_applicable"
    pending: list[dict[str, Any]] = []
    grouped_rows: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}

    for index, row in out.iterrows():
        if _text(row.get("identity_status", "")).lower() == "unresolved":
            out.at[index, "qualifier"] = ""
            out.at[index, "qualifier_rule"] = "identity_unresolved"
            out.at[index, "qualifier_status"] = "abstain"
            continue
        annotation = _text(row.get("annotation", ""))
        if (not annotation or ":" not in annotation or
                not annotation.split(":", 1)[1].strip() or
                _text(row.get("id", "")) == "Reason:"):
            out.at[index, "qualifier"] = ""
            continue
        status = _text(row.get("status", ""))
        if status.startswith("original") and _text(row.get("qualifier", "")):
            out.at[index, "qualifier_rule"] = "existing"
            out.at[index, "qualifier_status"] = "retained"
            continue

        species_id = _text(row.get("id", ""))
        species_name = _text(row.get("display_name", "")) or species_id
        entity_type = _text(row.get("type", ""))
        state_text = f"{species_id} {species_name}"
        rule = ""
        qualifier = "is"
        if entity_type.lower() == "complex":
            if _text(row.get("component_id", "")):
                rule, qualifier = "complex_component", "hasPart"
            else:
                rule = "default_is"
        elif _DEGRADED.search(state_text):
            rule = "explicit_degraded_state"
        elif _STATE.search(state_text):
            rule, qualifier = "explicit_state", "isVersionOf"

        if rule:
            out.at[index, "qualifier_rule"] = rule
            out.at[index, "qualifier_status"] = "predicted"
            out.at[index, "qualifier"] = qualifier
            if rule == "explicit_degraded_state":
                out.at[index, "qualifier_reason"] = "Explicit degraded state in the species label; assigned is."
            continue

        # Identity ranking can retain several tied accessions for one
        # biological identity. Their relationship to the SBML species is the
        # same task; qualifier prediction must not adjudicate that tie.
        identity = _text(row.get("identity", "")) or annotation
        group_key = (
            species_id, _text(row.get("component_id", "")), identity.upper(),
            annotation.split(":", 1)[0].upper(),
        )
        selected = {
            "id": annotation,
            "uri": annotation_resource(annotation),
            "name": _text(row.get("annotation_label", "")),
        }
        if group_key in grouped_rows:
            grouped_rows[group_key]["selected_identifiers"].append(selected)
            grouped_rows[group_key]["indexes"].append(index)
            continue
        item = {
            "row_id": str(index),
            "species_id": species_id,
            "species_name": species_name,
            "entity_type": entity_type,
            "selected_identifiers": [selected],
            "indexes": [index],
        }
        if effort == "high":
            item["reactions"] = _relevant_reactions(
                species_id, model_info.get("reactions", []),
            )
        grouped_rows[group_key] = item
        pending.append(item)

    for start in range(0, len(pending), 30):
        batch = pending[start:start + 30]
        expected = {item["row_id"] for item in batch}
        prompt = (
            "Choose the biological qualifier relating each SBML species to its already "
            "selected identifier. Do not change the identifier. Default to is when a URI "
            "is selected. Use isVersionOf only when the modeled species explicitly denotes "
            "a distinct state or form; do not infer this from the selected identifier's "
            "name or isoform label. Use hasPart only when the modeled species is a complex "
            "and the identifier names one of its components. A row may contain several "
            "already-tied accession IDs for one biological identity. Give that row one "
            "qualifier; do not use a tie to resolve identity or as evidence to abstain. "
            "Keep reasons brief.\n"
        )
        if effort == "high":
            prompt += f"Model notes: {model_info.get('model_notes', '')}\n"
        prompt += "Rows:\n" + json.dumps(
            [{key: value for key, value in item.items() if key != "indexes"}
             for item in batch], ensure_ascii=False,
        )
        decisions: Dict[str, Dict[str, str]] = {}
        error = ""
        for attempt in range(2):
            try:
                raw = query_llm(
                    prompt + (f"\nPrevious response was invalid: {error}. Return each row exactly once."
                              if attempt else ""),
                    developer_prompt="You assign SBML biological qualifiers to already selected identifiers.",
                    model=llm_model,
                    response_format=_qualifier_response_format(),
                    max_completion_tokens=max_completion_tokens,
                    usage_callback=usage_callback,
                )
                decisions = _parse_decisions(raw, expected)
                break
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
                logger.warning("Invalid qualifier prediction: %s", error)
            except Exception as exc:
                error = str(exc)
                logger.warning("Qualifier request failed: %s", error)
                break

        for item in batch:
            decision = decisions.get(item["row_id"], {"qualifier": "is", "reason": error})
            qualifier = decision["qualifier"]
            rule = "llm"
            reason = decision["reason"]
            # A database isoform label alone does not establish that the model
            # distinguishes that isoform or any other state of this species.
            # Explicit state and complex-component cases were handled above.
            if qualifier == "isVersionOf" and effort == "high" and _context_supports_state(item, model_info):
                rule = "llm_context_state"
            elif qualifier != "is":
                rule = "default_is"
                reason = ("No explicit model-side state evidence; defaulted to is."
                          if qualifier == "isVersionOf" else
                          "No modeled complex component relation; defaulted to is.")
                qualifier = "is"
            elif not decisions:
                rule = "default_is"
                reason = f"Qualifier response unavailable; defaulted to is. {reason}".strip()
            for index in item["indexes"]:
                out.at[index, "qualifier_rule"] = rule
                out.at[index, "qualifier_reason"] = reason[:300]
                out.at[index, "qualifier_status"] = "predicted"
                out.at[index, "qualifier"] = qualifier
    return out
