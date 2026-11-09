"""Decide whether selected species IDs are supported by the model.

Retrieval candidates remain visible when their exact biological identity is
unresolved. This decision precedes, and is independent of, qualifier choice.
"""

import html
import json
import logging
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

import libsbml
import pandas as pd

from core.llm_interface import query_llm


logger = logging.getLogger(__name__)
_DEGRADED = re.compile(r"(?:^|[^a-z0-9])degrad(?:ed|ation)(?:$|[^a-z0-9])", re.I)
_ISOFORM = re.compile(r"\b(?:isoforms?|splice variants?|transcript variants?)\b", re.I)
_ISOFORM_NUMBER = re.compile(r"\bisoform[-_ ]*(?:[0-9]+|alpha|beta|gamma|delta)\b", re.I)
_ABSTRACT = re.compile(r"(?:^|[^a-z0-9])(?:activity|capacity|burden|damage|response)(?:$|[^a-z0-9])", re.I)
_SUBUNIT_SUFFIX = re.compile(r"[A-Z][A-Z0-9]{2,}[A-D][1-9]$", re.I)


def _text(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _plain_notes(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value or "")).split())


def _brief(value: str, limit: int = 220) -> str:
    return " ".join(str(value or "").split())[:limit].rstrip()


def _model_context(model_file: Optional[str]) -> Dict[str, Any]:
    if not model_file:
        return {"model_notes": "", "species": {}, "reactions": {}}
    document = libsbml.readSBMLFromFile(model_file)
    model = document.getModel() if document else None
    if model is None:
        raise ValueError(f"Could not load SBML model: {model_file}")
    species = {
        item.getId(): {
            "name": item.getName() if item.isSetName() else "",
            "notes": _plain_notes(item.getNotesString()) if item.isSetNotes() else "",
        }
        for item in model.getListOfSpecies()
    }
    reactions: Dict[str, list[dict[str, Any]]] = defaultdict(list)
    for reaction in model.getListOfReactions():
        participants = {
            "reactants": [ref.getSpecies() for ref in reaction.getListOfReactants()],
            "products": [ref.getSpecies() for ref in reaction.getListOfProducts()],
            "modifiers": [ref.getSpecies() for ref in reaction.getListOfModifiers()],
        }
        for role, ids in participants.items():
            for species_id in ids:
                reactions[species_id].append({
                    "id": reaction.getId(),
                    "name": reaction.getName() if reaction.isSetName() else "",
                    "role": role[:-1],
                    "reactants": participants["reactants"],
                    "products": participants["products"],
                    "modifiers": participants["modifiers"],
                    "notes": _plain_notes(reaction.getNotesString())[:500]
                    if reaction.isSetNotes() else "",
                })
    return {
        "model_notes": _plain_notes(model.getNotesString()) if model.isSetNotes() else "",
        "species": species,
        "reactions": dict(reactions),
    }


def _mentions_alias(source: str, alias: str) -> bool:
    alias = alias.strip()
    if len(alias) < 3:
        return False
    return bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])", source, re.I))


def _relevant_model_notes(notes: str, species_id: str, display: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", notes)
    return " ".join(
        sentence for sentence in sentences
        if _mentions_alias(sentence, species_id) or _mentions_alias(sentence, display)
    )


def _component_is_explicit(rows: pd.DataFrame, source: str) -> bool:
    first = rows.iloc[0]
    aliases = [part.strip() for part in _text(first.get("component_name", "")).split(",")]
    aliases.extend([_text(first.get("annotation_label", "")), _text(first.get("identity", ""))])
    # Full generated phrases may contain the parent complex name. Require a
    # distinct component alias in the original SBML label or its notes.
    return any(_mentions_alias(source, alias) for alias in aliases if alias)


def _associated_component_source(species_id: str, context: Dict[str, Any]) -> str:
    """Names of partners explicitly assembled into or released from a complex."""
    partners: set[str] = set()
    for reaction in context.get("reactions", {}).get(species_id, []):
        if reaction["role"] == "product":
            partners.update(reaction["reactants"])
        elif reaction["role"] == "reactant":
            partners.update(reaction["products"])
    partners.discard(species_id)
    source = []
    for partner in sorted(partners):
        source.append(partner)
        source.append(context.get("species", {}).get(partner, {}).get("name", ""))
    return " ".join(source)


def _terminal_degradation(species_id: str, context: Dict[str, Any]) -> Optional[str]:
    reactions = context.get("reactions", {}).get(species_id, [])
    if not reactions or any(item["role"] != "product" for item in reactions):
        return None
    degradation = [item for item in reactions if
                   re.search(r"degrad", f"{item['id']} {item['name']}", re.I)]
    return degradation[0]["id"] if degradation else None


def _response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "aaaim_identity_support",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "units": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "unit_id": {"type": "string"},
                                "status": {"type": "string", "enum": ["supported", "unresolved"]},
                                "reason": {"type": "string"},
                                "note": {"type": "string"},
                            },
                            "required": ["unit_id", "status", "reason", "note"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["units"],
                "additionalProperties": False,
            },
        },
    }


def _parse_response(raw: str, expected: set[str]) -> Dict[str, Dict[str, str]]:
    data = json.loads(raw)
    units = data["units"]
    if not isinstance(units, list):
        raise ValueError("units must be an array")
    decisions: Dict[str, Dict[str, str]] = {}
    for item in units:
        unit_id = item["unit_id"]
        if unit_id not in expected or unit_id in decisions or item["status"] not in {"supported", "unresolved"}:
            raise ValueError("unexpected identity-support unit")
        if not isinstance(item["reason"], str) or not isinstance(item["note"], str):
            raise ValueError("reason and note must be strings")
        decisions[unit_id] = item
    if set(decisions) != expected:
        raise ValueError("missing identity-support decisions")
    return decisions


def assess_species_identity_support(
    recommendations_df: pd.DataFrame,
    *,
    model_file: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    effort: str = "low",
    max_completion_tokens: Optional[int] = None,
    usage_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> pd.DataFrame:
    """Mark selected species candidates as supported or unresolved.

    Clear name matches use the low-cost path. Ambiguous labels use original
    model/species notes and structured reaction roles when ``model_file`` is
    available. ``effort='high'`` also checks clear matches with that context.
    The selected IDs are never replaced or deleted from the table.
    """
    if effort not in {"low", "high"}:
        raise ValueError("effort must be 'low' or 'high'")
    out = recommendations_df.copy().reset_index(drop=True)
    out["identity_status"] = "not_applicable"
    out["identity_reason"] = ""
    out["Note"] = ""
    if out.empty:
        return out
    context = _model_context(model_file)
    pending: list[dict[str, Any]] = []

    def apply(indexes: list[int], status: str, reason: str, note: str = "") -> None:
        for index in indexes:
            out.at[index, "identity_status"] = status
            out.at[index, "identity_reason"] = _brief(reason)
            out.at[index, "Note"] = _brief(note) if status == "unresolved" else ""
            if status == "unresolved":
                out.at[index, "update_annotation"] = "ignore"
                out.at[index, "qualifier"] = ""

    groups: Dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in out.iterrows():
        species_id = _text(row.get("id", ""))
        if species_id and species_id != "Reason:":
            identity = _text(row.get("identity", "")) or _text(row.get("annotation", ""))
            groups[(species_id, _text(row.get("component_id", "")), identity)].append(index)
    base_counts: Dict[tuple[str, str], int] = defaultdict(int)
    for species_id, component_id, _identity in groups:
        base_counts[(species_id, component_id)] += 1

    for (species_id, component_id, identity), indexes in groups.items():
        rows = out.loc[indexes]
        first = rows.iloc[0]
        detail = context.get("species", {}).get(species_id, {})
        display = detail.get("name", "") or _text(first.get("display_name", "")) or species_id
        entity_type = _text(first.get("type", ""))
        source = f"{species_id} {display}"
        relevant_notes = _relevant_model_notes(context.get("model_notes", ""), species_id, display)
        evidence = " ".join([source, detail.get("notes", ""), relevant_notes])
        specific_evidence = " ".join([source, detail.get("notes", "")])
        candidates = rows[rows["annotation"].fillna("").astype(str).str.contains(":", regex=False)]

        if candidates.empty:
            if entity_type.lower() == "unknown" or _ABSTRACT.search(source):
                if re.search(r"damage", source, re.I):
                    note = f"{species_id} represents a modeled damage burden, not a molecular species."
                else:
                    note = f"{species_id} represents an abstract model state, not a molecular species."
            else:
                note = f"The molecular identity of {species_id} is unresolved in this model."
            apply(indexes, "unresolved", "No supported molecular identifier was selected.", note)
            continue
        if all(_text(value).startswith("original") for value in rows.get("status", pd.Series("", index=rows.index))):
            apply(indexes, "supported", "Existing annotation retained.")
            continue

        terminal_reaction = _terminal_degradation(species_id, context)
        if terminal_reaction:
            apply(indexes, "unresolved", "Terminal degradation product; no defined protein identity.",
                  f"{species_id} is a terminal product of {terminal_reaction} and represents degradation bookkeeping.")
            continue
        if _DEGRADED.search(source):
            apply(indexes, "unresolved", "Degraded label alone does not establish a molecular identity.",
                  f"{species_id} denotes a degraded state; its molecular identity is not specified.")
            continue
        if _ABSTRACT.search(source):
            apply(indexes, "unresolved", "Functional or burden variable, not an exact molecular identity.",
                  f"{species_id} is a modeled activity or burden variable; no exact molecular identity is assigned.")
            continue

        labels = [_text(value) for value in candidates.get("annotation_label", pd.Series(dtype=str))]
        if any(_ISOFORM.search(label) for label in labels) and not _ISOFORM_NUMBER.search(specific_evidence):
            apply(indexes, "unresolved", "Candidate resolves isoforms that the model does not distinguish.",
                  f"{species_id} is modeled without a specified isoform.")
            continue
        if entity_type.lower() == "complex" and component_id:
            component_evidence = f"{evidence} {_associated_component_source(species_id, context)}"
            if _component_is_explicit(rows, component_evidence):
                apply(indexes, "supported", "Component alias appears in the model source or notes.")
            else:
                apply(indexes, "unresolved", "Proposed component is not specified by the model.",
                      f"The exact subunit identities of {species_id} are not fully specified.")
            continue
        specific_subunit = [label for label in labels if _SUBUNIT_SUFFIX.fullmatch(label)]
        if specific_subunit and not any(_mentions_alias(specific_evidence, label) for label in specific_subunit):
            apply(indexes, "unresolved", "Subunit-specific candidate lacks model-side evidence.",
                  f"{species_id} is modeled without a specified subunit identity.")
            continue

        exact_label = bool(labels) and all(label and _mentions_alias(source, label) for label in labels)
        if exact_label and effort == "low":
            apply(indexes, "supported", "Selected label matches the original SBML species label.")
            continue
        unit_id = f"{species_id}|{component_id}" if component_id else species_id
        if base_counts[(species_id, component_id)] > 1:
            unit_id += f"|{identity}"
        item = {
            "unit_id": unit_id,
            "species_id": species_id,
            "species_name": display,
            "entity_type": entity_type,
            "selected_identifiers": [
                {"id": _text(row.get("annotation", "")), "name": _text(row.get("annotation_label", ""))}
                for _, row in candidates.iterrows()
            ],
            "indexes": indexes,
        }
        if model_file:
            item["species_notes"] = detail.get("notes", "")[:1200]
            item["model_notes"] = context.get("model_notes", "")[:1200]
            item["reactions"] = context.get("reactions", {}).get(species_id, [])[:12]
        pending.append(item)

    for start in range(0, len(pending), 20):
        batch = pending[start:start + 20]
        expected = {item["unit_id"] for item in batch}
        prompt = (
            "Decide whether each already-selected database identifier is supported as an "
            "identity of the original SBML species. Do not choose a qualifier or replace IDs. "
            "Use supported when the identifier names the underlying molecular entity "
            "represented by the species. An active, inactive, or modified species may be "
            "supported by a general protein entry; the separate qualifier stage will use "
            "isVersionOf. Do not require UniProt to encode the precise activation or "
            "modification state. "
            "Use unresolved for unspecified subunits, splice isoforms, inferred complex "
            "composition, abstract variables, or degradation bookkeeping products. A name "
            "match score or selected URI is not biological evidence. Prefer unresolved "
            "when the source is insufficient. For unresolved, write one short SBML note "
            "describing the model variable, without describing the pipeline.\nRows:\n"
            + json.dumps([{key: value for key, value in item.items() if key != "indexes"}
                          for item in batch], ensure_ascii=False)
        )
        decisions: Dict[str, Dict[str, str]] = {}
        for attempt in range(2):
            try:
                raw = query_llm(
                    prompt + ("\nReturn every unit exactly once." if attempt else ""),
                    developer_prompt="You assess model-side support for selected biological identities.",
                    model=llm_model,
                    response_format=_response_format(),
                    max_completion_tokens=max_completion_tokens,
                    usage_callback=usage_callback,
                )
                decisions = _parse_response(raw, expected)
                break
            except Exception as exc:
                logger.warning("Identity-support prediction failed: %s", exc)
        for item in batch:
            decision = decisions.get(item["unit_id"])
            if decision is None:
                apply(item["indexes"], "unresolved", "Identity-support response unavailable.",
                      f"The exact molecular identity of {item['species_id']} is unresolved in this model.")
            else:
                note = decision["note"] or f"The exact molecular identity of {item['species_id']} is unresolved in this model."
                apply(item["indexes"], decision["status"], decision["reason"], note)
    return out
