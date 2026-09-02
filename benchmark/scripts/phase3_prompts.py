"""Provider-independent Phase 3 prompt and context builders.

No API calls. Context variants are bounded and scanned for KEGG reaction-id leakage.
Ground-truth identifiers are never included in the payload; the answer key is a
separate artifact.

Usage::

    python benchmark/scripts/phase3_prompts.py
    python benchmark/scripts/phase3_prompts.py --write
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.scripts.phase3_common import (
    EVIDENCE_CSV,
    OUT_PILOT,
    OUT_PROMPTS,
    OUT_SPECIES_NAMES,
    PROMPT_TEMPLATE_VERSION,
    TOKENIZER_SCAFFOLD,
    assert_no_kegg_leakage,
    estimate_tokens,
    load_evaluable_corpus,
    parse_participant_ids,
    redact_kegg_reaction_ids,
    write_jsonl,
)
from benchmark.scripts.sample_phase3_pilot import build_pilot

logger = logging.getLogger("phase3_prompts")

CONTEXT_VARIANTS = ("target_only", "target_plus_model", "target_plus_neighborhood")
DEFAULT_NEIGHBORHOOD = 4
MAX_MODEL_NOTES_CHARS = 280
MAX_NEIGHBOR_EQ_CHARS = 160

OPEN_SET_INSTRUCTIONS = (
    "Identify the KEGG reaction that this SBML reaction most likely represents. "
    "You may propose up to three KEGG reaction identifiers (R followed by five digits), "
    "ordered from most to least confident, or abstain if the evidence is insufficient. "
    "Do not invent identifiers that you cannot support. Return JSON only."
)

STRUCTURED_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["abstain", "predictions", "rationale", "basis"],
    "properties": {
        "abstain": {
            "type": "boolean",
            "description": "True if evidence is insufficient to propose any KEGG reaction.",
        },
        "predictions": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["kegg_id", "confidence"],
                "properties": {
                    "kegg_id": {"type": "string", "pattern": "^R[0-9]{5}$"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "rationale": {"type": "string"},
        "basis": {
            "type": "string",
            "enum": ["recalled_knowledge", "supplied_evidence", "mixed"],
            "description": "Self-report only; not treated as proof of evidence use.",
        },
    },
}


def _species_index(model_id: str, evidence: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    sub = evidence[evidence.model_id == model_id]
    out: Dict[str, Dict[str, List[str]]] = {}
    for rec in sub.itertuples(index=False):
        sid = str(rec.species_id)
        entry = out.setdefault(sid, {"chebi": [], "kegg_compound": []})
        ann = str(rec.annotation)
        kind = str(rec.annotation_type)
        if kind == "chebi" and ann not in entry["chebi"]:
            entry["chebi"].append(ann)
        elif kind == "kegg_compound" and ann not in entry["kegg_compound"]:
            entry["kegg_compound"].append(ann)
    return out


def load_species_name_lookup(
    frame: pd.DataFrame | None = None,
) -> Dict[Tuple[str, str], str]:
    """``(model_id, species_id) -> species_name`` from the Phase 3 SBML table."""
    if frame is None:
        if not OUT_SPECIES_NAMES.exists():
            return {}
        frame = pd.read_csv(OUT_SPECIES_NAMES)
    lookup: Dict[Tuple[str, str], str] = {}
    for rec in frame.itertuples(index=False):
        lookup[(str(rec.model_id), str(rec.species_id))] = str(rec.species_name)
    return lookup


def _participant_block(
    equation: str,
    model_id: str,
    name_lookup: Dict[Tuple[str, str], str],
    evidence_idx: Dict[str, Dict[str, List[str]]],
) -> List[Dict[str, Any]]:
    """Join participant names by species id. Never zip unique ids with name lists."""
    ids = parse_participant_ids(equation)
    blocks = []
    for sid in ids:
        ev = evidence_idx.get(sid, {"chebi": [], "kegg_compound": []})
        blocks.append({
            "species_id": sid,
            "name": name_lookup.get((model_id, sid), sid),
            "chebi": list(ev["chebi"]),
            "kegg_compound": list(ev["kegg_compound"]),
        })
    return blocks


def _direction(equation: str) -> str:
    if "<=>" in equation:
        return "reversible"
    if "=>" in equation or "->" in equation:
        return "irreversible"
    return "unknown"


def _truncate(text: str, limit: int) -> str:
    text = redact_kegg_reaction_ids((text or "").strip().replace("\n", " "))
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _neighbors(
    target: pd.Series,
    model_rows: pd.DataFrame,
    *,
    k: int,
) -> List[Dict[str, Any]]:
    target_ids = set(parse_participant_ids(str(target.reaction_equation)))
    scored: List[Tuple[int, str, pd.Series]] = []
    for _, rec in model_rows.iterrows():
        if rec.reaction_id == target.reaction_id:
            continue
        shared = target_ids & set(parse_participant_ids(str(rec.reaction_equation)))
        if not shared:
            continue
        scored.append((len(shared), str(rec.reaction_id), rec))
    scored.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for n_shared, rid, rec in scored[:k]:
        out.append({
            "reaction_id": redact_kegg_reaction_ids(rid),
            "shared_participants": n_shared,
            "equation": _truncate(str(rec.reaction_equation), MAX_NEIGHBOR_EQ_CHARS),
            "name": redact_kegg_reaction_ids(str(rec.reaction_name or "")),
        })
    return out


def build_context(
    row: pd.Series,
    *,
    variant: str,
    corpus: pd.DataFrame,
    evidence: pd.DataFrame,
    neighborhood_k: int = DEFAULT_NEIGHBORHOOD,
    species_names: pd.DataFrame | Dict[Tuple[str, str], str] | None = None,
) -> Dict[str, Any]:
    if variant not in CONTEXT_VARIANTS:
        raise ValueError(f"unknown context variant {variant}")

    evidence_idx = _species_index(str(row.model_id), evidence)
    if isinstance(species_names, dict):
        name_lookup = species_names
    else:
        name_lookup = load_species_name_lookup(species_names)

    context: Dict[str, Any] = {
        "variant": variant,
        "template_version": PROMPT_TEMPLATE_VERSION,
        "model_id": str(row.model_id),
        "reaction_id": redact_kegg_reaction_ids(str(row.reaction_id)),
        "equation": str(row.reaction_equation or ""),
        "reaction_name": redact_kegg_reaction_ids(str(row.reaction_name or "")),
        "direction": _direction(str(row.reaction_equation or "")),
        "participants": _participant_block(
            str(row.reaction_equation or ""), str(row.model_id),
            name_lookup, evidence_idx),
    }

    if variant in ("target_plus_model", "target_plus_neighborhood"):
        context["model"] = {
            "title": redact_kegg_reaction_ids(str(row.model_title or row.model_name or "")),
            "description": _truncate(str(row.model_notes or ""), MAX_MODEL_NOTES_CHARS),
        }

    if variant == "target_plus_neighborhood":
        model_rows = corpus[corpus.model_id == row.model_id]
        context["neighborhood_k"] = int(neighborhood_k)
        context["neighbors"] = _neighbors(row, model_rows, k=neighborhood_k)

    assert_no_kegg_leakage(context, where=f"{row.model_id}/{row.reaction_id}/{variant}")
    return context


def render_prompt(context: Dict[str, Any]) -> Dict[str, Any]:
    """Turn a context dict into a provider-independent chat payload."""
    system = (
        "You are annotating a metabolic reaction from an SBML model against the KEGG "
        "reaction catalog. Use only the supplied reaction-local context. The catalog is "
        "open: no candidate list is provided. If the evidence is insufficient, abstain."
    )
    user_lines = [
        OPEN_SET_INSTRUCTIONS,
        "",
        f"SBML reaction id: {context['reaction_id']}",
        f"Equation: {context['equation']}",
        f"Direction: {context['direction']}",
    ]
    if context.get("reaction_name"):
        user_lines.append(f"Reaction name: {context['reaction_name']}")
    user_lines.append("Participants:")
    for p in context["participants"]:
        ids = []
        if p["chebi"]:
            ids.append("ChEBI " + ", ".join(p["chebi"]))
        if p["kegg_compound"]:
            ids.append("KEGG compound " + ", ".join(p["kegg_compound"]))
        extra = f" ({'; '.join(ids)})" if ids else ""
        user_lines.append(f"- {p['name']} [{p['species_id']}]{extra}")
    model = context.get("model")
    if model:
        user_lines.append(f"Model title: {model.get('title') or ''}")
        if model.get("description"):
            user_lines.append(f"Model context: {model['description']}")
    neighbors = context.get("neighbors") or []
    if neighbors:
        user_lines.append(
            f"Up to {context.get('neighborhood_k', len(neighbors))} neighboring reactions "
            "that share participants (not the full model):"
        )
        for n in neighbors:
            label = n["name"] or n["reaction_id"]
            user_lines.append(f"- {label}: {n['equation']}")
    user_lines.append("")
    user_lines.append(
        "Respond with JSON: {\"abstain\": bool, \"predictions\": "
        "[{\"kegg_id\": \"R#####\", \"confidence\": 0-1}, ...], "
        "\"rationale\": string, \"basis\": \"recalled_knowledge\"|\"supplied_evidence\"|\"mixed\"}."
    )
    user = "\n".join(user_lines)
    payload = {
        "template_version": PROMPT_TEMPLATE_VERSION,
        "variant": context["variant"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_schema": STRUCTURED_OUTPUT_SCHEMA,
        "max_output_tokens": 400,
    }
    assert_no_kegg_leakage(payload, where=f"prompt:{context['model_id']}/{context['reaction_id']}")
    payload["n_input_tokens_est"] = estimate_tokens(system) + estimate_tokens(user)
    payload["token_estimate_method"] = TOKENIZER_SCAFFOLD
    return payload


def build_pilot_prompts(
    *,
    neighborhood_k: int = DEFAULT_NEIGHBORHOOD,
    variants: Sequence[str] = CONTEXT_VARIANTS,
) -> List[Dict[str, Any]]:
    public, key, _ = build_pilot()
    corpus = load_evaluable_corpus()
    evidence = pd.read_csv(EVIDENCE_CSV)
    evidence["model_id"] = evidence.model_id.astype(str)
    evidence["species_id"] = evidence.species_id.astype(str)
    name_lookup = load_species_name_lookup()

    key_index = {(r.model_id, r.reaction_id) for r in key.itertuples(index=False)}
    rows = []
    for rec in public.itertuples(index=False):
        corpus_row = corpus[
            (corpus.model_id == rec.model_id) & (corpus.reaction_id == rec.reaction_id)
        ].iloc[0]
        for variant in variants:
            context = build_context(
                corpus_row, variant=variant, corpus=corpus, evidence=evidence,
                neighborhood_k=neighborhood_k, species_names=name_lookup,
            )
            prompt = render_prompt(context)
            record = {
                "sample_id": rec.sample_id,
                "model_id": rec.model_id,
                "reaction_id": rec.reaction_id,
                "cluster_id": rec.cluster_id,
                "stratum": rec.stratum,
                "variant": variant,
                "template_version": PROMPT_TEMPLATE_VERSION,
                "neighborhood_k": neighborhood_k if variant == "target_plus_neighborhood" else 0,
                "n_input_tokens_est": prompt["n_input_tokens_est"],
                "token_estimate_method": TOKENIZER_SCAFFOLD,
                "max_output_tokens": prompt["max_output_tokens"],
                "prompt": prompt,
            }
            assert (rec.model_id, rec.reaction_id) in key_index
            # Join keys may themselves be KEGG-shaped SBML ids; only the
            # model-visible prompt is required to be free of R##### tokens.
            assert_no_kegg_leakage(prompt, where=f"prompt:{rec.sample_id}/{variant}")
            rows.append(record)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Write pilot_prompts.jsonl under benchmark/phase3/")
    parser.add_argument("--neighborhood-k", type=int, default=DEFAULT_NEIGHBORHOOD)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = build_pilot_prompts(neighborhood_k=args.neighborhood_k)
    n_var = len({r["variant"] for r in rows})
    logger.info("built %d prompts (%d variants) for %d reactions; leakage checks passed",
                len(rows), n_var, len(rows) // max(n_var, 1))
    if args.write:
        write_jsonl(rows, OUT_PROMPTS)
        logger.info("wrote %s", OUT_PROMPTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
