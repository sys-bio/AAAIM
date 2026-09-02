"""Shared Phase 3 constants, IO, corpus join and stratum assignment.

Phase 3 artifacts live under ``benchmark/phase3/``, never inside the frozen Phase 1/2
tables in ``benchmark/data/``. This module only *reads* those frozen files.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "benchmark" / "data"
PHASE3_DIR = REPO_ROOT / "benchmark" / "phase3"

# Frozen Phase 2 inputs (read-only).
REACTIONS_CSV = DATA_DIR / "reactions.csv"
RETRIEVAL_CSV = DATA_DIR / "reaction_retrieval.csv"
STATUS_CSV = DATA_DIR / "candidate_status.csv"
STRATA_CSV = DATA_DIR / "reaction_strata.csv"
TEXT_CSV = DATA_DIR / "reaction_text.csv"
CLUSTERS_CSV = DATA_DIR / "model_clusters.csv"
CONTEXT_CSV = DATA_DIR / "model_context.csv"
EVIDENCE_CSV = DATA_DIR / "species_evidence.csv"

# Phase 3 outputs.
OUT_STRATA = PHASE3_DIR / "strata.csv"
OUT_STRATA_SUMMARY = PHASE3_DIR / "strata_summary.json"
OUT_SPLITS = PHASE3_DIR / "splits.csv"
OUT_SPLIT_SUMMARY = PHASE3_DIR / "split_summary.json"
OUT_OVERLAP = PHASE3_DIR / "target_overlap.json"
OUT_PILOT = PHASE3_DIR / "pilot_sample.csv"
OUT_PILOT_KEY = PHASE3_DIR / "pilot_answer_key.csv"
OUT_PILOT_SUMMARY = PHASE3_DIR / "pilot_summary.json"
OUT_PROMPTS = PHASE3_DIR / "pilot_prompts.jsonl"
OUT_COST = PHASE3_DIR / "cost_estimate.json"
PRICING_EXAMPLE = PHASE3_DIR / "pricing.example.json"

YEAST_CLUSTER = "CLU_BIOMD0000000042"
PHASE2_TAG = "benchmark-phase2-v1"
PHASE2_COMMIT = "19580572a70c2c138290aa6da06697a7cef9d7f6"
CONFIG_ID = "86938b48ab88"

STRATUM_UNCONSTRAINED = "unconstrained"
STRATUM_EMPTY = "empty_constrained"
STRATUM_ABSENT = "nonempty_answer_absent"
STRATUM_RERANK = "retrievable_rerank_failure"
STRATUM_TOP1 = "retrievable_top1_success"
STRATA = (
    STRATUM_UNCONSTRAINED,
    STRATUM_EMPTY,
    STRATUM_ABSENT,
    STRATUM_RERANK,
    STRATUM_TOP1,
)

STATUS_UNCONSTRAINED = "unconstrained_candidate_set"
STATUS_NO_CANDIDATES = "no_candidates"
STATUS_OK = "ok"

SPLITS = ("train", "validation", "test")
SPLIT_SEED = 20260902
SPLIT_ALGORITHM = "cluster_greedy_v1"
PILOT_SEED = 20260902
PROMPT_TEMPLATE_VERSION = "phase3-open-set-v1"

KEGG_REACTION_RE = re.compile(r"\bR\d{5}\b")
KEGG_REACTION_URI_RE = re.compile(r"kegg\.reaction[/:]R\d{5}", re.IGNORECASE)
KEGG_ID_STRICT = re.compile(r"^R\d{5}$")

# Tokens that look like KEGG reaction ids must never appear in a prompt payload.
LEAKAGE_PATTERNS = (KEGG_REACTION_RE, KEGG_REACTION_URI_RE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def parse_kegg_ids(value: Any) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [p for p in re.split(r"[;,\s]+", text) if KEGG_ID_STRICT.match(p)]


def assign_stratum(status: str, hit_any_exact: bool, hit_at_1_exact: bool) -> str:
    """Mutually exclusive Phase 2 outcome. Exact matching only."""
    if status == STATUS_UNCONSTRAINED:
        return STRATUM_UNCONSTRAINED
    if status == STATUS_NO_CANDIDATES:
        return STRATUM_EMPTY
    if not hit_any_exact:
        return STRATUM_ABSENT
    if not hit_at_1_exact:
        return STRATUM_RERANK
    return STRATUM_TOP1


def parse_participant_ids(equation: str) -> List[str]:
    """Species ids from an SBML-style reaction equation, order-preserving unique."""
    if not equation or not isinstance(equation, str):
        return []
    body = re.split(r"<?=>", equation, maxsplit=1)
    tokens: List[str] = []
    seen = set()
    for side in body:
        for part in re.split(r"\s*\+\s*", side):
            part = part.strip()
            if not part:
                continue
            part = re.sub(r"^\d+(?:\.\d+)?\s+", "", part)
            if part and part not in seen:
                seen.add(part)
                tokens.append(part)
    return tokens


def load_evaluable_corpus() -> pd.DataFrame:
    """One row per evaluable reaction, joined from frozen Phase 1/2 tables."""
    reactions = pd.read_csv(REACTIONS_CSV)
    reactions["model_id"] = reactions.model_id.astype(str)
    reactions["reaction_id"] = reactions.reaction_id.astype(str)
    evaluable = reactions[reactions.included_in_eval.astype(bool)].copy()

    retrieval = pd.read_csv(RETRIEVAL_CSV)
    retrieval["model_id"] = retrieval.model_id.astype(str)
    retrieval["reaction_id"] = retrieval.reaction_id.astype(str)

    strata = pd.read_csv(STRATA_CSV)
    strata["model_id"] = strata.model_id.astype(str)
    strata["reaction_id"] = strata.reaction_id.astype(str)

    text = pd.read_csv(TEXT_CSV)
    text["model_id"] = text.model_id.astype(str)
    text["reaction_id"] = text.reaction_id.astype(str)

    context = pd.read_csv(CONTEXT_CSV)
    context["model_id"] = context.model_id.astype(str)

    keys = ["model_id", "reaction_id"]
    frame = evaluable[keys + [
        "reaction_equation", "ground_truth_kegg_all", "ground_truth_kegg_primary",
        "num_ground_truth_ids", "is_exchange_ssx",
    ]].merge(
        retrieval[keys + [
            "cluster_id", "status", "candidate_set_size", "has_candidates",
            "hit_any_exact", "hit_at_1_exact", "hit_any_brite_orthology",
            "hit_at_1_brite_orthology", "first_hit_rank_exact",
        ]],
        on=keys, how="left", validate="one_to_one",
    ).merge(
        strata[keys + [
            "complexity_bucket", "species_annotation_source", "is_genome_scale",
            "num_participants", "any_missing_annotation",
        ]],
        on=keys, how="left", validate="one_to_one",
    ).merge(
        text[keys + [
            "model_name", "reaction_name", "has_reaction_name",
            "substrate_names", "product_names", "modifier_names",
            "query_text",
        ]],
        on=keys, how="left",
    ).merge(
        context[["model_id", "model_title", "model_notes", "num_reactions_total"]],
        on="model_id", how="left",
    )

    for col in ("hit_any_exact", "hit_at_1_exact", "has_candidates",
                "is_genome_scale", "hit_any_brite_orthology", "hit_at_1_brite_orthology"):
        frame[col] = frame[col].fillna(False).astype(bool)

    frame["stratum"] = [
        assign_stratum(str(s), bool(any_), bool(at1))
        for s, any_, at1 in zip(
            frame.status, frame.hit_any_exact, frame.hit_at_1_exact)
    ]
    frame["ground_truth_ids"] = frame.ground_truth_kegg_all.map(parse_kegg_ids)
    return frame.sort_values(keys).reset_index(drop=True)


def redact_kegg_reaction_ids(text: str) -> str:
    if not text:
        return ""
    text = KEGG_REACTION_URI_RE.sub("[REDACTED_KEGG_REACTION]", text)
    return KEGG_REACTION_RE.sub("[REDACTED_KEGG_REACTION]", text)


def find_kegg_leakage(payload: Any) -> List[str]:
    """Return unique KEGG reaction ids / URIs found anywhere in a JSON-able payload."""
    blob = json.dumps(payload, default=str)
    found = set(KEGG_REACTION_RE.findall(blob))
    found.update(KEGG_REACTION_URI_RE.findall(blob))
    return sorted(found)


def assert_no_kegg_leakage(payload: Any, *, where: str) -> None:
    leaked = find_kegg_leakage(payload)
    if leaked:
        raise ValueError(f"KEGG reaction-id leakage in {where}: {leaked[:10]}")


def estimate_tokens(text: str) -> int:
    """Tokenizer-free token estimate: ~4 characters per token, minimum 1 if nonempty."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
