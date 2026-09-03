"""Shared Phase 3 constants, IO, corpus join and stratum assignment.

Phase 3 artifacts live under ``benchmark/phase3/``, never inside the frozen Phase 1/2
tables in ``benchmark/data/``. This module only *reads* those frozen files.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import AbstractSet, Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

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
OUT_SPECIES_NAMES = PHASE3_DIR / "species_names.csv"
OUT_KEGG_CATALOG_IDS = PHASE3_DIR / "kegg_catalog_ids.json"
PRICING_EXAMPLE = PHASE3_DIR / "pricing.example.json"
PRICING_OPENAI_TERRA = PHASE3_DIR / "pricing.openai.gpt-5.6-terra.json"
OUT_SMOKE_DIR = PHASE3_DIR / "smoke"

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
# Exploratory LLM choices (variant, prompt, provider, abstention, tools) use
# validation only. Test is reserved for one frozen-method run.
PILOT_SPLIT = "validation"
FIT_SPLITS_TRAIN = ("train",)
FIT_SPLITS_TRAIN_VAL = ("train", "validation")
SPLIT_SEED = 20260902
SPLIT_ALGORITHM = "cluster_greedy_v1"
PILOT_SEED = 20260902
PROMPT_TEMPLATE_VERSION = "phase3-open-set-v3"

# Word-boundary detector (legacy). Underscores and letters are word characters,
# so it misses R##### embedded in SBML ids such as R_R06861_C3_cytop and R00678_Tdo.
KEGG_REACTION_LEGACY_WORD_BOUNDARY_RE = re.compile(r"\bR\d{5}\b")
# R followed by exactly five digits, bounded by non-digits (or string edges).
# Matches R00024, R_R06861_C3_cytop, R00678_Tdo, prefixR00024_suffix.
# Rejects R000240 (six or more digits).
KEGG_REACTION_RE = re.compile(r"(?<!\d)(R\d{5})(?!\d)")
KEGG_REACTION_URI_RE = re.compile(r"kegg\.reaction[/:]R\d{5}(?!\d)", re.IGNORECASE)
KEGG_ID_STRICT = re.compile(r"^R\d{5}$")
ID_MALFORMED = "malformed"
ID_ABSENT = "absent_from_catalog"
ID_IN_CATALOG = "in_catalog"

# Scaffolding only. Live runs must use the chosen model's tokenizer.
TOKENIZER_SCAFFOLD = "chars_div_4_scaffold"
# Conservative spend gate: 2 characters ≈ 1 token. English/scientific text is
# typically closer to 4 chars/token, so this overestimates input size.
TOKENIZER_CONSERVATIVE = "chars_div_2_conservative"
OUTPUT_SCHEMA_VERSION = "phase3-structured-v1"
DEFAULT_MODEL = "gpt-5.6-terra"
SMOKE_N_REACTIONS = 3
SMOKE_N_REQUESTS = 9
SMOKE_SELECTION_RULE = "seeded_round_robin_one_per_stratum_v1"
LIVE_TOKENIZER_REQUIRED = (
    "Live runs must use the chosen model's tokenizer or a conservative "
    "provider-specific bound; chars/4 is scaffolding only and must not gate spend."
)

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
    atomic_write_json(obj, path)


def atomic_write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def atomic_write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="\n", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    atomic_write_jsonl(rows, path)


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
    """Replace every embedded R##### (and kegg.reaction URIs) in a string."""
    if not text:
        return ""
    text = KEGG_REACTION_URI_RE.sub("[REDACTED_KEGG_REACTION]", text)
    return KEGG_REACTION_RE.sub("[REDACTED_KEGG_REACTION]", text)


def redact_kegg_in_obj(obj: Any) -> Any:
    """Redact KEGG reaction ids in every string inside a JSON-able object."""
    if isinstance(obj, str):
        return redact_kegg_reaction_ids(obj)
    if isinstance(obj, list):
        return [redact_kegg_in_obj(item) for item in obj]
    if isinstance(obj, dict):
        return {key: redact_kegg_in_obj(value) for key, value in obj.items()}
    return obj


def extract_kegg_reaction_ids(text: str) -> List[str]:
    """All R##### substrings using digit boundaries, not word boundaries."""
    if not text:
        return []
    return KEGG_REACTION_RE.findall(text)


def find_kegg_leakage(payload: Any) -> List[str]:
    """Return unique KEGG reaction ids found anywhere in a JSON-able payload."""
    blob = json.dumps(payload, default=str)
    return sorted(set(extract_kegg_reaction_ids(blob)))


def assert_no_kegg_leakage(payload: Any, *, where: str) -> None:
    leaked = find_kegg_leakage(payload)
    if leaked:
        raise ValueError(f"KEGG reaction-id leakage in {where}: {leaked[:10]}")


def classify_kegg_id(kegg_id: str, catalog: AbstractSet[str]) -> str:
    """Classify an identifier against syntax and the frozen KEGG catalog.

    ``R#####`` matching is well-formed syntax only. Existence is catalog membership.
    """
    if not kegg_id or not KEGG_ID_STRICT.match(kegg_id):
        return ID_MALFORMED
    if kegg_id not in catalog:
        return ID_ABSENT
    return ID_IN_CATALOG


def load_kegg_catalog_ids(path: Path | None = None) -> FrozenSet[str]:
    """Frozen Phase 2 KEGG reaction-id set (Phase 3 snapshot, not a live download)."""
    path = path or OUT_KEGG_CATALOG_IDS
    if not path.exists():
        return frozenset()
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    return frozenset(str(x) for x in ids)


def seen_targets_from_corpus(
    corpus: pd.DataFrame,
    *,
    fit_splits: Sequence[str],
) -> set:
    """Target KEGG ids appearing in the splits used to *fit* the model.

    Use ``fit_splits=("train",)`` when the retriever is trained on train only
    (method selection happens on validation). Use ``("train", "validation")``
    when the final retriever is refit on train+validation before the test run.
    """
    allowed = set(fit_splits)
    unknown = allowed - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown fit splits: {sorted(unknown)}")
    if "split" not in corpus.columns:
        raise ValueError("corpus must include a split column")
    subset = corpus[corpus.split.isin(allowed)]
    ids: set = set()
    values = subset["ground_truth_ids"] if "ground_truth_ids" in subset.columns else []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            ids.update(value)
        else:
            ids.update(parse_kegg_ids(value))
    if not ids and "ground_truth_kegg_all" in subset.columns:
        for value in subset["ground_truth_kegg_all"]:
            ids.update(parse_kegg_ids(value))
    return ids


def estimate_tokens(text: str) -> int:
    """Scaffolding token estimate: ~4 characters per token.

    Not valid for live spend. Call ``require_live_tokenizer`` before a paid run.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_tokens_conservative(text: str) -> int:
    """Conservative spend-gate estimate: 2 characters per token.

    This is an upper bound for English and scientific prose, not a billing tokenizer.
    """
    if not text:
        return 0
    return max(1, (len(text) + 1) // 2)


def require_live_tokenizer(method: str | None) -> None:
    """Refuse a live run that still uses the chars/4 scaffolding estimate."""
    if method is None or method == TOKENIZER_SCAFFOLD:
        raise RuntimeError(LIVE_TOKENIZER_REQUIRED)
