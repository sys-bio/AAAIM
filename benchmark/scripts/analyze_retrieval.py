"""Phase 2 step 3: candidate-retrieval ceiling.

Answers the question "how good could any reranker possibly be, given this candidate
generator?" — the upper bound on end-to-end accuracy, and the dividing line between
retrieval failures and reranking failures.

Reported for exact KEGG id matching and, separately, for equivalence-aware matching over
BRITE/orthology groups (see ``kegg_equivalence``).

Every metric is reported three ways, because seven genome-scale models hold ~78% of the
reactions and would otherwise dictate the headline number:

reaction_micro
    Mean over all reactions.
model_macro
    Mean over models of each model's reaction mean.
cluster_macro
    Mean over Phase 1 duplicate-clusters of each cluster's reaction mean.

Usage::

    python benchmark/scripts/analyze_retrieval.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.scripts.kegg_equivalence import (
    EQUIVALENCE_KINDS,
    coverage_stats,
    match_kinds,
)

DATA_DIR = REPO_ROOT / "benchmark" / "data"
REACTIONS_CSV = DATA_DIR / "reactions.csv"
CANDIDATES_CSV = DATA_DIR / "candidates.csv"
STATUS_CSV = DATA_DIR / "candidate_status.csv"
STRATA_CSV = DATA_DIR / "reaction_strata.csv"

OUT_PER_REACTION = DATA_DIR / "reaction_retrieval.csv"
OUT_BY_STRATUM = DATA_DIR / "retrieval_ceiling_by_stratum.csv"
OUT_JSON = DATA_DIR / "retrieval_ceiling.json"

CRITERIA = ("exact",) + EQUIVALENCE_KINDS
K_VALUES = (1, 3, 5, 10)

logger = logging.getLogger("analyze_retrieval")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")


def write_json(obj: Any, path: Path) -> None:
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def three_way(df: pd.DataFrame, column: str) -> Dict[str, Optional[float]]:
    """Reaction-micro, model-macro and cluster-macro averages of a 0/1 or numeric column."""
    if df.empty:
        return {"reaction_micro": None, "model_macro": None, "cluster_macro": None,
                "n_reactions": 0, "n_models": 0, "n_clusters": 0}
    series = df[column].astype(float)
    return {
        "reaction_micro": round(float(series.mean()), 4),
        "model_macro": round(float(df.groupby("model_id")[column].mean().mean()), 4),
        "cluster_macro": round(float(df.groupby("cluster_id")[column].mean().mean()), 4),
        "n_reactions": int(len(df)),
        "n_models": int(df.model_id.nunique()),
        "n_clusters": int(df.cluster_id.nunique()),
    }


def build_per_reaction() -> pd.DataFrame:
    """One row per evaluable reaction with candidate-set size and first-hit ranks."""
    reactions = pd.read_csv(REACTIONS_CSV)
    strata = pd.read_csv(STRATA_CSV)
    status = pd.read_csv(STATUS_CSV)
    candidates = pd.read_csv(CANDIDATES_CSV)

    evaluable = reactions[reactions.included_in_eval.astype(bool)].copy()
    evaluable["reaction_id"] = evaluable.reaction_id.astype(str)
    evaluable["model_id"] = evaluable.model_id.astype(str)

    strata["reaction_id"] = strata.reaction_id.astype(str)
    strata["model_id"] = strata.model_id.astype(str)
    status["reaction_id"] = status.reaction_id.astype(str)
    status["model_id"] = status.model_id.astype(str)

    stratum_cols = [
        "model_id", "reaction_id", "cluster_id", "num_participants", "complexity_bucket",
        "species_annotation_source", "num_participants_unannotated",
        "any_missing_annotation", "participant_annotation_coverage",
        "is_genome_scale", "model_reaction_count",
    ]
    df = evaluable.merge(strata[stratum_cols], on=["model_id", "reaction_id"], how="left")
    df = df.merge(
        status[["model_id", "reaction_id", "status", "num_candidates",
                "degenerate_set_size", "relaxation_level", "relaxation_direction"]],
        on=["model_id", "reaction_id"], how="left",
    )

    candidates["reaction_id"] = candidates.reaction_id.astype(str)
    candidates["model_id"] = candidates.model_id.astype(str)
    candidates = candidates.sort_values(["model_id", "reaction_id", "raw_rank"])
    by_reaction = {
        key: list(zip(sub.candidate_kegg.astype(str), sub.raw_rank.astype(int)))
        for key, sub in candidates.groupby(["model_id", "reaction_id"], sort=False)
    }

    rows: List[Dict[str, Any]] = []
    for r in df.itertuples():
        truth = {t for t in str(r.ground_truth_kegg_all).split("|") if t}
        ranked = by_reaction.get((r.model_id, r.reaction_id), [])

        row: Dict[str, Any] = {
            "model_id": r.model_id,
            "reaction_id": r.reaction_id,
            "cluster_id": r.cluster_id,
            "status": r.status if isinstance(r.status, str) else "missing_status",
            "candidate_set_size": len(ranked),
            "has_candidates": len(ranked) > 0,
            "num_ground_truth_ids": len(truth),
            "relaxation_level": int(r.relaxation_level) if pd.notna(r.relaxation_level) else 0,
            "relaxation_required": bool(pd.notna(r.relaxation_level) and r.relaxation_level > 0),
            "species_annotation_source": r.species_annotation_source,
            "any_missing_annotation": bool(r.any_missing_annotation),
            "num_participants": int(r.num_participants) if pd.notna(r.num_participants) else 0,
            "complexity_bucket": r.complexity_bucket,
            "is_genome_scale": bool(r.is_genome_scale),
            "degenerate_set_size": int(r.degenerate_set_size) if pd.notna(r.degenerate_set_size) else 0,
        }

        first_hit: Dict[str, Optional[int]] = {c: None for c in CRITERIA}
        for cand, rank in ranked:
            verdict = match_kinds(cand, truth)
            for crit in CRITERIA:
                if verdict[crit] and first_hit[crit] is None:
                    first_hit[crit] = int(rank)
            if all(first_hit[c] is not None for c in CRITERIA):
                break

        for crit in CRITERIA:
            fh = first_hit[crit]
            row[f"first_hit_rank_{crit}"] = fh
            row[f"hit_any_{crit}"] = fh is not None
            for k in K_VALUES:
                row[f"hit_at_{k}_{crit}"] = fh is not None and fh <= k
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate(per_reaction: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    subsets = {
        "all_evaluable": per_reaction,
        "with_candidates": per_reaction[per_reaction.has_candidates],
    }
    for subset_name, subset in subsets.items():
        block: Dict[str, Any] = {
            "candidate_set_size_mean": (
                round(float(subset.candidate_set_size.mean()), 2) if len(subset) else None
            ),
            "candidate_set_size_median": (
                float(subset.candidate_set_size.median()) if len(subset) else None
            ),
            "zero_candidate_pct": (
                round(100.0 * float((~subset.has_candidates).mean()), 2) if len(subset) else None
            ),
            "n_reactions": int(len(subset)),
        }
        for crit in CRITERIA:
            crit_block = {"recall_any_rank": three_way(subset, f"hit_any_{crit}")}
            for k in K_VALUES:
                crit_block[f"recall_at_{k}"] = three_way(subset, f"hit_at_{k}_{crit}")
            block[crit] = crit_block
        out[subset_name] = block
    return out


def by_stratum(per_reaction: pd.DataFrame) -> pd.DataFrame:
    """Recall/ceiling broken down by each stratification variable."""
    rows: List[Dict[str, Any]] = []
    strata_defs: Sequence[tuple] = (
        ("species_annotation_source", "species_annotation_source"),
        ("any_missing_annotation", "any_missing_annotation"),
        ("relaxation_required", "relaxation_required"),
        ("relaxation_level", "relaxation_level"),
        ("complexity_bucket", "complexity_bucket"),
        ("is_genome_scale", "is_genome_scale"),
        ("status", "status"),
    )
    for stratum_name, column in strata_defs:
        for value, sub in per_reaction.groupby(column, dropna=False):
            row: Dict[str, Any] = {
                "stratum": stratum_name,
                "value": str(value),
                "n_reactions": int(len(sub)),
                "n_models": int(sub.model_id.nunique()),
                "n_clusters": int(sub.cluster_id.nunique()),
                "zero_candidate_pct": round(100.0 * float((~sub.has_candidates).mean()), 2),
                "mean_candidate_set_size": round(float(sub.candidate_set_size.mean()), 2),
            }
            for crit in ("exact", "brite_orthology"):
                agg = three_way(sub, f"hit_any_{crit}")
                row[f"recall_any_{crit}_micro"] = agg["reaction_micro"]
                row[f"recall_any_{crit}_model_macro"] = agg["model_macro"]
                row[f"recall_any_{crit}_cluster_macro"] = agg["cluster_macro"]
                row[f"recall_at_1_{crit}_micro"] = three_way(sub, f"hit_at_1_{crit}")["reaction_micro"]
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["stratum", "value"]).reset_index(drop=True)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    per_reaction = build_per_reaction()
    write_csv(per_reaction, OUT_PER_REACTION)

    stratum_df = by_stratum(per_reaction)
    write_csv(stratum_df, OUT_BY_STRATUM)

    summary = {
        "criteria": list(CRITERIA),
        "k_values": list(K_VALUES),
        "equivalence_coverage": coverage_stats(),
        "status_counts": per_reaction.status.value_counts().sort_index().to_dict(),
        "metrics": aggregate(per_reaction),
        "averaging_definitions": {
            "reaction_micro": "mean over all reactions in the subset",
            "model_macro": "mean over models of each model's reaction mean",
            "cluster_macro": "mean over Phase 1 clusters of each cluster's reaction mean",
        },
        "inputs": {
            "reactions_csv_sha256": sha256_file(REACTIONS_CSV),
            "candidates_csv_sha256": sha256_file(CANDIDATES_CSV),
            "candidate_status_csv_sha256": sha256_file(STATUS_CSV),
            "reaction_strata_csv_sha256": sha256_file(STRATA_CSV),
        },
        "outputs": {
            "reaction_retrieval_csv_sha256": sha256_file(OUT_PER_REACTION),
            "retrieval_ceiling_by_stratum_csv_sha256": sha256_file(OUT_BY_STRATUM),
        },
    }
    write_json(summary, OUT_JSON)

    allev = summary["metrics"]["all_evaluable"]
    logger.info("evaluable reactions: %d", allev["n_reactions"])
    logger.info("zero-candidate: %.1f%% | mean candidate set size: %s",
                allev["zero_candidate_pct"], allev["candidate_set_size_mean"])
    for crit in ("exact", "brite_orthology"):
        a = allev[crit]["recall_any_rank"]
        logger.info(
            "%s recall(any rank): micro=%.3f model_macro=%.3f cluster_macro=%.3f",
            crit, a["reaction_micro"], a["model_macro"], a["cluster_macro"],
        )
        r1 = allev[crit]["recall_at_1"]
        logger.info("%s recall@1: micro=%.3f model_macro=%.3f cluster_macro=%.3f",
                    crit, r1["reaction_micro"], r1["model_macro"], r1["cluster_macro"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
