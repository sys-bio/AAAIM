"""Phase 2 diagnostics: candidate-set size, concentration and degeneracy.

Read-only analysis of the frozen candidate table. It answers "how big are the candidate
sets, where do the rows come from, and are large sets useful or noise?" without
re-running generation.

The central finding this script quantifies: candidate-set size is driven almost entirely
by how many reaction participants were mapped to KEGG compounds. ``filter_kegg_reactions``
keeps a KEGG reaction when the model's mapped compound keys are a *subset* of the KEGG
reaction's keys, so the fewer constraints a reaction has, the more of KEGG it matches.
Zero mapped participants matches all of KEGG and is recorded as
``unconstrained_candidate_set``; exactly one mapped participant, especially a ubiquitous
cofactor, still matches thousands of reactions and is kept as ``ok``.

Usage::

    python benchmark/scripts/candidate_diagnostics.py
    python benchmark/scripts/candidate_diagnostics.py --focus-model BIOMD0000001063
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.scripts.generate_candidates import (
    CANDIDATES_CSV,
    DATA_DIR,
    STATUS_CSV,
    STATUS_OK,
    sha256_file,
    write_csv,
    write_json,
)

RETRIEVAL_CSV = DATA_DIR / "reaction_retrieval.csv"

OUT_JSON = DATA_DIR / "candidate_diagnostics.json"
OUT_BY_STRATUM = DATA_DIR / "candidate_size_by_stratum.csv"
OUT_LARGEST = DATA_DIR / "candidate_largest_sets.csv"

SIZE_THRESHOLDS = (15, 50, 100, 1000, 10000)
PERCENTILES = (50, 75, 90, 95, 99)
CONCENTRATION_TOP_N = (1, 5, 10)
LARGEST_N = 20

# Above this many candidates a set is treated as weakly constrained for reporting. It is
# a reporting boundary only: nothing is filtered or capped anywhere in the pipeline.
WEAK_SET_THRESHOLD = 100

logger = logging.getLogger("candidate_diagnostics")


def _stats(series: pd.Series) -> Dict[str, Any]:
    if series.empty:
        return {"n": 0}
    s = series.astype(float)
    out: Dict[str, Any] = {
        "n": int(len(s)),
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "min": int(s.min()),
        "max": int(s.max()),
        "total_rows": int(s.sum()),
    }
    for p in PERCENTILES:
        out[f"p{p}"] = round(float(s.quantile(p / 100.0)), 2)
    for t in SIZE_THRESHOLDS:
        out[f"count_gt_{t}"] = int((s > t).sum())
    return out


def _load() -> Dict[str, pd.DataFrame]:
    status = pd.read_csv(STATUS_CSV)
    status["model_id"] = status.model_id.astype(str)
    status["reaction_id"] = status.reaction_id.astype(str)

    candidates = pd.read_csv(CANDIDATES_CSV)
    candidates["model_id"] = candidates.model_id.astype(str)
    candidates["reaction_id"] = candidates.reaction_id.astype(str)
    candidates["candidate_kegg"] = candidates.candidate_kegg.astype(str)

    retrieval = pd.read_csv(RETRIEVAL_CSV)
    retrieval["model_id"] = retrieval.model_id.astype(str)
    retrieval["reaction_id"] = retrieval.reaction_id.astype(str)

    # Reaction-level view: one row per evaluable reaction with size, status, strata and
    # whether the exact answer is retrievable.
    keep = ["model_id", "reaction_id", "status", "num_candidates", "filtered_species_count",
            "relaxation_level", "relaxation_direction", "degenerate_set_size"]
    view = status[keep].copy()
    retr_cols = ["model_id", "reaction_id", "cluster_id", "is_genome_scale",
                 "species_annotation_source", "complexity_bucket", "num_participants",
                 "hit_any_exact", "hit_at_1_exact"]
    view = view.merge(retrieval[retr_cols], on=["model_id", "reaction_id"], how="left")

    model_sizes = view.groupby("model_id").size().rename("model_reaction_count")
    view = view.merge(model_sizes, on="model_id", how="left")
    view["model_size_stratum"] = pd.cut(
        view.model_reaction_count,
        bins=[0, 10, 50, 200, 1000, 10 ** 9],
        labels=["1-10", "11-50", "51-200", "201-1000", ">1000"],
    ).astype(str)
    return {"status": status, "candidates": candidates, "retrieval": retrieval, "view": view}


def _by_group(view: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for value, grp in view.groupby(column, dropna=False, observed=True):
        stats = _stats(grp.num_candidates)
        rows.append({
            "stratum": column,
            "value": str(value),
            "n_reactions": stats["n"],
            "total_candidate_rows": stats["total_rows"],
            "mean_size": stats["mean"],
            "median_size": stats["median"],
            "p90_size": stats.get("p90"),
            "max_size": stats["max"],
            "count_gt_100": stats.get("count_gt_100"),
            "pct_retrievable_exact": (
                round(100.0 * float(grp.hit_any_exact.fillna(False).astype(bool).mean()), 2)
                if len(grp) else None
            ),
            "pct_top1_exact": (
                round(100.0 * float(grp.hit_at_1_exact.fillna(False).astype(bool).mean()), 2)
                if len(grp) else None
            ),
        })
    return rows


def diagnose(focus_model: Optional[str] = None) -> Dict[str, Any]:
    data = _load()
    view, candidates = data["view"], data["candidates"]
    total_rows = int(len(candidates))

    nonempty = view[view.num_candidates > 0]

    # Concentration by model.
    by_model = (candidates.groupby("model_id").size().sort_values(ascending=False)
                .rename("candidate_rows"))
    concentration = {}
    for n in CONCENTRATION_TOP_N:
        top = by_model.head(n)
        concentration[f"top_{n}_models"] = {
            "models": list(top.index),
            "candidate_rows": int(top.sum()),
            "share_of_all_rows": round(float(top.sum()) / total_rows, 4) if total_rows else None,
        }

    # Concentration by reaction.
    by_reaction = (candidates.groupby(["model_id", "reaction_id"]).size()
                   .sort_values(ascending=False))
    reaction_concentration = {}
    for n in (1, 10, 100):
        top = by_reaction.head(n)
        reaction_concentration[f"top_{n}_reactions"] = {
            "candidate_rows": int(top.sum()),
            "share_of_all_rows": round(float(top.sum()) / total_rows, 4) if total_rows else None,
        }

    largest = (by_reaction.head(LARGEST_N).rename("candidate_rows").reset_index()
               .merge(view, on=["model_id", "reaction_id"], how="left"))
    largest_out = largest[[
        "model_id", "reaction_id", "candidate_rows", "status", "filtered_species_count",
        "num_participants", "relaxation_level", "relaxation_direction",
        "hit_any_exact", "hit_at_1_exact",
    ]]

    # The mechanism: size as a function of how many participants were mapped.
    by_constraints: List[Dict[str, Any]] = []
    for value, grp in view[view.status == STATUS_OK].groupby("filtered_species_count"):
        by_constraints.append({
            "filtered_species_count": int(value),
            "n_reactions": int(len(grp)),
            "total_candidate_rows": int(grp.num_candidates.sum()),
            "mean_size": round(float(grp.num_candidates.mean()), 2),
            "median_size": round(float(grp.num_candidates.median()), 2),
            "max_size": int(grp.num_candidates.max()),
            "pct_retrievable_exact": round(
                100.0 * float(grp.hit_any_exact.fillna(False).astype(bool).mean()), 2),
            "pct_top1_exact": round(
                100.0 * float(grp.hit_at_1_exact.fillna(False).astype(bool).mean()), 2),
        })

    # Weakly-constrained sets: large, and empirically almost never containing the answer.
    ok = view[view.status == STATUS_OK]
    weak = ok[ok.num_candidates > WEAK_SET_THRESHOLD]
    strong = ok[ok.num_candidates <= WEAK_SET_THRESHOLD]
    degeneracy = {
        "weak_set_threshold": WEAK_SET_THRESHOLD,
        "weakly_constrained": {
            "n_reactions": int(len(weak)),
            "share_of_ok_reactions": round(float(len(weak)) / len(ok), 4) if len(ok) else None,
            "candidate_rows": int(weak.num_candidates.sum()),
            "share_of_all_rows": (
                round(float(weak.num_candidates.sum()) / total_rows, 4) if total_rows else None),
            "pct_retrievable_exact": (
                round(100.0 * float(weak.hit_any_exact.fillna(False).astype(bool).mean()), 2)
                if len(weak) else None),
            "pct_top1_exact": (
                round(100.0 * float(weak.hit_at_1_exact.fillna(False).astype(bool).mean()), 2)
                if len(weak) else None),
        },
        "normally_constrained": {
            "n_reactions": int(len(strong)),
            "candidate_rows": int(strong.num_candidates.sum()),
            "pct_retrievable_exact": (
                round(100.0 * float(strong.hit_any_exact.fillna(False).astype(bool).mean()), 2)
                if len(strong) else None),
            "pct_top1_exact": (
                round(100.0 * float(strong.hit_at_1_exact.fillna(False).astype(bool).mean()), 2)
                if len(strong) else None),
        },
    }

    duplicate_rows = int(candidates.duplicated(
        subset=["model_id", "reaction_id", "candidate_kegg"]).sum())

    stratum_rows: List[Dict[str, Any]] = []
    for column in ("status", "model_size_stratum", "relaxation_level",
                   "relaxation_direction", "is_genome_scale", "complexity_bucket",
                   "species_annotation_source"):
        stratum_rows.extend(_by_group(view, column))
    stratum_df = pd.DataFrame(stratum_rows).sort_values(
        ["stratum", "value"]).reset_index(drop=True)

    summary: Dict[str, Any] = {
        "totals": {
            "evaluable_reactions": int(len(view)),
            "candidate_rows": total_rows,
            "distinct_candidate_kegg_ids": int(candidates.candidate_kegg.nunique()),
            "reactions_with_candidates": int(len(nonempty)),
            "reactions_with_zero_candidates": int((view.num_candidates == 0).sum()),
            "zero_candidate_rate": round(float((view.num_candidates == 0).mean()), 4),
            "duplicate_candidate_rows": duplicate_rows,
        },
        "size_all_evaluable": _stats(view.num_candidates),
        "size_nonempty_only": _stats(nonempty.num_candidates),
        "model_concentration": concentration,
        "reaction_concentration": reaction_concentration,
        "size_by_mapped_participant_count": by_constraints,
        "degeneracy": degeneracy,
        "largest_reaction_sets": largest_out.to_dict("records"),
        "inputs": {
            "candidates_csv_sha256": sha256_file(CANDIDATES_CSV),
            "candidate_status_csv_sha256": sha256_file(STATUS_CSV),
            "reaction_retrieval_csv_sha256": sha256_file(RETRIEVAL_CSV),
        },
    }

    if focus_model:
        summary["focus_model"] = _focus(view, candidates, focus_model, total_rows)

    write_csv(stratum_df, OUT_BY_STRATUM)
    write_csv(largest_out.reset_index(drop=True), OUT_LARGEST)
    summary["outputs"] = {
        "candidate_size_by_stratum_csv_sha256": sha256_file(OUT_BY_STRATUM),
        "candidate_largest_sets_csv_sha256": sha256_file(OUT_LARGEST),
    }
    write_json(summary, OUT_JSON)
    return summary


def _focus(view: pd.DataFrame, candidates: pd.DataFrame, model_id: str,
           total_rows: int) -> Dict[str, Any]:
    """Explain one model's contribution to the candidate table."""
    mv = view[view.model_id == model_id]
    mc = candidates[candidates.model_id == model_id]
    if mv.empty:
        return {"model_id": model_id, "error": "model not present in status table"}

    ok = mv[mv.status == STATUS_OK]
    weak = ok[ok.num_candidates > WEAK_SET_THRESHOLD]
    rows = int(len(mc))

    # Does excluding this model move the headline metrics?
    others = view[view.model_id != model_id]
    def _micro(df: pd.DataFrame, col: str) -> Optional[float]:
        return round(float(df[col].fillna(False).astype(bool).mean()), 4) if len(df) else None

    return {
        "model_id": model_id,
        "evaluable_reactions": int(len(mv)),
        "candidate_rows": rows,
        "share_of_all_candidate_rows": round(float(rows) / total_rows, 4) if total_rows else None,
        "status_counts": mv.status.value_counts().sort_index().to_dict(),
        "duplicate_candidate_rows": int(mc.duplicated(
            subset=["reaction_id", "candidate_kegg"]).sum()),
        "reactions_by_mapped_participants": (
            ok.filtered_species_count.value_counts().sort_index().to_dict()),
        "weakly_constrained_reactions": {
            "n": int(len(weak)),
            "candidate_rows": int(weak.num_candidates.sum()),
            "share_of_model_rows": (
                round(float(weak.num_candidates.sum()) / rows, 4) if rows else None),
            "share_of_all_rows": (
                round(float(weak.num_candidates.sum()) / total_rows, 4) if total_rows else None),
            "pct_retrievable_exact": (
                round(100.0 * float(weak.hit_any_exact.fillna(False).astype(bool).mean()), 2)
                if len(weak) else None),
        },
        "relaxation_levels": mc.relaxation_level.value_counts().sort_index().to_dict(),
        "relaxation_directions": mc.relaxation_direction.value_counts().sort_index().to_dict(),
        "metric_impact": {
            "model_recall_any_exact": _micro(mv, "hit_any_exact"),
            "model_recall_at_1_exact": _micro(mv, "hit_at_1_exact"),
            "corpus_recall_any_exact_including_model": _micro(view, "hit_any_exact"),
            "corpus_recall_any_exact_excluding_model": _micro(others, "hit_any_exact"),
            "corpus_recall_at_1_exact_including_model": _micro(view, "hit_at_1_exact"),
            "corpus_recall_at_1_exact_excluding_model": _micro(others, "hit_at_1_exact"),
        },
        "largest_sets": (mv.sort_values("num_candidates", ascending=False)
                         .head(10)[["reaction_id", "num_candidates",
                                    "filtered_species_count", "num_participants",
                                    "hit_any_exact"]].to_dict("records")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focus-model", default="BIOMD0000001063",
                        help="Model to explain in detail (pass '' to skip)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    summary = diagnose(focus_model=args.focus_model or None)
    t = summary["totals"]
    logger.info("evaluable=%d candidate_rows=%d zero_candidate_rate=%.4f duplicates=%d",
                t["evaluable_reactions"], t["candidate_rows"], t["zero_candidate_rate"],
                t["duplicate_candidate_rows"])
    allsz, nonempty = summary["size_all_evaluable"], summary["size_nonempty_only"]
    logger.info("size all-evaluable: mean=%.2f median=%.2f p90=%.2f p99=%.2f max=%d",
                allsz["mean"], allsz["median"], allsz["p90"], allsz["p99"], allsz["max"])
    logger.info("size nonempty:      mean=%.2f median=%.2f p90=%.2f p99=%.2f max=%d",
                nonempty["mean"], nonempty["median"], nonempty["p90"], nonempty["p99"],
                nonempty["max"])
    top1 = summary["model_concentration"]["top_1_models"]
    logger.info("largest model %s contributes %d rows (%.1f%% of all)",
                top1["models"][0], top1["candidate_rows"],
                100.0 * top1["share_of_all_rows"])
    weak = summary["degeneracy"]["weakly_constrained"]
    logger.info("weakly-constrained (>%d candidates): %d reactions, %d rows (%.1f%% of all), "
                "exact retrievable in only %.1f%%",
                summary["degeneracy"]["weak_set_threshold"], weak["n_reactions"],
                weak["candidate_rows"], 100.0 * weak["share_of_all_rows"],
                weak["pct_retrievable_exact"])
    logger.info("wrote %s, %s, %s", OUT_JSON.name, OUT_BY_STRATUM.name, OUT_LARGEST.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
