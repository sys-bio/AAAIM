"""Leakage-resistant train/validation/test splits over Phase 1 model clusters.

Clusters are the partition units. No reaction is assigned independently of its cluster,
and the conservative yeast cluster ``CLU_BIOMD0000000042`` is kept intact because it is
already one Phase 1 cluster.

The assignment is a deterministic greedy fill plus a local-improvement pass. Target
shares are 60/15/25 train/validation/test by reaction count, with extra weight on the
rare ``retrievable_rerank_failure`` stratum so the held-out test split can support the
open-set pilot. Exact balance is impossible: seven genome-scale models hold 78% of
reactions and cannot be fractionally split.

Usage::

    python benchmark/scripts/build_phase3_splits.py
"""

from __future__ import annotations

import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.scripts.phase3_common import (
    CONFIG_ID,
    OUT_OVERLAP,
    OUT_SPLIT_SUMMARY,
    OUT_SPLITS,
    PHASE2_COMMIT,
    PHASE2_TAG,
    SPLIT_ALGORITHM,
    SPLIT_SEED,
    SPLITS,
    STRATA,
    YEAST_CLUSTER,
    load_evaluable_corpus,
    parse_kegg_ids,
    write_csv,
    write_json,
)
from benchmark.scripts.build_phase3_strata import build_strata

logger = logging.getLogger("build_phase3_splits")

TARGET_SHARES = {"train": 0.60, "validation": 0.15, "test": 0.25}

# Squared-error weights. Rare strata and genome-scale mass are the hard constraints.
LOSS_WEIGHTS = {
    "n_reactions": 4.0,
    "n_genome_scale": 3.0,
    "unconstrained": 1.5,
    "empty_constrained": 1.5,
    "nonempty_answer_absent": 2.0,
    "retrievable_rerank_failure": 4.0,
    "retrievable_top1_success": 1.0,
}


def _cluster_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, sub in frame.groupby("cluster_id", sort=True):
        row: Dict[str, Any] = {
            "cluster_id": cluster_id,
            "n_models": int(sub.model_id.nunique()),
            "n_reactions": int(len(sub)),
            "n_genome_scale": int(sub.is_genome_scale.sum()),
            "is_genome_scale": bool(sub.is_genome_scale.any()),
            "is_yeast_cluster": cluster_id == YEAST_CLUSTER,
        }
        for name in STRATA:
            row[name] = int((sub.stratum == name).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["n_reactions", "cluster_id"], ascending=[False, True]).reset_index(drop=True)


def _empty_totals() -> Dict[str, float]:
    totals = {"n_reactions": 0.0, "n_genome_scale": 0.0}
    totals.update({s: 0.0 for s in STRATA})
    return totals


def _add(totals: MutableMapping[str, float], cluster: Mapping[str, Any], sign: int = 1) -> None:
    totals["n_reactions"] += sign * cluster["n_reactions"]
    totals["n_genome_scale"] += sign * cluster["n_genome_scale"]
    for s in STRATA:
        totals[s] += sign * cluster[s]


def _loss(assigned: Mapping[str, Mapping[str, float]], corpus_totals: Mapping[str, float]) -> float:
    score = 0.0
    for split, share in TARGET_SHARES.items():
        got = assigned[split]
        for key, weight in LOSS_WEIGHTS.items():
            expected = share * corpus_totals[key]
            if expected <= 0 and got[key] == 0:
                continue
            denom = max(expected, 1.0)
            err = (got[key] - expected) / denom
            score += weight * err * err
    return score


def assign_clusters(
    stats: pd.DataFrame, *, seed: int = SPLIT_SEED,
) -> Dict[str, str]:
    """Greedy then local-search assignment of cluster_id -> split."""
    rng = random.Random(seed)
    tie_order = list(SPLITS)
    rng.shuffle(tie_order)

    corpus_totals = _empty_totals()
    for cluster in stats.itertuples(index=False):
        _add(corpus_totals, cluster._asdict())

    assigned_totals = {s: _empty_totals() for s in SPLITS}
    mapping: Dict[str, str] = {}

    for cluster in stats.itertuples(index=False):
        rec = cluster._asdict()
        best_split = None
        best_loss = math.inf
        for split in tie_order:
            _add(assigned_totals[split], rec, 1)
            score = _loss(assigned_totals, corpus_totals)
            _add(assigned_totals[split], rec, -1)
            if score < best_loss - 1e-12:
                best_loss = score
                best_split = split
        mapping[rec["cluster_id"]] = best_split  # type: ignore[assignment]
        _add(assigned_totals[best_split], rec, 1)

    improved = True
    while improved:
        improved = False
        for cluster in stats.itertuples(index=False):
            rec = cluster._asdict()
            current = mapping[rec["cluster_id"]]
            current_loss = _loss(assigned_totals, corpus_totals)
            best_split = current
            best_loss = current_loss
            for split in tie_order:
                if split == current:
                    continue
                _add(assigned_totals[current], rec, -1)
                _add(assigned_totals[split], rec, 1)
                score = _loss(assigned_totals, corpus_totals)
                _add(assigned_totals[split], rec, -1)
                _add(assigned_totals[current], rec, 1)
                if score < best_loss - 1e-12:
                    best_loss = score
                    best_split = split
            if best_split != current:
                _add(assigned_totals[current], rec, -1)
                _add(assigned_totals[best_split], rec, 1)
                mapping[rec["cluster_id"]] = best_split
                improved = True
    return mapping


def _overlap_report(frame: pd.DataFrame) -> Dict[str, Any]:
    """KEGG target overlap across splits. Overlap is allowed but must be reported."""
    by_split: Dict[str, Counter] = {s: Counter() for s in SPLITS}
    for rec in frame.itertuples(index=False):
        ids = rec.ground_truth_ids if isinstance(rec.ground_truth_ids, list) else parse_kegg_ids(
            rec.ground_truth_kegg_all)
        for kid in ids:
            by_split[rec.split][kid] += 1

    sets = {s: set(by_split[s]) for s in SPLITS}
    train_val = sets["train"] | sets["validation"]
    test_ids = sets["test"]
    seen = sorted(test_ids & train_val)
    unseen = sorted(test_ids - train_val)
    train_and_test = sorted(sets["train"] & sets["test"])

    def _freq_table(counter: Counter) -> List[Dict[str, Any]]:
        return [{"kegg_id": k, "n": int(n)} for k, n in sorted(counter.items())]

    return {
        "n_distinct_targets": {
            split: len(sets[split]) for split in SPLITS
        },
        "targets_in_train_and_test": train_and_test,
        "n_targets_in_train_and_test": len(train_and_test),
        "test_targets_seen_in_train_or_validation": seen,
        "n_test_targets_seen_in_train_or_validation": len(seen),
        "test_targets_never_seen_in_train_or_validation": unseen,
        "n_test_targets_never_seen_in_train_or_validation": len(unseen),
        "share_of_test_targets_seen": (
            round(len(seen) / len(test_ids), 4) if test_ids else None
        ),
        "frequency_by_split": {s: _freq_table(by_split[s]) for s in SPLITS},
        "note": "Overlap is expected: the catalog is a closed KEGG reaction set. "
                "Report test metrics separately for seen vs unseen targets.",
    }


def build_splits(corpus: pd.DataFrame | None = None, *, seed: int = SPLIT_SEED):
    frame = corpus if corpus is not None else load_evaluable_corpus()
    if "stratum" not in frame.columns:
        strata, _ = build_strata(frame)
        frame = frame.merge(
            strata[["model_id", "reaction_id", "stratum"]],
            on=["model_id", "reaction_id"], how="left",
        )

    stats = _cluster_stats(frame)
    mapping = assign_clusters(stats, seed=seed)
    frame = frame.copy()
    frame["split"] = frame.cluster_id.map(mapping)
    if frame.split.isna().any():
        missing = sorted(frame.loc[frame.split.isna(), "cluster_id"].unique())
        raise RuntimeError(f"unassigned clusters: {missing}")

    # Yeast cluster must land in exactly one split.
    yeast_splits = sorted(frame.loc[frame.cluster_id == YEAST_CLUSTER, "split"].unique())
    if yeast_splits and yeast_splits != [yeast_splits[0]]:
        raise RuntimeError("yeast cluster crossed splits")

    split_table = frame[[
        "model_id", "reaction_id", "cluster_id", "split", "stratum",
        "is_genome_scale", "complexity_bucket", "species_annotation_source",
        "status", "candidate_set_size",
    ]].sort_values(["split", "cluster_id", "model_id", "reaction_id"]).reset_index(drop=True)

    def _split_block(name: str) -> Dict[str, Any]:
        sub = frame[frame.split == name]
        return {
            "n_reactions": int(len(sub)),
            "n_models": int(sub.model_id.nunique()),
            "n_clusters": int(sub.cluster_id.nunique()),
            "n_genome_scale_reactions": int(sub.is_genome_scale.sum()),
            "share_of_reactions": round(len(sub) / len(frame), 4),
            "stratum_counts": sub.stratum.value_counts().reindex(STRATA).fillna(0).astype(int).to_dict(),
            "complexity_counts": sub.complexity_bucket.value_counts().sort_index().to_dict(),
            "species_source_counts": sub.species_annotation_source.value_counts().sort_index().to_dict(),
            "clusters": sorted(sub.cluster_id.unique()),
            "models": sorted(sub.model_id.unique()),
        }

    overlap = _overlap_report(frame)
    summary: Dict[str, Any] = {
        "phase2_tag": PHASE2_TAG,
        "phase2_commit": PHASE2_COMMIT,
        "config_id": CONFIG_ID,
        "algorithm": SPLIT_ALGORITHM,
        "seed": seed,
        "target_shares": TARGET_SHARES,
        "loss_weights": LOSS_WEIGHTS,
        "yeast_cluster": YEAST_CLUSTER,
        "yeast_cluster_split": yeast_splits[0] if yeast_splits else None,
        "n_evaluable": int(len(frame)),
        "n_clusters": int(frame.cluster_id.nunique()),
        "n_models": int(frame.model_id.nunique()),
        "splits": {name: _split_block(name) for name in SPLITS},
        "tradeoffs": (
            "Seven genome-scale models hold ~78% of reactions and cannot be fractionally "
            "split. The greedy objective therefore trades reaction-count share against "
            "stratum and genome-scale share. The rare retrievable_rerank_failure stratum "
            "(85 reactions) is up-weighted so the test split can feed the open-set pilot. "
            "Phase 1 cluster membership is never rewritten."
        ),
    }
    return split_table, summary, overlap, frame


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    table, summary, overlap, _ = build_splits()
    write_csv(table, OUT_SPLITS)
    write_json(summary, OUT_SPLIT_SUMMARY)
    # Frequency tables are large; keep the compact overlap summary in the dedicated file
    # and a short pointer in split_summary.
    compact = {k: v for k, v in overlap.items() if k != "frequency_by_split"}
    compact["frequency_by_split_rows"] = {
        s: len(overlap["frequency_by_split"][s]) for s in SPLITS
    }
    write_json(overlap, OUT_OVERLAP)
    summary["target_overlap"] = compact
    write_json(summary, OUT_SPLIT_SUMMARY)
    for name, block in summary["splits"].items():
        logger.info("%s: %d reactions, %d models, %d clusters (share=%.3f)",
                    name, block["n_reactions"], block["n_models"],
                    block["n_clusters"], block["share_of_reactions"])
    logger.info("test targets seen in train/val: %d; unseen: %d",
                overlap["n_test_targets_seen_in_train_or_validation"],
                overlap["n_test_targets_never_seen_in_train_or_validation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
