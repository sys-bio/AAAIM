"""Assign mutually exclusive Phase 2 outcome strata to every evaluable reaction.

Primary assignment uses exact KEGG matching so it matches the Phase 2 failure
decomposition. Equivalence-aware labels are stored alongside, never mixed in.

Usage::

    python benchmark/scripts/build_phase3_strata.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.phase3_common import (
    CONFIG_ID,
    OUT_STRATA,
    OUT_STRATA_SUMMARY,
    PHASE2_COMMIT,
    PHASE2_TAG,
    STRATA,
    load_evaluable_corpus,
    write_csv,
    write_json,
)

logger = logging.getLogger("build_phase3_strata")

STRATA_COLUMNS = [
    "model_id", "reaction_id", "cluster_id", "stratum",
    "status", "candidate_set_size", "has_candidates",
    "hit_any_exact", "hit_at_1_exact",
    "hit_any_brite_orthology", "hit_at_1_brite_orthology",
    "is_genome_scale", "complexity_bucket", "species_annotation_source",
    "num_participants", "num_ground_truth_ids",
]


def build_strata(corpus=None):
    frame = corpus if corpus is not None else load_evaluable_corpus()
    out = frame[STRATA_COLUMNS].copy()
    counts = out.stratum.value_counts().reindex(STRATA).fillna(0).astype(int)
    if int(counts.sum()) != len(out):
        raise RuntimeError("stratum counts do not cover the evaluable corpus")
    if out.duplicated(["model_id", "reaction_id"]).any():
        raise RuntimeError("duplicate reaction keys in strata")
    unlabeled = out[out.stratum.isna() | ~out.stratum.isin(STRATA)]
    if not unlabeled.empty:
        raise RuntimeError(f"unlabeled reactions: {len(unlabeled)}")
    summary: Dict[str, Any] = {
        "phase2_tag": PHASE2_TAG,
        "phase2_commit": PHASE2_COMMIT,
        "config_id": CONFIG_ID,
        "n_evaluable": int(len(out)),
        "counts": counts.to_dict(),
        "matching": "exact",
        "note": "Equivalence-aware hit columns are stored but do not assign the stratum.",
        "exhaustive": bool(int(counts.sum()) == len(out)),
        "mutually_exclusive": True,
    }
    return out, summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    table, summary = build_strata()
    write_csv(table, OUT_STRATA)
    write_json(summary, OUT_STRATA_SUMMARY)
    logger.info("wrote %s (%d rows); counts=%s", OUT_STRATA.name, len(table), summary["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
