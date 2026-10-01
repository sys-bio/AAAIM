"""Seeded open-set pilot sample drawn only from the validation split.

The exploratory LLM pilot chooses context variant, prompt, provider/model,
abstention behavior, and tool strategy. Those choices must not use test
reactions. Sequence:

    Train: fit the learned retriever.
    Validation: run the exploratory LLM pilot and choose the method.
    Test: run the final frozen method once.

Default quotas (200 reactions) unless a stratum has too few eligible validation
reactions, in which case every eligible reaction is taken and the shortfall is
recorded. Sampling never borrows from train or test, and the cluster split is
not rewritten to reach 200.

Within a stratum, clusters are visited round-robin so a few genome-scale models
cannot dominate. The within-cluster order is a seeded shuffle.

The sample CSV that would be sent to a model contains no ground-truth KEGG ids.
Those live only in ``pilot_answer_key.csv``.

Usage::

    python benchmark/scripts/sample_phase3_pilot.py
"""

from __future__ import annotations

import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.scripts.phase3_common import (
    CONFIG_ID,
    OUT_PILOT,
    OUT_PILOT_KEY,
    OUT_PILOT_SUMMARY,
    PHASE2_COMMIT,
    PHASE2_TAG,
    PILOT_SEED,
    PILOT_SPLIT,
    STRATA,
    STRATUM_ABSENT,
    STRATUM_EMPTY,
    STRATUM_RERANK,
    STRATUM_TOP1,
    STRATUM_UNCONSTRAINED,
    write_csv,
    write_json,
)
from benchmark.scripts.build_phase3_splits import build_splits

logger = logging.getLogger("sample_phase3_pilot")

DEFAULT_QUOTAS = {
    STRATUM_UNCONSTRAINED: 50,
    STRATUM_EMPTY: 50,
    STRATUM_ABSENT: 50,
    STRATUM_RERANK: 25,
    STRATUM_TOP1: 25,
}

SAMPLE_COLUMNS = [
    "sample_id", "model_id", "reaction_id", "cluster_id", "split", "stratum",
    "is_genome_scale", "complexity_bucket", "species_annotation_source",
    "status", "candidate_set_size", "selection_rule",
]
# Ground truth stays out of SAMPLE_COLUMNS on purpose.
KEY_COLUMNS = [
    "sample_id", "model_id", "reaction_id", "cluster_id", "stratum",
    "ground_truth_kegg_all", "ground_truth_kegg_primary", "num_ground_truth_ids",
    "hit_any_exact", "hit_at_1_exact", "hit_any_brite_orthology",
]


def sample_stratum(
    eligible: pd.DataFrame, quota: int, *, seed: int, stratum: str,
    source_split: str = PILOT_SPLIT,
) -> pd.DataFrame:
    """Round-robin across clusters; seeded shuffle within each cluster."""
    if eligible.empty:
        return eligible.iloc[0:0]
    if len(eligible) <= quota:
        out = eligible.copy()
        out["selection_rule"] = (
            f"all_eligible_{source_split}:{stratum}:n={len(eligible)}<=quota={quota}"
        )
        return out

    rng = random.Random(seed)
    by_cluster: Dict[str, List[int]] = defaultdict(list)
    for idx, rec in eligible.sort_values(["cluster_id", "model_id", "reaction_id"]).iterrows():
        by_cluster[rec.cluster_id].append(idx)
    clusters = sorted(by_cluster)
    for cluster_id in clusters:
        rng.shuffle(by_cluster[cluster_id])

    picked: List[int] = []
    while len(picked) < quota and any(by_cluster[c] for c in clusters):
        for cluster_id in clusters:
            if not by_cluster[cluster_id]:
                continue
            picked.append(by_cluster[cluster_id].pop())
            if len(picked) >= quota:
                break
    out = eligible.loc[picked].copy()
    out["selection_rule"] = (
        f"round_robin_clusters:seed={seed}:quota={quota}:n_clusters={len(clusters)}"
    )
    return out


def build_pilot(
    corpus: pd.DataFrame | None = None,
    *,
    seed: int = PILOT_SEED,
    quotas: Mapping[str, int] | None = None,
    source_split: str = PILOT_SPLIT,
):
    quotas = dict(quotas or DEFAULT_QUOTAS)
    if corpus is None:
        _, _, _, frame = build_splits()
    else:
        frame = corpus
    pool = frame[frame.split == source_split].copy()
    if pool.empty:
        raise RuntimeError(
            f"{source_split} split is empty; run build_phase3_splits.py first"
        )
    other = {"train", "validation", "test"} - {source_split}

    parts: List[pd.DataFrame] = []
    shortfalls: Dict[str, Any] = {}
    for stratum in STRATA:
        eligible = pool[pool.stratum == stratum]
        quota = int(quotas.get(stratum, 0))
        picked = sample_stratum(
            eligible, quota, seed=seed + STRATA.index(stratum), stratum=stratum,
            source_split=source_split,
        )
        if len(picked) < quota:
            shortfalls[stratum] = {
                "quota": quota,
                "eligible_in_source": int(len(eligible)),
                "selected": int(len(picked)),
                "shortfall": quota - int(len(picked)),
                "action": (
                    f"included every eligible {source_split} reaction; "
                    f"did not borrow from {'/'.join(sorted(other))}"
                ),
            }
        parts.append(picked)

    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sort_values(["stratum", "cluster_id", "model_id", "reaction_id"]).reset_index(drop=True)
    sample.insert(0, "sample_id", [f"P3P{i:04d}" for i in range(1, len(sample) + 1)])
    if sample.duplicated(["model_id", "reaction_id"]).any():
        raise RuntimeError("pilot sample contains duplicate reactions")
    if (sample.split != source_split).any():
        raise RuntimeError("pilot sample leaked a reaction from another split")

    public = sample[SAMPLE_COLUMNS].copy()
    key = sample[KEY_COLUMNS].copy()
    # Guard: the public table must not carry ground-truth ids.
    for banned in ("ground_truth_kegg_all", "ground_truth_kegg_primary", "ground_truth_ids"):
        if banned in public.columns:
            raise RuntimeError(f"ground-truth column {banned} leaked into the public sample")

    summary = {
        "phase2_tag": PHASE2_TAG,
        "phase2_commit": PHASE2_COMMIT,
        "config_id": CONFIG_ID,
        "seed": seed,
        "source_split": source_split,
        "quotas": quotas,
        "n_selected": int(len(public)),
        "counts_by_stratum": public.stratum.value_counts().reindex(STRATA).fillna(0).astype(int).to_dict(),
        "n_models": int(public.model_id.nunique()),
        "n_clusters": int(public.cluster_id.nunique()),
        "n_genome_scale": int(public.is_genome_scale.sum()),
        "shortfalls": shortfalls,
        "selection": (
            f"Only the {source_split} split is eligible. The exploratory pilot chooses "
            "method details here; test is reserved for one frozen-method run. Within "
            "each stratum, clusters are visited round-robin after a seeded "
            "within-cluster shuffle so large models cannot exhaust the quota. If a "
            f"stratum has fewer eligible {source_split} reactions than its quota, "
            "every eligible reaction is taken and the shortfall is recorded; "
            f"{'/'.join(sorted(other))} are never used as a backfill. The cluster "
            "split is not rewritten to reach the nominal 200."
        ),
        "answer_key": str(OUT_PILOT_KEY.name),
        "public_sample": str(OUT_PILOT.name),
    }
    return public, key, summary


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    public, key, summary = build_pilot()
    write_csv(public, OUT_PILOT)
    write_csv(key, OUT_PILOT_KEY)
    write_json(summary, OUT_PILOT_SUMMARY)
    logger.info("pilot n=%d models=%d clusters=%d shortfalls=%s",
                summary["n_selected"], summary["n_models"],
                summary["n_clusters"], list(summary["shortfalls"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
