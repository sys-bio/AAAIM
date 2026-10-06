"""Phase 3B selected-model export and validation-only retrieval fusion.

The two operations in this module deliberately have separate trust boundaries:
``freeze_fusion_rankings`` never opens an answer key, while
``evaluate_frozen_fusion`` refuses to score until the compressed ranking has a
recorded digest.  Archive restoration loads exclusively from an extracted local
Hugging Face directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import lzma
import os
import platform
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from benchmark.scripts import phase3_biencoder as biencoder
from benchmark.scripts import phase3_retrieval as retrieval
from benchmark.scripts.phase3_common import (
    PHASE3_DIR,
    REPO_ROOT,
    TRUE_RETRIEVAL_FAILURE_STRATA,
    _replace_with_retry,
    repo_relative_posix,
    sha256_file,
    sha256_portable,
    write_artifact_manifest,
    write_json,
)

FUSION_OUT = PHASE3_DIR / "phase3b_fusion"
DIST_DIR = REPO_ROOT / "benchmark" / "dist"
ARCHIVE_NAME = "aaaim-phase3b-selected-epoch1-inference.zip"
ARCHIVE_PATH = DIST_DIR / ARCHIVE_NAME
ARCHIVE_REGISTRY = DIST_DIR / "aaaim-phase3b-selected-epoch1-inference.registry.json"
CHECKPOINT = biencoder.FULL_OUT / "_checkpoints" / "best.pt"
SELECTED_METADATA = biencoder.FULL_OUT / "selected_checkpoint.json"
SELECTED_RANKINGS = biencoder.FULL_OUT / "rankings_epoch_1.jsonl"
BM25_RANKINGS = retrieval.OUT / "rankings_bm25.jsonl"
CATALOG_IDS = PHASE3_DIR / "kegg_catalog_ids.json"
MODEL_CACHE_SNAPSHOT = (
    biencoder.OUT / "_model_cache" / "models--BAAI--bge-small-en-v1.5"
    / "snapshots" / biencoder.MODEL_REVISION
)
FUSION_RANKING_NAME = "rankings_bm25_trained_epoch1_rrf.jsonl.xz"
FUSION_METHOD = "bm25_trained_epoch1_rrf"
FUSION_SCHEMA = "phase3b-fusion-ranking-v1"
RELEASE_SCHEMA = "phase3b-selected-inference-v1"
EXPECTED_CHECKPOINT_SHA256 = "8773b04f09916889b74c956e708044fe2c653fa764fe73c422ecc376ecae81c1"
EXPECTED_RANKING_SHA256 = "660c7ff55d787928050ba963cf4236788de21f661a523a5ebc1c47abbfc2f304"
EXPECTED_CONFIG_HASH = "0b808b72948311a44ebfd5869168751297df47a6afcfd1484a3b6907c9cabe88"
EXPECTED_DATASET_HASH = "142c9f0d404531cf2a8c59ddfabdc80a2858dddd4656a1becc1fb5c0ff292a05"
EXPECTED_EPOCH = 1
EXPECTED_CATALOG_SIZE = 12_312
EXPECTED_VALIDATION_SIZE = 969
RRF_K = 60
BOOTSTRAP_SEED = 20260902
BOOTSTRAP_REPLICATES = 10_000


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_xz_jsonl(path: Path) -> list[dict[str, Any]]:
    with lzma.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_xz_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Write deterministic XZ-compressed canonical JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with lzma.LZMAFile(tmp, "wb", format=lzma.FORMAT_XZ, preset=9) as raw:
        with io.TextIOWrapper(raw, encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
    _replace_with_retry(tmp, path)


def verify_selected_checkpoint() -> dict[str, Any]:
    """Fail closed if the selected checkpoint or its committed metadata drifted."""
    selected = json.loads(SELECTED_METADATA.read_text(encoding="utf-8"))
    record = selected["selected_checkpoint"]
    metrics = json.loads((biencoder.FULL_OUT / "metrics_epoch_1.json").read_text(encoding="utf-8"))
    initialization = json.loads((biencoder.FULL_OUT / "initialization.json").read_text(encoding="utf-8"))
    config = json.loads((biencoder.FULL_OUT / "full_training_config.json").read_text(encoding="utf-8"))
    actual = {
        "selected_epoch": selected.get("selected_epoch"),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "metadata_checkpoint_sha256": record.get("sha256"),
        "initializer": initialization.get("model"),
        "revision": initialization.get("revision"),
        "pooling": "cls",
        "normalization": "l2",
        "max_length": config["configuration"].get("max_length"),
        "config_hash": record.get("config_hash"),
        "dataset_hash": record.get("dataset_hash"),
        "recall_at_1": metrics["exact"]["recall_at_1"]["reaction_micro"],
        "recall_at_10": metrics["exact"]["recall_at_10"]["reaction_micro"],
        "ranking_sha256": sha256_file(SELECTED_RANKINGS),
    }
    expected = {
        "selected_epoch": EXPECTED_EPOCH,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "metadata_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "initializer": biencoder.MODEL_NAME,
        "revision": biencoder.MODEL_REVISION,
        "pooling": "cls",
        "normalization": "l2",
        "max_length": 256,
        "config_hash": EXPECTED_CONFIG_HASH,
        "dataset_hash": EXPECTED_DATASET_HASH,
        "recall_at_1": 0.840041,
        "recall_at_10": 0.921569,
        "ranking_sha256": EXPECTED_RANKING_SHA256,
    }
    if actual != expected:
        raise ValueError(f"selected Phase 3B checkpoint mismatch: {actual!r} != {expected!r}")
    return actual


def reciprocal_rank_fusion_complete(
    component_rankings: Sequence[Sequence[str]], catalog_ids: Sequence[str], *, k: int = RRF_K,
) -> list[str]:
    """Equal-weight RRF over complete catalog, with identifier tie-breaking.

    Frozen component files are 100 deep.  IDs absent from either file receive
    exactly zero, so all remaining catalog documents follow the positive-score
    union in ascending identifier order.
    """
    if k < 0:
        raise ValueError("RRF k must be non-negative")
    catalog = list(catalog_ids)
    if len(catalog) != len(set(catalog)):
        raise ValueError("catalog contains duplicate identifiers")
    catalog_set = set(catalog)
    scores: dict[str, float] = defaultdict(float)
    for ranked in component_rankings:
        ids = list(ranked)
        if len(ids) != len(set(ids)):
            raise ValueError("component ranking contains duplicate identifiers")
        outside = set(ids) - catalog_set
        if outside:
            raise ValueError(f"component ranking contains non-catalog IDs: {sorted(outside)[:3]}")
        for rank, identifier in enumerate(ids, 1):
            scores[identifier] += 1.0 / (k + rank)
    return sorted(catalog, key=lambda identifier: (-scores.get(identifier, 0.0), identifier))


def validate_fusion_rows(
    rows: Sequence[Mapping[str, Any]], catalog_ids: Sequence[str], *, expected: int = EXPECTED_VALIDATION_SIZE,
) -> None:
    if len(rows) != expected:
        raise ValueError(f"fusion ranking row count mismatch: {len(rows)} != {expected}")
    catalog = set(catalog_ids)
    if len(catalog) != EXPECTED_CATALOG_SIZE:
        raise ValueError(f"catalog size mismatch: {len(catalog)}")
    keys: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("split") != "validation":
            raise ValueError("fusion rankings are strictly validation-only")
        if row.get("method") != FUSION_METHOD or row.get("schema") != FUSION_SCHEMA:
            raise ValueError("fusion ranking schema/method mismatch")
        key = (str(row["model_id"]), str(row["reaction_id"]))
        if key in keys:
            raise ValueError(f"duplicate fusion ranking key: {key}")
        keys.add(key)
        ranked = list(row["ranked_ids"])
        if len(ranked) != EXPECTED_CATALOG_SIZE or len(ranked) != len(set(ranked)):
            raise ValueError(f"ranking is not a unique complete catalog permutation: {key}")
        if set(ranked) != catalog:
            raise ValueError(f"ranking catalog membership mismatch: {key}")


def _load_component_maps(path: Path, *, expected_method: str | None = None) -> dict[tuple[str, str], list[str]]:
    rows = _read_jsonl(path)
    result: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if row.get("split", "validation") != "validation":
            raise ValueError(f"non-validation component row in {path}")
        if expected_method is not None and row.get("method") != expected_method:
            raise ValueError(f"unexpected component method in {path}")
        key = (str(row["model_id"]), str(row["reaction_id"]))
        if key in result:
            raise ValueError(f"duplicate component key in {path}: {key}")
        result[key] = list(row["ranked_ids"])
    return result


def fusion_config() -> dict[str, Any]:
    catalog_meta = json.loads(CATALOG_IDS.read_text(encoding="utf-8"))
    return {
        "schema": FUSION_SCHEMA,
        "scope": "validation only; no held-out test reactions or labels",
        "method": FUSION_METHOD,
        "components": [
            {"method": "bm25", "path": repo_relative_posix(BM25_RANKINGS), "sha256": sha256_file(BM25_RANKINGS), "depth": 100, "weight": 1.0},
            {"method": "trained_biencoder_epoch1", "path": repo_relative_posix(SELECTED_RANKINGS), "sha256": sha256_file(SELECTED_RANKINGS), "depth": 100, "weight": 1.0},
        ],
        "rrf": {"k": RRF_K, "rank_origin": 1, "weights": "equal", "missing_contribution": 0.0, "tie_break": "KEGG identifier ascending"},
        "catalog": {
            "path": repo_relative_posix(CATALOG_IDS),
            "sha256": sha256_portable(CATALOG_IDS),
            "source_path": catalog_meta["source"],
            "source_sha256": sha256_file(retrieval.CATALOG),
            "size": catalog_meta["n"],
        },
        "construction": "label-free; complete-catalog permutation; no re-encoding, retraining, tuning, or validation-label access",
        "zero_score_tail": "IDs absent from both frozen 100-deep inputs have equal zero score and are ordered by ascending KEGG ID",
        "test_rows_read": 0,
    }


def freeze_fusion_rankings(out: Path = FUSION_OUT) -> Path:
    """Construct and hash the ranking without loading validation truth."""
    verify_selected_checkpoint()
    catalog_ids = list(json.loads(CATALOG_IDS.read_text(encoding="utf-8"))["ids"])
    if len(catalog_ids) != EXPECTED_CATALOG_SIZE or catalog_ids != sorted(catalog_ids):
        raise ValueError("frozen catalog ID list is not the expected sorted complete catalog")
    bm25 = _load_component_maps(BM25_RANKINGS, expected_method="bm25")
    trained = _load_component_maps(SELECTED_RANKINGS)
    if set(bm25) != set(trained):
        raise ValueError("BM25 and trained bi-encoder ranking populations differ")
    population = retrieval.load_query_population("validation")  # Model-visible fields only.
    expected_keys = set(zip(population.model_id.astype(str), population.reaction_id.astype(str)))
    if set(bm25) != expected_keys or len(expected_keys) != EXPECTED_VALIDATION_SIZE:
        raise ValueError("component rankings do not match the frozen validation query population")
    rows = [
        {
            "schema": FUSION_SCHEMA,
            "split": "validation",
            "model_id": key[0],
            "reaction_id": key[1],
            "method": FUSION_METHOD,
            "ranked_ids": reciprocal_rank_fusion_complete([bm25[key], trained[key]], catalog_ids),
        }
        for key in sorted(expected_keys)
    ]
    validate_fusion_rows(rows, catalog_ids, expected=EXPECTED_VALIDATION_SIZE)
    out.mkdir(parents=True, exist_ok=True)
    path = out / FUSION_RANKING_NAME
    _write_xz_jsonl(rows, path)
    freeze = {
        "schema": FUSION_SCHEMA,
        "path": f"benchmark/phase3/phase3b_fusion/{FUSION_RANKING_NAME}",
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "n_queries": len(rows),
        "ranking_depth": EXPECTED_CATALOG_SIZE,
        "catalog_permutation_verified": True,
        "frozen_before_ground_truth_join": True,
        "labels_loaded_during_ranking": False,
        "test_rows_read": 0,
    }
    write_json(fusion_config(), out / "config.json")
    write_json(freeze, out / "ranking_freeze.json")
    return path


def _metric_columns() -> list[str]:
    return (
        [f"recall_at_{k}_exact" for k in (1, 3, 5, 10)] + ["mrr_at_10_exact"]
        + [f"recall_at_{k}_brite_orthology" for k in (1, 3, 5, 10)] + ["mrr_at_10_brite_orthology"]
    )


def _metric_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {column: biencoder._metric_three_way(frame, column) for column in _metric_columns()}


def _rank_value(value: Any) -> int:
    return 10**9 if value == "" or value is None or pd.isna(value) else int(value)


def _transition_block(component: pd.DataFrame, fusion: pd.DataFrame, *, k: int) -> dict[str, int]:
    keys = ["model_id", "reaction_id"]
    column = f"recall_at_{k}_exact"
    joined = component[keys + [column]].merge(fusion[keys + [column]], on=keys, suffixes=("_component", "_fusion"), validate="one_to_one")
    component_hit = joined[f"{column}_component"].astype(bool)
    fusion_hit = joined[f"{column}_fusion"].astype(bool)
    return {
        "correct_both": int((component_hit & fusion_hit).sum()),
        "recovered_by_fusion": int((~component_hit & fusion_hit).sum()),
        "lost_by_fusion": int((component_hit & ~fusion_hit).sum()),
        "missed_both": int((~component_hit & ~fusion_hit).sum()),
    }


def _overlap_and_transitions(
    frames: Mapping[str, pd.DataFrame], rank_maps: Mapping[str, Mapping[tuple[str, str], Sequence[str]]],
) -> dict[str, Any]:
    bm = frames["bm25"].set_index(["model_id", "reaction_id"])
    trained = frames["trained_biencoder_epoch1"].set_index(["model_id", "reaction_id"])
    fused = frames[FUSION_METHOD].set_index(["model_id", "reaction_id"])
    universe = set(bm.index)
    bm10 = {key for key in universe if bool(bm.at[key, "recall_at_10_exact"])}
    tr10 = {key for key in universe if bool(trained.at[key, "recall_at_10_exact"])}
    fu10 = {key for key in universe if bool(fused.at[key, "recall_at_10_exact"])}
    rank_detail: dict[str, int] = {}
    for name, component in (("bm25", bm), ("trained_biencoder_epoch1", trained)):
        component_ranks = component.first_hit_rank_exact.map(_rank_value)
        fusion_ranks = fused.first_hit_rank_exact.map(_rank_value)
        rank_detail[f"demoted_vs_{name}"] = int((fusion_ranks > component_ranks).sum())
        rank_detail[f"promoted_vs_{name}"] = int((fusion_ranks < component_ranks).sum())
        rank_detail[f"lost_top1_vs_{name}"] = int(((component_ranks == 1) & (fusion_ranks != 1)).sum())
        rank_detail[f"lost_top10_vs_{name}"] = int(((component_ranks <= 10) & (fusion_ranks > 10)).sum())
    return {
        "scope": "exact validation reaction outcomes",
        "at_10": {
            "correct_for_bm25_only": len(bm10 - tr10),
            "correct_for_trained_biencoder_only": len(tr10 - bm10),
            "correct_for_both": len(bm10 & tr10),
            "missed_by_both": len(universe - (bm10 | tr10)),
            "recovered_by_fusion_beyond_bm25": len(fu10 - bm10),
            "recovered_by_fusion_beyond_trained_biencoder": len(fu10 - tr10),
            "recovered_by_fusion_beyond_both_components": len(fu10 - (bm10 | tr10)),
            "lost_by_fusion_relative_to_bm25": len(bm10 - fu10),
            "lost_by_fusion_relative_to_trained_biencoder": len(tr10 - fu10),
        },
        "component_to_fusion_transition_matrices": {
            method: {f"recall_at_{k}": _transition_block(frames[method], frames[FUSION_METHOD], k=k) for k in (1, 10)}
            for method in ("bm25", "trained_biencoder_epoch1")
        },
        "rank_transitions": rank_detail,
        "rank_sources": {method: len(rank_maps[method]) for method in rank_maps},
        "n_reactions": len(universe),
    }


def _qualitative_examples(
    truth: pd.DataFrame, frames: Mapping[str, pd.DataFrame], rank_maps: Mapping[str, Mapping[tuple[str, str], Sequence[str]]],
) -> dict[str, Any]:
    indexed = {name: frame.set_index(["model_id", "reaction_id"]) for name, frame in frames.items()}
    keys = sorted(set(indexed[FUSION_METHOD].index))
    bm, tr, fu = indexed["bm25"], indexed["trained_biencoder_epoch1"], indexed[FUSION_METHOD]
    predicates = {
        "bm25_only_at_10": lambda key: bool(bm.at[key, "recall_at_10_exact"]) and not bool(tr.at[key, "recall_at_10_exact"]),
        "trained_biencoder_only_at_10": lambda key: bool(tr.at[key, "recall_at_10_exact"]) and not bool(bm.at[key, "recall_at_10_exact"]),
        "correct_for_both_at_10": lambda key: bool(tr.at[key, "recall_at_10_exact"]) and bool(bm.at[key, "recall_at_10_exact"]),
        "missed_by_both_at_10": lambda key: not bool(tr.at[key, "recall_at_10_exact"]) and not bool(bm.at[key, "recall_at_10_exact"]),
        "fusion_recovers_beyond_bm25_at_10": lambda key: bool(fu.at[key, "recall_at_10_exact"]) and not bool(bm.at[key, "recall_at_10_exact"]),
        "fusion_recovers_beyond_trained_at_10": lambda key: bool(fu.at[key, "recall_at_10_exact"]) and not bool(tr.at[key, "recall_at_10_exact"]),
        "fusion_loses_bm25_hit_at_10": lambda key: bool(bm.at[key, "recall_at_10_exact"]) and not bool(fu.at[key, "recall_at_10_exact"]),
        "fusion_loses_trained_hit_at_10": lambda key: bool(tr.at[key, "recall_at_10_exact"]) and not bool(fu.at[key, "recall_at_10_exact"]),
    }
    truth_idx = truth.set_index(["model_id", "reaction_id"])
    examples = []
    for category, predicate in predicates.items():
        eligible = [key for key in keys if predicate(key)]
        if not eligible:
            examples.append({"category": category, "example": None})
            continue
        key = eligible[0]
        meta = truth_idx.loc[key]
        examples.append({
            "category": category,
            "selection": "lexicographically first eligible validation reaction",
            "model_id": key[0],
            "reaction_id": key[1],
            "stratum": meta.stratum,
            "seen_in_train": bool(meta.seen_in_train),
            "ground_truth_ids": list(meta.truth),
            "first_hit_rank_exact": {name: (None if _rank_value(frame.at[key, "first_hit_rank_exact"]) == 10**9 else _rank_value(frame.at[key, "first_hit_rank_exact"])) for name, frame in indexed.items()},
            "top10": {name: list(rank_maps[name][key][:10]) for name in rank_maps},
        })
    return {"selection_rule": "lexicographically first eligible validation reaction per prespecified category", "examples": examples, "test_rows_read": 0}


def _bootstrap(frames: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    comparisons = {}
    for reference in ("bm25", "trained_biencoder_epoch1", "bm25_bge_m3_rrf"):
        comparisons[f"fusion_minus_{reference}"] = {
            f"recall_at_{k}": biencoder.paired_cluster_bootstrap_strict(
                frames[FUSION_METHOD], frames[reference], f"recall_at_{k}_exact",
                seed=BOOTSTRAP_SEED, n_boot=BOOTSTRAP_REPLICATES,
            )
            for k in (1, 10)
        }
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "frozen validation cluster",
        "n_clusters": 12,
        "comparisons": comparisons,
        "interpretation_rule": "Do not claim superiority when the 95% percentile interval includes zero.",
    }


def _pareto_frontier(metrics: Mapping[str, Mapping[str, Any]], seen: Mapping[str, Any]) -> list[str]:
    methods = list(metrics)
    values = {
        method: (
            metrics[method]["recall_at_1_exact"]["reaction_micro"],
            metrics[method]["recall_at_10_exact"]["reaction_micro"],
            seen[method]["unseen"]["recall_at_10_exact"]["reaction_micro"],
        )
        for method in methods
    }
    return sorted(method for method in methods if not any(
        other != method
        and all(x >= y for x, y in zip(values[other], values[method]))
        and any(x > y for x, y in zip(values[other], values[method]))
        for other in methods
    ))


def _recommendation(
    metrics: Mapping[str, Mapping[str, Any]], seen: Mapping[str, Any], bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    fusion = metrics[FUSION_METHOD]
    trained = metrics["trained_biencoder_epoch1"]
    overall_delta = fusion["recall_at_10_exact"]["reaction_micro"] - trained["recall_at_10_exact"]["reaction_micro"]
    unseen_delta = seen[FUSION_METHOD]["unseen"]["recall_at_10_exact"]["reaction_micro"] - seen["trained_biencoder_epoch1"]["unseen"]["recall_at_10_exact"]["reaction_micro"]
    r1_delta = fusion["recall_at_1_exact"]["reaction_micro"] - trained["recall_at_1_exact"]["reaction_micro"]
    interval = bootstrap["comparisons"]["fusion_minus_trained_biencoder_epoch1"]["recall_at_10"]
    if overall_delta >= 0 and unseen_delta >= 0 and not interval["includes_zero"]:
        choice = FUSION_METHOD
        rationale = "Fusion improves the primary Recall@10 criterion with a cluster-bootstrap interval excluding zero and does not reduce unseen-target Recall@10."
    elif overall_delta >= 0 and unseen_delta >= 0 and r1_delta >= -0.01:
        choice = FUSION_METHOD
        rationale = "Fusion has the best validation evidence-set recall, including unseen targets, with no material (>0.01) Recall@1 harm; statistical uncertainty is retained."
    else:
        choice = "trained_biencoder_epoch1"
        rationale = "The prespecified fusion does not establish a sufficient Recall@10/unseen-target benefit to displace the selected trained retriever."
    return {
        "phase3c_evidence_retriever": choice,
        "evidence_depth": 10,
        "rationale": rationale,
        "fusion_minus_trained": {"recall_at_1": round(r1_delta, 6), "recall_at_10": round(overall_delta, 6), "unseen_recall_at_10": round(unseen_delta, 6)},
        "recall_at_1_material_harm_threshold": -0.01,
        "materially_harms_trained_recall_at_1": bool(r1_delta < -0.01),
        "pareto_frontier_recall1_recall10_unseen_recall10": _pareto_frontier(metrics, seen),
        "scope": "validation-only selection for future Phase 3C; Phase 3C was not started",
    }


def _report(
    metrics: Mapping[str, Mapping[str, Any]], seen: Mapping[str, Any], strata: Mapping[str, Any],
    overlap: Mapping[str, Any], bootstrap: Mapping[str, Any], recommendation: Mapping[str, Any],
) -> str:
    methods = list(metrics)
    lines = [
        "# Phase 3B BM25 + trained bi-encoder fusion", "",
        "The prespecified equal-weight RRF was built from frozen 100-deep BM25 and selected epoch-1 bi-encoder rankings, with `k=60`, one-indexed ranks, zero contribution for missing documents, and ascending KEGG-ID tie-breaking. Each output is a complete permutation of the frozen 12,312-reaction catalog. Rankings were hashed before validation truth was joined; no test row or label was read.", "",
        "## Exact validation metrics", "",
        "| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 | R@10 model macro | R@10 cluster macro |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        value = metrics[method]
        lines.append("| " + method + " | " + " | ".join(str(value[column]["reaction_micro"]) for column in ("recall_at_1_exact", "recall_at_3_exact", "recall_at_5_exact", "recall_at_10_exact", "mrr_at_10_exact")) + f" | {value['recall_at_10_exact']['model_macro']} | {value['recall_at_10_exact']['cluster_macro']} |")
    lines += ["", "## BRITE/orthology-aware validation metrics", "", "| Method | R@1 | R@3 | R@5 | R@10 | MRR@10 |", "|---|---:|---:|---:|---:|---:|"]
    for method in methods:
        value = metrics[method]
        lines.append("| " + method + " | " + " | ".join(str(value[column]["reaction_micro"]) for column in ("recall_at_1_brite_orthology", "recall_at_3_brite_orthology", "recall_at_5_brite_orthology", "recall_at_10_brite_orthology", "mrr_at_10_brite_orthology")) + " |")
    lines += ["", "## Evidence-set strata", "", "| Method | Seen R@10 (n=847) | Unseen R@10 (n=122) | True Phase 2 retrieval failures R@10 (n=524) | Phase 2 reranking failures R@10 (n=17) |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        lines.append(f"| {method} | {seen[method]['seen']['recall_at_10_exact']['reaction_micro']} | {seen[method]['unseen']['recall_at_10_exact']['reaction_micro']} | {strata[method]['true_retrieval_failure']['recall_at_10_exact']['reaction_micro']} | {strata[method]['rerank_failure']['recall_at_10_exact']['reaction_micro']} |")
    top10 = overlap["at_10"]
    lines += [
        "", "## Component overlap and fusion transitions", "",
        f"At Recall@10: BM25 only {top10['correct_for_bm25_only']}; trained bi-encoder only {top10['correct_for_trained_biencoder_only']}; both {top10['correct_for_both']}; missed by both {top10['missed_by_both']}. Fusion recovered {top10['recovered_by_fusion_beyond_bm25']} beyond BM25 and {top10['recovered_by_fusion_beyond_trained_biencoder']} beyond the trained model, while losing {top10['lost_by_fusion_relative_to_bm25']} BM25 hits and {top10['lost_by_fusion_relative_to_trained_biencoder']} trained-model hits.", "",
        f"The 17-row Phase 2 reranking-failure subset is a real small-stratum tradeoff: fusion R@10 is {strata[FUSION_METHOD]['rerank_failure']['recall_at_10_exact']['reaction_micro']} versus {strata['trained_biencoder_epoch1']['rerank_failure']['recall_at_10_exact']['reaction_micro']} for the trained model.", "",
        "## Paired cluster bootstrap", "",
        "10,000 percentile replicates use seed 20260902 and the 12 frozen validation clusters. Deltas are fusion minus reference reaction-micro recall.", "",
    ]
    for name, comparisons in bootstrap["comparisons"].items():
        for metric_name, value in comparisons.items():
            low, high = value["ci_95_percentile"]
            lines.append(f"- {name}, {metric_name}: {value['delta_selected_minus_reference']:+.6f}, 95% CI [{low:+.6f}, {high:+.6f}]; includes zero: {str(value['includes_zero']).lower()}.")
    lines += [
        "", "Only 12 clusters are available, so percentile intervals may be unstable; no superiority claim is made where zero is included.", "",
        "## Phase 3C recommendation", "", recommendation["rationale"], "",
        f"Recommended Top-10 evidence retriever: `{recommendation['phase3c_evidence_retriever']}`. Pareto frontier over Recall@1, Recall@10, and unseen-target Recall@10: {', '.join(recommendation['pareto_frontier_recall1_recall10_unseen_recall10'])}.", "",
        "This milestone did not start Phase 3C, run held-out test evaluation, train another model, tune fusion, or call an API.",
    ]
    return "\n".join(lines) + "\n"


def evaluate_frozen_fusion(out: Path = FUSION_OUT) -> list[Path]:
    """Load validation truth only after checking the frozen ranking digest."""
    ranking_path = out / FUSION_RANKING_NAME
    freeze_path = out / "ranking_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not freeze.get("frozen_before_ground_truth_join") or sha256_file(ranking_path) != freeze.get("sha256"):
        raise ValueError("fusion ranking must be frozen with a matching digest before label join")
    catalog_ids = list(json.loads(CATALOG_IDS.read_text(encoding="utf-8"))["ids"])
    fusion_rows = _read_xz_jsonl(ranking_path)
    validate_fusion_rows(fusion_rows, catalog_ids, expected=EXPECTED_VALIDATION_SIZE)
    truth = retrieval._truth_and_metadata()
    retrieval.reject_test_rows(truth)
    sources = {
        "phase2_rule_based": retrieval.OUT / "rankings_phase2_rule_based.jsonl",
        "bm25": BM25_RANKINGS,
        "bge_m3_dense": retrieval.OUT / "rankings_bge_m3_dense.jsonl",
        "bm25_bge_m3_rrf": retrieval.OUT / "rankings_bm25_bge_m3_rrf.jsonl",
        "trained_biencoder_epoch1": SELECTED_RANKINGS,
    }
    frames: dict[str, pd.DataFrame] = {}
    rank_maps: dict[str, dict[tuple[str, str], list[str]]] = {}
    for method, path in sources.items():
        frames[method], rank_maps[method] = biencoder.score_reference_rankings(path, truth, method=method)
    frames[FUSION_METHOD], rank_maps[FUSION_METHOD] = (
        retrieval.score_rankings(fusion_rows, truth),
        {(row["model_id"], row["reaction_id"]): list(row["ranked_ids"]) for row in fusion_rows},
    )
    if not frames[FUSION_METHOD].method.eq(FUSION_METHOD).all():
        raise ValueError("fusion scoring method mismatch")
    method_order = ["phase2_rule_based", "bm25", "bge_m3_dense", "bm25_bge_m3_rrf", "trained_biencoder_epoch1", FUSION_METHOD]
    frames = {method: frames[method] for method in method_order}
    rank_maps = {method: rank_maps[method] for method in method_order}
    metrics = {method: _metric_summary(frame) for method, frame in frames.items()}
    seen = {
        method: {
            "seen": _metric_summary(frame[frame.seen_in_train]),
            "unseen": _metric_summary(frame[~frame.seen_in_train]),
        }
        for method, frame in frames.items()
    }
    strata = {
        method: {
            "true_retrieval_failure": _metric_summary(frame[frame.stratum.isin(TRUE_RETRIEVAL_FAILURE_STRATA)]),
            "rerank_failure": _metric_summary(frame[frame.stratum.eq("retrievable_rerank_failure")]),
        }
        for method, frame in frames.items()
    }
    overlap = _overlap_and_transitions(frames, rank_maps)
    bootstrap = _bootstrap(frames)
    recommendation = _recommendation(metrics, seen, bootstrap)
    examples = _qualitative_examples(truth, frames, rank_maps)
    payloads = {
        "metrics_by_method.json": {"scope": "validation-only", "n_reactions": len(truth), "methods": metrics, "test_rows_read": 0},
        "seen_unseen_analysis.json": seen,
        "stratum_analysis.json": strata,
        "overlap_transition_analysis.json": overlap,
        "bootstrap_comparisons.json": bootstrap,
        "qualitative_examples.json": examples,
        "recommendation.json": recommendation,
        "safety_audit.json": {
            "ranking_labels_loaded": False,
            "ranking_frozen_sha256_before_label_join": freeze["sha256"],
            "evaluation_split": "validation",
            "validation_reactions": len(truth),
            "test_rows_read_ranked_or_scored": 0,
            "parameter_tuning": False,
            "reencoding": False,
            "retraining": False,
            "paid_api_calls": 0,
        },
    }
    paths = []
    for name, payload in payloads.items():
        path = out / name
        write_json(payload, path)
        paths.append(path)
    report = out / "REPORT.md"
    report.write_text(_report(metrics, seen, strata, overlap, bootstrap, recommendation), encoding="utf-8", newline="\n")
    paths.append(report)
    if sha256_file(ranking_path) != freeze["sha256"]:
        raise RuntimeError("fusion ranking changed during evaluation")
    return paths


def rebuild_fusion_twice(out: Path = FUSION_OUT) -> dict[str, Any]:
    """Rebuild every fusion artifact twice and install only byte-identical output."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_phase3b_fusion_a_", dir=out.parent) as first_name, tempfile.TemporaryDirectory(prefix="_phase3b_fusion_b_", dir=out.parent) as second_name:
        first, second = Path(first_name), Path(second_name)
        freeze_fusion_rankings(first)
        evaluate_frozen_fusion(first)
        freeze_fusion_rankings(second)
        evaluate_frozen_fusion(second)
        first_files = {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()}
        second_files = {path.name: path.read_bytes() for path in second.iterdir() if path.is_file()}
        if first_files != second_files:
            changed = sorted(name for name in set(first_files) | set(second_files) if first_files.get(name) != second_files.get(name))
            raise RuntimeError(f"fusion rebuild was not byte-identical: {changed}")
        out.mkdir(parents=True, exist_ok=True)
        for name, blob in first_files.items():
            destination = out / name
            tmp = destination.with_name(destination.name + ".tmp")
            tmp.write_bytes(blob)
            _replace_with_retry(tmp, destination)
    verification = {
        "passes": 2,
        "byte_identical": True,
        "files": [{"path": f"benchmark/phase3/phase3b_fusion/{name}", "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)} for name, blob in sorted(first_files.items())],
    }
    write_json(verification, out / "rebuild_verification.json")
    artifacts = [path for path in out.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    write_artifact_manifest(out, artifacts)
    return verification


def verify_fusion_manifest(out: Path = FUSION_OUT) -> list[str]:
    manifest = json.loads((out / "artifact_manifest.json").read_text(encoding="utf-8"))
    paths = [item["path"] for item in manifest["files"]]
    problems = []
    if len(paths) != len(set(paths)):
        problems.append("duplicate paths")
    for item in manifest["files"]:
        if "\\" in item["path"]:
            problems.append(f"non-POSIX path: {item['path']}")
        path = REPO_ROOT / item["path"]
        if not path.exists() or sha256_portable(path) != item["sha256"]:
            problems.append(f"digest mismatch: {item['path']}")
    return problems


def _archive_environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def _encode_local(model: Any, tokenizer: Any, texts: Sequence[str], device: Any, *, batch_size: int = 64) -> np.ndarray:
    return biencoder._encode_texts(model, tokenizer, texts, device, max_length=256, batch_size=batch_size)


def _fixture_inputs() -> dict[str, Any]:
    return {
        "queries": [
            "Equation: ATP + H2O => ADP + phosphate\nParticipants: ATP; water; ADP; phosphate",
            "Equation: glucose + oxygen => gluconolactone + hydrogen peroxide\nParticipants: glucose; oxygen; product",
        ],
        "documents": [
            {"id": "D1", "text": "Definition: ATP hydrolysis to ADP and phosphate\nNames: ATP phosphohydrolase reaction"},
            {"id": "D2", "text": "Definition: glucose oxidation with oxygen\nNames: glucose oxidase reaction"},
            {"id": "D3", "text": "Definition: amino acid transamination\nNames: aminotransferase reaction"},
            {"id": "D4", "text": "Definition: ADP phosphorylation to ATP\nNames: ATP synthase reaction"},
        ],
    }


def _load_archive_model(model_dir: Path, *, device_name: str = "cpu") -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
    device = torch.device(device_name)
    model.to(device)
    return model, tokenizer, device


def _make_fixture(model_dir: Path) -> dict[str, Any]:
    fixture = _fixture_inputs()
    model, tokenizer, device = _load_archive_model(model_dir, device_name="cpu")
    queries = _encode_local(model, tokenizer, fixture["queries"], device, batch_size=2)
    documents = _encode_local(model, tokenizer, [item["text"] for item in fixture["documents"]], device, batch_size=4)
    ids = [item["id"] for item in fixture["documents"]]
    rankings = retrieval.dense_rank(queries, documents, ids, topn=len(ids))
    fixture.update({
        "pooling": "cls",
        "normalization": "l2",
        "max_length": 256,
        "similarity": "inner product over normalized embeddings",
        "dtype_and_device": "float32 CPU",
        "query_embedding_sha256": hashlib.sha256(np.asarray(queries, dtype="<f4").tobytes()).hexdigest(),
        "document_embedding_sha256": hashlib.sha256(np.asarray(documents, dtype="<f4").tobytes()).hexdigest(),
        "expected_rankings": rankings,
        "comparison": "embedding digests are exact for the pinned environment; ranking equality is the cross-platform portable check",
    })
    return fixture


def _archive_readme() -> str:
    return f"""# AAAIM Phase 3B selected epoch-1 inference model

This archive is a local Hugging Face-compatible BERT model exported from the
validation-selected `{biencoder.MODEL_NAME}` epoch-1 checkpoint. It contains no
optimizer, scheduler, gradient-scaler, training cursor, or resume state and does
not require the source checkpoint or a Hugging Face cache.

Install a suitable PyTorch build, then the exact inference dependencies in
`requirements-inference.txt`. Load locally:

```python
from transformers import AutoModel, AutoTokenizer
import torch

path = "."
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModel.from_pretrained(path, local_files_only=True).eval()
texts = ["Equation: A => B\\nParticipants: A; B"]
tokens = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
with torch.no_grad():
    vectors = model(**tokens).last_hidden_state[:, 0]
vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
```

Use `phase3-retrieval-query-v1` and `phase3-retrieval-document-v1` as specified
in `inference_config.json`. Rank by descending inner product and resolve exact
ties by ascending document ID. Verify all payload files against
`archive_manifest.json`; that manifest excludes only itself to avoid a recursive
self-digest.

Resuming optimization would additionally require a full `.pt` checkpoint with
optimizer, scheduler, gradient scaler, training state/cursor, RNG-compatible
ordering, and the exact training configuration/dataset. Those are intentionally
absent because this is inference-only.
"""


def _stage_archive(stage: Path) -> None:
    """Materialize one deterministic inference-only archive tree."""
    verify_selected_checkpoint()
    import torch
    from safetensors.torch import save_file

    if not MODEL_CACHE_SNAPSHOT.is_dir():
        raise FileNotFoundError(f"pinned local model snapshot missing: {MODEL_CACHE_SNAPSHOT}")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload.get("schema") != biencoder.CHECKPOINT_SCHEMA or payload.get("config_hash") != EXPECTED_CONFIG_HASH or payload.get("dataset_hash") != EXPECTED_DATASET_HASH:
        raise ValueError("checkpoint payload provenance mismatch")
    stage.mkdir(parents=True, exist_ok=True)
    config = json.loads((MODEL_CACHE_SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    config["_name_or_path"] = biencoder.MODEL_NAME
    config["_commit_hash"] = biencoder.MODEL_REVISION
    (stage / "config.json").write_bytes(_json_bytes(config))
    for name in ("tokenizer.json", "vocab.txt", "special_tokens_map.json"):
        shutil.copyfile(MODEL_CACHE_SNAPSHOT / name, stage / name)
    tokenizer_config = json.loads((MODEL_CACHE_SNAPSHOT / "tokenizer_config.json").read_text(encoding="utf-8"))
    tokenizer_config["model_max_length"] = 256
    (stage / "tokenizer_config.json").write_bytes(_json_bytes(tokenizer_config))
    state = {key: value.detach().cpu().contiguous() for key, value in sorted(payload["model_state"].items())}
    save_file(state, stage / "model.safetensors", metadata={"format": "pt"})
    dataset = json.loads((biencoder.FULL_OUT / "dataset_summary.json").read_text(encoding="utf-8"))
    selected_metrics = json.loads((biencoder.FULL_OUT / "metrics_epoch_1.json").read_text(encoding="utf-8"))
    metadata = {
        "schema": RELEASE_SCHEMA,
        "model": biencoder.MODEL_NAME,
        "initializer_revision": biencoder.MODEL_REVISION,
        "selected_epoch": EXPECTED_EPOCH,
        "architecture": biencoder.MODEL_ARCHITECTURE,
        "pooling": "cls",
        "normalization": "l2",
        "max_sequence_length": 256,
        "similarity": "inner product over L2-normalized embeddings",
        "query_template": {"version": biencoder.QUERY_TEMPLATE, "format": "Equation: {normalized SBML reaction equation}\\nParticipants: {name} [species={SBML species id}; ChEBI={ids}; KEGG-compound={ids}]; ..."},
        "document_template": {"version": biencoder.DOCUMENT_TEMPLATE, "fields": ["DEFINITION", "NAME", "EQUATION", "ENZYME", "RCLASS", "BRITE"], "document_id_searchable": False},
        "validation_metrics": {kind: {metric: value for metric, value in selected_metrics[kind].items()} for kind in ("exact", "brite_orthology")},
        "source_checkpoint": {"sha256": EXPECTED_CHECKPOINT_SHA256, "bytes": CHECKPOINT.stat().st_size, "config_hash": EXPECTED_CONFIG_HASH, "dataset_hash": EXPECTED_DATASET_HASH},
        "catalog": {"path": repo_relative_posix(retrieval.CATALOG), "sha256": sha256_file(retrieval.CATALOG), "count": EXPECTED_CATALOG_SIZE},
        "relevant_input_digests": dataset["source_digests"] | {
            repo_relative_posix(CATALOG_IDS): sha256_portable(CATALOG_IDS),
            repo_relative_posix(SELECTED_RANKINGS): sha256_file(SELECTED_RANKINGS),
            repo_relative_posix(biencoder.FULL_OUT / "full_training_config.json"): sha256_portable(biencoder.FULL_OUT / "full_training_config.json"),
        },
        "excluded_training_state": ["optimizer", "scheduler", "gradient scaler", "training cursor", "RNG state", "original Hugging Face cache"],
    }
    (stage / "inference_config.json").write_bytes(_json_bytes(metadata))
    environment = {"export_environment": _archive_environment(), "training_environment": json.loads((biencoder.FULL_OUT / "environment.json").read_text(encoding="utf-8"))}
    (stage / "environment.json").write_bytes(_json_bytes(environment))
    requirements = "transformers==5.16.1\ntokenizers==0.23.1\nhuggingface-hub==1.29.0\nsafetensors==0.8.0\nnumpy==2.2.2\n"
    (stage / "requirements-inference.txt").write_text(requirements, encoding="utf-8", newline="\n")
    (stage / "README.md").write_text(_archive_readme(), encoding="utf-8", newline="\n")
    fixture = _make_fixture(stage)
    (stage / "inference_fixture.json").write_bytes(_json_bytes(fixture))
    files = []
    for path in sorted(stage.iterdir(), key=lambda item: item.name):
        if path.name == "archive_manifest.json" or not path.is_file():
            continue
        files.append({"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {"schema": RELEASE_SCHEMA, "self_exclusion": "archive_manifest.json is excluded because a file cannot contain its own stable digest", "n_payload_files": len(files), "files": files}
    (stage / "archive_manifest.json").write_bytes(_json_bytes(manifest))


def _write_deterministic_zip(stage: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, strict_timestamps=True) as archive:
        for path in sorted(stage.iterdir(), key=lambda item: item.name):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    _replace_with_retry(tmp, destination)


def verify_archive_contents(archive_path: Path = ARCHIVE_PATH) -> list[str]:
    problems = []
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        required = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "inference_config.json", "environment.json", "requirements-inference.txt", "README.md", "inference_fixture.json", "archive_manifest.json"}
        if set(names) != required:
            problems.append(f"archive member mismatch: {sorted(set(names) ^ required)}")
        if len(names) != len(set(names)):
            problems.append("duplicate archive members")
        manifest = json.loads(archive.read("archive_manifest.json"))
        for item in manifest["files"]:
            blob = archive.read(item["path"])
            if len(blob) != item["bytes"] or hashlib.sha256(blob).hexdigest() != item["sha256"]:
                problems.append(f"archive payload digest mismatch: {item['path']}")
        forbidden = {"best.pt", "optimizer.pt", "scheduler.pt", "scaler.pt", "training_state.json", "cursor.json"}
        if forbidden & set(names) or any("checkpoint" in name.lower() for name in names):
            problems.append("training/checkpoint state present in inference archive")
    return problems


def build_archive_twice(archive_path: Path = ARCHIVE_PATH) -> dict[str, Any]:
    """Require two independently staged, byte-identical deterministic ZIPs."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_phase3b_archive_", dir=archive_path.parent) as temp_name:
        root = Path(temp_name)
        stages = [root / "stage_a", root / "stage_b"]
        zips = [root / "a.zip", root / "b.zip"]
        for stage, target in zip(stages, zips):
            _stage_archive(stage)
            _write_deterministic_zip(stage, target)
        if zips[0].read_bytes() != zips[1].read_bytes():
            raise RuntimeError("independent inference ZIP builds were not byte-identical")
        shutil.copyfile(zips[0], archive_path)
    problems = verify_archive_contents(archive_path)
    if problems:
        raise ValueError(f"archive content verification failed: {problems}")
    with zipfile.ZipFile(archive_path) as archive:
        uncompressed = sum(item.file_size for item in archive.infolist())
    registry = {
        "schema": RELEASE_SCHEMA,
        "archive": {"filename": archive_path.name, "gitignored_path": repo_relative_posix(archive_path), "sha256": sha256_file(archive_path), "compressed_bytes": archive_path.stat().st_size, "uncompressed_bytes": uncompressed},
        "deterministic_build": {"independent_passes": 2, "byte_identical": True, "fixed_zip_timestamp": "1980-01-01T00:00:00", "compression": "deflate level 9"},
        "selected_model": verify_selected_checkpoint(),
        "contents_verified": True,
        "restoration": {"performed": False, "fixture_reproduced": False, "complete_validation_rankings_reproduced": False},
        "upload_performed": False,
    }
    write_json(registry, ARCHIVE_REGISTRY)
    return registry


def _verify_restored_fixture(model_dir: Path) -> dict[str, Any]:
    expected = json.loads((model_dir / "inference_fixture.json").read_text(encoding="utf-8"))
    actual = _make_fixture(model_dir)
    if actual["expected_rankings"] != expected["expected_rankings"]:
        raise RuntimeError("restored inference fixture ranking mismatch")
    exact_digests = actual["query_embedding_sha256"] == expected["query_embedding_sha256"] and actual["document_embedding_sha256"] == expected["document_embedding_sha256"]
    return {"rankings_exact": True, "embedding_digests_exact": exact_digests, "expected_rankings": actual["expected_rankings"]}


def restore_and_verify(archive_path: Path = ARCHIVE_PATH, *, batch_size: int = 64) -> dict[str, Any]:
    """Clean-room local load and exact ranked-ID regeneration on validation."""
    problems = verify_archive_contents(archive_path)
    if problems:
        raise ValueError(f"archive verification failed before restore: {problems}")
    prior_offline = {name: os.environ.get(name) for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        with tempfile.TemporaryDirectory(prefix="_phase3b_restore_", dir=PHASE3_DIR) as temp_name:
            restored = Path(temp_name) / "model"
            restored.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(restored)
            fixture = _verify_restored_fixture(restored)
            import torch

            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            model, tokenizer, device = _load_archive_model(restored, device_name=device_name)
            population = retrieval.load_query_population("validation")
            queries = [retrieval.query_text(row) for row in population.to_dict("records")]
            documents = retrieval.load_catalog()
            document_embeddings = _encode_local(model, tokenizer, [item["text"] for item in documents], device, batch_size=batch_size)
            query_embeddings = _encode_local(model, tokenizer, queries, device, batch_size=batch_size)
            ranks = retrieval.dense_rank(query_embeddings, document_embeddings, [item["kegg_id"] for item in documents], topn=100)
            regenerated = [{"schema": biencoder.FULL_RANKING_SCHEMA, "epoch": 1, "model_id": row["model_id"], "reaction_id": row["reaction_id"], "split": "validation", "ranked_ids": ids} for row, ids in zip(population.to_dict("records"), ranks)]
            biencoder.validate_full_ranking_rows(regenerated, epoch=1)
            expected = _read_jsonl(SELECTED_RANKINGS)
            exact = all(a["model_id"] == b["model_id"] and a["reaction_id"] == b["reaction_id"] and a["ranked_ids"] == b["ranked_ids"] for a, b in zip(regenerated, expected)) and len(regenerated) == len(expected)
            if not exact:
                mismatch = next((index for index, (a, b) in enumerate(zip(regenerated, expected)) if a["ranked_ids"] != b["ranked_ids"]), None)
                raise RuntimeError(f"restored model did not reproduce epoch-1 ranked KEGG IDs; first mismatch row: {mismatch}")
            result = {
                "archive_sha256": sha256_file(archive_path),
                "fresh_temporary_directory": True,
                "offline_environment": True,
                "loaded_from_extracted_archive_only": True,
                "source_checkpoint_accessed_for_restore": False,
                "source_model_cache_accessed_for_restore": False,
                "optimizer_scheduler_scaler_or_cursor_present": False,
                "fixture": fixture,
                "validation": {"n_queries": len(regenerated), "catalog_size": len(documents), "rank_depth": 100, "ranked_kegg_ids_exact": True, "expected_sha256": EXPECTED_RANKING_SHA256},
                "device": device_name,
                "test_rows_read": 0,
            }
    finally:
        for name, value in prior_offline.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    write_json(result, DIST_DIR / "aaaim-phase3b-selected-epoch1-inference.restoration.json")
    registry = json.loads(ARCHIVE_REGISTRY.read_text(encoding="utf-8"))
    registry["restoration"] = {"performed": True, "fixture_reproduced": True, "complete_validation_rankings_reproduced": True, "verification_path": "benchmark/dist/aaaim-phase3b-selected-epoch1-inference.restoration.json"}
    write_json(registry, ARCHIVE_REGISTRY)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify-selected", "build-archive", "verify-archive", "restore", "fusion", "verify-fusion"))
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--out", type=Path, default=FUSION_OUT)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    if args.command == "verify-selected":
        print(json.dumps(verify_selected_checkpoint(), sort_keys=True))
    elif args.command == "build-archive":
        print(json.dumps(build_archive_twice(args.archive), sort_keys=True))
    elif args.command == "verify-archive":
        problems = verify_archive_contents(args.archive)
        print(json.dumps({"archive_sha256": sha256_file(args.archive), "problems": problems, "n_problems": len(problems)}, sort_keys=True))
        return int(bool(problems))
    elif args.command == "restore":
        print(json.dumps(restore_and_verify(args.archive, batch_size=args.batch), sort_keys=True))
    elif args.command == "fusion":
        print(json.dumps(rebuild_fusion_twice(args.out), sort_keys=True))
    else:
        problems = verify_fusion_manifest(args.out)
        print(json.dumps({"problems": problems, "n_problems": len(problems)}, sort_keys=True))
        return int(bool(problems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
