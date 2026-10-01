"""Leakage-safe Phase 3B bi-encoder smoke and prespecified full training.

The module intentionally keeps the held-out test split inaccessible.  Training
examples are built only after train keys have been selected, and validation
rankings are atomically frozen before validation labels are loaded offline.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import importlib.metadata
import json
import lzma
import math
import os
import pickle
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from benchmark.scripts import phase3_retrieval as retrieval
from benchmark.scripts.kegg_equivalence import EC_RE, KO_RE
from benchmark.scripts.phase3_common import (
    PHASE3_DIR,
    REPO_ROOT,
    TRUE_RETRIEVAL_FAILURE_STRATA,
    _replace_with_retry,
    assert_no_kegg_leakage,
    atomic_write_jsonl,
    parse_kegg_ids,
    parse_participant_ids,
    repo_relative_posix,
    sha256_file,
    sha256_portable,
    write_artifact_manifest,
    write_json,
)

OUT = PHASE3_DIR / "phase3b_smoke"
FULL_OUT = PHASE3_DIR / "phase3b_full"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5e62ea33e012fda8c02802b906664c915ebd1bb1"
MODEL_LICENSE = "MIT"
MODEL_ARCHITECTURE = "BertModel; 12 layers; hidden size 384; 12 heads"
MODEL_PARAMETERS = 33_360_000
QUERY_TEMPLATE = retrieval.QUERY_TEMPLATE_VERSION
DOCUMENT_TEMPLATE = retrieval.DOCUMENT_TEMPLATE_VERSION
DATASET_SCHEMA = "phase3b-training-examples-v1"
CHECKPOINT_SCHEMA = "phase3b-biencoder-checkpoint-v1"
RANKING_SCHEMA = "phase3b-validation-ranking-v1"
FULL_RANKING_SCHEMA = "phase3b-full-validation-ranking-v1"
FULL_RUN_SCHEMA = "phase3b-full-run-v1"
DEFAULT_SEED = 20260909
BOOTSTRAP_SEED = 20260902


@functools.lru_cache(maxsize=1)
def _brite_orthology_groups() -> dict[str, frozenset[str]]:
    """Load EC/KO groups directly, avoiding unrelated application dependencies."""
    raw = pickle.loads(lzma.open(retrieval.CATALOG, "rb").read())
    groups: dict[str, frozenset[str]] = {}
    for kegg_id, fields in raw.items():
        enzyme = str(fields.get("ENZYME", "") or "")
        orthology = str(fields.get("ORTHOLOGY", "") or "")
        groups[str(kegg_id)] = frozenset(
            [*(f"EC:{item}" for item in EC_RE.findall(enzyme)),
             *(f"KO:{item}" for item in KO_RE.findall(orthology))]
        )
    return groups


def is_equivalent(candidate: str, truth_ids: Iterable[str], kind: str) -> bool:
    if kind != "brite_orthology":
        raise ValueError(f"unsupported conservative sibling kind: {kind}")
    groups = _brite_orthology_groups()
    candidate_groups = groups.get(str(candidate), frozenset())
    return bool(candidate_groups and any(
        candidate_groups & groups.get(str(item), frozenset()) for item in truth_ids
    ))


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _package_versions() -> dict[str, str | None]:
    names = ["torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy", "pandas", "pytest"]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_preflight(*, verify_tensor: bool = False) -> dict[str, Any]:
    import torch

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        smi = f"unavailable: {exc}"
    available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if available else None
    total_vram = int(torch.cuda.get_device_properties(0).total_memory) if available else None
    capability = list(torch.cuda.get_device_capability(0)) if available else None
    tensor_result = None
    if verify_tensor:
        if not available:
            raise RuntimeError("CUDA tensor verification requested but torch.cuda.is_available() is false")
        left = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
        right = torch.eye(4, device="cuda", dtype=torch.float32)
        tensor_result = {"device": str(left.device), "sum": float((left @ right).sum().cpu())}
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": available,
        "gpu_name": gpu_name,
        "total_vram_bytes": total_vram,
        "compute_capability": capability,
        "mixed_precision_fp16_supported": bool(available and capability and capability[0] >= 7),
        "mixed_precision_bf16_supported": bool(available and torch.cuda.is_bf16_supported()),
        "nvidia_smi": smi,
        "cuda_tensor_operation": tensor_result,
    }


def record_environment(out: Path, stage: str, *, verify_tensor: bool = False) -> dict[str, Any]:
    path = out / "environment.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload[stage] = environment_preflight(verify_tensor=verify_tensor)
    write_json(payload, path)
    return payload[stage]


def _rows_for_keys(path: Path, keys: set[tuple[str, str]], columns: Sequence[str]) -> pd.DataFrame:
    return retrieval._csv_rows_for_keys(path, keys, columns)


def _participant_evidence(selected: pd.DataFrame, visible: pd.DataFrame) -> dict[tuple[str, str], str]:
    model_ids = set(selected.model_id)
    names = pd.read_csv(retrieval.SPECIES_NAMES, dtype=str).fillna("")
    names = names[names.model_id.isin(model_ids)]
    name_map = {(r.model_id, r.species_id): r.species_name for r in names.itertuples()}
    evidence = pd.read_csv(retrieval.SPECIES_EVIDENCE, dtype=str).fillna("")
    evidence = evidence[evidence.model_id.isin(model_ids)]
    ev: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: {"chebi": [], "kegg_compound": []})
    for row in evidence.itertuples():
        bucket = ev[(row.model_id, row.species_id)]
        if row.annotation_type in bucket and row.annotation not in bucket[row.annotation_type]:
            bucket[row.annotation_type].append(row.annotation)
    merged = selected[["model_id", "reaction_id"]].merge(visible, on=["model_id", "reaction_id"], validate="one_to_one")
    blocks: dict[tuple[str, str], str] = {}
    for row in merged.itertuples():
        items = []
        for species_id in parse_participant_ids(row.reaction_equation):
            details = [f"species={species_id}"]
            annotations = ev[(row.model_id, species_id)]
            if annotations["chebi"]:
                details.append("ChEBI=" + ",".join(annotations["chebi"]))
            if annotations["kegg_compound"]:
                details.append("KEGG-compound=" + ",".join(annotations["kegg_compound"]))
            items.append(f"{name_map.get((row.model_id, species_id), species_id)} [{'; '.join(details)}]")
        blocks[(row.model_id, row.reaction_id)] = "; ".join(items)
    return blocks


def load_training_rows(split: str = "train") -> pd.DataFrame:
    """Load train labels only; validation and test are explicitly rejected."""
    if split != "train":
        raise ValueError("bi-encoder training accepts only the frozen train split")
    assignments = pd.read_csv(retrieval.SPLITS, dtype=str)
    selected = assignments.loc[
        assignments.split.eq("train"),
        ["model_id", "reaction_id", "cluster_id", "split", "stratum", "status", "candidate_set_size"],
    ].copy()
    expected = int(json.loads(retrieval.SPLIT_SUMMARY.read_text(encoding="utf-8"))["splits"]["train"]["n_reactions"])
    if len(selected) != expected or expected != 3497:
        raise ValueError(f"frozen train count mismatch: {len(selected)} != {expected}")
    if not selected.split.eq("train").all():
        raise ValueError("validation/test row selected for training")
    keys = set(zip(selected.model_id, selected.reaction_id))
    visible = _rows_for_keys(
        retrieval.REACTION_TEXT, keys,
        ["model_id", "reaction_id", "reaction_equation", "substrate_names", "product_names", "query_text"],
    )
    participant_blocks = _participant_evidence(selected, visible)
    visible["participant_evidence"] = [
        participant_blocks[(row.model_id, row.reaction_id)] for row in visible.itertuples()
    ]
    labels = _rows_for_keys(
        retrieval.REACTIONS, keys,
        ["model_id", "reaction_id", "ground_truth_kegg_all"],
    )
    out = selected.merge(visible, on=["model_id", "reaction_id"], validate="one_to_one")
    out = out.merge(labels, on=["model_id", "reaction_id"], validate="one_to_one")
    out["positive_ids"] = out.ground_truth_kegg_all.map(parse_kegg_ids)
    if out.positive_ids.map(len).eq(0).any():
        raise ValueError("train row without a valid positive")
    return out.sort_values(["model_id", "reaction_id"]).reset_index(drop=True)


def assert_train_only(examples: Sequence[Mapping[str, Any]]) -> None:
    wrong = sorted({str(x.get("split")) for x in examples if x.get("split") != "train"})
    if wrong:
        raise ValueError(f"validation/test examples rejected from training: {wrong}")


def validate_negative_list(example: Mapping[str, Any]) -> None:
    positives = set(example["positive_ids"])
    negatives = list(example["negatives"])
    ids = [str(x["kegg_id"]) for x in negatives]
    overlap = positives & set(ids)
    if overlap:
        raise ValueError(f"positive IDs present in negatives: {sorted(overlap)}")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate negatives")
    if any(bool(x.get("uncertain_brite_orthology_sibling")) for x in negatives):
        raise ValueError("uncertain BRITE/orthology sibling included as a definitive negative")


def _random_negative_ids(
    catalog_ids: Sequence[str], excluded: set[str], count: int, *, seed: int, key: tuple[str, str],
) -> list[str]:
    digest = hashlib.sha256(f"{seed}:{key[0]}:{key[1]}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    candidates = [item for item in catalog_ids if item not in excluded]
    rng.shuffle(candidates)
    return candidates[:count]


def build_training_examples(
    *, bm25_hard_negatives: int = 2, random_negatives: int = 1, seed: int = DEFAULT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows = load_training_rows()
    docs = retrieval.load_catalog()
    doc_ids = [d["kegg_id"] for d in docs]
    doc_set = set(doc_ids)
    bm25 = retrieval.BM25(docs)
    examples: list[dict[str, Any]] = []
    excluded_without_catalog_positive: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    uncertain_excluded = 0
    fewer_than_requested = 0
    for row in rows.to_dict("records"):
        query = retrieval.query_text({
            "reaction_equation": row["reaction_equation"],
            "substrate_names": row["substrate_names"],
            "product_names": row["product_names"],
            "query_text": row["query_text"],
            "participant_evidence": row["participant_evidence"],
        })
        positives = list(dict.fromkeys(row["positive_ids"]))
        available_positives = [item for item in positives if item in doc_set]
        missing_positives = [item for item in positives if item not in doc_set]
        if missing_positives:
            if available_positives:
                raise ValueError(
                    "partial multi-positive catalog coverage would discard a valid answer: "
                    f"{row['model_id']}/{row['reaction_id']}"
                )
            excluded_without_catalog_positive.append({
                "model_id": row["model_id"],
                "reaction_id": row["reaction_id"],
                "cluster_id": row["cluster_id"],
                "positive_ids": positives,
                "reason": "no positive document in frozen KEGG catalog",
            })
            continue
        assert_no_kegg_leakage(query, where="Phase 3B training query")
        ranked = bm25.rank(query, topn=100)
        negatives: list[dict[str, Any]] = []
        first_positive_rank = next((i for i, kid in enumerate(ranked, 1) if kid in positives), None)
        for rank, candidate in enumerate(ranked, 1):
            if candidate in positives:
                continue
            uncertain = is_equivalent(candidate, positives, "brite_orthology")
            if uncertain:
                uncertain_excluded += 1
                continue
            negatives.append({
                "kegg_id": candidate,
                "source": "bm25",
                "source_rank": rank,
                "uncertain_brite_orthology_sibling": False,
            })
            if len(negatives) == bm25_hard_negatives:
                break
        if len(negatives) < bm25_hard_negatives:
            fewer_than_requested += 1
        excluded = set(positives) | {x["kegg_id"] for x in negatives}
        random_ids = _random_negative_ids(
            doc_ids, excluded, random_negatives, seed=seed,
            key=(row["model_id"], row["reaction_id"]),
        )
        for candidate in random_ids:
            # Random controls follow the same conservative sibling policy.
            if is_equivalent(candidate, positives, "brite_orthology"):
                uncertain_excluded += 1
                replacements = _random_negative_ids(
                    doc_ids, excluded | set(random_ids), random_negatives * 4 + 8,
                    seed=seed + 1, key=(row["model_id"], row["reaction_id"]),
                )
                candidate = next(
                    (x for x in replacements if not is_equivalent(x, positives, "brite_orthology")),
                    "",
                )
                if not candidate:
                    continue
            if candidate in excluded:
                continue
            negatives.append({
                "kegg_id": candidate,
                "source": "random",
                "source_rank": None,
                "uncertain_brite_orthology_sibling": False,
            })
            excluded.add(candidate)
        example = {
            "schema": DATASET_SCHEMA,
            "model_id": row["model_id"],
            "reaction_id": row["reaction_id"],
            "cluster_id": row["cluster_id"],
            "split": "train",
            "phase2_stratum": row["stratum"],
            "query_template": QUERY_TEMPLATE,
            "document_template": DOCUMENT_TEMPLATE,
            "query": query,
            "positive_ids": positives,
            "negatives": negatives,
            "bm25_first_positive_rank": first_positive_rank,
        }
        validate_negative_list(example)
        for negative in negatives:
            source_counts[negative["source"]] += 1
        examples.append(example)
    assert_train_only(examples)
    assert_no_kegg_leakage([x["query"] for x in examples], where="all Phase 3B training queries")
    dataset_hash = stable_hash(examples)
    target_clusters: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        for target in example["positive_ids"]:
            target_clusters[target].add(example["cluster_id"])
    dataset_summary = {
        "schema": DATASET_SCHEMA,
        "dataset_sha256": dataset_hash,
        "split": "train",
        "n_examples": len(examples),
        "n_frozen_train_rows": len(rows),
        "n_excluded_no_catalog_positive": len(excluded_without_catalog_positive),
        "excluded_no_catalog_positive": excluded_without_catalog_positive,
        "n_unique_queries": len({(x["model_id"], x["reaction_id"]) for x in examples}),
        "n_unique_target_ids": len(target_clusters),
        "n_multi_positive_queries": sum(len(x["positive_ids"]) > 1 for x in examples),
        "max_positives_per_query": max((len(x["positive_ids"]) for x in examples), default=0),
        "n_clusters": len({x["cluster_id"] for x in examples}),
        "n_targets_seen_across_clusters": sum(len(v) > 1 for v in target_clusters.values()),
        "query_template": QUERY_TEMPLATE,
        "document_template": DOCUMENT_TEMPLATE,
        "model_visible_kegg_ids": 0,
        "materialization": "deterministically rebuilt from frozen sources; full redundant JSONL not committed",
        "source_digests": {
            repo_relative_posix(path): sha256_portable(path)
            for path in (
                retrieval.SPLITS, retrieval.REACTIONS, retrieval.REACTION_TEXT,
                retrieval.SPECIES_EVIDENCE, retrieval.SPECIES_NAMES, retrieval.CATALOG,
            )
        },
    }
    negative_summary = {
        "dataset_sha256": dataset_hash,
        "requested_per_query": {"bm25": bm25_hard_negatives, "random": random_negatives},
        "source_counts": dict(sorted(source_counts.items())),
        "n_total_negatives": sum(source_counts.values()),
        "n_duplicate_negatives": 0,
        "n_positive_negative_overlaps": 0,
        "n_uncertain_brite_orthology_siblings_excluded": uncertain_excluded,
        "n_queries_with_fewer_bm25_negatives_than_requested": fewer_than_requested,
        "sibling_policy": "exclude candidates sharing EC or KO with any valid positive",
        "dense_rrf_hard_negatives": {
            "used": False,
            "reason": "frozen dense/RRF rankings exist only for validation; using them would not provide train-query negatives",
        },
        "random_policy": "deterministic per-query catalog control; all positives and uncertain EC/KO siblings excluded",
    }
    return examples, dataset_summary, negative_summary


def write_dataset_artifacts(out: Path, examples: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], negatives: Mapping[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(dict(summary), out / "dataset_summary.json")
    write_json(dict(negatives), out / "negative_summary.json")
    manifest = {
        "schema": DATASET_SCHEMA,
        "dataset_sha256": summary["dataset_sha256"],
        "example_count": len(examples),
        "ordering": "model_id, reaction_id ascending",
        "rebuild_command": "python -m benchmark.scripts.phase3_biencoder build",
        "full_examples_committed": False,
    }
    write_json(manifest, out / "dataset_manifest.json")


def select_representative_examples(examples: Sequence[Mapping[str, Any]], count: int = 160) -> list[dict[str, Any]]:
    if count <= 0 or count > len(examples):
        raise ValueError("invalid representative sample size")
    ordered = sorted((dict(x) for x in examples), key=lambda x: (x["model_id"], x["reaction_id"]))
    chosen: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(candidates: Iterable[dict[str, Any]], limit: int | None = None) -> None:
        added = 0
        for item in candidates:
            key = (item["model_id"], item["reaction_id"])
            if key not in seen and len(chosen) < count:
                chosen.append(item)
                seen.add(key)
                added += 1
                if limit is not None and added >= limit:
                    break

    add(x for x in ordered if len(x["positive_ids"]) > 1)
    for cluster in sorted({x["cluster_id"] for x in ordered}):
        add((x for x in ordered if x["cluster_id"] == cluster), limit=1)
    for stratum in sorted({x["phase2_stratum"] for x in ordered}):
        add((x for x in ordered if x["phase2_stratum"] == stratum), limit=1)
    add(
        (x for x in ordered if not x["bm25_first_positive_rank"] or x["bm25_first_positive_rank"] > 100),
        limit=20,
    )
    add(
        (x for x in ordered if x["bm25_first_positive_rank"] and x["bm25_first_positive_rank"] <= 10),
        limit=20,
    )
    target_frequency = Counter(p for x in ordered for p in x["positive_ids"])
    add((x for x in ordered if min(target_frequency[p] for p in x["positive_ids"]) == 1), limit=20)
    # Hash ordering prevents the fill from collapsing onto one large model.
    add(sorted(ordered, key=lambda x: stable_hash([DEFAULT_SEED, x["model_id"], x["reaction_id"]])))
    return chosen


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = MODEL_NAME
    model_revision: str = MODEL_REVISION
    query_template: str = QUERY_TEMPLATE
    document_template: str = DOCUMENT_TEMPLATE
    max_length: int = 256
    physical_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    warmup_steps: int = 2
    max_optimizer_steps: int = 20
    epochs: int = 20
    temperature: float = 0.02
    max_grad_norm: float = 1.0
    seed: int = DEFAULT_SEED
    bm25_hard_negatives: int = 2
    random_negatives: int = 1
    mixed_precision: bool = True

    @property
    def hash(self) -> str:
        return stable_hash(asdict(self))


def validate_config(config: TrainingConfig) -> None:
    if config.model_name != MODEL_NAME or config.model_revision != MODEL_REVISION:
        raise ValueError("starting checkpoint identity/revision may not be silently substituted")
    if config.max_length <= 0 or config.physical_batch_size <= 0 or config.gradient_accumulation_steps <= 0:
        raise ValueError("invalid training dimensions")
    if not config.mixed_precision:
        raise ValueError("Phase 3B CUDA smoke training requires mixed precision")


def multi_positive_contrastive_loss(
    query_embeddings: Any, document_embeddings: Any, positive_counts: Sequence[int], group_sizes: Sequence[int],
    *, temperature: float,
) -> Any:
    """Per-query multi-positive InfoNCE over explicit candidates only."""
    import torch

    if len(positive_counts) != len(group_sizes) or len(group_sizes) != query_embeddings.shape[0]:
        raise ValueError("candidate group metadata mismatch")
    losses = []
    offset = 0
    for query, n_positive, size in zip(query_embeddings, positive_counts, group_sizes):
        if n_positive <= 0 or size < n_positive:
            raise ValueError("invalid positive/candidate counts")
        scores = document_embeddings[offset:offset + size] @ query / temperature
        losses.append(torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[:n_positive], dim=0))
        offset += size
    if offset != document_embeddings.shape[0]:
        raise ValueError("unused document embeddings")
    return torch.stack(losses).mean()


def _encode(model: Any, tokenized: Mapping[str, Any]) -> Any:
    import torch.nn.functional as functional

    return functional.normalize(model(**tokenized).last_hidden_state[:, 0], p=2, dim=1)


def _batch_loss(model: Any, tokenizer: Any, examples: Sequence[Mapping[str, Any]], docs: Mapping[str, str], config: TrainingConfig, device: Any) -> Any:
    queries = [str(x["query"]) for x in examples]
    document_texts: list[str] = []
    positive_counts: list[int] = []
    group_sizes: list[int] = []
    for example in examples:
        ids = list(example["positive_ids"]) + [x["kegg_id"] for x in example["negatives"]]
        positive_counts.append(len(example["positive_ids"]))
        group_sizes.append(len(ids))
        document_texts.extend(docs[x] for x in ids)
    query_tokens = tokenizer(queries, padding=True, truncation=True, max_length=config.max_length, return_tensors="pt")
    doc_tokens = tokenizer(document_texts, padding=True, truncation=True, max_length=config.max_length, return_tensors="pt")
    query_tokens = {k: v.to(device) for k, v in query_tokens.items()}
    doc_tokens = {k: v.to(device) for k, v in doc_tokens.items()}
    query_embeddings = _encode(model, query_tokens)
    document_embeddings = _encode(model, doc_tokens)
    return multi_positive_contrastive_loss(
        query_embeddings, document_embeddings, positive_counts, group_sizes, temperature=config.temperature,
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), tmp)
    _replace_with_retry(tmp, path)


def save_checkpoint(path: Path, *, model: Any, optimizer: Any, scheduler: Any, scaler: Any, state: Mapping[str, Any], config: TrainingConfig, dataset_hash: str) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "config_hash": config.hash,
        "dataset_hash": dataset_hash,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "training_state": dict(state),
    }
    _atomic_torch_save(payload, path)


def load_checkpoint(path: Path, *, model: Any, optimizer: Any, scheduler: Any, scaler: Any, config: TrainingConfig, dataset_hash: str) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    if payload.get("config_hash") != config.hash:
        raise ValueError("checkpoint invalidated by training configuration")
    if payload.get("dataset_hash") != dataset_hash:
        raise ValueError("checkpoint invalidated by training dataset")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["scaler_state"])
    return dict(payload["training_state"])


def _linear_schedule(optimizer: Any, *, warmup_steps: int, total_steps: int) -> Any:
    from torch.optim.lr_scheduler import LambdaLR

    def factor(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))
    return LambdaLR(optimizer, factor)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _epoch_order(length: int, seed: int, epoch: int) -> list[int]:
    order = list(range(length))
    random.Random(seed + epoch).shuffle(order)
    return order


def train(
    examples: Sequence[Mapping[str, Any]], config: TrainingConfig, *, out: Path, run_name: str,
    resume: bool = False, model_cache: Path | None = None, stop_after_steps: int | None = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    validate_config(config)
    assert_train_only(examples)
    if not torch.cuda.is_available():
        raise RuntimeError("representative Phase 3B smoke training requires CUDA")
    _seed_everything(config.seed)
    device = torch.device("cuda")
    cache = model_cache or out / "_model_cache"
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision, cache_dir=cache)
    model = AutoModel.from_pretrained(config.model_name, revision=config.model_revision, cache_dir=cache).to(device)
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved != config.model_revision:
        raise ValueError(f"resolved model revision mismatch: {resolved}")
    model.train()
    docs = {d["kegg_id"]: d["text"] for d in retrieval.load_catalog()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = _linear_schedule(optimizer, warmup_steps=config.warmup_steps, total_steps=config.max_optimizer_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=config.mixed_precision, init_scale=128.0)
    dataset_hash = stable_hash(list(examples))
    checkpoint = out / "_checkpoints" / run_name / "latest.pt"
    state: dict[str, Any] = {
        "optimizer_step": 0, "epoch": 0, "cursor": 0, "examples_processed": 0,
        "skipped_optimizer_updates": 0,
    }
    resumed_from = None
    previous_metrics: dict[str, Any] | None = None
    metrics_path = out / ("overfit_metrics.json" if run_name == "overfit" else "smoke_metrics.json")
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint absent: {checkpoint}")
        state = load_checkpoint(
            checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            config=config, dataset_hash=dataset_hash,
        )
        resumed_from = str(checkpoint)
        if metrics_path.exists():
            previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    elif checkpoint.exists():
        raise FileExistsError(f"checkpoint already exists; choose resume or a new run name: {checkpoint}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    starting_examples = int(state["examples_processed"])
    started = time.perf_counter()
    trajectory: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    target_steps = config.max_optimizer_steps if stop_after_steps is None else min(stop_after_steps, config.max_optimizer_steps)
    if target_steps < int(state["optimizer_step"]):
        raise ValueError("stop_after_steps precedes the resumed optimizer step")
    elapsed_offset = float(previous_metrics.get("elapsed_seconds", 0.0)) if previous_metrics else 0.0
    while int(state["optimizer_step"]) < target_steps:
        accumulated_loss = 0.0
        accumulated_examples = 0
        for _ in range(config.gradient_accumulation_steps):
            if int(state["epoch"]) >= config.epochs:
                break
            order = _epoch_order(len(examples), config.seed, int(state["epoch"]))
            cursor = int(state["cursor"])
            indices = order[cursor:cursor + config.physical_batch_size]
            if not indices:
                state["epoch"] = int(state["epoch"]) + 1
                state["cursor"] = 0
                continue
            batch = [examples[i] for i in indices]
            with torch.autocast("cuda", dtype=torch.float16, enabled=config.mixed_precision):
                loss = _batch_loss(model, tokenizer, batch, docs, config, device)
            scaler.scale(loss / config.gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.detach().cpu()) * len(batch)
            accumulated_examples += len(batch)
            state["cursor"] = cursor + len(batch)
            state["examples_processed"] = int(state["examples_processed"]) + len(batch)
            if int(state["cursor"]) >= len(examples):
                state["epoch"] = int(state["epoch"]) + 1
                state["cursor"] = 0
        if accumulated_examples == 0:
            break
        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm).detach().cpu())
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        skipped = scaler.get_scale() < scale_before
        if skipped:
            state["skipped_optimizer_updates"] = int(state.get("skipped_optimizer_updates", 0)) + 1
            continue
        scheduler.step()
        state["optimizer_step"] = int(state["optimizer_step"]) + 1
        elapsed = elapsed_offset + time.perf_counter() - started
        trajectory.append({
            "optimizer_step": int(state["optimizer_step"]),
            "loss": accumulated_loss / accumulated_examples,
            "grad_norm": grad_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "examples_processed_total": int(state["examples_processed"]),
            "elapsed_seconds": elapsed,
        })
    session_elapsed = time.perf_counter() - started
    elapsed = elapsed_offset + session_elapsed
    save_checkpoint(
        checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        state=state, config=config, dataset_hash=dataset_hash,
    )
    peak = int(torch.cuda.max_memory_allocated())
    if previous_metrics:
        peak = max(peak, int(previous_metrics.get("peak_allocated_vram_bytes", 0)))
    combined_trajectory = (list(previous_metrics.get("trajectory", [])) if previous_metrics else []) + trajectory
    result = {
        "run_name": run_name,
        "scope": "bounded smoke diagnostic; not scientific performance",
        "config": asdict(config),
        "config_hash": config.hash,
        "dataset_hash": dataset_hash,
        "n_examples_in_subset": len(examples),
        "trajectory": combined_trajectory,
        "initial_loss": combined_trajectory[0]["loss"] if combined_trajectory else None,
        "final_loss": combined_trajectory[-1]["loss"] if combined_trajectory else None,
        "loss_reduction_fraction": (
            1.0 - combined_trajectory[-1]["loss"] / combined_trajectory[0]["loss"]
            if combined_trajectory and combined_trajectory[0]["loss"] else None
        ),
        "optimizer_steps_completed": int(state["optimizer_step"]),
        "examples_processed": int(state["examples_processed"]),
        "elapsed_seconds": elapsed,
        "examples_per_second": int(state["examples_processed"]) / elapsed if elapsed else None,
        "session_examples_processed": int(state["examples_processed"]) - starting_examples,
        "session_elapsed_seconds": session_elapsed,
        "peak_allocated_vram_bytes": peak,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "mixed_precision": "fp16",
        "resolved_model_revision": resolved,
        "checkpoint_path": repo_relative_posix(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "resumed_from": resumed_from,
        "resume_verified": bool(resume and trajectory),
        "skipped_optimizer_updates": int(state.get("skipped_optimizer_updates", 0)),
        "training_state": state,
        "negative_pool_note": "each loss sees all valid positives plus explicit negatives for each physical-batch query; no cross-query or cross-forward-pass in-batch negatives",
    }
    write_json(result, metrics_path)
    return result


def _load_trained_model(config: TrainingConfig, checkpoint: Path, cache: Path) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, revision=config.model_revision, cache_dir=cache)
    model = AutoModel.from_pretrained(config.model_name, revision=config.model_revision, cache_dir=cache)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("config_hash") != config.hash:
        raise ValueError("ranking checkpoint/config mismatch")
    model.load_state_dict(payload["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, device


def _encode_texts(model: Any, tokenizer: Any, texts: Sequence[str], device: Any, *, max_length: int, batch_size: int) -> np.ndarray:
    import torch

    output = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            output.append(_encode(model, tokens).float().cpu().numpy())
    return np.vstack(output)


def make_ranking_rows(population: pd.DataFrame, ranks: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    rows = []
    for row, ids in zip(population.to_dict("records"), ranks):
        rows.append({
            "schema": RANKING_SCHEMA,
            "model_id": row["model_id"],
            "reaction_id": row["reaction_id"],
            "split": "validation",
            "ranked": [{"rank": rank, "kegg_id": kid} for rank, kid in enumerate(ids, 1)],
        })
    validate_ranking_rows(rows, expected=len(population))
    return rows


def validate_ranking_rows(rows: Sequence[Mapping[str, Any]], *, expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"ranking row count mismatch: {len(rows)} != {expected}")
    keys: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("split") != "validation":
            raise ValueError("ranking evaluation is validation-only")
        key = (str(row["model_id"]), str(row["reaction_id"]))
        if key in keys:
            raise ValueError("duplicate ranking key")
        keys.add(key)
        ranked = list(row["ranked"])
        ids = [x["kegg_id"] for x in ranked]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ranked ID")
        if [x["rank"] for x in ranked] != list(range(1, len(ranked) + 1)):
            raise ValueError("ranking positions must be consecutive and one-indexed")


def freeze_validation_rankings(config: TrainingConfig, *, out: Path, batch_size: int = 64) -> Path:
    import torch

    checkpoint = out / "_checkpoints" / "representative" / "latest.pt"
    model, tokenizer, device = _load_trained_model(config, checkpoint, out / "_model_cache")
    population = retrieval.load_query_population("validation")  # Label-free by construction.
    queries = [retrieval.query_text(row) for row in population.to_dict("records")]
    docs = retrieval.load_catalog()
    started = time.perf_counter()
    document_embeddings = _encode_texts(model, tokenizer, [x["text"] for x in docs], device, max_length=config.max_length, batch_size=batch_size)
    query_embeddings = _encode_texts(model, tokenizer, queries, device, max_length=config.max_length, batch_size=batch_size)
    ranks = retrieval.dense_rank(query_embeddings, document_embeddings, [x["kegg_id"] for x in docs], topn=100)
    rows = make_ranking_rows(population, ranks)
    path = out / "_rankings" / "validation_smoke_rankings.jsonl"
    atomic_write_jsonl(rows, path)
    write_json({
        "ranking_sha256": sha256_file(path),
        "n_queries": len(rows),
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "labels_loaded_during_ranking": False,
    }, out / "_rankings" / "runtime.json")
    return path


def _load_validation_truth_after_freeze(ranking_path: Path) -> pd.DataFrame:
    if not ranking_path.exists() or ranking_path.stat().st_size == 0:
        raise ValueError("validation rankings must be frozen before label join")
    truth = retrieval._truth_and_metadata()
    retrieval.reject_test_rows(truth)
    return truth


def score_frozen_validation_rankings(ranking_path: Path, *, out: Path) -> dict[str, Any]:
    frozen_digest = sha256_file(ranking_path)
    rows = [json.loads(line) for line in ranking_path.read_text(encoding="utf-8").splitlines() if line]
    validate_ranking_rows(rows, expected=969)
    truth = _load_validation_truth_after_freeze(ranking_path)
    ranked = {
        (x["model_id"], x["reaction_id"]): [item["kegg_id"] for item in x["ranked"]]
        for x in rows
    }
    scored = []
    for item in truth.itertuples():
        ids = ranked[(item.model_id, item.reaction_id)]
        targets = set(item.truth)
        scored.append({
            "seen_in_train": bool(item.seen_in_train),
            "true_retrieval_failure": item.stratum in TRUE_RETRIEVAL_FAILURE_STRATA,
            "r1": any(x in targets for x in ids[:1]),
            "r10": any(x in targets for x in ids[:10]),
        })
    frame = pd.DataFrame(scored)
    metrics = {
        "scope": "single validation-only smoke diagnostic; not checkpoint selection or scientific performance",
        "ranking_sha256_before_label_join": frozen_digest,
        "n_validation_reactions": len(frame),
        "recall_at_1": float(frame.r1.mean()),
        "recall_at_10": float(frame.r10.mean()),
        "unseen_target_recall_at_10": float(frame.loc[~frame.seen_in_train, "r10"].mean()),
        "unseen_target_n": int((~frame.seen_in_train).sum()),
        "true_retrieval_failure_recall_at_10": float(frame.loc[frame.true_retrieval_failure, "r10"].mean()),
        "true_retrieval_failure_n": int(frame.true_retrieval_failure.sum()),
        "test_rows_read": 0,
    }
    if sha256_file(ranking_path) != frozen_digest:
        raise RuntimeError("frozen ranking changed during offline scoring")
    write_json(metrics, out / "validation_smoke_metrics.json")
    return metrics


def _sample_summary(examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "n_examples": len(examples),
        "n_clusters": len({x["cluster_id"] for x in examples}),
        "phase2_strata": dict(sorted(Counter(x["phase2_stratum"] for x in examples).items())),
        "n_multi_positive": sum(len(x["positive_ids"]) > 1 for x in examples),
        "n_bm25_easy_at_10": sum(bool(x["bm25_first_positive_rank"] and x["bm25_first_positive_rank"] <= 10) for x in examples),
        "n_bm25_hard_after_100": sum(not x["bm25_first_positive_rank"] or x["bm25_first_positive_rank"] > 100 for x in examples),
        "keys": [{"model_id": x["model_id"], "reaction_id": x["reaction_id"]} for x in examples],
    }


def finalize_artifacts(out: Path) -> None:
    smoke = json.loads((out / "smoke_metrics.json").read_text(encoding="utf-8"))
    if smoke.get("resumed_from"):
        smoke["resumed_from"] = repo_relative_posix(out / "_checkpoints" / "representative" / "latest.pt")
        write_json(smoke, out / "smoke_metrics.json")
    overfit = json.loads((out / "overfit_metrics.json").read_text(encoding="utf-8"))
    environment = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    dataset = json.loads((out / "dataset_summary.json").read_text(encoding="utf-8"))
    negative = json.loads((out / "negative_summary.json").read_text(encoding="utf-8"))
    validation_path = out / "validation_smoke_metrics.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else None
    trajectory = smoke["trajectory"]
    elapsed = float(smoke["elapsed_seconds"])
    processed = int(smoke["examples_processed"])
    projected_epoch = elapsed * 3497 / max(1, processed)
    ranking_runtime_path = out / "_rankings" / "runtime.json"
    ranking_runtime = json.loads(ranking_runtime_path.read_text(encoding="utf-8")) if ranking_runtime_path.exists() else None
    runtime = {
        "representative_training_seconds": elapsed,
        "examples_per_second": smoke["examples_per_second"],
        "projected_full_epoch_seconds": projected_epoch,
        "projected_full_epoch_minutes": projected_epoch / 60,
        "peak_allocated_vram_bytes": smoke["peak_allocated_vram_bytes"],
        "checkpoint_bytes": smoke["checkpoint_bytes"],
        "projection_basis": "linear projection from bounded representative smoke optimization; excludes validation and startup",
        "validation_ranking_runtime": ranking_runtime,
    }
    write_json(runtime, out / "runtime.json")
    full_config = {
        "starting_checkpoint": MODEL_NAME,
        "revision": MODEL_REVISION,
        "max_length": 256,
        "physical_batch_size": smoke["config"]["physical_batch_size"],
        "explicit_negatives_per_query": {
            "bm25": negative["requested_per_query"]["bm25"],
            "random": negative["requested_per_query"]["random"],
        },
        "gradient_accumulation_steps": smoke["config"]["gradient_accumulation_steps"],
        "learning_rate": 2e-5,
        "warmup": "10% of optimizer steps",
        "epochs": 3,
        "evaluation_frequency": "once per epoch after atomically freezing all validation rankings",
        "early_stopping_metric": "validation Recall@1 (primary); preserve Recall@10, unseen-target Recall@10, and true-retrieval-failure Recall@10",
        "patience": 2,
        "estimated_seconds_per_epoch": projected_epoch,
        "estimated_peak_allocated_vram_bytes": smoke["peak_allocated_vram_bytes"],
        "estimated_checkpoint_bytes_each": smoke["checkpoint_bytes"],
        "estimated_checkpoint_storage_bytes_best_plus_latest": 2 * smoke["checkpoint_bytes"],
        "checkpoint_retention": "best plus latest, including optimizer/scaler/scheduler state",
        "local_training_practical": True,
        "cloud_gpu_rental_justified": False,
        "authorization": "proposal only; full training was not launched",
    }
    training_config_artifact = {
        "representative_smoke": smoke["config"],
        "representative_smoke_config_hash": smoke["config_hash"],
        "recommended_full_run": full_config,
        "model_metadata": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "architecture": MODEL_ARCHITECTURE,
            "parameters": MODEL_PARAMETERS,
            "official_source": "https://huggingface.co/BAAI/bge-small-en-v1.5",
            "cache_destination": "benchmark/phase3/phase3b_smoke/_model_cache",
            "observed_cache_bytes": 134410637,
            "role": "general retrieval starting checkpoint; becomes AAAIM biochemical retriever only through fine-tuning",
        },
    }
    write_json(training_config_artifact, out / "training_config.json")
    environment["environment_change"] = {
        "scope": "isolated project-local virtual environment; original Python environment unchanged",
        "path": "benchmark/phase3/_phase3b_env",
        "commands": [
            "python -m venv benchmark/phase3/_phase3b_env",
            "benchmark/phase3/_phase3b_env/Scripts/python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128",
            "benchmark/phase3/_phase3b_env/Scripts/python.exe -m pip install -r benchmark/requirements-phase3b.txt",
        ],
    }
    write_json(environment, out / "environment.json")
    report = [
        "# Phase 3B bi-encoder smoke milestone",
        "",
        "This is a bounded engineering diagnostic, not a scientific performance result. No held-out test labels were read and no full training run was launched.",
        "",
        "## Starting checkpoint",
        "",
        f"`{MODEL_NAME}` at immutable revision `{MODEL_REVISION}` (MIT; {MODEL_ARCHITECTURE}; approximately {MODEL_PARAMETERS:,} parameters). This is a general English retrieval encoder, not a biology-specialized model.",
        "",
        "## Leakage-safe data",
        "",
        f"{dataset['n_examples']:,} usable frozen-training queries, {dataset['n_unique_target_ids']:,} unique targets, {dataset['n_multi_positive_queries']} multi-positive queries, and {negative['n_total_negatives']:,} explicit negatives. {dataset['n_excluded_no_catalog_positive']} of the 3,497 assigned train rows are explicitly excluded because their only valid target has no document in the frozen catalog. All query/document text uses the frozen Phase 3 retrieval templates and passes the digit-bounded leakage scanner. EC/KO siblings are excluded as ambiguous negatives.",
        "",
        "## Measured smoke results",
        "",
        f"Overfit loss: {overfit['initial_loss']:.6f} to {overfit['final_loss']:.6f} ({overfit['loss_reduction_fraction']:.1%} reduction).",
        f"Representative loss: {smoke['initial_loss']:.6f} to {smoke['final_loss']:.6f}; {processed} examples in {elapsed:.2f}s ({smoke['examples_per_second']:.2f}/s); peak allocated VRAM {smoke['peak_allocated_vram_bytes'] / 2**30:.2f} GiB.",
        "Checkpoint and full optimizer/scaler/scheduler state were saved atomically; a separate resumed step is recorded in the smoke metrics.",
        "",
    ]
    if validation:
        report += [
            "## Validation smoke diagnostic",
            "",
            f"One frozen-ranking, validation-only diagnostic: R@1 {validation['recall_at_1']:.4f}, R@10 {validation['recall_at_10']:.4f}, unseen-target R@10 {validation['unseen_target_recall_at_10']:.4f}, and true-retrieval-failure R@10 {validation['true_retrieval_failure_recall_at_10']:.4f}. It was not used to select a checkpoint.",
            "",
        ]
    report += [
        "## Full-run recommendation (not executed)",
        "",
        f"Use batch {full_config['physical_batch_size']}, three explicit negatives, accumulation {full_config['gradient_accumulation_steps']}, maximum length 256, learning rate 2e-5, 10% warmup, and at most three epochs. Estimated training-only time is {projected_epoch / 60:.1f} minutes per epoch. Select on overall validation Recall@1 while retaining the other three prespecified metrics.",
        "",
        "The local RTX 3070 is sufficient for this configuration; cloud rental is not currently justified.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    included = [
        out / name for name in (
            "REPORT.md", "training_config.json", "environment.json", "dataset_manifest.json",
            "dataset_summary.json", "negative_summary.json", "overfit_metrics.json",
            "smoke_metrics.json", "runtime.json", "representative_sample.json",
            "validation_smoke_metrics.json",
        ) if (out / name).exists()
    ]
    write_artifact_manifest(out, included)


def verify_manifest(out: Path = OUT) -> list[str]:
    path = out / "artifact_manifest.json"
    before = path.read_bytes()
    manifest = json.loads(before)
    paths = [x["path"] for x in manifest["files"]]
    problems: list[str] = []
    if len(paths) != len(set(paths)):
        problems.append("duplicate paths")
    for item in manifest["files"]:
        if "\\" in item["path"]:
            problems.append(f"non-POSIX path: {item['path']}")
        artifact = REPO_ROOT / item["path"]
        if not artifact.exists() or sha256_portable(artifact) != item["sha256"]:
            problems.append(f"digest mismatch: {item['path']}")
    if path.read_bytes() != before:
        problems.append("manifest verification mutated the manifest")
    return problems


def full_training_config() -> TrainingConfig:
    """The single prespecified Phase 3B full-run configuration."""
    microbatches_per_epoch = math.ceil(3466 / 4)
    updates_per_epoch = math.ceil(microbatches_per_epoch / 2)
    total_updates = updates_per_epoch * 3
    return TrainingConfig(
        max_length=256,
        physical_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_steps=math.ceil(total_updates * 0.10),
        max_optimizer_steps=total_updates,
        epochs=3,
        temperature=0.02,
        max_grad_norm=1.0,
        seed=DEFAULT_SEED,
        bm25_hard_negatives=2,
        random_negatives=1,
        mixed_precision=True,
    )


def full_epoch_layout(n_examples: int, config: TrainingConfig, epoch: int) -> list[list[list[int]]]:
    """Physical batches grouped into optimizer updates without crossing epochs."""
    if epoch not in (1, 2, 3):
        raise ValueError("full training epoch must be 1, 2, or 3")
    order = _epoch_order(n_examples, config.seed, epoch - 1)
    physical = [order[i:i + config.physical_batch_size] for i in range(0, n_examples, config.physical_batch_size)]
    return [physical[i:i + config.gradient_accumulation_steps] for i in range(0, len(physical), config.gradient_accumulation_steps)]


def assert_full_initializer(initializer: str | Path | None) -> None:
    """Full runs initialize only from the pinned official checkpoint or full-run resume."""
    if initializer is None or str(initializer) == f"{MODEL_NAME}@{MODEL_REVISION}":
        return
    normalized = str(initializer).replace("\\", "/").lower()
    if "phase3b_smoke" in normalized or "overfit" in normalized or "representative" in normalized:
        raise ValueError("smoke checkpoints are forbidden as full-run initializers")
    if "phase3b_full/_checkpoints" not in normalized:
        raise ValueError("full-run resume checkpoint must come from the Phase 3B full output")


def select_best_epoch(epoch_metrics: Mapping[int, Mapping[str, Any]]) -> int:
    """Prespecified validation-only hierarchy with earlier-epoch tie breaking."""
    if set(epoch_metrics) != {0, 1, 2, 3}:
        raise ValueError("checkpoint selection requires epochs 0 through 3")
    return max(
        epoch_metrics,
        key=lambda epoch: (
            float(epoch_metrics[epoch]["exact"]["recall_at_1"]["reaction_micro"]),
            float(epoch_metrics[epoch]["exact"]["recall_at_10"]["reaction_micro"]),
            float(epoch_metrics[epoch]["seen_unseen"]["unseen"]["recall_at_10"]["reaction_micro"]),
            -epoch,
        ),
    )


def validate_full_ranking_rows(rows: Sequence[Mapping[str, Any]], *, epoch: int, expected: int = 969) -> None:
    if len(rows) != expected:
        raise ValueError(f"epoch {epoch} ranking count mismatch: {len(rows)} != {expected}")
    keys: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("schema") != FULL_RANKING_SCHEMA or row.get("split") != "validation":
            raise ValueError("full-run rankings must be validation-only")
        if int(row.get("epoch", -1)) != epoch:
            raise ValueError("ranking epoch mismatch")
        key = (str(row["model_id"]), str(row["reaction_id"]))
        if key in keys:
            raise ValueError("duplicate ranking key")
        keys.add(key)
        ids = list(row["ranked_ids"])
        if len(ids) != 100:
            raise ValueError("full-catalog validation ranking must contain exactly 100 IDs")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ranked ID")


def _read_full_ranking(path: Path, epoch: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    validate_full_ranking_rows(rows, epoch=epoch)
    return rows


def _load_official_start(config: TrainingConfig, cache: Path) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    assert_full_initializer(f"{MODEL_NAME}@{MODEL_REVISION}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, revision=config.model_revision, cache_dir=cache, local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        config.model_name, revision=config.model_revision, cache_dir=cache, local_files_only=True,
    )
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved != MODEL_REVISION:
        raise ValueError(f"official checkpoint revision mismatch: {resolved}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("full Phase 3B training requires CUDA")
    model.to(device)
    return model, tokenizer, device


def freeze_full_epoch_rankings(
    model: Any, tokenizer: Any, device: Any, config: TrainingConfig, *, epoch: int, out: Path,
    batch_size: int = 64,
) -> tuple[Path, dict[str, Any]]:
    """Freeze label-free validation rankings; never loads a label table."""
    import torch

    path = out / f"rankings_epoch_{epoch}.jsonl"
    runtime_path = out / f"ranking_runtime_epoch_{epoch}.json"
    if path.exists() or runtime_path.exists():
        raise FileExistsError(f"refusing to overwrite completed epoch-{epoch} rankings")
    population = retrieval.load_query_population("validation")
    queries = [retrieval.query_text(row) for row in population.to_dict("records")]
    assert_no_kegg_leakage(queries, where=f"full-run validation epoch {epoch}")
    docs = retrieval.load_catalog()
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    document_embeddings = _encode_texts(
        model, tokenizer, [x["text"] for x in docs], device,
        max_length=config.max_length, batch_size=batch_size,
    )
    query_embeddings = _encode_texts(
        model, tokenizer, queries, device, max_length=config.max_length, batch_size=batch_size,
    )
    ranks = retrieval.dense_rank(
        query_embeddings, document_embeddings, [x["kegg_id"] for x in docs], topn=100,
    )
    rows = [{
        "schema": FULL_RANKING_SCHEMA,
        "epoch": epoch,
        "model_id": row["model_id"],
        "reaction_id": row["reaction_id"],
        "split": "validation",
        "ranked_ids": ids,
    } for row, ids in zip(population.to_dict("records"), ranks)]
    validate_full_ranking_rows(rows, epoch=epoch)
    atomic_write_jsonl(rows, path)
    runtime = {
        "epoch": epoch,
        "ranking_sha256_before_label_join": sha256_file(path),
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "model_revision": MODEL_REVISION,
        "config_hash": config.hash,
        "labels_loaded_during_ranking": False,
        "n_queries": len(rows),
        "catalog_size": len(docs),
    }
    write_json(runtime, runtime_path)
    model.train()
    return path, runtime


def _first_rank(ids: Sequence[str], truths: set[str], *, brite: bool = False) -> int | None:
    for rank, candidate in enumerate(ids, 1):
        if candidate in truths or (brite and is_equivalent(candidate, truths, "brite_orthology")):
            return rank
    return None


def _metric_three_way(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    return {
        "reaction_micro": round(float(frame[column].astype(float).mean()), 6),
        "model_macro": round(float(frame.groupby("model_id")[column].mean().mean()), 6),
        "cluster_macro": round(float(frame.groupby("cluster_id")[column].mean().mean()), 6),
        "n_reactions": len(frame),
    }


def score_full_epoch_rankings(path: Path, *, epoch: int, out: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Join validation labels only after the epoch ranking is frozen."""
    frozen_digest = sha256_file(path)
    rows = _read_full_ranking(path, epoch)
    truth = _load_validation_truth_after_freeze(path)
    rank_map = {(x["model_id"], x["reaction_id"]): x["ranked_ids"] for x in rows}
    scored_rows = []
    for item in truth.itertuples():
        ids = rank_map[(item.model_id, item.reaction_id)]
        targets = set(item.truth)
        exact = _first_rank(ids, targets)
        brite = _first_rank(ids, targets, brite=True)
        scored = {
            "model_id": item.model_id,
            "reaction_id": item.reaction_id,
            "cluster_id": item.cluster_id,
            "stratum": item.stratum,
            "seen_in_train": bool(item.seen_in_train),
            "multi_positive": len(targets) > 1,
            "ground_truth_ids": list(item.truth),
            "first_hit_rank_exact": exact,
            "first_hit_rank_brite_orthology": brite,
            "mrr_at_10_exact": 0.0 if exact is None or exact > 10 else 1.0 / exact,
            "mrr_at_10_brite_orthology": 0.0 if brite is None or brite > 10 else 1.0 / brite,
        }
        for k in (1, 3, 5, 10):
            scored[f"recall_at_{k}_exact"] = exact is not None and exact <= k
            scored[f"recall_at_{k}_brite_orthology"] = brite is not None and brite <= k
        scored_rows.append(scored)
    frame = pd.DataFrame(scored_rows)
    exact = {
        f"recall_at_{k}": _metric_three_way(frame, f"recall_at_{k}_exact") for k in (1, 3, 5, 10)
    }
    exact["mrr_at_10"] = _metric_three_way(frame, "mrr_at_10_exact")
    brite_metrics = {
        f"recall_at_{k}": _metric_three_way(frame, f"recall_at_{k}_brite_orthology")
        for k in (1, 3, 5, 10)
    }
    brite_metrics["mrr_at_10"] = _metric_three_way(frame, "mrr_at_10_brite_orthology")

    def subset_metrics(subset: pd.DataFrame) -> dict[str, Any]:
        return {
            f"recall_at_{k}": _metric_three_way(subset, f"recall_at_{k}_exact") for k in (1, 3, 5, 10)
        } | {"mrr_at_10": _metric_three_way(subset, "mrr_at_10_exact")}

    metrics = {
        "schema": FULL_RUN_SCHEMA,
        "epoch": epoch,
        "scope": "validation-only; ranking frozen before label join; no test rows",
        "ranking_sha256_before_label_join": frozen_digest,
        "exact": exact,
        "brite_orthology": brite_metrics,
        "seen_unseen": {
            "seen": subset_metrics(frame[frame.seen_in_train]),
            "unseen": subset_metrics(frame[~frame.seen_in_train]),
        },
        "failure_strata": {
            "true_retrieval_failure": subset_metrics(frame[frame.stratum.isin(TRUE_RETRIEVAL_FAILURE_STRATA)]),
            "rerank_failure": subset_metrics(frame[frame.stratum.eq("retrievable_rerank_failure")]),
        },
        "multi_positive": {
            "multi": subset_metrics(frame[frame.multi_positive]),
            "single": subset_metrics(frame[~frame.multi_positive]),
        },
        "n_validation_reactions": len(frame),
        "test_rows_read": 0,
    }
    if sha256_file(path) != frozen_digest:
        raise RuntimeError("ranking mutated during offline scoring")
    write_json(metrics, out / f"metrics_epoch_{epoch}.json")
    return metrics, frame


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, tmp)
    _replace_with_retry(tmp, destination)


def _checkpoint_record(path: Path, *, epoch: int, role: str, config: TrainingConfig, dataset_hash: str) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "role": role,
        "path": repo_relative_posix(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "config_hash": config.hash,
        "dataset_hash": dataset_hash,
        "contains": ["model", "optimizer", "scheduler", "grad_scaler", "epoch", "cursor", "rng-compatible deterministic ordering"],
    }


def _full_checkpoint_paths(out: Path, epoch: int) -> tuple[Path, Path]:
    root = out / "_checkpoints"
    return root / f"epoch_{epoch}.pt", root / "latest.pt"


def load_full_resume_checkpoint(
    path: Path, *, model: Any, optimizer: Any, scheduler: Any, scaler: Any,
    config: TrainingConfig, dataset_hash: str,
) -> dict[str, Any]:
    assert_full_initializer(path)
    state = load_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        config=config, dataset_hash=dataset_hash,
    )
    if int(state.get("completed_epoch", -1)) not in (0, 1, 2, 3):
        raise ValueError("invalid full-run checkpoint epoch state")
    return state


def train_one_full_epoch(
    examples: Sequence[Mapping[str, Any]], config: TrainingConfig, *, epoch: int,
    model: Any, tokenizer: Any, device: Any, optimizer: Any, scheduler: Any, scaler: Any,
    state: dict[str, Any], out: Path, docs: Mapping[str, str], dataset_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train exactly one deterministic epoch, with resumable optimizer-group cursor."""
    import torch

    groups = full_epoch_layout(len(examples), config, epoch)
    if state.get("active_epoch") not in (None, epoch):
        raise ValueError("resume state points at a different active epoch")
    start_group = int(state.get("update_group_cursor", 0)) if state.get("active_epoch") == epoch else 0
    partial_path = out / "_checkpoints" / f"partial_epoch_{epoch}.json"
    partial = json.loads(partial_path.read_text(encoding="utf-8")) if start_group and partial_path.exists() else {
        "trajectory": [], "loss_weighted_sum": 0.0, "examples": 0, "grad_norms": [], "skipped": 0,
    }
    if start_group and len(partial["trajectory"]) != start_group:
        raise ValueError("partial epoch trajectory/cursor mismatch")
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    try:
        for group_index in range(start_group, len(groups)):
            physical_batches = groups[group_index]
            group_loss_sum = 0.0
            group_examples = 0
            for indices in physical_batches:
                batch = [examples[i] for i in indices]
                with torch.autocast("cuda", dtype=torch.float16, enabled=config.mixed_precision):
                    loss = _batch_loss(model, tokenizer, batch, docs, config, device)
                scaler.scale(loss / len(physical_batches)).backward()
                group_loss_sum += float(loss.detach().cpu()) * len(batch)
                group_examples += len(batch)
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm).detach().cpu())
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            skipped = scaler.get_scale() < scale_before
            if not skipped:
                scheduler.step()
                state["optimizer_step"] = int(state.get("optimizer_step", 0)) + 1
                partial["grad_norms"].append(grad_norm)
            else:
                partial["skipped"] = int(partial["skipped"]) + 1
            state["active_epoch"] = epoch
            state["update_group_cursor"] = group_index + 1
            state["examples_processed"] = int(state.get("examples_processed", 0)) + group_examples
            partial["loss_weighted_sum"] = float(partial["loss_weighted_sum"]) + group_loss_sum
            partial["examples"] = int(partial["examples"]) + group_examples
            partial["trajectory"].append({
                "epoch": epoch,
                "update_group": group_index + 1,
                "optimizer_step": int(state["optimizer_step"]),
                "mean_loss": group_loss_sum / group_examples,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": None if skipped else grad_norm,
                "examples": group_examples,
                "skipped": skipped,
            })
    except KeyboardInterrupt:
        write_json(partial, partial_path)
        _, latest = _full_checkpoint_paths(out, epoch)
        save_checkpoint(
            latest, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            state=state, config=config, dataset_hash=dataset_hash,
        )
        write_json(state, out / "run_state.json")
        raise
    if int(partial["examples"]) != len(examples):
        raise RuntimeError(f"epoch {epoch} processed {partial['examples']} != {len(examples)} examples")
    state.update({
        "completed_epoch": epoch,
        "active_epoch": None,
        "update_group_cursor": 0,
    })
    epoch_checkpoint, latest = _full_checkpoint_paths(out, epoch)
    if epoch_checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite completed checkpoint: {epoch_checkpoint}")
    save_checkpoint(
        epoch_checkpoint, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        state=state, config=config, dataset_hash=dataset_hash,
    )
    _atomic_copy(epoch_checkpoint, latest)
    elapsed = time.perf_counter() - started
    grad_norms = [float(x) for x in partial["grad_norms"]]
    epoch_metrics = {
        "epoch": epoch,
        "mean_training_loss": float(partial["loss_weighted_sum"]) / int(partial["examples"]),
        "loss_trajectory": partial["trajectory"],
        "learning_rate_trajectory": [x["learning_rate"] for x in partial["trajectory"]],
        "gradient_norm_summary": {
            "count": len(grad_norms),
            "min": min(grad_norms) if grad_norms else None,
            "mean": sum(grad_norms) / len(grad_norms) if grad_norms else None,
            "max": max(grad_norms) if grad_norms else None,
        },
        "optimizer_updates_epoch": len(groups) - int(partial["skipped"]),
        "optimizer_updates_total": int(state["optimizer_step"]),
        "examples_processed_epoch": int(partial["examples"]),
        "examples_processed_total": int(state["examples_processed"]),
        "skipped_optimizer_updates": int(partial["skipped"]),
        "training_seconds": elapsed,
        "examples_per_second": int(partial["examples"]) / elapsed,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "checkpoint": _checkpoint_record(
            epoch_checkpoint, epoch=epoch, role="completed_epoch", config=config, dataset_hash=dataset_hash,
        ),
        "resumed_within_epoch": start_group > 0,
    }
    write_json(epoch_metrics, out / f"training_metrics_epoch_{epoch}.json")
    write_json(state, out / "run_state.json")
    if partial_path.exists():
        partial_path.unlink()
    return epoch_metrics, state


def score_reference_rankings(path: Path, truth: pd.DataFrame, *, method: str) -> tuple[pd.DataFrame, dict[tuple[str, str], list[str]]]:
    """Score an already-frozen baseline without invoking its optional runtime stack."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rank_map = {(str(x["model_id"]), str(x["reaction_id"])): list(x["ranked_ids"]) for x in rows}
    expected = set(zip(truth.model_id.astype(str), truth.reaction_id.astype(str)))
    if set(rank_map) != expected or len(rows) != len(rank_map):
        raise ValueError(f"baseline population mismatch for {method}")
    scored = []
    for item in truth.itertuples():
        ids = rank_map[(item.model_id, item.reaction_id)]
        targets = set(item.truth)
        exact = _first_rank(ids, targets)
        brite = _first_rank(ids, targets, brite=True)
        row = {
            "model_id": item.model_id,
            "reaction_id": item.reaction_id,
            "cluster_id": item.cluster_id,
            "stratum": item.stratum,
            "seen_in_train": bool(item.seen_in_train),
            "multi_positive": len(targets) > 1,
            "ground_truth_ids": list(item.truth),
            "first_hit_rank_exact": exact,
            "first_hit_rank_brite_orthology": brite,
            "mrr_at_10_exact": 0.0 if exact is None or exact > 10 else 1.0 / exact,
            "mrr_at_10_brite_orthology": 0.0 if brite is None or brite > 10 else 1.0 / brite,
        }
        for k in (1, 3, 5, 10):
            row[f"recall_at_{k}_exact"] = exact is not None and exact <= k
            row[f"recall_at_{k}_brite_orthology"] = brite is not None and brite <= k
        scored.append(row)
    return pd.DataFrame(scored), rank_map


def summarize_scored_frame(frame: pd.DataFrame) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for kind in ("exact", "brite_orthology"):
        metrics[kind] = {
            f"recall_at_{k}": _metric_three_way(frame, f"recall_at_{k}_{kind}")
            for k in (1, 3, 5, 10)
        }
        metrics[kind]["mrr_at_10"] = _metric_three_way(frame, f"mrr_at_10_{kind}")
    return metrics


def paired_cluster_bootstrap_strict(
    a: pd.DataFrame, b: pd.DataFrame, column: str, *, seed: int = BOOTSTRAP_SEED,
    n_boot: int = 10_000,
) -> dict[str, Any]:
    """Paired reaction-micro difference with strict population/cluster alignment."""
    keys = ["model_id", "reaction_id"]
    a_small = a[keys + ["cluster_id", column]].copy()
    b_small = b[keys + ["cluster_id", column]].copy()
    if a_small.duplicated(keys).any() or b_small.duplicated(keys).any():
        raise ValueError("bootstrap input contains duplicate reactions")
    merged = a_small.merge(b_small, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
    if len(merged) != len(a_small) or len(merged) != len(b_small):
        raise ValueError("bootstrap populations are not aligned")
    if not merged.cluster_id_a.equals(merged.cluster_id_b):
        raise ValueError("bootstrap cluster assignments are not aligned")
    groups = sorted(merged.cluster_id_a.unique())
    if len(groups) != 12:
        raise ValueError(f"expected 12 frozen validation clusters, found {len(groups)}")
    per_cluster = {
        cluster: (
            float((part[f"{column}_a"].astype(float) - part[f"{column}_b"].astype(float)).sum()),
            len(part),
        )
        for cluster, part in merged.groupby("cluster_id_a")
    }
    rng = random.Random(seed)
    values = []
    for _ in range(n_boot):
        drawn = [rng.choice(groups) for _ in groups]
        numerator = sum(per_cluster[x][0] for x in drawn)
        denominator = sum(per_cluster[x][1] for x in drawn)
        values.append(numerator / denominator)
    values.sort()
    point = float(
        (merged[f"{column}_a"].astype(float) - merged[f"{column}_b"].astype(float)).mean()
    )
    low = values[int(0.025 * n_boot)]
    high = values[min(n_boot - 1, int(0.975 * n_boot))]
    return {
        "delta_selected_minus_reference": round(point, 6),
        "ci_95_percentile": [round(low, 6), round(high, 6)],
        "includes_zero": bool(low <= 0.0 <= high),
        "seed": seed,
        "n_boot": n_boot,
        "unit": "frozen validation cluster",
        "n_clusters": len(groups),
        "n_reactions": len(merged),
        "estimand": f"paired reaction-micro {column} difference",
        "limitation": "Only 12 validation clusters; percentile intervals may be unstable.",
    }


def build_transition_analysis(
    epoch0: pd.DataFrame, selected: pd.DataFrame, bm25: pd.DataFrame,
    *, epoch0_ranks: Mapping[tuple[str, str], Sequence[str]],
    selected_ranks: Mapping[tuple[str, str], Sequence[str]],
    bm25_ranks: Mapping[tuple[str, str], Sequence[str]],
    expected_reactions: int = 969,
) -> dict[str, Any]:
    keys = ["model_id", "reaction_id"]
    base = epoch0[keys + ["cluster_id", "stratum", "seen_in_train", "multi_positive", "ground_truth_ids", "first_hit_rank_exact"]]
    joined = base.merge(
        selected[keys + ["first_hit_rank_exact"]], on=keys,
        suffixes=("_epoch0", "_selected"), validate="one_to_one",
    ).merge(
        bm25[keys + ["first_hit_rank_exact"]].rename(columns={"first_hit_rank_exact": "first_hit_rank_bm25"}),
        on=keys, validate="one_to_one",
    )
    if len(joined) != expected_reactions:
        raise ValueError(f"transition population must contain all {expected_reactions} validation reactions")
    r0 = joined.first_hit_rank_exact_epoch0.fillna(101).astype(int)
    rs = joined.first_hit_rank_exact_selected.fillna(101).astype(int)
    rb = joined.first_hit_rank_bm25.fillna(101).astype(int)
    masks = {
        "incorrect_epoch0_to_correct_selected": (r0 != 1) & (rs == 1),
        "correct_epoch0_to_incorrect_selected": (r0 == 1) & (rs != 1),
        "improved_rank_without_top1": (rs < r0) & (rs > 1),
        "worsened_rank": rs > r0,
        "recovered_within_top10": (r0 > 10) & (rs <= 10),
        "lost_from_top10": (r0 <= 10) & (rs > 10),
    }
    top1_matrix = {
        "wrong_to_wrong": int(((r0 != 1) & (rs != 1)).sum()),
        "wrong_to_correct": int(((r0 != 1) & (rs == 1)).sum()),
        "correct_to_wrong": int(((r0 == 1) & (rs != 1)).sum()),
        "correct_to_correct": int(((r0 == 1) & (rs == 1)).sum()),
    }
    if sum(top1_matrix.values()) != len(joined):
        raise RuntimeError("Top-1 transition counts do not partition the population")

    def breakdown(mask: pd.Series) -> dict[str, Any]:
        part = joined[mask]
        return {
            "total": len(part),
            "seen": int(part.seen_in_train.sum()),
            "unseen": int((~part.seen_in_train).sum()),
            "true_retrieval_failure": int(part.stratum.isin(TRUE_RETRIEVAL_FAILURE_STRATA).sum()),
            "rerank_failure": int(part.stratum.eq("retrievable_rerank_failure").sum()),
            "multi_positive": int(part.multi_positive.sum()),
            "single_positive": int((~part.multi_positive).sum()),
            "by_stratum": {str(k): int(v) for k, v in sorted(part.stratum.value_counts().items())},
        }

    example_masks = {
        "successful_biochemical_retrieval_learned": masks["incorrect_epoch0_to_correct_selected"],
        "harmed_by_fine_tuning": masks["correct_epoch0_to_incorrect_selected"],
        "unseen_target_improvement": (~joined.seen_in_train) & (rs < r0),
        "unseen_target_regression": (~joined.seen_in_train) & (rs > r0),
        "bm25_success_biencoder_misses": (rb == 1) & (rs != 1),
        "biencoder_success_bm25_misses": (rs == 1) & (rb != 1),
    }
    population = retrieval.load_query_population("validation").set_index(keys)
    examples = []
    for category, mask in example_masks.items():
        candidates = joined[mask].sort_values(keys)
        if candidates.empty:
            examples.append({"category": category, "eligible_count": 0, "example": None})
            continue
        item = candidates.iloc[0]
        key = (str(item.model_id), str(item.reaction_id))
        query_row = population.loc[key]
        examples.append({
            "category": category,
            "eligible_count": len(candidates),
            "selection": "lexicographically first eligible validation reaction",
            "example": {
                "model_id": key[0],
                "reaction_id": key[1],
                "cluster_id": item.cluster_id,
                "stratum": item.stratum,
                "seen_in_train": bool(item.seen_in_train),
                "multi_positive": bool(item.multi_positive),
                "ground_truth_ids": list(item.ground_truth_ids),
                "query": retrieval.query_text(query_row.to_dict()),
                "first_hit_rank_exact": {
                    "epoch0": None if int(r0.loc[item.name]) == 101 else int(r0.loc[item.name]),
                    "selected": None if int(rs.loc[item.name]) == 101 else int(rs.loc[item.name]),
                    "bm25": None if int(rb.loc[item.name]) == 101 else int(rb.loc[item.name]),
                },
                "top10": {
                    "epoch0": list(epoch0_ranks[key][:10]),
                    "selected": list(selected_ranks[key][:10]),
                    "bm25": list(bm25_ranks[key][:10]),
                },
            },
        })
    return {
        "scope": "paired 969-reaction validation population; exact reaction matching",
        "n_reactions": len(joined),
        "rank_not_in_top100_value_for_comparison": 101,
        "top1_transition_matrix": top1_matrix,
        "transitions": {name: breakdown(mask) for name, mask in masks.items()},
        "examples": examples,
        "example_selection_is_mechanical": True,
    }


def _compact_exact(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: dict(metrics["exact"][name])
        for name in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10", "mrr_at_10")
    }


def _rank_map_from_full(path: Path, epoch: int) -> dict[tuple[str, str], list[str]]:
    return {(x["model_id"], x["reaction_id"]): list(x["ranked_ids"]) for x in _read_full_ranking(path, epoch)}


def _full_report(
    epoch_comparison: Mapping[str, Any], baseline: Mapping[str, Any], bootstrap: Mapping[str, Any],
    transition: Mapping[str, Any], runtime: Mapping[str, Any], selection: Mapping[str, Any],
    recommendation: Mapping[str, Any],
) -> str:
    lines = [
        "# Phase 3B full bi-encoder training", "",
        "One prespecified BGE-small configuration was trained for exactly three epochs. All checkpoint selection and analysis are validation-only: each 969-query, 12,312-document ranking was frozen before validation labels were joined, and no held-out test row or label was read.",
        "", "## Epoch learning curve", "",
        "| Epoch | Train loss | Updates | Train min | Rank min | Exact R@1 | R@3 | R@5 | R@10 | MRR@10 | Unseen R@10 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in epoch_comparison["epochs"]:
        loss = "—" if row["mean_training_loss"] is None else f"{row['mean_training_loss']:.6f}"
        lines.append(
            f"| {row['epoch']} | {loss} | {row['optimizer_updates_epoch']} | "
            f"{row['training_seconds']/60:.2f} | {row['ranking_seconds']/60:.2f} | "
            f"{row['recall_at_1']:.6f} | {row['recall_at_3']:.6f} | {row['recall_at_5']:.6f} | "
            f"{row['recall_at_10']:.6f} | {row['mrr_at_10']:.6f} | {row['unseen_recall_at_10']:.6f} |"
        )
    lines += [
        "", "## Validation-selected checkpoint", "",
        f"Epoch **{selection['selected_epoch']}** was selected by the prespecified hierarchy: exact reaction-micro R@1, then R@10, then unseen R@10, then earlier epoch. Its retained reference is `{selection['selected_reference']}`.",
        "", "## Frozen-baseline comparison", "",
        "| Method | Exact R@1 | Exact R@10 | Delta R@1 vs selected | Delta R@10 vs selected |",
        "|---|---:|---:|---:|---:|",
    ]
    selected_r1 = baseline["selected"]["exact"]["recall_at_1"]["reaction_micro"]
    selected_r10 = baseline["selected"]["exact"]["recall_at_10"]["reaction_micro"]
    for method, metrics in baseline["references"].items():
        r1 = metrics["exact"]["recall_at_1"]["reaction_micro"]
        r10 = metrics["exact"]["recall_at_10"]["reaction_micro"]
        lines.append(f"| {method} | {r1:.6f} | {r10:.6f} | {selected_r1-r1:+.6f} | {selected_r10-r10:+.6f} |")
    lines += ["", "## Paired cluster bootstrap", "", "10,000 percentile replicates use seed 20260902 and the 12 frozen validation clusters. Intervals containing zero are not evidence of superiority.", ""]
    for name, comparisons in bootstrap["comparisons"].items():
        for metric, result in comparisons.items():
            lo, hi = result["ci_95_percentile"]
            lines.append(f"- Selected vs {name}, {metric}: {result['delta_selected_minus_reference']:+.6f}, 95% CI [{lo:+.6f}, {hi:+.6f}]; includes zero: {str(result['includes_zero']).lower()}.")
    selected_seen = baseline["selected_strata"]
    lines += [
        "", "## Selected-checkpoint strata", "",
        f"Seen R@10: {selected_seen['seen_recall_at_10']:.6f}; unseen R@10: {selected_seen['unseen_recall_at_10']:.6f}; true-retrieval-failure R@10: {selected_seen['true_retrieval_failure_recall_at_10']:.6f}; rerank-failure R@10: {selected_seen['rerank_failure_recall_at_10']:.6f}.",
        "", "## What fine-tuning changed", "",
    ]
    for name, values in transition["transitions"].items():
        lines.append(f"- {name}: {values['total']} (seen {values['seen']}, unseen {values['unseen']}, multi-positive {values['multi_positive']}).")
    lines += ["", "Mechanically selected examples (lexicographically first eligible):", ""]
    for item in transition["examples"]:
        example = item["example"]
        if example is None:
            lines.append(f"- {item['category']}: no eligible reaction.")
        else:
            ranks = example["first_hit_rank_exact"]
            lines.append(f"- {item['category']}: `{example['model_id']}/{example['reaction_id']}`; epoch 0 rank {ranks['epoch0']}, selected rank {ranks['selected']}, BM25 rank {ranks['bm25']}.")
    lines += [
        "", "## Runtime, storage, and recommendation", "",
        f"Training took {runtime['total_training_seconds']/60:.2f} minutes and validation ranking took {runtime['total_ranking_seconds']/60:.2f} minutes. Peak allocated/reserved VRAM was {runtime['peak_allocated_vram_bytes']/2**30:.2f}/{runtime['peak_reserved_vram_bytes']/2**30:.2f} GiB. Three completed epoch checkpoints occupy {runtime['completed_checkpoint_bytes']/2**30:.2f} GiB; including the retained best and latest copies, checkpoint storage is {runtime['retained_checkpoint_bytes_on_disk']/2**30:.2f} GiB. All are ignored by Git.",
        "", recommendation["summary"], "",
        "The selected inference weights should be archived outside Git according to `archive_plan.json`; no upload was performed.",
    ]
    return "\n".join(lines) + "\n"


def build_full_derived(out: Path = FULL_OUT) -> list[Path]:
    """Rebuild all label-derived full-run summaries deterministically."""
    epoch_metrics: dict[int, dict[str, Any]] = {}
    epoch_frames: dict[int, pd.DataFrame] = {}
    for epoch in range(4):
        epoch_metrics[epoch], epoch_frames[epoch] = score_full_epoch_rankings(
            out / f"rankings_epoch_{epoch}.jsonl", epoch=epoch, out=out,
        )
    selected_epoch = select_best_epoch(epoch_metrics)
    config = full_training_config()
    dataset_hash = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))["dataset_sha256"]
    checkpoint_root = out / "_checkpoints"
    if selected_epoch:
        source = checkpoint_root / f"epoch_{selected_epoch}.pt"
        best = checkpoint_root / "best.pt"
        if best.exists() and sha256_file(best) != sha256_file(source):
            raise ValueError("existing retained best checkpoint does not match validation selection")
        if not best.exists():
            _atomic_copy(source, best)
        selected_reference = repo_relative_posix(best)
        selected_checkpoint = _checkpoint_record(
            best, epoch=selected_epoch, role="validation_selected_best", config=config,
            dataset_hash=dataset_hash,
        )
    else:
        selected_reference = f"{MODEL_NAME}@{MODEL_REVISION}"
        selected_checkpoint = {
            "epoch": 0, "role": "validation_selected_official_checkpoint",
            "path": selected_reference, "revision": MODEL_REVISION,
            "config_hash": config.hash, "dataset_hash": dataset_hash,
        }
    latest = checkpoint_root / "latest.pt"
    latest_checkpoint = _checkpoint_record(
        latest, epoch=3, role="latest_completed_epoch", config=config, dataset_hash=dataset_hash,
    )
    selection = {
        "selection_population": "validation only; epochs 0 through 3",
        "hierarchy": ["exact reaction-micro Recall@1", "exact reaction-micro Recall@10", "unseen-target Recall@10", "earlier epoch"],
        "selected_epoch": selected_epoch,
        "selected_reference": selected_reference,
        "selected_checkpoint": selected_checkpoint,
        "latest_checkpoint": latest_checkpoint,
        "test_rows_read": 0,
    }
    write_json(selection, out / "selected_checkpoint.json")

    epoch_rows = []
    trajectory = []
    for epoch in range(4):
        ranking_runtime = json.loads((out / f"ranking_runtime_epoch_{epoch}.json").read_text(encoding="utf-8"))
        training = None if epoch == 0 else json.loads((out / f"training_metrics_epoch_{epoch}.json").read_text(encoding="utf-8"))
        if training:
            trajectory.extend(training["loss_trajectory"])
        metrics = epoch_metrics[epoch]
        epoch_rows.append({
            "epoch": epoch,
            "mean_training_loss": None if training is None else training["mean_training_loss"],
            "optimizer_updates_epoch": 0 if training is None else training["optimizer_updates_epoch"],
            "examples_processed_epoch": 0 if training is None else training["examples_processed_epoch"],
            "skipped_optimizer_updates": 0 if training is None else training["skipped_optimizer_updates"],
            "training_seconds": 0.0 if training is None else training["training_seconds"],
            "ranking_seconds": ranking_runtime["seconds"],
            "recall_at_1": metrics["exact"]["recall_at_1"]["reaction_micro"],
            "recall_at_3": metrics["exact"]["recall_at_3"]["reaction_micro"],
            "recall_at_5": metrics["exact"]["recall_at_5"]["reaction_micro"],
            "recall_at_10": metrics["exact"]["recall_at_10"]["reaction_micro"],
            "mrr_at_10": metrics["exact"]["mrr_at_10"]["reaction_micro"],
            "model_macro_recall_at_10": metrics["exact"]["recall_at_10"]["model_macro"],
            "cluster_macro_recall_at_10": metrics["exact"]["recall_at_10"]["cluster_macro"],
            "unseen_recall_at_10": metrics["seen_unseen"]["unseen"]["recall_at_10"]["reaction_micro"],
            "brite_orthology_recall_at_10": metrics["brite_orthology"]["recall_at_10"]["reaction_micro"],
        })
    epoch_comparison = {
        "selected_epoch": selected_epoch,
        "epochs": epoch_rows,
        "selection_hierarchy_applied_without_early_stopping": True,
    }
    write_json(epoch_comparison, out / "epoch_comparison.json")
    write_json({"optimizer_groups": trajectory, "n_optimizer_groups": len(trajectory)}, out / "training_trajectory.json")

    truth = _load_validation_truth_after_freeze(out / "rankings_epoch_0.jsonl")
    baseline_files = {
        "phase2_rule_based": retrieval.OUT / "rankings_phase2_rule_based.jsonl",
        "bm25": retrieval.OUT / "rankings_bm25.jsonl",
        "bge_m3_dense": retrieval.OUT / "rankings_bge_m3_dense.jsonl",
        "bm25_bge_m3_rrf": retrieval.OUT / "rankings_bm25_bge_m3_rrf.jsonl",
    }
    baseline_frames: dict[str, pd.DataFrame] = {}
    baseline_ranks: dict[str, dict[tuple[str, str], list[str]]] = {}
    baseline_summaries = {}
    baseline_provenance = {}
    for method, path in baseline_files.items():
        baseline_frames[method], baseline_ranks[method] = score_reference_rankings(path, truth, method=method)
        baseline_summaries[method] = summarize_scored_frame(baseline_frames[method])
        baseline_provenance[method] = {"path": repo_relative_posix(path), "sha256": sha256_file(path)}
    selected_frame = epoch_frames[selected_epoch]
    selected_metrics = epoch_metrics[selected_epoch]
    baseline_comparison = {
        "selected_epoch": selected_epoch,
        "selected": summarize_scored_frame(selected_frame),
        "references": {"epoch0_bge_small": summarize_scored_frame(epoch_frames[0]), **baseline_summaries},
        "frozen_baseline_provenance": baseline_provenance,
        "selected_strata": {
            "seen_recall_at_10": selected_metrics["seen_unseen"]["seen"]["recall_at_10"]["reaction_micro"],
            "unseen_recall_at_10": selected_metrics["seen_unseen"]["unseen"]["recall_at_10"]["reaction_micro"],
            "true_retrieval_failure_recall_at_10": selected_metrics["failure_strata"]["true_retrieval_failure"]["recall_at_10"]["reaction_micro"],
            "rerank_failure_recall_at_10": selected_metrics["failure_strata"]["rerank_failure"]["recall_at_10"]["reaction_micro"],
        },
    }
    write_json(baseline_comparison, out / "baseline_comparison.json")
    write_json(
        {str(epoch): metrics["seen_unseen"] for epoch, metrics in epoch_metrics.items()},
        out / "seen_unseen_analysis.json",
    )
    write_json(
        {str(epoch): metrics["failure_strata"] for epoch, metrics in epoch_metrics.items()},
        out / "failure_stratum_analysis.json",
    )

    compare_frames = {
        "epoch0_bge_small": epoch_frames[0],
        "bge_m3_dense": baseline_frames["bge_m3_dense"],
        "bm25": baseline_frames["bm25"],
        "bm25_bge_m3_rrf": baseline_frames["bm25_bge_m3_rrf"],
    }
    bootstrap = {
        "selected_epoch": selected_epoch,
        "comparisons": {
            name: {
                "recall_at_1": paired_cluster_bootstrap_strict(selected_frame, frame, "recall_at_1_exact"),
                "recall_at_10": paired_cluster_bootstrap_strict(selected_frame, frame, "recall_at_10_exact"),
            }
            for name, frame in compare_frames.items()
        },
    }
    write_json(bootstrap, out / "bootstrap_comparisons.json")
    epoch0_ranks = _rank_map_from_full(out / "rankings_epoch_0.jsonl", 0)
    selected_ranks = _rank_map_from_full(out / f"rankings_epoch_{selected_epoch}.jsonl", selected_epoch)
    transition = build_transition_analysis(
        epoch_frames[0], selected_frame, baseline_frames["bm25"],
        epoch0_ranks=epoch0_ranks, selected_ranks=selected_ranks,
        bm25_ranks=baseline_ranks["bm25"],
    )
    write_json(transition, out / "transition_analysis.json")

    training_metrics = [json.loads((out / f"training_metrics_epoch_{epoch}.json").read_text(encoding="utf-8")) for epoch in (1, 2, 3)]
    ranking_metrics = [json.loads((out / f"ranking_runtime_epoch_{epoch}.json").read_text(encoding="utf-8")) for epoch in range(4)]
    completed = [checkpoint_root / f"epoch_{epoch}.pt" for epoch in (1, 2, 3)]
    runtime = {
        "total_training_seconds": sum(x["training_seconds"] for x in training_metrics),
        "total_ranking_seconds": sum(x["seconds"] for x in ranking_metrics),
        "peak_allocated_vram_bytes": max([x["peak_allocated_vram_bytes"] for x in training_metrics + ranking_metrics]),
        "peak_reserved_vram_bytes": max([x["peak_reserved_vram_bytes"] for x in training_metrics + ranking_metrics]),
        "completed_checkpoint_bytes": sum(x.stat().st_size for x in completed),
        "retained_checkpoint_bytes_on_disk": sum(
            x.stat().st_size for x in [*completed, checkpoint_root / "best.pt", latest]
            if x.exists()
        ),
        "checkpoint_records": [
            _checkpoint_record(path, epoch=epoch, role="completed_epoch", config=config, dataset_hash=dataset_hash)
            for epoch, path in zip((1, 2, 3), completed)
        ],
        "best_checkpoint": selected_checkpoint,
        "latest_checkpoint": latest_checkpoint,
    }
    write_json(runtime, out / "runtime_vram.json")

    before_last = epoch_rows[2]
    last = epoch_rows[3]
    r1_delta = last["recall_at_1"] - before_last["recall_at_1"]
    r10_delta = last["recall_at_10"] - before_last["recall_at_10"]
    unseen_delta = last["unseen_recall_at_10"] - before_last["unseen_recall_at_10"]
    plateaued = abs(r1_delta) < 1 / 969 and abs(r10_delta) < 1 / 969 and abs(unseen_delta) < 1 / 122
    still_improving = r1_delta >= 5 / 969 and r10_delta >= -1 / 969 and unseen_delta >= -1 / 122
    peaked_before = selected_epoch < 3
    overfitting_visible = last["recall_at_1"] < max(x["recall_at_1"] for x in epoch_rows[:3])
    if still_improving and not plateaued:
        summary = "Epoch 3 remained meaningfully better by the recorded rule; a separately authorized extension experiment is scientifically reasonable, but was not run."
    elif plateaued:
        summary = "Epoch 3 was plateaued relative to epoch 2; the validation curve does not justify automatically extending training."
    elif peaked_before or overfitting_visible:
        summary = "Validation performance peaked before epoch 3 or showed degradation; extending this run is not currently justified."
    else:
        summary = "Epoch 3 changed the validation trade-off without a clear continuing improvement signal; no automatic extension is justified."
    recommendation = {
        "selected_epoch": selected_epoch,
        "epoch3_minus_epoch2": {"recall_at_1": r1_delta, "recall_at_10": r10_delta, "unseen_recall_at_10": unseen_delta},
        "meaningful_improvement_rule": "R@1 gain of at least 5/969 with R@10 decline no worse than 1/969 and unseen R@10 decline no worse than 1/122",
        "performance_peaked_before_epoch3": peaked_before,
        "plateaued": plateaued,
        "epoch3_still_meaningfully_improving": still_improving and not plateaued,
        "overfitting_visible": overfitting_visible,
        "extension_run_executed": False,
        "summary": summary,
    }
    write_json(recommendation, out / "recommendation.json")
    archive_plan = {
        "status": "prepared only; no archive created or uploaded",
        "selected_epoch": selected_epoch,
        "selected_source": selected_reference,
        "inference_archive": {
            "required": ["model state_dict", "pinned tokenizer/config", "pooling and L2-normalization recipe", "query/document templates", "model revision", "training config and checksums"],
            "exclude": ["optimizer state", "scheduler state", "gradient scaler state"],
            "suggested_name": f"aaaim-phase3b-bge-small-epoch-{selected_epoch}-inference.tar.zst",
            "verification": "record SHA-256 and byte size after exporting; restore into the pinned BGE-small architecture and run a frozen validation-ranking digest check",
        },
        "resume_archive": {
            "required": ["selected/latest full .pt checkpoint", "optimizer", "scheduler", "gradient scaler", "training state/cursor", "full training config", "dataset hash", "pinned tokenizer/config"],
            "latest_checkpoint": latest_checkpoint,
            "note": "Resume archive is larger and is only needed for a separately authorized continuation experiment.",
        },
        "git_policy": "checkpoint, cache, environment, and archive bytes remain ignored and unstaged",
        "upload_performed": False,
    }
    write_json(archive_plan, out / "archive_plan.json")
    safety_audit = {
        "training": {
            "split": "train only",
            "model_visible_queries_digit_bounded_kegg_reaction_id_scan_passed": True,
            "n_queries": int(json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))["example_count"]),
            "test_keys_or_labels_used": 0,
        },
        "ranking": {
            "split": "validation only",
            "n_queries_per_epoch": [x["n_queries"] for x in ranking_metrics],
            "catalog_size_per_epoch": [x["catalog_size"] for x in ranking_metrics],
            "model_visible_queries_digit_bounded_kegg_reaction_id_scan_passed": True,
            "labels_loaded_during_ranking": [x["labels_loaded_during_ranking"] for x in ranking_metrics],
        },
        "evaluation": {
            "validation_labels_joined_only_after_ranking_digest_was_recorded": True,
            "test_rows_read_ranked_or_scored": 0,
        },
        "initialization": f"{MODEL_NAME}@{MODEL_REVISION}",
        "smoke_checkpoint_used": False,
        "paid_api_used": False,
        "cloud_or_rented_gpu_used": False,
        "external_model_used_beyond_authorized_checkpoint": False,
    }
    write_json(safety_audit, out / "safety_audit.json")
    (out / "REPORT.md").write_text(
        _full_report(epoch_comparison, baseline_comparison, bootstrap, transition, runtime, selection, recommendation),
        encoding="utf-8", newline="\n",
    )
    return [
        out / name for name in (
            "selected_checkpoint.json", "epoch_comparison.json", "training_trajectory.json",
            "baseline_comparison.json", "seen_unseen_analysis.json", "failure_stratum_analysis.json",
            "bootstrap_comparisons.json", "transition_analysis.json", "runtime_vram.json",
            "recommendation.json", "archive_plan.json", "REPORT.md",
            "safety_audit.json",
            *(f"metrics_epoch_{epoch}.json" for epoch in range(4)),
        )
    ]


def rebuild_full_reports_twice(out: Path = FULL_OUT) -> dict[str, Any]:
    first_paths = build_full_derived(out)
    first = {repo_relative_posix(path): path.read_bytes() for path in first_paths}
    second_paths = build_full_derived(out)
    second = {repo_relative_posix(path): path.read_bytes() for path in second_paths}
    if first != second:
        changed = sorted(set(first) | set(second) - {k for k in first if first.get(k) == second.get(k)})
        raise RuntimeError(f"derived full-run rebuild was not byte-identical: {changed}")
    result = {
        "passes": 2,
        "byte_identical": True,
        "files": [{"path": key, "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)} for key, value in sorted(first.items())],
    }
    write_json(result, out / "rebuild_verification.json")
    artifacts = [
        path for path in out.iterdir()
        if path.is_file() and not path.name.startswith("_") and path.name != "artifact_manifest.json"
    ]
    write_artifact_manifest(out, artifacts)
    return result


def run_full_training(out: Path = FULL_OUT, *, resume: bool = False) -> None:
    import torch

    config = full_training_config()
    validate_config(config)
    assert_full_initializer(f"{MODEL_NAME}@{MODEL_REVISION}")
    prior_run_markers = [out / "run_state.json", out / "rankings_epoch_0.jsonl", out / "_checkpoints" / "latest.pt"]
    if not resume and any(path.exists() for path in prior_run_markers):
        raise FileExistsError("full-run state already exists; use --resume rather than overwriting it")
    examples, dataset_summary, negative_summary = build_training_examples()
    assert_train_only(examples)
    if len(examples) != 3466 or dataset_summary["n_excluded_no_catalog_positive"] != 31:
        raise ValueError("prespecified full training population invariant failed")
    if negative_summary["source_counts"] != {"bm25": 6932, "random": 3466}:
        raise ValueError("prespecified explicit-negative recipe invariant failed")
    if config.max_optimizer_steps != 1302 or config.warmup_steps != 131:
        raise ValueError("prespecified full optimizer schedule invariant failed")
    write_dataset_artifacts(out, examples, dataset_summary, negative_summary)
    training_config = {
        "schema": FULL_RUN_SCHEMA,
        "configuration": asdict(config),
        "config_hash": config.hash,
        "usable_training_queries": len(examples),
        "physical_batches_per_epoch": math.ceil(len(examples) / config.physical_batch_size),
        "optimizer_updates_per_epoch": len(full_epoch_layout(len(examples), config, 1)),
        "total_optimizer_update_groups": config.max_optimizer_steps,
        "warmup_fraction": 0.10,
        "model_initialization": f"{MODEL_NAME}@{MODEL_REVISION}",
        "smoke_checkpoint_used": False,
        "negative_pool_note": "gradient accumulation does not enlarge the candidate pool; each forward pass sees only that query's positives plus two BM25 and one random explicit negative",
        "test_rows_read": 0,
    }
    write_json(training_config, out / "full_training_config.json")
    initialization = {
        "epoch": 0, "source": "original official checkpoint", "model": MODEL_NAME,
        "revision": MODEL_REVISION, "smoke_checkpoint_used": False,
        "cache": repo_relative_posix(OUT / "_model_cache"),
    }
    initialization_path = out / "initialization.json"
    if initialization_path.exists() and json.loads(initialization_path.read_text(encoding="utf-8")) != initialization:
        raise ValueError("existing initialization metadata is incompatible")
    write_json(initialization, initialization_path)
    _seed_everything(config.seed)
    model, tokenizer, device = _load_official_start(config, OUT / "_model_cache")
    docs = {d["kegg_id"]: d["text"] for d in retrieval.load_catalog()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = _linear_schedule(optimizer, warmup_steps=config.warmup_steps, total_steps=config.max_optimizer_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=128.0)
    dataset_hash = dataset_summary["dataset_sha256"]
    state: dict[str, Any] = {
        "completed_epoch": 0, "active_epoch": None, "update_group_cursor": 0,
        "optimizer_step": 0, "examples_processed": 0,
    }
    latest = out / "_checkpoints" / "latest.pt"
    if resume and latest.exists():
        state = load_full_resume_checkpoint(
            latest, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
            config=config, dataset_hash=dataset_hash,
        )
    elif resume and not (out / "rankings_epoch_0.jsonl").exists():
        raise FileNotFoundError("no resumable full-run checkpoint or epoch-zero ranking exists")

    def evaluate_current(epoch: int, *, existing_ok: bool) -> None:
        ranking = out / f"rankings_epoch_{epoch}.jsonl"
        if ranking.exists():
            if not existing_ok:
                raise FileExistsError(f"refusing to overwrite epoch-{epoch} ranking")
            _read_full_ranking(ranking, epoch)
        else:
            freeze_full_epoch_rankings(model, tokenizer, device, config, epoch=epoch, out=out)
        score_full_epoch_rankings(ranking, epoch=epoch, out=out)

    completed = int(state["completed_epoch"])
    active = state.get("active_epoch")
    if completed == 0 and active is None:
        evaluate_current(0, existing_ok=resume)
    else:
        for epoch in range(completed):
            if not (out / f"rankings_epoch_{epoch}.jsonl").exists():
                raise FileNotFoundError(f"completed epoch {epoch} is missing its frozen ranking")
        if not (out / f"rankings_epoch_{completed}.jsonl").exists():
            if active is not None:
                raise FileNotFoundError("cannot reconstruct a completed-epoch ranking from a mid-epoch model")
            evaluate_current(completed, existing_ok=False)
        else:
            score_full_epoch_rankings(out / f"rankings_epoch_{completed}.jsonl", epoch=completed, out=out)

    start_epoch = int(active) if active is not None else completed + 1
    for epoch in range(start_epoch, 4):
        _, state = train_one_full_epoch(
            examples, config, epoch=epoch, model=model, tokenizer=tokenizer, device=device,
            optimizer=optimizer, scheduler=scheduler, scaler=scaler, state=state, out=out,
            docs=docs, dataset_hash=dataset_hash,
        )
        evaluate_current(epoch, existing_ok=False)
    if int(state["completed_epoch"]) != 3 or int(state["examples_processed"]) != 3 * len(examples):
        raise RuntimeError("full training did not complete exactly three passes")
    rebuild_full_reports_twice(out)


def _write_runtime_config(out: Path, config: TrainingConfig) -> None:
    payload = {
        **asdict(config),
        "config_hash": config.hash,
        "model_license": MODEL_LICENSE,
        "model_architecture": MODEL_ARCHITECTURE,
        "model_parameter_scale": MODEL_PARAMETERS,
        "model_role": "general retrieval starting checkpoint to be fine-tuned for AAAIM biochemical retrieval",
        "expected_download": "134,410,637 bytes observed for pinned weights plus tokenizer/config cache",
        "cache_destination": repo_relative_posix(out / "_model_cache"),
        "loss": "multi-positive InfoNCE with per-query explicit candidates",
        "pooling": "CLS",
        "normalization": "L2",
        "similarity": "normalized inner product (cosine)",
        "shared_encoder": True,
    }
    write_json(payload, out / "_runtime_training_config.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
        "preflight", "build", "overfit", "smoke", "validate", "finalize", "verify",
        "full", "full-rebuild", "full-verify",
    ])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--stage", default="initial_active_environment")
    parser.add_argument("--verify-tensor", action="store_true")
    parser.add_argument("--sample-size", type=int, default=160)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out = args.out or (FULL_OUT if args.command.startswith("full") else OUT)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "preflight":
        print(json.dumps(record_environment(args.out, args.stage, verify_tensor=args.verify_tensor), indent=2, sort_keys=True))
        return 0
    if args.command in {"verify", "full-verify"}:
        problems = verify_manifest(args.out)
        print(json.dumps({"problems": problems, "n_problems": len(problems)}))
        return int(bool(problems))
    if args.command == "full-rebuild":
        print(json.dumps(rebuild_full_reports_twice(args.out), indent=2, sort_keys=True))
        return 0
    if args.command == "full":
        run_full_training(args.out, resume=args.resume)
        return 0
    if args.command == "finalize":
        finalize_artifacts(args.out)
        return 0
    examples, dataset_summary, negative_summary = build_training_examples()
    write_dataset_artifacts(args.out, examples, dataset_summary, negative_summary)
    if args.command == "build":
        print(json.dumps({"dataset": dataset_summary, "negatives": negative_summary}, indent=2, sort_keys=True))
        return 0
    representative = select_representative_examples(examples, args.sample_size)
    write_json(_sample_summary(representative), args.out / "representative_sample.json")
    if args.command == "overfit":
        subset = representative[:12]
        config = TrainingConfig(
            physical_batch_size=2, gradient_accumulation_steps=1,
            learning_rate=5e-5, warmup_steps=2, max_optimizer_steps=40,
            epochs=20,
        )
        _write_runtime_config(args.out, config)
        train(subset, config, out=args.out, run_name="overfit", resume=args.resume, stop_after_steps=args.steps)
        return 0
    config = TrainingConfig(max_optimizer_steps=21)
    _write_runtime_config(args.out, config)
    if args.command == "smoke":
        train(
            representative, config, out=args.out, run_name="representative", resume=args.resume,
            stop_after_steps=args.steps,
        )
        return 0
    if args.command == "validate":
        ranking_path = freeze_validation_rankings(config, out=args.out)
        print(json.dumps(score_frozen_validation_rankings(ranking_path, out=args.out), indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
