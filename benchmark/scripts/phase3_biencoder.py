"""Leakage-safe Phase 3B bi-encoder preparation and bounded smoke training.

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
DEFAULT_SEED = 20260909


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
    parser.add_argument("command", choices=["preflight", "build", "overfit", "smoke", "validate", "finalize", "verify"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--stage", default="initial_active_environment")
    parser.add_argument("--verify-tensor", action="store_true")
    parser.add_argument("--sample-size", type=int, default=160)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.command == "preflight":
        print(json.dumps(record_environment(args.out, args.stage, verify_tensor=args.verify_tensor), indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        problems = verify_manifest(args.out)
        print(json.dumps({"problems": problems, "n_problems": len(problems)}))
        return int(bool(problems))
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
