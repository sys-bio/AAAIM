"""Freeze the deterministic validation-only 60-reaction formal audit sample.

This module performs mechanical sampling from the frozen Prompt 1 inventory.
It does not load held-out labels, perform inference, or make biological
judgments.  The sampling rules and stable tie-breakers are constants so that
selection is prespecified before any selected case is inspected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from benchmark.scripts.phase3_common import (
    PHASE3_DIR,
    REPO_ROOT,
    atomic_write_json,
    sha256_portable,
    write_artifact_manifest,
)
from benchmark.scripts.phase3_error_inventory import parse_multilabel_kegg_ids

OUT = PHASE3_DIR / "error_audit"
INVENTORY = OUT / "all_validation_outcomes.csv"
SPLITS = PHASE3_DIR / "splits.csv"
SPLITS_DIGEST = "7df9566693ea1f2c170fe9d442fde47d8927ee6668a9d5c1c2e762d52cf55a4b"
SOURCE_PROMPT1_COMMIT = "46844578de89df7db4d0deca258be4f33ccd2ead"
SAMPLING_SEED = 20260910
BLINDED_SEED = SAMPLING_SEED + 1
ALGORITHM_VERSION = "phase3-formal-audit-sampling-v1"

RANDOM = "random_control"
FAILURE = "phase3c_failure_harm"
DISAGREEMENT = "cross_method_disagreement"
SUSPICIOUS = "mechanically_suspicious"
MANUAL_EDGE = "manual_nomination_or_documented_edge_case"

CATEGORY_QUOTAS = {
    RANDOM: 20,
    FAILURE: 15,
    DISAGREEMENT: 10,
    SUSPICIOUS: 10,
    MANUAL_EDGE: 5,
}
SELECTION_ORDER = [RANDOM, MANUAL_EDGE, FAILURE, DISAGREEMENT, SUSPICIOUS]
PRESENTATION_ORDER = [RANDOM, FAILURE, DISAGREEMENT, SUSPICIOUS, MANUAL_EDGE]

FAILURE_TYPES = [
    "fusion_top1_correct_grounded_incorrect",
    "fusion_top1_correct_grounded_abstained",
    "truth_fusion_ranks2_10_grounded_incorrect",
    "truth_fusion_ranks2_10_grounded_abstained",
    "truth_absent_fusion_top10_grounded_incorrect_in_evidence",
    "other_grounded_nonexact",
    "evidence_compliance_case",
]

DISAGREEMENT_TYPES = [
    "bm25_correct_trained_incorrect",
    "trained_correct_bm25_incorrect",
    "fusion_correct_components_differ",
    "grounded_correct_phase3a_incorrect",
    "phase3a_correct_grounded_incorrect",
    "fusion_correct_grounded_harms",
    "fusion_incorrect_grounded_recovers",
    "exact_mismatch_brite_orthology_match",
    "large_ground_truth_rank_change",
    "one_method_abstains_another_correct",
]

SUSPICION_RULES = [
    "multiple_ground_truth_kegg_ids",
    "ground_truth_id_absent_from_frozen_catalog",
    "exact_mismatch_resolved_by_frozen_brite_orthology",
]

DOCUMENTED_EDGE_SOURCES = {
    "benchmark/phase3/phase3c_validation/qualitative_examples.json":
        "33e91e50e9be45d06d8669ede2f12823ffcf032f5a67e175a71a68758cbab990",
    "benchmark/phase3/phase3b_fusion/qualitative_examples.json":
        "187232dad72996b3c49f357066beda79ff015a3bdc5dec388c0147ab003f91c8",
    "benchmark/phase3/retrieval_baselines/qualitative_examples.json":
        "2502b9373ec9edecc10711d2648119edeadb61bd3678b9550135a55f4fc46aae",
}

PROMPT1_ARTIFACT_DIGESTS = {
    "README.md": "108cb3648d9b285c716abc5ba31f4bb6d2d755248e00ef5388c22b7bf6d7a08f",
    "all_method_disagreements.csv": "7eea9fcb73a3ec0f5d57f03bb13d98742e8fbe03be4f908c10c9ae807fb6d995",
    "all_validation_outcomes.csv": "14a70bc7144ed47d08e2e2dafe74ae286e792e00758a1424003c560de43c4eb6",
    "deterministic_rebuild.json": "4461180ee9bc8ef0503641899f4aecfae4ee0de7fe2696531f514299fa41fadf",
    "inventory_summary.json": "66654f7096fcd76be3d69aa8ea92d8b325bad4b9c70257575bdf204c7cbbd8d3",
    "phase3c_abstentions.csv": "39b768af179cdef205ca31a1dab62b771d7fa0d8206cbfef9e16a42ae3392e68",
    "phase3c_compliance_cases.csv": "36af96b47707e9b53e4d72f931aa53c52180c517689087748f1f36903530cacc",
    "phase3c_incorrect_selections.csv": "53a7db0908edaee033d98ac695dddda1f2d243341622b4a55f7b9c9a52409f32",
    "phase3c_noncorrect_review.csv": "a2fd3e4016112ebc8bc11e13903df49e9f49b6313695a9457d867b69bf846b7c",
    "source_provenance.json": "519757f37f578515be387147b54ffa4df73dab929b9a678445e3d06157f2d133",
}
PROMPT1_MANIFEST_DIGEST_AT_COMMIT = "28d4f25aa1cd705f83ed6cc1c395fd48171956292e113903c37080ef88787c73"

EARLIER_FROZEN_DIGESTS = {
    "benchmark/manifest/model_registry.json": "8ba6de7b4171309828a043fa83835dcb1c47d202952152fa945119fa8cb1e367",
    "benchmark/PHASE2_MANIFEST.json": "438f6058bea5d85cabdb4174459e340827fda34ceb6778a38901600af0510ba8",
    "benchmark/phase3/validation/artifact_manifest.json": "8001818f261f861c0f463012821cc06ee5f1651a000d1ac48ca99e2437817693",
    "benchmark/phase3/validation_rescue_2048/artifact_manifest.json": "6f89c2ea102e228982d3dbd0bb846f2505bba88bf8c99362be062358dec33ca1",
    "benchmark/phase3/validation_rescue_2048/sensitivity/artifact_manifest.json": "5c269b63150a03974dbe999533d7fc7b9ab3e7b528711a3d49950e9108d3601e",
    "benchmark/phase3/retrieval_baselines/artifact_manifest.json": "32b80d73f6d078977ed7e2559b14bc7b1a116a5980463b06589e7c31c730f08e",
    "benchmark/phase3/phase3b_smoke/artifact_manifest.json": "f95ee881a0517f208462db4ad85016713b562a917ce19577ec2a592655bf2007",
    "benchmark/phase3/phase3b_full/artifact_manifest.json": "47aab9b3bea664e7078b9839619fb1c984305a1e5fa69fdd093ea2c066a5e0d9",
    "benchmark/phase3/phase3b_fusion/artifact_manifest.json": "6f4f6107c85174ed546818ce61b0e42f1e8db7c8ec723940fa164011a769ab7b",
    "benchmark/phase3/phase3c_smoke/artifact_manifest.json": "ce7ebcd2459e22498fa1d3cee22696d0ea7f7402a63436503cd84678a10f7128",
    "benchmark/phase3/phase3c_validation/artifact_manifest.json": "376c7e34e20afb5e72f45a40dc698330e85ca5489c6e50eecd842f63995c3647",
}

# The first row is the repeated header inside the supplied CSV block.  It is
# deliberately retained so the diagnostic artifact records it as malformed.
SUPPLIED_NOMINATIONS = [
    {
        "model_id": "model_id",
        "reaction_id": "reaction_id",
        "nomination_reason": "nomination_reason",
        "prior_source": "prior_source",
    },
    {
        "model_id": "BIOMD0000000013",
        "reaction_id": "E12",
        "nomination_reason": "R01429 contains xylonolactone which is not reflected in the reaction E12",
        "prior_source": "prior manual observation",
    },
]

# Targeted split membership copied from the frozen split artifact for supplied
# keys only.  Rebuilds therefore never scan or materialize held-out rows.
FROZEN_NOMINATION_SPLIT_MEMBERSHIP = {
    ("BIOMD0000000013", "E12"): ("train",),
}

MANUAL_FIELDS = [
    "model_id", "reaction_id", "nomination_reason", "prior_source",
    "audit_id", "validation_status", "matching_status",
]
DIAGNOSTIC_FIELDS = [
    "entry_number", "model_id", "reaction_id", "nomination_reason", "prior_source",
    "status", "matching_status", "validation_status", "test_status", "diagnostic",
]
BLINDED_FIELDS = ["blinded_order", "audit_id", "sample_id", "model_id", "reaction_id"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"field list required for empty CSV: {path.name}")
        fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hash(seed: int, namespace: str, *parts: str) -> str:
    payload = "\x1f".join([str(seed), namespace, *map(str, parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _key(row: Mapping[str, str]) -> tuple[str, str]:
    return str(row["model_id"]), str(row["reaction_id"])


def verify_frozen_sources() -> None:
    for name, digest in PROMPT1_ARTIFACT_DIGESTS.items():
        path = OUT / name
        actual = sha256_portable(path) if path.is_file() else "missing"
        if actual != digest:
            raise ValueError(f"Prompt 1 artifact changed: {name}: {actual} != {digest}")
    for relative, digest in {**DOCUMENTED_EDGE_SOURCES, **EARLIER_FROZEN_DIGESTS}.items():
        path = REPO_ROOT / relative
        actual = sha256_portable(path) if path.is_file() else "missing"
        if actual != digest:
            raise ValueError(f"earlier frozen artifact changed: {relative}: {actual} != {digest}")
    if sha256_portable(SPLITS) != SPLITS_DIGEST:
        raise ValueError("frozen Phase 3 split artifact changed")


def _targeted_split_lookup(keys: set[tuple[str, str]]) -> dict[tuple[str, str], tuple[str, ...]]:
    """Return preverified membership for supplied keys without scanning split rows."""
    return {key: FROZEN_NOMINATION_SPLIT_MEMBERSHIP[key] for key in keys if key in FROZEN_NOMINATION_SPLIT_MEMBERSHIP}


def match_nominations(
    nominations: Sequence[Mapping[str, str]],
    outcomes: Sequence[Mapping[str, str]],
    *,
    split_lookup: Callable[[set[tuple[str, str]]], Mapping[tuple[str, str], Sequence[str]]] = _targeted_split_lookup,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_key: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in outcomes:
        by_key[_key(row)].append(row)
    unresolved_keys: set[tuple[str, str]] = set()
    preliminary: list[tuple[int, Mapping[str, str], str]] = []
    for number, nomination in enumerate(nominations, 1):
        model = str(nomination.get("model_id") or "")
        reaction = str(nomination.get("reaction_id") or "")
        malformed = (
            not model or not reaction
            or (model == "model_id" and reaction == "reaction_id")
            or not str(nomination.get("nomination_reason") or "")
            or not str(nomination.get("prior_source") or "")
        )
        if malformed:
            preliminary.append((number, nomination, "malformed"))
        elif len(by_key[(model, reaction)]) == 1:
            preliminary.append((number, nomination, "matched"))
        elif len(by_key[(model, reaction)]) > 1:
            preliminary.append((number, nomination, "ambiguous"))
        else:
            preliminary.append((number, nomination, "unmatched"))
            unresolved_keys.add((model, reaction))
    memberships = split_lookup(unresolved_keys) if unresolved_keys else {}
    accepted: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []
    for number, nomination, state in preliminary:
        model = str(nomination.get("model_id") or "")
        reaction = str(nomination.get("reaction_id") or "")
        base = {
            "entry_number": str(number),
            "model_id": model,
            "reaction_id": reaction,
            "nomination_reason": str(nomination.get("nomination_reason") or ""),
            "prior_source": str(nomination.get("prior_source") or ""),
        }
        if state == "matched":
            match = by_key[(model, reaction)][0]
            if match.get("split") != "validation":
                raise ValueError("non-validation row exists in all_validation_outcomes.csv")
            accepted.append({
                "model_id": model,
                "reaction_id": reaction,
                "nomination_reason": base["nomination_reason"],
                "prior_source": base["prior_source"],
                "audit_id": str(match["audit_id"]),
                "validation_status": "confirmed_validation",
                "matching_status": "unique_exact_match",
            })
            continue
        if state == "malformed":
            values = ("unresolved_malformed", "malformed", "not_checked", "not_checked", "required fields missing or repeated CSV header row")
        elif state == "ambiguous":
            values = ("unresolved_ambiguous", "ambiguous_exact_key", "not_confirmed", "not_checked", "multiple validation inventory rows matched exact key")
        else:
            splits = sorted(set(map(str, memberships.get((model, reaction), []))))
            if "test" in splits:
                values = ("rejected_held_out_test", "not_in_validation_inventory", "not_validation", "confirmed_test_membership_only", "held-out nomination rejected; no test label read or recorded")
            elif splits:
                values = ("unresolved_not_validation", "not_in_validation_inventory", "not_validation", "confirmed_not_test", f"exact key belongs only to non-validation split(s): {';'.join(splits)}")
            else:
                values = ("unresolved_not_found", "no_exact_match", "not_confirmed", "not_found", "exact key was not found; no substitution made")
        diagnostics.append({
            **base,
            "status": values[0],
            "matching_status": values[1],
            "validation_status": values[2],
            "test_status": values[3],
            "diagnostic": values[4],
        })
    if len(accepted) > CATEGORY_QUOTAS[MANUAL_EDGE]:
        raise ValueError("more than five valid manual nominations; user selection required")
    return accepted, diagnostics


def load_documented_edges(outcome_keys: set[tuple[str, str]]) -> dict[tuple[str, str], list[str]]:
    edges: dict[tuple[str, str], list[str]] = defaultdict(list)
    for relative in DOCUMENTED_EDGE_SOURCES:
        payload = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
        for item in payload.get("examples") or []:
            example = item.get("example") if isinstance(item.get("example"), dict) else item
            key = (str(example.get("model_id") or ""), str(example.get("reaction_id") or ""))
            if key in outcome_keys:
                reason = f"{relative}:{item.get('category', 'documented_example')}"
                if reason not in edges[key]:
                    edges[key].append(reason)
    return dict(edges)


def failure_type(row: Mapping[str, str]) -> str | None:
    if row["phase3c_exact"] == "true":
        return None
    abstain = row["phase3c_abstain"] == "true"
    if row["fusion_exact"] == "true":
        return FAILURE_TYPES[1 if abstain else 0]
    if row["retrieval_state"] == "truth_at_fusion_ranks2_10":
        return FAILURE_TYPES[3 if abstain else 2]
    if row["retrieval_state"] == "truth_absent_from_fusion_top10" and not abstain:
        return FAILURE_TYPES[4]
    if row["phase3c_evidence_compliant"] == "false":
        return FAILURE_TYPES[6]
    return FAILURE_TYPES[5]


def disagreement_types(row: Mapping[str, str]) -> list[str]:
    exact = lambda field: row[field] == "true"
    reasons: list[str] = []
    if exact("bm25_exact") and not exact("trained_epoch1_exact"):
        reasons.append(DISAGREEMENT_TYPES[0])
    if exact("trained_epoch1_exact") and not exact("bm25_exact"):
        reasons.append(DISAGREEMENT_TYPES[1])
    if exact("fusion_exact") and row["bm25_top1"] != row["trained_epoch1_top1"]:
        reasons.append(DISAGREEMENT_TYPES[2])
    if exact("phase3c_exact") and not exact("phase3a_target_only_exact"):
        reasons.append(DISAGREEMENT_TYPES[3])
    if exact("phase3a_target_only_exact") and not exact("phase3c_exact"):
        reasons.append(DISAGREEMENT_TYPES[4])
    if exact("fusion_exact") and not exact("phase3c_exact"):
        reasons.append(DISAGREEMENT_TYPES[5])
    if not exact("fusion_exact") and exact("phase3c_exact"):
        reasons.append(DISAGREEMENT_TYPES[6])
    method_pairs = [
        ("phase2_exact", "phase2_brite"), ("bm25_exact", "bm25_brite"),
        ("trained_epoch1_exact", "trained_epoch1_brite"), ("fusion_exact", "fusion_brite"),
        ("phase3a_target_only_exact", "phase3a_target_only_brite"),
        ("phase3c_exact", "phase3c_brite"),
    ]
    if any(not exact(a) and exact(b) for a, b in method_pairs):
        reasons.append(DISAGREEMENT_TYPES[7])
    ranks = [int(row[field]) for field in (
        "phase2_ground_truth_rank", "bm25_ground_truth_rank", "trained_epoch1_ground_truth_rank", "fusion_ground_truth_rank"
    ) if row.get(field)]
    rank_fields = [row.get(field, "") for field in (
        "phase2_ground_truth_rank", "bm25_ground_truth_rank", "trained_epoch1_ground_truth_rank", "fusion_ground_truth_rank"
    )]
    if (ranks and max(ranks) - min(ranks) >= 10) or (ranks and any(not value for value in rank_fields)):
        reasons.append(DISAGREEMENT_TYPES[8])
    any_correct = any(exact(field) for field in (
        "phase2_exact", "bm25_exact", "trained_epoch1_exact", "fusion_exact", "phase3a_target_only_exact", "phase3c_exact"
    ))
    if any_correct and (row["phase3a_target_only_abstain"] == "true" or row["phase3c_abstain"] == "true"):
        reasons.append(DISAGREEMENT_TYPES[9])
    return reasons


def suspicion_rules(row: Mapping[str, str], documented: bool) -> list[str]:
    rules: list[str] = []
    if len(parse_multilabel_kegg_ids(row["ground_truth_kegg_ids"])) > 1:
        rules.append(SUSPICION_RULES[0])
    if row["ground_truth_catalog_presence_status"] != "all_in_catalog":
        rules.append(SUSPICION_RULES[1])
    if DISAGREEMENT_TYPES[7] in disagreement_types(row):
        rules.append(SUSPICION_RULES[2])
    return rules


def build_eligibility(rows: Sequence[Mapping[str, str]]) -> dict[str, dict[str, Any]]:
    keys = {_key(row) for row in rows}
    documented = load_documented_edges(keys)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _key(row)
        result[row["audit_id"]] = {
            "failure_type": failure_type(row),
            "failure_types": list(dict.fromkeys([
                *([failure_type(row)] if failure_type(row) else []),
                *([FAILURE_TYPES[6]] if row["phase3c_evidence_compliant"] == "false" else []),
            ])),
            "disagreement_types": disagreement_types(row),
            "suspicion_rules": suspicion_rules(row, key in documented),
            "documented_edges": documented.get(key, []),
            "manual_nomination": None,
        }
    return result


def _balanced_pick_one(
    candidates: Sequence[Mapping[str, str]],
    *,
    seed: int,
    namespace: str,
    selected: Sequence[Mapping[str, Any]],
    category_selected: Sequence[Mapping[str, Any]],
    balance_fields: Sequence[str] = (),
) -> Mapping[str, str]:
    cluster_total = Counter(item["row"]["cluster_id"] for item in selected)
    model_total = Counter(item["row"]["model_id"] for item in selected)
    cluster_category = Counter(item["row"]["cluster_id"] for item in category_selected)
    model_category = Counter(item["row"]["model_id"] for item in category_selected)
    extra_total = {
        field: Counter(item["row"][field] for item in selected)
        for field in balance_fields
    }
    extra_category = {
        field: Counter(item["row"][field] for item in category_selected)
        for field in balance_fields
    }
    return min(candidates, key=lambda row: (
        cluster_total[row["cluster_id"]],
        cluster_category[row["cluster_id"]],
        model_total[row["model_id"]],
        model_category[row["model_id"]],
        *(count for field in balance_fields for count in (
            extra_total[field][row[field]], extra_category[field][row[field]],
        )),
        _stable_hash(seed, namespace, row["audit_id"]),
        row["audit_id"],
    ))


def cluster_round_robin(
    candidates: Sequence[Mapping[str, str]],
    quota: int,
    *,
    seed: int,
    namespace: str,
    selected: Sequence[Mapping[str, Any]],
    category_selected: Sequence[Mapping[str, Any]],
    balance_fields: Sequence[str] = (),
) -> list[Mapping[str, str]]:
    remaining = {row["audit_id"]: row for row in candidates}
    chosen: list[Mapping[str, str]] = []
    while remaining and len(chosen) < quota:
        cluster_total = Counter(item["row"]["cluster_id"] for item in [*selected, *({"row": row} for row in chosen)])
        category_counts = Counter(item["row"]["cluster_id"] for item in category_selected)
        category_counts.update(row["cluster_id"] for row in chosen)
        clusters = sorted(
            {row["cluster_id"] for row in remaining.values()},
            key=lambda cluster: (
                cluster_total[cluster], category_counts[cluster],
                _stable_hash(seed, namespace + ":cluster", cluster), cluster,
            ),
        )
        made_progress = False
        for cluster in clusters:
            pool = [row for row in remaining.values() if row["cluster_id"] == cluster]
            if not pool:
                continue
            wrapped_selected = [*selected, *({"row": row} for row in chosen)]
            wrapped_category = [*category_selected, *({"row": row} for row in chosen)]
            row = _balanced_pick_one(
                pool, seed=seed, namespace=namespace + ":row", selected=wrapped_selected,
                category_selected=wrapped_category, balance_fields=balance_fields,
            )
            chosen.append(row)
            del remaining[row["audit_id"]]
            made_progress = True
            if len(chosen) == quota:
                break
        if not made_progress:
            break
    return chosen


def select_random_controls(rows: Sequence[Mapping[str, str]], seed: int = SAMPLING_SEED) -> list[Mapping[str, str]]:
    allowed = ("audit_id", "sample_id", "model_id", "reaction_id", "cluster_id", "split")
    stripped = [{field: row[field] for field in allowed} for row in rows]
    if {row["split"] for row in stripped} != {"validation"}:
        raise ValueError("random-control pool is not validation-only")
    return cluster_round_robin(
        stripped, CATEGORY_QUOTAS[RANDOM], seed=seed, namespace="random-control-v1",
        selected=[], category_selected=[],
    )


def _record_selection(
    selected: list[dict[str, Any]], category_selected: list[dict[str, Any]], row: Mapping[str, str],
    *, category: str, detail: str, priority: str, refill_reason: str,
) -> None:
    cluster_before = sum(item["row"]["cluster_id"] == row["cluster_id"] for item in selected)
    model_before = sum(item["row"]["model_id"] == row["model_id"] for item in selected)
    item = {
        "row": row,
        "primary_category": category,
        "primary_detail": detail,
        "selection_priority": priority,
        "refill_reason": refill_reason,
        "balance_cluster_count_at_selection": cluster_before,
        "balance_model_count_at_selection": model_before,
        "global_selection_order": len(selected) + 1,
    }
    selected.append(item)
    category_selected.append(item)


def _qualified_reasons(meta: Mapping[str, Any], primary: str) -> list[str]:
    reasons: list[str] = []
    if meta.get("manual_nomination"):
        reasons.append("manual_nomination")
    for edge in meta["documented_edges"]:
        reasons.append(f"documented_edge_case:{edge}")
    if meta["failure_type"]:
        reasons.append(f"category:{FAILURE}")
        reasons.extend(f"failure_type:{value}" for value in meta["failure_types"])
    if meta["disagreement_types"]:
        reasons.append(f"category:{DISAGREEMENT}")
        reasons.extend(f"disagreement:{value}" for value in meta["disagreement_types"])
    if meta["suspicion_rules"]:
        reasons.append(f"category:{SUSPICIOUS}")
        reasons.extend(f"suspicion:{value}" for value in meta["suspicion_rules"])
    marker = f"category:{primary}"
    return sorted(dict.fromkeys(value for value in reasons if value != marker))


def select_sample(
    rows: Sequence[Mapping[str, str]],
    eligibility: dict[str, dict[str, Any]],
    accepted: Sequence[Mapping[str, str]],
    *,
    seed: int = SAMPLING_SEED,
) -> list[dict[str, Any]]:
    by_audit = {row["audit_id"]: row for row in rows}
    for nomination in accepted:
        eligibility[nomination["audit_id"]]["manual_nomination"] = dict(nomination)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()

    random_category: list[dict[str, Any]] = []
    for stripped in select_random_controls(rows, seed):
        row = by_audit[stripped["audit_id"]]
        _record_selection(selected, random_category, row, category=RANDOM, detail="seeded_cluster_balanced_control", priority="", refill_reason="initial_random_control_selection")
        used.add(row["audit_id"])

    manual_category: list[dict[str, Any]] = []
    for nomination in sorted(accepted, key=lambda item: item["audit_id"]):
        if nomination["audit_id"] in used:
            continue
        row = by_audit[nomination["audit_id"]]
        _record_selection(selected, manual_category, row, category=MANUAL_EDGE, detail="manual_nomination", priority="0", refill_reason="accepted_manual_nomination")
        used.add(row["audit_id"])
    needed = CATEGORY_QUOTAS[MANUAL_EDGE] - len(manual_category)
    edge_pool = [row for row in rows if row["audit_id"] not in used and eligibility[row["audit_id"]]["documented_edges"]]
    fills = cluster_round_robin(edge_pool, needed, seed=seed, namespace="documented-edge-fill-v1", selected=selected, category_selected=manual_category)
    if len(fills) != needed:
        raise ValueError(f"only {len(fills)} documented validation edge cases available for {needed} fills")
    for row in fills:
        detail = "documented_edge_case_fill"
        _record_selection(selected, manual_category, row, category=MANUAL_EDGE, detail=detail, priority="1", refill_reason="fewer_than_five_valid_nominations_or_random_overlap")
        used.add(row["audit_id"])

    failure_category: list[dict[str, Any]] = []
    for priority, subtype in enumerate(FAILURE_TYPES, 1):
        needed = CATEGORY_QUOTAS[FAILURE] - len(failure_category)
        if needed <= 0:
            break
        pool = [row for row in rows if row["audit_id"] not in used and eligibility[row["audit_id"]]["failure_type"] == subtype]
        picks = cluster_round_robin(
            pool, needed, seed=seed, namespace=f"failure-{priority}-v1", selected=selected,
            category_selected=failure_category,
            balance_fields=("corrected_phase2_stratum", "target_seen_in_train"),
        )
        for row in picks:
            _record_selection(selected, failure_category, row, category=FAILURE, detail=subtype, priority=str(priority), refill_reason="priority_pool_after_deduplication")
            used.add(row["audit_id"])
    if len(failure_category) != CATEGORY_QUOTAS[FAILURE]:
        raise ValueError("insufficient Phase 3C non-exact outcomes after deduplication")

    disagreement_category: list[dict[str, Any]] = []
    for priority, subtype in enumerate(DISAGREEMENT_TYPES, 1):
        if len(disagreement_category) == CATEGORY_QUOTAS[DISAGREEMENT]:
            break
        pool = [row for row in rows if row["audit_id"] not in used and subtype in eligibility[row["audit_id"]]["disagreement_types"]]
        if not pool:
            continue
        row = _balanced_pick_one(pool, seed=seed, namespace=f"disagreement-{priority}-v1", selected=selected, category_selected=disagreement_category)
        _record_selection(selected, disagreement_category, row, category=DISAGREEMENT, detail=subtype, priority=str(priority), refill_reason="one_per_available_subtype_diversity_pass")
        used.add(row["audit_id"])
    needed = CATEGORY_QUOTAS[DISAGREEMENT] - len(disagreement_category)
    if needed:
        pool = [row for row in rows if row["audit_id"] not in used and eligibility[row["audit_id"]]["disagreement_types"]]
        for row in cluster_round_robin(pool, needed, seed=seed, namespace="disagreement-refill-v1", selected=selected, category_selected=disagreement_category):
            subtype = eligibility[row["audit_id"]]["disagreement_types"][0]
            _record_selection(selected, disagreement_category, row, category=DISAGREEMENT, detail=subtype, priority=str(DISAGREEMENT_TYPES.index(subtype) + 1), refill_reason="diversity_pass_shortfall_cluster_balanced_refill")
            used.add(row["audit_id"])
    if len(disagreement_category) != CATEGORY_QUOTAS[DISAGREEMENT]:
        raise ValueError("insufficient cross-method disagreement cases after deduplication")

    suspicious_category: list[dict[str, Any]] = []
    for priority, rule in enumerate(SUSPICION_RULES, 1):
        if len(suspicious_category) == CATEGORY_QUOTAS[SUSPICIOUS]:
            break
        pool = [row for row in rows if row["audit_id"] not in used and rule in eligibility[row["audit_id"]]["suspicion_rules"]]
        if not pool:
            continue
        row = _balanced_pick_one(pool, seed=seed, namespace=f"suspicion-{priority}-v1", selected=selected, category_selected=suspicious_category)
        _record_selection(selected, suspicious_category, row, category=SUSPICIOUS, detail=rule, priority=str(priority), refill_reason="one_per_available_mechanical_rule_pass")
        used.add(row["audit_id"])
    needed = CATEGORY_QUOTAS[SUSPICIOUS] - len(suspicious_category)
    if needed:
        pool = [row for row in rows if row["audit_id"] not in used and eligibility[row["audit_id"]]["suspicion_rules"]]
        for row in cluster_round_robin(pool, needed, seed=seed, namespace="suspicion-refill-v1", selected=selected, category_selected=suspicious_category):
            rule = eligibility[row["audit_id"]]["suspicion_rules"][0]
            _record_selection(selected, suspicious_category, row, category=SUSPICIOUS, detail=rule, priority=str(SUSPICION_RULES.index(rule) + 1), refill_reason="mechanical_rule_diversity_shortfall_cluster_balanced_refill")
            used.add(row["audit_id"])
    if len(suspicious_category) != CATEGORY_QUOTAS[SUSPICIOUS]:
        raise ValueError("insufficient mechanically suspicious cases after deduplication")

    if len(selected) != 60 or len(used) != 60:
        raise ValueError("formal audit sample is not exactly 60 unique reactions")
    for item in selected:
        item["secondary_eligibility_reasons"] = _qualified_reasons(eligibility[item["row"]["audit_id"]], item["primary_category"])
    return selected


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def eligibility_counts(rows: Sequence[Mapping[str, str]], eligibility: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "phase3-formal-audit-eligibility-counts-v1",
        "validation_population": len(rows),
        "primary_category_raw_eligibility": {
            RANDOM: len(rows),
            FAILURE: sum(bool(eligibility[row["audit_id"]]["failure_type"]) for row in rows),
            DISAGREEMENT: sum(bool(eligibility[row["audit_id"]]["disagreement_types"]) for row in rows),
            SUSPICIOUS: sum(bool(eligibility[row["audit_id"]]["suspicion_rules"]) for row in rows),
            MANUAL_EDGE: sum(bool(eligibility[row["audit_id"]]["manual_nomination"] or eligibility[row["audit_id"]]["documented_edges"]) for row in rows),
        },
        "failure_type_availability": {
            name: sum(name in meta["failure_types"] for meta in eligibility.values())
            for name in FAILURE_TYPES
        },
        "disagreement_subtype_availability": {name: sum(name in meta["disagreement_types"] for meta in eligibility.values()) for name in DISAGREEMENT_TYPES},
        "mechanical_rule_availability": {name: sum(name in meta["suspicion_rules"] for meta in eligibility.values()) for name in SUSPICION_RULES},
        "documented_edge_case_unique_validation_reactions": sum(bool(meta["documented_edges"]) for meta in eligibility.values()),
    }


def _qualifies(category: str, meta: Mapping[str, Any]) -> bool:
    if category == RANDOM:
        return True
    if category == MANUAL_EDGE:
        return bool(meta["manual_nomination"] or meta["documented_edges"])
    if category == FAILURE:
        return bool(meta["failure_type"])
    if category == DISAGREEMENT:
        return bool(meta["disagreement_types"])
    if category == SUSPICIOUS:
        return bool(meta["suspicion_rules"])
    raise ValueError(f"unknown category: {category}")


def add_stage_eligibility_counts(
    counts: dict[str, Any], rows: Sequence[Mapping[str, str]], eligibility: Mapping[str, Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> None:
    selected_by_category = {
        category: {item["row"]["audit_id"] for item in selected if item["primary_category"] == category}
        for category in CATEGORY_QUOTAS
    }
    used: set[str] = set()
    stages: dict[str, Any] = {}
    for category in SELECTION_ORDER:
        raw = {row["audit_id"] for row in rows if _qualifies(category, eligibility[row["audit_id"]])}
        available = raw - used
        details: dict[str, Any] = {
            "raw_eligible": len(raw),
            "deduplicated_by_prior_primary_assignment": len(raw & used),
            "eligible_at_selection_stage": len(available),
            "selected": len(selected_by_category[category]),
        }
        if category == FAILURE:
            details["subtype_availability_at_selection_stage"] = {
                subtype: sum(
                    audit_id in available and subtype in meta["failure_types"]
                    for audit_id, meta in eligibility.items()
                )
                for subtype in FAILURE_TYPES
            }
        elif category == DISAGREEMENT:
            details["subtype_availability_at_selection_stage"] = {
                subtype: sum(
                    audit_id in available and subtype in meta["disagreement_types"]
                    for audit_id, meta in eligibility.items()
                )
                for subtype in DISAGREEMENT_TYPES
            }
        elif category == SUSPICIOUS:
            details["rule_availability_at_selection_stage"] = {
                rule: sum(
                    audit_id in available and rule in meta["suspicion_rules"]
                    for audit_id, meta in eligibility.items()
                )
                for rule in SUSPICION_RULES
            }
        stages[category] = details
        used.update(selected_by_category[category])
    counts["selection_stage_eligibility"] = stages


def distribution(
    selected: Sequence[Mapping[str, Any]], eligibility: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [item["row"] for item in selected]
    secondary = [reason for item in selected for reason in item["secondary_eligibility_reasons"]]
    cluster = _counter(row["cluster_id"] for row in rows)
    model = _counter(row["model_id"] for row in rows)
    return {
        "schema": "phase3-formal-audit-sampling-distribution-v1",
        "total_unique_reactions": len({_key(row) for row in rows}),
        "primary_category": _counter(item["primary_category"] for item in selected),
        "cluster": cluster,
        "model": model,
        "corrected_phase2_stratum": _counter(row["corrected_phase2_stratum"] for row in rows),
        "seen_unseen": _counter("seen" if row["target_seen_in_train"] == "true" else "unseen" for row in rows),
        "retrieval_state": _counter(row["retrieval_state"] for row in rows),
        "represented_validation_clusters": len(cluster),
        "represented_models": len(model),
        "maximum_cases_from_any_cluster": max(cluster.values()),
        "maximum_cases_from_any_model": max(model.values()),
        "secondary_reason_overlap": _counter(secondary),
        "cases_with_secondary_eligibility": sum(bool(item["secondary_eligibility_reasons"]) for item in selected),
        "failure_type_primary_representation": _counter(
            item["primary_detail"] for item in selected if item["primary_category"] == FAILURE
        ),
        "disagreement_subtype_representation_in_subset": {
            subtype: sum(
                subtype in eligibility[item["row"]["audit_id"]]["disagreement_types"]
                for item in selected if item["primary_category"] == DISAGREEMENT
            )
            for subtype in DISAGREEMENT_TYPES
        },
        "mechanical_rule_representation_in_subset": {
            rule: sum(
                rule in eligibility[item["row"]["audit_id"]]["suspicion_rules"]
                for item in selected if item["primary_category"] == SUSPICIOUS
            )
            for rule in SUSPICION_RULES
        },
        "balance_relaxations": [],
    }


def sampling_config() -> dict[str, Any]:
    return {
        "schema": "phase3-formal-audit-sampling-config-v1",
        "algorithm_version": ALGORITHM_VERSION,
        "source_prompt1_commit": SOURCE_PROMPT1_COMMIT,
        "source_prompt1_manifest_sha256_at_commit": PROMPT1_MANIFEST_DIGEST_AT_COMMIT,
        "source_prompt1_artifact_digests": PROMPT1_ARTIFACT_DIGESTS,
        "sampling_seed": SAMPLING_SEED,
        "blinded_seed": BLINDED_SEED,
        "blinded_seed_derivation": "sampling_seed + 1",
        "category_quotas": CATEGORY_QUOTAS,
        "category_selection_order": SELECTION_ORDER,
        "category_presentation_order": PRESENTATION_ORDER,
        "failure_priority_order": FAILURE_TYPES,
        "disagreement_diversity_order": DISAGREEMENT_TYPES,
        "mechanical_suspicion_rule_order": SUSPICION_RULES,
        "random_control_allowed_selection_fields": ["audit_id", "sample_id", "model_id", "reaction_id", "cluster_id", "split"],
        "random_control_excluded_features": ["correctness", "abstention", "disagreement", "suspicion", "catalog_presence", "method_outcome"],
        "cluster_balancing": "lowest prior total cluster count, then lowest category cluster count, then seeded SHA-256 cluster tie-break; one case per cluster per round",
        "model_balancing": "within the chosen cluster, lowest prior total model count, then lowest category model count, then seeded SHA-256 audit-ID tie-break",
        "failure_additional_balancing": "after cluster and model balance, prefer the least represented corrected stratum and seen/unseen group within the failure category",
        "deduplication": "first assigned primary category wins; later categories refill from remaining eligible reactions",
        "stable_tie_break": "SHA-256(seed, namespace, stable identifier), then audit_id",
        "documented_edge_sources": DOCUMENTED_EDGE_SOURCES,
        "nomination_split_membership_source": {
            "path": "benchmark/phase3/splits.csv",
            "sha256": SPLITS_DIGEST,
            "targeted_membership_only": {
                f"{model_id},{reaction_id}": list(splits)
                for (model_id, reaction_id), splits in FROZEN_NOMINATION_SPLIT_MEMBERSHIP.items()
            },
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
        },
        "earlier_frozen_artifact_digests": EARLIER_FROZEN_DIGESTS,
        "builds_required_for_freeze": 2,
        "biological_judgments_made": False,
        "test_rows_loaded": 0,
        "test_labels_loaded": 0,
        "api_calls": 0,
        "new_inference": False,
    }


def _formal_rows(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    category_rank = {name: number for number, name in enumerate(PRESENTATION_ORDER)}
    ordered = sorted(selected, key=lambda item: (category_rank[item["primary_category"]], item["global_selection_order"]))
    category_counts: Counter[str] = Counter()
    result: list[dict[str, str]] = []
    for item in ordered:
        row = item["row"]
        category_counts[item["primary_category"]] += 1
        result.append({
            "audit_id": row["audit_id"],
            "sample_id": row["sample_id"],
            "model_id": row["model_id"],
            "reaction_id": row["reaction_id"],
            "cluster_id": row["cluster_id"],
            "split": row["split"],
            "primary_category": item["primary_category"],
            "primary_category_rank": str(category_counts[item["primary_category"]]),
            "secondary_eligibility_reasons": _json(item["secondary_eligibility_reasons"]),
            "selection_priority": item["selection_priority"],
            "global_selection_order": str(item["global_selection_order"]),
        })
    return result


def _crosswalk_rows(selected: Sequence[Mapping[str, Any]], eligibility: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    formal = _formal_rows(selected)
    by_audit = {item["row"]["audit_id"]: item for item in selected}
    rows: list[dict[str, str]] = []
    for base in formal:
        item = by_audit[base["audit_id"]]
        source = item["row"]
        meta = eligibility[source["audit_id"]]
        nomination = meta["manual_nomination"] or {}
        rows.append({
            **base,
            "primary_detail": item["primary_detail"],
            "refill_reason": item["refill_reason"],
            "balance_cluster_count_at_selection": str(item["balance_cluster_count_at_selection"]),
            "balance_model_count_at_selection": str(item["balance_model_count_at_selection"]),
            "sampling_seed": str(SAMPLING_SEED),
            "blinded_seed": str(BLINDED_SEED),
            "corrected_phase2_stratum": source["corrected_phase2_stratum"],
            "target_seen_in_train": source["target_seen_in_train"],
            "retrieval_state": source["retrieval_state"],
            "ground_truth_id_count": source["ground_truth_id_count"],
            "ground_truth_catalog_presence_status": source["ground_truth_catalog_presence_status"],
            "failure_type": meta["failure_type"] or "",
            "failure_types": _json(meta["failure_types"]),
            "disagreement_types": _json(meta["disagreement_types"]),
            "suspicion_rules": _json(meta["suspicion_rules"]),
            "documented_edge_sources": _json(meta["documented_edges"]),
            "manual_nomination_reason": str(nomination.get("nomination_reason") or ""),
            "manual_nomination_prior_source": str(nomination.get("prior_source") or ""),
            "bm25_exact": source["bm25_exact"],
            "trained_epoch1_exact": source["trained_epoch1_exact"],
            "fusion_exact": source["fusion_exact"],
            "phase3a_target_only_abstain": source["phase3a_target_only_abstain"],
            "phase3a_target_only_exact": source["phase3a_target_only_exact"],
            "phase3c_abstain": source["phase3c_abstain"],
            "phase3c_exact": source["phase3c_exact"],
            "phase3c_evidence_compliant": source["phase3c_evidence_compliant"],
        })
    return rows


def _blinded_rows(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    ordered = sorted((item["row"] for item in selected), key=lambda row: (_stable_hash(BLINDED_SEED, "blinded-order-v1", row["audit_id"]), row["audit_id"]))
    return [{
        "blinded_order": str(number),
        "audit_id": row["audit_id"],
        "sample_id": row["sample_id"],
        "model_id": row["model_id"],
        "reaction_id": row["reaction_id"],
    } for number, row in enumerate(ordered, 1)]


def _report(
    selected: Sequence[Mapping[str, Any]], accepted: Sequence[Mapping[str, str]], diagnostics: Sequence[Mapping[str, str]],
    counts: Mapping[str, Any], dist: Mapping[str, Any],
) -> str:
    category_lines = "\n".join(f"| `{name}` | {CATEGORY_QUOTAS[name]} | {dist['primary_category'].get(name, 0)} |" for name in PRESENTATION_ORDER)
    diagnostic_lines = "\n".join(f"- `{row['model_id']},{row['reaction_id']}`: {row['status']} — {row['diagnostic']}" for row in diagnostics) or "- None."
    fills = [item for item in selected if item["primary_category"] == MANUAL_EDGE and item["primary_detail"] == "documented_edge_case_fill"]
    fill_lines = "\n".join(f"- `{item['row']['audit_id']}` from {', '.join(json.loads(_json(item['secondary_eligibility_reasons'])))}" for item in fills)
    rejected = sum(row["status"].startswith("rejected_") for row in diagnostics)
    unresolved = sum(row["status"].startswith("unresolved_") for row in diagnostics)
    stage_lines = "\n".join(
        f"- `{category}`: {values['raw_eligible']} raw, {values['deduplicated_by_prior_primary_assignment']} already assigned, {values['eligible_at_selection_stage']} available, {values['selected']} selected."
        for category, values in counts["selection_stage_eligibility"].items()
    )
    return f"""# Formal 60-reaction audit sampling report

## Freeze provenance

The formal sample was built from Prompt 1 commit `{SOURCE_PROMPT1_COMMIT}` using `{ALGORITHM_VERSION}`. The Prompt 1 manifest digest at that commit was `{PROMPT1_MANIFEST_DIGEST_AT_COMMIT}`; all individual Prompt 1 artifact digests are frozen in `sampling_config.json`. Sampling seed: `{SAMPLING_SEED}`. Blinded-order seed: `{BLINDED_SEED}`, derived as sampling seed plus one.

All Phase 1, Phase 2, Phase 3A, retrieval-baseline, Phase 3B, Phase 3C, and Prompt 1 digest gates passed before sampling. No held-out test labels or rows were loaded.

## Prespecified algorithm

Random controls were selected first from all 163 validation reactions using only stable identity, cluster, model, and split fields. Clusters are visited round-robin by their lowest current representation; seeded SHA-256 breaks ties. Within a cluster, the least represented model is preferred, followed by a seeded audit-ID tie-break. Outcome, correctness, abstention, disagreement, suspicion, and catalog fields cannot affect random-control selection. Within each Phase 3C failure priority pool, corrected stratum and seen/unseen representation are additional tie-break balances after cluster and model.

Primary assignment order was: {', '.join(f'`{value}`' for value in SELECTION_ORDER)}. A first assignment always wins. Later categories exclude selected audit IDs and refill deterministically. Phase 3C cases exhaust the seven priority pools in the configured order. Disagreement and suspicion categories take one balanced case per available subtype/rule before a cluster-balanced refill. No balance constraint required relaxation.

## Eligibility before selection

- Validation population: {counts['validation_population']}.
- Raw Phase 3C non-exact eligibility: {counts['primary_category_raw_eligibility'][FAILURE]}.
- Raw cross-method disagreement eligibility: {counts['primary_category_raw_eligibility'][DISAGREEMENT]}.
- Raw mechanical-suspicion eligibility: {counts['primary_category_raw_eligibility'][SUSPICIOUS]}.
- Previously documented unique validation edge cases: {counts['documented_edge_case_unique_validation_reactions']}.
- Failure-type availability: `{_json(counts['failure_type_availability'])}`.
- Disagreement-subtype availability: `{_json(counts['disagreement_subtype_availability'])}`.
- Mechanical-rule availability: `{_json(counts['mechanical_rule_availability'])}`.

Stage eligibility after first-assignment deduplication:

{stage_lines}

Failure subtype availability at its selection stage: `{_json(counts['selection_stage_eligibility'][FAILURE]['subtype_availability_at_selection_stage'])}`. Disagreement subtype availability at its selection stage: `{_json(counts['selection_stage_eligibility'][DISAGREEMENT]['subtype_availability_at_selection_stage'])}`. Mechanical-rule availability at its selection stage: `{_json(counts['selection_stage_eligibility'][SUSPICIOUS]['rule_availability_at_selection_stage'])}`.

## Manual nominations and documented fills

Accepted nominations: {len(accepted)}. Rejected nominations: {rejected}. Unresolved nominations: {unresolved}.

{diagnostic_lines}

Documented edge-case fills ({len(fills)}):

{fill_lines}

No corrected KEGG label was inferred, and nomination was not treated as evidence that a frozen label is wrong.

## Exact sample accounting

| Primary category | Quota | Selected |
| --- | ---: | ---: |
{category_lines}

Unique reactions: {dist['total_unique_reactions']}. Failure-type representation: `{_json(dist['failure_type_primary_representation'])}`. Disagreement subtype representation (counting every subtype attached to the 10 selected disagreement cases): `{_json(dist['disagreement_subtype_representation_in_subset'])}`. Mechanical-rule representation: `{_json(dist['mechanical_rule_representation_in_subset'])}`. Full secondary overlaps are in `sampling_distribution.json` and the private crosswalk.

## Distribution

- Clusters ({dist['represented_validation_clusters']} represented; maximum {dist['maximum_cases_from_any_cluster']} cases): `{_json(dist['cluster'])}`.
- Models ({dist['represented_models']} represented; maximum {dist['maximum_cases_from_any_model']} cases): `{_json(dist['model'])}`.
- Corrected strata: `{_json(dist['corrected_phase2_stratum'])}`.
- Seen/unseen: `{_json(dist['seen_unseen'])}`.
- Retrieval states: `{_json(dist['retrieval_state'])}`.
- Cases with at least one secondary eligibility reason: {dist['cases_with_secondary_eligibility']}.

## Blinding preparation

`formal_audit_blinded_order.csv` contains only blinded order, stable audit ID, sample ID, model ID, and reaction ID. It excludes category, suspicion, failure, correctness, method outcome, priority, and random/error-enriched status. Its deterministic seed differs from the sampling seed, and its order differs from category presentation and selection order. Prompt 3 may use this skeleton to build Pass 1 materials; no reviewer forms or evidence packets were created here.

## Limitations

The full 60-case sample deliberately contains 40 error-enriched or edge-enriched cases, so it cannot estimate population label-error prevalence. Only the 20 outcome-independent random controls support an approximately unbiased prevalence estimate, subject to finite-sample uncertainty and the prespecified cluster/model balancing design. Mechanical flags, system failures, disagreements, and nominations are review triggers—not biological verdicts. No biological adjudication, reviewer verdict, API call, new inference, label change, or corrected-label sensitivity analysis was performed.
"""


def verify_sample_invariants(
    formal: Sequence[Mapping[str, str]], crosswalk: Sequence[Mapping[str, str]], blinded: Sequence[Mapping[str, str]],
) -> None:
    keys = lambda values: {(row["model_id"], row["reaction_id"]) for row in values}
    if len(formal) != 60 or len(keys(formal)) != 60:
        raise ValueError("formal sample must contain exactly 60 unique reactions")
    if {row["split"] for row in formal} != {"validation"}:
        raise ValueError("non-validation reaction entered formal sample")
    if _counter(row["primary_category"] for row in formal) != dict(sorted(CATEGORY_QUOTAS.items())):
        raise ValueError("formal sample category quota mismatch")
    if keys(formal) != keys(crosswalk) or keys(formal) != keys(blinded):
        raise ValueError("formal CSV, crosswalk, and blinded skeleton do not align")
    if list(blinded[0]) != BLINDED_FIELDS or any(set(row) != set(BLINDED_FIELDS) for row in blinded):
        raise ValueError("blinded skeleton schema leaked private fields")
    category_order = [row["audit_id"] for row in formal]
    blind_order = [row["audit_id"] for row in blinded]
    if category_order == blind_order:
        raise ValueError("blinded order did not differ from category presentation order")


def build_sampling_bundle(out: Path = OUT) -> dict[str, Any]:
    verify_frozen_sources()
    rows = _read_csv(INVENTORY)
    if len(rows) != 163 or len({_key(row) for row in rows}) != 163 or {row["split"] for row in rows} != {"validation"}:
        raise ValueError("Prompt 1 validation inventory invariant failed")
    if [row["audit_id"] for row in rows] != [f"P3EA{number:04d}" for number in range(1, 164)]:
        raise ValueError("Prompt 1 stable audit IDs changed")
    accepted, diagnostics = match_nominations(SUPPLIED_NOMINATIONS, rows)
    eligibility = build_eligibility(rows)
    for nomination in accepted:
        eligibility[nomination["audit_id"]]["manual_nomination"] = dict(nomination)
    selected = select_sample(rows, eligibility, accepted)
    counts = eligibility_counts(rows, eligibility)
    add_stage_eligibility_counts(counts, rows, eligibility, selected)
    formal = _formal_rows(selected)
    crosswalk = _crosswalk_rows(selected, eligibility)
    blinded = _blinded_rows(selected)
    verify_sample_invariants(formal, crosswalk, blinded)
    dist = distribution(selected, eligibility)

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "formal_audit_sample.csv", formal)
    _write_jsonl(out / "formal_audit_sample.jsonl", formal)
    _write_csv(out / "formal_audit_crosswalk.csv", crosswalk)
    _write_csv(out / "formal_audit_blinded_order.csv", blinded, BLINDED_FIELDS)
    _write_csv(out / "manual_nominations.csv", accepted, MANUAL_FIELDS)
    _write_csv(out / "manual_nomination_diagnostics.csv", diagnostics, DIAGNOSTIC_FIELDS)
    atomic_write_json(sampling_config(), out / "sampling_config.json")
    atomic_write_json(counts, out / "sampling_eligibility_counts.json")
    atomic_write_json(dist, out / "sampling_distribution.json")
    (out / "SAMPLING_REPORT.md").write_text(_report(selected, accepted, diagnostics, counts, dist), encoding="utf-8", newline="\n")
    return {
        "selected": selected,
        "accepted": accepted,
        "diagnostics": diagnostics,
        "eligibility_counts": counts,
        "distribution": dist,
    }


SAMPLING_OUTPUTS = [
    "formal_audit_sample.csv", "formal_audit_sample.jsonl", "formal_audit_crosswalk.csv",
    "formal_audit_blinded_order.csv", "manual_nominations.csv", "manual_nomination_diagnostics.csv",
    "sampling_config.json", "sampling_eligibility_counts.json", "sampling_distribution.json", "SAMPLING_REPORT.md",
]


def build_twice(out: Path = OUT) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="_error_sampling_a_", dir=PHASE3_DIR) as first_name, tempfile.TemporaryDirectory(prefix="_error_sampling_b_", dir=PHASE3_DIR) as second_name:
        first, second = Path(first_name), Path(second_name)
        build_sampling_bundle(first)
        build_sampling_bundle(second)
        first_bytes = {name: (first / name).read_bytes() for name in SAMPLING_OUTPUTS}
        second_bytes = {name: (second / name).read_bytes() for name in SAMPLING_OUTPUTS}
        if first_bytes != second_bytes:
            raise ValueError("sampling rebuilds are not byte-identical")
    result = build_sampling_bundle(out)
    artifacts = [path for path in out.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    write_artifact_manifest(out, artifacts)
    return result


def verify_manifest(out: Path = OUT) -> list[str]:
    manifest_path = out / "artifact_manifest.json"
    before = manifest_path.read_bytes()
    manifest = json.loads(before)
    problems: list[str] = []
    paths = [str(item["path"]) for item in manifest.get("files") or []]
    if len(paths) != len(set(paths)):
        problems.append("duplicate manifest paths")
    for item in manifest.get("files") or []:
        artifact = REPO_ROOT / item["path"]
        if not artifact.is_file() or sha256_portable(artifact) != item["sha256"]:
            problems.append(f"digest mismatch: {item['path']}")
    try:
        formal = _read_csv(out / "formal_audit_sample.csv")
        crosswalk = _read_csv(out / "formal_audit_crosswalk.csv")
        blinded = _read_csv(out / "formal_audit_blinded_order.csv")
        verify_sample_invariants(formal, crosswalk, blinded)
    except (OSError, ValueError) as exc:
        problems.append(f"sample invariant failure: {exc}")
    if manifest_path.read_bytes() != before:
        problems.append("read-only verification mutated manifest")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        problems = verify_manifest(args.out)
        print(json.dumps({"n_problems": len(problems), "problems": problems}, sort_keys=True))
        return int(bool(problems))
    result = build_twice(args.out)
    print(json.dumps({
        "accepted_nominations": len(result["accepted"]),
        "diagnostic_nominations": len(result["diagnostics"]),
        "distribution": result["distribution"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
