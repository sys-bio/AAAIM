"""Build deterministic validation-only inventories for the reaction-label audit.

This milestone is descriptive inventory construction. It reads only frozen
validation artifacts, performs no inference or biological adjudication, and
never opens the held-out test set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import os
import pickle
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

from benchmark.scripts.kegg_equivalence import match_kinds
from benchmark.scripts.phase3_common import (
    PHASE3_DIR,
    atomic_write_json,
    repo_relative_posix,
    sha256_portable,
    write_artifact_manifest,
)

SOURCE_COMMIT = "c61f2ff5e259e82f4eddc5725861a67fa51998b9"
OUT = PHASE3_DIR / "error_audit"
VALIDATION = PHASE3_DIR / "phase3c_validation"
SAMPLE = PHASE3_DIR / "pilot_sample.csv"
ANSWER_KEY = PHASE3_DIR / "pilot_answer_key.csv"
SCORED = VALIDATION / "answer_key_join.jsonl"
EVIDENCE = VALIDATION / "frozen_evidence.jsonl"
RESPONSES = VALIDATION / "frozen_responses.jsonl"
REQUESTS = VALIDATION / "requests.jsonl"
PHASE2_RANKINGS = PHASE3_DIR / "retrieval_baselines" / "rankings_phase2_rule_based.jsonl"
BM25_RANKINGS = PHASE3_DIR / "retrieval_baselines" / "rankings_bm25.jsonl"
TRAINED_RANKINGS = PHASE3_DIR / "phase3b_full" / "rankings_epoch_1.jsonl"
FUSION_RANKINGS = PHASE3_DIR / "phase3b_fusion" / "rankings_bm25_trained_epoch1_rrf.jsonl.xz"
CATALOG = REPO_ROOT / "data" / "kegg" / "kegg_reaction_features.lzma"

SOURCE_DIGESTS = {
    "benchmark/phase3/pilot_sample.csv": "be086250023be617df278ab62549756b23bd477f8f5d6de92531608dcbf1088a",
    "benchmark/phase3/pilot_answer_key.csv": "62415034be49a11dd672e0940675f8ebe827cb217b79a773386f1c4d20bb4c83",
    "benchmark/phase3/phase3c_validation/answer_key_join.jsonl": "01844afb38ff036a0c139896e1b0ea70163b8e0324c74c952e431bba01f4ad43",
    "benchmark/phase3/phase3c_validation/frozen_evidence.jsonl": "b6045e9a77c22c884642b567dd8f011a76ad4f943b2fa3e33b96f87fcf490a53",
    "benchmark/phase3/phase3c_validation/frozen_responses.jsonl": "72e14d505b67a2aebeaebd844de26164da0d404bd7cff7df662cd765cf344999",
    "benchmark/phase3/phase3c_validation/requests.jsonl": "ce962521b6503ed217c22dd640ab7cc46a5f8ead1f49c747107904cf746ccf95",
    "benchmark/phase3/retrieval_baselines/rankings_phase2_rule_based.jsonl": "e22954bb0b8cee75ff17511d96fc42b923217003388f3e4e4a7e03200ef31c0d",
    "benchmark/phase3/retrieval_baselines/rankings_bm25.jsonl": "f0b700248e388bd75fb7bca9c05a46ecbb6b5802861a8631eadb944e0b1572f0",
    "benchmark/phase3/phase3b_full/rankings_epoch_1.jsonl": "660c7ff55d787928050ba963cf4236788de21f661a523a5ebc1c47abbfc2f304",
    "benchmark/phase3/phase3b_fusion/rankings_bm25_trained_epoch1_rrf.jsonl.xz": "9601149e0297831ebfbfb4fde24b725a9b00287cf1b4d74270bf2921be138b31",
    "data/kegg/kegg_reaction_features.lzma": "00acb4de1bbfb1ddae0298a09d218f93eb22241844179439ff34a2912fba046c",
}
SOURCE_PATHS = tuple(SOURCE_DIGESTS)

EXPECTED = {
    "total": 163,
    "grounded_exact": 102,
    "incorrect_selections": 22,
    "abstentions": 39,
    "noncorrect": 61,
    "unsupported": 0,
    "compliance_cases": 2,
}

HUMAN_REVIEW_FIELDS = (
    "human_review_status",
    "human_label_verdict",
    "human_corrected_ground_truth_kegg_ids",
    "human_error_type",
    "human_notes",
    "human_reviewer",
    "human_review_date",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def parse_multilabel_kegg_ids(value: Any) -> list[str]:
    """Parse legacy pipe or current semicolon multi-label values without loss."""
    tokens = re.split(r"[;|,\s]+", str(value or "").strip())
    out: list[str] = []
    for token in tokens:
        if not token:
            continue
        if not re.fullmatch(r"R[0-9]{5}", token):
            raise ValueError(f"malformed KEGG reaction identifier in multi-label value: {token!r}")
        if token not in out:
            out.append(token)
    return out


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _by_sample(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out = {str(row["sample_id"]): row for row in rows}
    if len(out) != EXPECTED["total"]:
        raise ValueError("sample-keyed frozen artifact is not exactly 163 unique rows")
    return out


def _ranking_map(path: Path, wanted: set[tuple[str, str]]) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for row in _read_jsonl(path):
        key = (str(row["model_id"]), str(row["reaction_id"]))
        if key in wanted:
            if key in out:
                raise ValueError(f"duplicate ranking row in {repo_relative_posix(path)}: {key}")
            out[key] = [str(value) for value in row.get("ranked_ids") or []]
    if set(out) != wanted:
        raise ValueError(f"ranking coverage mismatch in {repo_relative_posix(path)}")
    return out


def _rank_truth(ranking: Sequence[str], truth: set[str]) -> int | None:
    return next((rank for rank, identifier in enumerate(ranking, 1) if identifier in truth), None)


def _method_match(prediction: str | None, truth: set[str]) -> tuple[bool, bool]:
    if not prediction:
        return False, False
    kinds = match_kinds(prediction, truth)
    return bool(kinds["exact"]), bool(kinds["brite_orthology"])


def _normalized_equation(request: Mapping[str, Any]) -> str:
    payload = request.get("payload") or {}
    text = str(payload.get("input") or "")
    if not text.startswith("TARGET\n") or "\nEVIDENCE\n" not in text:
        raise ValueError("unexpected frozen Phase 3C request format")
    target_text = text[len("TARGET\n"):].split("\nEVIDENCE\n", 1)[0]
    target = json.loads(target_text)
    query = str(target.get("query") or "")
    first_line = query.splitlines()[0]
    if not first_line.startswith("Equation: "):
        raise ValueError("frozen target query lacks normalized equation")
    return re.sub(r"\s+", " ", first_line[len("Equation: "):].strip())


def _catalog_bundle(truth_ids: Sequence[str], catalog: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    membership = {identifier: ("in_catalog" if identifier in catalog else "absent_from_catalog") for identifier in truth_ids}
    if all(value == "in_catalog" for value in membership.values()):
        status = "all_in_catalog"
    elif all(value == "absent_from_catalog" for value in membership.values()):
        status = "absent_from_catalog"
    else:
        status = "partially_in_catalog"
    records = [
        {
            "catalog_membership": membership[identifier],
            "kegg_id": identifier,
            "record": dict(catalog[identifier]) if identifier in catalog else None,
        }
        for identifier in truth_ids
    ]
    return status, _canonical_json({"membership": membership, "records": records})


def _transition(grounded_exact: bool, fusion_exact: bool) -> str:
    if grounded_exact and fusion_exact:
        return "both_correct"
    if grounded_exact:
        return "grounded_only_correct"
    if fusion_exact:
        return "fusion_only_correct"
    return "neither_correct"


def _mechanical_category(state: str, exact: bool, abstain: bool) -> str:
    if exact:
        return "grounded_exact"
    if state == "truth_at_fusion_rank1":
        return "abstention_when_fusion_top1_correct" if abstain else "incorrect_prediction_when_fusion_top1_correct"
    if state == "truth_at_fusion_ranks2_10":
        return "abstention_when_truth_at_fusion_ranks2_10" if abstain else "incorrect_prediction_when_truth_at_fusion_ranks2_10"
    return "abstention_when_truth_absent_from_fusion_top10" if abstain else "incorrect_in_evidence_prediction_when_truth_absent_from_fusion_top10"


def _disagreement_reasons(row: Mapping[str, str]) -> list[str]:
    predictions = [
        row[name] for name in (
            "phase2_prediction", "bm25_top1", "trained_epoch1_top1", "fusion_top1",
            "phase3a_target_only_prediction", "phase3c_prediction",
        ) if row[name]
    ]
    reasons: list[str] = []
    if len(set(predictions)) > 1:
        reasons.append("method_top1_disagreement")
    for prefix in ("phase2", "bm25", "trained_epoch1", "fusion", "phase3a_target_only", "phase3c"):
        if row.get(f"{prefix}_prediction") or prefix in {"bm25", "trained_epoch1", "fusion"}:
            if row[f"{prefix}_exact"] != row[f"{prefix}_brite"]:
                reasons.append("exact_and_brite_correctness_differ")
                break
    available = [
        (bool(row["phase2_prediction"]), row["phase2_exact"] == "true"),
        (True, row["bm25_exact"] == "true"),
        (True, row["trained_epoch1_exact"] == "true"),
        (True, row["fusion_exact"] == "true"),
        (bool(row["phase3a_target_only_prediction"]), row["phase3a_target_only_exact"] == "true"),
        (bool(row["phase3c_prediction"]), row["phase3c_exact"] == "true"),
    ]
    exact_values = [exact for is_available, exact in available if is_available]
    if any(exact_values) and not all(exact_values):
        reasons.append("one_method_correct_another_incorrect")
    if row["grounded_vs_fusion_transition"] == "grounded_only_correct":
        reasons.append("grounded_helps_fusion")
    if row["grounded_vs_fusion_transition"] == "fusion_only_correct":
        reasons.append("grounded_harms_fusion")
    phase3a_outcome = "ABSTAIN" if row["phase3a_target_only_abstain"] == "true" else row["phase3a_target_only_prediction"]
    phase3c_outcome = "ABSTAIN" if row["phase3c_abstain"] == "true" else row["phase3c_prediction"]
    if phase3a_outcome != phase3c_outcome:
        reasons.append("direct_open_set_and_grounded_differ")
    if (row["phase3a_target_only_abstain"] == "true" or row["phase3c_abstain"] == "true") and any(exact_values):
        reasons.append("one_method_abstains_while_another_is_correct")
    if row["ground_truth_catalog_presence_status"] != "all_in_catalog":
        reasons.append("ground_truth_absent_from_catalog")
    if int(row["ground_truth_id_count"]) > 1:
        reasons.append("multiple_ground_truth_identifiers")
    return list(dict.fromkeys(reasons))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty inventory: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent CSV schema for {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def verify_sources() -> None:
    for relative, expected in SOURCE_DIGESTS.items():
        path = REPO_ROOT / relative
        actual = sha256_portable(path) if path.is_file() else "missing"
        if actual != expected:
            raise ValueError(f"frozen source digest mismatch: {relative}: {actual} != {expected}")


def build_rows() -> list[dict[str, str]]:
    verify_sources()
    samples = sorted(_read_csv(SAMPLE), key=lambda row: row["sample_id"])
    if len(samples) != EXPECTED["total"] or len({row["sample_id"] for row in samples}) != EXPECTED["total"]:
        raise ValueError("frozen sample is not exactly 163 unique sample IDs")
    if {row["split"] for row in samples} != {"validation"}:
        raise ValueError("non-validation row entered error inventory")
    keys = {(row["model_id"], row["reaction_id"]) for row in samples}
    if len(keys) != EXPECTED["total"]:
        raise ValueError("duplicate reaction key in frozen sample")

    answers = _by_sample(_read_csv(ANSWER_KEY))
    scored = _by_sample(_read_jsonl(SCORED))
    evidence = _by_sample(_read_jsonl(EVIDENCE))
    responses = _by_sample(_read_jsonl(RESPONSES))
    requests = _by_sample(_read_jsonl(REQUESTS))
    phase2 = _ranking_map(PHASE2_RANKINGS, keys)
    bm25 = _ranking_map(BM25_RANKINGS, keys)
    trained = _ranking_map(TRAINED_RANKINGS, keys)
    fusion = _ranking_map(FUSION_RANKINGS, keys)
    catalog = pickle.loads(lzma.open(CATALOG, "rb").read())
    if len(catalog) != 12_312:
        raise ValueError("frozen KEGG catalog size changed")

    source_paths_json = _canonical_json(list(SOURCE_PATHS))
    source_digests_json = _canonical_json(SOURCE_DIGESTS)
    rows: list[dict[str, str]] = []
    for number, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        key = (sample["model_id"], sample["reaction_id"])
        answer = answers[sid]
        score = scored[sid]
        response = responses[sid]
        request = requests[sid]
        evidence_row = evidence[sid]
        truth_ids = parse_multilabel_kegg_ids(answer["ground_truth_kegg_all"])
        if set(truth_ids) != set(score["ground_truth_ids"]) or len(truth_ids) != int(answer["num_ground_truth_ids"]):
            raise ValueError(f"multi-label ground truth mismatch for {sid}")
        truth = set(truth_ids)
        p2_rank = _rank_truth(phase2[key], truth)
        bm25_rank = _rank_truth(bm25[key], truth)
        trained_rank = _rank_truth(trained[key], truth)
        fusion_rank = _rank_truth(fusion[key], truth)
        state = (
            "truth_at_fusion_rank1" if fusion_rank == 1
            else "truth_at_fusion_ranks2_10" if fusion_rank is not None and fusion_rank <= 10
            else "truth_absent_from_fusion_top10"
        )
        expected_state = {
            "A_truth_at_rank1": "truth_at_fusion_rank1",
            "B_truth_at_ranks2_10": "truth_at_fusion_ranks2_10",
            "C_truth_absent_top10": "truth_absent_from_fusion_top10",
        }[str(score["retrieval_state"])]
        scored_top10_rank = fusion_rank if fusion_rank is not None and fusion_rank <= 10 else None
        if state != expected_state or scored_top10_rank != score["truth_fused_rank"]:
            raise ValueError(f"retrieval-state mismatch for {sid}")
        records = list(evidence_row["records"])
        if len(records) != 10 or [record["kegg_id"] for record in records] != fusion[key][:10]:
            raise ValueError(f"incomplete or mismatched frozen Top-10 evidence for {sid}")
        annotation = response.get("annotation") or {}
        abstain = bool(annotation.get("abstain"))
        prediction = None if abstain else annotation.get("predicted_kegg_id")
        grounded_exact, grounded_brite = _method_match(prediction, truth)
        if grounded_exact != bool(score["grounded_exact"]) or grounded_brite != bool(score["grounded_brite_orthology"]):
            raise ValueError(f"grounded score mismatch for {sid}")
        p2_prediction = phase2[key][0] if phase2[key] else None
        bm25_prediction = bm25[key][0]
        trained_prediction = trained[key][0]
        fusion_prediction = fusion[key][0]
        p2_exact, p2_brite = _method_match(p2_prediction, truth)
        bm25_exact, bm25_brite = _method_match(bm25_prediction, truth)
        trained_exact, trained_brite = _method_match(trained_prediction, truth)
        fusion_exact, fusion_brite = _method_match(fusion_prediction, truth)
        if fusion_exact != bool(score["fusion_top1_exact"]) or trained_prediction != score["trained_biencoder_top1_kegg_id"]:
            raise ValueError(f"frozen comparator mismatch for {sid}")
        phase3a_prediction = None if score["phase3a_target_only_abstain"] else score["phase3a_target_only_kegg_id"]
        catalog_status, catalog_json = _catalog_bundle(truth_ids, catalog)
        compliant = bool(score["grounded_evidence_compliant"])
        category = _mechanical_category(state, grounded_exact, abstain)
        row = {
            "audit_id": f"P3EA{number:04d}",
            "sample_id": sid,
            "model_id": sample["model_id"],
            "reaction_id": sample["reaction_id"],
            "cluster_id": sample["cluster_id"],
            "split": sample["split"],
            "corrected_phase2_stratum": str(score["corrected_stratum"]),
            "target_seen_in_train": _bool(score["target_seen_in_train"]),
            "normalized_reaction_equation": _normalized_equation(request),
            "ground_truth_kegg_ids": ";".join(truth_ids),
            "ground_truth_kegg_primary": answer["ground_truth_kegg_primary"],
            "ground_truth_id_count": str(len(truth_ids)),
            "ground_truth_catalog_presence_status": catalog_status,
            "ground_truth_catalog_records_json": catalog_json,
            "phase2_status": sample["status"],
            "phase2_candidate_set_size": sample["candidate_set_size"],
            "phase2_prediction": p2_prediction or "",
            "phase2_ground_truth_rank": "" if p2_rank is None else str(p2_rank),
            "phase2_exact": _bool(p2_exact),
            "phase2_brite": _bool(p2_brite),
            "bm25_top1": bm25_prediction,
            "bm25_ground_truth_rank": "" if bm25_rank is None else str(bm25_rank),
            "bm25_exact": _bool(bm25_exact),
            "bm25_brite": _bool(bm25_brite),
            "trained_epoch1_top1": trained_prediction,
            "trained_epoch1_ground_truth_rank": "" if trained_rank is None else str(trained_rank),
            "trained_epoch1_exact": _bool(trained_exact),
            "trained_epoch1_brite": _bool(trained_brite),
            "fusion_top1": fusion_prediction,
            "fusion_ground_truth_rank": "" if fusion_rank is None else str(fusion_rank),
            "fusion_exact": _bool(fusion_exact),
            "fusion_brite": _bool(fusion_brite),
            "retrieval_state": state,
            "phase3a_target_only_prediction": phase3a_prediction or "",
            "phase3a_target_only_abstain": _bool(score["phase3a_target_only_abstain"]),
            "phase3a_target_only_exact": _bool(score["phase3a_target_only_exact"]),
            "phase3a_target_only_brite": _bool(score["phase3a_target_only_brite_orthology"]),
            "phase3c_prediction": prediction or "",
            "phase3c_abstain": _bool(abstain),
            "phase3c_exact": _bool(grounded_exact),
            "phase3c_brite": _bool(grounded_brite),
            "phase3c_confidence": str(annotation.get("confidence", "")),
            "phase3c_selected_evidence_rank": str(annotation.get("selected_evidence_rank") or ""),
            "phase3c_reasoning_summary": str(annotation.get("reasoning_summary") or ""),
            "phase3c_abstention_reason": str(annotation.get("abstention_reason") or ""),
            "phase3c_supporting_evidence_ids": ";".join(annotation.get("supporting_evidence_ids") or []),
            "phase3c_evidence_compliant": _bool(compliant),
            "phase3c_compliance_problems_json": _canonical_json(response.get("compliance_problems") or []),
            "phase3c_unsupported_or_fabricated": _bool(score["grounded_unsupported"]),
            "grounded_vs_fusion_transition": _transition(grounded_exact, fusion_exact),
            "mechanical_outcome_category": category,
            "compliance_issue": _bool(not compliant),
            "review_flags": "compliance_issue" if not compliant else "",
            "fused_top10_evidence_ids": ";".join(record["kegg_id"] for record in records),
            "fused_top10_evidence_records_json": _canonical_json(records),
            "source_artifact_paths_json": source_paths_json,
            "source_artifact_digests_json": source_digests_json,
            **{field: "" for field in HUMAN_REVIEW_FIELDS},
            "disagreement_reasons": "",
        }
        row["disagreement_reasons"] = ";".join(_disagreement_reasons(row))
        rows.append(row)
    verify_accounting(rows)
    return rows


def verify_accounting(rows: Sequence[Mapping[str, str]]) -> dict[str, int]:
    counts = {
        "total": len(rows),
        "grounded_exact": sum(row["phase3c_exact"] == "true" for row in rows),
        "incorrect_selections": sum(row["phase3c_exact"] == "false" and row["phase3c_abstain"] == "false" for row in rows),
        "abstentions": sum(row["phase3c_abstain"] == "true" for row in rows),
        "noncorrect": sum(row["phase3c_exact"] == "false" for row in rows),
        "unsupported": sum(row["phase3c_unsupported_or_fabricated"] == "true" for row in rows),
        "compliance_cases": sum(row["phase3c_evidence_compliant"] == "false" for row in rows),
    }
    if counts != EXPECTED:
        raise ValueError(f"frozen Phase 3C accounting discrepancy: {counts} != {EXPECTED}")
    compliance = [row for row in rows if row["phase3c_evidence_compliant"] == "false"]
    if any(row["phase3c_abstain"] != "true" or "supporting evidence" not in row["phase3c_compliance_problems_json"] for row in compliance):
        raise ValueError("compliance cases are not the two expected cited abstentions")
    return counts


def _readme_text(summary: Mapping[str, Any]) -> str:
    return f"""# Phase 3 reaction-label error-audit inventories

This directory is an inventory-only milestone built deterministically from the frozen 163-reaction Phase 3C validation pilot. No held-out test data, new inference, API calls, live KEGG queries, or biological adjudication were used.

## Files

- `all_validation_outcomes.csv`: one provenance-rich row for every validation reaction.
- `phase3c_noncorrect_review.csv`: all {summary['noncorrect']} grounded outcomes that were not exactly correct; this is the main file to inspect next.
- `phase3c_incorrect_selections.csv`: the {summary['incorrect_selections']} non-abstained incorrect selections.
- `phase3c_abstentions.csv`: the {summary['abstentions']} abstentions, kept separate because an abstention is not automatically an error.
- `phase3c_compliance_cases.csv`: the {summary['compliance_cases']} mechanically detected evidence-compliance cases.
- `all_method_disagreements.csv`: {summary['method_disagreements']} error-enriched cases where frozen methods or correctness notions disagree.
- `source_provenance.json`, `inventory_summary.json`, and `deterministic_rebuild.json`: frozen inputs, accounting, and byte-rebuild evidence.

These inventories are review queues, not estimates of label-error prevalence. System agreement or disagreement is not biological evidence that a label is right or wrong, and all human-review fields are intentionally blank.

Prompt 2 will select the formal 60-case audit and construct Pass 1/Pass 2 review packets under a separately authorized, prespecified sampling and adjudication protocol. That work has not started here.
"""


def build_inventory_bundle(out: Path = OUT) -> dict[str, Any]:
    rows = build_rows()
    noncorrect = [row for row in rows if row["phase3c_exact"] == "false"]
    incorrect = [row for row in noncorrect if row["phase3c_abstain"] == "false"]
    abstentions = [row for row in noncorrect if row["phase3c_abstain"] == "true"]
    compliance = [row for row in rows if row["phase3c_evidence_compliant"] == "false"]
    disagreements = [row for row in rows if row["disagreement_reasons"]]
    outputs = {
        "all_validation_outcomes.csv": rows,
        "phase3c_noncorrect_review.csv": noncorrect,
        "phase3c_incorrect_selections.csv": incorrect,
        "phase3c_abstentions.csv": abstentions,
        "phase3c_compliance_cases.csv": compliance,
        "all_method_disagreements.csv": disagreements,
    }
    for name, selected in outputs.items():
        _write_csv(out / name, selected)
    summary = {
        "schema": "phase3-error-audit-inventory-summary-v1",
        **verify_accounting(rows),
        "method_disagreements": len(disagreements),
        "validation_only": True,
        "test_rows_loaded": 0,
        "test_labels_loaded": 0,
        "api_calls": 0,
        "new_inference": False,
        "biological_adjudication": False,
        "labels_modified": False,
        "formal_60_case_sample_built": False,
        "inventory_files": sorted(outputs),
    }
    atomic_write_json(summary, out / "inventory_summary.json")
    provenance = {
        "schema": "phase3-error-audit-source-provenance-v1",
        "source_phase3c_commit": SOURCE_COMMIT,
        "source_artifacts": [{"path": path, "sha256": digest} for path, digest in SOURCE_DIGESTS.items()],
        "hash_function": "sha256_portable (LF-normalized for text)",
        "split": "validation",
        "test_rows_loaded": 0,
        "test_labels_loaded": 0,
        "live_sources": [],
        "api_calls": 0,
    }
    atomic_write_json(provenance, out / "source_provenance.json")
    readme = out / "README.md"
    readme.write_text(_readme_text(summary), encoding="utf-8", newline="\n")
    return summary


def build_twice(out: Path = OUT) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_error_inventory_a_", dir=PHASE3_DIR) as first_name, tempfile.TemporaryDirectory(prefix="_error_inventory_b_", dir=PHASE3_DIR) as second_name:
        first, second = Path(first_name), Path(second_name)
        build_inventory_bundle(first)
        build_inventory_bundle(second)
        first_files = {path.name: path.read_bytes() for path in first.iterdir() if path.is_file()}
        second_files = {path.name: path.read_bytes() for path in second.iterdir() if path.is_file()}
        names = sorted(set(first_files) | set(second_files))
        comparisons = [
            {
                "path": name,
                "byte_equal": first_files.get(name) == second_files.get(name),
                "sha256": hashlib.sha256(first_files[name]).hexdigest() if name in first_files else None,
            }
            for name in names
        ]
        all_equal = set(first_files) == set(second_files) and all(item["byte_equal"] for item in comparisons)
    if not all_equal:
        raise ValueError("inventory rebuilds are not byte-identical")
    summary = build_inventory_bundle(out)
    atomic_write_json({
        "schema": "phase3-error-audit-deterministic-rebuild-v1",
        "builds": 2,
        "all_byte_equal": True,
        "files": comparisons,
        "command": "python -m benchmark.scripts.phase3_error_inventory",
    }, out / "deterministic_rebuild.json")
    write_manifest(out)
    return summary


def write_manifest(out: Path = OUT) -> None:
    artifacts = [path for path in out.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    write_artifact_manifest(out, artifacts)


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
    if manifest_path.read_bytes() != before:
        problems.append("read-only verification mutated manifest")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="verify the committed inventory manifest without writing")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.verify:
        problems = verify_manifest(args.out)
        print(json.dumps({"n_problems": len(problems), "problems": problems}, sort_keys=True))
        return int(bool(problems))
    summary = build_twice(args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
