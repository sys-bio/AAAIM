"""Tests for Phase 3A validation offline evaluation helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.phase3_common import (
    ORIGINAL_VALIDATION_RESULTS_SHA256,
    OUT_VALIDATION_DIR,
    STRATUM_EMPTY,
    STRATUM_RERANK,
    STRATUM_TOP1,
    TRUE_RETRIEVAL_FAILURE_STRATA,
    sha256_portable,
)
from benchmark.scripts.phase3_openai_eval import (
    build_manifest,
    compose_rescue_sensitivity,
    recovery_block,
    variant_metrics,
    verify_manifest,
)


def _row(**overrides):
    base = {
        "sample_id": "P1",
        "model_id": "M1",
        "reaction_id": "rxn",
        "cluster_id": "C1",
        "stratum": STRATUM_EMPTY,
        "variant": "target_only",
        "abstain": False,
        "answered": True,
        "exact_top1": False,
        "exact_top3": False,
        "brite_top1": False,
        "brite_top3": False,
        "parse_error": None,
        "terminal_status": "succeeded",
        "n_malformed_ids": 0,
        "n_absent_from_catalog_ids": 0,
        "n_in_catalog_ids": 1,
        "open_set_outcome": "incorrect_in_catalog",
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


def test_recovery_rate_excludes_abstention_and_uses_explicit_denominator():
    rows = [
        _row(sample_id="a", exact_top1=True, open_set_outcome="correct_top1"),
        _row(sample_id="b", abstain=True, answered=False, exact_top1=False,
             n_in_catalog_ids=0, open_set_outcome="abstain"),
        _row(sample_id="c", exact_top1=False),
        _row(sample_id="d", stratum=STRATUM_TOP1, exact_top1=True,
             open_set_outcome="correct_top1"),
        _row(sample_id="e", stratum=STRATUM_RERANK, exact_top1=True,
             open_set_outcome="correct_top1"),
    ]
    rec = recovery_block(rows)
    assert rec["true_retrieval_failure"]["numerator"] == 1
    assert rec["true_retrieval_failure"]["denominator"] == 3
    assert rec["all_retrieval_failure"]["denominator"] == 3
    assert rec["retrievable_rerank_failure"]["numerator"] == 1
    assert rec["retrievable_rerank_failure"]["denominator"] == 1
    assert rec["true_retrieval_failure"]["recovery_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert rec["empty_constrained"]["numerator"] == 1
    assert rec["all_non_top1"]["denominator"] == 4
    metrics = variant_metrics(rows)
    assert metrics["n_incorrect_in_catalog"] == 1
    assert metrics["n_reactions"] == 5
    assert STRATUM_RERANK not in TRUE_RETRIEVAL_FAILURE_STRATA
    assert set(TRUE_RETRIEVAL_FAILURE_STRATA) == {
        "unconstrained", "empty_constrained", "nonempty_answer_absent",
    }


def test_rerank_failure_cannot_enter_true_retrieval_failure_denominator():
    rows = [
        _row(sample_id="e", stratum=STRATUM_RERANK, exact_top1=True,
             open_set_outcome="correct_top1"),
    ]
    rec = recovery_block(rows)
    assert rec["true_retrieval_failure"]["denominator"] == 0
    assert rec["true_retrieval_failure"]["numerator"] == 0
    assert rec["retrievable_rerank_failure"]["numerator"] == 1


def test_recovery_block_raises_if_rerank_is_classified_as_retrieval_failure(monkeypatch):
    import benchmark.scripts.phase3_openai_eval as ev
    monkeypatch.setattr(
        ev, "TRUE_RETRIEVAL_FAILURE_STRATA",
        ("unconstrained", "empty_constrained", STRATUM_RERANK),
    )
    with pytest.raises(AssertionError, match="must not be a true retrieval-failure"):
        ev.recovery_block([_row(stratum=STRATUM_RERANK)])


def test_schema_invalid_is_not_incorrect_in_catalog():
    rows = [
        _row(sample_id="s", terminal_status="schema_invalid", parse_error="unparseable",
             answered=False, n_in_catalog_ids=0, open_set_outcome="schema_invalid",
             exact_top1=False),
        _row(sample_id="c", exact_top1=True, open_set_outcome="correct_top1"),
    ]
    metrics = variant_metrics(rows)
    assert metrics["n_schema_invalid"] == 1
    assert metrics["n_incorrect_in_catalog"] == 0
    assert metrics["exact_top1_d"] == 2
    assert metrics["schema_invalid_counted_as_unsuccessful"] is True


def test_compose_rescue_sensitivity_substitutes_only_original_invalid_keys():
    original = [
        _row(sample_id="ok", terminal_status="succeeded", exact_top1=True,
             open_set_outcome="correct_top1"),
        _row(sample_id="bad", terminal_status="schema_invalid", exact_top1=False,
             answered=False, open_set_outcome="schema_invalid",
             api_error="LengthFinishReasonError"),
    ]
    rescue = [
        _row(sample_id="bad", terminal_status="succeeded", exact_top1=True,
             open_set_outcome="correct_top1"),
        _row(sample_id="ok", terminal_status="succeeded", exact_top1=False,
             open_set_outcome="incorrect_in_catalog"),
    ]
    composed, stats = compose_rescue_sensitivity(original, rescue)
    by_id = {row["sample_id"]: row for row in composed}
    assert by_id["ok"]["exact_top1"] is True
    assert by_id["bad"]["sensitivity_source"] == "rescue_2048"
    assert stats["n_rescued_successfully"] == 1
    assert stats["n_still_schema_invalid"] == 0
    assert stats["n_original_length_truncated"] == 1


def test_build_manifest_unique_posix_paths(tmp_path):
    a = tmp_path / "a.json"
    a.write_text("{}\n", encoding="utf-8")
    manifest = build_manifest(tmp_path, [a, a], root=tmp_path)
    assert manifest["n_files"] == 1
    assert manifest["files"][0]["path"] == "a.json"
    assert "\\" not in manifest["files"][0]["path"]


def test_manifest_verify_fails_on_corrupt_copy_and_leaves_original_results(tmp_path):
    original = OUT_VALIDATION_DIR / "results.jsonl"
    before = sha256_portable(original)
    assert before == ORIGINAL_VALIDATION_RESULTS_SHA256
    copied = tmp_path / "eval.json"
    copied.write_text('{"ok": true}\n', encoding="utf-8")
    manifest = build_manifest(tmp_path, [copied], root=tmp_path)
    man_path = tmp_path / "artifact_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert verify_manifest(man_path, root=tmp_path) == []
    copied.write_text('{"ok": false}\n', encoding="utf-8")
    problems = verify_manifest(man_path, root=tmp_path)
    assert problems, "corrupt copy must fail verification"
    assert sha256_portable(original) == before
    assert sha256_portable(original) == ORIGINAL_VALIDATION_RESULTS_SHA256
    assert original.exists()
