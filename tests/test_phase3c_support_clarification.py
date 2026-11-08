"""Correctness and task-specific evidence compliance are separate endpoints."""
import json

from benchmark.scripts import phase3c_support_clarification as clarification


def row(**changes):
    value = {"sample_id": "case-1", "phase3a_unsupported_in_catalog": True,
             "phase3a_target_only_exact": True, "grounded_answered": True,
             "grounded_abstain": False, "grounded_unsupported": False,
             "grounded_exact": False, "grounded_evidence_compliant": True,
             "fusion_top1_exact": False}
    return {**value, **changes}


def test_correct_direct_answer_is_not_a_biological_error():
    result = clarification.support_counts([row()])
    assert result["direct_in_catalog_without_retrieved_evidence"] == 1
    assert result["direct_reference_correct_without_retrieved_evidence"] == 1
    assert result["direct_reference_incorrect_without_retrieved_evidence"] == 0
    assert not result["support_outcomes_have_same_definition"]
    assert not result["support_interval_establishes_reduced_biological_hallucination"]


def test_zero_outside_evidence_does_not_mean_zero_reference_errors():
    rows = [row(), row(sample_id="case-2", grounded_answered=False, grounded_abstain=True,
                       grounded_evidence_compliant=False)]
    result = clarification.support_counts(rows)
    assert result["grounded_out_of_supplied_evidence"] == 0
    assert result["grounded_in_evidence_reference_incorrect"] == 1
    assert result["grounded_full_evidence_compliant"] == 1
    assert result["grounded_abstained"] == 1


def test_reference_correctness_does_not_imply_in_evidence_compliance():
    result = clarification.support_counts([row(grounded_unsupported=True, grounded_exact=True,
                                                grounded_evidence_compliant=False)])
    assert result["grounded_exact_correct_all"] == 1
    assert result["grounded_out_of_supplied_evidence"] == 1
    assert result["grounded_in_evidence_reference_incorrect"] == 0


def test_frozen_validation_counts_preserve_accuracy_result():
    source = clarification.ROOT / "benchmark/phase3/phase3c_validation/answer_key_join.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = clarification.support_counts(rows)
    assert result["n_validation"] == 163
    assert result["direct_in_catalog_without_retrieved_evidence"] == 130
    assert result["direct_reference_correct_without_retrieved_evidence"] == 49
    assert result["direct_reference_incorrect_without_retrieved_evidence"] == 81
    assert result["grounded_exact_correct_all"] == 102
    assert result["fusion_exact_correct_all"] == 93
    assert result["grounded_in_evidence_reference_incorrect"] == 22
    assert result["grounded_full_evidence_compliant"] == 161
