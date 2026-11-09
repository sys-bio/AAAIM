"""Synthetic-only tests for personal-export analysis; no biological judgments."""
import copy

import pytest

from benchmark.scripts import phase3_error_analysis as analysis
from benchmark.scripts import phase3_error_reviewer as reviewer


def fixture(phase):
    bundle = {"dataset_id": f"synthetic-pass{phase}", "dataset_digest": "synthetic",
              "cases": [{"audit_id": f"SYN{i:03}"} for i in range(60)]}
    schema = reviewer.REVIEW_SCHEMA[f"pass{phase}"]
    rows = []
    for case in bundle["cases"]:
        row = {field: "" for field in schema["fields"]}
        row.update(case, complete=True, biological_rationale="Synthetic missing-evidence rationale.")
        if phase == 1:
            row.update(verdict="unresolved", needs_second_review=False)
        else:
            row.update(prediction_assessment="unresolved", needs_adjudication=False, failure_taxonomy=[])
        rows.append(row)
    payload = {"dataset_id": bundle["dataset_id"], "dataset_digest": bundle["dataset_digest"],
               "review_schema_version": reviewer.REVIEW_SCHEMA_VERSION, "review_kind": f"formal_pass{phase}",
               "reviews": rows, "completion_count": 60, "complete": True}
    return sign(payload), bundle


def sign(payload):
    payload["canonical_review_digest"] = reviewer._digest({
        k: v for k, v in payload.items() if k not in {"canonical_review_digest", "export_timestamp"}})
    return payload


@pytest.mark.parametrize("phase", [1, 2])
def test_complete_unresolved_and_blank_optional_fields_are_valid(phase):
    payload, bundle = fixture(phase)
    assert analysis.validate(payload, bundle, phase) == []


@pytest.mark.parametrize("phase", [1, 2])
@pytest.mark.parametrize("mutation", ["duplicate", "foreign", "digest", "required", "boolean", "enum", "missing"])
def test_reject_invalid_exports(phase, mutation):
    payload, bundle = fixture(phase)
    row = payload["reviews"][0]
    if mutation == "duplicate":
        row["audit_id"] = payload["reviews"][1]["audit_id"]
    elif mutation == "foreign":
        row["audit_id"] = "SUP-SYNTHETIC"
    elif mutation == "required":
        row["biological_rationale"] = " "
    elif mutation == "boolean":
        row["complete"] = "true"
    elif mutation == "enum":
        row["confidence"] = "invalid"
    elif mutation == "missing":
        del row["confidence"]
    sign(payload)
    if mutation == "digest":
        payload["canonical_review_digest"] = "bad"
    assert analysis.validate(payload, bundle, phase)


def test_multilabel_counts_are_case_counts_separate_from_exclusive_outcomes():
    row = {"verdict": "unresolved", "prediction_assessment": "unresolved", "kegg_mappable": "",
           "exact_matching_too_strict": "", "failure_taxonomy": ["retrieval_failure", "llm_selection_failure"],
           "needs_second_review": False, "needs_adjudication": False}
    other = copy.deepcopy(row)
    other["failure_taxonomy"] = []
    result = analysis.aggregate([row, other])
    assert result["n"] == 2
    assert result["prediction_assessment"] == {"unresolved": 2}
    assert result["failure_taxonomy_multilabel"] == {"retrieval_failure": 1, "llm_selection_failure": 1}
    assert result["taxonomy_blank_cases"] == 1
    assert "unresolved biological judgment" in analysis.reasons(row)


def test_no_followup_inferred_from_blank_taxonomy():
    row = {"verdict": "label_correct", "prediction_assessment": "prediction_correct",
           "failure_taxonomy": [], "exact_matching_too_strict": "",
           "needs_second_review": False, "needs_adjudication": False}
    assert analysis.reasons(row) == []


def test_abstained_output_with_prediction_assessment_needs_target_clarification():
    row = {"verdict": "label_correct", "prediction_assessment": "prediction_correct",
           "failure_taxonomy": [], "exact_matching_too_strict": "", "phase3c_abstain": "true",
           "needs_second_review": False, "needs_adjudication": False}
    assert len(analysis.reasons(row)) == 1
    assert "clarify assessment target" in analysis.reasons(row)[0]


def test_revision_comparison_reports_all_changed_fields_and_other_cases():
    old = [{"audit_id": "SYN-A", "assessment": "prediction_correct", "tags": []},
           {"audit_id": "SYN-B", "assessment": "unresolved", "tags": []}]
    new = copy.deepcopy(old)
    new[0].update(assessment="abstention_inappropriate", tags=["llm_overabstention"])
    new[1]["assessment"] = "insufficient_information"
    changes = analysis.compare_reviews(old, new[::-1])
    assert len(changes) == 3
    assert changes[0] == {"audit_id": "SYN-A", "field": "assessment",
                          "before": "prediction_correct", "after": "abstention_inappropriate"}
    assert changes[-1]["audit_id"] == "SYN-B"
    assert analysis.compare_reviews(old, old[::-1]) == []


def test_revision_comparison_rejects_duplicate_or_different_case_sets():
    rows = [{"audit_id": "SYN-A"}, {"audit_id": "SYN-B"}]
    with pytest.raises(ValueError):
        analysis.compare_reviews(rows, [rows[0], rows[0]])
    with pytest.raises(ValueError):
        analysis.compare_reviews(rows, [{"audit_id": "SYN-C"}, rows[1]])
