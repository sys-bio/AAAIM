"""Focused invariants for the validation-only reaction error inventories."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmark.scripts import phase3_error_inventory as inventory


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("error-audit")
    summary = inventory.build_inventory_bundle(out)
    rows = _csv_rows(out / "all_validation_outcomes.csv")
    return out, summary, rows


def test_exact_frozen_accounting_and_unique_validation_keys(bundle):
    _, summary, rows = bundle
    assert len(rows) == summary["total"] == 163
    assert len({row["sample_id"] for row in rows}) == 163
    assert len({(row["model_id"], row["reaction_id"]) for row in rows}) == 163
    assert {row["split"] for row in rows} == {"validation"}
    assert summary["grounded_exact"] == 102
    assert summary["incorrect_selections"] == 22
    assert summary["abstentions"] == 39
    assert summary["noncorrect"] == 61
    assert summary["unsupported"] == 0
    assert summary["compliance_cases"] == 2


def test_noncorrrect_subsets_are_complete_and_disjoint(bundle):
    out, _, _ = bundle
    noncorrect = _csv_rows(out / "phase3c_noncorrect_review.csv")
    incorrect = _csv_rows(out / "phase3c_incorrect_selections.csv")
    abstentions = _csv_rows(out / "phase3c_abstentions.csv")
    assert len(noncorrect) == 61
    assert len(incorrect) == 22
    assert len(abstentions) == 39
    assert {row["sample_id"] for row in incorrect}.isdisjoint({row["sample_id"] for row in abstentions})
    assert {row["sample_id"] for row in noncorrect} == {
        row["sample_id"] for row in incorrect + abstentions
    }
    assert {row["phase3c_abstain"] for row in incorrect} == {"false"}
    assert {row["phase3c_abstain"] for row in abstentions} == {"true"}


def test_two_compliance_cases_are_cited_abstentions(bundle):
    out, _, _ = bundle
    cases = _csv_rows(out / "phase3c_compliance_cases.csv")
    assert len(cases) == 2
    assert {row["phase3c_abstain"] for row in cases} == {"true"}
    assert all("supporting evidence" in row["phase3c_compliance_problems_json"] for row in cases)
    assert all(row["phase3c_supporting_evidence_ids"] for row in cases)


def test_multilabel_parser_accepts_semicolon_and_legacy_pipe_without_loss(bundle):
    assert inventory.parse_multilabel_kegg_ids("R00001;R00002") == ["R00001", "R00002"]
    assert inventory.parse_multilabel_kegg_ids("R00001|R00002") == ["R00001", "R00002"]
    assert inventory.parse_multilabel_kegg_ids("R00001; R00002|R00003") == ["R00001", "R00002", "R00003"]
    with pytest.raises(ValueError, match="malformed"):
        inventory.parse_multilabel_kegg_ids("R00001|not-an-id")
    _, _, rows = bundle
    assert all("|" not in row["ground_truth_kegg_ids"] for row in rows)
    assert all(len(inventory.parse_multilabel_kegg_ids(row["ground_truth_kegg_ids"])) == int(row["ground_truth_id_count"]) for row in rows)


def test_stable_audit_ids_and_retrieval_states(bundle):
    _, _, rows = bundle
    assert [row["audit_id"] for row in rows] == [f"P3EA{number:04d}" for number in range(1, 164)]
    assert {row["retrieval_state"] for row in rows} == {
        "truth_at_fusion_rank1",
        "truth_at_fusion_ranks2_10",
        "truth_absent_from_fusion_top10",
    }


def test_review_cases_have_complete_frozen_top10_and_blank_human_fields(bundle):
    out, _, _ = bundle
    rows = _csv_rows(out / "phase3c_noncorrect_review.csv")
    for row in rows:
        evidence_ids = row["fused_top10_evidence_ids"].split(";")
        records = json.loads(row["fused_top10_evidence_records_json"])
        assert len(evidence_ids) == len(records) == 10
        assert evidence_ids == [record["kegg_id"] for record in records]
        assert [record["fused_rank"] for record in records] == list(range(1, 11))
        assert all(row[field] == "" for field in inventory.HUMAN_REVIEW_FIELDS)


def test_catalog_membership_representation_is_explicit(bundle):
    _, _, rows = bundle
    valid_statuses = {"all_in_catalog", "partially_in_catalog", "absent_from_catalog"}
    for row in rows:
        assert row["ground_truth_catalog_presence_status"] in valid_statuses
        payload = json.loads(row["ground_truth_catalog_records_json"])
        assert set(payload) == {"membership", "records"}
        assert set(payload["membership"]) == set(inventory.parse_multilabel_kegg_ids(row["ground_truth_kegg_ids"]))
        assert set(payload["membership"].values()) <= {"in_catalog", "absent_from_catalog"}


def test_source_provenance_is_complete_and_test_sealed(bundle):
    _, summary, rows = bundle
    expected_paths = set(inventory.SOURCE_DIGESTS)
    first_paths = set(json.loads(rows[0]["source_artifact_paths_json"]))
    first_digests = json.loads(rows[0]["source_artifact_digests_json"])
    assert first_paths == expected_paths == set(first_digests)
    assert all(row["source_artifact_paths_json"] == rows[0]["source_artifact_paths_json"] for row in rows)
    assert all(row["source_artifact_digests_json"] == rows[0]["source_artifact_digests_json"] for row in rows)
    assert summary["validation_only"] is True
    assert summary["test_rows_loaded"] == summary["test_labels_loaded"] == 0
    assert summary["api_calls"] == 0
    assert summary["new_inference"] is False


def test_frozen_source_hashes_are_unchanged():
    inventory.verify_sources()


def test_method_disagreement_inventory_matches_summary(bundle):
    out, summary, rows = bundle
    disagreements = _csv_rows(out / "all_method_disagreements.csv")
    assert len(disagreements) == summary["method_disagreements"]
    assert {row["sample_id"] for row in disagreements} == {
        row["sample_id"] for row in rows if row["disagreement_reasons"]
    }
    assert all(row["disagreement_reasons"] for row in disagreements)


def test_mechanical_categories_separate_abstentions_from_selections(bundle):
    _, _, rows = bundle
    for row in rows:
        category = row["mechanical_outcome_category"]
        if row["phase3c_exact"] == "true":
            assert category == "grounded_exact"
        elif row["phase3c_abstain"] == "true":
            assert category.startswith("abstention_")
        else:
            assert category.startswith("incorrect_")


def test_bundle_rebuild_is_byte_deterministic(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    inventory.build_inventory_bundle(first)
    inventory.build_inventory_bundle(second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_manifest_verification_is_read_only(tmp_path):
    out = tmp_path / "manifested"
    inventory.build_twice(out)
    manifest = out / "artifact_manifest.json"
    before = manifest.read_bytes()
    assert inventory.verify_manifest(out) == []
    assert manifest.read_bytes() == before
