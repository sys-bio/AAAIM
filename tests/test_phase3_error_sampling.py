"""Focused invariants for deterministic formal error-audit sampling."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmark.scripts import phase3_error_sampling as sampling


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path: Path) -> list[dict[str, str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture(scope="module")
def frozen_rows():
    return _csv(sampling.INVENTORY)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    out = tmp_path_factory.mktemp("formal-audit-sampling")
    result = sampling.build_sampling_bundle(out)
    return out, result


def test_exact_60_unique_validation_reactions_and_category_quotas(bundle):
    out, _ = bundle
    rows = _csv(out / "formal_audit_sample.csv")
    assert len(rows) == 60
    assert len({(row["model_id"], row["reaction_id"]) for row in rows}) == 60
    assert {row["split"] for row in rows} == {"validation"}
    counts = {name: sum(row["primary_category"] == name for row in rows) for name in sampling.CATEGORY_QUOTAS}
    assert counts == sampling.CATEGORY_QUOTAS


def test_stable_prompt1_audit_ids_are_reused(bundle, frozen_rows):
    out, _ = bundle
    expected = {(row["model_id"], row["reaction_id"]): row["audit_id"] for row in frozen_rows}
    assert [row["audit_id"] for row in frozen_rows] == [f"P3EA{number:04d}" for number in range(1, 164)]
    for row in _csv(out / "formal_audit_sample.csv"):
        assert row["audit_id"] == expected[(row["model_id"], row["reaction_id"])]


def test_random_controls_are_outcome_independent_and_deterministic(frozen_rows):
    first = [row["audit_id"] for row in sampling.select_random_controls(frozen_rows)]
    changed = []
    for row in frozen_rows:
        copy = dict(row)
        for field in copy:
            if field not in {"audit_id", "sample_id", "model_id", "reaction_id", "cluster_id", "split"}:
                copy[field] = "deliberately-perturbed"
        changed.append(copy)
    second = [row["audit_id"] for row in sampling.select_random_controls(changed)]
    assert first == second
    assert first == [row["audit_id"] for row in sampling.select_random_controls(frozen_rows, 20260910)]
    assert first != [row["audit_id"] for row in sampling.select_random_controls(frozen_rows, 20260911)]
    by_audit = {row["audit_id"]: row for row in frozen_rows}
    cluster_counts = {
        cluster: sum(by_audit[audit_id]["cluster_id"] == cluster for audit_id in first)
        for cluster in {row["cluster_id"] for row in frozen_rows}
    }
    assert set(cluster_counts.values()) <= {1, 2}


def test_cluster_round_robin_and_tie_breaking_are_deterministic():
    rows = [
        {"audit_id": f"A{number}", "model_id": f"M{number % 2}", "reaction_id": f"R{number}", "cluster_id": cluster, "split": "validation", "sample_id": f"S{number}"}
        for number, cluster in enumerate(["C1", "C1", "C1", "C2", "C2", "C2", "C3", "C3", "C3"], 1)
    ]
    kwargs = dict(seed=17, namespace="unit", selected=[], category_selected=[])
    first = sampling.cluster_round_robin(rows, 6, **kwargs)
    second = sampling.cluster_round_robin(list(reversed(rows)), 6, **kwargs)
    assert [row["audit_id"] for row in first] == [row["audit_id"] for row in second]
    counts = {cluster: sum(row["cluster_id"] == cluster for row in first) for cluster in {"C1", "C2", "C3"}}
    assert counts == {"C1": 2, "C2": 2, "C3": 2}


def test_failure_priority_deduplication_refill_and_secondary_reasons(bundle, frozen_rows):
    _, result = bundle
    selected = result["selected"]
    assert len({item["row"]["audit_id"] for item in selected}) == 60
    failure_priorities = [int(item["selection_priority"]) for item in selected if item["primary_category"] == sampling.FAILURE]
    assert failure_priorities == sorted(failure_priorities)
    assert all(isinstance(item["secondary_eligibility_reasons"], list) for item in selected)
    eligibility = sampling.build_eligibility(frozen_rows)
    for item in selected:
        reasons = item["secondary_eligibility_reasons"]
        assert len(reasons) == len(set(reasons))
        meta = eligibility[item["row"]["audit_id"]]
        assert all(f"failure_type:{value}" in reasons for value in meta["failure_types"])
        assert all(f"disagreement:{value}" in reasons for value in meta["disagreement_types"])
        assert all(f"suspicion:{value}" in reasons for value in meta["suspicion_rules"])
        assert all(f"documented_edge_case:{value}" in reasons for value in meta["documented_edges"])


def test_valid_malformed_ambiguous_and_test_nomination_handling_has_no_label_exposure():
    outcomes = [
        {"model_id": "M1", "reaction_id": "R1", "audit_id": "A1", "split": "validation"},
        {"model_id": "M2", "reaction_id": "R2", "audit_id": "A2", "split": "validation"},
        {"model_id": "M2", "reaction_id": "R2", "audit_id": "A3", "split": "validation"},
    ]
    nominations = [
        {"model_id": "M1", "reaction_id": "R1", "nomination_reason": "valid reason", "prior_source": "source"},
        {"model_id": "model_id", "reaction_id": "reaction_id", "nomination_reason": "nomination_reason", "prior_source": "prior_source"},
        {"model_id": "M2", "reaction_id": "R2", "nomination_reason": "ambiguous", "prior_source": "source"},
        {"model_id": "MT", "reaction_id": "RT", "nomination_reason": "test", "prior_source": "source"},
    ]
    accepted, diagnostics = sampling.match_nominations(
        nominations, outcomes, split_lookup=lambda keys: {("MT", "RT"): ["test"]}
    )
    assert len(accepted) == 1
    assert accepted[0]["nomination_reason"] == "valid reason"
    assert {row["status"] for row in diagnostics} == {
        "unresolved_malformed", "unresolved_ambiguous", "rejected_held_out_test"
    }
    assert all("label" not in key for row in diagnostics for key in row)
    test_row = next(row for row in diagnostics if row["status"] == "rejected_held_out_test")
    assert test_row["diagnostic"] == "held-out nomination rejected; no test label read or recorded"


def test_supplied_nominations_are_preserved_and_fewer_than_five_are_filled(bundle):
    out, result = bundle
    assert result["accepted"] == []
    diagnostics = _csv(out / "manual_nomination_diagnostics.csv")
    assert len(diagnostics) == 2
    unresolved = next(row for row in diagnostics if row["model_id"] == "BIOMD0000000013")
    assert unresolved["nomination_reason"] == "R01429 contains xylonolactone which is not reflected in the reaction E12"
    assert unresolved["prior_source"] == "prior manual observation"
    assert unresolved["status"] == "unresolved_not_validation"
    manual = [item for item in result["selected"] if item["primary_category"] == sampling.MANUAL_EDGE]
    assert len(manual) == 5
    assert {item["primary_detail"] for item in manual} == {"documented_edge_case_fill"}
    assert _csv(out / "manual_nominations.csv") == []


def test_nomination_already_random_retains_random_primary_and_refills(frozen_rows):
    random_audit = sampling.select_random_controls(frozen_rows)[0]["audit_id"]
    row = next(row for row in frozen_rows if row["audit_id"] == random_audit)
    accepted = [{
        "model_id": row["model_id"], "reaction_id": row["reaction_id"], "nomination_reason": "unit",
        "prior_source": "unit", "audit_id": row["audit_id"], "validation_status": "confirmed_validation",
        "matching_status": "unique_exact_match",
    }]
    eligibility = sampling.build_eligibility(frozen_rows)
    selected = sampling.select_sample(frozen_rows, eligibility, accepted)
    item = next(item for item in selected if item["row"]["audit_id"] == random_audit)
    assert item["primary_category"] == sampling.RANDOM
    assert "manual_nomination" in item["secondary_eligibility_reasons"]
    assert sum(value["primary_category"] == sampling.MANUAL_EDGE for value in selected) == 5


def test_multilabel_parsing_is_used_for_mechanical_eligibility(frozen_rows):
    multi = next(row for row in frozen_rows if int(row["ground_truth_id_count"]) > 1)
    ids = sampling.parse_multilabel_kegg_ids(multi["ground_truth_kegg_ids"])
    assert len(ids) == int(multi["ground_truth_id_count"])
    assert sampling.SUSPICION_RULES[0] in sampling.suspicion_rules(multi, False)


def test_csv_jsonl_crosswalk_and_blinded_artifacts_align_exactly(bundle):
    out, _ = bundle
    formal = _csv(out / "formal_audit_sample.csv")
    assert formal == _jsonl(out / "formal_audit_sample.jsonl")
    crosswalk = _csv(out / "formal_audit_crosswalk.csv")
    blinded = _csv(out / "formal_audit_blinded_order.csv")
    keyset = lambda rows: {(row["audit_id"], row["model_id"], row["reaction_id"]) for row in rows}
    assert keyset(formal) == keyset(crosswalk) == keyset(blinded)


def test_blinded_skeleton_has_allowlist_only_and_distinct_deterministic_order(bundle):
    out, _ = bundle
    blinded = _csv(out / "formal_audit_blinded_order.csv")
    assert list(blinded[0]) == sampling.BLINDED_FIELDS
    forbidden_fragments = ("category", "suspicion", "correct", "outcome", "failure", "priority", "random")
    assert not any(fragment in field for field in blinded[0] for fragment in forbidden_fragments)
    formal_order = [row["audit_id"] for row in _csv(out / "formal_audit_sample.csv")]
    blind_order = [row["audit_id"] for row in blinded]
    selection_order = [item["row"]["audit_id"] for item in sorted(bundle[1]["selected"], key=lambda item: item["global_selection_order"])]
    assert blind_order != formal_order
    assert blind_order != selection_order
    assert sampling.BLINDED_SEED != sampling.SAMPLING_SEED
    assert blind_order == [row["audit_id"] for row in sampling._blinded_rows(bundle[1]["selected"])]


def test_byte_identical_rebuilds(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    sampling.build_sampling_bundle(first)
    sampling.build_sampling_bundle(second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }


def test_manifest_verification_is_read_only(tmp_path):
    out = tmp_path / "manifested"
    sampling.build_twice(out)
    manifest = out / "artifact_manifest.json"
    before = manifest.read_bytes()
    assert sampling.verify_manifest(out) == []
    assert manifest.read_bytes() == before


def test_prompt1_and_earlier_frozen_artifacts_are_unchanged():
    sampling.verify_frozen_sources()


def test_no_review_packets_or_biological_judgments_created(bundle):
    out, _ = bundle
    assert not any("pass1" in path.name.lower() or "pass2" in path.name.lower() for path in out.iterdir())
    config = json.loads((out / "sampling_config.json").read_text(encoding="utf-8"))
    assert config["biological_judgments_made"] is False
    assert config["test_rows_loaded"] == config["test_labels_loaded"] == 0
    assert config["api_calls"] == 0
    assert config["new_inference"] is False
