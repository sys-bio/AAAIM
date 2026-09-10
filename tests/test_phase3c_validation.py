"""Offline integrity tests for the 163-row Phase 3C paired validation pilot."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.scripts import phase3c_validation as validation
from benchmark.scripts import phase3c_grounded as grounded
from benchmark.scripts.phase3_modes import FileCache


@pytest.fixture(scope="module")
def index():
    return grounded.EvidenceIndex()


@pytest.fixture(scope="module")
def cold_plan(index, tmp_path_factory):
    return validation.build_plan(index=index, validation_cache=tmp_path_factory.mktemp("p3c-validation-cold"))


def _annotation(item):
    first = item["evidence"][0]
    return grounded.GroundedAnnotation(
        abstain=False,
        predicted_kegg_id=first.kegg_id,
        selected_evidence_rank=1,
        supporting_evidence_ids=[first.evidence_id],
        confidence=0.8,
        reasoning_summary="Mocked evidence selection.",
        abstention_reason=None,
    )


def _response(item):
    usage = SimpleNamespace(
        input_tokens=500,
        output_tokens=100,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens_details=SimpleNamespace(reasoning_tokens=20),
    )
    return SimpleNamespace(
        output_parsed=_annotation(item), usage=usage,
        model=grounded.MODEL, id="resp_mock",
    )


def _metric_row(**updates):
    row = {
        "sample_id": "s", "cluster_id": "c", "retrieval_state": "C_truth_absent_top10",
        "grounded_answered": False, "grounded_exact": False,
        "grounded_brite_orthology": False, "grounded_abstain": True,
        "grounded_evidence_compliant": True, "grounded_incorrect_in_evidence": False,
        "grounded_unsupported": False, "grounded_schema_invalid": False,
        "fusion_top1_exact": False, "phase3a_target_only_exact": False,
        "phase3a_target_only_brite_orthology": False,
        "phase3a_unsupported_in_catalog": False,
        "phase3a_incorrect_unsupported_in_catalog": False,
    }
    row.update(updates)
    return row


def test_population_is_exact_frozen_163_validation_reactions(cold_plan):
    population = cold_plan["population"]
    assert len(population) == 163
    assert len({row["sample_id"] for row in population}) == 163
    assert len({(row["model_id"], row["reaction_id"]) for row in population}) == 163
    assert {row["split"] for row in population} == {"validation"}
    assert grounded.sha256_portable(validation.PILOT_SAMPLE) == validation.EXPECTED_SAMPLE_SHA256


def test_all_five_compatible_smoke_entries_are_reused(cold_plan):
    assert len(cold_plan["cache_hits"]) == 5
    assert {row["source"] for row in cold_plan["cache_hits"]} == {"phase3c_smoke"}
    assert len(cold_plan["pending"]) == 158


def test_request_plan_is_answer_key_free_and_evidence_digest_bound(cold_plan):
    assert cold_plan["preflight"]["answer_key_read"] is False
    for item, request in zip(cold_plan["planned"], cold_plan["request_rows"]):
        assert not (validation.FORBIDDEN_REQUEST_KEYS & validation._request_keys(request["payload"]))
        assert request["payload"]["tools"] == []
        assert request["payload"]["store"] is False
        assert request["provider_payload_digest"] == grounded._cache_key(request["payload"])
        assert request["cache_id"] == validation._canonical_digest({
            "experimental_unit": [request["model_id"], request["reaction_id"]],
            "provider_payload_digest": request["provider_payload_digest"],
        })
        assert request["evidence_digest"] == validation._canonical_digest([record.to_dict() for record in item["evidence"]])


def test_dry_run_cost_and_request_caps_pass(cold_plan):
    preflight = cold_plan["preflight"]
    assert preflight["planned_rows"] == 163
    assert preflight["pending_calls"] == preflight["authorized_new_calls"] == 158
    assert preflight["expected_cost_usd"] <= preflight["cost_cap_usd"] == 3.50
    assert preflight["automatic_retries"] == 0
    assert preflight["test_rows_read"] == 0
    assert len({row["cache_id"] for row in cold_plan["request_rows"]}) == 163


def test_retrieval_state_classification_accounting():
    rows = [
        _metric_row(sample_id="a", retrieval_state="A_truth_at_rank1", grounded_answered=True, grounded_exact=True, grounded_abstain=False, fusion_top1_exact=True),
        _metric_row(sample_id="b", retrieval_state="B_truth_at_ranks2_10", grounded_answered=True, grounded_exact=True, grounded_abstain=False),
        _metric_row(sample_id="c"),
    ]
    result = validation.retrieval_state_analysis(rows)
    assert result["A_truth_at_rank1"]["preserved_correct"]["count"] == 1
    assert result["B_truth_at_ranks2_10"]["promoted_truth"]["count"] == 1
    assert result["C_truth_absent_top10"]["abstained"]["count"] == 1
    assert sum(block["n"] for block in result.values()) == len(rows)


def test_paired_transition_accounting_is_exhaustive():
    rows = [
        _metric_row(grounded_exact=True, fusion_top1_exact=True, grounded_answered=True, grounded_abstain=False),
        _metric_row(grounded_exact=True, fusion_top1_exact=False, grounded_answered=True, grounded_abstain=False),
        _metric_row(grounded_exact=False, fusion_top1_exact=True),
        _metric_row(grounded_exact=False, fusion_top1_exact=False),
    ]
    result = validation.paired_transitions(rows, "fusion_top1")
    assert [result[key] for key in ("both_correct", "grounded_only_correct", "comparator_only_correct", "neither_correct")] == [1, 1, 1, 1]
    assert sum(result[key] for key in ("both_correct", "grounded_only_correct", "comparator_only_correct", "neither_correct")) == result["n"]


def test_cluster_bootstrap_is_deterministic():
    rows = [
        _metric_row(sample_id="a", cluster_id="c1", grounded_exact=True),
        _metric_row(sample_id="b", cluster_id="c1", fusion_top1_exact=True),
        _metric_row(sample_id="c", cluster_id="c2", grounded_exact=True),
        _metric_row(sample_id="d", cluster_id="c2"),
    ]
    first = validation._bootstrap_delta(rows, "grounded_exact", "fusion_top1_exact")
    second = validation._bootstrap_delta(rows, "grounded_exact", "fusion_top1_exact")
    assert first == second
    assert first["replicates"] == 10_000
    assert first["seed"] == 20260902
    assert first["n_clusters"] == 2


def test_interrupted_run_is_resumable_without_repurchase(cold_plan, tmp_path):
    items = [item for item in cold_plan["planned"] if item["cache_source"] is None][:2]
    subset = {**cold_plan, "planned": items}
    cache_dir = tmp_path / "cache"
    calls = []

    def parse(payload):
        item = items[len(calls)]
        calls.append(item["cache_id"])
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return _response(item)

    with pytest.raises(RuntimeError, match="no retry"):
        validation.run_plan(subset, execute=True, cache_only=False, validation_cache=cache_dir, parse_fn=parse)
    assert len(calls) == 2
    replay_calls = []
    rows, summary = validation.run_plan(
        subset, execute=True, cache_only=False, validation_cache=cache_dir,
        parse_fn=lambda payload: replay_calls.append(payload),
    )
    assert len(rows) == 2
    assert replay_calls == []
    assert summary["api_calls_this_invocation"] == 0


def test_cache_only_full_replay_makes_zero_calls(cold_plan, tmp_path):
    cache = FileCache(tmp_path / "cache")
    for item in cold_plan["planned"]:
        if item["cache_source"] is None:
            cache.put(item["cache_id"], {
                "sample_id": item["sample"]["sample_id"], "cache_id": item["cache_id"],
                "terminal_status": "succeeded", "compliance_problems": [],
                "annotation": _annotation(item).model_dump(mode="json"), "cost_usd": 0.01,
                "usage": {}, "model_requested": grounded.MODEL, "model_returned": grounded.MODEL,
                "response_id": "resp_mock", "evidence_digest": item["evidence_digest"],
            })
    rows, summary = validation.run_plan(
        cold_plan, execute=False, cache_only=True, validation_cache=tmp_path / "cache",
        parse_fn=lambda payload: pytest.fail("cache-only constructed a provider call"),
    )
    assert len(rows) == 163
    assert summary["api_calls_this_invocation"] == 0
    assert summary["cache_hits_this_invocation"] == 163


def test_persistent_attempt_cap_and_unresolved_attempt_fail_closed(cold_plan, tmp_path):
    item = next(item for item in cold_plan["planned"] if item["cache_source"] is None)
    subset = {**cold_plan, "planned": [item]}
    cache_dir = tmp_path / "attempts"
    cache_dir.mkdir()
    ledger = {"attempts": [{
        "sample_id": item["sample"]["sample_id"], "cache_id": item["cache_id"],
        "reserved_cost_usd": 0.01, "actual_cost_usd": None, "status": "attempted",
    }]}
    (cache_dir / "_attempt_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unresolved prior attempt"):
        validation.run_plan(subset, execute=True, cache_only=False, validation_cache=cache_dir, parse_fn=lambda payload: pytest.fail("must not call"))


def test_langchain_provenance_and_native_parity(cold_plan, index):
    report = validation.langchain_parity(cold_plan, index)
    assert report["byte_equivalent"] is True
    assert report["n_queries_compared"] == 163
    assert report["demonstration_api_calls"] == 0
    assert report["paid_calls_used_langchain_wrapper"] is False
    assert report["paid_calls_used_autonomous_provider_tool_calls"] is False
    assert report["provider_request_tools"] == []


def test_manifest_verification_is_read_only(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    grounded.write_artifact_manifest(tmp_path, [artifact], root=tmp_path)
    original_root = validation.REPO_ROOT
    try:
        validation.REPO_ROOT = tmp_path
        before = (tmp_path / "artifact_manifest.json").read_bytes()
        assert validation.verify_manifest(tmp_path) == []
        assert (tmp_path / "artifact_manifest.json").read_bytes() == before
    finally:
        validation.REPO_ROOT = original_root


def test_preflight_artifacts_rebuild_byte_identically(cold_plan, index, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    validation.write_preflight(cold_plan, index, first)
    validation.write_preflight(cold_plan, index, second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {path.name: path.read_bytes() for path in second.iterdir()}


@pytest.mark.skipif(not (validation.OUT / "frozen_responses.jsonl").exists(), reason="paid validation responses not frozen yet")
def test_derived_evaluation_rebuilds_byte_identically(cold_plan, tmp_path):
    responses = validation._read_jsonl(validation.OUT / "frozen_responses.jsonl")
    first, second = tmp_path / "eval1", tmp_path / "eval2"
    validation.write_evaluation(cold_plan, responses, first)
    validation.write_evaluation(cold_plan, responses, second)
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {path.name: path.read_bytes() for path in second.iterdir()}
    rebuild = validation.write_deterministic_rebuild(cold_plan, responses, tmp_path / "rebuild")
    assert json.loads(rebuild.read_text(encoding="utf-8"))["all_byte_equal"] is True


@pytest.mark.skipif(not (validation.OUT / "frozen_responses.jsonl").exists(), reason="paid validation responses not frozen yet")
def test_frozen_predictions_obey_evidence_constraint(cold_plan):
    responses = validation._read_jsonl(validation.OUT / "frozen_responses.jsonl")
    rows = validation.build_scored_rows(cold_plan, responses)
    assert len(rows) == 163
    assert sum(row["grounded_unsupported"] for row in rows) == 0
    assert all(not row["grounded_answered"] or row["grounded_predicted_kegg_id"] in row["evidence_ids"] for row in rows)


@pytest.mark.skipif(not (validation.OUT / "cost_usage_report.json").exists(), reason="paid validation evaluation not frozen yet")
def test_decision_analysis_answers_all_prespecified_questions():
    result = validation.decision_analysis()
    questions = result["questions"]
    assert len(questions) == 7
    assert questions["1_grounding_reduces_unsupported_guessing"]["answer"] is True
    assert questions["5_final_system"]["answer"] in {"fusion_alone", "grounded_llm_on_every_reaction"}
    assert questions["5_final_system"]["prespecified_selective_routing"] is False
    assert questions["6_all_969_validation_scientifically_necessary"]["answer"] is False


@pytest.mark.skipif(not (validation.OUT / "artifact_manifest.json").exists(), reason="validation manifest not frozen yet")
def test_committed_manifest_excludes_raw_scratch():
    manifest = json.loads((validation.OUT / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert all(not Path(item["path"]).name.startswith("_") for item in manifest["files"])
