"""Offline tests for the Phase 3C grounded smoke protocol."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.scripts import phase3c_grounded as grounded


@pytest.fixture(scope="module")
def index():
    return grounded.EvidenceIndex()


@pytest.fixture(scope="module")
def plan(index, tmp_path_factory):
    return grounded.build_plan(index, tmp_path_factory.mktemp("cache"))


def _annotation(**overrides):
    values = {
        "abstain": False,
        "predicted_kegg_id": "R00001",
        "selected_evidence_rank": 1,
        "supporting_evidence_ids": ["E01"],
        "confidence": 0.8,
        "reasoning_summary": "The selected equation best matches the target.",
        "abstention_reason": None,
    }
    values.update(overrides)
    return grounded.GroundedAnnotation(**values)


def _fake_response(annotation, n_in=500, n_out=100):
    usage = SimpleNamespace(
        input_tokens=n_in,
        output_tokens=n_out,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        output_tokens_details=SimpleNamespace(reasoning_tokens=10),
    )
    return SimpleNamespace(output_parsed=annotation, usage=usage, model=grounded.MODEL, id="resp_mock")


def test_frozen_retrieval_is_deterministic_and_complete(index):
    query = index.query_for("BIOMD0000000017", "R11")
    first = grounded.retrieve_kegg_evidence(query, index=index)
    second = grounded.retrieve_kegg_evidence(query, index=index)
    assert grounded.canonical_evidence_bytes(first) == grounded.canonical_evidence_bytes(second)
    assert len(first) == 10
    assert [row.fused_rank for row in first] == list(range(1, 11))
    assert len({row.kegg_id for row in first}) == 10
    assert all(row.provenance["fusion_sha256"] == grounded.EXPECTED_FUSION_SHA256 for row in first)


def test_query_must_be_validation_only_and_exact(index):
    query = index.query_for("BIOMD0000000017", "R11")
    with pytest.raises(ValueError, match="exactly match"):
        index.retrieve(grounded.RetrievalQuery(query.model_id, query.reaction_id, query.text + " changed"))
    with pytest.raises(KeyError, match="validation"):
        index.query_for("not", "present")


def test_mechanical_sample_has_all_strata_and_both_presence_states(plan):
    samples = plan["samples"]
    answers = plan["answer_key"]
    assert len(samples) == len(answers) == 5
    assert [row["corrected_stratum"] for row in samples] == list(grounded.STRATUM_PRESENCE)
    assert {row["answer_present_in_fused_top10"] for row in answers} == {False, True}
    assert all(row["split"] == "validation" for row in samples)


def test_sample_is_truth_free_and_answer_key_is_separate(plan):
    forbidden = {"ground_truth_ids", "answer_present_in_fused_top10", "first_ground_truth_evidence_rank"}
    assert all(not (forbidden & set(row)) for row in plan["samples"])
    assert all("ground_truth_ids" in row for row in plan["answer_key"])


def test_requests_do_not_contain_labels_or_test_data(plan):
    answers = {identifier for row in plan["answer_key"] for identifier in row["ground_truth_ids"]}
    for row in plan["request_rows"]:
        payload = row["payload"]
        assert payload["tools"] == []
        assert payload["store"] is False
        assert "model_id" not in payload and "reaction_id" not in payload
        assert "ground_truth" not in json.dumps(payload)
        # Answers may legitimately occur as retrieval evidence; no answer-key field or marker may occur.
        assert "answer_present_in_fused_top10" not in json.dumps(payload)
        assert set(grounded.parse_kegg_ids(payload["input"])).issubset({r["kegg_id"] for e in plan["evidence_rows"] if e["sample_id"] == row["sample_id"] for r in e["records"]})
    assert answers  # keep the test honest: the sealed key is non-empty.


def test_langchain_tool_is_byte_equivalent_to_native(index):
    pytest.importorskip("langchain_core")
    query = index.query_for("BIOMD0000000017", "R11")
    native = index.retrieve(query)
    adapted = grounded.run_langchain_tool_step(index, query)
    assert grounded.canonical_evidence_bytes(native) == grounded.canonical_evidence_bytes(adapted)


def test_non_abstention_evidence_compliance_rules(index):
    evidence = index.retrieve(index.query_for("BIOMD0000000017", "R11"))
    good = _annotation(predicted_kegg_id=evidence[0].kegg_id)
    assert grounded.validate_grounded_annotation(good, evidence) == []
    unsupported = _annotation(predicted_kegg_id="R99999")
    assert "prediction is absent from supplied evidence" in grounded.validate_grounded_annotation(unsupported, evidence)
    bad_rank = _annotation(predicted_kegg_id=evidence[1].kegg_id)
    assert "predicted_kegg_id does not match selected_evidence_rank" in grounded.validate_grounded_annotation(bad_rank, evidence)
    missing_support = _annotation(predicted_kegg_id=evidence[0].kegg_id, supporting_evidence_ids=[])
    assert grounded.validate_grounded_annotation(missing_support, evidence)


def test_abstention_schema_rules(index):
    evidence = index.retrieve(index.query_for("BIOMD0000000017", "R11"))
    good = _annotation(abstain=True, predicted_kegg_id=None, selected_evidence_rank=None, supporting_evidence_ids=[], confidence=0, abstention_reason="Evidence is ambiguous.")
    assert grounded.validate_grounded_annotation(good, evidence) == []
    bad = _annotation(abstain=True, abstention_reason=None)
    problems = grounded.validate_grounded_annotation(bad, evidence)
    assert any("abstention" in problem for problem in problems)


def test_pydantic_schema_is_strict():
    with pytest.raises(Exception):
        _annotation(extra_field="forbidden")
    with pytest.raises(Exception):
        _annotation(confidence=1.1)
    required = set(grounded.GroundedAnnotation.model_json_schema()["required"])
    assert required == {"abstain", "predicted_kegg_id", "selected_evidence_rank", "supporting_evidence_ids", "confidence", "reasoning_summary", "abstention_reason"}


def test_dry_run_caps_and_zero_calls(plan):
    dry = plan["dry_run"]
    assert dry["api_calls"] == 0
    assert dry["request_cap"] == dry["attempt_cap"] == 5
    assert dry["automatic_retries"] == 0
    assert dry["worst_case_cost_usd"] <= 0.50
    assert dry["test_rows_read"] == 0


def test_cache_is_atomic_compatible_and_cache_only_replays(plan, tmp_path):
    cache_dir = tmp_path / "_cache"
    frozen_before = grounded.RESPONSES_PATH.read_bytes() if grounded.RESPONSES_PATH.exists() else None
    annotations = {}
    for item in plan["planned"]:
        first = item["evidence"][0]
        annotations[item["cache_id"]] = _annotation(predicted_kegg_id=first.kegg_id)

    calls = []
    def parse(payload):
        calls.append(payload)
        return _fake_response(annotations[grounded._cache_key(payload)])

    rows, summary = grounded.run_plan(plan, execute=True, cache_only=False, cache_dir=cache_dir, parse_fn=parse)
    assert summary["api_calls"] == summary["attempts"] == 5
    assert len(calls) == len(rows) == 5
    assert not list(cache_dir.glob("*.tmp"))
    replay, replay_summary = grounded.run_plan(plan, execute=False, cache_only=True, cache_dir=cache_dir)
    assert replay_summary["api_calls"] == replay_summary["attempts"] == 0
    assert replay_summary["cache_hits"] == 5
    assert all(row["cache_hit"] for row in replay)
    if frozen_before is not None:
        assert grounded.RESPONSES_PATH.read_bytes() == frozen_before


def test_cache_only_fails_closed_on_miss(plan, tmp_path):
    with pytest.raises(RuntimeError, match="would call"):
        grounded.run_plan(plan, execute=False, cache_only=True, cache_dir=tmp_path / "empty")


def test_persistent_attempt_cap_blocks_resume(plan, tmp_path):
    cache_dir = tmp_path / "attempts"
    cache_dir.mkdir()
    (cache_dir / "_attempt_ledger.json").write_text(json.dumps({"attempts": [{"cache_id": str(i), "reserved_cost_usd": 0.01, "actual_cost_usd": None, "status": "attempted"} for i in range(5)]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="persistent total-attempt"):
        grounded.run_plan(plan, execute=True, cache_only=False, cache_dir=cache_dir, parse_fn=lambda _: None)


def test_manifest_verification_is_read_only(tmp_path):
    artifact = tmp_path / "x.json"
    artifact.write_text("{}\n", encoding="utf-8")
    grounded.write_artifact_manifest(tmp_path, [artifact], root=tmp_path)
    original_root = grounded.REPO_ROOT
    try:
        grounded.REPO_ROOT = tmp_path
        before = (tmp_path / "artifact_manifest.json").read_bytes()
        assert grounded.verify_artifacts(tmp_path) == []
        assert (tmp_path / "artifact_manifest.json").read_bytes() == before
    finally:
        grounded.REPO_ROOT = original_root


def test_deterministic_artifact_rebuilds(plan, tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    grounded.write_plan(plan, first)
    grounded.write_plan(plan, second)
    for name in ("sample.jsonl", "answer_key.jsonl", "frozen_evidence.jsonl", "frozen_requests.jsonl", "dry_run_plan.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
