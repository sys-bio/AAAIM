import copy
import json
from pathlib import Path

import pytest

from benchmark.scripts import luna_output_budget as budget
from benchmark.scripts import luna_model250728_analysis as original


@pytest.fixture
def evidence():
    return [{"evidence_id": "E01", "kegg_id": "R00001", "fused_rank": 1,
             "fused_score": .03, "bm25_rank": 1, "trained_biencoder_rank": 2,
             "name": "example", "definition": "example", "equation": "A => B",
             "enzyme": "", "rclass": "", "brite": ""}]


@pytest.fixture
def answer():
    return {"abstain": False, "predicted_kegg_id": "R00001", "selected_evidence_rank": 1,
            "supporting_evidence_ids": ["E01"], "confidence": .9,
            "reasoning_summary": "Supplied E01 represents the target.", "abstention_reason": None}


def response(answer, **extra):
    return {"status": "completed", "model": budget.MODEL,
            "usage": {"input_tokens": 1000, "output_tokens": 150,
                      "output_tokens_details": {"reasoning_tokens": 30}},
            "output": [{"type": "message", "status": "completed", "content": [
                {"type": "output_text", "text": json.dumps(answer)}]}], **extra}


def write_plan(path, evidence, count=3, limit=2048):
    path.mkdir()
    requests, manifest = [], []
    for i in range(count):
        body = original.serialize_provider_body(f"A{i} => B{i}", evidence)
        body["max_output_tokens"] = limit
        unit = {"model_id": original.MODEL_ID, "reaction_id": f"r{i}"}
        key = original.unit_cache_key(unit, budget.MODEL, original.digest(body))
        requests.append({"unit": unit, "unit_cache_key": key, "provider_payload": body})
        manifest.append({"unit": unit, "unit_cache_key": key,
                         "provider_payload_sha256": original.digest(body), "reservation": budget.reservation(body)})
    original.atomic_jsonl(path / "requests.jsonl", requests)
    original.atomic_jsonl(path / "request_manifest.jsonl", manifest)
    total = sum(row["reservation"]["reservation_nusd"] for row in manifest)
    plan = {"request_count": count, "max_output_tokens": limit, "maximum_attempts": count,
            "reservation_nusd": total, "hard_cap_nusd": total,
            "requests_sha256": original.sha256_file(path / "requests.jsonl"),
            "request_manifest_sha256": original.sha256_file(path / "request_manifest.jsonl")}
    original.atomic_json(path / "plan.json", plan)
    original.atomic_json(path / "plan_identity.json", {"plan_digest": original.digest(plan)})
    return requests


@pytest.mark.parametrize("visible", [False, True])
def test_incomplete_is_failure_even_with_a_parseable_answer(evidence, answer, visible):
    raw = response(answer, status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    if not visible:
        raw["output"] = [{"type": "reasoning", "summary": []}]
    result = budget.classify_response(raw, evidence)
    assert result["terminal_status"] == "failure_incomplete"
    assert result["biological_abstain"] is None and result["annotation"] is None
    assert result["incomplete_details"]["reason"] == "max_output_tokens"
    assert result["usage"]["output_tokens"] == 150


def test_incomplete_message_is_detected_even_if_response_says_completed(evidence, answer):
    raw = response(answer)
    raw["output"][0]["status"] = "incomplete"
    assert budget.classify_response(raw, evidence)["terminal_status"] == "failure_incomplete"


@pytest.mark.parametrize("kind,expected", [
    ("empty", "failure_empty"), ("invalid", "failure_schema_invalid"),
    ("refusal", "failure_refusal"), ("provider", "failure_provider"),
    ("model", "failure_model_mismatch"), ("compliance", "failure_evidence_compliance"),
])
def test_failures_are_never_biological_abstentions(evidence, answer, kind, expected):
    raw = response(answer)
    if kind == "empty":
        raw["output"] = []
    elif kind == "invalid":
        raw["output"][0]["content"][0]["text"] = '{"abstain":true}'
    elif kind == "refusal":
        raw["output"][0]["content"] = [{"type": "refusal", "refusal": "declined"}]
    elif kind == "provider":
        raw["status"] = "failed"
    elif kind == "model":
        raw["model"] = "gpt-5.6-terra"
    else:
        answer.update({"abstain": True, "predicted_kegg_id": None, "selected_evidence_rank": None,
                       "abstention_reason": "Insufficient evidence."})
        raw = response(answer)  # Still claims E01 support: invalid abstention.
    result = budget.classify_response(raw, evidence)
    assert result["terminal_status"] == expected
    assert result["biological_abstain"] is None


def test_valid_abstention_and_selection_remain_distinct(evidence, answer):
    assert budget.classify_response(response(answer), evidence)["terminal_status"] == "succeeded"
    answer.update({"abstain": True, "predicted_kegg_id": None, "selected_evidence_rank": None,
                   "supporting_evidence_ids": [], "abstention_reason": "Insufficient evidence."})
    result = budget.classify_response(response(answer), evidence)
    assert result["terminal_status"] == "abstained" and result["biological_abstain"] is True


def test_denied_execution_never_invokes_callback_or_reserves(tmp_path, evidence):
    plan = tmp_path / "plan"
    rows = write_plan(plan, evidence)
    run = budget.RunState(plan, tmp_path / "state")
    def forbidden(_):
        pytest.fail("provider boundary invoked")
    with pytest.raises(ValueError, match="not authorized"):
        run.run_one(rows[0]["unit_cache_key"], forbidden)
    assert not run._read()["attempts"]


def test_interruption_before_recording_retains_reservation_and_never_resends(tmp_path, evidence):
    plan, state = tmp_path / "plan", tmp_path / "state"
    rows = write_plan(plan, evidence)
    run = budget.RunState(plan, state)
    key = rows[0]["unit_cache_key"]
    calls = []
    def interrupted(_):
        calls.append(1)
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        run.run_one(key, interrupted, authorized_plan_digest=run.plan_digest)
    resumed = budget.RunState(plan, state)
    result = resumed.run_one(key, interrupted, authorized_plan_digest=resumed.plan_digest)
    assert result["terminal_status"] == "failure_interrupted_or_uncertain"
    assert len(calls) == 1
    assert resumed.total(resumed._read()) == resumed.by_key[key]["reservation"]["reservation_nusd"]
    assert len(resumed.coverage()) == 3
    assert sum(row["status"] == "not_attempted" for row in resumed.coverage()) == 2


def test_provider_exception_is_recorded_once_and_retains_full_cost(tmp_path, evidence):
    plan = tmp_path / "plan"
    rows = write_plan(plan, evidence)
    run = budget.RunState(plan, tmp_path / "state")
    key, calls = rows[0]["unit_cache_key"], []
    def timeout(_):
        calls.append(1)
        raise TimeoutError("not retained in the record")
    first = run.run_one(key, timeout, authorized_plan_digest=run.plan_digest)
    resumed = budget.RunState(plan, run.directory)
    assert resumed.run_one(key, timeout, authorized_plan_digest=run.plan_digest) == first
    assert len(calls) == 1 and first["biological_abstain"] is None
    assert first["raw_response"]["error"] == {"type": "TimeoutError"}
    with pytest.raises(ValueError, match="only valid completed"):
        resumed.settle_verified(key, 1, "fixture bill")


def test_cache_rebuilds_from_journal_after_interruption_and_billing_settles(tmp_path, evidence, answer, monkeypatch):
    plan, state = tmp_path / "plan", tmp_path / "state"
    rows = write_plan(plan, evidence)
    run = budget.RunState(plan, state)
    key = rows[0]["unit_cache_key"]
    run.reserve(key)
    monkeypatch.setattr(run, "_write_cache", lambda *args: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        run.record(key, response(answer))
    resumed = budget.RunState(plan, state)
    assert resumed.cached_result(key)["terminal_status"] == "succeeded"
    assert (state / "cache" / f"{key}.json").exists()
    with pytest.raises(ValueError, match="already attempted"):
        resumed.reserve(key)
    resumed.settle_verified(key, 1000, "synthetic verified invoice")
    assert budget.RunState(plan, state).total(resumed._read()) == 1000


def test_cache_corruption_blocks_repurchase(tmp_path, evidence, answer):
    plan = tmp_path / "plan"
    rows = write_plan(plan, evidence)
    run = budget.RunState(plan, tmp_path / "state")
    key = rows[0]["unit_cache_key"]
    run.reserve(key)
    run.record(key, response(answer))
    (run.directory / "cache" / f"{key}.json").write_text('{}', encoding="utf-8")
    with pytest.raises(ValueError, match="cache/journal mismatch"):
        run.cached_result(key)


def test_limit_changes_model_specific_cache_identity_and_resume_plan_guard(tmp_path, evidence):
    first, second = tmp_path / "512", tmp_path / "2048"
    rows_a = write_plan(first, evidence, limit=512)
    rows_b = write_plan(second, evidence, limit=2048)
    assert rows_a[0]["unit_cache_key"] != rows_b[0]["unit_cache_key"]
    budget.RunState(first, tmp_path / "state")
    with pytest.raises(ValueError, match="journal plan/cap mismatch"):
        budget.RunState(second, tmp_path / "state")


def test_all_541_attempts_fit_even_with_every_reservation_retained(tmp_path, evidence):
    plan = tmp_path / "plan"
    rows = write_plan(plan, evidence, count=541)
    run = budget.RunState(plan, tmp_path / "state")
    for row in rows:
        run.reserve(row["unit_cache_key"])
    resumed = budget.RunState(plan, run.directory)
    assert len(resumed.coverage()) == 541
    assert resumed.total(resumed._read()) == resumed.plan["hard_cap_nusd"]
    assert all(row["biological_abstain"] is None for row in resumed.coverage())


def test_real_frozen_541_requests_change_only_output_limit():
    source = original.REPO_ROOT / "tests/useCase/results/MODEL2507280001/selected_rrf_analysis"
    if not source.exists():
        pytest.skip("local ignored Luna artifacts are not installed")
    rows = budget.read_jsonl(source / "requests.jsonl")
    assert len(rows) == len({row["unit"]["reaction_id"] for row in rows}) == 541
    for old in rows:
        body = copy.deepcopy(old["provider_payload"])
        body["max_output_tokens"] = 2048
        assert budget.reservation(body)["reservation_nusd"] > budget.reservation(old["provider_payload"])["reservation_nusd"]
        assert original.unit_cache_key(old["unit"], budget.MODEL, original.digest(body)) != old["unit_cache_key"]
        body["max_output_tokens"] = 512
        assert body == old["provider_payload"]


def test_actual_both_541_plans_survive_incomplete_results_and_retained_reservations(tmp_path):
    base = original.REPO_ROOT / "tests/useCase/results/MODEL2507280001"
    out = base / "selected_rrf_output_budget_20261107"
    if not out.exists():
        pytest.skip("local ignored output-budget plans are not installed")
    report = budget.verify_artifacts(base / "selected_rrf_analysis", out)
    assert report["status"] == "passed"
    for limit in (512, 2048):
        run = budget.RunState(out / f"max_output_{limit}", tmp_path / str(limit))
        for row in run.manifest:
            run.reserve(row["unit_cache_key"])
        key = run.manifest[0]["unit_cache_key"]
        run.record(key, {"status": "incomplete", "model": budget.MODEL,
                        "incomplete_details": {"reason": "max_output_tokens"},
                        "usage": {"output_tokens": limit}, "output": []})
        resumed = budget.RunState(run.plan_dir, run.directory)
        assert resumed.total(resumed._read()) == resumed.plan["reservation_nusd"]
        assert resumed.total(resumed._read()) <= resumed.plan["hard_cap_nusd"]
        coverage = resumed.coverage()
        assert len(coverage) == 541
        assert sum(row["status"] == "failure_incomplete" for row in coverage) == 1
        assert all(row["biological_abstain"] is None for row in coverage)
        with pytest.raises(ValueError, match="already attempted"):
            resumed.reserve(key)
