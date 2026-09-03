"""Mocked tests for the Phase 3A OpenAI smoke-test runner.

No live network calls. Secrets are never asserted, logged, or written.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.phase3_common import (
    OUT_PILOT,
    OUT_PROMPTS,
    PHASE3_DIR,
    PILOT_SEED,
    PILOT_SPLIT,
    PRICING_OPENAI_TERRA,
    PROMPT_TEMPLATE_VERSION,
    SMOKE_N_REQUESTS,
    STRATA,
    TOKENIZER_CONSERVATIVE,
    TOKENIZER_SCAFFOLD,
    estimate_tokens_conservative,
    require_live_tokenizer,
)
from benchmark.scripts.phase3_modes import parse_structured_output
from benchmark.scripts.phase3_openai_run import (
    SECRET_ENV,
    BudgetExceeded,
    ClosedListBlocked,
    InferenceSettings,
    LeakageBlocked,
    MissingAPIKeyError,
    Phase3StructuredOutput,
    RunConfig,
    assert_env_file_protected,
    assert_no_closed_list_or_answer_key,
    attach_variants,
    call_with_retries,
    env_file_is_tracked,
    finalize_request,
    interpret_parsed_payload,
    interpret_response,
    is_retryable,
    load_prompt_rows,
    main,
    plan_run,
    plan_smoke_run,
    require_api_key_for_live,
    run_planned,
    select_smoke_reactions,
    select_validation_reactions,
    usage_from_response,
    write_plan_artifacts,
)
from benchmark.scripts.phase3_prompts import CONTEXT_VARIANTS, SYSTEM_DIRECT

FAKE_KEY = "sk-test-secret-value-not-for-commit"
live_only = pytest.mark.skipif(
    not (PHASE3_DIR / "pilot_sample.csv").exists(),
    reason="Phase 3 artifacts not built",
)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prompt_row(sample_id, model_id, reaction_id, variant,
                stratum="empty_constrained", extra_user=""):
    user = (
        "Identify the KEGG reaction that this SBML reaction most likely represents.\n"
        f"SBML reaction id: {reaction_id}\n"
        "Equation: A_c => B_c\n"
        f"{extra_user}"
    )
    return {
        "sample_id": sample_id,
        "model_id": model_id,
        "reaction_id": reaction_id,
        "cluster_id": f"CLU_{model_id}",
        "stratum": stratum,
        "variant": variant,
        "template_version": PROMPT_TEMPLATE_VERSION,
        "prompt": {
            "messages": [
                {"role": "system", "content": SYSTEM_DIRECT},
                {"role": "user", "content": user},
            ],
            "mode": "direct_open_set",
            "system_direct": SYSTEM_DIRECT,
            "system_tool_assisted": "tool instruction",
            "template_version": PROMPT_TEMPLATE_VERSION,
            "variant": variant,
            "response_schema": {"type": "object"},
        },
    }


def _sample_frame() -> pd.DataFrame:
    rows = []
    for i, stratum in enumerate(STRATA[:3], start=1):
        rows.append({
            "sample_id": f"P{i}",
            "model_id": f"M{i}",
            "reaction_id": f"rxn{i}",
            "cluster_id": f"C{i}",
            "split": PILOT_SPLIT,
            "stratum": stratum,
        })
    return pd.DataFrame(rows)


def _all_prompt_rows(frame: pd.DataFrame) -> list:
    rows = []
    for rec in frame.itertuples(index=False):
        for variant in CONTEXT_VARIANTS:
            rows.append(_prompt_row(
                rec.sample_id, rec.model_id, rec.reaction_id, variant, rec.stratum,
            ))
    return rows


def _workspace(tmp_path: Path, *, sample=None, prompts=None, answer_key=True) -> dict:
    sample = sample if sample is not None else _sample_frame()
    prompts = prompts if prompts is not None else _all_prompt_rows(sample)
    sample_path = tmp_path / "pilot_sample.csv"
    prompts_path = tmp_path / "pilot_prompts.jsonl"
    pricing_path = tmp_path / "pricing.json"
    key_path = tmp_path / "pilot_answer_key.csv"
    sample.to_csv(sample_path, index=False, lineterminator="\n")
    _write_jsonl(prompts_path, prompts)
    pricing_path.write_text(PRICING_OPENAI_TERRA.read_text(encoding="utf-8"), encoding="utf-8")
    if answer_key:
        pd.DataFrame({
            "sample_id": sample.sample_id,
            "model_id": sample.model_id,
            "reaction_id": sample.reaction_id,
            "cluster_id": sample.cluster_id,
            "stratum": sample.stratum,
            "ground_truth_kegg_all": ["R00024"] * len(sample),
            "ground_truth_kegg_primary": ["R00024"] * len(sample),
            "num_ground_truth_ids": [1] * len(sample),
        }).to_csv(key_path, index=False, lineterminator="\n")
    return {
        "sample_path": sample_path,
        "prompts_path": prompts_path,
        "pricing_path": pricing_path,
        "answer_key_path": key_path,
        "out_dir": tmp_path / "smoke",
        "cache_dir": tmp_path / "_response_cache",
    }


def _config(paths: dict, **overrides) -> RunConfig:
    cfg = RunConfig(
        sample_path=paths["sample_path"],
        prompts_path=paths["prompts_path"],
        answer_key_path=paths["answer_key_path"],
        pricing_path=paths["pricing_path"],
        out_dir=paths["out_dir"],
        cache_dir=paths["cache_dir"],
        load_env=False,
        evaluate=False,
        sleep=lambda _s: None,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _parsed(abstain=False, kegg_id="R00024", confidence=0.8):
    preds = [] if abstain else [{"kegg_id": kegg_id, "confidence": confidence}]
    return Phase3StructuredOutput(
        abstain=abstain,
        predictions=preds,
        rationale="mock",
        basis="recalled_knowledge",
    )


def _usage(input_tokens=80, output_tokens=40, cached=0, reasoning=10):
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_tokens_details=SimpleNamespace(cached_tokens=cached),
        output_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )


def _response(parsed=None, *, model="gpt-5.6-terra", usage="default",
              refusal=None, response_id="resp_test"):
    usage_obj = _usage() if usage == "default" else usage
    if refusal is not None:
        return SimpleNamespace(
            id=response_id,
            model=model,
            usage=usage_obj,
            output_parsed=None,
            output=[SimpleNamespace(
                content=[SimpleNamespace(type="refusal", refusal=refusal)],
            )],
        )
    return SimpleNamespace(
        id=response_id,
        model=model,
        usage=usage_obj,
        output_parsed=parsed if parsed is not None else _parsed(),
        output=[],
    )


def test_conservative_tokenizer_is_allowed_for_live_gate():
    assert estimate_tokens_conservative("abcd") == 2
    require_live_tokenizer(TOKENIZER_CONSERVATIVE)
    with pytest.raises(RuntimeError, match="tokenizer"):
        require_live_tokenizer(TOKENIZER_SCAFFOLD)


def test_env_helpers_do_not_expose_key(monkeypatch, caplog):
    monkeypatch.setenv(SECRET_ENV, FAKE_KEY)
    require_api_key_for_live()
    assert env_file_is_tracked() is False
    assert_env_file_protected()
    assert FAKE_KEY not in caplog.text


def test_missing_key_errors_only_for_live(monkeypatch, tmp_path):
    monkeypatch.delenv(SECRET_ENV, raising=False)
    paths = _workspace(tmp_path)
    cfg = _config(paths, execute=False)
    plan = plan_smoke_run(cfg)
    outcome = run_planned(plan, cfg)
    assert outcome["summary"]["api_calls"] == 0
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY is not set"):
        require_api_key_for_live()


def test_tracked_env_file_blocks_live(monkeypatch):
    monkeypatch.setattr(
        "benchmark.scripts.phase3_openai_run.env_file_is_tracked",
        lambda repo_root=None: True,
    )
    with pytest.raises(Exception, match="tracked"):
        assert_env_file_protected()


def test_dry_run_performs_zero_api_calls(tmp_path):
    paths = _workspace(tmp_path)
    calls = {"n": 0}

    def parse_fn(_kwargs):
        calls["n"] += 1
        return _response()

    cfg = _config(paths, execute=False, parse_fn=parse_fn)
    plan = plan_smoke_run(cfg)
    write_plan_artifacts(plan, cfg.out_dir)
    outcome = run_planned(plan, cfg)
    assert calls["n"] == 0
    assert outcome["summary"]["api_calls"] == 0
    assert plan["n_requests"] == SMOKE_N_REQUESTS
    assert all(row["terminal_status"] == "dry_run" for row in outcome["rows"])


def test_smoke_selects_nine_requests_three_variants_deterministically():
    sample = _sample_frame()
    a = select_smoke_reactions(sample, seed=PILOT_SEED)
    b = select_smoke_reactions(sample, seed=PILOT_SEED)
    assert list(zip(a.sample_id, a.reaction_id)) == list(zip(b.sample_id, b.reaction_id))
    assert len(a) == 3
    assert set(a.stratum) == set(STRATA[:3])
    planned = attach_variants(a, _all_prompt_rows(sample))
    assert len(planned) == 9
    by_rxn = {}
    for row in planned:
        by_rxn.setdefault(row["sample_id"], set()).add(row["variant"])
    assert all(v == set(CONTEXT_VARIANTS) for v in by_rxn.values())


def test_test_split_rows_are_refused():
    sample = _sample_frame()
    sample.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="validation-only"):
        select_smoke_reactions(sample)
    with pytest.raises(ValueError, match="validation-only"):
        select_validation_reactions(sample)


def test_plan_does_not_read_answer_key(tmp_path):
    src = inspect.getsource(plan_run)
    assert "answer_key_path" not in src
    assert "OUT_PILOT_KEY" not in src
    assert "pilot_answer_key" not in src
    paths = _workspace(tmp_path, answer_key=False)
    cfg = _config(paths, answer_key_path=tmp_path / "missing_answer_key.csv")
    plan = plan_smoke_run(cfg)
    assert plan["answer_key_read"] is False
    assert plan["test_split_read"] is False
    write_plan_artifacts(plan, cfg.out_dir)
    blob = (cfg.out_dir / "plan.json").read_text(encoding="utf-8")
    public = json.loads(blob)
    for req in public["requests"]:
        assert "ground_truth" not in req
        assert "answer_key" not in req
        assert "candidates" not in req


def test_leakage_aborts_before_api_call(tmp_path, monkeypatch):
    sample = _sample_frame()
    prompts = _all_prompt_rows(sample)
    prompts[0]["prompt"]["messages"][1]["content"] += "\nsee R00024"
    paths = _workspace(tmp_path, sample=sample, prompts=prompts)
    calls = {"n": 0}

    def parse_fn(_kwargs):
        calls["n"] += 1
        return _response()

    monkeypatch.setattr(
        "benchmark.scripts.phase3_openai_run.redact_kegg_reaction_ids", lambda text: text,
    )
    monkeypatch.setattr(
        "benchmark.scripts.phase3_openai_run.redact_kegg_in_obj", lambda obj: obj,
    )
    cfg = _config(paths, execute=True, parse_fn=parse_fn)
    with pytest.raises(LeakageBlocked):
        plan_smoke_run(cfg)
    assert calls["n"] == 0


def test_closed_candidate_list_is_rejected():
    with pytest.raises(ClosedListBlocked):
        assert_no_closed_list_or_answer_key(
            {"system": "x", "user": "y", "candidates": ["R00024"]},
            where="test",
        )
    with pytest.raises(ClosedListBlocked):
        assert_no_closed_list_or_answer_key(
            {"system": "x", "user": json.dumps({"ground_truth_kegg_all": "R00024"})},
            where="test",
        )


def test_structured_parse_and_catalog_membership():
    catalog = {"R00024"}
    ok = interpret_parsed_payload({
        "abstain": False,
        "predictions": [{"kegg_id": "R00024", "confidence": 0.9}],
        "rationale": "ok", "basis": "recalled_knowledge",
    }, catalog=catalog, raw_text="{}")
    assert ok["terminal_status"] == "succeeded"
    assert ok["predictions"][0].in_catalog is True

    absent = interpret_parsed_payload({
        "abstain": False,
        "predictions": [{"kegg_id": "R99999", "confidence": 0.4}],
        "rationale": "syntax", "basis": "recalled_knowledge",
    }, catalog=catalog, raw_text="{}")
    assert absent["predictions"][0].in_catalog is False
    assert absent["predictions"][0].well_formed is True

    dups = interpret_parsed_payload({
        "abstain": False,
        "predictions": [
            {"kegg_id": "R00024", "confidence": 0.9},
            {"kegg_id": "R00024", "confidence": 0.1},
        ],
        "rationale": "dup", "basis": "recalled_knowledge",
    }, catalog=catalog, raw_text="{}")
    assert dups["terminal_status"] == "compliance_error"
    assert "duplicate_predicted_ids" in dups["parse_error"]

    empty = interpret_parsed_payload({
        "abstain": False, "predictions": [],
        "rationale": "none", "basis": "recalled_knowledge",
    }, catalog=catalog, raw_text="{}")
    assert empty["abstain"] is False
    assert empty["parse_error"] == "abstain_false_without_predictions"

    malformed = parse_structured_output(json.dumps({
        "abstain": False,
        "predictions": [{"kegg_id": "R1", "confidence": 0.2}],
        "rationale": "bad", "basis": "recalled_knowledge",
    }), catalog=catalog)
    assert malformed["predictions"][0].well_formed is False


def test_refusal_is_not_an_abstention():
    interpreted = interpret_response(_response(refusal="policy"), catalog={"R00024"})
    assert interpreted["terminal_status"] == "refused"
    assert interpreted["abstain"] is False
    assert interpreted["predictions"] == []
    assert interpreted["refusal"] == "policy"


def test_invalid_schema_is_a_failure_not_abstention():
    interpreted = interpret_response(
        SimpleNamespace(output_parsed=None, output=[], usage=None, model="gpt-5.6-terra"),
        catalog={"R00024"},
    )
    assert interpreted["terminal_status"] == "schema_invalid"
    assert interpreted["abstain"] is False


def test_cache_hit_avoids_second_api_call(tmp_path):
    paths = _workspace(tmp_path)
    calls = {"n": 0}

    def parse_fn(_kwargs):
        calls["n"] += 1
        return _response(_parsed(), model="gpt-5.6-terra-returned")

    cfg = _config(paths, execute=True, parse_fn=parse_fn)
    plan = plan_smoke_run(cfg)
    first = run_planned(plan, cfg)
    assert calls["n"] == 9
    assert first["summary"]["api_calls"] == 9
    second = run_planned(plan, cfg)
    assert calls["n"] == 9
    assert second["summary"]["api_calls"] == 0
    assert second["summary"]["n_cached"] == 9
    assert {row["model_returned"] for row in second["rows"]} == {"gpt-5.6-terra-returned"}


def test_interrupted_run_resumes_from_cached_successes(tmp_path):
    paths = _workspace(tmp_path)
    calls = {"n": 0}

    def parse_fn(_kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            raise KeyboardInterrupt()
        return _response()

    cfg = _config(paths, execute=True, parse_fn=parse_fn)
    plan = plan_smoke_run(cfg)
    with pytest.raises(KeyboardInterrupt):
        run_planned(plan, cfg)
    assert calls["n"] == 4
    results = (cfg.out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(results) == 3

    def parse_fn_resume(_kwargs):
        calls["n"] += 1
        return _response()

    cfg.parse_fn = parse_fn_resume
    resumed = run_planned(plan, cfg)
    assert resumed["summary"]["n_cached"] == 3
    assert resumed["summary"]["api_calls"] == 6
    assert len(resumed["rows"]) == 9


def test_retry_limits_transient_errors_only():
    sleeps = []

    class Transient(Exception):
        status_code = 429

    class Auth(Exception):
        status_code = 401

    n = {"t": 0}

    def flaky(_kwargs):
        n["t"] += 1
        if n["t"] < 3:
            raise Transient("rate")
        return _response()

    resp, attempts = call_with_retries(
        flaky, {}, max_retries=2, sleep=lambda s: sleeps.append(s),
    )
    assert attempts == 3
    assert len(sleeps) == 2
    assert is_retryable(Transient()) is True
    assert is_retryable(Auth()) is False

    def auth(_kwargs):
        raise Auth("nope")

    with pytest.raises(Auth):
        call_with_retries(auth, {}, max_retries=5, sleep=lambda _s: None)


def test_preflight_budget_rejection(tmp_path):
    paths = _workspace(tmp_path)
    cfg = _config(paths, max_cost_usd=1e-9)
    with pytest.raises(BudgetExceeded, match="preflight"):
        plan_smoke_run(cfg)


def test_runtime_budget_stops_before_next_call(tmp_path):
    paths = _workspace(tmp_path)
    calls = {"n": 0}

    def parse_fn(_kwargs):
        calls["n"] += 1
        return _response(_parsed(), usage=_usage(input_tokens=100, output_tokens=3000))

    cfg = _config(
        paths, execute=True, parse_fn=parse_fn, max_cost_usd=0.03,
        max_output_tokens=200, max_retries=0,
    )
    plan = plan_smoke_run(cfg)
    outcome = run_planned(plan, cfg)
    assert calls["n"] == 1
    statuses = [row["terminal_status"] for row in outcome["rows"]]
    assert "budget_stopped" in statuses
    assert statuses.count("succeeded") == 1


def test_output_token_accounting_and_missing_usage(tmp_path):
    usage = usage_from_response(_response())
    assert usage["input_tokens"] == 80
    assert usage["output_tokens"] == 40
    assert usage["cached_input_tokens"] == 0
    assert usage["reasoning_tokens"] == 10
    assert usage["usage_missing"] is False
    missing = usage_from_response(SimpleNamespace(usage=None))
    assert missing["usage_missing"] is True

    paths = _workspace(tmp_path)

    def parse_fn(_kwargs):
        return _response(_parsed(), usage=None)

    cfg = _config(paths, execute=True, parse_fn=parse_fn, max_cost_usd=1.0)
    plan = plan_smoke_run(cfg)
    outcome = run_planned(plan, cfg)
    assert outcome["rows"][0]["usage_missing"] is True
    assert outcome["rows"][0]["cost_used_fallback"] is True


def test_logs_and_artifacts_never_contain_api_key(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv(SECRET_ENV, FAKE_KEY)
    paths = _workspace(tmp_path)
    cfg = _config(paths, execute=False)
    plan = plan_smoke_run(cfg)
    write_plan_artifacts(plan, cfg.out_dir)
    run_planned(plan, cfg)
    blob = ""
    for path in cfg.out_dir.glob("*"):
        blob += path.read_text(encoding="utf-8")
    assert FAKE_KEY not in blob
    assert FAKE_KEY not in caplog.text
    assert "sk-test-secret" not in blob


def test_main_dry_run_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv(SECRET_ENV, raising=False)
    paths = _workspace(tmp_path)
    code = main([
        "--no-dotenv",
        "--sample", str(paths["sample_path"]),
        "--prompts", str(paths["prompts_path"]),
        "--pricing", str(paths["pricing_path"]),
        "--out-dir", str(paths["out_dir"]),
        "--cache-dir", str(paths["cache_dir"]),
        "--skip-eval",
        "--max-cost-usd", "1.00",
    ])
    assert code == 0
    plan = json.loads((paths["out_dir"] / "plan.json").read_text(encoding="utf-8"))
    assert plan["n_requests"] == 9
    assert plan["model"] == "gpt-5.6-terra"


@live_only
def test_live_artifacts_select_exactly_nine_validation_prompts():
    sample = pd.read_csv(OUT_PILOT)
    selected = select_smoke_reactions(sample)
    assert (selected.split == "validation").all()
    prompts = load_prompt_rows(OUT_PROMPTS)
    planned_rows = attach_variants(selected, prompts)
    assert len(planned_rows) == 9
    settings = InferenceSettings()
    finalized = [finalize_request(row, settings) for row in planned_rows]
    assert len(finalized) == 9
    assert {item.variant for item in finalized} == set(CONTEXT_VARIANTS)
    assert len({item.sample_id for item in finalized}) == 3
    for item in finalized:
        assert "ground_truth" not in item.user
        assert "candidate list" not in item.user.lower()
    again = select_smoke_reactions(sample)
    assert list(selected.sample_id) == list(again.sample_id)


def test_validation_budget_gate_allows_theoretical_max_above_cap():
    from benchmark.scripts.phase3_openai_run import assert_budget_gate
    estimate = {
        "expected_usd_from_smoke_prior": 1.6,
        "worst_case_usd": 20.0,
    }
    assert_budget_gate(estimate, 5.0, require_worst_under_cap=False)
    with pytest.raises(BudgetExceeded, match="worst-case"):
        assert_budget_gate(estimate, 5.0, require_worst_under_cap=True)
    with pytest.raises(BudgetExceeded, match="expected"):
        assert_budget_gate(estimate, 1.0, require_worst_under_cap=False)


def test_validation_plan_cannot_access_answer_key(tmp_path, monkeypatch):
    from benchmark.scripts.phase3_openai_run import plan_run
    paths = _workspace(tmp_path)
    real_read = pd.read_csv

    def guarded(path, *args, **kwargs):
        text = str(path)
        if "pilot_answer_key" in text or "answer_key" in Path(text).name:
            raise AssertionError("answer key read during request construction")
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)

    original_read_text = Path.read_text

    def guarded_read_text(self, *args, **kwargs):
        if "pilot_answer_key" in self.name or "answer_key" in self.name:
            raise AssertionError("answer key path opened during request construction")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    cfg = _config(
        paths, execute=False, profile="validation", n_reactions=3, max_requests=9,
        max_cost_usd=5.0,
    )
    plan = plan_run(cfg)
    assert plan["n_requests"] == 9
    assert plan["answer_key_read"] is False
    assert plan["profile"] == "validation"


def test_validation_refuses_to_overwrite_smoke_dir(tmp_path):
    from benchmark.scripts.phase3_common import OUT_SMOKE_DIR
    from benchmark.scripts.phase3_openai_run import plan_run
    paths = _workspace(tmp_path)
    cfg = _config(
        paths, execute=False, profile="validation", n_reactions=3, max_requests=9,
        max_cost_usd=5.0, out_dir=OUT_SMOKE_DIR,
    )
    with pytest.raises(ValueError, match="must not overwrite smoke"):
        plan_run(cfg)


@live_only
def test_live_validation_plan_has_489_rows_and_reuses_smoke_cache():
    from benchmark.scripts.phase3_common import VALIDATION_N_REQUESTS
    from benchmark.scripts.phase3_modes import CACHE_DIR
    from benchmark.scripts.phase3_openai_run import (
        audit_planned_requests, plan_run,
    )
    sample = pd.read_csv(OUT_PILOT)
    cfg = RunConfig(
        sample_path=OUT_PILOT,
        prompts_path=OUT_PROMPTS,
        pricing_path=PRICING_OPENAI_TERRA,
        out_dir=PHASE3_DIR / "validation",
        cache_dir=CACHE_DIR,
        execute=False,
        profile="validation",
        n_reactions=163,
        max_requests=VALIDATION_N_REQUESTS,
        max_cost_usd=5.0,
        evaluate=False,
        load_env=False,
    )
    plan = plan_run(cfg)
    audit = audit_planned_requests(plan, sample=sample, require_all_sample_ids=True)
    assert audit["ok"] is True
    assert plan["n_requests"] == 489
    assert plan["n_reactions"] == 163
    assert plan["n_compatible_cache_hits"] >= 9
    assert plan["n_new_calls_max"] == 489 - plan["n_compatible_cache_hits"]
    smoke_plan = json.loads((PHASE3_DIR / "smoke" / "plan.json").read_text(encoding="utf-8"))
    smoke_ids = {req["cache_id"] for req in smoke_plan["requests"]}
    assert smoke_ids <= set(plan["cache_hit_ids"])
    assert plan["inference"]["reasoning_effort"] == "low"
    assert plan["inference"]["max_output_tokens"] == 1024
    assert plan["template_version"] == PROMPT_TEMPLATE_VERSION
    assert plan["estimate"]["expected_usd_from_smoke_prior"] < 5.0


def test_select_rescue_prompt_rows_rejects_succeeded_keys_and_requires_26():
    from benchmark.scripts.phase3_openai_run import select_rescue_prompt_rows
    original = []
    prompts = []
    for i in range(25):
        original.append({"sample_id": f"s{i}", "variant": "target_only", "terminal_status": "schema_invalid"})
        prompts.append(_prompt_row(f"s{i}", f"M{i}", f"rxn{i}", "target_only"))
    original.append({"sample_id": "ok", "variant": "target_only", "terminal_status": "succeeded"})
    prompts.append(_prompt_row("ok", "Mok", "rxnok", "target_only"))
    with pytest.raises(ValueError, match="must select 26"):
        select_rescue_prompt_rows(original, prompts)
    original[-1] = {
        "sample_id": "dup", "variant": "target_only", "terminal_status": "schema_invalid",
    }
    original.append({"sample_id": "dup", "variant": "target_only", "terminal_status": "succeeded"})
    prompts.append(_prompt_row("dup", "Mdup", "rxndup", "target_only"))
    with pytest.raises(ValueError, match="also succeeded"):
        select_rescue_prompt_rows(original, prompts)


def test_rescue_plan_refuses_to_overwrite_original_validation(tmp_path):
    from benchmark.scripts.phase3_common import OUT_VALIDATION_DIR
    from benchmark.scripts.phase3_openai_run import plan_run
    paths = _workspace(tmp_path)
    cfg = _config(
        paths,
        execute=False,
        profile="rescue_schema_invalid",
        n_reactions=26,
        max_requests=26,
        max_cost_usd=1.0,
        max_output_tokens=2048,
        max_retries=0,
        out_dir=OUT_VALIDATION_DIR,
    )
    with pytest.raises(ValueError, match="must not overwrite original validation"):
        plan_run(cfg)


@live_only
def test_live_rescue_plan_selects_only_original_schema_invalid_keys(tmp_path):
    from benchmark.scripts.phase3_common import (
        ORIGINAL_VALIDATION_RESULTS_SHA256,
        OUT_VALIDATION_DIR,
        RESCUE_N_REQUESTS,
        sha256_portable,
    )
    from benchmark.scripts.phase3_modes import CACHE_DIR
    from benchmark.scripts.phase3_openai_run import (
        audit_planned_requests, load_result_rows, plan_run,
    )
    original_path = OUT_VALIDATION_DIR / "results.jsonl"
    before = sha256_portable(original_path)
    assert before == ORIGINAL_VALIDATION_RESULTS_SHA256
    original = load_result_rows(original_path)
    invalid_keys = {
        (str(row["sample_id"]), str(row["variant"]))
        for row in original if row.get("terminal_status") == "schema_invalid"
    }
    succeeded_keys = {
        (str(row["sample_id"]), str(row["variant"]))
        for row in original if row.get("terminal_status") == "succeeded"
    }
    sample = pd.read_csv(OUT_PILOT)
    cfg = RunConfig(
        sample_path=OUT_PILOT,
        prompts_path=OUT_PROMPTS,
        pricing_path=PRICING_OPENAI_TERRA,
        out_dir=tmp_path / "rescue",
        cache_dir=CACHE_DIR,
        execute=False,
        profile="rescue_schema_invalid",
        n_reactions=26,
        max_requests=26,
        max_cost_usd=1.0,
        max_output_tokens=2048,
        max_retries=0,
        evaluate=False,
        load_env=False,
    )
    plan = plan_run(cfg)
    audit = audit_planned_requests(plan, sample=sample, require_all_sample_ids=False)
    assert audit["ok"] is True
    assert plan["n_requests"] == RESCUE_N_REQUESTS
    assert plan["inference"]["max_output_tokens"] == 2048
    assert plan["inference"]["max_retries"] == 0
    assert plan["answer_key_read"] is False
    assert plan["test_split_read"] is False
    keys = {(item.sample_id, item.variant) for item in plan["planned"]}
    assert keys == invalid_keys
    assert not (keys & succeeded_keys)
    assert plan["estimate"]["expected_usd_no_retry_at_max_output"] < 1.0
    assert all(item.settings.max_output_tokens == 2048 for item in plan["planned"])
    assert sha256_portable(original_path) == before
    if "split" in sample.columns:
        selected_ids = {item.sample_id for item in plan["planned"]}
        splits = set(sample.loc[sample.sample_id.astype(str).isin(selected_ids), "split"].astype(str))
        assert splits == {PILOT_SPLIT}

