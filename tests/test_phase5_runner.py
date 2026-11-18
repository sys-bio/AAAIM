"""Fault injection and real-plan checks; no provider network or held-out loader."""
import copy
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.scripts import phase5_runner as r
from benchmark.scripts import phase5_matched_analysis as a
from benchmark.scripts.phase5_llm_plan import render_input, provider_payload
from benchmark.scripts.phase5_billing import request_input_token_bound, request_reservation_nusd
from benchmark.scripts.phase5_perturb import digest


def make_fixture(path, n_pairs=2, model=None):
    model = model or r.MODEL
    evidence = {"evidence_id": "E01", "kegg_id": "R99991", "fused_rank": 1, "fused_score": .03,
        "bm25_rank": 1, "trained_biencoder_rank": 1, "name": "SYNTHETIC", "definition": "SYNTHETIC",
        "equation": "SYNTHETIC A => B", "enzyme": "", "rclass": "", "brite": ""}
    body = provider_payload({"model": model, "instructions": "Synthetic tests only", "input": render_input("Synthetic query", [evidence]),
        "max_output_tokens": 2048, "reasoning": {"effort": "low"}, "tools": [], "store": False})
    body_bytes, input_bound = request_input_token_bound(body, r.PRICING_BY_MODEL[model])
    unit_reservation = request_reservation_nusd(body, r.PRICING_BY_MODEL[model], input_bound)
    units, bodies, order, refs = [], {}, [], []
    for i in range(n_pairs):
        sid = f"SYNTHETIC_SAMPLE_{i}"
        refs.append({"sample_id": sid, "model_id": "SYNTHETIC_MODEL", "reaction_id": f"SYNTHETIC_RXN_{i}",
                     "cluster_id": f"SYNTHETIC_CLUSTER_{i}", "corrected_stratum": "synthetic_only",
                     "ground_truth_ids": ["R99991" if i == 0 else "R99992"], "grounded_exact": i == 0,
                     "grounded_answered": True, "grounded_abstain": False})
        for arm in ("clean", "degraded"):
            condition = "clean" if arm == "clean" else r.CONDITION
            key = r.unit_key(condition, sid, digest(body), requested_model=model)
            u = {"unit_key": key, "condition": condition, "arm": arm, "sample_id": sid, "model_id": "SYNTHETIC_MODEL",
                 "reaction_id": f"SYNTHETIC_RXN_{i}", "provider_body_digest": digest(body), "split": "validation",
                 "canonical_provider_body_bytes": body_bytes, "input_token_bound": input_bound,
                 "reservation_nusd": unit_reservation}
            units.append(u)
            bodies[key] = copy.deepcopy(body)
            order.append(key)
    for name, value in {"units.json": units, "order.json": order, "_requests/bodies.json": bodies}.items():
        r.atomic(path / name, value)
    plan = {"schema": "phase5-synthetic-plan-v1", "split": "validation", "n_units": 2*n_pairs, "n_pairs": n_pairs,
        "held_out_execution": "disabled", "runner_source_sha256": r.sha(Path(r.__file__)),
        "settings": r.settings(body), "settings_digest": digest(r.settings(body)),
        "expected_returned_models": [model], "expected_service_tiers": ["default"], "attempt_cap": 2*n_pairs,
        "reservation_nusd": unit_reservation, "dollar_cap_nusd": r.CEILING_NUSD, "endpoint": r.ENDPOINT,
        "pricing": r.PRICING_BY_MODEL[model], "source_directory": ".", "source_sha256": {}, "duplicate_policy": r.UNIT_POLICY,
        "files_sha256": {name: r.sha(path / name) for name in ("units.json", "order.json", "_requests/bodies.json")}}
    plan["plan_digest"] = digest(plan)
    r.atomic(path / "plan.json", plan)
    return path, plan, refs


def approval(plan):
    return {**r.authorization_template(plan), "status": "approved", "approved_by": "synthetic test only",
            "recorded_at": "2026-11-06T00:00:00+00:00", "approval_reference": "synthetic fixture; no paid authorization",
            "billing_assumptions_confirmed": True}


def response(body, *, abstain=False):
    ann = {"abstain": abstain, "predicted_kegg_id": None if abstain else "R99991",
           "selected_evidence_rank": None if abstain else 1, "supporting_evidence_ids": [] if abstain else ["E01"],
           "confidence": .9, "reasoning_summary": "Synthetic response only", "abstention_reason": "Synthetic abstention" if abstain else None}
    return {"id": "SYNTHETIC_RESPONSE", "model": body.get("model", r.MODEL), "service_tier": "default", "status": "completed",
            "usage": {"input_tokens": 300, "output_tokens": 100, "output_tokens_details": {"reasoning_tokens": 40}},
            "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(ann)}]}]}


def execute(package, plan, provider, **kwargs):
    return r.run(package, execute=True, auth=approval(plan), confirmation=plan["plan_digest"],
                 provider_factory=lambda: provider, synthetic=True, **kwargs)


def rewrite_plan(path, plan, **changes):
    plan = {k: v for k, v in {**plan, **changes}.items() if k != "plan_digest"}
    plan["plan_digest"] = digest(plan)
    r.atomic(path / "plan.json", plan)
    return plan


def test_gpt61_uses_separate_cache_namespace_and_revised_maximum_reservation():
    body_hash = "a" * 64
    key56 = r.unit_key("clean", "P3P0001", body_hash, requested_model=r.MODEL)
    key61 = r.unit_key("clean", "P3P0001", body_hash, requested_model=r.GPT61_MODEL)
    assert key56 != key61
    pricing = r.PRICING_BY_MODEL[r.GPT61_MODEL]
    assert (pricing["standard_input_per_million"], pricing["standard_output_per_million"],
            pricing["implicit_30m_write_per_million"]) == (1, 5, 1.25)
    body = provider_payload({"model": r.GPT61_MODEL, "instructions": "synthetic", "input": "x"*5000,
        "max_output_tokens": 2048, "reasoning": {"effort": "low"}, "tools": [], "store": False,
        "text": {"format": {"type": "json_schema", "schema": {"type": "object"}}}})
    body_bytes, bound = request_input_token_bound(body, pricing)
    reservation = request_reservation_nusd(body, pricing, bound)
    assert bound == body_bytes + 4096
    assert reservation == (bound*1.25 + 2048*5)*2*1000


def test_gpt61_request_bound_budget_covers_326_synthetic_units_and_releases_verified_excess(tmp_path):
    path, plan, _ = make_fixture(tmp_path, n_pairs=163, model=r.GPT61_MODEL)
    units = r.read(path / "units.json")
    full_reserved = sum(u["reservation_nusd"] for u in units)
    assert full_reserved < plan["dollar_cap_nusd"]
    sent = []
    summary = execute(path, plan, lambda body: (sent.append(body) or response(body)))
    ledger = r.read(path / "_synthetic_run/ledger.json")
    assert len(sent) == len(ledger["attempts"]) == 326
    assert summary["charged_ceiling_usd"] < full_reserved / 1e9
    assert all(e["charged_nusd"] < e["reserved_nusd"] for e in ledger["attempts"])
    assert sum(e["charged_nusd"] for e in ledger["attempts"]) <= plan["dollar_cap_nusd"]


def test_gpt61_uncertain_synthetic_attempt_retains_its_full_request_reservation(tmp_path):
    path, plan, _ = make_fixture(tmp_path, n_pairs=1, model=r.GPT61_MODEL)
    with pytest.raises(RuntimeError, match="reconciliation"):
        execute(path, plan, lambda body: (_ for _ in ()).throw(TimeoutError("synthetic uncertain dispatch")))
    entry = r.read(path / "_synthetic_run/ledger.json")["attempts"][0]
    assert entry["charged_nusd"] == entry["reserved_nusd"]
    assert entry["reserved_nusd"] == plan["reservation_nusd"]


def test_duplicate_bodies_are_separate_unit_attempts_and_resume_is_free(tmp_path):
    path, plan, _ = make_fixture(tmp_path)
    sent = []
    def provider(body):
        sent.append(copy.deepcopy(body))
        return response(body)
    summary = execute(path, plan, provider)
    assert len(sent) == summary["units_recorded"] == 4
    assert len({digest(b) for b in sent}) == 1
    _, units, bodies, order = r.load_package(path)
    assert sent == [bodies[key] for key in order]
    assert len({u["unit_key"] for u in units.values()}) == 4
    execute(path, plan, lambda b: pytest.fail("resume repurchased a completed unit"), resume=True)
    ledger = r.read(path / "_synthetic_run/ledger.json")
    assert len(ledger["attempts"]) == 4
    assert all(e["charged_nusd"] == 300*2500+100*12000 for e in ledger["attempts"])


def test_dry_run_has_no_provider_secret_network_or_held_out_access(tmp_path, monkeypatch):
    path, _, _ = make_fixture(tmp_path)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network"))
    monkeypatch.setattr(r, "live_provider", lambda: pytest.fail("provider construction"))
    original = Path.open
    def guarded(self, *args, **kwargs):
        assert self.suffix not in {".csv", ".lzma", ".env"} and "held_out" not in self.parts
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    summary = r.run(path)
    assert summary["paid_attempts"] == summary["provider_constructions"] == summary["held_out_rows_read"] == 0
    assert not (path / "_run").exists() and not (path / "_synthetic_run").exists()


@pytest.mark.parametrize("field,value", [("plan_digest", "stale"), ("settings_digest", "altered"), ("attempt_cap", 317), ("dollar_cap_nusd", 1), ("model", "wrong"), ("endpoint", "wrong")])
def test_authorization_must_match_every_frozen_constraint(tmp_path, field, value):
    path, plan, _ = make_fixture(tmp_path)
    auth = {**approval(plan), field: value}
    with pytest.raises(PermissionError):
        r.run(path, execute=True, auth=auth, confirmation=plan["plan_digest"], synthetic=True,
              provider_factory=lambda: pytest.fail("provider touched"))
    assert not (path / "_synthetic_run").exists()


def test_current_zero_authorization_cannot_dispatch(tmp_path):
    path, plan, _ = make_fixture(tmp_path)
    with pytest.raises(PermissionError):
        r.run(path, execute=True, auth=r.authorization_template(plan), confirmation=plan["plan_digest"], synthetic=True)


def test_reservation_is_persisted_before_any_provider_construction(tmp_path):
    path, plan, _ = make_fixture(tmp_path)
    def factory():
        ledger = r.read(path / "_synthetic_run/ledger.json")
        assert len(ledger["attempts"]) == 1
        assert ledger["attempts"][0]["status"] == "reserved"
        assert ledger["attempts"][0]["reserved_nusd"] == plan["reservation_nusd"]
        raise KeyboardInterrupt("synthetic process loss before dispatch")
    with pytest.raises(KeyboardInterrupt):
        r.run(path, execute=True, auth=approval(plan), confirmation=plan["plan_digest"], synthetic=True, provider_factory=factory)
    with pytest.raises(RuntimeError, match="uncertain"):
        execute(path, plan, lambda b: pytest.fail("repurchased unknown attempt"), resume=True)
    assert len(r.read(path / "_synthetic_run/ledger.json")["attempts"]) == 1


def test_interruption_during_provider_requires_documented_reconciliation(tmp_path):
    path, plan, _ = make_fixture(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        execute(path, plan, lambda b: (_ for _ in ()).throw(KeyboardInterrupt()))
    key = r.read(path / "order.json")[0]
    r.reconcile(path, {"plan_digest": plan["plan_digest"], "unit_key": key, "operator": "synthetic test",
                       "reason": "provider result unavailable", "evidence_reference": "synthetic fault injection",
                       "decision": "retain_reservation_no_retry"}, synthetic=True)
    sent = []
    execute(path, plan, lambda b: (sent.append(b) or response(b)), resume=True)
    assert len(sent) == 3
    entry = r.read(path / "_synthetic_run/ledger.json")["attempts"][0]
    assert entry["charged_nusd"] == plan["reservation_nusd"] and entry["result_status"] == "interrupted_unavailable"


def test_resume_recovers_cache_written_before_ledger_completion(tmp_path, monkeypatch):
    path, plan, _ = make_fixture(tmp_path)
    original = r.atomic
    def crash_after_cache(pathname, obj):
        if pathname.name == "ledger.json" and obj["attempts"] and obj["attempts"][0]["status"] == "completed":
            raise KeyboardInterrupt("synthetic crash between durable cache and ledger completion")
        original(pathname, obj)
    monkeypatch.setattr(r, "atomic", crash_after_cache)
    with pytest.raises(KeyboardInterrupt):
        execute(path, plan, response)
    monkeypatch.setattr(r, "atomic", original)
    sent = []
    execute(path, plan, lambda b: (sent.append(b) or response(b)), resume=True)
    assert len(sent) == 3


@pytest.mark.parametrize("kind", ["provider", "missing_usage", "model", "tier", "usage_bound"])
def test_uncertain_or_mismatched_results_pause_and_retain_reservation(tmp_path, kind):
    path, plan, _ = make_fixture(tmp_path)
    def bad(body):
        if kind == "provider":
            raise TimeoutError("synthetic")
        result = response(body)
        if kind == "missing_usage": result["usage"] = None
        if kind == "model": result["model"] = "SYNTHETIC_OTHER_SNAPSHOT"
        if kind == "tier": result["service_tier"] = "priority"
        if kind == "usage_bound": result["usage"]["input_tokens"] = 922001
        return result
    with pytest.raises(RuntimeError, match="reconciliation"):
        execute(path, plan, bad)
    entry = r.read(path / "_synthetic_run/ledger.json")["attempts"][0]
    assert entry["charged_nusd"] == plan["reservation_nusd"]
    with pytest.raises(RuntimeError, match="reconciliation"):
        execute(path, plan, lambda b: pytest.fail("unexpected call"), resume=True)


@pytest.mark.parametrize("kind", ["schema", "compliance"])
def test_schema_and_compliance_failures_remain_terminal_denominator_units(tmp_path, kind):
    path, plan, refs = make_fixture(tmp_path)
    def bad(body):
        result = response(body)
        text = json.loads(result["output"][0]["content"][0]["text"])
        if kind == "schema": text["extra"] = "forbidden"
        else: text["predicted_kegg_id"] = "R99992"
        result["output"][0]["content"][0]["text"] = json.dumps(text)
        return result
    execute(path, plan, bad)
    execute(path, plan, lambda b: pytest.fail("failed unit repurchased"), resume=True)
    frozen = path / "_analysis/responses.json"
    a.freeze(path, frozen, synthetic=True)
    report = a.analyze(path, frozen, refs, synthetic=True)
    assert all(v["n"] == v["failures"] == 2 and v["correct"] == 0 for v in report["arms"].values())
    assert all(e["charged_nusd"] == plan["reservation_nusd"] for e in r.read(path / "_synthetic_run/ledger.json")["attempts"])


@pytest.mark.parametrize("cap", ["attempt", "dollar"])
def test_caps_stop_before_dispatch_across_resumes(tmp_path, cap):
    path, plan, _ = make_fixture(tmp_path)
    plan = rewrite_plan(path, plan, **({"attempt_cap": 1} if cap == "attempt" else {"dollar_cap_nusd": plan["reservation_nusd"]-1}))
    calls = []
    with pytest.raises(RuntimeError, match="cap"):
        execute(path, plan, lambda b: (calls.append(b) or response(b)))
    assert len(calls) == (1 if cap == "attempt" else 0)
    with pytest.raises(RuntimeError, match="cap"):
        execute(path, plan, lambda b: pytest.fail("cap bypass"), resume=True)


def test_exclusive_plan_lock_excludes_another_process(tmp_path):
    lock = tmp_path / "_execution.lock"
    code = "from pathlib import Path\nfrom benchmark.scripts.phase5_runner import exclusive\nimport sys\ntry:\n with exclusive(Path(sys.argv[1])): pass\nexcept RuntimeError:\n sys.exit(42)\n"
    with r.exclusive(lock):
        proc = subprocess.run([sys.executable, "-c", code, str(lock)], cwd=r.ROOT, capture_output=True)
        assert proc.returncode == 42, proc.stderr
    with r.exclusive(lock):
        pass  # released by OS, no stale file deletion required


def test_stale_artifacts_and_incompatible_caches_fail_closed(tmp_path):
    path, plan, _ = make_fixture(tmp_path)
    execute(path, plan, response)
    key = r.read(path / "order.json")[0]
    cache = path / "_synthetic_run/cache" / (key + ".json")
    envelope = r.read(cache)
    envelope["unit"]["sample_id"] = "ANOTHER_SYNTHETIC_UNIT"
    r.atomic(cache, envelope)
    with pytest.raises(RuntimeError, match="incompatible"):
        execute(path, plan, lambda b: pytest.fail("cache mismatch repurchased"), resume=True)
    r.atomic(path / "order.json", [])
    with pytest.raises(ValueError, match="altered locked artifact"):
        r.run(path)


def test_lost_reservation_ledger_cannot_repurchase_an_orphaned_cache(tmp_path):
    path, plan, _ = make_fixture(tmp_path)
    execute(path, plan, response)
    ledger_path = path / "_synthetic_run/ledger.json"
    ledger = r.read(ledger_path)
    ledger["attempts"].pop()
    r.atomic(ledger_path, ledger)
    with pytest.raises(RuntimeError, match="unledgered"):
        execute(path, plan, lambda b: pytest.fail("repurchase after ledger loss"), resume=True)


def test_source_plan_change_invalidates_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    path, plan, _ = make_fixture(tmp_path)
    source = path / "source.json"
    r.atomic(source, {"synthetic": "original"})
    plan = rewrite_plan(path, plan, source_directory=path.relative_to(r.ROOT).as_posix(), source_sha256={"source.json": r.sha(source)})
    r.atomic(source, {"synthetic": "changed"})
    with pytest.raises(ValueError, match="stale source"):
        r.run(path)


def test_ledger_cost_cannot_disagree_with_cached_result(tmp_path):
    path, plan, _ = make_fixture(tmp_path)
    execute(path, plan, response)
    ledger_path = path / "_synthetic_run/ledger.json"
    ledger = r.read(ledger_path)
    ledger["attempts"][0]["charged_nusd"] = 0
    r.atomic(ledger_path, ledger)
    with pytest.raises(RuntimeError, match="cost mismatch"):
        execute(path, plan, lambda b: pytest.fail("altered ledger cost"), resume=True)


def test_input_and_output_bound_counts_reasoning_once():
    pathless = {"expected_returned_models": [r.MODEL], "expected_service_tiers": ["default"], "pricing": r.PRICING, "reservation_nusd": r.RESERVATION_NUSD}
    body = {"max_output_tokens": 2048, "input": "TARGET\n{}\nEVIDENCE\n[]"}
    raw = response(body)
    raw["usage"] = {"input_tokens": 922000, "output_tokens": 2048, "output_tokens_details": {"reasoning_tokens": 2048}}
    raw["service_tier"] = "priority"
    result = r.assess(raw, body, pathless)
    assert result["accounted_usage_ceiling_nusd"] == r.RESERVATION_NUSD
    assert r.RESERVATION_NUSD == 9_293_728_000


def test_live_adapter_sends_exact_body_and_disables_automatic_retries(monkeypatch, tmp_path):
    import httpx
    import openai
    path, _, _ = make_fixture(tmp_path)
    body = next(iter(r.read(path / "_requests/bodies.json").values()))
    captured, constructed = [], []
    def transport(request):
        captured.append(json.loads(request.content))
        assert str(request.url) == r.ENDPOINT + "/responses"
        return httpx.Response(200, json=response(body))
    original = openai.OpenAI
    def factory(**kwargs):
        constructed.append(kwargs)
        return original(**kwargs, api_key="SYNTHETIC_NOT_A_CREDENTIAL", http_client=httpx.Client(transport=httpx.MockTransport(transport)))
    monkeypatch.setattr(openai, "OpenAI", factory)
    call = r.live_provider()
    call(body)
    assert captured == [body]
    assert constructed[0]["max_retries"] == 0 and constructed[0]["base_url"] == r.ENDPOINT


def test_analysis_freezes_before_labels_and_uses_paired_denominators(tmp_path):
    path, plan, refs = make_fixture(tmp_path)
    execute(path, plan, response)
    frozen = path / "_analysis/responses.json"
    a.freeze(path, frozen, synthetic=True)
    def labels():
        assert r.read(frozen)["reference_rows_read"] == 0
        return refs
    report = a.analyze(path, frozen, labels, synthetic=True)
    assert report["n_pairs"] == 2
    assert report["arms"]["clean"]["accuracy"] == .5
    assert report["arms"]["degraded"]["coverage"] == 1
    assert report["paired_uncertainty"]["changes"]["accuracy"]["ci95"] == [0, 0]
    assert report["paired_uncertainty"] == a.paired_bootstrap(report["case_results"])
    bad = r.read(frozen)
    bad["responses"].pop()
    r.atomic(frozen, bad)
    with pytest.raises(ValueError, match="freeze"):
        a.analyze(path, frozen, lambda: pytest.fail("labels read before valid freeze"), synthetic=True)


def test_incomplete_run_cannot_freeze_or_read_references(tmp_path):
    path, plan, _ = make_fixture(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        execute(path, plan, lambda b: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(ValueError, match="all planned"):
        a.freeze(path, path / "_analysis/responses.json", synthetic=True)


def test_abstentions_and_zero_coverage_bootstrap_draws_are_explicit(tmp_path):
    path, plan, refs = make_fixture(tmp_path)
    flags = iter([False, True, True, False])
    execute(path, plan, lambda b: response(b, abstain=next(flags)))
    frozen = path / "_analysis/responses.json"
    a.freeze(path, frozen, synthetic=True)
    report = a.analyze(path, frozen, refs, synthetic=True)
    assert report["arms"]["clean"]["n"] == report["arms"]["degraded"]["n"] == 2
    assert report["arms"]["clean"]["appropriate_reference_based_abstentions"] == 1
    assert report["arms"]["degraded"]["retrievable_reference_abstentions"] == 1
    assert report["arms"]["clean"]["selective_accuracy"] == 1
    assert report["arms"]["degraded"]["selective_accuracy"] == 0
    assert report["paired_uncertainty"]["changes"]["selective_accuracy"]["defined_bootstrap_draws"] < 2000


def test_real_326_unit_package_is_reproducible_without_network(tmp_path, monkeypatch):
    source = r.ROOT / "benchmark/phase5/validation/matched_prompt_v2"
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network during lock/dry-run"))
    one, two = tmp_path / "one", tmp_path / "two"
    p1, p2 = r.build_package(source, one), r.build_package(source, two)
    assert p1["plan_digest"] == p2["plan_digest"]
    assert r.state_paths(one, p1) == r.state_paths(two, p2)  # copies cannot repurchase units
    assert r.run(one)["planned_units"] == 326 and r.run(two)["distinct_provider_bodies"] == 317
    groups = r.read(one / "duplicate_bodies.json")["groups"]
    assert sum(len(g["units"])-1 for g in groups) == 9
    assert all(len({u["unit_key"] for u in g["units"]}) == len(g["units"]) for g in groups)
    source_paths = p1["source_sha256"]
    assert all(r.sha(source / p) == h for p, h in source_paths.items())
