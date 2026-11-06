"""Synthetic-only tests of the draft freeze's fail-closed runner semantics."""
import copy
import socket
from pathlib import Path

import pytest

from benchmark.scripts import phase4_freeze as freeze


def test_dry_run_uses_no_network_or_held_out_files(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected network or file access")
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    result = freeze.synthetic_dry_run()
    assert result["provider_network_calls"] == result["test_files_opened"] == 0
    assert result["injected_fake_provider_calls"] == 1
    assert result["cache_replay_identical"]


def test_execution_fails_before_opening_any_data(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsealed input")
    monkeypatch.setattr(Path, "read_text", forbidden)
    with pytest.raises(SystemExit, match="execution disabled"):
        freeze.main(["--execute", "--freeze", "SYNTHETIC_MUST_NOT_OPEN.json"])


def test_budget_stops_before_call_and_keeps_failed_attempt_reservations():
    budget = freeze.Budget(.1, 1)
    cache, calls = {}, []
    def fail(payload):
        calls.append(payload)
        raise RuntimeError("synthetic failure")
    payload = {"synthetic": True}
    result = freeze.replay_or_attempt(payload, [], budget, cache, fail)
    assert result["status"] == "failed" and result["abstain"]
    assert freeze.replay_or_attempt(payload, [], budget, cache, fail) == result
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="cap"):
        freeze.replay_or_attempt({"synthetic": "second"}, [], budget, cache, fail)
    assert len(calls) == 1
    assert budget.reserved_usd == pytest.approx(.090112)


@pytest.mark.parametrize("cap", [0.0, -.1, float("nan"), float("inf")])
def test_invalid_or_unfunded_budget_never_calls_provider(cap):
    def forbidden(*args):
        raise AssertionError("provider must not be called")
    with pytest.raises((ValueError, RuntimeError)):
        freeze.replay_or_attempt({"synthetic": True}, [], freeze.Budget(cap, 1), {}, forbidden)


def test_cache_key_changes_with_prompt_model_and_routing_payload():
    first = {"model": "synthetic", "input": "test", "max_output_tokens": 20}
    assert freeze.canonical_digest(first) == freeze.canonical_digest(dict(reversed(list(first.items()))))
    for key in first:
        altered = {**first, key: "changed"}
        assert freeze.canonical_digest(first) != freeze.canonical_digest(altered)


def test_query_leakage_and_oversize_payload_rejected():
    with pytest.raises(ValueError, match="identifier"):
        freeze.request_payload("synthetic reaction r12345", [])
    with pytest.raises(ValueError, match="byte limit"):
        freeze.request_payload("synthetic" * 10000, [])
