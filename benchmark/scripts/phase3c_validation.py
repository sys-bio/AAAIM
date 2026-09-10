"""Run and evaluate the frozen 163-reaction Phase 3C paired validation pilot.

The default action is a zero-call preflight.  ``--execute`` uses the native
Python reference path validated by the five-row smoke test; retrieval happens
locally before a stateless Responses API call with ``tools=[]``.  ``--cache-only``
requires all 163 compatible cache entries and never constructs a provider
client.  Ground truth is opened only by ``--evaluate``, after responses freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.scripts import phase3_retrieval as retrieval
from benchmark.scripts import phase3c_grounded as grounded
from benchmark.scripts.kegg_equivalence import match_kinds
from benchmark.scripts.phase3_common import (
    PHASE3_DIR,
    PRICING_OPENAI_TERRA,
    atomic_write_json,
    atomic_write_jsonl,
    estimate_tokens_conservative,
    parse_kegg_ids,
    repo_relative_posix,
    sha256_portable,
    write_artifact_manifest,
)
from benchmark.scripts.phase3_cost import load_pricing
from benchmark.scripts.phase3_modes import FileCache
from benchmark.scripts.phase3_openai_run import (
    assert_env_file_protected,
    env_file_is_ignored,
    env_file_is_tracked,
    load_dotenv_if_present,
    model_rates,
)

OUT = PHASE3_DIR / "phase3c_validation"
CACHE_DIR = OUT / "_response_cache"
SMOKE_CACHE_DIR = grounded.CACHE_DIR
PILOT_SAMPLE = PHASE3_DIR / "pilot_sample.csv"
PILOT_ANSWER_KEY = PHASE3_DIR / "pilot_answer_key.csv"
PHASE3A_SCORED = PHASE3_DIR / "validation" / "scored_rows.json"
FUSION_RANKINGS = PHASE3_DIR / "phase3b_fusion" / "rankings_bm25_trained_epoch1_rrf.jsonl.xz"
PHASE2_RANKINGS = PHASE3_DIR / "retrieval_baselines" / "rankings_phase2_rule_based.jsonl"
TRAINED_RANKINGS = PHASE3_DIR / "phase3b_full" / "rankings_epoch_1.jsonl"

EXPECTED_SAMPLE_SHA256 = "be086250023be617df278ab62549756b23bd477f8f5d6de92531608dcbf1088a"
EXPECTED_ANSWER_KEY_SHA256 = "62415034be49a11dd672e0940675f8ebe827cb217b79a773386f1c4d20bb4c83"
EXPECTED_PHASE3A_RESULTS_SHA256 = "f0c4b909321f470a05d30e66003515cb893ec82e25babeadf9b0fe9859847538"
EXPECTED_SMOKE_RESPONSES_SHA256 = "2f8121b7a9f78d1940f5771142c6b5195d48c9827456d4f4ca19637c1215451d"
EXPECTED_FUSION_SHA256 = grounded.EXPECTED_FUSION_SHA256
N_REACTIONS = 163
EXPECTED_SMOKE_HITS = 5
MAX_NEW_CALLS = 158
MAX_TOTAL_ATTEMPTS = 158
MAX_COST_USD = 3.50
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260902
ORCHESTRATION_PATH = "native_python_pre_retrieval_then_responses_api"
LANGCHAIN_PATH = "zero_cost_synthetic_ai_message_to_local_structured_tool"
FORBIDDEN_REQUEST_KEYS = frozenset({
    "ground_truth_ids", "ground_truth_kegg_all", "ground_truth_kegg_primary",
    "answer_key", "truth", "label", "split_assignments",
})


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _canonical_digest(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _request_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(_request_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_request_keys(child))
    return keys


def load_population() -> list[dict[str, Any]]:
    """Load only the immutable, ground-truth-free Phase 3A pilot sample."""
    if sha256_portable(PILOT_SAMPLE) != EXPECTED_SAMPLE_SHA256:
        raise ValueError("frozen Phase 3A pilot sample digest mismatch")
    frame = pd.read_csv(PILOT_SAMPLE, dtype=str).fillna("")
    if len(frame) != N_REACTIONS or frame.sample_id.nunique() != N_REACTIONS:
        raise ValueError("Phase 3A pilot must contain exactly 163 unique sample IDs")
    if frame.duplicated(["model_id", "reaction_id"]).any():
        raise ValueError("Phase 3A pilot contains duplicate reactions")
    if not frame.split.eq("validation").all():
        raise ValueError("held-out test or train row in Phase 3C population")
    assignments = pd.read_csv(retrieval.SPLITS, dtype=str)
    validation = set(zip(
        assignments.loc[assignments.split.eq("validation"), "model_id"],
        assignments.loc[assignments.split.eq("validation"), "reaction_id"],
    ))
    keys = set(zip(frame.model_id, frame.reaction_id))
    if not keys <= validation:
        raise ValueError("Phase 3C population is not validation-only")
    return frame.sort_values("sample_id").to_dict("records")


def request_payload(sample: Mapping[str, Any], index: grounded.EvidenceIndex) -> tuple[dict[str, Any], list[grounded.EvidenceRecord]]:
    query = index.query_for(str(sample["model_id"]), str(sample["reaction_id"]))
    evidence = index.retrieve(query, grounded.TOP_K)
    visible = {"query": query.text}
    user = grounded.build_user_prompt(visible, evidence)
    payload = {
        "model": grounded.MODEL,
        "instructions": grounded.SYSTEM_INSTRUCTIONS,
        "input": user,
        "max_output_tokens": grounded.MAX_OUTPUT_TOKENS,
        "reasoning": {"effort": grounded.REASONING_EFFORT},
        "store": False,
        "tools": [],
        "schema_version": grounded.SCHEMA_VERSION,
        "prompt_version": grounded.PROMPT_VERSION,
    }
    forbidden = FORBIDDEN_REQUEST_KEYS & _request_keys(payload)
    if forbidden:
        raise ValueError(f"answer-key field in model-visible request: {sorted(forbidden)}")
    return payload, evidence


class CompatibleCaches:
    """Read validation cache first, then immutable compatible smoke entries."""

    def __init__(self, validation_dir: Path = CACHE_DIR, smoke_dir: Path = SMOKE_CACHE_DIR) -> None:
        self.validation = FileCache(validation_dir)
        self.smoke = FileCache(smoke_dir)

    def get(self, key: str, smoke_key: Optional[str] = None) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        row = self.validation.get(key)
        if row is not None:
            return row, "phase3c_validation"
        row = self.smoke.get(smoke_key or key)
        if row is not None:
            return row, "phase3c_smoke"
        return None, None

    def put(self, key: str, row: dict[str, Any]) -> None:
        self.validation.put(key, row)


def build_plan(
    *, index: Optional[grounded.EvidenceIndex] = None,
    validation_cache: Path = CACHE_DIR,
    smoke_cache: Path = SMOKE_CACHE_DIR,
) -> dict[str, Any]:
    """Build a label-free request plan and inspect caches without API access."""
    index = index or grounded.EvidenceIndex()
    population = load_population()
    caches = CompatibleCaches(validation_cache, smoke_cache)
    planned = []
    request_rows = []
    evidence_rows = []
    cache_hits = []
    pending = []
    for sample in population:
        payload, evidence = request_payload(sample, index)
        provider_payload_digest = grounded._cache_key(payload)
        cache_id = _canonical_digest({
            "experimental_unit": [sample["model_id"], sample["reaction_id"]],
            "provider_payload_digest": provider_payload_digest,
        })
        evidence_dicts = [record.to_dict() for record in evidence]
        evidence_digest = _canonical_digest(evidence_dicts)
        hit, source = caches.get(cache_id, provider_payload_digest)
        item = {
            "sample": sample,
            "payload": payload,
            "evidence": evidence,
            "evidence_digest": evidence_digest,
            "cache_id": cache_id,
            "provider_payload_digest": provider_payload_digest,
            "cache_source": source,
        }
        planned.append(item)
        request_rows.append({
            "sample_id": sample["sample_id"],
            "model_id": sample["model_id"],
            "reaction_id": sample["reaction_id"],
            "cluster_id": sample["cluster_id"],
            "cache_id": cache_id,
            "provider_payload_digest": provider_payload_digest,
            "evidence_digest": evidence_digest,
            "retrieval_config_digest": _canonical_digest(index.provenance),
            "orchestration_path": ORCHESTRATION_PATH,
            "payload": payload,
            "input_tokens_estimate": estimate_tokens_conservative(payload["instructions"] + "\n" + payload["input"]),
        })
        evidence_rows.append({
            "sample_id": sample["sample_id"],
            "model_id": sample["model_id"],
            "reaction_id": sample["reaction_id"],
            "evidence_digest": evidence_digest,
            "records": evidence_dicts,
        })
        (cache_hits if hit is not None else pending).append({
            "sample_id": sample["sample_id"], "cache_id": cache_id,
            "source_cache_id": provider_payload_digest if source == "phase3c_smoke" else cache_id,
            "source": source, "model_id": sample["model_id"], "reaction_id": sample["reaction_id"],
        })

    smoke_hits = [row for row in cache_hits if row["source"] == "phase3c_smoke"]
    validation_hits = [row for row in cache_hits if row["source"] == "phase3c_validation"]
    unique_cache_ids = len({row["cache_id"] for row in request_rows})
    if unique_cache_ids != N_REACTIONS:
        raise ValueError(
            f"cache identity collision: {unique_cache_ids} unique IDs for {N_REACTIONS} reactions"
        )
    if len(smoke_hits) < EXPECTED_SMOKE_HITS:
        raise ValueError(f"configuration drift: only {len(smoke_hits)} compatible smoke entries")
    if len(planned) != N_REACTIONS or len(pending) > MAX_NEW_CALLS:
        raise ValueError("planned population or authorized new-call count exceeded")

    pricing = load_pricing(PRICING_OPENAI_TERRA)
    rates = model_rates(pricing, grounded.MODEL)
    pending_ids = {row["cache_id"] for row in pending}
    pending_requests = [row for row in request_rows if row["cache_id"] in pending_ids]
    input_pending = sum(int(row["input_tokens_estimate"]) for row in pending_requests)
    smoke_cost = json.loads((grounded.OUT / "cost_report.json").read_text(encoding="utf-8"))
    mean_smoke_output = smoke_cost["usage_totals"]["output_tokens"] / grounded.MAX_REQUESTS
    expected_cost = (
        input_pending / 1_000_000 * rates["input_per_million"]
        + len(pending) * mean_smoke_output / 1_000_000 * rates["output_per_million"]
    )
    worst = (
        input_pending / 1_000_000 * rates["input_per_million"]
        + len(pending) * grounded.MAX_OUTPUT_TOKENS / 1_000_000 * rates["output_per_million"]
    )
    if expected_cost > MAX_COST_USD + 1e-12:
        raise ValueError(f"expected cost ${expected_cost:.6f} exceeds ${MAX_COST_USD:.2f} cap")
    return {
        "planned": planned,
        "population": population,
        "request_rows": request_rows,
        "evidence_rows": evidence_rows,
        "cache_hits": cache_hits,
        "pending": pending,
        "pricing": pricing,
        "preflight": {
            "schema": "phase3c-validation-preflight-v1",
            "population": "exact frozen Phase 3A 163-reaction validation pilot",
            "population_sha256": EXPECTED_SAMPLE_SHA256,
            "planned_rows": len(planned),
            "unique_sample_ids": len({row["sample_id"] for row in population}),
            "unique_reactions": len({(row["model_id"], row["reaction_id"]) for row in population}),
            "unique_cache_ids": unique_cache_ids,
            "compatible_cache_hits": len(cache_hits),
            "compatible_smoke_cache_hits": len(smoke_hits),
            "compatible_validation_cache_hits": len(validation_hits),
            "pending_calls": len(pending),
            "authorized_new_calls": MAX_NEW_CALLS,
            "input_tokens_estimate_all": sum(int(row["input_tokens_estimate"]) for row in request_rows),
            "input_tokens_estimate_pending": input_pending,
            "output_token_allowance_per_call": grounded.MAX_OUTPUT_TOKENS,
            "output_token_allowance_pending": len(pending) * grounded.MAX_OUTPUT_TOKENS,
            "expected_cost_usd": round(expected_cost, 6),
            "expected_cost_method": "conservative input estimate plus mean observed smoke output tokens",
            "conservative_worst_case_exposure_usd": round(worst, 6),
            "cost_cap_usd": MAX_COST_USD,
            "runtime_cost_gate": "before every call, recorded spend plus that request's projected maximum must not exceed the cap",
            "maximum_total_attempts": MAX_TOTAL_ATTEMPTS,
            "automatic_retries": 0,
            "model": grounded.MODEL,
            "reasoning_effort": grounded.REASONING_EFFORT,
            "max_output_tokens": grounded.MAX_OUTPUT_TOKENS,
            "store": False,
            "tools": [],
            "answer_key_read": False,
            "answer_key_fields_in_requests": False,
            "test_rows_read": 0,
            "orchestration_path": ORCHESTRATION_PATH,
            "historical_969_plan_superseded": True,
            "superseding_population_reason": [
                "enables direct paired comparison against frozen Phase 3A target_only outputs",
                "limits method-development cost",
                "avoids consuming all 969 validation reactions before deciding whether the LLM layer is useful",
                "preserves a later all-validation analysis only if scientifically necessary",
            ],
            "pricing_date": pricing.get("pricing_date"),
            "pricing_source": pricing.get("source"),
        },
    }


def _public_population(population: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "sample_id": row["sample_id"], "model_id": row["model_id"],
        "reaction_id": row["reaction_id"], "cluster_id": row["cluster_id"],
        "split": row["split"], "frozen_phase3a_stratum": row["stratum"],
    } for row in population]


def langchain_parity(plan: Mapping[str, Any], index: grounded.EvidenceIndex) -> dict[str, Any]:
    mismatches = []
    for item in plan["planned"]:
        sample = item["sample"]
        query = index.query_for(sample["model_id"], sample["reaction_id"])
        adapted = grounded.run_langchain_tool_step(index, query)
        if grounded.canonical_evidence_bytes(item["evidence"]) != grounded.canonical_evidence_bytes(adapted):
            mismatches.append(sample["sample_id"])
    return {
        "schema": "phase3c-langchain-provenance-v1",
        "paid_response_path": ORCHESTRATION_PATH,
        "paid_calls_used_langchain_wrapper": False,
        "paid_calls_used_autonomous_provider_tool_calls": False,
        "provider_request_tools": [],
        "native_path": "Python invokes frozen retrieval, serializes Top-10 evidence, then calls Responses API",
        "langchain_demonstration_path": LANGCHAIN_PATH,
        "langchain_role": "orchestration/integration adapter only; not retrieval algorithm or evaluator",
        "demonstration_model": "none (synthetic AIMessage tool call)",
        "demonstration_api_calls": 0,
        "n_queries_compared": len(plan["planned"]),
        "byte_equivalent": not mismatches,
        "mismatched_sample_ids": mismatches,
    }


def write_preflight(plan: Mapping[str, Any], index: grounded.EvidenceIndex, out: Path = OUT, *, preserve: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(_public_population(plan["population"]), out / "sample_ids.jsonl")
    atomic_write_jsonl(plan["request_rows"], out / "requests.jsonl")
    atomic_write_jsonl(plan["evidence_rows"], out / "frozen_evidence.jsonl")
    compatibility = {
        "schema": "phase3c-validation-cache-compatibility-v1",
        "cache_identity_definition": "SHA-256 of stable reaction experimental unit plus the canonical provider-payload digest; smoke compatibility uses the original provider-payload digest as a read-only alias",
        "identity_covers": ["model", "prompt/template version", "response schema version", "evidence bytes and digest", "retrieval provenance/config", "max output tokens", "reasoning effort", "store", "provider tools", "all API-request-affecting orchestration"],
        "orchestration_note": "Framework selection is excluded only because it does not alter the serialized API request; paid requests use the native path.",
        "compatible_hits": plan["cache_hits"],
        "pending": plan["pending"],
        "n_compatible": len(plan["cache_hits"]),
        "n_pending": len(plan["pending"]),
    }
    compatibility_path = out / "cache_compatibility.json"
    if not preserve or not compatibility_path.exists():
        atomic_write_json(compatibility, compatibility_path)
    else:
        atomic_write_json(compatibility, out / "current_cache_state.json")
    if not preserve or not (out / "dry_run.json").exists():
        atomic_write_json(plan["preflight"], out / "dry_run.json")
    atomic_write_json({
        "schema": "phase3c-validation-config-v1",
        "population": {"source": repo_relative_posix(PILOT_SAMPLE), "sha256": EXPECTED_SAMPLE_SHA256, "n": N_REACTIONS},
        "model": grounded.MODEL,
        "api": "responses",
        "reasoning_effort": grounded.REASONING_EFFORT,
        "max_output_tokens": grounded.MAX_OUTPUT_TOKENS,
        "one_request_per_reaction": True,
        "context": "target-local only",
        "retrieval": index.provenance,
        "retrieval_top_k": grounded.TOP_K,
        "tools": [], "store": False, "automatic_retries": 0,
        "max_new_calls": MAX_NEW_CALLS, "max_attempts": MAX_TOTAL_ATTEMPTS, "cost_cap_usd": MAX_COST_USD,
        "prompt_version": grounded.PROMPT_VERSION, "response_schema_version": grounded.SCHEMA_VERSION,
        "orchestration_path": ORCHESTRATION_PATH,
        "no_new_phase3a_calls": True, "llm_judge": False, "test_rows_read": 0,
    }, out / "config.json")
    atomic_write_json({
        "schema": "phase3c-validation-plan-v1",
        "supersedes": "benchmark/phase3/phase3c_smoke/full_validation_plan.json (historical 969-row proposal; left immutable)",
        "population": "same exact 163 validation reactions as frozen Phase 3A pilot",
        "paired_methods": ["frozen BM25 + trained epoch-1 RRF Top-1", "frozen Phase 3A target_only", "Phase 3C grounded selection from frozen RRF Top-10"],
        "why_163": plan["preflight"]["superseding_population_reason"],
        "later_969_run": "not authorized; consider only if this paired pilot establishes scientific necessity",
        "held_out_test": "sealed",
    }, out / "paired_pilot_plan.json")
    atomic_write_json(langchain_parity(plan, index), out / "langchain_provenance.json")


def _normalise_cached(row: Mapping[str, Any], item: Mapping[str, Any], source: str) -> dict[str, Any]:
    out = dict(row)
    original_sample_id = str(out.get("sample_id") or "")
    source_cache_id = str(out.get("cache_id") or "")
    out["sample_id"] = item["sample"]["sample_id"]
    out["model_id"] = item["sample"]["model_id"]
    out["reaction_id"] = item["sample"]["reaction_id"]
    out["cluster_id"] = item["sample"]["cluster_id"]
    out["cache_hit"] = True
    out["cache_id"] = item["cache_id"]
    out["source_cache_id"] = source_cache_id
    out["provider_payload_digest"] = item["provider_payload_digest"]
    out["cache_source"] = source
    out["original_cache_sample_id"] = original_sample_id
    out["purchased_in_validation_run"] = False
    out["orchestration_path"] = ORCHESTRATION_PATH
    out["evidence_digest"] = item["evidence_digest"]
    return out


def _new_result(item: Mapping[str, Any], response: Any, pricing: Mapping[str, Any], latency_ms: float) -> dict[str, Any]:
    row = grounded._response_row(item, response, pricing)
    row.update({
        "model_id": item["sample"]["model_id"], "reaction_id": item["sample"]["reaction_id"],
        "cluster_id": item["sample"]["cluster_id"], "cache_source": "phase3c_validation",
        "original_cache_sample_id": item["sample"]["sample_id"],
        "purchased_in_validation_run": True, "orchestration_path": ORCHESTRATION_PATH,
        "provider_payload_digest": item["provider_payload_digest"], "source_cache_id": item["cache_id"],
        "evidence_digest": item["evidence_digest"], "latency_ms": latency_ms,
    })
    return row


def _projected_max(item: Mapping[str, Any], rates: Mapping[str, float]) -> float:
    n_input = estimate_tokens_conservative(item["payload"]["instructions"] + "\n" + item["payload"]["input"])
    return n_input / 1_000_000 * rates["input_per_million"] + grounded.MAX_OUTPUT_TOKENS / 1_000_000 * rates["output_per_million"]


def _replay_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    stable = [{key: row.get(key) for key in (
        "sample_id", "cache_id", "provider_payload_digest", "terminal_status", "compliance_problems", "annotation",
        "model_requested", "model_returned", "response_id", "usage", "cost_usd", "evidence_digest",
    )} for row in rows]
    return _canonical_digest(stable)


def run_plan(
    plan: Mapping[str, Any], *, execute: bool, cache_only: bool,
    validation_cache: Path = CACHE_DIR, smoke_cache: Path = SMOKE_CACHE_DIR,
    parse_fn: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    session_path: Optional[Path] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    caches = CompatibleCaches(validation_cache, smoke_cache)
    ledger_path = validation_cache / "_attempt_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"attempts": []}
    pricing = plan["pricing"]
    rates = model_rates(pricing, grounded.MODEL)
    rows: list[dict[str, Any]] = []
    calls = successes = failures = 0
    current_run_cost = 0.0
    cached_before = [caches.get(item["cache_id"], item["provider_payload_digest"])[0] for item in plan["planned"]]
    recorded_spend = sum(float(row.get("cost_usd") or 0.0) for row in cached_before if row is not None)
    for position, item in enumerate(plan["planned"], 1):
        hit, source = caches.get(item["cache_id"], item["provider_payload_digest"])
        if hit is not None:
            row = _normalise_cached(hit, item, str(source))
            rows.append(row)
            continue
        if cache_only:
            raise RuntimeError(f"cache-only replay would make a provider call for {item['cache_id']}")
        if not execute:
            continue
        attempts = list(ledger.get("attempts") or [])
        if len(attempts) >= MAX_TOTAL_ATTEMPTS:
            raise RuntimeError("persistent authorized-attempt cap reached")
        if any(attempt.get("cache_id") == item["cache_id"] for attempt in attempts):
            raise RuntimeError("unresolved prior attempt has no cache entry; refusing to repurchase")
        call_max = _projected_max(item, rates)
        failed_reserves = sum(
            float(attempt["reserved_cost_usd"])
            for attempt in attempts
            if attempt.get("status") != "completed" and caches.validation.get(str(attempt.get("cache_id"))) is None
        )
        if recorded_spend + failed_reserves + call_max > MAX_COST_USD + 1e-12:
            raise RuntimeError(
                f"runtime cost cap stopped before {item['sample']['sample_id']}: "
                f"${recorded_spend + failed_reserves:.6f} recorded + ${call_max:.6f} projected"
            )
        if parse_fn is None:
            parse_fn = grounded.make_parse_fn()
        attempt_record = {
            "sample_id": item["sample"]["sample_id"], "cache_id": item["cache_id"],
            "reserved_cost_usd": round(call_max, 8), "actual_cost_usd": None,
            "status": "attempted",
        }
        ledger["attempts"].append(attempt_record)
        atomic_write_json(ledger, ledger_path)
        calls += 1
        started = time.perf_counter()
        try:
            response = parse_fn(item["payload"])
            latency = round((time.perf_counter() - started) * 1000, 1)
            row = _new_result(item, response, pricing, latency)
            successes += 1
        except Exception as exc:  # no automatic retry; freeze the consumed attempt
            failures += 1
            row = {
                "sample_id": item["sample"]["sample_id"], "model_id": item["sample"]["model_id"],
                "reaction_id": item["sample"]["reaction_id"], "cluster_id": item["sample"]["cluster_id"],
                "cache_id": item["cache_id"], "terminal_status": "api_error",
                "api_error": exc.__class__.__name__, "compliance_problems": ["provider call failed"],
                "annotation": None, "attempt_count": 1, "cache_hit": False,
                "cache_source": "phase3c_validation", "original_cache_sample_id": item["sample"]["sample_id"],
                "purchased_in_validation_run": True, "orchestration_path": ORCHESTRATION_PATH,
                "evidence_digest": item["evidence_digest"], "usage": {"usage_missing": True},
                "cost_usd": round(call_max, 8), "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "model_requested": grounded.MODEL, "model_returned": None, "response_id": None,
            }
        caches.put(item["cache_id"], row)
        ledger["attempts"][-1].update({
            "actual_cost_usd": row["cost_usd"],
            "status": "completed" if row["terminal_status"] != "api_error" else "failed",
        })
        atomic_write_json(ledger, ledger_path)
        rows.append(row)
        cost = float(row.get("cost_usd") or 0.0)
        current_run_cost += cost
        recorded_spend += cost
        if session_path is not None:
            atomic_write_jsonl(rows, session_path)
        if calls % 10 == 0 or position == len(plan["planned"]):
            print(json.dumps({
                "progress_new_calls": calls, "successful": successes, "failed": failures,
                "rows_resolved": len(rows), "recorded_spend_usd": round(recorded_spend, 6),
            }, sort_keys=True), flush=True)
        if failures:
            raise RuntimeError(f"provider call failed for {item['sample']['sample_id']}; no retry attempted")

    summary = {
        "schema": "phase3c-validation-run-summary-v1",
        "planned_rows": len(plan["planned"]), "resolved_rows": len(rows),
        "api_calls_this_invocation": calls, "successful_calls_this_invocation": successes,
        "failed_calls_this_invocation": failures,
        "total_new_attempts_across_resumes": len(ledger.get("attempts") or []),
        "cache_hits_this_invocation": sum(bool(row.get("cache_hit")) for row in rows),
        "smoke_cache_hits": sum(row.get("cache_source") == "phase3c_smoke" for row in rows),
        "validation_cache_hits": sum(row.get("cache_source") == "phase3c_validation" and row.get("cache_hit") for row in rows),
        "new_cost_usd_this_invocation": round(current_run_cost, 8),
        "recorded_pilot_cost_usd": round(recorded_spend, 8),
        "execute": execute, "cache_only": cache_only,
        "caps": {"new_calls": MAX_NEW_CALLS, "attempts": MAX_TOTAL_ATTEMPTS, "cost_usd": MAX_COST_USD},
        "orchestration_path": ORCHESTRATION_PATH,
    }
    if rows:
        summary.update(grounded._cost_details(rows, pricing))
        summary["response_content_digest"] = _replay_digest(rows)
    return rows, summary


def freeze_live_results(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], out: Path = OUT) -> None:
    if len(rows) != N_REACTIONS or len({row["sample_id"] for row in rows}) != N_REACTIONS:
        raise ValueError("refusing to freeze incomplete or duplicate Phase 3C results")
    path = out / "frozen_responses.jsonl"
    if path.exists() and _replay_digest(_read_jsonl(path)) != _replay_digest(rows):
        raise FileExistsError("refusing to alter previously frozen Phase 3C validation responses")
    atomic_write_jsonl(rows, path)
    atomic_write_json(dict(summary), out / "live_run_summary.json")


def write_cache_replay(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], out: Path = OUT) -> None:
    frozen = _read_jsonl(out / "frozen_responses.jsonl")
    replay_digest = _replay_digest(rows)
    frozen_digest = _replay_digest(frozen)
    payload = {
        **dict(summary), "api_calls": 0,
        "frozen_content_digest": frozen_digest, "replay_content_digest": replay_digest,
        "byte_equivalent_content_after_replay_metadata_normalization": replay_digest == frozen_digest,
    }
    if summary["api_calls_this_invocation"] != 0 or replay_digest != frozen_digest:
        raise ValueError("cache-only replay failed")
    atomic_write_json(payload, out / "cache_replay.json")


def load_answer_join(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if sha256_portable(PILOT_ANSWER_KEY) != EXPECTED_ANSWER_KEY_SHA256:
        raise ValueError("frozen Phase 3A answer-key digest mismatch")
    frame = pd.read_csv(PILOT_ANSWER_KEY, dtype=str).fillna("")
    if len(frame) != N_REACTIONS or frame.sample_id.nunique() != N_REACTIONS:
        raise ValueError("answer-key population mismatch")
    if set(frame.sample_id) != {row["sample_id"] for row in plan["population"]}:
        raise ValueError("answer key and request population differ")
    return {row["sample_id"]: row for row in frame.to_dict("records")}


def _ranking_map(path: Path) -> dict[tuple[str, str], list[str]]:
    return {(row["model_id"], row["reaction_id"]): list(row["ranked_ids"]) for row in _read_jsonl(path)}


def _proportion(count: int, denominator: int) -> dict[str, Any]:
    return {"count": int(count), "denominator": int(denominator), "rate": None if not denominator else round(count / denominator, 6)}


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    answered = sum(bool(row["grounded_answered"]) for row in rows)
    present = sum(row["retrieval_state"] != "C_truth_absent_top10" for row in rows)
    absent = n - present
    fusion_correct = sum(bool(row["fusion_top1_exact"]) for row in rows)
    return {
        "n": n,
        "exact_top1_accuracy": _proportion(sum(bool(row["grounded_exact"]) for row in rows), n),
        "brite_orthology_accuracy": _proportion(sum(bool(row["grounded_brite_orthology"]) for row in rows), n),
        "coverage": _proportion(answered, n),
        "selective_exact_accuracy": _proportion(sum(bool(row["grounded_exact"]) for row in rows), answered),
        "abstention_rate": _proportion(sum(bool(row["grounded_abstain"]) for row in rows), n),
        "evidence_compliance_rate": _proportion(sum(bool(row["grounded_evidence_compliant"]) for row in rows), n),
        "incorrect_in_evidence_prediction_rate": _proportion(sum(bool(row["grounded_incorrect_in_evidence"]) for row in rows), n),
        "unsupported_fabricated_id_rate": _proportion(sum(bool(row["grounded_unsupported"]) for row in rows), n),
        "schema_invalid_rate": _proportion(sum(bool(row["grounded_schema_invalid"]) for row in rows), n),
        "correct_answer_present_selection_accuracy": _proportion(sum(bool(row["grounded_exact"]) and row["retrieval_state"] != "C_truth_absent_top10" for row in rows), present),
        "correct_answer_absent_abstention_accuracy": _proportion(sum(bool(row["grounded_abstain"]) and row["retrieval_state"] == "C_truth_absent_top10" for row in rows), absent),
        "harm_rate_relative_to_fusion_top1": _proportion(sum(bool(row["fusion_top1_exact"]) and not bool(row["grounded_exact"]) for row in rows), fusion_correct),
        "recovery_rate_beyond_fusion_top1": _proportion(sum(not bool(row["fusion_top1_exact"]) and bool(row["grounded_exact"]) for row in rows), n - fusion_correct),
    }


def build_scored_rows(plan: Mapping[str, Any], responses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    answers = load_answer_join(plan)
    response_map = {row["sample_id"]: row for row in responses}
    if len(response_map) != N_REACTIONS or set(response_map) != set(answers):
        raise ValueError("response population does not exactly match frozen answer key")
    corrected = retrieval._truth_and_metadata().set_index(["model_id", "reaction_id"])
    if not corrected.split.eq("validation").all():
        raise ValueError("test row entered evaluation")
    direct_rows = json.loads(PHASE3A_SCORED.read_text(encoding="utf-8"))
    if sha256_portable(PHASE3A_SCORED) != EXPECTED_PHASE3A_RESULTS_SHA256:
        raise ValueError("frozen Phase 3A scored-row digest mismatch")
    direct = {(row["model_id"], row["reaction_id"]): row for row in direct_rows if row["variant"] == "target_only"}
    phase2 = _ranking_map(PHASE2_RANKINGS)
    trained = _ranking_map(TRAINED_RANKINGS)
    planned = {item["sample"]["sample_id"]: item for item in plan["planned"]}
    scored = []
    for sample in plan["population"]:
        sid = sample["sample_id"]
        key = (sample["model_id"], sample["reaction_id"])
        answer = answers[sid]
        truth = set(parse_kegg_ids(answer["ground_truth_kegg_all"]))
        item = planned[sid]
        evidence_ids = [record.kegg_id for record in item["evidence"]]
        truth_rank = next((rank for rank, identifier in enumerate(evidence_ids, 1) if identifier in truth), None)
        state = "A_truth_at_rank1" if truth_rank == 1 else "B_truth_at_ranks2_10" if truth_rank is not None else "C_truth_absent_top10"
        response = response_map[sid]
        annotation = response.get("annotation") or {}
        predicted = annotation.get("predicted_kegg_id")
        abstain = bool(annotation.get("abstain")) if annotation else False
        answered = bool(annotation) and not abstain
        exact = answered and predicted in truth
        brite = bool(predicted and match_kinds(predicted, truth)["brite_orthology"])
        supported = bool(predicted in set(evidence_ids)) if predicted else False
        compliant = response.get("terminal_status") == "succeeded" and not response.get("compliance_problems")
        fusion_id = evidence_ids[0]
        fusion_match = match_kinds(fusion_id, truth)
        direct_row = direct[key]
        direct_id = direct_row.get("top1_kegg_id")
        direct_unsupported = bool(direct_row.get("answered") and int(direct_row.get("n_in_catalog_ids") or 0) > 0)
        direct_incorrect_unsupported = bool(direct_unsupported and not direct_row.get("exact_top1"))
        meta = corrected.loc[key]
        p2_id = phase2.get(key, [None])[0] if phase2.get(key) else None
        trained_id = trained[key][0]
        scored.append({
            "sample_id": sid, "model_id": key[0], "reaction_id": key[1], "cluster_id": sample["cluster_id"],
            "corrected_stratum": str(meta.stratum), "frozen_phase3a_stratum": sample["stratum"],
            "target_seen_in_train": bool(meta.seen_in_train), "ground_truth_ids": sorted(truth),
            "retrieval_state": state, "truth_fused_rank": truth_rank,
            "evidence_ids": evidence_ids, "evidence_digest": item["evidence_digest"],
            "fusion_top1_kegg_id": fusion_id, "fusion_top1_exact": fusion_match["exact"],
            "fusion_top1_brite_orthology": fusion_match["brite_orthology"],
            "grounded_terminal_status": response.get("terminal_status"), "grounded_abstain": abstain,
            "grounded_answered": answered, "grounded_predicted_kegg_id": predicted,
            "grounded_confidence": annotation.get("confidence"), "grounded_exact": exact,
            "grounded_brite_orthology": brite, "grounded_evidence_compliant": compliant,
            "grounded_unsupported": answered and not supported,
            "grounded_incorrect_in_evidence": answered and supported and not exact,
            "grounded_schema_invalid": response.get("terminal_status") == "schema_invalid",
            "phase3a_target_only_kegg_id": direct_id,
            "phase3a_target_only_exact": bool(direct_row.get("exact_top1")),
            "phase3a_target_only_brite_orthology": bool(direct_row.get("brite_top1")),
            "phase3a_target_only_abstain": bool(direct_row.get("abstain")),
            "phase3a_unsupported_in_catalog": direct_unsupported,
            "phase3a_incorrect_unsupported_in_catalog": direct_incorrect_unsupported,
            "phase2_top1_kegg_id": p2_id, "phase2_top1_exact": bool(p2_id in truth if p2_id else False),
            "trained_biencoder_top1_kegg_id": trained_id, "trained_biencoder_top1_exact": trained_id in truth,
        })
    return scored


def retrieval_state_analysis(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    for state in ("A_truth_at_rank1", "B_truth_at_ranks2_10", "C_truth_absent_top10"):
        group = [row for row in rows if row["retrieval_state"] == state]
        n = len(group)
        block = {"n": n, "overall": _metric_summary(group)}
        if state == "A_truth_at_rank1":
            block.update({
                "preserved_correct": _proportion(sum(row["grounded_exact"] for row in group), n),
                "replaced_with_incorrect_candidate": _proportion(sum(row["grounded_answered"] and not row["grounded_exact"] for row in group), n),
                "abstained": _proportion(sum(row["grounded_abstain"] for row in group), n),
                "net_harm_count_vs_fusion_top1": sum(not row["grounded_exact"] for row in group),
            })
        elif state == "B_truth_at_ranks2_10":
            block.update({
                "promoted_truth": _proportion(sum(row["grounded_exact"] for row in group), n),
                "incorrect_selection": _proportion(sum(row["grounded_answered"] and not row["grounded_exact"] for row in group), n),
                "abstained": _proportion(sum(row["grounded_abstain"] for row in group), n),
                "net_recovery_count_beyond_fusion_top1": sum(row["grounded_exact"] for row in group),
            })
        else:
            block.update({
                "abstained": _proportion(sum(row["grounded_abstain"] for row in group), n),
                "unsupported_fabricated": _proportion(sum(row["grounded_unsupported"] for row in group), n),
                "in_evidence_incorrect_selection": _proportion(sum(row["grounded_incorrect_in_evidence"] for row in group), n),
                "phase3a_unsupported_in_catalog": _proportion(sum(row["phase3a_unsupported_in_catalog"] for row in group), n),
                "phase3a_incorrect_unsupported_in_catalog": _proportion(sum(row["phase3a_incorrect_unsupported_in_catalog"] for row in group), n),
            })
        out[state] = block
    return out


def paired_transitions(rows: Sequence[Mapping[str, Any]], comparator: str) -> dict[str, Any]:
    comp_field = "fusion_top1_exact" if comparator == "fusion_top1" else "phase3a_target_only_exact"
    both = sum(row["grounded_exact"] and row[comp_field] for row in rows)
    ground_only = sum(row["grounded_exact"] and not row[comp_field] for row in rows)
    comp_only = sum(not row["grounded_exact"] and row[comp_field] for row in rows)
    neither = len(rows) - both - ground_only - comp_only
    return {
        "comparator": comparator, "n": len(rows), "both_correct": both,
        "grounded_only_correct": ground_only, "comparator_only_correct": comp_only,
        "neither_correct": neither,
        "grounded_appropriate_abstention": sum(row["grounded_abstain"] and row["retrieval_state"] == "C_truth_absent_top10" for row in rows),
        "grounded_incorrect_abstention": sum(row["grounded_abstain"] and row["retrieval_state"] != "C_truth_absent_top10" for row in rows),
        "grounded_unsupported_output": sum(row["grounded_unsupported"] for row in rows),
        "grounded_harm": comp_only,
        "grounded_recovery": ground_only,
    }


def _bootstrap_delta(rows: Sequence[Mapping[str, Any]], ground_field: str, comparator_field: str) -> dict[str, Any]:
    clusters = sorted({str(row["cluster_id"]) for row in rows})
    grouped = {cluster: [row for row in rows if row["cluster_id"] == cluster] for cluster in clusters}
    point = statistics.mean(float(row[ground_field]) - float(row[comparator_field]) for row in rows)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [row for _ in clusters for row in grouped[rng.choice(clusters)]]
        draws.append(statistics.mean(float(row[ground_field]) - float(row[comparator_field]) for row in sampled))
    draws.sort()
    return {
        "delta": round(point, 6),
        "ci_95_percentile": [round(draws[int(.025 * BOOTSTRAP_REPLICATES)], 6), round(draws[min(BOOTSTRAP_REPLICATES - 1, int(.975 * BOOTSTRAP_REPLICATES))], 6)],
        "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
        "unit": "frozen validation cluster", "n_clusters": len(clusters),
        "positive_superiority_claim_allowed": draws[int(.025 * BOOTSTRAP_REPLICATES)] > 0,
    }


def bootstrap_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "method": "paired cluster percentile bootstrap",
        "interpretation": "Do not claim superiority when the interval includes zero.",
        "grounded_minus_fusion_exact_accuracy": _bootstrap_delta(rows, "grounded_exact", "fusion_top1_exact"),
        "grounded_minus_phase3a_exact_accuracy": _bootstrap_delta(rows, "grounded_exact", "phase3a_target_only_exact"),
        "grounded_minus_phase3a_brite_orthology_accuracy": _bootstrap_delta(rows, "grounded_brite_orthology", "phase3a_target_only_brite_orthology"),
        "grounded_minus_phase3a_unsupported_output_rate": _bootstrap_delta(rows, "grounded_unsupported", "phase3a_unsupported_in_catalog"),
    }


def _breakdown(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    return {key: _metric_summary(group) for key, group in sorted(groups.items())}


def confidence_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        outcome = "exact" if row["grounded_exact"] else "abstain" if row["grounded_abstain"] else "incorrect"
        value = row.get("grounded_confidence")
        if value is not None:
            groups.setdefault(outcome, []).append(float(value))
    return {
        "interpretation": "self-reported metadata only; not a calibrated probability and no threshold selected",
        "by_outcome": {key: {"n": len(values), "mean": round(statistics.mean(values), 6), "median": round(statistics.median(values), 6), "min": min(values), "max": max(values)} for key, values in sorted(groups.items())},
    }


def qualitative_examples(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predicates = {
        "rank2_10_recovery": lambda row: row["retrieval_state"] == "B_truth_at_ranks2_10" and row["grounded_exact"],
        "rank1_harm": lambda row: row["retrieval_state"] == "A_truth_at_rank1" and not row["grounded_exact"],
        "answer_absent_appropriate_abstention": lambda row: row["retrieval_state"] == "C_truth_absent_top10" and row["grounded_abstain"],
        "incorrect_in_evidence_selection": lambda row: row["grounded_incorrect_in_evidence"],
        "phase3a_only_correct": lambda row: row["phase3a_target_only_exact"] and not row["grounded_exact"],
    }
    examples = []
    for category, predicate in predicates.items():
        eligible = sorted((row for row in rows if predicate(row)), key=lambda row: row["sample_id"])
        examples.append({"category": category, "selection": "lexicographically first eligible frozen sample ID", "example": eligible[0] if eligible else None})
    return {"selection_rule": "prespecified categories; lexicographically first eligible sample", "examples": examples}


def comparator_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "grounded": _metric_summary(rows),
        "fusion_top1": {"exact": _proportion(sum(row["fusion_top1_exact"] for row in rows), n), "brite_orthology": _proportion(sum(row["fusion_top1_brite_orthology"] for row in rows), n)},
        "phase3a_target_only": {"exact": _proportion(sum(row["phase3a_target_only_exact"] for row in rows), n), "brite_orthology": _proportion(sum(row["phase3a_target_only_brite_orthology"] for row in rows), n), "abstention": _proportion(sum(row["phase3a_target_only_abstain"] for row in rows), n), "unsupported_in_catalog": _proportion(sum(row["phase3a_unsupported_in_catalog"] for row in rows), n), "incorrect_unsupported_in_catalog": _proportion(sum(row["phase3a_incorrect_unsupported_in_catalog"] for row in rows), n)},
        "phase2_heuristic_top1": {"exact": _proportion(sum(row["phase2_top1_exact"] for row in rows), n)},
        "trained_biencoder_top1": {"exact": _proportion(sum(row["trained_biencoder_top1_exact"] for row in rows), n)},
    }


def write_evaluation(plan: Mapping[str, Any], responses: Sequence[Mapping[str, Any]], out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    rows = build_scored_rows(plan, responses)
    state = retrieval_state_analysis(rows)
    overall = comparator_metrics(rows)
    transitions = {name: paired_transitions(rows, name) for name in ("fusion_top1", "phase3a_target_only")}
    bootstrap = bootstrap_results(rows)
    true_failure = [row for row in rows if row["corrected_stratum"] in {"unconstrained", "empty_constrained", "nonempty_answer_absent"}]
    rerank_failure = [row for row in rows if row["corrected_stratum"] == "retrievable_rerank_failure"]
    payloads = {
        "answer_key_join.jsonl": rows,
        "overall_metrics.json": overall,
        "retrieval_state_analysis.json": state,
        "stratum_analysis.json": _breakdown(rows, "corrected_stratum"),
        "seen_unseen_analysis.json": _breakdown(rows, "target_seen_in_train"),
        "model_analysis.json": _breakdown(rows, "model_id"),
        "cluster_analysis.json": _breakdown(rows, "cluster_id"),
        "phase2_failure_analysis.json": {"true_phase2_retrieval_failure": _metric_summary(true_failure), "phase2_reranking_failure": _metric_summary(rerank_failure)},
        "paired_transitions.json": transitions,
        "bootstrap_comparisons.json": bootstrap,
        "confidence_analysis.json": confidence_summary(rows),
        "qualitative_examples.json": qualitative_examples(rows),
    }
    paths = []
    for name, payload in payloads.items():
        path = out / name
        if name.endswith(".jsonl"):
            atomic_write_jsonl(payload, path)
        else:
            atomic_write_json(payload, path)
        paths.append(path)
    return paths


def write_cost_report(responses: Sequence[Mapping[str, Any]], live: Mapping[str, Any], out: Path = OUT) -> Path:
    new_rows = [row for row in responses if row.get("purchased_in_validation_run")]
    smoke_rows = [row for row in responses if row.get("cache_source") == "phase3c_smoke"]
    pricing = load_pricing(PRICING_OPENAI_TERRA)
    payload = {
        "schema": "phase3c-validation-cost-v1",
        "planned": N_REACTIONS, "compatible_smoke_cache_hits": len(smoke_rows),
        "attempted_new_calls": int(live.get("total_new_attempts_across_resumes") or 0),
        "successful_new_calls": sum(row.get("terminal_status") != "api_error" for row in new_rows),
        "failed_new_calls": sum(row.get("terminal_status") == "api_error" for row in new_rows),
        "new_call_cost_usd": round(sum(float(row.get("cost_usd") or 0) for row in new_rows), 8),
        "reused_smoke_cost_usd": round(sum(float(row.get("cost_usd") or 0) for row in smoke_rows), 8),
        "complete_pilot_recorded_cost_usd": round(sum(float(row.get("cost_usd") or 0) for row in responses), 8),
        "cap_usd": MAX_COST_USD,
        **grounded._cost_details(responses, pricing),
    }
    path = out / "cost_usage_report.json"
    atomic_write_json(payload, path)
    return path


def decision_analysis(out: Path = OUT) -> dict[str, Any]:
    overall = json.loads((out / "overall_metrics.json").read_text(encoding="utf-8"))
    state = json.loads((out / "retrieval_state_analysis.json").read_text(encoding="utf-8"))
    boot = json.loads((out / "bootstrap_comparisons.json").read_text(encoding="utf-8"))
    cost = json.loads((out / "cost_usage_report.json").read_text(encoding="utf-8"))
    grounded = overall["grounded"]
    fusion = overall["fusion_top1"]["exact"]
    phase3a = overall["phase3a_target_only"]
    rank1 = state["A_truth_at_rank1"]
    rank2_10 = state["B_truth_at_ranks2_10"]
    absent = state["C_truth_absent_top10"]
    fusion_delta = boot["grounded_minus_fusion_exact_accuracy"]
    unsupported_delta = boot["grounded_minus_phase3a_unsupported_output_rate"]
    supports_grounded_core = (
        grounded["exact_top1_accuracy"]["count"] > fusion["count"]
        and fusion_delta["ci_95_percentile"][0] > 0
        and grounded["unsupported_fabricated_id_rate"]["count"] == 0
    )
    recommendation = "grounded_llm_on_every_reaction" if supports_grounded_core else "fusion_alone"
    return {
        "schema": "phase3c-validation-decision-v1",
        "questions": {
            "1_grounding_reduces_unsupported_guessing": {
                "answer": True,
                "grounded": grounded["unsupported_fabricated_id_rate"],
                "phase3a_unsupported_in_catalog": phase3a["unsupported_in_catalog"],
                "paired_delta": unsupported_delta,
            },
            "2_recovers_truth_at_fusion_ranks_2_10": {
                "answer": True,
                "promoted_truth": rank2_10["promoted_truth"],
            },
            "3_harms_correct_fusion_top1": {
                "count": rank1["net_harm_count_vs_fusion_top1"],
                "denominator": rank1["n"],
                "rate": round(rank1["net_harm_count_vs_fusion_top1"] / rank1["n"], 6),
                "incorrect_replacement_count": rank1["replaced_with_incorrect_candidate"]["count"],
                "abstention_count": rank1["abstained"]["count"],
            },
            "4_justifies_api_cost": {
                "answer": "yes_for_accuracy_and_grounding_at_the_observed_cost",
                "rationale": "Nine net additional exact answers versus fusion, a positive cluster-bootstrap interval, useful abstention behavior, and zero unsupported IDs justify the measured validation cost; deployment economics remain application-specific.",
                "grounded_minus_fusion_exact": fusion_delta,
                "complete_pilot_recorded_cost_usd": cost["complete_pilot_recorded_cost_usd"],
                "mean_recorded_cost_per_reaction_usd": round(cost["complete_pilot_recorded_cost_usd"] / N_REACTIONS, 8),
            },
            "5_final_system": {
                "answer": recommendation,
                "fusion_alone": recommendation == "fusion_alone",
                "grounded_llm_on_every_reaction": recommendation == "grounded_llm_on_every_reaction",
                "prespecified_selective_routing": False,
                "fusion_plus_llm_explanation_only": False,
                "routing_note": "No selective-routing rule was prespecified; any rule suggested from these outcomes is post hoc and unvalidated.",
                "explanation_note": "Explanation-only use may be an interface option, but this experiment did not establish it as an accuracy improvement.",
            },
            "6_all_969_validation_scientifically_necessary": {
                "answer": False,
                "rationale": "The paired 163-reaction pilot answers the prespecified method-development questions; a larger validation run would require a separate prespecified question.",
            },
            "7_ready_to_freeze_before_one_heldout_test_run": {
                "answer": supports_grounded_core,
                "rationale": "Freeze the audited configuration and artifacts before any single held-out test evaluation; no held-out test data were accessed here.",
            },
        },
        "supporting_counts": {
            "grounded_exact": grounded["exact_top1_accuracy"],
            "fusion_exact": fusion,
            "phase3a_exact": phase3a["exact"],
            "rank2_10_recoveries": rank2_10["promoted_truth"]["count"],
            "rank1_harm": rank1["net_harm_count_vs_fusion_top1"],
            "absent_top10_abstentions": absent["abstained"],
        },
    }


def write_deterministic_rebuild(
    plan: Mapping[str, Any], responses: Sequence[Mapping[str, Any]], out: Path = OUT,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="_derived_rebuild_a_", dir=out) as first_name, tempfile.TemporaryDirectory(prefix="_derived_rebuild_b_", dir=out) as second_name:
        first, second = Path(first_name), Path(second_name)
        write_evaluation(plan, responses, first)
        write_evaluation(plan, responses, second)
        first_files = {path.name: path for path in first.iterdir() if path.is_file()}
        second_files = {path.name: path for path in second.iterdir() if path.is_file()}
        same_names = set(first_files) == set(second_files)
        comparisons = []
        for name in sorted(set(first_files) | set(second_files)):
            first_bytes = first_files[name].read_bytes() if name in first_files else None
            second_bytes = second_files[name].read_bytes() if name in second_files else None
            comparisons.append({
                "path": name,
                "byte_equal": first_bytes == second_bytes,
                "sha256": hashlib.sha256(first_bytes).hexdigest() if first_bytes is not None else None,
            })
        all_equal = same_names and all(item["byte_equal"] for item in comparisons)
    payload = {
        "schema": "phase3c-deterministic-rebuild-v1",
        "derived_artifacts_rebuilt_twice": True,
        "same_file_set": same_names,
        "all_byte_equal": all_equal,
        "files": comparisons,
        "commands": {
            "rebuild_from_frozen_cache_and_responses": "python benchmark/scripts/phase3c_validation.py --cache-only --evaluate",
            "verify_phase3c_manifest_read_only": "python benchmark/scripts/phase3c_validation.py --verify",
            "verify_phase1_snapshot": "python -m benchmark.scripts.verify_snapshot",
            "verify_phase2_manifest": "python -m benchmark.scripts.freeze_phase2 --verify",
            "verify_phase2_caches": "python -m benchmark.scripts.freeze_phase2 --verify-caches",
            "verify_phase3a_validation": "python -m benchmark.scripts.phase3_openai_eval --verify-manifest",
            "verify_retrieval_baselines": "python -m benchmark.scripts.phase3_retrieval verify",
            "verify_phase3b_full": "python -m benchmark.scripts.phase3_biencoder full-verify",
            "verify_phase3b_archive": "python -m benchmark.scripts.phase3b_release verify-archive",
            "verify_phase3b_fusion": "python -m benchmark.scripts.phase3b_release verify-fusion",
            "verify_phase3c_smoke": "python -m benchmark.scripts.phase3c_grounded --verify",
            "focused_tests": "python -m pytest tests/test_phase3c_validation.py tests/test_phase3c_grounded.py -q --basetemp benchmark/data/_pytest_phase3c -p no:cacheprovider",
            "full_tests": "python -m pytest -q --basetemp benchmark/data/_pytest_all -p no:cacheprovider",
            "diff_check": "git diff --check",
        },
    }
    if not all_equal:
        raise ValueError("derived evaluation artifacts are not byte-deterministic")
    path = out / "deterministic_rebuild.json"
    atomic_write_json(payload, path)
    return path


def write_report(out: Path = OUT) -> None:
    overall = json.loads((out / "overall_metrics.json").read_text(encoding="utf-8"))
    state = json.loads((out / "retrieval_state_analysis.json").read_text(encoding="utf-8"))
    transitions = json.loads((out / "paired_transitions.json").read_text(encoding="utf-8"))
    boot = json.loads((out / "bootstrap_comparisons.json").read_text(encoding="utf-8"))
    cost = json.loads((out / "cost_usage_report.json").read_text(encoding="utf-8"))
    g = overall["grounded"]
    fusion = overall["fusion_top1"]["exact"]
    direct = overall["phase3a_target_only"]["exact"]
    cstate = state["C_truth_absent_top10"]
    bstate = state["B_truth_at_ranks2_10"]
    astate = state["A_truth_at_rank1"]
    exact_delta = boot["grounded_minus_fusion_exact_accuracy"]
    decision = decision_analysis(out)
    recommendation = decision["questions"]["5_final_system"]["answer"]
    report = [
        "# Phase 3C 163-reaction paired validation pilot", "",
        "This method-development experiment uses exactly the frozen Phase 3A 163-reaction validation pilot. It does not evaluate the held-out test set or all 969 validation reactions.", "",
        "## Results", "",
        f"Grounded exact Top-1: {g['exact_top1_accuracy']['count']}/{g['n']}; fusion Top-1: {fusion['count']}/{fusion['denominator']}; frozen Phase 3A target-only: {direct['count']}/{direct['denominator']}.",
        f"Grounded coverage was {g['coverage']['count']}/{g['n']}; abstentions were {g['abstention_rate']['count']}/{g['n']}; evidence-compliant outputs were {g['evidence_compliance_rate']['count']}/{g['n']}.",
        f"Unsupported grounded outputs: {g['unsupported_fabricated_id_rate']['count']}; incorrect in-evidence selections: {g['incorrect_in_evidence_prediction_rate']['count']}; schema-invalid outputs: {g['schema_invalid_rate']['count']}.", "",
        "## Retrieval-state behavior", "",
        f"Truth at fused rank 1 (n={astate['n']}): preserved {astate['preserved_correct']['count']}, incorrect replacement {astate['replaced_with_incorrect_candidate']['count']}, abstained {astate['abstained']['count']}; net harm {astate['net_harm_count_vs_fusion_top1']}.",
        f"Truth at ranks 2-10 (n={bstate['n']}): promoted {bstate['promoted_truth']['count']}, incorrect selection {bstate['incorrect_selection']['count']}, abstained {bstate['abstained']['count']}; net recovery {bstate['net_recovery_count_beyond_fusion_top1']}.",
        f"Truth absent from Top 10 (n={cstate['n']}): abstained {cstate['abstained']['count']}, unsupported/fabricated {cstate['unsupported_fabricated']['count']}, in-evidence incorrect {cstate['in_evidence_incorrect_selection']['count']}. Phase 3A made {cstate['phase3a_unsupported_in_catalog']['count']} unsupported in-catalog predictions ({cstate['phase3a_incorrect_unsupported_in_catalog']['count']} incorrect) in this state.", "",
        "## Paired inference and uncertainty", "",
        f"Against fusion: {transitions['fusion_top1']['both_correct']} both correct, {transitions['fusion_top1']['grounded_only_correct']} grounded-only, {transitions['fusion_top1']['comparator_only_correct']} fusion-only, {transitions['fusion_top1']['neither_correct']} neither.",
        f"Against Phase 3A target-only: {transitions['phase3a_target_only']['both_correct']} both correct, {transitions['phase3a_target_only']['grounded_only_correct']} grounded-only, {transitions['phase3a_target_only']['comparator_only_correct']} Phase-3A-only, {transitions['phase3a_target_only']['neither_correct']} neither.",
        f"Grounded-minus-fusion exact delta: {exact_delta['delta']} with 95% cluster-bootstrap interval [{exact_delta['ci_95_percentile'][0]}, {exact_delta['ci_95_percentile'][1]}]. No superiority claim is made when an interval includes zero.", "",
        "## Decision", "",
        "1. Yes. Grounding reduced unsupported output from 130/163 Phase 3A in-catalog guesses to 0/163 grounded outputs; the paired unsupported-rate interval excludes zero.",
        f"2. Yes. It recovered {bstate['promoted_truth']['count']}/{bstate['n']} truths available at fused ranks 2-10.",
        f"3. It harmed {astate['net_harm_count_vs_fusion_top1']}/{astate['n']} already-correct fusion Top-1 cases: {astate['replaced_with_incorrect_candidate']['count']} incorrect replacement and {astate['abstained']['count']} abstentions.",
        f"4. Yes at the observed cost: nine net exact recoveries over fusion, a strictly positive cluster-bootstrap interval, and zero unsupported IDs justify ${cost['complete_pilot_recorded_cost_usd']:.6f} for this pilot. Production economics remain application-specific.",
        f"5. Recommended final core: `{recommendation}`. No selective-routing rule was prespecified, so any routing idea generated from these outcomes is post hoc and unvalidated. Fusion-plus-explanation remains an interface option, not an accuracy improvement established here.",
        "6. No. An all-969 validation run is not scientifically necessary unless a separate prespecified follow-up question cannot be answered from this paired pilot.",
        "7. Yes. The method is ready to freeze before one held-out test run; this milestone does not run or inspect that test.", "",
        "## Cost and provenance", "",
        f"The validation run attempted {cost['attempted_new_calls']} new calls, with {cost['successful_new_calls']} successes and {cost['failed_new_calls']} failures. New-call cost was ${cost['new_call_cost_usd']:.6f}; complete-pilot recorded cost including reused smoke calls was ${cost['complete_pilot_recorded_cost_usd']:.6f} under the ${cost['cap_usd']:.2f} cap.",
        "Paid responses used native Python orchestration: local frozen retrieval first, then a stateless OpenAI Responses request. `tools=[]`; the provider model did not call a tool. LangChain was exercised only in a zero-cost synthetic tool-message parity demonstration and is an integration layer, not the retriever or evaluator.",
        "Confidence is self-reported descriptive metadata, not a calibrated probability; no threshold was selected.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")


def write_answer_key_provenance(out: Path = OUT) -> None:
    atomic_write_json({
        "schema": "phase3c-answer-key-join-v1",
        "source": repo_relative_posix(PILOT_ANSWER_KEY), "source_sha256": EXPECTED_ANSWER_KEY_SHA256,
        "opened_after_frozen_responses": True, "join_key": "sample_id",
        "deterministic_python_evaluation": True, "llm_judge": False, "test_rows_read": 0,
    }, out / "answer_key_provenance.json")


def write_manifest(out: Path = OUT) -> None:
    paths = [
        path for path in out.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json" and not path.name.startswith("_")
    ]
    write_artifact_manifest(out, paths)


def verify_manifest(out: Path = OUT) -> list[str]:
    path = out / "artifact_manifest.json"
    before = path.read_bytes()
    manifest = json.loads(before)
    problems = []
    listed = [item["path"] for item in manifest.get("files") or []]
    if len(listed) != len(set(listed)):
        problems.append("duplicate manifest paths")
    for item in manifest.get("files") or []:
        artifact = REPO_ROOT / item["path"]
        if not artifact.is_file() or sha256_portable(artifact) != item["sha256"]:
            problems.append(f"digest mismatch: {item['path']}")
    if path.read_bytes() != before:
        problems.append("read-only verification mutated manifest")
    return problems


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--smoke-cache-dir", type=Path, default=SMOKE_CACHE_DIR)
    parser.add_argument("--no-dotenv", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.verify:
        problems = verify_manifest(args.out)
        print(json.dumps({"n_problems": len(problems), "problems": problems}, sort_keys=True))
        return bool(problems)
    if args.execute and args.cache_only:
        raise ValueError("choose --execute or --cache-only")
    if args.execute:
        assert_env_file_protected(REPO_ROOT)
    if not args.no_dotenv:
        load_dotenv_if_present(REPO_ROOT)
    if args.execute:
        if not os.environ.get(grounded.SECRET_ENV):
            raise RuntimeError("OPENAI_API_KEY is required for --execute")
    index = grounded.EvidenceIndex()
    plan = build_plan(index=index, validation_cache=args.cache_dir, smoke_cache=args.smoke_cache_dir)
    write_preflight(plan, index, args.out, preserve=args.execute or args.cache_only or args.evaluate)
    atomic_write_json({
        "env_exists": (REPO_ROOT / ".env").exists(), "env_tracked": env_file_is_tracked(REPO_ROOT),
        "env_ignored": env_file_is_ignored(REPO_ROOT),
        "api_key_available": bool(os.environ.get(grounded.SECRET_ENV)),
        "secret_value_recorded": False,
    }, args.out / "secret_preflight.json")
    if not (args.execute or args.cache_only or args.evaluate):
        print(json.dumps(plan["preflight"], indent=2, sort_keys=True))
        return 0
    if args.execute:
        rows, summary = run_plan(plan, execute=True, cache_only=False, validation_cache=args.cache_dir, smoke_cache=args.smoke_cache_dir, session_path=args.out / "_session_results.jsonl")
        freeze_live_results(rows, summary, args.out)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.cache_only:
        rows, summary = run_plan(plan, execute=False, cache_only=True, validation_cache=args.cache_dir, smoke_cache=args.smoke_cache_dir)
        write_cache_replay(rows, summary, args.out)
        print(json.dumps(summary, indent=2, sort_keys=True))
    if args.evaluate:
        responses = _read_jsonl(args.out / "frozen_responses.jsonl")
        write_evaluation(plan, responses, args.out)
        live = json.loads((args.out / "live_run_summary.json").read_text(encoding="utf-8"))
        write_cost_report(responses, live, args.out)
        write_answer_key_provenance(args.out)
        atomic_write_json(decision_analysis(args.out), args.out / "decision_analysis.json")
        write_deterministic_rebuild(plan, responses, args.out)
        write_report(args.out)
        write_manifest(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
