"""Offline output-budget assessment and durable result accounting for Luna's case.

No provider transport is implemented here. Read frozen validation responses and
the existing case request plan only; never load benchmark test reactions.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from collections import Counter
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from benchmark.scripts import luna_model250728_analysis as original
from benchmark.scripts.phase3c_grounded import (
    EvidenceRecord, GroundedAnnotation, validate_grounded_annotation,
)

MODEL = "gpt-6.1-terra"
PRICING = {"input_per_million": "1.00", "cache_read_per_million": "0.05",
           "cache_write_30m_per_million": "1.25", "output_per_million": "5.00"}
DOCS = {
    "reasoning": "https://developers.openai.com/api/docs/guides/reasoning",
    "output_usage": "https://developers.openai.com/api/docs/guides/token-counting",
    "pricing": "https://developers.openai.com/api/docs/pricing",
    "caching": "https://developers.openai.com/api/docs/guides/prompt-caching",
    "model": "https://developers.openai.com/api/docs/models/gpt-6.1-terra",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def tree_checksums(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): original.sha256_file(p)
            for p in sorted(root.rglob("*")) if p.is_file()}


def durable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(original.canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    points = np.percentile(values, [0, 50, 90, 95, 99, 100]).tolist()
    return {"n": len(values), "mean": sum(values) / len(values),
            **dict(zip(("min", "median", "p90", "p95", "p99", "max"), points))}


def historical_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce the historical SDK serialization without constructing a client."""
    from openai.resources.responses import Responses
    resource = object.__new__(Responses)
    resource._post = lambda url, *, body, **kwargs: body
    actual = {k: v for k, v in payload.items() if k not in {"schema_version", "prompt_version"}}
    return resource.parse(**actual, text_format=GroundedAnnotation)


def inspect_history(repo: Path, cache_repo: Path | None = None) -> dict[str, Any]:
    validation = repo / "benchmark/phase3/phase3c_validation"
    smoke = repo / "benchmark/phase3/phase3c_smoke"
    rows = read_jsonl(validation / "frozen_responses.jsonl")
    requests = read_jsonl(validation / "requests.jsonl")
    smoke_rows = read_jsonl(smoke / "frozen_responses.jsonl")
    if len(rows) != 163 or len(requests) != 163:
        raise ValueError("unexpected frozen grounded validation population")
    by_sample = {row["sample_id"]: row for row in rows}
    historical_bytes = sum(len(original.canonical_json(historical_body(row["payload"]))) for row in requests)
    total_input = sum(row["usage"]["input_tokens"] for row in rows)
    total_output = sum(row["usage"]["output_tokens"] for row in rows)
    schema_errors = []
    empty = []
    for row in rows:
        try:
            GroundedAnnotation.model_validate(row["annotation"], strict=True)
        except (ValueError, KeyError, TypeError):
            schema_errors.append(row["sample_id"])
        if not row.get("annotation") or not (row["annotation"].get("reasoning_summary") or "").strip():
            empty.append(row["sample_id"])
    ids = {row.get("response_id") for row in rows}
    config = json.loads((validation / "config.json").read_text(encoding="utf-8"))
    cache_audit = {}
    if cache_repo is not None:
        for phase in ("phase3c_smoke", "phase3c_validation"):
            directory = cache_repo / "benchmark/phase3" / phase / "_response_cache"
            cached = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json")) if not p.name.startswith("_")]
            ledger_path = directory / "_attempt_ledger.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"attempts": []}
            cache_audit[phase] = {"normalized_response_count": len(cached),
                "status_counts": dict(Counter(row.get("terminal_status") for row in cached)),
                "model_counts": dict(Counter(row.get("model_returned") for row in cached)),
                "attempt_count": len(ledger["attempts"]),
                "attempt_status_counts": dict(Counter(row.get("status") for row in ledger["attempts"])),
                "raw_provider_status_retained": sum("status" in row or "incomplete_details" in row for row in cached),
                "file_checksums": tree_checksums(directory)}
    return {
        "sources": {p.relative_to(repo).as_posix(): original.sha256_file(p) for p in
                    (validation / "frozen_responses.jsonl", validation / "requests.jsonl",
                     validation / "config.json", smoke / "frozen_responses.jsonl")},
        "model_counts": dict(Counter(row.get("model_returned") for row in rows)),
        "n_unique_responses": len(ids), "validation_count": len(rows),
        "smoke_count": len(smoke_rows),
        "smoke_already_in_validation": sum(row.get("response_id") in ids for row in smoke_rows),
        "output_limit": config["max_output_tokens"], "reasoning_effort": config["reasoning_effort"],
        "output_tokens_including_reasoning_and_formatting": distribution([row["usage"]["output_tokens"] for row in rows]),
        "reasoning_tokens_subset": distribution([row["usage"]["reasoning_tokens"] for row in rows]),
        "input_tokens": distribution([row["usage"]["input_tokens"] for row in rows]),
        "successful_output_tokens": distribution([row["usage"]["output_tokens"] for row in rows if row["terminal_status"] == "succeeded"]),
        "reported_usage_missing": sum(bool(row["usage"].get("usage_missing")) for row in rows),
        "terminal_status_counts": dict(Counter(row["terminal_status"] for row in rows)),
        "schema_invalid_on_revalidation": schema_errors, "empty_structured_answers": empty,
        "at_least_450_output_tokens": [row["sample_id"] for row in rows if row["usage"]["output_tokens"] >= 450],
        "exceeding_512": [row["sample_id"] for row in rows if row["usage"]["output_tokens"] > 512],
        "recorded_incomplete_or_truncated": sum(row["terminal_status"] in {"incomplete", "truncated"} for row in rows),
        "termination_limitation": "Frozen records retain normalized terminal status, not original provider status/incomplete_details or raw output. No recorded truncation is not independent proof that every raw termination was completed.",
        "gpt61_measured_grounded_responses": 0,
        "permitted_historical_cache_audit": cache_audit,
        "gpt61_limitation": "GPT-5.6 Terra observations do not validate the required budget of GPT-6.1 Terra or the harder 541-reaction use case.",
        "expectation_calibration": {
            "actual_input_tokens": total_input, "complete_historical_provider_body_bytes": historical_bytes,
            "input_tokens_per_complete_body_byte": total_input / historical_bytes,
            "mean_total_output_tokens": total_output / len(rows),
            "assumption": "Transfer GPT-5.6 grounded input-token/complete-body-byte ratio and mean total output to GPT-6.1 Terra and this case; heuristic scenario only, not a measured GPT-6.1 expectation or a bound.",
        },
        "all_validation_samples_aligned": set(by_sample) == {row["sample_id"] for row in requests},
    }


def reservation(body: Mapping[str, Any]) -> dict[str, Any]:
    limit = body["max_output_tokens"]
    if limit not in (512, 2048):
        raise ValueError("output limit must be an assessed option")
    size = len(original.canonical_json(body))
    bounded_input = size + 4096
    if bounded_input > 272_000:
        raise ValueError("short-context pricing bound exceeded")
    usd = (Decimal(bounded_input) * Decimal(PRICING["cache_write_30m_per_million"])
           + Decimal(limit) * Decimal(PRICING["output_per_million"])) / Decimal(1_000_000)
    nanos = int((usd * 1_000_000_000).to_integral_value(rounding=ROUND_CEILING))
    return {"canonical_provider_body_bytes": size, "input_token_upper_bound": bounded_input,
            "max_billable_output_tokens_including_reasoning_and_formatting": limit,
            "reservation_nusd": nanos, "reservation_usd": nanos / 1_000_000_000}


def verify_plan(plan: Mapping[str, Any], requests: list[dict[str, Any]], manifest: list[dict[str, Any]]) -> None:
    if len(requests) != plan["request_count"] or len(requests) != len(manifest):
        raise ValueError("plan coverage count mismatch")
    units, keys = set(), set()
    for row, meta in zip(requests, manifest):
        body, unit = row["provider_payload"], row["unit"]
        unit_id = (unit["model_id"], unit["reaction_id"])
        if unit_id in units or row["unit_cache_key"] in keys:
            raise ValueError("duplicate unit or cache key")
        units.add(unit_id)
        keys.add(row["unit_cache_key"])
        if body["model"] != MODEL or body["max_output_tokens"] != plan["max_output_tokens"]:
            raise ValueError("model/output limit mismatch")
        if body.get("reasoning") != {"effort": "low"} or body.get("service_tier") != "default":
            raise ValueError("reasoning/tier mismatch")
        if body.get("store") is not False or body.get("tools") != [] or not body["text"]["format"]["strict"]:
            raise ValueError("provider configuration mismatch")
        key = original.unit_cache_key(unit, MODEL, original.digest(body))
        if key != row["unit_cache_key"] or meta["unit_cache_key"] != key or meta["unit"] != unit:
            raise ValueError("unit/cache identity mismatch")
        if meta["provider_payload_sha256"] != original.digest(body) or meta["reservation"] != reservation(body):
            raise ValueError("payload digest/reservation mismatch")
    total = sum(row["reservation"]["reservation_nusd"] for row in manifest)
    if total != plan["reservation_nusd"] or total > plan["hard_cap_nusd"]:
        raise ValueError("full retained reservations exceed cap or disagree with plan")


def build_options(source: Path, out: Path, repo: Path, *, reported_balance: str = "19.97", cache_repo: Path | None = None) -> dict[str, Any]:
    before = tree_checksums(source)
    requests = read_jsonl(source / "requests.jsonl")
    if len(requests) != 541 or len({row["unit"]["reaction_id"] for row in requests}) != 541:
        raise ValueError("frozen source must contain exactly 541 unique units")
    if any(row["provider_payload"]["max_output_tokens"] != 512 for row in requests):
        raise ValueError("unexpected prior output limit")
    original.verify_provider_plan(requests, read_jsonl(source / "request_manifest.jsonl"))
    history = inspect_history(repo, cache_repo)
    out.mkdir(parents=True, exist_ok=False)
    original.atomic_json(out / "prior_plan_preserved_checksums.json", before)
    prior = out / "prior_512_plan_v1"
    prior.mkdir()
    for name in ("requests.jsonl", "request_manifest.jsonl", "analysis_summary.json", "pricing_snapshot.json", "execution_state_template.json", "dry_run_report.json", "REPORT.md"):
        (prior / name).write_bytes((source / name).read_bytes())
        if original.sha256_file(prior / name) != before[name]:
            raise ValueError("prior plan versioned copy checksum mismatch")
    original.atomic_json(out / "historical_usage_assessment.json", history)
    original.atomic_json(out / "pricing_snapshot.json", {
        "checked": "2026-11-07", "rates_usd_per_million": PRICING, "official_sources": DOCS,
        "tier": "default Standard", "context": "every bounded input is below 272,000 tokens",
        "cache_policy": "Default implicit 30-minute writes; reserve ALL bounded input at $1.25/M with no cache-read discount. Cache writes replace ordinary input rate; they are not added to it.",
        "exclusions": "No tools, images, Batch/Flex/Fast, 12-hour cache retention, regional/FedRAMP endpoints, taxes, currency conversion, or unrelated account use. Global api.openai.com only. Reprice/re-authorize if these assumptions change.",
        "input_bound": "Canonical complete provider body UTF-8 bytes as a conservative text-token bound plus 4,096 framing/schema/system tokens per request; assumption, not provider-certified token count. No context-capacity reservation.",
        "output_bound": "max_output_tokens covers reasoning, visible structured answer, and non-visible formatting; reasoning is not added twice.",
    })
    options = []
    calibration = history["expectation_calibration"]
    # Actual input tokens do not change when only the maximum OUTPUT changes.
    # Use one shared input scenario; the conservative serialized-byte bound
    # still includes the one extra digit in 2048 independently for each option.
    shared_expected_input = sum(len(original.canonical_json(row["provider_payload"])) for row in requests) * calibration["input_tokens_per_complete_body_byte"]
    for limit, cap in ((512, 12), (2048, 16)):
        directory = out / f"max_output_{limit}"
        directory.mkdir()
        revised, manifest = [], []
        for old in requests:
            row = copy.deepcopy(old)
            body = row["provider_payload"]
            body["max_output_tokens"] = limit
            unchanged = copy.deepcopy(body)
            unchanged["max_output_tokens"] = 512
            if unchanged != old["provider_payload"]:
                raise ValueError("provider input changed beyond intended output limit")
            key = original.unit_cache_key(row["unit"], MODEL, original.digest(body))
            if (limit == 512 and key != old["unit_cache_key"]) or (limit == 2048 and key == old["unit_cache_key"]):
                raise ValueError("output-limit/cache-identity compatibility violation")
            row["unit_cache_key"] = key
            revised.append(row)
            manifest.append({"unit": row["unit"], "unit_cache_key": key,
                             "provider_payload_sha256": original.digest(body),
                             "maximum_attempts": 1, "reservation": reservation(body)})
        original.atomic_jsonl(directory / "requests.jsonl", revised)
        original.atomic_jsonl(directory / "request_manifest.jsonl", manifest)
        total = sum(row["reservation"]["reservation_nusd"] for row in manifest)
        expected_input = shared_expected_input
        expected_output = len(revised) * calibration["mean_total_output_tokens"]
        expected_standard = (expected_input + expected_output * 5) / 1_000_000
        expected_write = (expected_input * 1.25 + expected_output * 5) / 1_000_000
        plan = {
            "schema": "luna-grounded-output-budget-v2", "request_count": len(revised),
            "model": MODEL, "max_output_tokens": limit, "reasoning_effort": "low",
            "prompt_version": original.PROMPT_VERSION, "strict_schema": "GroundedAnnotation",
            "service_tier": "default", "store": False, "tools": [],
            "endpoint": "https://api.openai.com/v1/responses", "default_cache_lifetime": "30m implicit",
            "response_handling": "Use raw Responses.create transport with this exact strict-schema body, then locally parse after checking provider status; avoid losing incomplete_details to a parser exception.",
            "maximum_attempts": 541, "automatic_retries": 0,
            "source_plan_checksums_digest": original.digest(before),
            "requests_sha256": original.sha256_file(directory / "requests.jsonl"),
            "request_manifest_sha256": original.sha256_file(directory / "request_manifest.jsonl"),
            "pricing": PRICING, "reservation_policy": "complete_body_bytes_plus4096_all_input_30m_write_and_max_total_output_v1",
            "reservation_nusd": total, "hard_cap_nusd": cap * 1_000_000_000,
            "expected_cost_scenario": {"estimated_input_tokens": expected_input,
                "estimated_total_output_tokens": expected_output,
                "usd_at_ordinary_input_rate": expected_standard, "usd_at_all_30m_write_rate": expected_write,
                "assumption": calibration["assumption"], "cache_read_savings_assumed": False},
            "reported_balance_usd": reported_balance, "balance_account_verified": False,
            "credit_unallocated_after_cap_usd": str(Decimal(reported_balance) - cap),
            "funds_allocated_to_phase5_usd": "0", "execution_authorized": False,
            "provider_temporal_drift": "Identical requests do not guarantee identical hosted model behavior across time.",
        }
        verify_plan(plan, revised, manifest)
        plan_digest = original.digest(plan)
        original.atomic_json(directory / "plan.json", plan)
        original.atomic_json(directory / "plan_identity.json", {"plan_digest": plan_digest,
                             "definition": "SHA256 of canonical complete plan.json object"})
        options.append({"max_output_tokens": limit, "plan_digest": plan_digest,
            "reservation_usd": total / 1_000_000_000, "hard_cap_usd": cap,
            "all_541_retained_reservations_fit": total <= cap * 1_000_000_000,
            "expected_cost_scenario": plan["expected_cost_scenario"],
            "credit_unallocated_after_cap_usd": plan["credit_unallocated_after_cap_usd"]})
    after = tree_checksums(source)
    if after != before:
        raise ValueError("prior plan was modified")
    if cache_repo is not None:
        for phase, audit in history["permitted_historical_cache_audit"].items():
            if tree_checksums(cache_repo / "benchmark/phase3" / phase / "_response_cache") != audit["file_checksums"]:
                raise ValueError("historical cache modified")
    comparison = {"options": options, "preferred_output_limit": 2048,
        "reason": "No measured GPT-6.1 grounded responses validate a reduction; preserve established output headroom.",
        "same_541_units_and_all_other_provider_fields": True, "prior_plan_preserved": True,
        "provider_calls": 0, "retrieval_rerun": False, "balance_usd": reported_balance,
        "balance_source": "user-reported current dashboard value; no independent billing mechanism available",
        "phase5": "paused; no funds allocated", "paid_execution_authorized": False}
    original.atomic_json(out / "budget_comparison.json", comparison)
    write_proposal(out, history, comparison)
    return comparison


def write_proposal(out: Path, history: Mapping[str, Any], comparison: Mapping[str, Any]) -> None:
    usage = history["output_tokens_including_reasoning_and_formatting"]
    lines = ["# Luna MODEL2507280001 output-budget execution proposal", "",
        "Status: prepared only; no inference or spending authorized. Phase 5 remains paused.", "",
        "Prefer 2,048 total output tokens. The 163 unique GPT-5.6 Terra grounded responses used a 2,048 limit: "
        f"median {usage['median']:g}, P90 {usage['p90']:g}, P95 {usage['p95']:g}, P99 {usage['p99']:g}, max {usage['max']:g}. "
        "All were below 512; none reached 450. Output totals include reasoning/formatting; reasoning is a subset. "
        "There are zero measured GPT-6.1 Terra grounded responses. Historical usage cannot validate reducing its budget.", "",
        "Frozen normalized statuses: 161 succeeded, 2 evidence-compliance errors (abstentions incorrectly claiming support). "
        "All 163 validate against the structural schema and have nonempty answer content. No incomplete/truncated/empty/schema-invalid "
        "response is recorded, but the old artifacts omitted provider status/incomplete_details. This limits retrospective termination checks.", "",
        "The read-only local historical attempt journals record 5 smoke plus 158 new validation attempts as completed. The smoke responses "
        "are already present in the 163-case validation table and are not counted twice. Their raw provider termination fields were not retained.", "",
        "Official API behavior: the output cap includes reasoning, answer text and hidden formatting. Reaching it may return "
        "status=incomplete, incomplete_details.reason=max_output_tokens, including empty visible output; usage can still be billed. "
        "New accounting records this as a failure, never a biological abstention.", "",
        "| Max output | Heuristic expected cost (ordinary to all cache-write input) | Full retained reservation | Proposed cap | Unallocated from reported $19.97 |",
        "|---|---:|---:|---:|---:|"]
    for option in comparison["options"]:
        e = option["expected_cost_scenario"]
        lines.append(f"| {option['max_output_tokens']} | ${e['usd_at_ordinary_input_rate']:.6f}–${e['usd_at_all_30m_write_rate']:.6f} | ${option['reservation_usd']:.9f} | ${option['hard_cap_usd']:.2f} | ${option['credit_unallocated_after_cap_usd']} |")
    lines += ["", "Expected cost is a scenario calibrated using GPT-5.6 grounded actual input tokens per complete request byte and mean "
        "total output tokens, transferred to this different model/case. It is not measured GPT-6.1 usage; real cost may differ. "
        "Both scenarios assume one attempt per unit and no cached-input discounts.", "",
        "The expected-input scenario is identical across output limits because the tokenized prompt/schema is unchanged. A one-byte "
        "difference in the serialized limit affects only the deliberately pessimistic complete-body-byte bound.", "",
        "Reservation formula per request: ((canonical complete request UTF-8 bytes + 4,096) × $1.25 + output limit × $5) / 1,000,000. "
        "Use integer nanodollars rounded upward per request. All bounded inputs are short context. This retains the previous conservative "
        "framing allowance, rather than reserving the full model context. It is not a provider-certified exact token count. "
        "Cache writes replace ordinary input billing, rather than adding to it. Standard global endpoint only; no tools, regional/FedRAMP uplift, "
        "tax, foreign exchange, unrelated account use or alternative tiers are included. Reprice if those assumptions change.", "",
        "Both proposed caps cover all 541 single attempts with every reservation retained. Failed, invalid or uncertain attempts keep their "
        "full reservation. Only a successful valid selection/abstention with separately verified billing may settle to lower cost. No retries "
        "or silent repurchases. The durable plan-bound journal is authoritative; response caches are reconstructable from it after interruption. "
        "Every unit stays in coverage, with failure status separate from biological abstention.", "",
        "The future live adapter must use max_retries=0 and return the raw Responses result for status/incomplete_details checks before "
        "local schema parsing; the complete strict-schema request body is unchanged by that parsing choice. The offline runner exposes "
        "an injected transport boundary for synthetic tests only and denies it unless the exact plan digest is supplied explicitly. "
        "An outstanding reservation after interruption blocks resend, including when it is unknown whether the provider received the request. "
        "A stale writer lock requires explicit recovery.", "",
        "Available balance: $19.97 reported by the user, independently unverified. No authorized account/billing connector is available. "
        "The $16 proposal would leave $3.97 unallocated if that balance is still current. No money is allocated to Phase 5. "
        "Any desired larger reserve for later work requires a scope/budget decision, not a silent output-limit/population reduction.", "",
        "Exact authorization identities (model=gpt-6.1-terra; reasoning=low; Responses API; strict GroundedAnnotation; store=false; tools=[]; "
        "service_tier=default; 541 units; one attempt/unit; zero automatic retries):", ""]
    for option in comparison["options"]:
        lines.append(f"- max_output_tokens={option['max_output_tokens']}, hard cap=${option['hard_cap_usd']:.2f}, plan SHA256 `{option['plan_digest']}`.")
    lines += ["", "The older 512/$12 proposal and its exact artifacts remain preserved as historical planning; this assessment recommends "
        "the new 2,048/$16 plan. Identical payloads cannot eliminate hosted-model version or temporal drift. This proposal authorizes no execution.", "",
        "Official sources: " + "; ".join(f"[{name}]({url})" for name, url in DOCS.items()), ""]
    (out / "EXECUTION_PROPOSAL.md").write_text("\n".join(lines), encoding="utf-8")


def verify_artifacts(source: Path, out: Path) -> dict[str, Any]:
    """Read-only check of both complete provider plans and preserved v1 inputs."""
    expected = json.loads((out / "prior_plan_preserved_checksums.json").read_text(encoding="utf-8"))
    if tree_checksums(source) != expected:
        raise ValueError("prior source plan changed")
    for path in (out / "prior_512_plan_v1").iterdir():
        if path.is_file() and original.sha256_file(path) != expected[path.name]:
            raise ValueError("versioned prior-plan copy mismatch")
    prior = read_jsonl(source / "requests.jsonl")
    options = []
    option_keys = []
    for limit in (512, 2048):
        directory = out / f"max_output_{limit}"
        plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
        identity = json.loads((directory / "plan_identity.json").read_text(encoding="utf-8"))
        rows = read_jsonl(directory / "requests.jsonl")
        manifest = read_jsonl(directory / "request_manifest.jsonl")
        if plan["request_count"] != 541 or identity["plan_digest"] != original.digest(plan):
            raise ValueError("incorrect population or plan digest")
        for name, field in (("requests.jsonl", "requests_sha256"), ("request_manifest.jsonl", "request_manifest_sha256")):
            if original.sha256_file(directory / name) != plan[field]:
                raise ValueError("request/manifest checksum mismatch")
        verify_plan(plan, rows, manifest)
        for row, old in zip(rows, prior):
            body = copy.deepcopy(row["provider_payload"])
            body["max_output_tokens"] = 512
            if body != old["provider_payload"] or row["unit"] != old["unit"]:
                raise ValueError("unintended model input/unit change")
            if limit == 512 and row != old:
                raise ValueError("512 control differs from historical complete request")
        option_keys.append({row["unit_cache_key"] for row in rows})
        bounds = [row["reservation"]["input_token_upper_bound"] for row in manifest]
        options.append({"max_output_tokens": limit, "plan_digest": identity["plan_digest"],
            "request_count": len(rows), "unique_unit_keys": len(option_keys[-1]),
            "input_upper_bound_min": min(bounds), "input_upper_bound_max": max(bounds),
            "reservation_nusd": plan["reservation_nusd"], "hard_cap_nusd": plan["hard_cap_nusd"],
            "all_reservations_retained_fits_cap": plan["reservation_nusd"] <= plan["hard_cap_nusd"]})
    if option_keys[0] & option_keys[1]:
        raise ValueError("changed output limit reused an old unit cache identity")
    return {"status": "passed", "prior_plan_preserved": True, "versioned_prior_plan_copies_verified": True,
        "512_complete_requests_identical_to_prior": True, "2048_only_provider_change_is_output_limit": True,
        "output_option_cache_keys_disjoint": True, "options": options, "provider_calls": 0,
        "heldout_access": False, "retrieval_or_gpu_inference": False}


def classify_response(raw: Mapping[str, Any], evidence: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Preserve termination and distinguish failures from valid biological abstention."""
    raw = dict(raw)
    result = {"provider_status": raw.get("status"), "incomplete_details": raw.get("incomplete_details"),
              "usage": raw.get("usage"), "raw_response": raw,
              "terminal_status": "failure_provider", "annotation": None, "biological_abstain": None}
    items = raw.get("output") or []
    if raw.get("status") == "incomplete" or raw.get("incomplete_details") or any(item.get("status") == "incomplete" for item in items):
        result["terminal_status"] = "failure_incomplete"
        return result
    if raw.get("model") != MODEL:
        result["terminal_status"] = "failure_model_mismatch"
        return result
    if raw.get("status") != "completed":
        return result
    content = [part for item in items for part in (item.get("content") or [])]
    if any(part.get("type") == "refusal" for part in content):
        result["terminal_status"] = "failure_refusal"
        return result
    text = raw.get("output_text") or "".join(part.get("text", "") for part in content if part.get("type") == "output_text")
    if not text.strip():
        result["terminal_status"] = "failure_empty"
        return result
    try:
        annotation = GroundedAnnotation.model_validate(json.loads(text), strict=True)
    except (ValueError, TypeError):
        result["terminal_status"] = "failure_schema_invalid"
        return result
    records = [EvidenceRecord(**{key: value for key, value in row.items() if key in EvidenceRecord.__dataclass_fields__ and key != "provenance"}, provenance={}) for row in evidence]
    problems = validate_grounded_annotation(annotation, records)
    if problems:
        result.update({"terminal_status": "failure_evidence_compliance", "compliance_problems": problems})
        return result
    result.update({"terminal_status": "abstained" if annotation.abstain else "succeeded",
                   "biological_abstain": annotation.abstain, "annotation": annotation.model_dump(mode="json")})
    return result


class RunState:
    """Plan-bound durable state for one serial executor; no API transport or retries.

    Journal commits precede derived cache writes. Any reservation survives a
    crash before completion recording. Lock contention fails closed; stale locks
    need explicit recovery rather than risking a second paid attempt.
    """
    def __init__(self, plan_dir: Path, state_dir: Path):
        self.plan_dir, self.directory = Path(plan_dir), Path(state_dir)
        self.plan = json.loads((self.plan_dir / "plan.json").read_text(encoding="utf-8"))
        self.plan_digest = original.digest(self.plan)
        identity = json.loads((self.plan_dir / "plan_identity.json").read_text(encoding="utf-8"))
        if identity["plan_digest"] != self.plan_digest:
            raise ValueError("plan digest mismatch")
        self.requests = read_jsonl(self.plan_dir / "requests.jsonl")
        self.manifest = read_jsonl(self.plan_dir / "request_manifest.jsonl")
        if original.sha256_file(self.plan_dir / "requests.jsonl") != self.plan["requests_sha256"] or original.sha256_file(self.plan_dir / "request_manifest.jsonl") != self.plan["request_manifest_sha256"]:
            raise ValueError("frozen request file digest mismatch")
        verify_plan(self.plan, self.requests, self.manifest)
        self.by_key = {row["unit_cache_key"]: row for row in self.manifest}
        self.request_by_key = {row["unit_cache_key"]: row for row in self.requests}
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "journal.json"
        with self._lock():
            if not self.path.exists():
                durable_json(self.path, {"schema": "luna-grounded-journal-v2", "plan_digest": self.plan_digest,
                    "hard_cap_nusd": self.plan["hard_cap_nusd"], "attempts": {}})
            self._read()

    @contextmanager
    def _lock(self):
        lock = self.directory / "writer.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ValueError("state already locked; no automatic recovery or resend") from exc
        try:
            os.close(fd)
            yield
        finally:
            lock.unlink()

    def _read(self) -> dict[str, Any]:
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("schema") != "luna-grounded-journal-v2" or state.get("plan_digest") != self.plan_digest or state.get("hard_cap_nusd") != self.plan["hard_cap_nusd"]:
            raise ValueError("journal plan/cap mismatch")
        return state

    @staticmethod
    def total(state: Mapping[str, Any]) -> int:
        return sum(attempt.get("verified_cost_nusd", attempt["reservation_nusd"]) for attempt in state["attempts"].values())

    def reserve(self, key: str) -> None:
        with self._lock():
            state = self._read()
            if key not in self.by_key:
                raise ValueError("unknown planned unit")
            if key in state["attempts"]:
                raise ValueError("unit already attempted; retries and repurchases prohibited")
            if (self.directory / "cache" / f"{key}.json").exists():
                raise ValueError("orphan cache requires explicit audit")
            amount = self.by_key[key]["reservation"]["reservation_nusd"]
            if len(state["attempts"]) >= self.plan["maximum_attempts"] or self.total(state) + amount > self.plan["hard_cap_nusd"]:
                raise ValueError("attempt or budget cap exceeded")
            state["attempts"][key] = {"reservation_nusd": amount, "status": "reserved_before_send_or_uncertain"}
            durable_json(self.path, state)

    def _write_cache(self, key: str, result: Mapping[str, Any]) -> None:
        durable_json(self.directory / "cache" / f"{key}.json", {
            "plan_digest": self.plan_digest, "provider_payload_sha256": self.by_key[key]["provider_payload_sha256"],
            "unit_cache_key": key, "result": result})

    def record(self, key: str, raw_response: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock():
            state = self._read()
            attempt = state["attempts"].get(key)
            if not attempt or "result" in attempt:
                raise ValueError("requires an outstanding reserved attempt")
            body = self.request_by_key[key]["provider_payload"]
            evidence = json.loads(body["input"].split("\nEVIDENCE\n", 1)[1])
            result = classify_response(raw_response, evidence)
            attempt.update({"status": result["terminal_status"], "result": result})
            durable_json(self.path, state)
            self._write_cache(key, result)
            return result

    def cached_result(self, key: str) -> dict[str, Any] | None:
        with self._lock():
            state = self._read()
            result = state["attempts"].get(key, {}).get("result")
            if result is None:
                return None
            cache = self.directory / "cache" / f"{key}.json"
            if not cache.exists():
                self._write_cache(key, result)
            stored = json.loads(cache.read_text(encoding="utf-8"))
            if stored != {"plan_digest": self.plan_digest, "provider_payload_sha256": self.by_key[key]["provider_payload_sha256"], "unit_cache_key": key, "result": result}:
                raise ValueError("cache/journal mismatch; no automatic repurchase")
            return result

    def settle_verified(self, key: str, actual_cost_nusd: int, billing_reference: str) -> None:
        with self._lock():
            state = self._read()
            attempt = state["attempts"][key]
            if attempt["status"] not in {"succeeded", "abstained"} or not billing_reference.strip():
                raise ValueError("only valid completed results with verified billing may release reservations")
            if type(actual_cost_nusd) is not int or not 0 <= actual_cost_nusd <= attempt["reservation_nusd"] or "verified_cost_nusd" in attempt:
                raise ValueError("invalid or duplicate billing settlement")
            attempt.update({"verified_cost_nusd": actual_cost_nusd, "billing_reference": billing_reference})
            durable_json(self.path, state)

    def coverage(self) -> list[dict[str, Any]]:
        with self._lock():
            state = self._read()
            return [{"unit": row["unit"], "unit_cache_key": row["unit_cache_key"],
                     "status": state["attempts"].get(row["unit_cache_key"], {}).get("status", "not_attempted"),
                     "biological_abstain": state["attempts"].get(row["unit_cache_key"], {}).get("result", {}).get("biological_abstain")}
                    for row in self.manifest]

    def run_one(self, key: str, call: Any, *, authorized_plan_digest: str | None = None) -> dict[str, Any]:
        """Injected transport boundary for tests/future executor; denied by default.

        A future live adapter must use SDK max_retries=0 and the exact frozen
        endpoint/body. This offline module never constructs such an adapter.
        """
        if authorized_plan_digest != self.plan_digest:
            raise ValueError("execution not authorized for this exact plan")
        cached = self.cached_result(key)
        if cached is not None:
            return cached
        with self._lock():
            if key in self._read()["attempts"]:
                return {"terminal_status": "failure_interrupted_or_uncertain", "biological_abstain": None}
        self.reserve(key)
        try:
            raw = call(copy.deepcopy(self.request_by_key[key]["provider_payload"]))
        except Exception as exc:
            # Store the exception class only, avoiding URLs/credentials in text.
            raw = {"status": "failed", "model": MODEL, "error": {"type": type(exc).__name__}}
        return self.record(key, raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=original.REPO_ROOT)
    parser.add_argument("--reported-balance", default="19.97")
    parser.add_argument("--historical-cache-repo", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_artifacts(args.source, args.out) if args.verify_only else build_options(args.source, args.out, args.repo, reported_balance=args.reported_balance, cache_repo=args.historical_cache_repo)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
