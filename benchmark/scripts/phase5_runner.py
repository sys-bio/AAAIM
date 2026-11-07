"""Frozen validation-only Phase 5 execution. Dry-run unless explicitly authorized.

No catalog, reference labels, population loader or historical response cache is used.
The immutable package is distinct from the preserved matched_prompt_v2 artifacts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from benchmark.scripts import phase3c_grounded as grounded
from benchmark.scripts.phase5_billing import request_input_token_bound, request_reservation_nusd
from benchmark.scripts.phase5_llm_plan import PINNED_SDK_VERSION, PINNED_PYDANTIC_VERSION, PROMPT_VERSION, CONDITION, provider_payload, split_input
from benchmark.scripts.phase5_perturb import digest

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "https://api.openai.com/v1"
MODEL = "gpt-5.6-terra"
GPT61_MODEL = "gpt-6.1-terra"
SUPPORTED_MODELS = {MODEL, GPT61_MODEL}
UNIT_POLICY = "one attempt per (condition,sample_id); own completed result only; no cross-unit body deduplication"
PRICING = {
    "verified_date": "2026-11-06", "currency": "USD", "standard_input_per_million": 2,
    "standard_output_per_million": 12, "standard_cache_read_per_million": .2,
    "implicit_30m_write_per_million": 2.5, "fast_multiplier": 2,
    "long_context_threshold": 272000, "long_context_input_multiplier": 2, "long_context_output_multiplier": 1.5,
    "max_input_tokens": 922000, "max_output_tokens_including_reasoning": 2048,
    "source": "https://developers.openai.com/api/docs/pricing",
    "model_source": "https://developers.openai.com/api/docs/models/gpt-5.6-terra",
    "cache_source": "https://developers.openai.com/api/docs/guides/prompt-caching",
    "output_bound_source": "https://developers.openai.com/api/docs/guides/reasoning",
    "assumptions": "global endpoint; Standard or at most Fast; implicit default 30m caching; no tools, explicit cache writes, regional processing or account-specific surcharges; reverify before live authorization",
}
PRICING_BY_MODEL = {
    MODEL: PRICING,
    GPT61_MODEL: {**PRICING, "standard_input_per_million": 1, "standard_output_per_million": 5,
        "standard_cache_read_per_million": .05, "implicit_30m_write_per_million": 1.25,
        "source": "https://developers.openai.com/api/docs/pricing",
        "model_source": "https://developers.openai.com/api/docs/models/gpt-6.1-terra"},
}
# Integer nano-dollars avoid floating-point budget errors. Reserve the entire
# documented model input maximum, NOT a chars/tokens estimate or guessed wrapper
# allowance. Includes implicit cache writes and unexpected Fast processing.
RESERVATION_NUSD_BY_MODEL = {model: int(922000 * pricing["implicit_30m_write_per_million"] * 4 * 1000
        + 2048 * pricing["standard_output_per_million"] * 3 * 1000)
        for model, pricing in PRICING_BY_MODEL.items()}  # max input/write/Fast/long context and output/reasoning
RESERVATION_NUSD = RESERVATION_NUSD_BY_MODEL[MODEL]  # retained GPT-5.6 plan constant
CEILING_NUSD = 40_000_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def native(path):
    """Keep atomic hashed-cache filenames usable in deeply nested Windows worktrees."""
    path = Path(path)
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path.resolve()))
    return path


def read(path):
    return json.loads(native(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in native(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(path):
    return hashlib.sha256(native(path).read_bytes()).hexdigest()


def atomic(path, obj):
    """Flush file contents before atomic replacement; unique temp names, same FS."""
    path = native(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def immutable(path, obj):
    if native(path).exists():
        raise ValueError(f"preserve existing artifact: {path}")
    atomic(path, obj)


@contextmanager
def exclusive(path):
    """OS releases the advisory lock after crash; never delete a lock pathname."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if f.seek(0, os.SEEK_END) == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            lock, unlock = lambda: msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1), lambda: msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock, unlock = lambda: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB), lambda: fcntl.flock(f, fcntl.LOCK_UN)
        try:
            lock()
        except OSError as exc:
            raise RuntimeError("another execution/reconciliation holds the plan lock") from exc
        try:
            yield
        finally:
            f.seek(0)
            unlock()


def unit_key(condition, sample_id, body_digest, requested_model=MODEL):
    return digest({"cache_key_schema": "phase5-unit-v2", "model": requested_model,
                   "condition": condition, "sample_id": sample_id, "provider_body_digest": body_digest})


def settings(body):
    return {k: v for k, v in body.items() if k != "input"}


def build_package(source, destination):
    """Copy exact requests; freeze order, lineage and duplicate policy, no inference."""
    if destination.exists():
        raise ValueError("choose a new runner package; existing packages are preserved")
    full = rows(source / "_requests/clean.jsonl") + rows(source / "_requests/degraded.jsonl")
    public = rows(source / "request_manifest.jsonl")
    sidecars = rows(source / "provenance_sidecars.jsonl")
    if len(full) != 326 or len(public) != 326 or len(sidecars) != 326:
        raise ValueError("exact 326-unit source plan required")
    units, bodies, groups, by_arm = [], {}, defaultdict(list), {}
    for request, manifest, sidecar in zip(full, public, sidecars):
        sid, arm, body = request["sample_id"], request["arm"], request["payload"]
        condition = "clean" if arm == "clean" else CONDITION if arm == "degraded" else None
        body_hash = digest(body)
        if set(body) != {"model", "instructions", "input", "max_output_tokens", "reasoning", "tools", "store", "text"} or body["store"] is not False:
            raise ValueError("unexpected provider fields; reservation/design must be reverified")
        if condition is None or request["split"] != "validation" or request["compatible_historical_clean_body"]:
            raise ValueError("only corrected validation units with new matched baselines are allowed")
        if manifest != {k: v for k, v in request.items() if k not in {"payload", "sidecar"}}:
            raise ValueError("source public manifest does not match complete request")
        if sidecar != {"sample_id": sid, "arm": arm, **request["sidecar"]} or sidecar["condition"] != condition:
            raise ValueError("source provenance sidecar mismatch")
        if body_hash != request["canonical_provider_payload_digest"] or digest(provider_payload(body)) != body_hash:
            raise ValueError("provider-body identity mismatch")
        if body["model"] not in SUPPORTED_MODELS or body["max_output_tokens"] != 2048 or body["reasoning"] != {"effort": "low"} or body["tools"] != []:
            raise ValueError("provider settings drift")
        unit_pricing = PRICING_BY_MODEL[body["model"]]
        body_bytes, input_bound = request_input_token_bound(body, unit_pricing)
        unit_reservation_nusd = request_reservation_nusd(body, unit_pricing, input_bound)
        if (("canonical_provider_body_bytes" in manifest and manifest["canonical_provider_body_bytes"] != body_bytes)
                or ("input_token_reservation_bound" in manifest and manifest["input_token_reservation_bound"] != input_bound)):
            raise ValueError("request billing-bound metadata does not match its complete provider body")
        visible = split_input(body["input"])
        if any("provenance" in e for e in visible["EVIDENCE"]):
            raise ValueError("operational provenance leaked into evidence")
        key = unit_key(condition, sid, body_hash, requested_model=body["model"])
        if (arm, sid) in by_arm:
            raise ValueError("duplicate experimental unit")
        item = {"unit_key": key, "condition": condition, "arm": arm, "sample_id": sid,
                "model_id": request["model_id"], "reaction_id": request["reaction_id"], "split": "validation",
                "provider_body_digest": body_hash, "sidecar_digest": digest(sidecar),
                "canonical_provider_body_bytes": body_bytes, "input_token_bound": input_bound,
                "reservation_nusd": unit_reservation_nusd}
        units.append(item)
        bodies[key] = body
        by_arm[arm, sid] = item
        groups[body_hash].append({"unit_key": key, "condition": condition, "sample_id": sid})
    ids = {u["sample_id"] for u in units}
    if len(ids) != 163 or len(groups) != 317 or set(by_arm) != {(a, s) for a in ("clean", "degraded") for s in ids}:
        raise ValueError("163 matched pairs / 317 distinct bodies required")
    first_settings = settings(next(iter(bodies.values())))
    if any(digest(settings(b)) != digest(first_settings) for b in bodies.values()):
        raise ValueError("all 326 units must have identical provider settings/schema")
    schedule = read(source / "paired_execution_schedule.json")
    order = [by_arm[r["arm"], r["sample_id"]]["unit_key"] for r in schedule]
    if len(order) != 326 or set(order) != set(bodies) or len(set(order)) != 326:
        raise ValueError("execution schedule does not align")
    for i in range(0, 326, 2):
        if units_by_key(units)[order[i]]["sample_id"] != units_by_key(units)[order[i+1]]["sample_id"]:
            raise ValueError("paired calls must remain adjacent")
    # Verify the original deterministic randomized order, not merely adjacency.
    expected = []
    for sid in sorted(ids, key=lambda s: digest([20261106, s])):
        pair = [by_arm[a, sid]["unit_key"] for a in ("clean", "degraded")]
        if int(digest([20261106, sid, "arm_order"])[0], 16) % 2:
            pair.reverse()
        expected += pair
    if order != expected:
        raise ValueError("source order differs from the fixed hash-seeded order")
    destination.mkdir(parents=True)
    atomic(destination / "units.json", units)
    atomic(destination / "order.json", order)
    atomic(destination / "_requests/bodies.json", bodies)
    atomic(destination / "sidecars.json", sidecars)
    atomic(destination / "duplicate_bodies.json", {"units": 326, "distinct_bodies": 317, "policy": UNIT_POLICY,
        "interpretation": "separate stochastic calls, not additional independent biological samples; sample remains 163 paired reactions",
        "groups": [{"body_digest": h, "units": g} for h, g in sorted(groups.items()) if len(g) > 1]})
    inputs = ["request_manifest.jsonl", "provenance_sidecars.jsonl", "paired_execution_schedule.json", "dry_run.json", "_requests/clean.jsonl", "_requests/degraded.jsonl"]
    old_estimate = read(source / "dry_run.json")
    inp, n = old_estimate["input_token_estimate_including_schema"], 326
    selected_model = first_settings["model"]
    if selected_model not in PRICING_BY_MODEL:
        raise ValueError("pricing/risk policy not verified for requested model")
    pricing = PRICING_BY_MODEL[selected_model]
    unit_reservations = [unit["reservation_nusd"] for unit in units]
    reservation_nusd = max(unit_reservations)
    total_reservations_nusd = sum(unit_reservations)
    standard_input = pricing["standard_input_per_million"]
    standard_write = pricing["implicit_30m_write_per_million"]
    standard_output = pricing["standard_output_per_million"]
    policy = {"schema": "phase5-persistent-plan-v2", "cache_key_schema": "phase5-unit-v2", "prompt_version": PROMPT_VERSION, "split": "validation",
        "n_units": n, "n_pairs": 163, "unique_provider_bodies": 317, "attempt_cap": n,
        "settings": first_settings, "settings_digest": digest(first_settings), "endpoint": ENDPOINT,
        "expected_returned_models": [selected_model], "expected_service_tiers": ["default"], "duplicate_policy": UNIT_POLICY,
        "sdk_version": PINNED_SDK_VERSION, "pydantic_version": PINNED_PYDANTIC_VERSION,
        "pricing": pricing, "reservation_nusd": reservation_nusd,
        "reservation_policy": {"input_token_bound": "complete canonical provider JSON body UTF-8 byte count plus 4096 input-token provider-framing allowance; report usage is checked against this per-unit bound",
            "all_input_at_default_30m_cache_write_rate": True, "cache_read_discount_assumed": False,
            "maximum_output_tokens_including_reasoning": 2048, "maximum_service_multiplier_reserved": pricing["fast_multiplier"],
            "long_context_uplift_only_if_per_unit_input_bound_exceeds_threshold": pricing["long_context_threshold"],
            "failed_or_uncertain_attempts": "retain the complete per-unit reservation; if metered usage implies a larger charge, account at the larger amount and stop before any next dispatch"},
        "total_reservations_nusd": total_reservations_nusd, "dollar_cap_nusd": CEILING_NUSD,
        "estimates": {"standard_no_write_mean_output_usd": (inp*standard_input+n*(23612/163)*standard_output)/1e6,
            "standard_all_30m_writes_mean_output_usd": (inp*standard_write+n*(23612/163)*standard_output)/1e6,
            "standard_all_30m_writes_max_output_usd": (inp*standard_write+n*2048*standard_output)/1e6,
            "all_attempts_worst_case_request_bound_reservations_usd": total_reservations_nusd/1e9,
            "maximum_single_request_reservation_usd": reservation_nusd/1e9,
            "maximum_input_token_bound": max(unit["input_token_bound"] for unit in units),
            "maximum_complete_body_bytes": max(unit["canonical_provider_body_bytes"] for unit in units),
            "interpretation": "per-request reservations use the frozen complete body plus explicit framing allowance, maximum output, cache-write pricing and Fast multiplier; verified successful usage releases the excess; failures retain reservations"},
        "source_directory": source.relative_to(ROOT).as_posix(), "source_sha256": {p: sha(source / p) for p in inputs},
        "files_sha256": {p: sha(destination / p) for p in ["units.json", "order.json", "_requests/bodies.json", "sidecars.json", "duplicate_bodies.json"]},
        "runner_source_sha256": sha(Path(__file__)), "currently_authorized_usd": 0, "held_out_execution": "disabled",
        "temporal_limit": "interleaved within-pair randomized order; log identity/tier/time; hidden backend drift and stochasticity remain even with unchanged aliases and request bodies"}
    policy["plan_digest"] = digest(policy)
    atomic(destination / "plan.json", policy)
    atomic(destination / "authorization.template.json", authorization_template(policy))
    return policy


def units_by_key(units):
    return {u["unit_key"]: u for u in units}


def load_package(path):
    plan = read(path / "plan.json")
    if digest({k: v for k, v in plan.items() if k != "plan_digest"}) != plan["plan_digest"]:
        raise ValueError("altered plan digest")
    if sha(Path(__file__)) != plan["runner_source_sha256"]:
        raise ValueError("runner source changed since plan lock")
    if plan["split"] != "validation" or plan["held_out_execution"] != "disabled":
        raise ValueError("held-out execution is disabled")
    for name, expected in plan["files_sha256"].items():
        if sha(path / name) != expected:
            raise ValueError("altered locked artifact: " + name)
    source = ROOT / plan["source_directory"]
    for name, expected in plan["source_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError("stale source plan: " + name)
    units, bodies, order = read(path / "units.json"), read(path / "_requests/bodies.json"), read(path / "order.json")
    if len(units) != plan["n_units"] or len(order) != len(set(order)) or set(order) != set(bodies) or set(order) != set(units_by_key(units)):
        raise ValueError("locked unit/order alignment")
    for u in units:
        b = bodies[u["unit_key"]]
        unit_pricing = PRICING_BY_MODEL.get(b["model"])
        if unit_pricing is None:
            raise ValueError("unsupported model in locked request")
        body_bytes, input_bound = request_input_token_bound(b, unit_pricing)
        expected_reservation = request_reservation_nusd(b, unit_pricing, input_bound)
        if (u["split"] != "validation" or digest(b) != u["provider_body_digest"]
                or u["unit_key"] != unit_key(u["condition"], u["sample_id"], digest(b), requested_model=b["model"])
                or u.get("canonical_provider_body_bytes", body_bytes) != body_bytes
                or ("input_token_bound" in u and u["input_token_bound"] != input_bound)
                or ("reservation_nusd" in u and u["reservation_nusd"] != expected_reservation)):
            raise ValueError("unit cache identity mismatch")
        if digest(settings(b)) != plan["settings_digest"]:
            raise ValueError("locked settings mismatch")
    return plan, units_by_key(units), bodies, order


def authorization_template(plan):
    return {"status": "unapproved", "approved_by": None, "approval_reference": None, "recorded_at": None,
            **{k: plan[k] for k in ["plan_digest", "settings_digest", "attempt_cap", "dollar_cap_nusd", "endpoint", "expected_returned_models"]},
            "model": plan["settings"]["model"], "pricing_digest": digest(plan["pricing"]),
            "billing_assumptions_confirmed": False, "scope": "Phase 5 matched validation only; no held-out execution"}


def authorize(plan, auth, confirmation):
    expected = authorization_template(plan)
    if confirmation != plan["plan_digest"] or auth.get("status") != "approved" or not all(auth.get(k) for k in ("approved_by", "approval_reference", "recorded_at", "billing_assumptions_confirmed")):
        raise PermissionError("explicit recorded authorization and exact plan confirmation required; current authorization is zero")
    for k in expected.keys() - {"status", "approved_by", "approval_reference", "recorded_at", "billing_assumptions_confirmed"}:
        if auth.get(k) != expected[k]:
            raise PermissionError("authorization does not match locked " + k)


def live_provider():
    """Only invoked after authorization and durable reservation, never in dry-run."""
    import openai
    import pydantic
    from openai.types.responses import Response
    if openai.__version__ != PINNED_SDK_VERSION or pydantic.__version__ != PINNED_PYDANTIC_VERSION:
        raise RuntimeError("pinned SDK/serializer required")
    # Explicit endpoint avoids OPENAI_BASE_URL overrides. No implicit .env reads.
    client = openai.OpenAI(base_url=ENDPOINT, max_retries=0, timeout=120)

    def call(body):
        # SDK low-level post dispatches the exact frozen JSON body. No runtime
        # prompt transformation, extra body defaults, parse call or SDK retries.
        return client.post("/responses", body=copy.deepcopy(body), cast_to=Response).model_dump(mode="json")
    return call


def assess(raw, body, plan, input_token_bound=None, reservation_nusd=None):
    flags, annotation = [], None
    identity_ok = raw.get("model") in plan["expected_returned_models"]
    tier = raw.get("service_tier")
    if not identity_ok:
        flags.append("provider_identity_mismatch")
    if tier not in plan["expected_service_tiers"]:
        flags.append("provider_tier_mismatch")
    usage = raw.get("usage")
    accounted, usage_valid = None, False
    if isinstance(usage, dict):
        i, o = usage.get("input_tokens"), usage.get("output_tokens")
        reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
        if input_token_bound is None:
            _, input_token_bound = request_input_token_bound(body, plan["pricing"])
        if reservation_nusd is None:
            reservation_nusd = plan["reservation_nusd"]
        usage_valid = all(type(v) is int and v >= 0 for v in (i, o, reasoning)) and i <= plan["pricing"]["max_input_tokens"] and reasoning <= o <= body["max_output_tokens"]
        request_bound_ok = usage_valid and i <= input_token_bound
        if usage_valid and not request_bound_ok:
            flags.append("request_input_bound_violation")
        if usage_valid and identity_ok and tier in {"default", "priority", "fast"}:
            # Treat ALL input as cache writes at 30m price, no assumed discount.
            # Usage total output already INCLUDES reasoning; never add it twice.
            multiplier = 1 if tier == "default" else 2
            long = i > plan["pricing"]["long_context_threshold"]
            input_rate = plan["pricing"]["implicit_30m_write_per_million"] * 1000
            output_rate = plan["pricing"]["standard_output_per_million"] * 1000
            accounted = i * input_rate * multiplier * (2 if long else 1) + o * output_rate * multiplier * (1.5 if long else 1)
            accounted = int(accounted)
    if not usage_valid:
        flags.append("usage_missing_or_out_of_bound")
    if raw.get("status") != "completed":
        flags.append("provider_response_failure")
    try:
        texts = [part["text"] for item in raw.get("output", []) if item.get("type") == "message"
                 for part in item.get("content", []) if part.get("type") == "output_text"]
        if len(texts) != 1:
            raise ValueError("exactly one structured output expected")
        parsed = grounded.GroundedAnnotation.model_validate_json(texts[0], strict=True)
        evidence = [grounded.EvidenceRecord(**r, provenance={}) for r in split_input(body["input"])["EVIDENCE"]]
        problems = grounded.validate_grounded_annotation(parsed, evidence)
        annotation = parsed.model_dump(mode="json")
        if problems:
            flags.append("evidence_compliance_failure")
    except (ValueError, TypeError, KeyError):
        problems = ["missing/refused/invalid strict structured annotation"]
        flags.append("parse_schema_failure")
    if accounted is not None and accounted > (reservation_nusd or plan["reservation_nusd"]):
        flags.append("reservation_bound_violation")
    return {"status": flags[0] if flags else "succeeded", "flags": flags, "annotation": annotation,
            "compliance_problems": problems, "usage": usage, "accounted_usage_ceiling_nusd": accounted,
            "cost_is_upper_bound_not_invoice": True, "model_returned": raw.get("model"), "service_tier": tier,
            "response_id": raw.get("id"), "pause_required": bool(set(flags) - {"evidence_compliance_failure", "parse_schema_failure", "provider_response_failure"})}


def cache_result(state, entry, package, unit):
    path = state / "cache" / (unit["unit_key"] + ".json")
    if not native(path).exists():
        if entry["status"] == "completed":
            raise RuntimeError("completed cache missing; no repurchase allowed")
        return None
    envelope = read(path)
    if envelope["plan_digest"] != package["plan_digest"] or envelope["unit"] != unit or digest(envelope["result"]) != envelope["result_digest"]:
        raise RuntimeError("incompatible unit cache")
    if entry.get("result_digest") and entry["result_digest"] != envelope["result_digest"]:
        raise RuntimeError("ledger/cache digest mismatch")
    if entry["status"] == "completed":
        result = envelope["result"]
        metered = result.get("accounted_usage_ceiling_nusd")
        if entry.get("reconciliation") or metered is None:
            expected_charge = max(entry.get("charged_nusd", 0), entry["reserved_nusd"]) if entry.get("reconciliation") else entry["reserved_nusd"]
        elif result["status"] == "succeeded":
            expected_charge = metered
        else:
            expected_charge = max(entry["reserved_nusd"], metered)
        if entry.get("charged_nusd") != expected_charge:
            raise RuntimeError("ledger/cache cost mismatch")
    return envelope["result"]


def validate_ledger(ledger, plan, units, order, auth_digest=None, kind=None):
    if ledger["plan_digest"] != plan["plan_digest"] or ledger["order_digest"] != digest(order):
        raise RuntimeError("ledger plan/order mismatch")
    if auth_digest is not None and ledger["authorization_digest"] != auth_digest or kind is not None and ledger["provider_kind"] != kind:
        raise RuntimeError("ledger authorization/provider cohort mismatch")
    keys = [e["unit_key"] for e in ledger["attempts"]]
    if keys != order[:len(keys)] or len(keys) > plan["attempt_cap"]:
        raise RuntimeError("ledger attempt order/limit mismatch")
    for e in ledger["attempts"]:
        if e["unit_key"] not in units or e["reserved_nusd"] != units[e["unit_key"]].get("reservation_nusd", plan["reservation_nusd"]) or e["status"] not in {"reserved", "completed"}:
            raise RuntimeError("invalid ledger reservation")
        charge = e.get("charged_nusd", e["reserved_nusd"])
        if type(charge) is not int or charge < 0:
            raise RuntimeError("invalid ledger accounting")
    if sum(e.get("charged_nusd", e["reserved_nusd"]) for e in ledger["attempts"]) > plan["dollar_cap_nusd"]:
        raise RuntimeError("persistent budget exceeded")


def state_paths(package, plan, synthetic=False):
    # All copies/reproductions of the same production plan share one ledger and
    # OS lock. A second package pathname must not create another purchase cohort.
    base = package if synthetic else ROOT / "benchmark/phase5/_runner_state" / plan["plan_digest"]
    return base / ("_synthetic_run" if synthetic else "run"), base / "_execution.lock"


def run(package, *, execute=False, auth=None, confirmation=None, resume=False, provider_factory=None, synthetic=False):
    plan, units, bodies, order = load_package(package)
    if not execute:
        # Do not construct provider, access secrets, acquire a live lock or create
        # an execution ledger. Validation touches only frozen validation plans.
        return {"mode": "dry-run", "plan_digest": plan["plan_digest"], "planned_units": len(order),
                "distinct_provider_bodies": len({digest(b) for b in bodies.values()}), "provider_constructions": 0,
                "paid_attempts": 0, "currently_authorized_usd": 0, "held_out_rows_read": 0,
                "per_call_reservation_usd": plan["reservation_nusd"]/1e9,
                "total_worst_case_request_bound_reservations_usd": plan.get("total_reservations_nusd", plan["reservation_nusd"]*len(order))/1e9,
                "proposed_cap_usd": plan["dollar_cap_nusd"]/1e9}
    if not synthetic and (plan["schema"] != "phase5-persistent-plan-v2" or plan["cache_key_schema"] != "phase5-unit-v2" or plan["n_units"] != 326 or len({digest(b) for b in bodies.values()}) != 317):
        raise ValueError("live execution requires the exact production 326-unit validation plan")
    authorize(plan, auth or {}, confirmation)
    if provider_factory is not None and not synthetic:
        raise ValueError("injected provider results require an isolated synthetic cohort")
    kind = "synthetic" if synthetic else "live"
    state, lock = state_paths(package, plan, synthetic)
    with exclusive(lock):
        path = state / "ledger.json"
        if path.exists() != resume:
            raise RuntimeError("use --resume for existing state; a new run must have no ledger")
        if not path.exists() and state.exists():
            raise RuntimeError("execution state exists without its ledger; fail closed, no purchases")
        ledger = read(path) if resume else {"plan_digest": plan["plan_digest"], "order_digest": digest(order),
            "authorization_digest": digest(auth), "authorization": auth, "provider_kind": kind, "started_at": now(),
            "endpoint": plan["endpoint"], "sdk_version": PINNED_SDK_VERSION, "attempts": []}
        validate_ledger(ledger, plan, units, order, digest(auth), kind)
        cache_dir = native(state / "cache")
        known = tuple(e["unit_key"] + "." for e in ledger["attempts"])
        if cache_dir.exists() and any(not f.name.startswith(known) for f in cache_dir.iterdir()):
            raise RuntimeError("unledgered cache indicates lost state; fail closed, no purchases")
        atomic(path, ledger)
        provider = None
        for index, key in enumerate(order):
            unit = units[key]
            if index < len(ledger["attempts"]):
                entry = ledger["attempts"][index]
                result = cache_result(state, entry, plan, unit)
                if result is None:
                    raise RuntimeError("interrupted attempt has uncertain provider status; explicit reconciliation required; no retry")
                # Recover the cache-written/ledger-not-completed crash window.
                if entry["status"] == "reserved":
                    complete(entry, result)
                    atomic(path, ledger)
                if result["pause_required"] and not entry.get("reconciliation"):
                    raise RuntimeError("recorded provider/usage/identity uncertainty requires explicit reconciliation")
                continue
            charge = sum(e.get("charged_nusd", e["reserved_nusd"]) for e in ledger["attempts"])
            unit_reservation = unit.get("reservation_nusd", plan["reservation_nusd"])
            if len(ledger["attempts"]) >= plan["attempt_cap"] or charge + unit_reservation > plan["dollar_cap_nusd"]:
                raise RuntimeError("persistent attempt/budget cap stopped before dispatch")
            entry = {"unit_key": key, "status": "reserved", "reserved_nusd": unit_reservation,
                     "started_at": now(), "requested_model": bodies[key]["model"], "provider_body_digest": digest(bodies[key])}
            ledger["attempts"].append(entry)
            atomic(path, ledger)  # Durable reservation BEFORE provider creation/dispatch.
            started = time.perf_counter()
            try:
                if provider is None:
                    provider = (provider_factory or live_provider)()
                raw = provider(copy.deepcopy(bodies[key]))
                result = assess(raw, bodies[key], plan, unit.get("input_token_bound"), unit_reservation)
                result["raw_response"] = raw
            except Exception as exc:
                result = {"status": "provider_failure", "flags": ["provider_failure"], "annotation": None,
                          "error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None),
                          "usage": None, "accounted_usage_ceiling_nusd": None, "model_returned": None,
                          "pause_required": True}
            result.update(started_at=entry["started_at"], finished_at=now(), latency_ms=round((time.perf_counter()-started)*1000, 3),
                          model_requested=bodies[key]["model"], endpoint=plan["endpoint"], sdk_version=PINNED_SDK_VERSION)
            envelope = {"plan_digest": plan["plan_digest"], "unit": unit, "result": result, "result_digest": digest(result)}
            atomic(state / "cache" / (key + ".json"), envelope)
            complete(entry, result)
            atomic(path, ledger)
            if result["pause_required"]:
                raise RuntimeError("attempt recorded; explicit reconciliation required before resume")
        return {"mode": kind, "units_recorded": len(ledger["attempts"]), "charged_ceiling_usd": sum(e["charged_nusd"] for e in ledger["attempts"])/1e9,
                "plan_digest": plan["plan_digest"], "paid_attempts": 0 if synthetic else len(ledger["attempts"])}


def complete(entry, result):
    entry.update(status="completed", result_digest=digest(result), result_status=result["status"], finished_at=result["finished_at"])
    # Provider/parse/compliance failures retain their full reservation, even if
    # usage exists. Only a successful, fully metered response releases the bound.
    metered = result.get("accounted_usage_ceiling_nusd")
    if metered is None:
        entry["charged_nusd"] = entry["reserved_nusd"]
    elif result["status"] == "succeeded":
        entry["charged_nusd"] = metered
    else:
        entry["charged_nusd"] = max(entry["reserved_nusd"], metered)


def reconcile(package, record, *, synthetic=False):
    """Resolve uncertainty conservatively, never permit a second unit attempt.

This v1 supports retaining the entire financial reservation only. Invoice-based
release would require a separately reviewed reconciliation implementation.
"""
    plan, units, _, order = load_package(package)
    if record.get("plan_digest") != plan["plan_digest"] or record.get("decision") != "retain_reservation_no_retry" or not all(record.get(k) for k in ("unit_key", "operator", "reason", "evidence_reference")):
        raise ValueError("explicit documented no-retry reconciliation required")
    state, lock = state_paths(package, plan, synthetic)
    with exclusive(lock):
        ledger = read(state / "ledger.json")
        validate_ledger(ledger, plan, units, order)
        entry = next(e for e in ledger["attempts"] if e["unit_key"] == record["unit_key"])
        if entry.get("reconciliation"):
            raise ValueError("existing reconciliation is immutable")
        result = cache_result(state, entry, plan, units[entry["unit_key"]])
        if result is None:
            result = {"status": "interrupted_unavailable", "flags": ["interrupted_unavailable"], "annotation": None,
                      "usage": None, "accounted_usage_ceiling_nusd": None, "model_returned": None,
                      "model_requested": plan["settings"]["model"], "pause_required": True,
                      "started_at": entry["started_at"], "finished_at": now(), "latency_ms": None}
            atomic(state / "cache" / (entry["unit_key"] + ".json"), {"plan_digest": plan["plan_digest"], "unit": units[entry["unit_key"]], "result": result, "result_digest": digest(result)})
        complete(entry, result)
        entry["charged_nusd"] = max(entry.get("charged_nusd", 0), entry["reserved_nusd"])
        entry["reconciliation"] = {**record, "recorded_at": now()}
        atomic(state / "ledger.json", ledger)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--build-from", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--authorization", type=Path)
    p.add_argument("--confirm-plan")
    p.add_argument("--reconcile-record", type=Path)
    a = p.parse_args(argv)
    if a.build_from:
        if a.execute or a.resume or a.reconcile_record:
            p.error("package building is dry-run only")
        build_package(a.build_from.resolve(), a.package.resolve())
    if a.reconcile_record:
        if a.execute:
            p.error("reconciliation cannot dispatch")
        reconcile(a.package, read(a.reconcile_record))
        return 0
    result = run(a.package, execute=a.execute, resume=a.resume, auth=read(a.authorization) if a.authorization else None, confirmation=a.confirm_plan)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
