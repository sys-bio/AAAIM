"""Matched clean/degraded validation requests, with provenance outside the prompt.

Also reconstruct the real ea241ea clean path and compare every provider field.
Only cached retrieval is read; no provider client, GPU or held-out loader is used.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

from benchmark.scripts import phase3c_grounded as grounded
from benchmark.scripts.phase5_billing import PROVIDER_WRAPPER_ALLOWANCE_TOKENS, canonical_body_bytes
from benchmark.scripts.phase3_common import estimate_tokens_conservative, sha256_portable
from benchmark.scripts.phase5_perturb import digest
from benchmark.scripts.phase5_robustness import ROOT, assets, load, read_rows, write, write_rows, validate_rank_input_alignment

CONDITION = "remove_0.5_s20261106"
PROMPT_VERSION = "phase5-grounded-no-provenance-v2"
PINNED_SDK_VERSION = "1.78.1"
PINNED_PYDANTIC_VERSION = "2.13.5"
VISIBLE_FIELDS = frozenset({"evidence_id", "kegg_id", "fused_rank", "fused_score", "bm25_rank", "trained_biencoder_rank", "name", "definition", "equation", "enzyme", "rclass", "brite"})
NUMERICAL_FIELDS = frozenset({"fused_rank", "fused_score", "bm25_rank", "trained_biencoder_rank"})


def provider_payload(request_spec):
    """Capture the complete SDK JSON body without creating a client or transport.

The historical make_parse_fn drops ONLY prompt_version/schema_version, then
calls Responses.parse(text_format=GroundedAnnotation). Use the installed pinned
SDK's real serializer, including its generated structured-output schema. Input
strings are preserved literally; no model-visible fields are ignored for hashes.
"""
    import openai
    import pydantic
    if openai.__version__ != PINNED_SDK_VERSION or pydantic.__version__ != PINNED_PYDANTIC_VERSION:
        raise RuntimeError("historical provider serialization requires pinned OpenAI/Pydantic versions; reverify the wire-schema contract before changing them")
    from openai.resources.responses import Responses
    from openai._utils import strip_not_given
    resource = object.__new__(Responses)
    resource._post = lambda url, *, body, **kwargs: strip_not_given(body)
    kwargs = {k: copy.deepcopy(v) for k, v in request_spec.items() if k not in {"prompt_version", "schema_version"}}
    # A prepared provider body already contains the explicit structured format.
    if "format" in kwargs.get("text", {}):
        return resource.parse(**kwargs)
    return resource.parse(**kwargs, text_format=grounded.GroundedAnnotation)


def scientific_records(catalog, ranking):
    bm, trained = [{value: i for i, value in enumerate(ranking[method], 1)} for method in ("bm25", "trained")]
    records = []
    for position, kid in enumerate(ranking["fusion"][:10], 1):
        raw = catalog[kid]
        records.append({"evidence_id": f"E{position:02}", "kegg_id": kid, "fused_rank": position,
                        "fused_score": round(sum(1/(60+m[kid]) for m in (bm, trained) if kid in m), 12),
                        "bm25_rank": bm.get(kid), "trained_biencoder_rank": trained.get(kid),
                        **{output: str(raw.get(field) or "") for output, field in [("name", "NAME"), ("definition", "DEFINITION"), ("equation", "EQUATION"), ("enzyme", "ENZYME"), ("rclass", "RCLASS"), ("brite", "BRITE")]}})
    return records


def legacy_evidence(catalog, ranking, variant, config, prediction_digest):
    """Exact original ea241ea construction, retained ONLY as a diagnostic."""
    provenance = {**config["retrieval"], "fusion_path": "phase5 recomputed validation rankings",
                  "fusion_sha256": prediction_digest, "query_digest": variant["query_digest"]}
    return [{**r, "provenance": copy.deepcopy(provenance)} for r in scientific_records(catalog, ranking)]


def visible_evidence(records):
    values = []
    for record in records:
        if set(record) - VISIBLE_FIELDS - {"provenance"} or not VISIBLE_FIELDS.issubset(record):
            raise ValueError("unknown or missing model-visible evidence field")
        values.append({k: copy.deepcopy(record[k]) for k in sorted(VISIBLE_FIELDS)})
    return values


def render_input(query, records):
    return "TARGET\n" + json.dumps({"query": query}, sort_keys=True, ensure_ascii=False) + "\nEVIDENCE\n" + json.dumps(visible_evidence(records), sort_keys=True, ensure_ascii=False)


def split_input(text):
    target, evidence = text.split("\nEVIDENCE\n", 1)
    return {"TARGET": json.loads(target.removeprefix("TARGET\n")), "EVIDENCE": json.loads(evidence)}


def payload_for(query, records, template, clean_records=None, model_override=None):
    """Return a v2 complete provider body and exact historical-body compatibility.

clean_records is accepted for older callers but never shortcuts serialization or
identity checks. Operational metadata cannot mask changes in actual inputs.
"""
    historical = provider_payload(template)
    payload = copy.deepcopy(historical)
    payload["input"] = render_input(query, records)
    if model_override is not None:
        payload["model"] = model_override
    return payload, digest(payload) == digest(historical)


_MISSING = object()


def field_differences(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        return [d for k in sorted(before.keys() | after.keys())
                for d in field_differences(before.get(k, _MISSING), after.get(k, _MISSING), f"{path}.{k}" if path else k)]
    if isinstance(before, list) and isinstance(after, list):
        return [d for i in range(max(len(before), len(after)))
                for d in field_differences(before[i] if i < len(before) else _MISSING,
                                           after[i] if i < len(after) else _MISSING, f"{path}[{i}]")]
    if before is not _MISSING and after is not _MISSING and digest(before) == digest(after):
        return []
    return [{"path": path, "before_present": before is not _MISSING, "after_present": after is not _MISSING,
             "before": None if before is _MISSING else before, "after": None if after is _MISSING else after}]


def categorize(path):
    if ".provenance" in path:
        return "provenance"
    if path.startswith("input.EVIDENCE") and path.rsplit(".", 1)[-1] in NUMERICAL_FIELDS:
        return "numerical"
    if path.startswith("input."):
        return "biological_evidence"
    return "provider_settings"


def compare_payloads(before, after):
    """Complete literal JSON-body equality plus explanatory nested input diffs."""
    raw = field_differences(before, after)
    semantic = [d for d in raw if d["path"] != "input"]
    if before.get("input") != after.get("input"):
        leaves = field_differences(split_input(before["input"]), split_input(after["input"]), "input")
        semantic.extend(leaves or [{"path": "input", "before": before["input"], "after": after["input"],
                                    "before_present": True, "after_present": True, "category": "serialization"}])
    for d in semantic:
        d.setdefault("category", categorize(d["path"]))
    # Do not repeat large prompt strings: literal hashes and expanded leaf values
    # explain them; raw complete bodies remain available in ignored plans.
    literal = [{**d, "before": digest(d["before"]), "after": digest(d["after"]), "values_are_canonical_digests": True}
               if d["path"] == "input" else d for d in raw]
    return {"complete_provider_payload_equal": digest(before) == digest(after), "before_digest": digest(before), "after_digest": digest(after),
            "literal_provider_field_differences": literal, "field_differences": semantic}


def read_context(source, out):
    variants = {(r["condition"], r["sample_id"]): r for r in read_rows(out / "derived_inputs.jsonl") if r["condition"] in {"clean", CONDITION}}
    ranks = {(r["condition"], r["sample_id"]): r for r in read_rows(out / "frozen_rankings.jsonl") if r["condition"] in {"clean", CONDITION}}
    original = {r["sample_id"]: r for r in read_rows(source / "benchmark/phase3/phase3c_validation/requests.jsonl")}
    ids = sorted(original)
    if len(ids) != 163 or set(variants) != {(c, sid) for c in ("clean", CONDITION) for sid in ids} or set(ranks) != set(variants):
        raise ValueError("exact 163 aligned validation cases required for both arms")
    freeze = load(out / "prediction_freeze.json")
    if sha256_portable(out / "frozen_rankings.jsonl") != freeze["sha256"]:
        raise ValueError("frozen retrieval artifact digest mismatch")
    for key in ranks:
        validate_rank_input_alignment(ranks[key], variants[key])
        old = original[key[1]]
        if any(variants[key][k] != old[k] for k in ("model_id", "reaction_id")):
            raise ValueError("experimental-unit identity mismatch")
    catalog, _ = assets(source)
    return {"ids": ids, "variants": variants, "ranks": ranks, "original": original, "catalog": catalog,
            "config": load(source / "benchmark/phase3/phase3c_validation/config.json"), "prediction_digest": freeze["sha256"],
            "study_config": load(ROOT / "benchmark/phase5/study_config.json")}


def clean_control(context):
    rows, categories, extra = [], Counter(), []
    equality = Counter()
    for sid in context["ids"]:
        variant, ranking = context["variants"]["clean", sid], context["ranks"]["clean", sid]
        historical = context["original"][sid]["payload"]
        records = legacy_evidence(context["catalog"], ranking, variant, context["config"], context["prediction_digest"])
        spec = copy.deepcopy(historical)
        spec["input"] = grounded.build_user_prompt({"query": variant["query"]}, [grounded.EvidenceRecord(**r) for r in records])
        diff = compare_payloads(provider_payload(historical), provider_payload(spec))
        old_records = split_input(historical["input"])["EVIDENCE"]
        equality["candidate_ids_and_order"] += [r["kegg_id"] for r in old_records] == [r["kegg_id"] for r in records]
        for field in VISIBLE_FIELDS:
            equality[field] += [r[field] for r in old_records] == [r[field] for r in records]
        diff.update(sample_id=sid, legacy_request_spec_digest=digest(spec))
        categories.update(d["category"] for d in diff["field_differences"])
        if any(d["category"] != "provenance" for d in diff["field_differences"]):
            extra.append(sid)
        rows.append(diff)
    return {"schema": "phase5-real-clean-control-v1", "n": len(rows), "complete_equal_count": sum(r["complete_provider_payload_equal"] for r in rows),
            "differing_cases": sum(not r["complete_provider_payload_equal"] for r in rows),
            "leaf_difference_counts": {k: categories[k] for k in ("provenance", "numerical", "biological_evidence", "provider_settings", "serialization")},
            "cases_with_equal_candidate_fields": dict(equality), "cases_with_non_provenance_differences": extra, "cases": rows,
            "definition": "actual ea241ea construction, complete SDK JSON bodies, literal input strings plus expanded field-level diagnostics; no numerical tolerances or ignored provider inputs"}


def matched_requests(context, model_override=None):
    requests = []
    for arm, condition in [("clean", "clean"), ("degraded", CONDITION)]:
        for sid in context["ids"]:
            v, r = context["variants"][condition, sid], context["ranks"][condition, sid]
            records = scientific_records(context["catalog"], r)
            payload, compatible = payload_for(v["query"], records, context["original"][sid]["payload"], model_override=model_override)
            sidecar = {"prompt_version": PROMPT_VERSION, "condition": condition, "query_digest": v["query_digest"],
                       "sdk_version": PINNED_SDK_VERSION, "pydantic_version": PINNED_PYDANTIC_VERSION,
                       "response_schema_digest": digest(payload["text"]["format"]),
                       "ranking_artifact": "benchmark/phase5/validation/frozen_rankings.jsonl",
                       "ranking_artifact_sha256_portable": context["prediction_digest"],
                       "ranking_key": {"condition": condition, "sample_id": sid},
                       "catalog_sha256": context["config"]["retrieval"]["catalog_sha256"],
                       "checkpoint_sha256": context["study_config"]["trained_checkpoint_sha256"],
                       "retrieval_settings": {"rrf_k": 60, "component_depth": 100, "evidence_depth": 10},
                       "model_requested": payload["model"], "hosted_model_snapshot": "not immutable; cohort drift must be monitored"}
            complete_body_bytes = canonical_body_bytes(payload)
            requests.append({"sample_id": sid, "arm": arm, "model_id": v["model_id"], "reaction_id": v["reaction_id"],
                             "split": "validation", "payload": payload, "canonical_provider_payload_digest": digest(payload),
                             "canonical_provider_body_bytes": complete_body_bytes,
                             "input_token_reservation_bound": complete_body_bytes + PROVIDER_WRAPPER_ALLOWANCE_TOKENS,
                             "cache_key": digest({"experimental_unit": [v["model_id"], v["reaction_id"]], "provider_payload": payload}),
                             "compatible_historical_clean_body": compatible,
                             "estimated_input_tokens": estimate_tokens_conservative(payload["instructions"] + "\n" + payload["input"] + json.dumps(payload["text"], ensure_ascii=False)),
                             "sidecar": sidecar})
    pairs = {sid: [r for r in requests if r["sample_id"] == sid] for sid in context["ids"]}
    for sid, pair in pairs.items():
        clean, degraded = pair
        if {k: v for k, v in clean["payload"].items() if k != "input"} != {k: v for k, v in degraded["payload"].items() if k != "input"}:
            raise ValueError(f"unmatched provider settings: {sid}")
    return requests


def plan(source, out, plan_name="matched_prompt_v2", publish_current=True, model_override=None):
    context = read_context(source, out)
    destination = out / plan_name
    if destination.exists():
        raise ValueError("choose a new plan name; existing plans are preserved")
    destination.mkdir(parents=True)
    diagnostic = clean_control(context)
    write(destination / "clean_control_diagnostic.json", diagnostic)
    flat = [{"sample_id": r["sample_id"], "category": d["category"], "field": d["path"],
             "before": json.dumps(d["before"], ensure_ascii=False), "after": json.dumps(d["after"], ensure_ascii=False),
             "before_present": d["before_present"], "after_present": d["after_present"]}
            for r in diagnostic["cases"] for d in r["field_differences"]]
    with (destination / "clean_control_differences.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    requests = matched_requests(context, model_override=model_override)
    corrected_cases = [{"sample_id": r["sample_id"], **compare_payloads(
        provider_payload(context["original"][r["sample_id"]]["payload"]), r["payload"])}
        for r in requests if r["arm"] == "clean"]
    corrected_categories = Counter(d["category"] for r in corrected_cases for d in r["field_differences"])
    write(destination / "corrected_clean_control_diagnostic.json", {
        "schema": "phase5-v2-real-clean-control-v1", "prompt_version": PROMPT_VERSION,
        "n": len(corrected_cases), "complete_equal_count": sum(r["complete_provider_payload_equal"] for r in corrected_cases),
        "leaf_difference_counts": dict(corrected_categories), "cases": corrected_cases,
        "definition": "actual matched_requests clean-arm complete provider bodies versus historical requests; all provider fields and literal input strings retained"})
    pair_diffs = []
    for sid in context["ids"]:
        clean, degraded = [r for r in requests if r["sample_id"] == sid]
        differences = compare_payloads(clean["payload"], degraded["payload"])
        if any(d["category"] in {"provider_settings", "provenance"} for d in differences["field_differences"]):
            raise ValueError("paired requests differ in an unintended field")
        pair_diffs.append({"sample_id": sid, **differences})
    write_rows(destination / "matched_pair_differences.jsonl", pair_diffs)
    request_dir = destination / "_requests"
    request_dir.mkdir()
    for arm in ("clean", "degraded"):
        write_rows(request_dir / f"{arm}.jsonl", [r for r in requests if r["arm"] == arm])
    public = [{k: v for k, v in r.items() if k not in {"payload", "sidecar"}} for r in requests]
    write_rows(destination / "request_manifest.jsonl", public)
    write_rows(destination / "provenance_sidecars.jsonl", [{"sample_id": r["sample_id"], "arm": r["arm"], **r["sidecar"]} for r in requests])
    # Pair order is deterministic and interleaved, with order within each pair
    # balanced by a fixed hash; no arm/time metadata enters provider input.
    schedule = []
    for sid in sorted(context["ids"], key=lambda s: digest([20261106, s])):
        pair = [r for r in requests if r["sample_id"] == sid]
        if int(digest([20261106, sid, "arm_order"])[0], 16) % 2:
            pair.reverse()
        schedule.extend({"sample_id": r["sample_id"], "arm": r["arm"], "cache_key": r["cache_key"]} for r in pair)
    write(destination / "paired_execution_schedule.json", schedule)
    selected_model = requests[0]["payload"]["model"]
    if selected_model == "gpt-6.1-terra":
        prices = {"source": "https://developers.openai.com/api/docs/pricing",
                  "model_source": "https://developers.openai.com/api/docs/models/gpt-6.1-terra",
                  "verified_date_local": "2026-11-06", "model": selected_model,
                  "service_tier": "Standard", "currency": "USD", "input_per_million": 1.0,
                  "cached_input_per_million": 0.05, "output_per_million": 5.0,
                  "implicit_30m_cache_write_per_million": 1.25, "implicit_30m_write_per_million": 1.25,
                  "short_context_max_input_tokens": 272000, "fast_multiplier": 2,
                  "long_context_threshold": 272000, "long_context_input_multiplier": 2,
                  "long_context_output_multiplier": 1.5, "max_input_tokens": 922000,
                  "max_output_tokens_including_reasoning": 2048,
                  "source_note": "Official public Standard pricing; account-level access and modifiers are not verified. Per-request reservations use complete body bytes plus a 4,096-token framing allowance, default 30-minute cache-write input pricing, Fast multiplier, maximum output, and long-context uplifts only if the request-specific bound crosses the published threshold."}
    else:
        prices = load(ROOT / "benchmark/phase5/pricing.openai.gpt-5.6-terra.20261106.json")
        prices = {**prices, "implicit_30m_cache_write_per_million": prices.get("implicit_30m_write_per_million", 2.5),
                  "implicit_30m_write_per_million": prices.get("implicit_30m_write_per_million", 2.5),
                  "fast_multiplier": 2, "long_context_threshold": 272000,
                  "long_context_input_multiplier": 2, "long_context_output_multiplier": 1.5,
                  "max_input_tokens": 922000, "max_output_tokens_including_reasoning": 2048}
    pending = [r for r in requests if not r["compatible_historical_clean_body"]]
    input_estimate = sum(r["estimated_input_tokens"] for r in pending)
    mean_output = 23612/163
    input_rate, output_rate = prices["input_per_million"], prices["output_per_million"]
    mean_cost = (input_estimate * input_rate + len(pending)*mean_output*output_rate)/1e6
    max_output_cost = (input_estimate * input_rate + len(pending)*2048*output_rate)/1e6
    cache_write = prices.get("implicit_30m_cache_write_per_million", input_rate)
    cached_cost = (input_estimate * cache_write + len(pending)*mean_output*output_rate)/1e6
    cached_max_cost = (input_estimate * cache_write + len(pending)*2048*output_rate)/1e6
    from benchmark.scripts.phase5_billing import request_reservation_nusd
    per_request_reservations = [request_reservation_nusd(
        request["payload"], prices, request["input_token_reservation_bound"])
        for request in pending]
    bounded_reservation_total = sum(per_request_reservations)
    maximum_reservation_per_call = max(per_request_reservations)
    unique_bodies = len({r["canonical_provider_payload_digest"] for r in requests})
    summary = {"schema": "phase5-matched-grounded-plan-v2", "prompt_version": PROMPT_VERSION,
               "n_validation_pairs": 163, "n_clean_requests": 163, "n_degraded_requests": 163,
               "degraded_condition": CONDITION, "new_calls_needed": len(pending), "attempt_cap": len(pending),
               "compatible_historical_clean_cache_hits": len(requests)-len(pending),
               "unique_provider_bodies": unique_bodies,
               "cross_unit_deduplication": f"not applied; retain Phase 3C experimental-unit cache policy ({unique_bodies} unique bodies, {len(requests)} case/arm requests); duplicate prompts do not establish independent biological observations",
               "cache_scope": "experimental unit plus complete provider body; execution cohort/model-identity gate required",
               "new_calls_made": 0, "currently_authorized_usd": 0, "automatic_retries": 0, "test_access": "disabled",
               "model": selected_model, "max_output_tokens": 2048, "reasoning_effort": "low", "tools": [],
               "persistent_runner_package": "persistent_runner_gpt61_v2" if selected_model == "gpt-6.1-terra" else "persistent_runner_v1",
               "sdk_serializer_version": importlib.metadata.version("openai"), "prices": prices,
               "pydantic_serializer_version": importlib.metadata.version("pydantic"),
               "input_token_estimate_including_schema": input_estimate,
               "estimated_cost_usd_at_historical_mean_output": mean_cost,
               "estimated_cost_usd_at_max_output_tokens": max_output_cost,
               "standard_all_30m_cache_writes_mean_output_usd": cached_cost,
               "standard_all_30m_cache_writes_max_output_usd": cached_max_cost,
               "persistent_runner_complete_request_bound_reservation_total_usd": bounded_reservation_total/1e9,
               "persistent_runner_complete_request_bound_maximum_reservation_per_call_usd": maximum_reservation_per_call/1e9,
               "canonical_provider_body_bytes_total": sum(r["canonical_provider_body_bytes"] for r in requests),
               "canonical_provider_body_bytes_maximum": max(r["canonical_provider_body_bytes"] for r in requests),
               "input_token_reservation_bound_total": sum(r["input_token_reservation_bound"] for r in requests),
               "input_token_reservation_bound_maximum": max(r["input_token_reservation_bound"] for r in requests),
               "provider_wrapper_allowance_tokens_per_request": PROVIDER_WRAPPER_ALLOWANCE_TOKENS,
               "assumed_output_tokens_per_call": mean_output, "estimates_not_guarantees": True,
               "conservative_32768_input_reservation_total_usd": len(pending)*(32768*input_rate + 2048*output_rate)/1e6,
               "proposed_hard_cap_usd": 40, "payload_byte_limit": 32768,
               "payloads_exceeding_byte_limit": sum(len(json.dumps(r["payload"], ensure_ascii=False).encode("utf-8")) > 32768 for r in requests),
               "request_manifest": f"{plan_name}/request_manifest.jsonl", "status": "dry-run only; paid execution disabled",
               "historical_clean_not_a_matched_baseline": True,
               "numerical_drift_policy": "retain actual recomputed ranks/scores in both arms; no historical value substitution or tolerance-based cache equivalence",
               "temporal_drift_limit": "identical request bodies are necessary, not sufficient, for compatible hosted behavior; run paired arms interleaved in one cohort, record returned model/SDK/endpoint and pause for identity drift; historical response times differ",
               "requested_decision": f"separate authorization for {len(pending)} validation calls across matched clean/degraded arms, one attempt each, USD 40 hard cap; no held-out evaluation",
               "command": "python -m benchmark.scripts.phase5_llm_plan --dry-run --source-root SOURCE_ROOT --out benchmark/phase5/validation --plan-name NEW_PLAN_NAME"}
    write(destination / "dry_run.json", summary)
    control_text = ["# Phase 5 clean-control diagnostic and corrected paired design", "",
        "The actual original ea241ea construction was run on all 163 unperturbed inputs; all reconstructed request-spec digests match independently captured original-builder digests in superseded/ea241ea/legacy_clean_request_spec_digests.json.",
        f"Complete historical provider-body matches: {diagnostic['complete_equal_count']}/163. All body fields, literal input strings, candidate fields/order, component ranks and scores are checked. SDK 1.78.1 is intercepted before transport; its structured-output JSON schema is included. Only the two internal version tags excluded by the historical runner itself are omitted from provider bodies.", "",
        "## Differing fields", "", "```json", json.dumps({k: v for k, v in diagnostic.items() if k not in {"cases", "definition"}}, indent=2), "```", ""]
    for case in diagnostic["cases"]:
        other = [d for d in case["field_differences"] if d["category"] != "provenance"]
        if other:
            control_text += [case["sample_id"], "", "```json", json.dumps(other, indent=2), "```", ""]
    control_text += ["The 4,890 provenance leaf changes are new fusion path, fusion artifact digest and query digest in each of 1,630 records. No biological field, candidate ID/order or provider setting changed in the legacy clean diagnostic. P3P0096 retains its recomputed rank 69 and score 0.020909832721 at E07; these are not silently replaced with historical values.",
        "Full leaf differences, including missing-versus-null flags, are in clean_control_differences.csv and clean_control_diagnostic.json. Literal input-string digest differences are also recorded, so semantic or tolerance comparisons never establish false complete-body equality.", "",
        "## Corrected design", "",
        f"Prompt phase5-grounded-no-provenance-v2 includes only scientific evidence fields. Accurate operational lineage is in provenance_sidecars.jsonl and never copied from historical clean rankings into changed evidence. Both clean and degraded arms are rebuilt from the same frozen Phase 5 retrieval artifact with the same serializer, model ({model_override or context['config']['model']}), instructions, schema and settings. Only actual perturbation-derived target/evidence fields differ within this model's paired arms. No retrieval experiments were rerun.",
        "Historical v1 clean responses are not a matched baseline for v2. Both arms need new outputs. matched_pair_differences.jsonl records all actual paired input changes. Complete provider JSON bodies, including exact input strings and strict schema, determine payload hashes; experimental-unit identity also remains part of the cache policy. Identical operational sidecar changes do not invalidate an unchanged body; actual body changes always do.",
        "corrected_clean_control_diagnostic.json separately compares all 163 actual v2 clean-arm bodies with historical requests and expands every changed field. This is distinct from the legacy diagnostic: v2 removes historical visible provenance instead of substituting new metadata. No complete historical bodies match.",
        f"The {len(requests)} case/arm calls contain {unique_bodies} distinct provider bodies. We retain the existing experimental-unit cache policy rather than silently changing cross-reaction reuse; duplicate prompts do not establish independent biological observations. Identical requests still do not guarantee identical hosted-model behavior: run matched pairs interleaved, record returned model/SDK/endpoint/time and stop on declared identity drift. Even an unchanged returned alias cannot rule out hidden backend changes. Final inference remains development evidence on a reused pilot.", ""]
    (destination / "CLEAN_CONTROL.md").write_text("\n".join(control_text), encoding="utf-8")
    if publish_current:
        write(out / "llm_dry_run.json", summary)
        write_rows(out / "llm_request_manifest.jsonl", public)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan-name", default="matched_prompt_v2")
    parser.add_argument("--model", choices=["gpt-5.6-terra", "gpt-6.1-terra"], default=None)
    parser.add_argument("--no-publish-current", action="store_true", help="preserve current manifest pointers during independent reproduction")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        raise SystemExit("Paid execution disabled; matched validation plan and budget require separate authorization.")
    if not args.dry_run:
        parser.error("--dry-run is required")
    out = args.out.resolve()
    out.relative_to((ROOT / "benchmark/phase5").resolve())
    plan(args.source_root.resolve(), out, args.plan_name, publish_current=not args.no_publish_current, model_override=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
