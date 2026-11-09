"""Build the prespecified degraded-validation request plan; execution disabled.

Only byte/canonical-identical clean requests may reuse frozen outputs. No clean
answer is ever assigned to a changed request. No provider is imported/created.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from benchmark.scripts import phase3c_grounded as grounded
from benchmark.scripts.phase3_common import estimate_tokens_conservative
from benchmark.scripts.phase5_perturb import digest
from benchmark.scripts.phase5_robustness import ROOT, assets, load, read_rows, write, write_rows

CONDITION = "remove_0.5_s20261106"


def payload_for(query, records, clean_payload, clean_records):
    """Preserve exact clean payload only if actual query AND evidence match."""
    clean_query = json.loads(clean_payload["input"].split("TARGET\n", 1)[1].split("\nEVIDENCE\n", 1)[0])["query"]
    if query == clean_query and records == clean_records:
        return copy.deepcopy(clean_payload), True
    payload = copy.deepcopy(clean_payload)
    payload["input"] = grounded.build_user_prompt({"query": query}, [grounded.EvidenceRecord(**r) for r in records])
    return payload, payload == clean_payload


def plan(source, out):
    variants = {r["sample_id"]: r for r in read_rows(out / "derived_inputs.jsonl") if r["condition"] == CONDITION}
    ranks = {r["sample_id"]: r for r in read_rows(out / "frozen_rankings.jsonl") if r["condition"] == CONDITION}
    old_requests = {r["sample_id"]: r for r in read_rows(source / "benchmark/phase3/phase3c_validation/requests.jsonl")}
    old_evidence = {r["sample_id"]: r["records"] for r in read_rows(source / "benchmark/phase3/phase3c_validation/frozen_evidence.jsonl")}
    if len(variants) != 163 or set(variants) != set(ranks) or set(variants) != set(old_requests):
        raise ValueError("LLM plan requires exact 163 aligned validation cases")
    catalog, _ = assets(source)
    config = load(source / "benchmark/phase3/phase3c_validation/config.json")
    prediction_digest = load(out / "prediction_freeze.json")["sha256"]
    requests, public = [], []
    for sid in sorted(variants):
        v, ranking = variants[sid], ranks[sid]
        bm, trained = [{value: i for i, value in enumerate(ranking[method], 1)} for method in ("bm25", "trained")]
        records = []
        for position, kid in enumerate(ranking["fusion"][:10], 1):
            raw = catalog[kid]
            provenance = {**config["retrieval"], "fusion_path": "phase5 recomputed validation rankings",
                          "fusion_sha256": prediction_digest, "query_digest": v["query_digest"]}
            records.append({"evidence_id": f"E{position:02}", "kegg_id": kid, "fused_rank": position,
                            "fused_score": round(sum(1/(60+m[kid]) for m in (bm, trained) if kid in m), 12),
                            "bm25_rank": bm.get(kid), "trained_biencoder_rank": trained.get(kid),
                            **{output: str(raw.get(field) or "") for output, field in [("name", "NAME"), ("definition", "DEFINITION"), ("equation", "EQUATION"), ("enzyme", "ENZYME"), ("rclass", "RCLASS"), ("brite", "BRITE")]},
                            "provenance": provenance})
        payload, compatible = payload_for(v["query"], records, old_requests[sid]["payload"], old_evidence[sid])
        estimated = estimate_tokens_conservative(payload["instructions"] + "\n" + payload["input"])
        requests.append({"sample_id": sid, "model_id": v["model_id"], "reaction_id": v["reaction_id"],
                         "condition": CONDITION, "split": "validation", "payload": payload,
                         "canonical_payload_digest": digest(payload), "compatible_clean_response": compatible,
                         "estimated_input_tokens": estimated, "evidence": records})
        public.append({k: value for k, value in requests[-1].items() if k not in {"payload", "evidence"}})
    request_dir = out / "_requests"
    request_dir.mkdir(exist_ok=True)
    write_rows(request_dir / "primary_grounded_plan.jsonl", requests)
    write_rows(out / "llm_request_manifest.jsonl", public)
    prices = load(ROOT / "benchmark/phase5/pricing.openai.gpt-5.6-terra.20261106.json")
    missing = [r for r in requests if not r["compatible_clean_response"]]
    input_estimate = sum(r["estimated_input_tokens"] for r in missing)
    # Output expectation is a planning assumption informed by historical mean,
    # not a fabricated output for any degraded case. No cache discount assumed.
    typical_output = 23612/163
    estimate = (input_estimate*prices["input_per_million"] + len(missing)*typical_output*prices["output_per_million"])/1e6
    max_output_estimate = (input_estimate*prices["input_per_million"] + len(missing)*2048*prices["output_per_million"])/1e6
    bound = len(missing)*(.090112)
    summary = {"schema": "phase5-grounded-validation-plan-v1", "condition": CONDITION, "n_validation": 163,
               "new_calls_needed": len(missing), "compatible_clean_cache_hits": len(requests)-len(missing),
               "new_calls_made": 0, "model": config["model"], "max_output_tokens": 2048, "reasoning_effort": "low",
               "tools": [], "automatic_retries": 0, "test_access": "disabled", "execution_status": "dry-run only; paid calls not authorized",
               "expected_value": "measure paired clean/degraded grounded correctness, coverage and abstention for 50% identifier removal instead of inferring it from retrieval",
               "prices": prices, "input_token_estimate": input_estimate,
               "estimated_cost_usd_at_historical_mean_output": estimate,
               "estimated_cost_usd_at_max_output_tokens": max_output_estimate,
               "estimates_not_guarantees": True, "assumed_output_tokens_per_call": typical_output,
               "conservative_32768_input_reservation_total_usd": bound, "proposed_hard_cap_usd": 15,
               "regional_modifiers_not_included": True, "currently_authorized_usd": 0,
               "requested_decision": "authorize this validation-only 163-row plan with at most its missing new calls, one attempt each, and a USD 15 hard ceiling; retain selected model and all fixed settings",
               "payload_byte_limit": 32768, "payloads_exceeding_byte_limit": sum(len(json.dumps(r["payload"], ensure_ascii=False).encode("utf-8")) > 32768 for r in requests),
               "command": "python -m benchmark.scripts.phase5_llm_plan --source-root SOURCE_ROOT --out benchmark/phase5/validation --dry-run",
               "live_command_status": "no live command exposed in this task; integrate approved frozen plan into a separately authorized persistent budget/cache runner before any paid execution"}
    write(out / "llm_dry_run.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute:
        raise SystemExit("Paid execution disabled; validation plan and budget require separate authorization.")
    if not args.dry_run:
        parser.error("--dry-run is required")
    plan(args.source_root.resolve(), args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
