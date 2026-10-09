"""Prepare or explicitly execute selected Phase 3C on MODEL2507280001.

Default behavior is a zero-call preflight.  Live execution is deliberately gated by
``--execute --confirm-live --max-cost-usd`` and ``OPENAI_API_KEY``.  The provider
request, schema, prompt, model, reasoning effort, and fusion evidence format are the
selected Phase 3C implementation from ``benchmark.scripts.phase3c_grounded``.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import lzma
import os
import pickle
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.scripts import phase3b_release as release
from benchmark.scripts import phase3c_grounded as grounded
from benchmark.scripts.phase3_common import (
    PRICING_OPENAI_TERRA,
    atomic_write_json,
    atomic_write_jsonl,
    estimate_tokens_conservative,
)
from benchmark.scripts.phase3_cost import load_pricing
from benchmark.scripts.phase3_modes import FileCache
from benchmark.scripts.phase3_openai_run import (
    assert_env_file_protected,
    load_dotenv_if_present,
    model_rates,
)


MODEL_ID = "MODEL2507280001"
ARCHIVE_SHA256 = "3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53"
CHECKPOINT_SHA256 = "8773b04f09916889b74c956e708044fe2c653fa764fe73c422ecc376ecae81c1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_plan() -> dict:
    review = pd.read_csv(OUT / "phase3b_fusion_reaction_review.csv").fillna("")
    eligible = review.loc[
        (~review["assessment"].str.startswith("not_applicable"))
        & (review["mapped_participant_fraction"].astype(float) >= 0.75)
    ].copy()
    if len(eligible) != 264:
        raise ValueError(f"expected 264 adequately mapped internal reactions, found {len(eligible)}")

    bm = {row["reaction_id"]: row["ranked_ids"] for row in read_jsonl(OUT / "phase3b_bm25_rankings_top100.jsonl")}
    trained = {row["reaction_id"]: row["ranked_ids"] for row in read_jsonl(OUT / "phase3b_trained_epoch1_rankings_top100.jsonl")}
    fusion = {row["reaction_id"]: row["ranked_ids"] for row in read_jsonl(OUT / "phase3b_fusion_rankings_top100.jsonl")}
    catalog = pickle.loads(lzma.open(ROOT / "data" / "kegg" / "kegg_reaction_features.lzma", "rb").read())
    provenance = {
        "schema": grounded.EVIDENCE_SCHEMA,
        "method": release.FUSION_METHOD,
        "rrf_k": release.RRF_K,
        "catalog_size": len(catalog),
        "selected_checkpoint_sha256": CHECKPOINT_SHA256,
        "selected_inference_archive_sha256": ARCHIVE_SHA256,
        "query_template": "phase3-retrieval-query-v1",
        "document_template": "phase3-retrieval-document-v1",
    }
    pricing = load_pricing(PRICING_OPENAI_TERRA)
    rates = model_rates(pricing, grounded.MODEL)
    samples = []
    evidence_rows = []
    request_rows = []
    planned = []
    for position, row in enumerate(eligible.sort_values("reaction_id").to_dict("records"), 1):
        rid = str(row["reaction_id"])
        bm_rank = {kid: rank for rank, kid in enumerate(bm[rid], 1)}
        tr_rank = {kid: rank for rank, kid in enumerate(trained[rid], 1)}
        evidence = []
        for fused_rank, kid in enumerate(fusion[rid][:grounded.TOP_K], 1):
            fields = catalog[kid]
            score = (1 / (release.RRF_K + bm_rank[kid]) if kid in bm_rank else 0.0)
            score += (1 / (release.RRF_K + tr_rank[kid]) if kid in tr_rank else 0.0)
            evidence.append(grounded.EvidenceRecord(
                evidence_id=f"E{fused_rank:02d}",
                kegg_id=kid,
                fused_rank=fused_rank,
                fused_score=round(score, 12),
                bm25_rank=bm_rank.get(kid),
                trained_biencoder_rank=tr_rank.get(kid),
                name=str(fields.get("NAME") or ""),
                definition=str(fields.get("DEFINITION") or ""),
                equation=str(fields.get("EQUATION") or ""),
                enzyme=str(fields.get("ENZYME") or ""),
                rclass=str(fields.get("RCLASS") or ""),
                brite=str(fields.get("BRITE") or ""),
                provenance=provenance,
            ))
        sample = {
            "sample_id": f"MODEL2507280001-P3C-{position:04d}",
            "model_id": MODEL_ID,
            "reaction_id": rid,
            "query": str(row["query_text"]),
        }
        user_prompt = grounded.build_user_prompt(sample, evidence)
        payload = {
            "model": grounded.MODEL,
            "instructions": grounded.SYSTEM_INSTRUCTIONS,
            "input": user_prompt,
            "max_output_tokens": grounded.MAX_OUTPUT_TOKENS,
            "reasoning": {"effort": grounded.REASONING_EFFORT},
            "store": False,
            "tools": [],
            "schema_version": grounded.SCHEMA_VERSION,
            "prompt_version": grounded.PROMPT_VERSION,
        }
        cache_id = grounded._cache_key(payload)
        token_estimate = estimate_tokens_conservative(grounded.SYSTEM_INSTRUCTIONS + "\n" + user_prompt)
        samples.append(sample)
        evidence_rows.append({"sample_id": sample["sample_id"], "reaction_id": rid, "records": [item.to_dict() for item in evidence]})
        request_rows.append({
            "sample_id": sample["sample_id"],
            "reaction_id": rid,
            "cache_id": cache_id,
            "payload": payload,
            "input_tokens_estimate": token_estimate,
            "evidence_sha256": sha256_bytes(grounded.canonical_evidence_bytes(evidence)),
        })
        planned.append({"sample": sample, "evidence": evidence, "payload": payload, "cache_id": cache_id})

    input_tokens = sum(row["input_tokens_estimate"] for row in request_rows)
    max_cost = input_tokens / 1_000_000 * rates["input_per_million"]
    max_cost += len(request_rows) * grounded.MAX_OUTPUT_TOKENS / 1_000_000 * rates["output_per_million"]
    return {
        "samples": samples,
        "evidence_rows": evidence_rows,
        "request_rows": request_rows,
        "planned": planned,
        "preflight": {
            "schema": "model2507280001-phase3c-preflight-v1",
            "api_calls": 0,
            "population": "264 adequately mapped internal reactions; transport/biomass/pseudo and insufficient-mapping rows excluded before requests",
            "n_requests": len(request_rows),
            "one_request_per_reaction": True,
            "model": grounded.MODEL,
            "reasoning_effort": grounded.REASONING_EFFORT,
            "max_output_tokens": grounded.MAX_OUTPUT_TOKENS,
            "retrieval_top_k": grounded.TOP_K,
            "store": False,
            "tools": [],
            "prompt_version": grounded.PROMPT_VERSION,
            "schema_version": grounded.SCHEMA_VERSION,
            "automatic_retries": 0,
            "input_tokens_estimate": input_tokens,
            "worst_case_cost_usd": round(max_cost, 6),
            "pricing_date": pricing.get("pricing_date"),
            "rates": rates,
            "credential_environment_variable": grounded.SECRET_ENV,
            "credential_available_in_process": bool(os.environ.get(grounded.SECRET_ENV)),
            "dotenv_exists": (ROOT / ".env").is_file(),
            "external_call_status": "not_attempted",
            "methodology": "selected grounded_llm_on_every_reaction over Phase 3B fusion Top-10",
        },
    }


def write_preflight(plan: dict) -> None:
    atomic_write_jsonl(plan["samples"], OUT / "phase3c_preflight_samples.jsonl")
    atomic_write_jsonl(plan["evidence_rows"], OUT / "phase3c_preflight_evidence.jsonl")
    atomic_write_jsonl(plan["request_rows"], OUT / "phase3c_preflight_requests.jsonl")
    atomic_write_json(plan["preflight"], OUT / "phase3c_preflight_summary.json")


def execute(plan: dict, max_cost_usd: float) -> tuple[list[dict], dict]:
    cache_dir = OUT / "_phase3c_response_cache"
    cache = FileCache(cache_dir)
    parse = grounded.make_parse_fn()
    pricing = load_pricing(PRICING_OPENAI_TERRA)
    rates = model_rates(pricing, grounded.MODEL)
    responses: list[dict] = []
    spent = 0.0
    calls = 0
    ledger_path = OUT / "phase3c_attempt_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"attempts": []}
    for item in plan["planned"]:
        cached = cache.get(item["cache_id"])
        if cached is not None:
            responses.append(dict(cached))
            spent += float(cached.get("cost_usd") or 0.0)
            continue
        input_estimate = estimate_tokens_conservative(item["payload"]["instructions"] + item["payload"]["input"])
        call_max = input_estimate / 1_000_000 * rates["input_per_million"]
        call_max += grounded.MAX_OUTPUT_TOKENS / 1_000_000 * rates["output_per_million"]
        if spent + call_max > max_cost_usd + 1e-12:
            raise RuntimeError(f"pre-call cost gate stopped at ${spent:.6f}; next-call maximum ${call_max:.6f}; cap ${max_cost_usd:.2f}")
        attempt = {"sample_id": item["sample"]["sample_id"], "cache_id": item["cache_id"], "status": "attempted", "reserved_cost_usd": round(call_max, 8), "actual_cost_usd": None}
        ledger["attempts"].append(attempt)
        atomic_write_json(ledger, ledger_path)
        started = time.perf_counter()
        try:
            response = parse(item["payload"])
            row = grounded._response_row(item, response, pricing)
        except Exception:
            attempt["status"] = "failed"
            atomic_write_json(ledger, ledger_path)
            raise
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        cache.put(item["cache_id"], row)
        attempt["status"] = "completed"
        attempt["actual_cost_usd"] = row["cost_usd"]
        atomic_write_json(ledger, ledger_path)
        responses.append(row)
        spent += float(row.get("cost_usd") or 0.0)
        calls += 1
        atomic_write_jsonl(responses, OUT / "phase3c_grounded_responses.jsonl")
    summary = {"api_calls_this_invocation": calls, "rows": len(responses), "spent_usd_including_cached_calls": round(spent, 8), "cost_cap_usd": max_cost_usd, "model": grounded.MODEL}
    atomic_write_json(summary, OUT / "phase3c_grounded_run_summary.json")
    return responses, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--no-dotenv", action="store_true")
    args = parser.parse_args()
    plan = load_plan()
    write_preflight(plan)
    if not args.execute:
        print(json.dumps(plan["preflight"], indent=2, sort_keys=True))
        return 0
    if not args.confirm_live or args.max_cost_usd is None or args.max_cost_usd <= 0:
        raise RuntimeError("live mode requires --confirm-live and a positive --max-cost-usd")
    assert_env_file_protected(ROOT)
    if not args.no_dotenv:
        load_dotenv_if_present(ROOT)
    if not os.environ.get(grounded.SECRET_ENV):
        raise RuntimeError("OPENAI_API_KEY is required for --execute; no external call was made")
    _, summary = execute(plan, args.max_cost_usd)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
