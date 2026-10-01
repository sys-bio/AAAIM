"""Phase 3C evidence-grounded validation smoke experiment.

The scientific reference path is framework independent: it reads the frozen
Phase 3B fusion ranking, materializes structured KEGG evidence, and makes at
most one stateless OpenAI Responses API call per planned validation reaction.
LangChain is an optional, thin adapter over the exact same retrieval function.

Dry-run is the default.  Live execution requires ``--execute`` and is capped at
five calls, five total attempts (no retries), and USD 0.50.  ``--cache-only``
proves that a completed run can be replayed without network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from benchmark.scripts import phase3_retrieval as retrieval
from benchmark.scripts import phase3b_release as release
from benchmark.scripts.phase3_common import (
    PHASE3_DIR,
    PRICING_OPENAI_TERRA,
    atomic_write_json,
    atomic_write_jsonl,
    estimate_tokens_conservative,
    parse_kegg_ids,
    repo_relative_posix,
    sha256_file,
    sha256_portable,
    write_artifact_manifest,
)
from benchmark.scripts.phase3_cost import load_pricing
from benchmark.scripts.phase3_modes import FileCache
from benchmark.scripts.phase3_openai_run import (
    SECRET_ENV,
    assert_env_file_protected,
    cost_from_usage,
    load_dotenv_if_present,
    model_rates,
    usage_from_response,
)

OUT = PHASE3_DIR / "phase3c_smoke"
CACHE_DIR = OUT / "_response_cache"
SAMPLE_PATH = OUT / "sample.jsonl"
ANSWER_KEY_PATH = OUT / "answer_key.jsonl"
EVIDENCE_PATH = OUT / "frozen_evidence.jsonl"
REQUESTS_PATH = OUT / "frozen_requests.jsonl"
RESPONSES_PATH = OUT / "frozen_responses.jsonl"
RESULTS_PATH = OUT / "eval.json"
MODEL = "gpt-5.6-terra"
MAX_OUTPUT_TOKENS = 2048
REASONING_EFFORT = "low"
MAX_REQUESTS = 5
MAX_ATTEMPTS = 5
MAX_COST_USD = 0.50
TOP_K = 10
SCHEMA_VERSION = "phase3c-grounded-output-v1"
EVIDENCE_SCHEMA = "phase3c-kegg-evidence-v1"
PROMPT_VERSION = "phase3c-grounded-prompt-v1"
SAMPLE_RULE = "lexicographically_first_by_corrected_stratum_and_prespecified_answer_presence"
EXPECTED_FUSION_SHA256 = "9601149e0297831ebfbfb4fde24b725a9b00287cf1b4d74270bf2921be138b31"

STRATUM_PRESENCE = {
    "unconstrained": False,
    "empty_constrained": True,
    "nonempty_answer_absent": False,
    "retrievable_rerank_failure": True,
    "retrievable_top1_success": True,
}


@dataclass(frozen=True)
class RetrievalQuery:
    """Label-free key and target-local text used by the frozen retriever."""

    model_id: str
    reaction_id: str
    text: str


@dataclass(frozen=True)
class EvidenceRecord:
    """A ranked catalog record presented verbatim to the grounded model."""

    evidence_id: str
    kegg_id: str
    fused_rank: int
    fused_score: float
    bm25_rank: Optional[int]
    trained_biencoder_rank: Optional[int]
    name: str
    definition: str
    equation: str
    enzyme: str
    rclass: str
    brite: str
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GroundedAnnotation(BaseModel):
    """Strict provider schema; semantic evidence rules are checked afterward."""

    model_config = ConfigDict(extra="forbid")

    abstain: bool
    predicted_kegg_id: Optional[str] = Field(pattern=r"^R[0-9]{5}$")
    selected_evidence_rank: Optional[int] = Field(ge=1, le=TOP_K)
    supporting_evidence_ids: list[str] = Field(max_length=TOP_K)
    confidence: float = Field(ge=0, le=1)
    reasoning_summary: str
    abstention_reason: Optional[str]


class EvidenceIndex:
    """Read-only adapter around frozen Phase 3B RRF rankings and KEGG catalog."""

    def __init__(self) -> None:
        freeze = json.loads((release.FUSION_OUT / "ranking_freeze.json").read_text(encoding="utf-8"))
        fusion_path = release.FUSION_OUT / release.FUSION_RANKING_NAME
        if freeze.get("sha256") != EXPECTED_FUSION_SHA256 or sha256_file(fusion_path) != EXPECTED_FUSION_SHA256:
            raise ValueError("frozen Phase 3B fusion ranking digest mismatch")
        if not freeze.get("frozen_before_ground_truth_join") or freeze.get("test_rows_read") != 0:
            raise ValueError("fusion ranking does not satisfy the validation-only freeze contract")

        config = json.loads((release.FUSION_OUT / "config.json").read_text(encoding="utf-8"))
        if config.get("rrf", {}).get("k") != 60 or config.get("catalog", {}).get("size") != 12_312:
            raise ValueError("frozen RRF/catalog configuration drift")
        self.provenance = {
            "schema": EVIDENCE_SCHEMA,
            "method": release.FUSION_METHOD,
            "fusion_path": repo_relative_posix(fusion_path),
            "fusion_sha256": EXPECTED_FUSION_SHA256,
            "rrf_k": 60,
            "catalog_size": 12_312,
            "bm25_sha256": config["components"][0]["sha256"],
            "trained_epoch1_sha256": config["components"][1]["sha256"],
            "catalog_sha256": config["catalog"]["source_sha256"],
        }
        self.fusion = {
            (row["model_id"], row["reaction_id"]): row["ranked_ids"]
            for row in release._read_xz_jsonl(fusion_path)
        }
        self.bm25 = release._load_component_maps(release.BM25_RANKINGS, expected_method="bm25")
        self.trained = release._load_component_maps(release.SELECTED_RANKINGS)
        visible = retrieval.load_query_population("validation")
        self.queries = {
            (str(row["model_id"]), str(row["reaction_id"])): retrieval.query_text(row)
            for row in visible.to_dict("records")
        }
        raw = pickle.loads(lzma.open(retrieval.CATALOG, "rb").read())
        if len(raw) != 12_312:
            raise ValueError("frozen KEGG catalog size mismatch")
        self.catalog: Mapping[str, Mapping[str, Any]] = raw

    def query_for(self, model_id: str, reaction_id: str) -> RetrievalQuery:
        key = (model_id, reaction_id)
        if key not in self.queries:
            raise KeyError("query is not in the frozen validation population")
        return RetrievalQuery(model_id=model_id, reaction_id=reaction_id, text=self.queries[key])

    def retrieve(self, query: RetrievalQuery, top_k: int = TOP_K) -> list[EvidenceRecord]:
        if not 1 <= top_k <= TOP_K:
            raise ValueError(f"top_k must be between 1 and {TOP_K}")
        key = (query.model_id, query.reaction_id)
        if key not in self.fusion or self.queries.get(key) != query.text:
            raise ValueError("query does not exactly match the frozen target-local validation query")
        bm_rank = {identifier: rank for rank, identifier in enumerate(self.bm25[key], 1)}
        tr_rank = {identifier: rank for rank, identifier in enumerate(self.trained[key], 1)}
        records: list[EvidenceRecord] = []
        for fused_rank, identifier in enumerate(self.fusion[key][:top_k], 1):
            fields = self.catalog[identifier]
            score = (1.0 / (60 + bm_rank[identifier]) if identifier in bm_rank else 0.0)
            score += (1.0 / (60 + tr_rank[identifier]) if identifier in tr_rank else 0.0)
            records.append(EvidenceRecord(
                evidence_id=f"E{fused_rank:02d}",
                kegg_id=identifier,
                fused_rank=fused_rank,
                fused_score=round(score, 12),
                bm25_rank=bm_rank.get(identifier),
                trained_biencoder_rank=tr_rank.get(identifier),
                name=str(fields.get("NAME") or ""),
                definition=str(fields.get("DEFINITION") or ""),
                equation=str(fields.get("EQUATION") or ""),
                enzyme=str(fields.get("ENZYME") or ""),
                rclass=str(fields.get("RCLASS") or ""),
                brite=str(fields.get("BRITE") or ""),
                provenance=dict(self.provenance),
            ))
        return records


def retrieve_kegg_evidence(
    query: RetrievalQuery, top_k: int = TOP_K, *, index: Optional[EvidenceIndex] = None,
) -> list[EvidenceRecord]:
    """Framework-independent public retrieval interface."""
    return (index or EvidenceIndex()).retrieve(query, top_k)


def canonical_evidence_bytes(evidence: Sequence[EvidenceRecord | Mapping[str, Any]]) -> bytes:
    rows = [item.to_dict() if isinstance(item, EvidenceRecord) else dict(item) for item in evidence]
    return (json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def langchain_retrieval_tool(index: EvidenceIndex):
    """Return a LangChain StructuredTool with no change to retrieval semantics."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("langchain-core==1.6.2 is required for the adapter") from exc

    def _retrieve(model_id: str, reaction_id: str, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        records = retrieve_kegg_evidence(
            RetrievalQuery(model_id=model_id, reaction_id=reaction_id, text=query),
            top_k=top_k,
            index=index,
        )
        return [record.to_dict() for record in records]

    return StructuredTool.from_function(
        func=_retrieve,
        name="retrieve_kegg_evidence",
        description="Retrieve frozen KEGG reaction evidence for one target-local validation query.",
    )


def run_langchain_tool_step(index: EvidenceIndex, query: RetrievalQuery, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Minimal agent-message/tool step used to audit parity without another LLM."""
    from langchain_core.messages import AIMessage

    tool = langchain_retrieval_tool(index)
    agent_message = AIMessage(content="", tool_calls=[{
        "name": tool.name,
        "args": {"model_id": query.model_id, "reaction_id": query.reaction_id, "query": query.text, "top_k": top_k},
        "id": "phase3c-deterministic-tool-call",
        "type": "tool_call",
    }])
    return tool.invoke(agent_message.tool_calls[0]["args"])


def _sample_id(position: int) -> str:
    return f"P3C{position:04d}"


def select_smoke(index: EvidenceIndex) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select labels mechanically, then seal them in a separate answer key."""
    truth = retrieval._truth_and_metadata()
    if not truth.split.eq("validation").all():
        raise ValueError("non-validation row entered Phase 3C selection")
    samples: list[dict[str, Any]] = []
    answer_key: list[dict[str, Any]] = []
    for position, (stratum, desired_presence) in enumerate(STRATUM_PRESENCE.items(), 1):
        selected = None
        for row in truth.loc[truth.stratum.eq(stratum)].sort_values(["model_id", "reaction_id"]).itertuples():
            evidence = index.retrieve(index.query_for(row.model_id, row.reaction_id))
            identifiers = {item.kegg_id for item in evidence}
            present = any(identifier in identifiers for identifier in row.truth)
            if present == desired_presence:
                selected = (row, evidence, present)
                break
        if selected is None:
            raise ValueError(f"no eligible reaction for {stratum} presence={desired_presence}")
        row, evidence, present = selected
        sid = _sample_id(position)
        samples.append({
            "sample_id": sid,
            "model_id": row.model_id,
            "reaction_id": row.reaction_id,
            "cluster_id": row.cluster_id,
            "split": "validation",
            "corrected_stratum": stratum,
            "query": index.queries[(row.model_id, row.reaction_id)],
        })
        hit_rank = next((item.fused_rank for item in evidence if item.kegg_id in set(row.truth)), None)
        answer_key.append({
            "sample_id": sid,
            "model_id": row.model_id,
            "reaction_id": row.reaction_id,
            "ground_truth_ids": list(row.truth),
            "answer_present_in_fused_top10": present,
            "first_ground_truth_evidence_rank": hit_rank,
        })
    return samples, answer_key


SYSTEM_INSTRUCTIONS = """You annotate one biochemical reaction using only the supplied TARGET and EVIDENCE records. Do not use recalled knowledge, external tools, web search, or identifiers absent from EVIDENCE. If you answer, predicted_kegg_id must be the kegg_id at selected_evidence_rank, and supporting_evidence_ids must name one or more supplied evidence_id values including that selected record. If the evidence is insufficient, abstain. Return only the required structured response."""


def build_user_prompt(sample: Mapping[str, Any], evidence: Sequence[EvidenceRecord]) -> str:
    target = {"query": sample["query"]}
    evidence_rows = [item.to_dict() for item in evidence]
    return "TARGET\n" + json.dumps(target, sort_keys=True, ensure_ascii=False) + "\nEVIDENCE\n" + json.dumps(evidence_rows, sort_keys=True, ensure_ascii=False)


def validate_grounded_annotation(annotation: GroundedAnnotation, evidence: Sequence[EvidenceRecord]) -> list[str]:
    problems: list[str] = []
    by_id = {item.evidence_id: item for item in evidence}
    by_rank = {item.fused_rank: item for item in evidence}
    if len(annotation.supporting_evidence_ids) != len(set(annotation.supporting_evidence_ids)):
        problems.append("duplicate supporting_evidence_ids")
    unknown = sorted(set(annotation.supporting_evidence_ids) - set(by_id))
    if unknown:
        problems.append(f"unknown supporting evidence: {unknown}")
    if annotation.abstain:
        if annotation.predicted_kegg_id is not None or annotation.selected_evidence_rank is not None:
            problems.append("abstention must not select an identifier or evidence rank")
        if annotation.supporting_evidence_ids:
            problems.append("abstention must not claim supporting evidence")
        if not (annotation.abstention_reason or "").strip():
            problems.append("abstention_reason is required when abstaining")
    else:
        if annotation.predicted_kegg_id is None or annotation.selected_evidence_rank is None:
            problems.append("non-abstention requires predicted_kegg_id and selected_evidence_rank")
        selected = by_rank.get(annotation.selected_evidence_rank or -1)
        if selected is not None and selected.kegg_id != annotation.predicted_kegg_id:
            problems.append("predicted_kegg_id does not match selected_evidence_rank")
        if selected is not None and selected.evidence_id not in annotation.supporting_evidence_ids:
            problems.append("selected record is absent from supporting_evidence_ids")
        if not annotation.supporting_evidence_ids:
            problems.append("non-abstention requires supporting_evidence_ids")
        if annotation.predicted_kegg_id not in {item.kegg_id for item in evidence}:
            problems.append("prediction is absent from supplied evidence")
        if (annotation.abstention_reason or "").strip():
            problems.append("non-abstention must not provide abstention_reason")
    if not annotation.reasoning_summary.strip():
        problems.append("reasoning_summary is empty")
    return problems


def _cache_key(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def build_plan(index: EvidenceIndex, cache_dir: Path = CACHE_DIR) -> dict[str, Any]:
    samples, answer_key = select_smoke(index)
    cache = FileCache(cache_dir)
    planned = []
    evidence_rows = []
    request_rows = []
    for sample in samples:
        query = index.query_for(sample["model_id"], sample["reaction_id"])
        evidence = index.retrieve(query)
        user = build_user_prompt(sample, evidence)
        payload = {
            "model": MODEL,
            "instructions": SYSTEM_INSTRUCTIONS,
            "input": user,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "reasoning": {"effort": REASONING_EFFORT},
            "store": False,
            "tools": [],
            "schema_version": SCHEMA_VERSION,
            "prompt_version": PROMPT_VERSION,
        }
        key = _cache_key(payload)
        item = {"sample": sample, "query": query, "evidence": evidence, "payload": payload, "cache_id": key}
        planned.append(item)
        evidence_rows.append({"sample_id": sample["sample_id"], "records": [record.to_dict() for record in evidence]})
        request_rows.append({
            "sample_id": sample["sample_id"],
            "cache_id": key,
            "payload": payload,
            "input_tokens_estimate": estimate_tokens_conservative(SYSTEM_INSTRUCTIONS + "\n" + user),
        })
    rates = model_rates(load_pricing(PRICING_OPENAI_TERRA), MODEL)
    pending = [item for item in planned if cache.get(item["cache_id"]) is None]
    input_tokens = sum(estimate_tokens_conservative(item["payload"]["instructions"] + "\n" + item["payload"]["input"]) for item in pending)
    worst = (input_tokens / 1_000_000) * rates["input_per_million"] + (len(pending) * MAX_OUTPUT_TOKENS / 1_000_000) * rates["output_per_million"]
    if len(planned) != MAX_REQUESTS or len(pending) > MAX_ATTEMPTS or worst > MAX_COST_USD:
        raise ValueError("Phase 3C request/attempt/cost cap gate failed")
    return {
        "samples": samples,
        "answer_key": answer_key,
        "planned": planned,
        "evidence_rows": evidence_rows,
        "request_rows": request_rows,
        "dry_run": {
            "schema": "phase3c-dry-run-v1",
            "selection_rule": SAMPLE_RULE,
            "planned_ids_and_strata": [{"sample_id": s["sample_id"], "model_id": s["model_id"], "reaction_id": s["reaction_id"], "corrected_stratum": s["corrected_stratum"]} for s in samples],
            "answer_presence_by_sample": {row["sample_id"]: row["answer_present_in_fused_top10"] for row in answer_key},
            "n_cache_hits": len(planned) - len(pending),
            "n_pending": len(pending),
            "pending_cache_ids": [item["cache_id"] for item in pending],
            "input_tokens_estimate_pending": input_tokens,
            "max_output_tokens_per_call": MAX_OUTPUT_TOKENS,
            "max_output_tokens_pending": len(pending) * MAX_OUTPUT_TOKENS,
            "worst_case_cost_usd": round(worst, 6),
            "cost_cap_usd": MAX_COST_USD,
            "request_cap": MAX_REQUESTS,
            "attempt_cap": MAX_ATTEMPTS,
            "automatic_retries": 0,
            "api_calls": 0,
            "answer_key_read_for_selection_only": True,
            "answer_key_in_requests": False,
            "test_rows_read": 0,
        },
    }


def write_plan(plan: Mapping[str, Any], out: Path = OUT, *, preserve_dry_run: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(plan["samples"], out / SAMPLE_PATH.name)
    atomic_write_jsonl(plan["answer_key"], out / ANSWER_KEY_PATH.name)
    atomic_write_jsonl(plan["evidence_rows"], out / EVIDENCE_PATH.name)
    atomic_write_jsonl(plan["request_rows"], out / REQUESTS_PATH.name)
    dry_path = out / "dry_run_plan.json"
    if not preserve_dry_run or not dry_path.exists():
        atomic_write_json(plan["dry_run"], dry_path)


def make_parse_fn() -> Callable[[Mapping[str, Any]], Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openai==1.78.1 is required") from exc
    if not os.environ.get(SECRET_ENV):
        raise RuntimeError("OPENAI_API_KEY is required only for --execute")
    client = OpenAI(api_key=os.environ[SECRET_ENV], max_retries=0)

    def _call(payload: Mapping[str, Any]) -> Any:
        kwargs = {key: value for key, value in payload.items() if key not in {"schema_version", "prompt_version"}}
        return client.responses.parse(**kwargs, text_format=GroundedAnnotation)
    return _call


def _response_row(item: Mapping[str, Any], response: Any, pricing: Mapping[str, Any]) -> dict[str, Any]:
    parsed = response.output_parsed
    annotation = parsed if isinstance(parsed, GroundedAnnotation) else GroundedAnnotation.model_validate(parsed)
    problems = validate_grounded_annotation(annotation, item["evidence"])
    usage = usage_from_response(response)
    rates = model_rates(pricing, MODEL)
    max_cost = (estimate_tokens_conservative(item["payload"]["instructions"] + item["payload"]["input"]) / 1_000_000) * rates["input_per_million"] + (MAX_OUTPUT_TOKENS / 1_000_000) * rates["output_per_million"]
    cost = cost_from_usage(usage, rates=rates, fallback_usd=max_cost)
    return {
        "sample_id": item["sample"]["sample_id"],
        "cache_id": item["cache_id"],
        "terminal_status": "succeeded" if not problems else "compliance_error",
        "compliance_problems": problems,
        "annotation": annotation.model_dump(mode="json"),
        "model_requested": MODEL,
        "model_returned": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "attempt_count": 1,
        "cache_hit": False,
        "usage": usage,
        "cost_usd": round(float(cost["usd"]), 8),
    }


def _cost_details(rows: Sequence[Mapping[str, Any]], pricing: Mapping[str, Any]) -> dict[str, Any]:
    token_names = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens")
    totals = {name: sum(int((row.get("usage") or {}).get(name) or 0) for row in rows) for name in token_names}
    return {
        "usage_totals": totals,
        "per_sample": [{
            "sample_id": row["sample_id"],
            "cost_usd": row.get("cost_usd"),
            **{name: (row.get("usage") or {}).get(name) for name in token_names},
        } for row in rows],
        "pricing_date": pricing.get("pricing_date"),
        "pricing_source": pricing.get("source"),
        "rates": model_rates(pricing, MODEL),
    }


def run_plan(
    plan: Mapping[str, Any], *, execute: bool, cache_only: bool, cache_dir: Path = CACHE_DIR,
    parse_fn: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    response_path: Optional[Path] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = FileCache(cache_dir)
    ledger_path = cache_dir / "_attempt_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {"attempts": []}
    pricing = load_pricing(PRICING_OPENAI_TERRA)
    rows: list[dict[str, Any]] = []
    calls = attempts = 0
    spent = 0.0
    for item in plan["planned"]:
        cached = cache.get(item["cache_id"])
        if cached is not None:
            row = dict(cached)
            row["cache_hit"] = True
            rows.append(row)
            spent += float(row.get("cost_usd") or 0.0)
            continue
        if cache_only:
            raise RuntimeError(f"cache-only replay would call the API for {item['cache_id']}")
        if not execute:
            continue
        if calls >= MAX_REQUESTS or attempts >= MAX_ATTEMPTS:
            raise RuntimeError("live request/attempt cap reached")
        rates = model_rates(pricing, MODEL)
        call_max = (estimate_tokens_conservative(item["payload"]["instructions"] + item["payload"]["input"]) / 1_000_000) * rates["input_per_million"] + (MAX_OUTPUT_TOKENS / 1_000_000) * rates["output_per_million"]
        prior_attempts = list(ledger.get("attempts") or [])
        prior_reserved = sum(float(x.get("actual_cost_usd") if x.get("actual_cost_usd") is not None else x["reserved_cost_usd"]) for x in prior_attempts)
        if len(prior_attempts) >= MAX_ATTEMPTS:
            raise RuntimeError("persistent total-attempt cap reached")
        if prior_reserved + call_max > MAX_COST_USD + 1e-12:
            raise RuntimeError("pre-call cost cap gate stopped execution")
        if parse_fn is None:
            parse_fn = make_parse_fn()
        attempts += 1
        ledger["attempts"].append({
            "cache_id": item["cache_id"],
            "reserved_cost_usd": round(call_max, 8),
            "actual_cost_usd": None,
            "status": "attempted",
        })
        atomic_write_json(ledger, ledger_path)
        started = time.perf_counter()
        response = parse_fn(item["payload"])
        calls += 1
        row = _response_row(item, response, pricing)
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        cache.put(item["cache_id"], row)
        ledger["attempts"][-1]["actual_cost_usd"] = row["cost_usd"]
        ledger["attempts"][-1]["status"] = "completed"
        atomic_write_json(ledger, ledger_path)
        rows.append(row)
        spent += float(row["cost_usd"])
        if response_path is not None:
            atomic_write_jsonl(rows, response_path)
    summary = {
        "api_calls": calls,
        "attempts": attempts,
        "total_attempts_across_resumes": len(ledger.get("attempts") or []),
        "cache_hits": sum(bool(row.get("cache_hit")) for row in rows),
        "rows": len(rows),
        "spent_usd_including_cached_original_calls": round(spent, 8),
        "execute": execute,
        "cache_only": cache_only,
        "caps": {"requests": MAX_REQUESTS, "attempts": MAX_ATTEMPTS, "cost_usd": MAX_COST_USD},
    }
    summary.update(_cost_details(rows, pricing))
    return rows, summary


def evaluate(rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> dict[str, Any]:
    key = {row["sample_id"]: row for row in plan["answer_key"]}
    baseline_rows = json.loads((PHASE3_DIR / "validation" / "scored_rows.json").read_text(encoding="utf-8"))
    baseline = {(row["model_id"], row["reaction_id"]): row for row in baseline_rows if row["variant"] == "target_only"}
    samples = {row["sample_id"]: row for row in plan["samples"]}
    scored = []
    for row in rows:
        sid = row["sample_id"]
        answer = key[sid]
        sample = samples[sid]
        annotation = row["annotation"]
        predicted = annotation.get("predicted_kegg_id")
        exact = (not annotation["abstain"]) and predicted in set(answer["ground_truth_ids"])
        direct = baseline.get((sample["model_id"], sample["reaction_id"]))
        scored.append({
            "sample_id": sid,
            "model_id": sample["model_id"],
            "reaction_id": sample["reaction_id"],
            "corrected_stratum": sample["corrected_stratum"],
            "answer_present_in_fused_top10": answer["answer_present_in_fused_top10"],
            "fusion_top1_exact": answer["first_ground_truth_evidence_rank"] == 1,
            "grounded_terminal_status": row["terminal_status"],
            "grounded_abstained": annotation["abstain"],
            "grounded_predicted_kegg_id": predicted,
            "grounded_exact": bool(exact),
            "grounded_evidence_compliant": not row["compliance_problems"],
            "direct_target_only_available": direct is not None,
            "direct_target_only_abstained": direct.get("abstain") if direct else None,
            "direct_target_only_exact_top1": direct.get("exact_top1") if direct else None,
            "direct_target_only_terminal_status": direct.get("terminal_status") if direct else None,
        })
    return {
        "scope": "five mechanically selected validation reactions; descriptive only",
        "percentages_reported": False,
        "n_rows": len(scored),
        "rows": scored,
        "counts": {
            "grounded_exact": sum(row["grounded_exact"] for row in scored),
            "grounded_abstained": sum(row["grounded_abstained"] for row in scored),
            "grounded_evidence_compliant": sum(row["grounded_evidence_compliant"] for row in scored),
            "fusion_top1_exact": sum(row["fusion_top1_exact"] for row in scored),
            "direct_target_only_exact_top1": sum(row["direct_target_only_exact_top1"] is True for row in scored),
        },
        "test_rows_read": 0,
        "new_phase3a_calls": 0,
    }


def write_postrun(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], plan: Mapping[str, Any], out: Path = OUT) -> None:
    atomic_write_jsonl(rows, out / RESPONSES_PATH.name)
    atomic_write_json(dict(summary), out / "cost_report.json")
    evaluation = evaluate(rows, plan)
    atomic_write_json(evaluation, out / RESULTS_PATH.name)
    atomic_write_json({
        "scope": "planned full frozen validation comparison; do not execute in Phase 3C smoke",
        "population": "all 969 frozen validation reactions; no held-out test rows",
        "methods": ["phase3b_fusion_top1", "phase3c_grounded_llm_over_fusion_top10", "frozen_phase3a_aligned_target_only", "abstention"],
        "primary_metrics": ["exact_accuracy", "selective_accuracy", "coverage", "abstention_rate", "evidence_compliance_rate"],
        "secondary_metrics": ["BRITE_orthology_accuracy", "per_stratum_counts", "seen_unseen_counts", "schema_failure_count"],
        "uncertainty": {"method": "paired cluster bootstrap", "replicates": 10_000, "seed": 20260902, "unit": "frozen validation cluster", "report": "point deltas and 95% percentile intervals"},
        "cost_reporting": ["input/output/reasoning/cached tokens", "latency", "per-row and total USD", "cache reuse"],
        "go_no_go": "Proceed only if evidence compliance is complete, grounded exact accuracy improves over fusion Top1 with an interval excluding zero or yields a prespecified useful coverage/accuracy tradeoff, and observed cost remains inside a separately approved cap.",
        "execution_authorized": False,
        "test_rows_read": 0,
    }, out / "full_validation_plan.json")
    report = [
        "# Phase 3C grounded smoke report", "",
        "This is a descriptive five-reaction validation smoke test, not an accuracy estimate.", "",
        f"Grounded exact answers: {evaluation['counts']['grounded_exact']} of {evaluation['n_rows']}.",
        f"Grounded abstentions: {evaluation['counts']['grounded_abstained']} of {evaluation['n_rows']}.",
        f"Evidence-compliant outputs: {evaluation['counts']['grounded_evidence_compliant']} of {evaluation['n_rows']}.",
        f"Frozen fusion Top-1 exact answers: {evaluation['counts']['fusion_top1_exact']} of {evaluation['n_rows']}.",
        f"Frozen Phase 3A aligned target-only exact answers: {evaluation['counts']['direct_target_only_exact_top1']} of {evaluation['n_rows']}.", "",
        f"The five live calls cost ${float(summary['spent_usd_including_cached_original_calls']):.6f}; cache-only replay made zero calls.", "",
        "All requests were target-local and contained only the frozen Top-10 evidence. No provider tools, web access, memory, tracing, planning, held-out test data, or answer labels were available to the model.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    atomic_write_json({
        "model": MODEL, "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "tools": [], "store": False,
        "top_k": TOP_K, "rrf_k": 60, "automatic_retries": 0,
        "required_versions": {"openai": "1.78.1", "pydantic": "2.13.5", "langchain-core": "1.6.2"},
        "schemas": {"output": SCHEMA_VERSION, "evidence": EVIDENCE_SCHEMA, "prompt": PROMPT_VERSION},
    }, out / "config.json")


def freeze_original_cached_run(plan: Mapping[str, Any], cache_dir: Path = CACHE_DIR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Recover the original live rows without changing their cache-hit flags."""
    cache = FileCache(cache_dir)
    rows = []
    for item in plan["planned"]:
        row = cache.get(item["cache_id"])
        if row is None:
            raise RuntimeError(f"missing cache entry for {item['cache_id']}")
        rows.append(dict(row))
    ledger_path = cache_dir / "_attempt_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    attempts = list(ledger.get("attempts") or [])
    summary = {
        "api_calls": sum(item.get("status") == "completed" for item in attempts),
        "attempts": len(attempts),
        "cache_hits": 0,
        "cache_only": False,
        "caps": {"attempts": MAX_ATTEMPTS, "cost_usd": MAX_COST_USD, "requests": MAX_REQUESTS},
        "execute": True,
        "rows": len(rows),
        "spent_usd_including_cached_original_calls": round(sum(float(row.get("cost_usd") or 0.0) for row in rows), 8),
        "total_attempts_across_resumes": len(attempts),
    }
    summary.update(_cost_details(rows, load_pricing(PRICING_OPENAI_TERRA)))
    return rows, summary


def verify_existing_manifests() -> dict[str, Any]:
    """Read-only verification of every pre-Phase-3C Phase 3 artifact manifest."""
    checked = []
    problems = []
    for manifest_path in sorted(PHASE3_DIR.rglob("artifact_manifest.json")):
        if OUT in manifest_path.parents:
            continue
        relative_parts = manifest_path.relative_to(PHASE3_DIR).parts
        if any(part.startswith("_") for part in relative_parts):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        local = []
        for item in manifest.get("files") or []:
            path = REPO_ROOT / item["path"]
            if not path.is_file():
                local.append(f"missing: {item['path']}")
            elif sha256_portable(path) != item["sha256"]:
                local.append(f"digest mismatch: {item['path']}")
        record = {"manifest": repo_relative_posix(manifest_path), "n_files": len(manifest.get("files") or []), "problems": local}
        checked.append(record)
        problems.extend(f"{record['manifest']}: {problem}" for problem in local)
    return {"scope": "all pre-Phase-3C Phase 3 artifact manifests", "read_only": True, "n_manifests": len(checked), "n_problems": len(problems), "manifests": checked, "problems": problems}


def verify_artifacts(out: Path = OUT) -> list[str]:
    manifest_path = out / "artifact_manifest.json"
    before = manifest_path.read_bytes()
    manifest = json.loads(before)
    problems = []
    paths = [item["path"] for item in manifest["files"]]
    if len(paths) != len(set(paths)):
        problems.append("duplicate manifest paths")
    for item in manifest["files"]:
        path = REPO_ROOT / item["path"]
        if not path.is_file() or sha256_portable(path) != item["sha256"]:
            problems.append(f"digest mismatch: {item['path']}")
    if manifest_path.read_bytes() != before:
        problems.append("read-only verification mutated manifest")
    return problems


def write_manifest(out: Path = OUT) -> None:
    artifacts = [path for path in out.iterdir() if path.is_file() and path.name != "artifact_manifest.json"]
    write_artifact_manifest(out, artifacts)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--no-dotenv", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.verify:
        problems = verify_artifacts(args.out)
        print(json.dumps({"problems": problems, "n_problems": len(problems)}, sort_keys=True))
        return bool(problems)
    index = EvidenceIndex()
    plan = build_plan(index, args.cache_dir)
    write_plan(plan, args.out, preserve_dry_run=args.execute or args.cache_only)
    if not args.execute and not args.cache_only:
        print(json.dumps(plan["dry_run"], indent=2, sort_keys=True))
        return 0
    if args.execute:
        assert_env_file_protected(REPO_ROOT)
        if not args.no_dotenv:
            load_dotenv_if_present(REPO_ROOT)
    rows, summary = run_plan(
        plan,
        execute=args.execute,
        cache_only=args.cache_only,
        cache_dir=args.cache_dir,
        response_path=args.out / RESPONSES_PATH.name if args.execute else None,
    )
    if args.cache_only:
        atomic_write_json(summary, args.out / "cache_verify.json")
        original_rows, original_summary = freeze_original_cached_run(plan, args.cache_dir)
        write_postrun(original_rows, original_summary, plan, args.out)
    else:
        write_postrun(rows, summary, plan, args.out)
    atomic_write_json(verify_existing_manifests(), args.out / "prior_manifest_verification.json")
    write_manifest(args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
