"""Provider requests and semantic validation for grounded reaction recommendations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .retrieval import CONFIG, load_retriever, sha256
from .species_evidence import CONTRACT_VERSION, build_queries

SYSTEM_INSTRUCTIONS = "You annotate one biochemical reaction using only the supplied TARGET and EVIDENCE records. Do not use recalled knowledge, external tools, web search, or identifiers absent from EVIDENCE. If you answer, predicted_kegg_id must be the kegg_id at selected_evidence_rank, and supporting_evidence_ids must name one or more supplied evidence_id values including that selected record. If the evidence is insufficient, abstain. Return only the required structured response."
VISIBLE_FIELDS = ("evidence_id", "kegg_id", "fused_rank", "fused_score", "bm25_rank",
                  "trained_biencoder_rank", "name", "definition", "equation", "enzyme", "rclass", "brite")


class GroundedAnnotation(BaseModel):
    """Strict provider schema; semantic evidence rules are checked afterward."""

    model_config = ConfigDict(extra="forbid")
    abstain: bool
    predicted_kegg_id: Optional[str] = Field(pattern=r"^R[0-9]{5}$")
    selected_evidence_rank: Optional[int] = Field(ge=1, le=10)
    supporting_evidence_ids: list[str] = Field(max_length=10)
    confidence: float = Field(ge=0, le=1)
    reasoning_summary: str
    abstention_reason: Optional[str]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def render_input(query, evidence):
    visible = [{key: row[key] for key in VISIBLE_FIELDS} for row in evidence]
    return "TARGET\n" + json.dumps({"query": query}, sort_keys=True, ensure_ascii=False) + "\nEVIDENCE\n" + json.dumps(visible, sort_keys=True, ensure_ascii=False)


def provider_payload(query, evidence, *, model="gpt-6.1-terra", reasoning_effort="low", max_output_tokens=2048):
    """Capture the SDK's complete wire body locally, without a client or network."""
    from openai.resources.responses import Responses
    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    if reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
        raise ValueError("Unsupported reasoning effort")
    resource = object.__new__(Responses)
    def capture(url, *, body, **kwargs):
        body.update(kwargs.get("options", {}).get("extra_json") or {})
        return body
    resource._post = capture
    return resource.parse(model=model, instructions=SYSTEM_INSTRUCTIONS,
        input=render_input(query, evidence), max_output_tokens=max_output_tokens,
        reasoning={"effort": reasoning_effort}, store=False, tools=[], text_format=GroundedAnnotation,
        extra_body={"service_tier": "default"})


def validate_evidence(evidence):
    if len(evidence) != 10 or [e["fused_rank"] for e in evidence] != list(range(1, 11)):
        raise ValueError("Expected exactly ten ordered evidence records")
    if len({e["kegg_id"] for e in evidence}) != 10 or [e["evidence_id"] for e in evidence] != [f"E{i:02d}" for i in range(1, 11)]:
        raise ValueError("Duplicate/invalid evidence identifiers")
    for row in evidence:
        if not all(key in row for key in VISIBLE_FIELDS):
            raise ValueError("Incomplete evidence record")


def classify_response(raw, evidence, model):
    """Failures never become biological abstentions. Preserve raw status and usage."""
    if not isinstance(raw, dict):
        return "provider_error", None, ["response is not an object"]
    if raw.get("status") == "incomplete":
        return "incomplete", None, [str(raw.get("incomplete_details"))]
    if raw.get("status") != "completed" or raw.get("error"):
        return "provider_error", None, [str(raw.get("error") or raw.get("status"))]
    if any(item.get("status") == "incomplete" for item in raw.get("output", [])):
        return "incomplete", None, ["incomplete output message"]
    # Snapshots must be explicitly configured rather than silently accepted as another model.
    if raw.get("model") != model:
        return "provider_error", None, ["response model differs from requested model"]
    blocks = [block for output in raw.get("output", []) if output.get("type") == "message"
              for block in output.get("content", [])]
    if any(block.get("type") == "refusal" for block in blocks):
        return "refusal", None, ["provider refusal"]
    messages = [block.get("text", "") for block in blocks if block.get("type") == "output_text"]
    if not messages or not "".join(messages).strip():
        return "empty_response", None, ["no answer text"]
    try:
        # Strict parsing prevents strings/numbers being coerced into booleans/IDs.
        answer = GroundedAnnotation.model_validate_json("".join(messages), strict=True)
    except ValueError as exc:
        return "schema_invalid", None, [str(exc)]
    errors = []
    support = answer.supporting_evidence_ids
    by_id = {e["evidence_id"]: e for e in evidence}
    by_rank = {e["fused_rank"]: e for e in evidence}
    if len(support) != len(set(support)) or set(support) - set(by_id):
        errors.append("duplicate or unknown supporting evidence")
    if answer.abstain:
        if answer.predicted_kegg_id is not None or answer.selected_evidence_rank is not None or support:
            errors.append("abstention must not select evidence")
        if not (answer.abstention_reason or "").strip():
            errors.append("abstention reason required")
    else:
        selected = by_rank.get(answer.selected_evidence_rank)
        if (selected is None or selected["kegg_id"] != answer.predicted_kegg_id
                or selected["evidence_id"] not in support):
            errors.append("selection does not match supplied evidence/rank/support")
        if (answer.abstention_reason or "").strip():
            errors.append("selection has abstention reason")
    if not answer.reasoning_summary.strip():
        errors.append("empty reasoning summary")
    if errors:
        return "evidence_noncompliant", None, errors
    return ("abstained" if answer.abstain else "selected"), answer.model_dump(), []


def execute_payload(body):
    """One attempt, no SDK automatic retries. Called only outside dry-run."""
    from openai import OpenAI
    client = OpenAI(max_retries=0)
    payload = dict(body)
    tier = payload.pop("service_tier")
    return client.responses.create(**payload, extra_body={"service_tier": tier}).model_dump(mode="json")


def annotate_grounded(model_file, *, species_recommendations_df=None, resources=None,
                      retriever=None, transport=None, dry_run=False, llm_model="gpt-6.1-terra",
                      reasoning_effort="low", max_output_tokens=2048, save_to=None, species_stage=None):
    from core.feedback import AnnotationResult
    error = None
    input_error = False
    species_table = species_recommendations_df
    try:
        if isinstance(species_table, (str, Path)):
            species_table = pd.read_csv(species_table, dtype=str).fillna("")
        queries = build_queries(model_file, species_table)
    except (ValueError, OSError) as exc:
        # A valid SBML still defines coverage when its recommendation input fails.
        queries = build_queries(model_file)
        error, input_error = str(exc), True
    try:
        if retriever is None and not error:
            if resources is None:
                raise ValueError("reaction_resources is required; install the verified local encoder and catalog")
            retriever = load_retriever(resources)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        error = str(exc)
    rows, artifacts, usage = [], [], []
    source_provenance = {"sbml_sha256": sha256(model_file), "species_contract": CONTRACT_VERSION}
    if isinstance(species_recommendations_df, (str, Path)) and Path(species_recommendations_df).is_file():
        source_provenance["species_csv_sha256"] = sha256(species_recommendations_df)
    if isinstance(species_table, pd.DataFrame):
        source_provenance["species_table_sha256"] = digest(species_table.fillna("").to_dict("records"))
    for query in queries:
        rid = query["reaction_id"]
        row = {"id": rid, "display_name": query["display_name"], "annotation": "", "database": "kegg",
               "type": "reaction", "qualifier": "is", "update_annotation": "ignore",
               "status": ("input_error" if input_error else "resource_error") if error else "planned", "biological_abstain": None,
               "error": error, "request_sha256": None}
        artifact = {**query, "method": CONFIG, "source_provenance": source_provenance,
                    "query_sha256": hashlib.sha256(query["query"].encode("utf-8")).hexdigest(),
                    "species_contract": CONTRACT_VERSION,
                    "retrieval_provenance": getattr(retriever, "provenance", None)}
        if not error:
            try:
                evidence = retriever.retrieve(query["query"])
                validate_evidence(evidence)
                body = provider_payload(query["query"], evidence, model=llm_model,
                    reasoning_effort=reasoning_effort, max_output_tokens=max_output_tokens)
                row["request_sha256"] = digest(body)
                artifact.update(evidence=evidence, provider_payload=body,
                                request_sha256=digest(body), cache_key=digest({"schema": "rrf-grounded-request-v1", "provider_payload": body}))
            except (ValueError, OSError, RuntimeError, KeyError) as exc:
                row.update(status="input_resource_error", error=str(exc))
            else:
                if not dry_run:
                    try:
                        raw = (transport or execute_payload)(body)
                        artifact["response"] = raw
                        if isinstance(raw, dict) and raw.get("usage"):
                            usage.append({"reaction_id": rid, "model": raw.get("model"), **raw["usage"]})
                        status, answer, errors = classify_response(raw, evidence, llm_model)
                        row.update(status=status, error="; ".join(errors) or None)
                        if answer:
                            row.update(biological_abstain=answer["abstain"],
                                annotation="KEGG:" + answer["predicted_kegg_id"] if not answer["abstain"] else "",
                                match_score=answer["confidence"], reasoning_summary=answer["reasoning_summary"])
                            artifact["answer"] = answer
                    except Exception as exc:
                        row.update(status="provider_error", error=f"{type(exc).__name__}: {exc}")
        artifacts.append({**artifact, "result": dict(row)})
        rows.append(row)
    frame = pd.DataFrame(rows, columns=None if rows else ["id", "annotation", "status", "biological_abstain"])
    counts = frame.status.value_counts().to_dict()
    metrics = {"method": "rrf_grounded", "configuration": {**CONFIG, "species_contract": CONTRACT_VERSION,
        "model": llm_model, "reasoning_effort": reasoning_effort, "max_output_tokens": max_output_tokens,
        "service_tier": "default"}, "coverage": {"total_reactions": len(queries), "status_counts": counts},
        "dry_run": dry_run, "provider_attempts": 0 if dry_run else sum("provider_payload" in a and a["result"]["status"] != "planned" for a in artifacts),
        "llm_usage": {"records": usage, "input_tokens": sum(r.get("input_tokens", 0) for r in usage),
                      "output_tokens": sum(r.get("output_tokens", 0) for r in usage)}}
    if species_stage is not None:
        metrics["species_stage"] = species_stage
    prefix = Path(save_to or str(model_file))
    csv_path = str(prefix) + "_reactions.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    Path(str(prefix) + "_reaction_evidence.jsonl").write_text("".join(json.dumps(a, ensure_ascii=False, sort_keys=True) + "\n" for a in artifacts), encoding="utf-8")
    Path(str(prefix) + "_reaction_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    result = AnnotationResult(frame, metrics, model_file=str(model_file), conversation_history=[],
        entities_to_evaluate=[q["reaction_id"] for q in queries], entity_type="reaction", database="kegg",
        method="rrf_grounded", llm_model=llm_model, top_k=10, n_return=1, csv_path=csv_path)
    result.reaction_recommendations_df = frame
    result.species_recommendations_df = species_table if isinstance(species_table, pd.DataFrame) else None
    result.reaction_evidence = artifacts
    return result
