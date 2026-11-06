"""Prepare a draft held-out freeze and exercise a synthetic-only bounded runner.

This CLI cannot unseal or execute held-out data. --execute fails before loading
any request data or importing a provider. The generic injected runner exists to
test budgeting, caching and failure semantics without making network calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from benchmark.scripts import phase3c_grounded as grounded
from benchmark.scripts.phase4_routing import ROOT, PHASE3, load, sha, write


@dataclass
class Budget:
    cap_usd: float
    max_attempts: int
    reserved_usd: float = 0.0
    attempts: int = 0

    def reserve(self, worst_case_usd):
        if not all(math.isfinite(v) and v >= 0 for v in (worst_case_usd, self.cap_usd, self.reserved_usd)) or self.max_attempts < 0:
            raise ValueError("invalid budget")
        if self.attempts >= self.max_attempts or self.reserved_usd + worst_case_usd > self.cap_usd + 1e-12:
            raise RuntimeError("hard budget/attempt cap reached before provider call")
        self.attempts += 1
        # Keep pessimistic reservations, including failed calls; no speculative refunds.
        self.reserved_usd += worst_case_usd


def canonical_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def request_payload(query, evidence):
    if re.search(r"R\d{5}", query, flags=re.I):
        raise ValueError("KEGG reaction identifier in target query")
    prompt = grounded.build_user_prompt({"query": query}, evidence)
    payload = {"model": grounded.MODEL, "instructions": grounded.SYSTEM_INSTRUCTIONS, "input": prompt,
               "tools": [], "store": False, "reasoning": {"effort": grounded.REASONING_EFFORT},
               "max_output_tokens": grounded.MAX_OUTPUT_TOKENS,
               "text": {"format": {"type": "json_schema", "name": "grounded_annotation", "strict": True,
                                    "schema": grounded.GroundedAnnotation.model_json_schema()}}}
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > 32768:
        raise ValueError("request exceeds prespecified serialized byte limit; abstain, never truncate ad hoc")
    return payload


def replay_or_attempt(payload, evidence, budget, cache, provider):
    """Network-independent injectable runner; immutable cache entries include failures."""
    key = canonical_digest(payload)
    if key in cache:
        entry = cache[key]
        if entry["payload_digest"] != key:
            raise ValueError("incompatible cache payload digest")
        if entry["status"] == "succeeded":
            annotation = grounded.GroundedAnnotation.model_validate(entry["annotation"])
            if grounded.validate_grounded_annotation(annotation, evidence):
                raise ValueError("cached answer incompatible with evidence")
        return entry
    # Frozen short-context snapshot rates; refresh requires approval before final freeze.
    worst_case = (32768 * 2.0 + grounded.MAX_OUTPUT_TOKENS * 12.0)/1_000_000
    budget.reserve(worst_case)
    try:
        response = provider(payload)
        annotation = grounded.GroundedAnnotation.model_validate(response)
        problems = grounded.validate_grounded_annotation(annotation, evidence)
        if problems:
            raise ValueError("; ".join(problems))
        entry = {"status": "succeeded", "annotation": annotation.model_dump(), "payload_digest": key}
    except Exception as error:
        # Failure remains a reported abstention; no automatic retry or selective rerun.
        entry = {"status": "failed", "annotation": None, "payload_digest": key,
                 "abstain": True, "error_type": type(error).__name__}
    cache[key] = entry
    return entry


def synthetic_dry_run():
    evidence = [grounded.EvidenceRecord(
        evidence_id="E01", kegg_id="R99991", fused_rank=1, fused_score=.03,
        bm25_rank=1, trained_biencoder_rank=1, name="Synthetic candidate",
        definition="Synthetic validation fixture only", equation="C00001 <=> C00002",
        enzyme="", rclass="", brite="", provenance={"synthetic": True})]
    payload = request_payload("Synthetic substrate => synthetic product", evidence)
    budget, cache, provider_calls = Budget(.10, 1), {}, []
    def fake_provider(value):
        provider_calls.append(value)
        return {"abstain": True, "predicted_kegg_id": None, "selected_evidence_rank": None,
                "supporting_evidence_ids": [], "confidence": .2, "reasoning_summary": "Synthetic insufficient evidence.",
                "abstention_reason": "Synthetic fixture"}
    first = replay_or_attempt(payload, evidence, budget, cache, fake_provider)
    second = replay_or_attempt(payload, evidence, budget, cache, fake_provider)
    return {"synthetic_only": True, "test_files_opened": 0, "provider_network_calls": 0,
            "injected_fake_provider_calls": len(provider_calls), "cache_replay_identical": first == second,
            "budget_reservation_usd": budget.reserved_usd, "status": first["status"],
            "payload_digest": canonical_digest(payload), "prompt_schema_validated": True,
            "held_out_executor_available": False}


def prepare(study, out):
    if out.exists():
        raise ValueError("refusing to overwrite freeze output")
    out.mkdir(parents=True)
    selection = load(study / "selection.json")
    provenance = load(study / "provenance.json")
    metrics = load(study / "metrics.json")
    prior = load(PHASE3 / "phase3c_validation/config.json")
    checkpoint = load(PHASE3 / "phase3b_full/selected_checkpoint.json")
    selected = selection["selected_method"]
    selected_config = {"method": selected,
        "rule": "frozen grounded on every reaction" if selected == "grounded" else "fusion Top-1" if selected == "fusion" else "component agreement plus relative RRF margin gate; fusion if gate else grounded",
        "threshold": float(selected.split("_")[1]) if selected.startswith("routed_") else None,
        "reference_policy": "unchanged frozen multi-label KEGG reference; no human-corrected labels",
        "grounded": prior,
        "bi_encoder": {"initializer": "BAAI/bge-small-en-v1.5", "revision": "5e62ea33e012fda8c02802b906664c915ebd1bb1",
                       "selected_checkpoint": checkpoint["selected_checkpoint"]},
        "retrieval": load(PHASE3 / "phase3b_fusion/config.json")}
    write(out / "selected_method.json", selected_config)
    (out / "system_prompt.txt").write_text(grounded.SYSTEM_INSTRUCTIONS + "\n", encoding="utf-8")
    write(out / "response_schema.json", grounded.GroundedAnnotation.model_json_schema())
    selected_cost = metrics[selected]["replayed_cost_usd"]
    freeze = {
        "schema": "phase4-evaluation-freeze-draft-v1", "status": "draft",
        "selected_method": selected, "validation_selection_complete": True,
        "base_code_revision": provenance["git_base_revision"], "code_state": provenance["source_state"],
        "source_and_baseline_hashes": provenance["inputs"],
        "held_out_status": "sealed; no rows, labels, predictions or diagnostics opened",
        "blockers": [
            "Explicitly authorized held-out executor and label-sealing integration are not yet implemented; this CLI intentionally cannot run held-out inputs.",
            "No held-out API budget is approved. Frozen September price snapshot must be confirmed before final budget approval; no price lookup performed in this task.",
            "Hosted gpt-5.6-terra is the recorded requested/returned identity, not an immutable provider snapshot. Lock availability/identity and policy for drift before unsealing."
        ],
        "primary_metric": "exact correct / all prespecified held-out eligible reactions, counting abstentions/failures as noncorrect",
        "secondary_metrics": ["coverage", "accuracy among answered", "cluster/model macro exact accuracy", "BRITE/orthology-aware secondary accuracy", "per retrieval stratum", "risk-coverage and partial/normalized AURC", "reference-absence detection", "correctness detection among answered", "cost/tokens/attempts/provider latency and end-to-end latency if measured"],
        "comparisons": ["selected method", "frozen fusion Top-1", "frozen grounded all-reaction baseline"],
        "uncertainty": {"scheme": "paired cluster bootstrap, resample test model clusters only after predictions freeze", "replicates": 10000, "seed": 20261105, "interval": "percentile 95%", "no_test_driven_threshold_changes": True},
        "failure_rules": {"automatic_retries": 0, "max_attempts_per_reaction": 1, "cache_key": "canonical full provider payload including model, prompt, evidence, schema, options", "cache": "persist successes and failures atomically, validate digest before reuse", "schema_or_compliance_failure": "reported abstention, count in denominator; do not silently drop", "missing_resources": "abort preflight before unsealing", "oversize_payload": "explicit abstention, no ad hoc truncation", "resume": "same locked requests only, never retry completed failures selectively"},
        "exclusions": "prespecified frozen benchmark eligibility only; no new test-dependent exclusions",
        "leakage": "target-local scrubbed queries; no KEGG reaction ID in query, no reference or audit fields in routing/request; labels joined by independent stage only after predictions are frozen",
        "budget": {"currently_authorized_usd": 0, "proposed_run_population_from_roadmap_only": 1350,
                   "rough_cost_usd_at_pilot_mix_and_snapshot_rates": selected_cost/163*1350,
                   "not_a_test_prompt_cost_quote": True,
                   "max_serialized_payload_bytes": 32768, "max_output_tokens": 2048,
                   "reservation_usd_per_possible_call_at_snapshot_rates": .090112,
                   "worst_case_all_1350_calls_snapshot_usd": 121.6512,
                   "proposed_hard_cap_usd_for_all_call_upper_bound": 125,
                   "max_attempts_upper_bound": 1350,
                   "mechanism": "reserve pessimistic full per-call maximum BEFORE each call, retain failed-call reservations; stop before count/dollar cap; no live execution authorized"},
        "test_change_policy": "one locked test run; no test-driven model/threshold/prompt changes or selective reruns; technical amendments require new documented protocol and invalidate a confirmatory interpretation if outcomes informed them",
        "later_execution_command": "python -m benchmark.scripts.phase4_freeze --freeze benchmark/phase4/freeze/evaluation_freeze.json --execute",
        "execution_command_status": "intentionally fail-closed placeholder; not a working held-out evaluator",
        "dry_run_command": "python -m benchmark.scripts.phase4_freeze --dry-run --out benchmark/phase4/_dry_run_reproduced.json",
    }
    write(out / "evaluation_freeze.json", freeze)
    write(out / "synthetic_dry_run.json", synthetic_dry_run())
    manifest = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    write(out / "package_checksums.json", manifest)
    return freeze


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--study", type=Path, default=ROOT / "benchmark/phase4/validation")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.execute:
        # Fail before even opening a passed freeze path; no data unlock adapter exists.
        raise SystemExit("Held-out execution disabled: draft freeze; complete executor, identity and budget gates first.")
    if args.prepare:
        result = prepare(args.study, args.out or ROOT / "benchmark/phase4/freeze")
        print(json.dumps({"status": result["status"], "selected_method": result["selected_method"], "blockers": result["blockers"]}, indent=2))
    elif args.dry_run:
        result = synthetic_dry_run()
        if args.out:
            if args.out.exists():
                raise ValueError("refusing to overwrite dry-run output")
            write(args.out, result)
        print(json.dumps(result, indent=2))
    else:
        parser.error("select --prepare or --dry-run; held-out execution is disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
