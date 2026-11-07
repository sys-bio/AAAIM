"""Freeze aligned responses first, then analyze only the frozen validation join.

Failed units remain in denominators; 326 outputs are 163 paired reactions.
"""
from __future__ import annotations

import argparse
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from benchmark.scripts import phase5_runner as runner
from benchmark.scripts.phase5_perturb import digest


def freeze(package, destination, *, synthetic=False):
    plan, units, _, order = runner.load_package(package)
    state, lock = runner.state_paths(package, plan, synthetic)
    if destination.exists():
        raise ValueError("response freezes are immutable; choose a new destination")
    with runner.exclusive(lock):
        ledger = runner.read(state / "ledger.json")
        runner.validate_ledger(ledger, plan, units, order, kind="synthetic" if synthetic else "live")
        if len(ledger["attempts"]) != len(order):
            raise ValueError("all planned units must be terminal before the reference join; incomplete run cannot be frozen")
        frozen = []
        for entry in ledger["attempts"]:
            unit = units[entry["unit_key"]]
            result = runner.cache_result(state, entry, plan, unit)
            if entry["status"] != "completed" or result is None or result["pause_required"] and not entry.get("reconciliation"):
                raise ValueError("unresolved attempt; reconcile without retry before freezing")
            frozen.append({"unit": unit, "result": {k: v for k, v in result.items() if k != "raw_response"},
                           "result_digest_in_cache": entry["result_digest"], "charged_nusd": entry["charged_nusd"],
                           "reconciliation": entry.get("reconciliation")})
        runner.immutable(destination, {"schema": "phase5-response-freeze-v1", "plan_digest": plan["plan_digest"],
            "response_digest": digest(frozen), "responses": frozen, "frozen_at": runner.now(),
            "frozen_before_reference_join": True, "provider_kind": ledger["provider_kind"],
            "ledger_digest": digest(ledger), "order_digest": digest(order), "reference_rows_read": 0,
            "endpoint": ledger["endpoint"], "sdk_version": ledger["sdk_version"]})
    return runner.read(destination)


def metrics(items):
    n = len(items)
    answered = sum(r["answered"] for r in items)
    correct = sum(r["correct"] for r in items)
    return {"n": n, "correct": correct, "accuracy": correct/n if n else None,
            "answered": answered, "coverage": answered/n if n else None,
            "selective_accuracy": correct/answered if answered else None,
            "abstained": sum(r["abstained"] for r in items), "failures": sum(r["failed"] for r in items),
            "abstention_rate": sum(r["abstained"] for r in items)/n if n else None,
            "failure_rate": sum(r["failed"] for r in items)/n if n else None,
            "appropriate_reference_based_abstentions": sum(r["appropriate_abstention"] for r in items),
            "retrievable_reference_abstentions": sum(r["retrievable_abstention"] for r in items),
            "reference_absent_from_evidence": sum(not r["reference_in_evidence"] for r in items)}


def paired_bootstrap(pairs, replicates=2000, seed=20261106):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["cluster_id"]].append(pair)
    clusters, rng = sorted(groups), random.Random(seed)

    def changes(sample):
        a, b = [metrics([p[arm] for p in sample]) for arm in ("clean", "degraded")]
        return {key: b[key]-a[key] if a[key] is not None and b[key] is not None else None
                for key in ("accuracy", "coverage", "selective_accuracy", "abstention_rate", "failure_rate")}

    point = changes(pairs)
    draws = defaultdict(list)
    for _ in range(replicates):
        sample = [p for _ in clusters for p in groups[rng.choice(clusters)]]
        for key, value in changes(sample).items():
            if value is not None:
                draws[key].append(value)
    result = {}
    for key, value in point.items():
        values = sorted(draws[key])
        result[key] = {"degraded_minus_clean": value,
                       "ci95": [values[int(.025*len(values))], values[min(len(values)-1, int(.975*len(values)))]] if values else None,
                       "defined_bootstrap_draws": len(values)}
    return {"method": "paired percentile cluster bootstrap; same reaction/condition pair in every draw",
            "n_biological_reactions": len(pairs), "n_clusters": len(clusters), "replicates": replicates, "seed": seed,
            "interpretation": "descriptive development-stratified pilot, not population-wide rates; selective accuracy may be undefined in zero-coverage draws", "changes": result}


def analyze(package, response_freeze, reference_rows, *, synthetic=False):
    """References are supplied only AFTER a frozen-response contract is verified."""
    plan, units, bodies, order = runner.load_package(package)
    f = runner.read(response_freeze)
    if not f["frozen_before_reference_join"] or f["plan_digest"] != plan["plan_digest"] or f["response_digest"] != digest(f["responses"]) or f["order_digest"] != digest(order):
        raise ValueError("invalid response freeze")
    if (f["provider_kind"] == "synthetic") != synthetic:
        raise ValueError("synthetic responses cannot be analyzed as live findings")
    if [r["unit"]["unit_key"] for r in f["responses"]] != order or any(r["unit"] != units[r["unit"]["unit_key"]] for r in f["responses"]):
        raise ValueError("complete response/unit alignment required")
    # Caller may pass a lazy loader: tests and CLI prove that validation and
    # alignment above occur BEFORE any reference label file is opened.
    refs = reference_rows() if callable(reference_rows) else reference_rows
    refs_by_id = {r["sample_id"]: r for r in refs}
    if len(refs_by_id) != len(refs) or set(refs_by_id) != {u["sample_id"] for u in units.values()}:
        raise ValueError("exact validation reference alignment required")
    per_case, pairs = {}, []
    for row in f["responses"]:
        u, result = row["unit"], row["result"]
        ref = refs_by_id[u["sample_id"]]
        if (ref["model_id"], ref["reaction_id"]) != (u["model_id"], u["reaction_id"]):
            raise ValueError("reference experimental-unit identity mismatch")
        a = result.get("annotation") or {}
        valid = result["status"] == "succeeded"
        evidence = runner.split_input(bodies[u["unit_key"]]["input"])["EVIDENCE"]
        present = bool(set(ref["ground_truth_ids"]) & {e["kegg_id"] for e in evidence})
        abstain = valid and bool(a.get("abstain"))
        item = {"sample_id": u["sample_id"], "condition": u["condition"], "cluster_id": ref["cluster_id"],
                "stratum": ref["corrected_stratum"], "status": result["status"],
                "answered": valid and not a["abstain"], "abstained": abstain, "failed": not valid,
                "correct": valid and not a["abstain"] and a["predicted_kegg_id"] in ref["ground_truth_ids"],
                "reference_in_evidence": present, "appropriate_abstention": abstain and not present,
                "retrievable_abstention": abstain and present}
        per_case[u["sample_id"], u["arm"]] = item
    for sid in sorted(refs_by_id):
        ref = refs_by_id[sid]
        pairs.append({"sample_id": sid, "cluster_id": ref["cluster_id"], "stratum": ref["corrected_stratum"],
                      "clean": per_case[sid, "clean"], "degraded": per_case[sid, "degraded"]})
    arms = {arm: metrics([p[arm] for p in pairs]) for arm in ("clean", "degraded")}
    historical = {"n": len(refs), "correct": sum(r["grounded_exact"] for r in refs),
                  "answered": sum(r["grounded_answered"] for r in refs), "abstained": sum(r["grounded_abstain"] for r in refs)}
    gaps, times = [], []
    for i in range(0, len(f["responses"]), 2):
        a, b = [r["result"] for r in f["responses"][i:i+2]]
        if a.get("started_at") and b.get("started_at"):
            gaps.append(abs((runner.datetime.fromisoformat(b["started_at"])-runner.datetime.fromisoformat(a["started_at"])).total_seconds()))
        times.extend(r[k] for r in (a, b) for k in ("started_at", "finished_at") if r.get(k))
    by_stratum = {s: {arm: metrics([p[arm] for p in pairs if p["stratum"] == s]) for arm in arms} for s in sorted({p["stratum"] for p in pairs})}
    return {"schema": "phase5-matched-analysis-v1", "synthetic": synthetic, "plan_digest": plan["plan_digest"],
        "response_freeze_digest": sha_freeze(response_freeze), "n_pairs": len(pairs), "arms": arms,
        "paired_uncertainty": paired_bootstrap(pairs), "by_stratum": by_stratum,
        "historical_clean_descriptive_only": {**historical, "new_clean_correct_minus_historical": arms["clean"]["correct"]-historical["correct"],
            "new_clean_answered_minus_historical": arms["clean"]["answered"]-historical["answered"],
            "new_clean_abstained_minus_historical": arms["clean"]["abstained"]-historical["abstained"],
            "limitation": "different requested model, changed prompt, time and possible hosted-model drift; historical GPT-5.6 outputs are descriptive only and the primary comparison uses only the new matched GPT-6.1 clean baseline",
            "historical_model": "gpt-5.6-terra", "new_model": plan["settings"]["model"]},
        "resource_report": {"attempts": len(f["responses"]), "accounted_cost_ceiling_usd": sum(r["charged_nusd"] for r in f["responses"])/1e9,
            "requested_models": dict(Counter(r["result"].get("model_requested") for r in f["responses"])),
            "returned_models": dict(Counter(str(r["result"].get("model_returned")) for r in f["responses"])),
            "service_tiers": dict(Counter(str(r["result"].get("service_tier")) for r in f["responses"])),
            "usage_totals_known_only": {key: sum((row["result"].get("usage") or {}).get(key) or 0 for row in f["responses"]) for key in ("input_tokens", "output_tokens")},
            "time_start": min(times) if times else None, "time_end": max(times) if times else None,
            "pair_start_gap_seconds_max": max(gaps) if gaps else None,
            "pair_start_gap_seconds_median": statistics.median(gaps) if gaps else None,
            "endpoint": f["endpoint"], "sdk_version": f["sdk_version"], "duplicate_policy": plan["duplicate_policy"]},
        "limitations": ["163 paired reactions, not 326 independent biological samples", "separate duplicate-body calls are repeated stochastic observations",
                        "stratified reused development pilot; no population-wide rate claim", "interleaving and matching aliases cannot eliminate hidden hosted-model or temporal drift"],
        "case_results": pairs}


def sha_freeze(path):
    return runner.sha(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--response-freeze", type=Path, required=True)
    p.add_argument("--freeze-only", action="store_true")
    p.add_argument("--source-root", type=Path)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    if a.freeze_only:
        freeze(a.package, a.response_freeze)
        return 0
    if a.source_root is None or a.out is None:
        p.error("--source-root and --out are required for analysis")
    result = analyze(a.package, a.response_freeze, lambda: runner.rows(a.source_root / "benchmark/phase3/phase3c_validation/answer_key_join.jsonl"))
    runner.immutable(a.out, result)
    print({k: result[k] for k in ("n_pairs", "arms", "paired_uncertainty")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
