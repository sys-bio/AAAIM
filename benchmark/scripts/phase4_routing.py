"""Validation-only replay of predeclared routing policies; never imports a provider.

python -m benchmark.scripts.phase4_routing --out benchmark/phase4/validation
Inputs are an explicit allowlist of frozen Phase 3 validation files. Route
decisions are computed and serialized before the evaluation answer key is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from benchmark.scripts.phase3_common import sha256_portable

ROOT = Path(__file__).resolve().parents[2]
PHASE3 = ROOT / "benchmark/phase3"
CONFIG = ROOT / "benchmark/phase4/study_config.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def keyed(data):
    result = {r["sample_id"]: r for r in data}
    if len(result) != len(data):
        raise ValueError("duplicate sample IDs")
    return result


@dataclass(frozen=True)
class Features:
    """Only prediction-time signals; no truth, strata, clusters, or audit fields."""
    top1_id: str
    margin: float
    component_agreement: bool


def extract_features(records):
    if len(records) < 2:
        raise ValueError("at least two evidence records required")
    first, second = records[:2]
    s1, s2 = float(first["fused_score"]), float(second["fused_score"])
    if not all(math.isfinite(v) for v in (s1, s2)) or not 0 <= s2 <= s1:
        raise ValueError("invalid descending finite retrieval scores")
    return Features(first["kegg_id"], (s1-s2)/s1 if s1 else 0.0,
                    first["bm25_rank"] == 1 and first["trained_biencoder_rank"] == 1)


def gate(features, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold outside [0,1]")
    return features.component_agreement and features.margin >= threshold


def predict(features, annotation, method, threshold=None):
    """Label-free route/output projection; missing grounded output is fatal."""
    if method not in {"fusion", "grounded", "threshold", "routed", "post_llm_threshold"}:
        raise ValueError("unknown policy")
    if method == "fusion" or (method in {"threshold", "routed"} and gate(features, threshold)):
        return {"prediction": features.top1_id, "score": features.margin, "llm_call": False, "route": "fusion"}
    if method == "threshold":
        return {"prediction": None, "score": 0.0, "llm_call": False, "route": "abstain_gate"}
    if annotation is None:
        raise ValueError("missing compatible grounded response; cannot simulate counterfactual")
    confidence = float(annotation["confidence"])
    if not 0 <= confidence <= 1 or not math.isfinite(confidence):
        raise ValueError("invalid self-reported confidence")
    prediction = None if annotation["abstain"] else annotation["predicted_kegg_id"]
    if method == "post_llm_threshold" and confidence < threshold:
        prediction = None
    return {"prediction": prediction, "score": confidence, "llm_call": True, "route": "grounded"}


def risk_coverage(correct, answered, scores):
    """Tie-grouped right-step integral up to native maximum coverage."""
    n = len(correct)
    if not n:
        raise ValueError("empty population")
    groups = defaultdict(list)
    for c, a, s in zip(correct, answered, scores):
        if a:
            groups[float(s)].append(bool(c))
    curve, count, errors, previous, area = [], 0, 0, 0.0, 0.0
    for score, values in sorted(groups.items(), reverse=True):
        count += len(values)
        errors += sum(not c for c in values)
        coverage, risk = count/n, errors/count
        area += (coverage-previous)*risk
        previous = coverage
        curve.append({"score": score, "coverage": coverage, "risk": risk, "answered": count})
    return {"curve": curve, "native_max_coverage": previous, "aurc_partial": area,
            "aurc_normalized": area/previous if previous else None}


def detection(target, score):
    if not target or len(set(target)) < 2:
        return {"n": len(target), "positives": sum(target), "auroc": None, "average_precision": None}
    return {"n": len(target), "positives": sum(target),
            "auroc": float(roc_auc_score(target, score)),
            "average_precision": float(average_precision_score(target, score))}


def metrics(data):
    n = len(data)
    correct = [r["correct"] for r in data]
    answered = [r["prediction"] is not None for r in data]
    k, a = sum(correct), sum(answered)
    result = {"n": n, "correct": k, "answered": a, "abstained": n-a,
              "accuracy": k/n, "coverage": a/n, "selective_accuracy": k/a if a else None,
              "selective_risk": 1-k/a if a else None,
              "simulated_llm_calls": sum(r["llm_call"] for r in data),
              "simulated_call_rate": sum(r["llm_call"] for r in data)/n,
              "replayed_cost_usd": sum(r["cost_usd"] for r in data),
              "replayed_input_tokens": sum(r["input_tokens"] for r in data),
              "replayed_cached_input_tokens": sum(r["cached_input_tokens"] for r in data),
              "replayed_output_tokens": sum(r["output_tokens"] for r in data),
              "replayed_serial_provider_latency_ms": sum(r["latency_ms"] for r in data),
              "reference_absence_consistent_abstention": sum(not p and not r["reference_present"] for r, p in zip(data, answered)),
              "reference_present_abstention": sum(not p and r["reference_present"] for r, p in zip(data, answered)),
              "answered_reference_absent": sum(p and not r["reference_present"] for r, p in zip(data, answered))}
    for dimension in ("cluster_id", "model_id"):
        groups = defaultdict(list)
        for r in data:
            groups[r[dimension]].append(r["correct"])
        result[f"{dimension}_macro_accuracy"] = float(np.mean([np.mean(v) for v in groups.values()]))
    result["risk_coverage"] = risk_coverage(correct, answered, [r["score"] for r in data])
    result["correctness_detection_answered_only"] = detection(
        [c for c, a in zip(correct, answered) if a], [r["score"] for r, a in zip(data, answered) if a])
    return result


def paired_bootstrap(data, reference, config):
    if [r["sample_id"] for r in data] != [r["sample_id"] for r in reference]:
        raise ValueError("bootstrap requires aligned pairs")
    clusters = sorted({r["cluster_id"] for r in data})
    totals = np.array([[sum(r["cluster_id"] == c for r in data),
                        sum(r["correct"] for r in data if r["cluster_id"] == c),
                        sum(r["correct"] for r in reference if r["cluster_id"] == c),
                        sum(r["llm_call"] for r in data if r["cluster_id"] == c)] for c in clusters], dtype=float)
    rng = np.random.default_rng(config["bootstrap_seed"])
    draw = rng.integers(0, len(clusters), size=(config["bootstrap_replicates"], len(clusters)))
    sums = totals[draw].sum(axis=1)
    accuracy, delta, calls = sums[:, 1]/sums[:, 0], (sums[:, 1]-sums[:, 2])/sums[:, 0], sums[:, 3]/sums[:, 0]
    return {"clusters": len(clusters), "replicates": config["bootstrap_replicates"], "seed": config["bootstrap_seed"],
            "accuracy_ci95": np.quantile(accuracy, [.025, .975]).tolist(),
            "accuracy_delta_vs_grounded_ci95": np.quantile(delta, [.025, .975]).tolist(),
            "simulated_call_rate_ci95": np.quantile(calls, [.025, .975]).tolist()}


def select_method(results, config):
    baseline = results["grounded"]
    eligible = []
    reasons = {}
    for name, result in results.items():
        if not name.startswith("routed_"):
            continue
        checks = {
            "accuracy_preserved": result["correct"] >= baseline["correct"],
            "coverage_preserved": result["answered"] >= baseline["answered"],
            "selective_risk_preserved": result["selective_risk"] is not None and result["selective_risk"] <= baseline["selective_risk"],
            "call_reduction": result["simulated_call_rate"] <= 1-config["min_call_reduction"],
            "cluster_screen": result["bootstrap"]["accuracy_delta_vs_grounded_ci95"][0] >= config["min_delta_lower_bound"],
        }
        reasons[name] = checks
        if all(checks.values()):
            eligible.append(name)
    selected = max(eligible, key=lambda k: (results[k]["correct"], -results[k]["simulated_llm_calls"], float(k.split("_")[1]))) if eligible else ("fusion" if results["fusion"]["correct"] > baseline["correct"] else "grounded")
    return {"selected_method": selected, "eligible_routed_candidates": eligible, "criteria": reasons,
            "scope": "validation-selected cached replay only; not independent confirmation"}


def csv_write(path, records):
    if not records:
        raise ValueError("empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def verify_prior():
    from benchmark.scripts import phase3_error_inventory, phase3_error_sampling, phase3_error_reviewer, phase3c_grounded, phase3c_validation, phase3b_release, phase3_biencoder
    return {"inventory": phase3_error_inventory.verify_manifest(),
            "sampling": phase3_error_sampling.verify_manifest(),
            "reviewer": phase3_error_reviewer.verify_manifest(),
            "phase3c_smoke": phase3c_grounded.verify_artifacts(),
            "phase3c_validation": phase3c_validation.verify_manifest(),
            "phase3b_full": phase3_biencoder.verify_manifest(phase3_biencoder.FULL_OUT),
            "phase3b_fusion": phase3b_release.verify_fusion_manifest()}


def plot_results(results, out):
    # The optional local plotting dependency directory is ignored, never global.
    deps = ROOT / "benchmark/phase4/_deps"
    if deps.exists():
        sys.path.append(str(deps))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "benchmark/phase4/_mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for name, result in results.items():
        if name.startswith("threshold_"):
            continue
        curve = result["risk_coverage"]["curve"]
        axes[0].step([r["coverage"] for r in curve], [r["risk"] for r in curve], where="pre", label=name)
        axes[1].scatter(result["simulated_call_rate"], result["accuracy"], label=name)
    axes[0].set(xlabel="Coverage / all 163 validation reactions", ylabel="Selective risk (exact-reference errors / answered)", title="Native answered outputs only; no extrapolation", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Simulated grounded-call fraction", ylabel="Exact accuracy / all eligible", title="Cached replay; retrieval overhead unmeasured", xlim=(-.03, 1.03), ylim=(0, 1))
    axes[0].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.25)
    fig.savefig(out / "validation_tradeoffs.svg")
    fig.savefig(out / "validation_tradeoffs.png", dpi=170)
    plt.close(fig)
    return {"matplotlib": matplotlib.__version__}


def report_results(results, selection, out):
    lines = ["# Phase 4 cached validation routing results", "",
             f"Selected method: **{selection['selected_method']}** by the criteria written in benchmark/PHASE4.md before execution.", "",
             "All comparisons use the same 163 validation reactions and unchanged frozen references. No human review judgments were loaded. No training, new inference, provider calls, held-out data access, or external biological queries occurred.", "",
             "| Method | Correct / 163 | Answered / 163 | Correct / answered | Simulated grounded calls | Replayed USD |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['correct']}/163 | {m['answered']}/163 | {m['correct']}/{m['answered']} | {m['simulated_llm_calls']}/163 | {m['replayed_cost_usd']:.6f} |")
    lines += ["", "## Selection and resource interpretation", "",
              "No candidate may be promoted solely for saving calls. All five criteria in selection.json must pass. The three coarse gates were not expanded after results were observed. A negative development result retains the existing stronger method.",
              "Frozen provider cost/usage/serial latency are summed only for replay routes choosing the grounded output. The present study's actual new calls and provider cost are zero. Historical calls were already paid; replay savings are not refunds. Both retrievers still run for all queries. Retrieval and routing overhead is unmeasured, so end-to-end speedup is not established.",
              "", "## Uncertainty, abstention, and detection", "",
              "metrics.json contains paired 2,000-replicate cluster-bootstrap intervals for accuracy, delta versus grounded and simulated call rate; 12 clusters, seed 20261105. Threshold selection and prior use of these validation cases make these intervals descriptive, with selection optimism and no multiplicity correction.",
              "Reference-absence-consistent abstention means no accepted frozen reference ID appears in the Top-10 evidence. Reference-present abstention means one does. Neither category establishes biological appropriateness. The post-LLM threshold still consumes all 163 simulated calls.",
              "Risk-coverage curves use native answered cases only, exact-score ties as groups, right-step partial area and area divided by maximum native coverage. Methods with different native coverage have different integration ranges. Grounded confidence and retrieval margins are uncalibrated. Routed mixed-score curves combine unlike scales and are diagnostic only. Correctness detection is evaluated among answered cases; absence detection uses negative retrieval margin and low gate confidence. Values and confusion counts are in metrics.json and answer_absence_detection.json. No calibrator was fit: compatible training confidence/output data are unavailable and the reused small pilot is unsuitable for both fitting and selecting one.",
              "", "## Coverage and limitations", "",
              "subgroups.csv reports denominators, exact accuracy, coverage, selective accuracy and call counts by corrected retrieval stratum, model cluster and model. The 163 pilot cases were stratified for development; unweighted micro accuracy describes this pilot, not population-wide accuracy. No weighting is invented.",
              "This is a replay of cached outputs under unchanged requests and evidence, not a newly executed end-to-end system or fabricated counterfactual model answers. Two baseline abstentions carry evidence-compliance errors and are retained rather than repaired or excluded. Output compliance does not establish biological correctness. Missing or mismatched cache entries abort replay.",
              "Full-catalog retrieval already handles rule-based empty/absent candidates; no redundant retrieval component was added. No controlled answer-absent data, new labels, calibration fitting, or audit-based policy optimization was performed. The single-reviewer audit remains complete; it is descriptive development evidence, not an independent evaluation. Thresholds/model choices require later sealed evaluation.",
              "", "## Reproduction and freeze status", "",
              "From the isolated Phase 4 worktree:", "", "```powershell",
              "python -m benchmark.scripts.phase4_routing --out benchmark/phase4/_validation_reproduced",
              "python -m benchmark.scripts.phase4_freeze --prepare --study benchmark/phase4/_validation_reproduced --out benchmark/phase4/_freeze_reproduced",
              "python -m benchmark.scripts.phase4_freeze --dry-run --out benchmark/phase4/_dry_run_reproduced.json", "```", "",
              "Choose new output paths on every replay; existing output directories are never overwritten. Runtime: existing NumPy/scikit-learn/Pydantic/benchmark dependencies plus Matplotlib for exported PNG/SVG plots. Source and input hashes, software versions, base revision and prior manifest results are in provenance.json. Checksum files pin the uncommitted Phase 4 source state; no commit was created.",
              "The selected method is validation-selected. The evaluation package is explicitly a draft: no operational held-out executor, approved budget, or immutable hosted-provider identity has been finalized. The documented --execute command deliberately fails closed before reading data. See freeze/evaluation_freeze.json for exact blockers and proposed budget limits. Do not execute or unlock the test.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run(out):
    if out.exists():
        raise ValueError("refusing to overwrite existing study output")
    config = load(CONFIG)
    if config["thresholds"] != [.05, .1, .2] or config["new_api_calls_allowed"] != 0:
        raise ValueError("protocol/config drift")
    verification = verify_prior()
    if any(verification.values()):
        raise ValueError(f"prior manifest failure: {verification}")
    base = PHASE3 / "phase3c_validation"
    files = [base / n for n in ("sample_ids.jsonl", "frozen_evidence.jsonl", "frozen_responses.jsonl", "requests.jsonl", "answer_key_join.jsonl", "config.json", "artifact_manifest.json")]
    files += [CONFIG, ROOT / "benchmark/PHASE4.md", Path(__file__), ROOT / "benchmark/scripts/phase4_freeze.py", ROOT / "tests/test_phase4_routing.py", ROOT / "tests/test_phase4_freeze.py",
              ROOT / "benchmark/scripts/phase3c_grounded.py", ROOT / "benchmark/scripts/phase3c_validation.py", ROOT / "benchmark/scripts/phase3_biencoder.py", ROOT / "benchmark/scripts/phase3_retrieval.py",
              PHASE3 / "phase3b_full/selected_checkpoint.json", PHASE3 / "phase3b_full/full_training_config.json", PHASE3 / "phase3b_fusion/config.json", PHASE3 / "pricing.openai.gpt-5.6-terra.json"]
    before = {p.relative_to(ROOT).as_posix(): {"sha256_bytes": sha(p), "sha256_portable": sha256_portable(p)} for p in files}
    samples = rows(base / "sample_ids.jsonl")
    if len(samples) != 163 or len(keyed(samples)) != 163 or any(r["split"] != "validation" for r in samples):
        raise ValueError("only the exact 163 validation cases are eligible")
    evidence, responses, requests = [keyed(rows(base / name)) for name in ("frozen_evidence.jsonl", "frozen_responses.jsonl", "requests.jsonl")]
    expected = set(keyed(samples))
    if any(set(m) != expected for m in (evidence, responses, requests)):
        raise ValueError("missing cached outputs; replay would fabricate a route")
    policies = [("fusion", "fusion", None), ("grounded", "grounded", None)]
    policies += [(f"{kind}_{t:g}", kind, t) for kind in ("threshold", "routed") for t in config["thresholds"]]
    policies += [("post_llm_threshold_0.8", "post_llm_threshold", config["post_llm_confidence_threshold"])]
    predictions, feature_rows = defaultdict(list), []
    for sample in samples:
        sid = sample["sample_id"]
        e, r, request = evidence[sid], responses[sid], requests[sid]
        for source in (e, r, request):
            if any(source[k] != sample[k] for k in ("model_id", "reaction_id")) or source["evidence_digest"] != e["evidence_digest"]:
                raise ValueError("cached identity/evidence digest mismatch")
        # Two frozen abstentions carry compliance errors; retain them exactly as
        # the Phase 3C baseline did rather than dropping or repairing them.
        if r["terminal_status"] not in {"succeeded", "compliance_error"} or not isinstance(r.get("annotation"), dict):
            raise ValueError("missing completed cached annotation")
        target = json.loads(request["payload"]["input"].split("TARGET\n", 1)[1].split("\nEVIDENCE\n", 1)[0])
        if set(target) != {"query"} or re.search(r"R\d{5}", target["query"], flags=re.I):
            raise ValueError("reference-shaped KEGG reaction ID in target query")
        f = extract_features(e["records"])
        feature_rows.append({"sample_id": sid, **asdict(f)})
        for name, method, threshold in policies:
            predictions[name].append({"sample_id": sid, **predict(f, r["annotation"], method, threshold)})
    out.mkdir(parents=True)
    # Policy decisions are frozen before any reference labels are read here.
    write(out / "label_free_predictions.json", dict(predictions))
    csv_write(out / "label_free_features.csv", feature_rows)
    prediction_digest = sha(out / "label_free_predictions.json")
    truth = keyed(rows(base / "answer_key_join.jsonl"))
    if set(truth) != expected:
        raise ValueError("reference join coverage mismatch")
    evaluated, results = {}, {}
    for name, prediction_rows in predictions.items():
        scored = []
        for prediction in prediction_rows:
            sid = prediction["sample_id"]
            t, r, e = truth[sid], responses[sid], evidence[sid]
            if t["evidence_digest"] != e["evidence_digest"] or t["fusion_top1_kegg_id"] != e["records"][0]["kegg_id"]:
                raise ValueError("reference/frozen output mismatch")
            for key in ("model_id", "reaction_id", "cluster_id"):
                if t[key] != r[key]:
                    raise ValueError("reference identity mismatch")
            reference_ids = set(t["ground_truth_ids"])
            if not reference_ids:
                raise ValueError("missing frozen reference labels")
            use = prediction["llm_call"]
            usage = r["usage"]
            if usage.get("usage_missing") or any(r.get(k) is None for k in ("cost_usd", "latency_ms")):
                raise ValueError("resource measurements unavailable; do not invent zeros")
            scored.append({**prediction, "correct": prediction["prediction"] in reference_ids,
                           "reference_present": bool(reference_ids & {c["kegg_id"] for c in e["records"]}),
                           "cluster_id": t["cluster_id"], "model_id": t["model_id"], "stratum": t["corrected_stratum"],
                           "cost_usd": r["cost_usd"] if use else 0.0, "latency_ms": r["latency_ms"] if use else 0.0,
                           "input_tokens": usage["input_tokens"] if use else 0,
                           "cached_input_tokens": usage["cached_input_tokens"] if use else 0,
                           "output_tokens": usage["output_tokens"] if use else 0})
        evaluated[name] = scored
        results[name] = metrics(scored)
    if results["grounded"]["correct"] != 102 or results["fusion"]["correct"] != 93 or results["grounded"]["answered"] != 124:
        raise ValueError("frozen baseline reproduction failed")
    for name, scored in evaluated.items():
        results[name]["bootstrap"] = paired_bootstrap(scored, evaluated["grounded"], config)
    selection = select_method(results, config)
    write(out / "selection.json", selection)
    write(out / "metrics.json", results)
    table = [{"method": name, **{k: v for k, v in result.items() if not isinstance(v, dict)}} for name, result in results.items()]
    csv_write(out / "comparison.csv", table)
    subgroup = []
    for name, scored in evaluated.items():
        for dim in ("stratum", "cluster_id", "model_id"):
            for key in sorted({r[dim] for r in scored}):
                subset = [r for r in scored if r[dim] == key]
                m = metrics(subset)
                subgroup.append({"method": name, "dimension": dim, "group": key,
                                 **{k: m[k] for k in ("n", "correct", "answered", "abstained", "accuracy", "coverage", "selective_accuracy", "simulated_llm_calls")}})
    csv_write(out / "subgroups.csv", subgroup)
    csv_write(out / "evaluated_predictions.csv", [{"method": name, **r} for name, data in evaluated.items() for r in data])
    margins = {r["sample_id"]: r["margin"] for r in feature_rows}
    absent = [not r["reference_present"] for r in evaluated["fusion"]]
    absence = {"negative_margin_detection": detection(absent, [-margins[r["sample_id"]] for r in evaluated["fusion"]])}
    for threshold in config["thresholds"]:
        flags = [not gate(Features(**{k: f[k] for k in ("top1_id", "margin", "component_agreement")}), threshold) for f in feature_rows]
        absence[str(threshold)] = {"true_positive_absent": sum(a and f for a, f in zip(absent, flags)),
                                   "false_positive_absent": sum(not a and f for a, f in zip(absent, flags)),
                                   "false_negative_absent": sum(a and not f for a, f in zip(absent, flags)),
                                   "true_negative_absent": sum(not a and not f for a, f in zip(absent, flags))}
    write(out / "answer_absence_detection.json", absence)
    versions = plot_results(results, out)
    report_results(results, selection, out)
    write(out / "provenance.json", {"git_base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
          "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
          "source_state": "uncommitted Phase 4 files pinned by checksums; no commit created",
          "inputs": before, "prediction_digest_before_reference_join": prediction_digest,
          "prior_read_only_verifiers": verification, "python": platform.python_version(), "numpy": np.__version__, **versions,
          "test_rows_read": 0, "human_review_files_loaded_by_study": 0, "new_api_calls": 0, "new_cost_usd": 0,
          "matched_training_confidence_outputs_available": False, "calibrator_fitted": False})
    for relative, record in before.items():
        if sha(ROOT / relative) != record["sha256_bytes"]:
            raise ValueError(f"input mutated: {relative}")
    if sha(out / "label_free_predictions.json") != prediction_digest:
        raise ValueError("prediction freeze mutated after reference join")
    print(json.dumps({"selected": selection, "comparison": table}, indent=2))
    return selection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark/phase4/validation")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.relative_to((ROOT / "benchmark/phase4").resolve())
    run(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
