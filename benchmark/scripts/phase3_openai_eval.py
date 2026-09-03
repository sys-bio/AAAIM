"""Offline evaluation for the Phase 3A direct open-set validation pilot.

Join the answer key only after responses are frozen. The experimental unit is
the reaction (163), not the 489 prompt rows. Test-split reactions are never
loaded into the scored population.

Usage::

    python benchmark/scripts/phase3_openai_eval.py
    python benchmark/scripts/phase3_openai_eval.py --results-dir benchmark/phase3/validation
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.scripts.phase3_common import (
    OUT_PILOT,
    OUT_PILOT_KEY,
    OUT_SPLITS,
    OUT_VALIDATION_DIR,
    PILOT_SPLIT,
    REACTIONS_CSV,
    RETRIEVAL_FAILURE_STRATA,
    STRATA,
    STRATUM_TOP1,
    TEXT_CSV,
    atomic_write_json,
    parse_kegg_ids,
    sha256_file,
)
from benchmark.scripts.phase3_eval import score_results
from benchmark.scripts.phase3_modes import ModeResult, Prediction
from benchmark.scripts.phase3_openai_run import row_to_mode_result
from benchmark.scripts.phase3_prompts import CONTEXT_VARIANTS

logger = logging.getLogger("phase3_openai_eval")

NONCANONICAL_RE = re.compile(
    r"\b(bind|binding|complex|dissociat|transport|exchange|signaling|signalling|"
    r"receptor|phosphorylat|dephosphorylat)\b",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_train_target_ids() -> set:
    """KEGG ids appearing in the *train* split only.

    The frozen Phase 2 reaction table is read only to resolve train keys.
    Validation and test rows are dropped before any aggregation. Test reactions
    are never scored here.
    """
    splits = pd.read_csv(OUT_SPLITS, usecols=["model_id", "reaction_id", "split"])
    train_keys = splits.loc[splits.split.astype(str) == "train", ["model_id", "reaction_id"]]
    reactions = pd.read_csv(
        REACTIONS_CSV, usecols=["model_id", "reaction_id", "ground_truth_kegg_all"],
    )
    train = reactions.merge(train_keys, on=["model_id", "reaction_id"], how="inner")
    ids: set = set()
    for value in train.ground_truth_kegg_all:
        ids.update(parse_kegg_ids(value))
    return ids


def load_answer_key(path: Path = OUT_PILOT_KEY) -> pd.DataFrame:
    return pd.read_csv(path)


def answer_key_map(key: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
    return {
        (str(rec.model_id), str(rec.reaction_id)): parse_kegg_ids(rec.ground_truth_kegg_all)
        for rec in key.itertuples(index=False)
    }


def _open_set_outcome(row: Mapping[str, Any]) -> str:
    if row.get("parse_error"):
        return "compliance_error"
    if row.get("abstain"):
        return "abstain"
    if row.get("exact_top1"):
        return "correct_top1"
    if row.get("n_malformed_ids"):
        return "incorrect_malformed"
    if row.get("n_absent_from_catalog_ids") and not row.get("n_in_catalog_ids"):
        return "incorrect_absent_from_catalog"
    if row.get("n_in_catalog_ids"):
        return "incorrect_in_catalog"
    return "incorrect_unanswered"


def _enrich_score_rows(
    score: Mapping[str, Any],
    result_rows: Sequence[Mapping[str, Any]],
    key: pd.DataFrame,
    seen_targets: set,
    sample: pd.DataFrame,
) -> List[Dict[str, Any]]:
    keyed_results = {
        (r["sample_id"], r["variant"]): r for r in result_rows
    }
    key_lookup = {
        (str(rec.model_id), str(rec.reaction_id)): rec for rec in key.itertuples(index=False)
    }
    sample_lookup = {
        str(rec.sample_id): rec for rec in sample.itertuples(index=False)
    }
    out = []
    for row in score["rows"]:
        raw = keyed_results.get((row["sample_id"], row["variant"]), {})
        rec = key_lookup.get((row["model_id"], row["reaction_id"]))
        samp = sample_lookup.get(str(row["sample_id"]))
        truth = parse_kegg_ids(getattr(rec, "ground_truth_kegg_all", "")) if rec is not None else []
        n_gt = int(getattr(rec, "num_ground_truth_ids", 0) or len(truth))
        enriched = dict(row)
        enriched["open_set_outcome"] = _open_set_outcome(row)
        enriched["terminal_status"] = raw.get("terminal_status")
        enriched["confidence"] = row.get("confidence")
        enriched["cost_usd"] = raw.get("cost_usd")
        enriched["n_input_tokens"] = raw.get("n_input_tokens")
        enriched["n_output_tokens"] = raw.get("n_output_tokens")
        enriched["n_reasoning_tokens"] = raw.get("n_reasoning_tokens")
        enriched["latency_ms"] = raw.get("latency_ms")
        enriched["cache_hit"] = raw.get("cache_hit")
        enriched["rationale"] = raw.get("rationale") or ""
        enriched["num_ground_truth_ids"] = n_gt
        enriched["multi_target"] = n_gt > 1
        enriched["target_seen_in_train"] = any(t in seen_targets for t in truth)
        if samp is not None:
            enriched["species_annotation_source"] = str(getattr(samp, "species_annotation_source", "") or "")
            enriched["complexity_bucket"] = str(getattr(samp, "complexity_bucket", "") or "")
            enriched["status"] = str(getattr(samp, "status", "") or "")
            enriched["candidate_set_size"] = getattr(samp, "candidate_set_size", None)
            enriched["is_genome_scale"] = bool(getattr(samp, "is_genome_scale", False))
        preds = raw.get("predictions") or []
        enriched["top1_kegg_id"] = preds[0]["kegg_id"] if preds else None
        out.append(enriched)
    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(sum(values) / len(values)), 4)


def _rate(n: int, d: int) -> Optional[float]:
    if d == 0:
        return None
    return round(n / d, 4)


def variant_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    n_abstain = sum(1 for r in rows if r.get("abstain"))
    n_answered = sum(1 for r in rows if r.get("answered"))
    n_exact = sum(1 for r in rows if r.get("exact_top1"))
    n_exact3 = sum(1 for r in rows if r.get("exact_top3"))
    n_brite = sum(1 for r in rows if r.get("brite_top1"))
    n_brite3 = sum(1 for r in rows if r.get("brite_top3"))
    n_schema = sum(1 for r in rows if r.get("terminal_status") == "schema_invalid")
    n_compliance = sum(1 for r in rows if r.get("parse_error"))
    n_refused = sum(1 for r in rows if r.get("terminal_status") == "refused")
    n_malformed = sum(1 for r in rows if r.get("n_malformed_ids"))
    n_absent = sum(1 for r in rows if r.get("n_absent_from_catalog_ids"))
    n_incorrect_in_catalog = sum(
        1 for r in rows if r.get("open_set_outcome") == "incorrect_in_catalog"
    )
    n_control_abstain = sum(
        1 for r in rows if r.get("abstain") and r.get("stratum") == STRATUM_TOP1
    )
    n_failure_abstain = sum(
        1 for r in rows
        if r.get("abstain") and r.get("stratum") in RETRIEVAL_FAILURE_STRATA
    )
    return {
        "n_reactions": n,
        "exact_top1": _rate(n_exact, n),
        "exact_top1_n": n_exact,
        "exact_top1_d": n,
        "exact_top3": _rate(n_exact3, n),
        "brite_top1": _rate(n_brite, n),
        "brite_top3": _rate(n_brite3, n),
        "abstention_rate": _rate(n_abstain, n),
        "n_abstain": n_abstain,
        "coverage": _rate(n_answered, n),
        "n_answered": n_answered,
        "selective_exact_top1": _rate(n_exact, n_answered) if n_answered else None,
        "selective_exact_top1_n": n_exact,
        "selective_exact_top1_d": n_answered,
        "incorrect_in_catalog_rate": _rate(n_incorrect_in_catalog, n),
        "n_incorrect_in_catalog": n_incorrect_in_catalog,
        "malformed_id_rate": _rate(n_malformed, n),
        "absent_from_catalog_rate": _rate(n_absent, n),
        "schema_invalid_rate": _rate(n_schema, n),
        "compliance_error_rate": _rate(n_compliance, n),
        "refusal_rate": _rate(n_refused, n),
        "n_incorrect_abstention_on_top1_control": n_control_abstain,
        "incorrect_abstention_on_top1_control_rate": _rate(
            n_control_abstain, sum(1 for r in rows if r.get("stratum") == STRATUM_TOP1)
        ),
        "n_abstention_on_retrieval_failure": n_failure_abstain,
        "note": (
            "Abstention is not recovery. Incorrect-abstention on the top-1 control "
            "is operational: the frozen heuristic already recovered an exact id."
        ),
    }


def recovery_block(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    failure = [r for r in rows if r.get("stratum") in RETRIEVAL_FAILURE_STRATA]
    n_fail = len(failure)
    n_correct = sum(1 for r in failure if r.get("exact_top1"))
    out["all_retrieval_failure"] = {
        "numerator": n_correct,
        "denominator": n_fail,
        "recovery_rate": _rate(n_correct, n_fail),
        "definition": (
            "exact_top1 / reactions in unconstrained + empty_constrained + "
            "nonempty_answer_absent + retrievable_rerank_failure. Abstention is not recovery."
        ),
        "n_abstain": sum(1 for r in failure if r.get("abstain")),
        "n_incorrect_in_catalog": sum(
            1 for r in failure if r.get("open_set_outcome") == "incorrect_in_catalog"
        ),
    }
    for stratum in STRATA:
        sub = [r for r in rows if r.get("stratum") == stratum]
        n = len(sub)
        n_ok = sum(1 for r in sub if r.get("exact_top1"))
        label = "control_exact_top1" if stratum == STRATUM_TOP1 else "recovery_rate"
        out[stratum] = {
            "numerator": n_ok,
            "denominator": n,
            label: _rate(n_ok, n),
            "abstention_rate": _rate(sum(1 for r in sub if r.get("abstain")), n),
            "coverage": _rate(sum(1 for r in sub if r.get("answered")), n),
            "selective_exact_top1": _rate(
                n_ok, sum(1 for r in sub if r.get("answered"))
            ),
        }
    return out


def _index_by_sample(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    return {str(r["sample_id"]): r for r in rows}


def paired_context_comparison(
    by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    baseline: str = "target_only",
    n_boot: int = 1000,
    seed: int = 20260902,
) -> Dict[str, Any]:
    base = _index_by_sample(by_variant[baseline])
    out: Dict[str, Any] = {}
    for variant in CONTEXT_VARIANTS:
        if variant == baseline:
            continue
        other = _index_by_sample(by_variant[variant])
        ids = sorted(set(base) & set(other))
        helped = harmed = unchanged = abstention_only = 0
        agree = 0
        conf_deltas = []
        acc_deltas = []
        cov_deltas = []
        sel_pairs = []
        clusters = []
        for sid in ids:
            a, b = base[sid], other[sid]
            clusters.append(str(a.get("cluster_id") or ""))
            a_ok, b_ok = bool(a.get("exact_top1")), bool(b.get("exact_top1"))
            a_ans, b_ans = bool(a.get("answered")), bool(b.get("answered"))
            acc_deltas.append(float(b_ok) - float(a_ok))
            cov_deltas.append(float(b_ans) - float(a_ans))
            if a_ok != b_ok:
                if b_ok:
                    helped += 1
                else:
                    harmed += 1
            elif a.get("abstain") != b.get("abstain"):
                abstention_only += 1
            else:
                unchanged += 1
            if (a.get("top1_kegg_id") or None) == (b.get("top1_kegg_id") or None) and bool(
                a.get("abstain")
            ) == bool(b.get("abstain")):
                agree += 1
            if a.get("confidence") is not None and b.get("confidence") is not None:
                conf_deltas.append(float(b["confidence"]) - float(a["confidence"]))
            if a_ans or b_ans:
                sel_pairs.append((a_ok if a_ans else None, b_ok if b_ans else None))
        out[f"{variant}_vs_{baseline}"] = {
            "n_paired_reactions": len(ids),
            "n_helped": helped,
            "n_harmed": harmed,
            "n_unchanged": unchanged,
            "n_changed_only_in_abstention": abstention_only,
            "accuracy_delta": _mean(acc_deltas),
            "coverage_delta": _mean(cov_deltas),
            "prediction_agreement_rate": _rate(agree, len(ids)),
            "mean_confidence_delta_answered_pairs": _mean(conf_deltas),
            "accuracy_delta_cluster_bootstrap": _paired_cluster_ci(
                ids, clusters, acc_deltas, n_boot=n_boot, seed=seed,
            ),
            "coverage_delta_cluster_bootstrap": _paired_cluster_ci(
                ids, clusters, cov_deltas, n_boot=n_boot, seed=seed,
            ),
            "limitation": (
                "Intervals are exploratory. Validation has 12 clusters; related "
                "models share biology, so reaction-level tests overstate precision."
            ),
        }
    return out


def _paired_cluster_ci(
    ids: Sequence[str],
    clusters: Sequence[str],
    deltas: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    by_cluster: Dict[str, List[float]] = defaultdict(list)
    for cluster, delta in zip(clusters, deltas):
        by_cluster[cluster].append(delta)
    keys = list(by_cluster)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        drawn = []
        for key in (rng.choice(keys) for _ in keys):
            drawn.extend(by_cluster[key])
        stats.append(sum(drawn) / len(drawn) if drawn else 0.0)
    stats.sort()
    return {
        "mean": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "low": round(stats[int(0.025 * n_boot)], 4) if stats else None,
        "high": round(stats[min(len(stats) - 1, int(0.975 * n_boot))], 4) if stats else None,
        "n": len(deltas),
        "n_clusters": len(keys),
        "n_boot": n_boot,
        "method": "cluster_bootstrap_paired_delta",
    }


def calibration_report(rows: Sequence[Mapping[str, Any]], n_bins: int = 5) -> Dict[str, Any]:
    answered = [r for r in rows if r.get("answered") and r.get("confidence") is not None]
    correct = [float(r["confidence"]) for r in answered if r.get("exact_top1")]
    incorrect = [float(r["confidence"]) for r in answered if not r.get("exact_top1")]
    abstained = [r for r in rows if r.get("abstain")]
    brier = None
    if answered:
        brier = round(
            sum((float(r["confidence"]) - (1.0 if r.get("exact_top1") else 0.0)) ** 2
                for r in answered) / len(answered),
            4,
        )
    bins = []
    if answered:
        for i in range(n_bins):
            lo, hi = i / n_bins, (i + 1) / n_bins
            members = [
                r for r in answered
                if lo <= float(r["confidence"]) < hi or (i == n_bins - 1 and float(r["confidence"]) == 1.0)
            ]
            if not members:
                bins.append({"lo": lo, "hi": hi, "n": 0})
                continue
            acc = sum(1 for r in members if r.get("exact_top1")) / len(members)
            conf = sum(float(r["confidence"]) for r in members) / len(members)
            bins.append({
                "lo": lo, "hi": hi, "n": len(members),
                "accuracy": round(acc, 4), "mean_confidence": round(conf, 4),
                "gap": round(abs(acc - conf), 4),
            })
    ece = None
    if answered and bins:
        ece = round(
            sum(b["n"] * b.get("gap", 0) for b in bins if b["n"]) / len(answered),
            4,
        )
    from benchmark.scripts.phase3_eval import coverage_curve
    return {
        "n_answered_with_confidence": len(answered),
        "n_abstain": len(abstained),
        "mean_confidence_correct": _mean(correct),
        "mean_confidence_incorrect": _mean(incorrect),
        "brier_score": brier,
        "ece": ece,
        "reliability_bins": bins,
        "coverage_curve": coverage_curve(list(rows)),
        "note": (
            "Brier/ECE are exploratory. Self-reported confidence is not assumed "
            "to be a calibrated probability."
        ),
    }


def stratify(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field))].append(row)
    return {key: variant_metrics(vals) for key, vals in sorted(groups.items())}


def cost_report(result_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n_reactions = len({r["sample_id"] for r in result_rows})
    n_cached = sum(1 for r in result_rows if r.get("cache_hit"))
    n_purchased_this_file = sum(1 for r in result_rows if not r.get("cache_hit"))
    total = sum(float(r.get("cost_usd") or 0) for r in result_rows)
    by_variant = {}
    for variant in CONTEXT_VARIANTS:
        sub = [r for r in result_rows if r.get("variant") == variant]
        by_variant[variant] = {
            "n": len(sub),
            "cost_usd": round(sum(float(r.get("cost_usd") or 0) for r in sub), 6),
            "input_tokens": sum(int(r["n_input_tokens"]) for r in sub if r.get("n_input_tokens") is not None),
            "output_tokens": sum(int(r["n_output_tokens"]) for r in sub if r.get("n_output_tokens") is not None),
            "reasoning_tokens": sum(int(r["n_reasoning_tokens"]) for r in sub if r.get("n_reasoning_tokens") is not None),
        }
    latencies = [float(r["latency_ms"]) for r in result_rows if r.get("latency_ms") is not None]
    latencies.sort()
    return {
        "n_live_this_file": n_purchased_this_file,
        "n_cached": n_cached,
        "total_cost_usd": round(total, 6),
        "cost_per_reaction_usd": round(total / n_reactions, 6) if n_reactions else None,
        "input_tokens": sum(int(r["n_input_tokens"]) for r in result_rows if r.get("n_input_tokens") is not None),
        "cached_input_tokens": sum(
            int(r["n_cached_input_tokens"]) for r in result_rows if r.get("n_cached_input_tokens") is not None
        ),
        "output_tokens": sum(int(r["n_output_tokens"]) for r in result_rows if r.get("n_output_tokens") is not None),
        "reasoning_tokens": sum(
            int(r["n_reasoning_tokens"]) for r in result_rows if r.get("n_reasoning_tokens") is not None
        ),
        "by_variant": by_variant,
        "latency_ms": {
            "n": len(latencies),
            "mean": _mean(latencies),
            "p50": latencies[len(latencies) // 2] if latencies else None,
            "p90": latencies[int(0.9 * (len(latencies) - 1))] if latencies else None,
            "max": latencies[-1] if latencies else None,
        },
        "note": (
            "Dollar values are calculated from recorded usage and the frozen "
            "pricing snapshot. cache_hit means the row was replayed in this "
            "file; cost_usd is still the original purchase."
        ),
    }


def attach_text(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], str]:
    keys = {(str(r["model_id"]), str(r["reaction_id"])) for r in rows}
    text = pd.read_csv(TEXT_CSV)
    text["model_id"] = text.model_id.astype(str)
    text["reaction_id"] = text.reaction_id.astype(str)
    out = {}
    for rec in text.itertuples(index=False):
        key = (str(rec.model_id), str(rec.reaction_id))
        if key in keys:
            out[key] = str(getattr(rec, "query_text", "") or getattr(rec, "reaction_name", "") or "")
    return out


def failure_taxonomy(rows_by_variant: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """Deterministic mechanical selection of representative cases."""
    baseline = list(rows_by_variant["target_only"])
    neighborhood = _index_by_sample(rows_by_variant["target_plus_neighborhood"])
    model_ctx = _index_by_sample(rows_by_variant["target_plus_model"])
    text = attach_text(baseline)

    def pack(row, tag, hypothesis):
        key = (row["model_id"], row["reaction_id"])
        return {
            "tag": tag,
            "sample_id": row["sample_id"],
            "model_id": row["model_id"],
            "reaction_id": row["reaction_id"],
            "cluster_id": row.get("cluster_id"),
            "stratum": row.get("stratum"),
            "variant": row.get("variant"),
            "open_set_outcome": row.get("open_set_outcome"),
            "exact_top1": row.get("exact_top1"),
            "abstain": row.get("abstain"),
            "confidence": row.get("confidence"),
            "top1_kegg_id": row.get("top1_kegg_id"),
            "rationale": (row.get("rationale") or "")[:400],
            "query_text": (text.get(key) or "")[:240],
            "hypothesis": hypothesis,
        }

    cases: List[Dict[str, Any]] = []
    recovered = [
        r for r in baseline
        if r.get("stratum") in RETRIEVAL_FAILURE_STRATA and r.get("exact_top1")
    ]
    recovered.sort(key=lambda r: float(r.get("confidence") or 0), reverse=True)
    if recovered:
        cases.append(pack(recovered[0], "correct_open_set_recovery", "parametric_knowledge_success"))

    wrong = [
        r for r in baseline
        if r.get("open_set_outcome") == "incorrect_in_catalog"
    ]
    wrong.sort(key=lambda r: float(r.get("confidence") or 0), reverse=True)
    if wrong:
        cases.append(pack(wrong[0], "confident_valid_but_wrong", "hallucinated_or_near_miss_identifier"))

    abstain_fail = [
        r for r in baseline
        if r.get("abstain") and r.get("stratum") in RETRIEVAL_FAILURE_STRATA
        and NONCANONICAL_RE.search((r.get("rationale") or "") + (text.get((r["model_id"], r["reaction_id"])) or ""))
    ]
    if abstain_fail:
        cases.append(pack(abstain_fail[0], "appropriate_abstention_noncanonical", "insufficient_or_noncanonical_event"))

    control_abstain = [
        r for r in baseline if r.get("abstain") and r.get("stratum") == STRATUM_TOP1
    ]
    if control_abstain:
        cases.append(pack(control_abstain[0], "incorrect_abstention_on_control", "insufficient_prompt_evidence_or_over_abstention"))

    helped_n, harmed_n, helped_m, harmed_m = [], [], [], []
    for row in baseline:
        sid = row["sample_id"]
        nb = neighborhood.get(sid)
        md = model_ctx.get(sid)
        if nb and (not row.get("exact_top1")) and nb.get("exact_top1"):
            helped_n.append(nb)
        if nb and row.get("exact_top1") and not nb.get("exact_top1"):
            harmed_n.append(nb)
        if md and (not row.get("exact_top1")) and md.get("exact_top1"):
            helped_m.append(md)
        if md and row.get("exact_top1") and not md.get("exact_top1"):
            harmed_m.append(md)
    if helped_m:
        cases.append(pack(helped_m[0], "helped_by_model_context", "context_added_identifying_cue"))
    if helped_n:
        cases.append(pack(helped_n[0], "helped_by_neighborhood_context", "context_added_identifying_cue"))
    if harmed_m:
        cases.append(pack(harmed_m[0], "harmed_by_model_context", "context_distraction"))
    if harmed_n:
        cases.append(pack(harmed_n[0], "harmed_by_neighborhood_context", "context_distraction"))

    noncanonical = [
        r for r in baseline
        if NONCANONICAL_RE.search((r.get("rationale") or "") + (text.get((r["model_id"], r["reaction_id"])) or ""))
    ]
    if noncanonical:
        cases.append(pack(noncanonical[0], "noncanonical_sbml_event", "possible_source_model_curation_or_non_kegg_event"))
    return cases


def recommend(metrics_by_variant: Mapping[str, Mapping[str, Any]], paired: Mapping[str, Any]) -> Dict[str, Any]:
    ranked = sorted(
        metrics_by_variant.items(),
        key=lambda kv: (
            kv[1].get("exact_top1") or 0.0,
            kv[1].get("selective_exact_top1") or 0.0,
            kv[1].get("coverage") or 0.0,
        ),
        reverse=True,
    )
    deltas = []
    for key, block in paired.items():
        ci = block.get("accuracy_delta_cluster_bootstrap") or {}
        deltas.append((key, ci.get("mean"), ci.get("low"), ci.get("high")))
    uncertain = any(
        low is not None and high is not None and low <= 0 <= high
        for _, _, low, high in deltas
    )
    incorrect_rates = [m.get("incorrect_in_catalog_rate") or 0.0 for m in metrics_by_variant.values()]
    high_false_id = max(incorrect_rates) >= 0.4
    return {
        "preferred_context_variant": None if uncertain else ranked[0][0],
        "ranking_by_exact_top1": [name for name, _ in ranked],
        "working_context_for_followup": "target_only",
        "retain_direct_open_set": "selective_method_development_only",
        "use_raw_confidence_threshold": False,
        "paired_deltas_cross_zero": uncertain,
        "high_incorrect_in_catalog_rate": high_false_id,
        "another_validation_experiment_before_test_freeze": True,
        "notes": [
            "Do not freeze a winner from a trivial numerical gap if cluster intervals include zero.",
            "Direct open-set value depends on recovery in retrieval-failure strata, not control-set accuracy alone.",
            "Held-out test evaluation is not part of this decision.",
            "Self-reported confidence is not treated as a calibrated probability.",
            "A non-abstained valid in-catalog miss is an unsupported open-set prediction.",
            "target_only is the working default because added context did not beat uncertainty and costs more.",
        ],
        "paired_accuracy_intervals": deltas,
    }


def build_manifest(out_dir: Path, extra: Sequence[Path]) -> Dict[str, Any]:
    files = []
    for path in extra:
        if not path.exists():
            continue
        files.append({
            "path": str(path.relative_to(REPO_ROOT)) if str(path).startswith(str(REPO_ROOT)) else str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "n_lines": (
                sum(1 for _ in path.open(encoding="utf-8")) if path.suffix in {".jsonl", ".csv"} else None
            ),
        })
    return {"n_files": len(files), "files": files}


def evaluate_validation_dir(
    results_dir: Path = OUT_VALIDATION_DIR,
    *,
    answer_key_path: Path = OUT_PILOT_KEY,
    sample_path: Path = OUT_PILOT,
) -> Dict[str, Any]:
    result_rows = load_jsonl(results_dir / "results.jsonl")
    sample = pd.read_csv(sample_path)
    if "split" in sample.columns:
        splits = set(sample["split"].astype(str))
        if splits != {PILOT_SPLIT}:
            raise ValueError(
                f"evaluation sample must be validation-only; found splits {sorted(splits)}"
            )
    sample_ids = set(sample["sample_id"].astype(str))
    result_ids = {str(r["sample_id"]) for r in result_rows}
    extra = sorted(result_ids - sample_ids)
    if extra:
        raise ValueError(f"result rows are not in the validation sample: {extra[:10]}")
    key = load_answer_key(answer_key_path)
    seen = load_train_target_ids()
    mapping = answer_key_map(key)
    mode_results = [row_to_mode_result(row) for row in result_rows]
    scored = score_results(
        mode_results, mapping, seen_targets=seen, seen_definition="train",
    )
    enriched = _enrich_score_rows(scored, result_rows, key, seen, sample)
    by_variant = {
        variant: [r for r in enriched if r.get("variant") == variant]
        for variant in CONTEXT_VARIANTS
    }
    primary = {variant: variant_metrics(rows) for variant, rows in by_variant.items()}
    recovery = {variant: recovery_block(rows) for variant, rows in by_variant.items()}
    paired = paired_context_comparison(by_variant)
    calibration = {variant: calibration_report(rows) for variant, rows in by_variant.items()}
    seen_unseen = {}
    for variant, rows in by_variant.items():
        seen_unseen[variant] = {
            "seen_in_train": variant_metrics([r for r in rows if r.get("target_seen_in_train")]),
            "unseen_in_train": variant_metrics([r for r in rows if not r.get("target_seen_in_train")]),
        }
    extra = {}
    for variant, rows in by_variant.items():
        extra[variant] = {
            "by_cluster": stratify(rows, "cluster_id"),
            "by_model": stratify(rows, "model_id"),
            "by_multi_target": {
                "single": variant_metrics([r for r in rows if not r.get("multi_target")]),
                "multiple": variant_metrics([r for r in rows if r.get("multi_target")]),
            },
            "by_species_annotation_source": stratify(rows, "species_annotation_source"),
            "by_complexity_bucket": stratify(rows, "complexity_bucket"),
            "by_phase2_status": stratify(rows, "status"),
        }
    cost = cost_report(result_rows)
    n_correct_predictions = sum(1 for r in enriched if r.get("exact_top1"))
    if n_correct_predictions:
        cost["cost_per_correct_prediction_usd"] = round(cost["total_cost_usd"] / n_correct_predictions, 6)
    else:
        cost["cost_per_correct_prediction_usd"] = None
    taxonomy = failure_taxonomy(by_variant)
    recommendation = recommend(primary, paired)
    payload = {
        "n_result_rows": len(result_rows),
        "n_reactions": int(pd.Series([r["sample_id"] for r in result_rows]).nunique()),
        "experimental_unit": "reaction",
        "seen_target_definition": "train",
        "primary_by_variant": primary,
        "recovery_by_variant": recovery,
        "paired_context_comparison": paired,
        "calibration_by_variant": calibration,
        "seen_unseen_by_variant": seen_unseen,
        "additional_stratification": extra,
        "cost": cost,
        "failure_taxonomy": taxonomy,
        "recommendation": recommendation,
        "scored_rows": enriched,
    }
    return payload


def write_eval_artifacts(payload: Mapping[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        {k: payload[k] for k in payload if k != "scored_rows"},
        out_dir / "eval.json",
    )
    atomic_write_json(payload["primary_by_variant"], out_dir / "metrics_by_variant.json")
    atomic_write_json(payload["recovery_by_variant"], out_dir / "stratum_analysis.json")
    atomic_write_json(payload["paired_context_comparison"], out_dir / "context_comparison.json")
    atomic_write_json(payload["calibration_by_variant"], out_dir / "calibration.json")
    atomic_write_json(payload["cost"], out_dir / "cost_report.json")
    atomic_write_json(payload["failure_taxonomy"], out_dir / "failure_taxonomy.json")
    atomic_write_json(payload["recommendation"], out_dir / "recommendation.json")
    atomic_write_json(payload["scored_rows"], out_dir / "scored_rows.json")
    paths = [
        out_dir / name for name in (
            "plan.json", "preflight.json", "run_config.json", "requests.jsonl",
            "results.jsonl", "summary.json", "execute_session.json",
            "cache_verify.json", "eval.json", "metrics_by_variant.json",
            "cache_verify.json", "eval.json", "metrics_by_variant.json",
            "stratum_analysis.json", "context_comparison.json", "calibration.json",
            "cost_report.json", "failure_taxonomy.json", "recommendation.json",
            "scored_rows.json",
        )
    ]
    atomic_write_json(build_manifest(out_dir, paths), out_dir / "artifact_manifest.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=OUT_VALIDATION_DIR)
    parser.add_argument("--answer-key", type=Path, default=OUT_PILOT_KEY)
    parser.add_argument("--sample", type=Path, default=OUT_PILOT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload = evaluate_validation_dir(
        args.results_dir, answer_key_path=args.answer_key, sample_path=args.sample,
    )
    write_eval_artifacts(payload, args.results_dir)
    logger.info(
        "evaluated %d rows / %d reactions; wrote artifacts under %s",
        payload["n_result_rows"], payload["n_reactions"], args.results_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
