"""Offline evaluation for Phase 3 mode results.

Scores mocked or future cached outputs. Never requires a network call. An abstention
is not treated as a hallucinated identifier: it is wrong for full-coverage accuracy
and is handled separately in selective accuracy.

Usage (library)::

    from benchmark.scripts.phase3_eval import score_results
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from benchmark.scripts.phase3_common import (
    ID_ABSENT,
    ID_IN_CATALOG,
    ID_MALFORMED,
    KEGG_ID_STRICT,
    STRATA,
    parse_kegg_ids,
)
from benchmark.scripts.phase3_modes import ModeResult, Prediction, evidence_identifier_list


def _default_equiv(candidate: str, truth_ids: Iterable[str], kind: str) -> bool:
    if kind == "exact":
        return candidate in set(truth_ids)
    from benchmark.scripts.kegg_equivalence import is_equivalent
    return is_equivalent(candidate, truth_ids, kind)


EquivFn = Callable[[str, Iterable[str], str], bool]


def _hit(pred: str, truth: Sequence[str], *, kind: str, equiv: EquivFn) -> bool:
    if kind == "exact":
        return pred in set(truth)
    return bool(equiv(pred, truth, kind))


def _topk_hit(
    preds: Sequence[Prediction], truth: Sequence[str], k: int, *, kind: str, equiv: EquivFn,
) -> bool:
    for pred in preds[:k]:
        if pred.valid_kegg_id and _hit(pred.kegg_id, truth, kind=kind, equiv=equiv):
            return True
    return False


def _reciprocal_rank(
    preds: Sequence[Prediction], truth: Sequence[str], *, kind: str, equiv: EquivFn,
) -> float:
    for i, pred in enumerate(preds, start=1):
        if pred.valid_kegg_id and _hit(pred.kegg_id, truth, kind=kind, equiv=equiv):
            return 1.0 / i
    return 0.0


def _pred_id_class(p: Prediction) -> str:
    if p.in_catalog or p.valid_kegg_id:
        return ID_IN_CATALOG
    if p.id_class == ID_ABSENT or p.well_formed:
        return ID_ABSENT
    if p.kegg_id and KEGG_ID_STRICT.match(p.kegg_id):
        return ID_ABSENT
    return ID_MALFORMED


def _evidence_outcome(result: ModeResult, exact_top1: bool) -> str:
    """Label using support of the evaluated (top-1) prediction only.

    Lower-ranked support must not promote top-1 to evidence-backed.
    """
    has_retrieval = bool(evidence_identifier_list(result.evidence))
    top1_supported = bool(
        result.predictions and result.predictions[0].prediction_supported_by_evidence
    )
    answered = (not result.abstain) and bool(result.predictions)
    if result.abstain and has_retrieval:
        return "abstained_after_retrieval"
    if exact_top1 and top1_supported:
        return "correct_and_evidence_supported"
    if exact_top1 and not top1_supported:
        return "correct_but_unsupported"
    if answered and (not exact_top1) and top1_supported:
        return "incorrect_despite_evidence"
    if answered:
        return "incorrect_unsupported"
    return "unanswered"


def score_one(
    result: ModeResult,
    truth_ids: Sequence[str],
    *,
    equiv: EquivFn = _default_equiv,
) -> Dict[str, Any]:
    preds = result.predictions
    malformed = [p.kegg_id for p in preds if _pred_id_class(p) == ID_MALFORMED]
    absent = [p.kegg_id for p in preds if _pred_id_class(p) == ID_ABSENT]
    in_catalog = [p for p in preds if _pred_id_class(p) == ID_IN_CATALOG]
    invalid = malformed + absent
    answered = (not result.abstain) and bool(preds)
    exact_top1 = False if result.abstain else _topk_hit(
        preds, truth_ids, 1, kind="exact", equiv=equiv)
    row: Dict[str, Any] = {
        "sample_id": result.sample_id,
        "model_id": result.model_id,
        "reaction_id": result.reaction_id,
        "cluster_id": result.cluster_id,
        "stratum": result.stratum,
        "mode": result.mode,
        "variant": result.variant,
        "abstain": result.abstain,
        "answered": answered,
        "n_predictions": len(preds),
        "n_valid_kegg_ids": len(in_catalog),
        "n_invalid_kegg_ids": len(invalid),
        "n_malformed_ids": len(malformed),
        "n_absent_from_catalog_ids": len(absent),
        "n_in_catalog_ids": len(in_catalog),
        "invalid_kegg_ids": invalid,
        "malformed_ids": malformed,
        "absent_from_catalog_ids": absent,
        "evidence_backed": (
            (not result.abstain)
            and bool(preds)
            and bool(preds[0].prediction_supported_by_evidence)
        ),
        "top1_supported_by_evidence": (
            bool(preds) and bool(preds[0].prediction_supported_by_evidence)
        ),
        "prediction_supported_by_evidence": [
            p.prediction_supported_by_evidence for p in preds
        ],
        "supporting_evidence_ids": [list(p.supporting_evidence_ids) for p in preds],
        "parse_error": result.parse_error,
        "exact_top1": exact_top1,
        "exact_top3": False if result.abstain else _topk_hit(
            preds, truth_ids, 3, kind="exact", equiv=equiv),
        "brite_top1": False if result.abstain else _topk_hit(
            preds, truth_ids, 1, kind="brite_orthology", equiv=equiv),
        "brite_top3": False if result.abstain else _topk_hit(
            preds, truth_ids, 3, kind="brite_orthology", equiv=equiv),
        "mrr_exact": 0.0 if result.abstain else _reciprocal_rank(
            preds, truth_ids, kind="exact", equiv=equiv),
        "recall_at_1_exact": False if result.abstain else _topk_hit(
            preds, truth_ids, 1, kind="exact", equiv=equiv),
        "recall_at_3_exact": False if result.abstain else _topk_hit(
            preds, truth_ids, 3, kind="exact", equiv=equiv),
        "recall_at_5_exact": False if result.abstain else _topk_hit(
            preds, truth_ids, 5, kind="exact", equiv=equiv),
        "recall_at_10_exact": False if result.abstain else _topk_hit(
            preds, truth_ids, 10, kind="exact", equiv=equiv),
        "evidence_outcome": _evidence_outcome(result, exact_top1),
    }
    return row


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(sum(values) / len(values)), 4)


def _cluster_macro(rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
    by = defaultdict(list)
    for row in rows:
        by[row["cluster_id"]].append(float(row[field]))
    if not by:
        return None
    return _mean([sum(v) / len(v) for v in by.values()])


def _bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    n_boot: int = 1000,
    seed: int = 20260902,
    cluster: bool = True,
) -> Dict[str, Any]:
    """Cluster-aware bootstrap. Small-n intervals are wide and should not be overread."""
    if not rows:
        return {"mean": None, "low": None, "high": None, "n": 0, "method": "none"}
    rng = random.Random(seed)
    if cluster:
        groups = defaultdict(list)
        for row in rows:
            groups[row["cluster_id"]].append(row)
        keys = list(groups)
        stats = []
        for _ in range(n_boot):
            drawn = []
            for key in (rng.choice(keys) for _ in keys):
                drawn.extend(groups[key])
            stats.append(sum(float(r[field]) for r in drawn) / len(drawn))
        method = "cluster_bootstrap"
    else:
        stats = []
        n = len(rows)
        for _ in range(n_boot):
            drawn = [rows[rng.randrange(n)] for _ in range(n)]
            stats.append(sum(float(r[field]) for r in drawn) / n)
        method = "reaction_bootstrap"
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[min(len(stats) - 1, int(0.975 * n_boot))]
    return {
        "mean": round(sum(float(r[field]) for r in rows) / len(rows), 4),
        "low": round(lo, 4),
        "high": round(hi, 4),
        "n": len(rows),
        "n_boot": n_boot,
        "method": method,
        "limitation": (
            "The pilot is small and clustered. These intervals describe resampling "
            "uncertainty under the observed clusters, not a license to claim significance."
        ),
    }


def _subset_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    n_abstain = sum(1 for r in rows if r["abstain"])
    n_answered = sum(1 for r in rows if r["answered"])
    n_malformed = sum(r["n_malformed_ids"] for r in rows)
    n_absent = sum(r["n_absent_from_catalog_ids"] for r in rows)
    n_in_catalog = sum(r["n_in_catalog_ids"] for r in rows)
    n_id = n_malformed + n_absent + n_in_catalog
    n_evidence = sum(1 for r in rows if r["evidence_backed"])
    answered = [r for r in rows if r["answered"]]
    outcomes = [r.get("evidence_outcome") for r in rows]
    return {
        "n": n,
        "abstention_rate": _mean([1.0 if r["abstain"] else 0.0 for r in rows]),
        "answered_rate": _mean([1.0 if r["answered"] else 0.0 for r in rows]),
        "valid_kegg_id_rate": None if n_id == 0 else round(n_in_catalog / n_id, 4),
        "malformed_id_rate": None if n_id == 0 else round(n_malformed / n_id, 4),
        "absent_from_catalog_id_rate": None if n_id == 0 else round(n_absent / n_id, 4),
        "in_catalog_id_rate": None if n_id == 0 else round(n_in_catalog / n_id, 4),
        "exact_top1": _mean([r["exact_top1"] for r in rows]),
        "exact_top3": _mean([r["exact_top3"] for r in rows]),
        "brite_top1": _mean([r["brite_top1"] for r in rows]),
        "brite_top3": _mean([r["brite_top3"] for r in rows]),
        "selective_exact_top1": _mean([r["exact_top1"] for r in answered]),
        "mrr_exact": _mean([r["mrr_exact"] for r in rows]),
        "recall_at_1_exact": _mean([r["recall_at_1_exact"] for r in rows]),
        "recall_at_3_exact": _mean([r["recall_at_3_exact"] for r in rows]),
        "recall_at_5_exact": _mean([r["recall_at_5_exact"] for r in rows]),
        "recall_at_10_exact": _mean([r["recall_at_10_exact"] for r in rows]),
        "cluster_macro_exact_top1": _cluster_macro(rows, "exact_top1"),
        "evidence_backed_rate": _mean([1.0 if r["evidence_backed"] else 0.0 for r in rows]),
        "evidence_backed_exact_top1": _mean(
            [r["exact_top1"] for r in rows if r["evidence_backed"]]),
        "correct_and_evidence_supported_rate": _mean(
            [1.0 if o == "correct_and_evidence_supported" else 0.0 for o in outcomes]),
        "correct_but_unsupported_rate": _mean(
            [1.0 if o == "correct_but_unsupported" else 0.0 for o in outcomes]),
        "incorrect_despite_evidence_rate": _mean(
            [1.0 if o == "incorrect_despite_evidence" else 0.0 for o in outcomes]),
        "abstained_after_retrieval_rate": _mean(
            [1.0 if o == "abstained_after_retrieval" else 0.0 for o in outcomes]),
        "n_abstain": n_abstain,
        "n_answered": n_answered,
        "n_evidence_backed": n_evidence,
    }


def coverage_curve(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Accuracy versus coverage by dropping the lowest-confidence answered cases.

    Abstentions are already uncovered. Remaining answers are sorted by the first
    prediction's confidence when present.
    """
    answered = [r for r in rows if r.get("answered")]
    def _conf(r):
        return 0.0
    # Confidence lives on ModeResult, not the score row. Callers may pass it through.
    ranked = sorted(answered, key=lambda r: float(r.get("confidence") or 0.0), reverse=True)
    curve = []
    correct = 0
    for i, row in enumerate(ranked, start=1):
        correct += 1 if row["exact_top1"] else 0
        curve.append({
            "n_answered": i,
            "coverage": round(i / len(rows), 4) if rows else 0.0,
            "selective_accuracy": round(correct / i, 4),
        })
    return curve


def score_results(
    results: Sequence[ModeResult],
    answer_key: Mapping[Tuple[str, str], Sequence[str]],
    *,
    seen_targets: Optional[set] = None,
    seen_definition: Optional[str] = None,
    equiv: EquivFn = _default_equiv,
) -> Dict[str, Any]:
    """Score mode results.

    ``seen_targets`` must be the KEGG ids present in the data actually used to
    *fit* the model. Pass ``seen_definition="train"`` for a retriever trained on
    train only, or ``"train+validation"`` if the final retriever is refit after
    method selection. Do not treat a raw id set as "train" without saying so.
    """
    if seen_targets is not None and not seen_definition:
        seen_definition = "caller_supplied"
    rows: List[Dict[str, Any]] = []
    for result in results:
        truth = list(answer_key.get((result.model_id, result.reaction_id), []))
        row = score_one(result, truth, equiv=equiv)
        if seen_targets is not None:
            row["target_seen_in_fit"] = any(t in seen_targets for t in truth)
        if result.predictions:
            row["confidence"] = result.predictions[0].confidence
        rows.append(row)

    overall = _subset_metrics(rows)
    overall["exact_top1_ci"] = _bootstrap_ci(rows, "exact_top1")
    by_stratum = {
        s: _subset_metrics([r for r in rows if r["stratum"] == s]) for s in STRATA
    }
    by_variant: Dict[str, Any] = {}
    for variant in sorted({r["variant"] for r in rows}):
        by_variant[variant] = _subset_metrics([r for r in rows if r["variant"] == variant])
    by_mode: Dict[str, Any] = {}
    for mode in sorted({r["mode"] for r in rows}):
        by_mode[mode] = _subset_metrics([r for r in rows if r["mode"] == mode])

    seen_metrics = None
    unseen_metrics = None
    if seen_targets is not None:
        seen_metrics = _subset_metrics([r for r in rows if r.get("target_seen_in_fit")])
        unseen_metrics = _subset_metrics([r for r in rows if not r.get("target_seen_in_fit")])

    return {
        "n": len(rows),
        "overall": overall,
        "by_stratum": by_stratum,
        "by_variant": by_variant,
        "by_mode": by_mode,
        "seen_target_definition": seen_definition,
        "seen_fit_target": seen_metrics,
        "unseen_fit_target": unseen_metrics,
        "coverage_curve": coverage_curve(rows),
        "rows": rows,
    }
