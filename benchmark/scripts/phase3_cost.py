"""Offline token and cost estimation for the Phase 3 open-set pilot.

No API calls. Pricing is loaded from a user-supplied file; the committed example
file is labelled EXAMPLE ONLY and is not a quote.

A live run is blocked unless the sample is frozen, leakage tests have passed, this
estimator has been printed, caching is in place, and the user has approved the
provider, model, sample size, and budget.

Usage::

    python benchmark/scripts/phase3_cost.py
    python benchmark/scripts/phase3_cost.py --pricing benchmark/phase3/pricing.example.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.scripts.phase3_common import (
    OUT_COST,
    PRICING_EXAMPLE,
    STRATA,
    TOKENIZER_SCAFFOLD,
    estimate_tokens,
    load_evaluable_corpus,
    write_json,
)
from benchmark.scripts.phase3_prompts import (
    CONTEXT_VARIANTS,
    DEFAULT_NEIGHBORHOOD,
    build_pilot_prompts,
)

logger = logging.getLogger("phase3_cost")

DEFAULT_MAX_OUTPUT = 400


def load_pricing(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("source"):
        raise ValueError("pricing file must record a source")
    if not payload.get("pricing_date"):
        raise ValueError("pricing file must record a pricing_date")
    return payload


def _usd(n_tokens: int, per_million: float) -> float:
    return round((n_tokens / 1_000_000.0) * per_million, 6)


def estimate_cost(
    prompts: Sequence[Mapping[str, Any]],
    pricing: Mapping[str, Any],
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT,
) -> Dict[str, Any]:
    by_variant: Dict[str, List[int]] = {v: [] for v in CONTEXT_VARIANTS}
    by_stratum: Dict[str, List[int]] = {s: [] for s in STRATA}
    for row in prompts:
        n = int(row["n_input_tokens_est"])
        by_variant.setdefault(row["variant"], []).append(n)
        by_stratum.setdefault(row["stratum"], []).append(n)

    n_calls = len(prompts)
    n_input = int(sum(int(r["n_input_tokens_est"]) for r in prompts))
    n_output_planned = n_calls * max_output_tokens

    models_out = {}
    for name, rates in (pricing.get("models") or {}).items():
        in_rate = float(rates["input_per_million"])
        out_rate = float(rates["output_per_million"])
        expected = _usd(n_input, in_rate) + _usd(n_output_planned, out_rate)
        # Worst case: every call uses the planned max output tokens (already) and
        # input is as estimated. Double output as a buffer for retries.
        worst = _usd(n_input, in_rate) + _usd(2 * n_output_planned, out_rate)
        models_out[name] = {
            "input_per_million": in_rate,
            "output_per_million": out_rate,
            "expected_usd": round(expected, 4),
            "worst_case_usd": round(worst, 4),
        }

    def _dist(values: List[int]) -> Dict[str, Any]:
        if not values:
            return {"n": 0}
        s = sorted(values)
        return {
            "n": len(s),
            "mean": round(sum(s) / len(s), 1),
            "min": s[0],
            "p50": s[len(s) // 2],
            "p90": s[int(0.9 * (len(s) - 1))],
            "max": s[-1],
            "total": int(sum(s)),
        }

    return {
        "pricing_date": pricing.get("pricing_date"),
        "pricing_source": pricing.get("source"),
        "currency": pricing.get("currency", "USD"),
        "n_calls": n_calls,
        "n_reactions": len({(r["model_id"], r["reaction_id"]) for r in prompts}),
        "n_variants": len({r["variant"] for r in prompts}),
        "n_input_tokens_est": n_input,
        "planned_max_output_tokens_per_call": max_output_tokens,
        "planned_max_output_tokens_total": n_output_planned,
        "models": models_out,
        "by_variant": {k: _dist(v) for k, v in by_variant.items()},
        "by_stratum": {k: _dist(v) for k, v in by_stratum.items()},
        "gate": {
            "sample_frozen": True,
            "leakage_tests_required": True,
            "cache_required": True,
            "explicit_approval_required": [
                "provider", "model", "sample_size", "budget",
            ],
            "tokenizer": {
                "method": TOKENIZER_SCAFFOLD,
                "live_run_blocked_with_this_method": True,
                "required_before_live": (
                    "chosen model tokenizer or a conservative provider-specific bound"
                ),
            },
            "live_calls_blocked_until_approval": True,
        },
    }


def whole_model_counterfactual(corpus, sample_keys) -> Dict[str, Any]:
    """Tokens if every prompt included the full model's reaction list.

    This is the quadratic anti-pattern Phase 3 is designed to avoid: repeating the
    entire reaction chain once per target.
    """
    by_model = {
        mid: sub.query_text.fillna("").astype(str).tolist()
        for mid, sub in corpus.groupby("model_id")
    }
    n_calls = 0
    n_tokens = 0
    for model_id, reaction_id in sample_keys:
        texts = by_model.get(model_id, [])
        n_calls += 1
        n_tokens += sum(estimate_tokens(t) for t in texts)
    return {
        "n_calls": n_calls,
        "n_input_tokens_est": int(n_tokens),
        "note": "Every target would receive every reaction string from its model. "
                "Cost grows with (reactions in model) per call, i.e. roughly quadratically "
                "in model size across a full evaluation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing", type=Path, default=PRICING_EXAMPLE)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    prompts = build_pilot_prompts(neighborhood_k=DEFAULT_NEIGHBORHOOD)
    pricing = load_pricing(args.pricing)
    estimate = estimate_cost(prompts, pricing, max_output_tokens=args.max_output_tokens)

    corpus = load_evaluable_corpus()
    keys = sorted({(r["model_id"], r["reaction_id"]) for r in prompts})
    counterfactual = whole_model_counterfactual(corpus, keys)
    estimate["whole_model_context_counterfactual"] = counterfactual
    bounded_total = estimate["n_input_tokens_est"]
    whole_total = counterfactual["n_input_tokens_est"]
    estimate["bounded_vs_whole_model"] = {
        "bounded_input_tokens": bounded_total,
        "whole_model_input_tokens": whole_total,
        "ratio_whole_over_bounded": (
            round(whole_total / bounded_total, 2) if bounded_total else None
        ),
        "note": "Bounded variants do not repeat the full reaction chain for each target.",
    }
    write_json(estimate, OUT_COST)
    logger.info("calls=%d input_tokens=%d whole_model_tokens=%d",
                estimate["n_calls"], bounded_total, whole_total)
    for name, block in estimate["models"].items():
        logger.info("example %s: expected $%.4f worst $%.4f (%s)",
                    name, block["expected_usd"], block["worst_case_usd"],
                    estimate["pricing_source"])
    logger.info("wrote %s", OUT_COST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
