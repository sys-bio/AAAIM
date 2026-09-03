"""Tests for Phase 3A validation offline evaluation helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts.phase3_common import RETRIEVAL_FAILURE_STRATA, STRATUM_EMPTY, STRATUM_TOP1
from benchmark.scripts.phase3_openai_eval import recovery_block, variant_metrics


def _row(**overrides):
    base = {
        "sample_id": "P1",
        "model_id": "M1",
        "reaction_id": "rxn",
        "cluster_id": "C1",
        "stratum": STRATUM_EMPTY,
        "variant": "target_only",
        "abstain": False,
        "answered": True,
        "exact_top1": False,
        "exact_top3": False,
        "brite_top1": False,
        "brite_top3": False,
        "parse_error": None,
        "terminal_status": "succeeded",
        "n_malformed_ids": 0,
        "n_absent_from_catalog_ids": 0,
        "n_in_catalog_ids": 1,
        "open_set_outcome": "incorrect_in_catalog",
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


def test_recovery_rate_excludes_abstention_and_uses_explicit_denominator():
    rows = [
        _row(sample_id="a", exact_top1=True, open_set_outcome="correct_top1"),
        _row(sample_id="b", abstain=True, answered=False, exact_top1=False,
             n_in_catalog_ids=0, open_set_outcome="abstain"),
        _row(sample_id="c", exact_top1=False),
        _row(sample_id="d", stratum=STRATUM_TOP1, exact_top1=True,
             open_set_outcome="correct_top1"),
    ]
    rec = recovery_block(rows)
    assert rec["all_retrieval_failure"]["numerator"] == 1
    assert rec["all_retrieval_failure"]["denominator"] == 3
    assert rec["all_retrieval_failure"]["recovery_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert rec["empty_constrained"]["numerator"] == 1
    metrics = variant_metrics(rows)
    assert metrics["n_incorrect_in_catalog"] == 1
    assert metrics["n_reactions"] == 4
    assert set(RETRIEVAL_FAILURE_STRATA)
