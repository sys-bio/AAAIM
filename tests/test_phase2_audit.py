"""Tests for the Phase 2 artifact audit, diagnostics and failure reporting.

Two kinds of test live here:

*Synthetic* tests build small artifact sets in a tmp dir and assert that each invariant
actually fires when violated. An invariant that cannot fail is worthless, so every check
is tested against a deliberately corrupted input as well as a clean one.

*Live* tests read the real frozen artifacts when they are present and skip otherwise, so
the suite stays runnable on a fresh clone that has not done the multi-day generation pass.

Run with::

    python -m pytest tests/test_phase2_audit.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts import audit_phase2, candidate_diagnostics
from benchmark.scripts.rank_baselines import _rate, failure_decomposition

DATA_DIR = REPO_ROOT / "benchmark" / "data"
EXPECTED_CONFIG_ID = "86938b48ab88"


# ----------------------------------------------------------------------------------
# Synthetic artifact fixtures


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """A tiny but fully consistent Phase 2 artifact set that passes every invariant.

    M1/R1 is `ok` with two candidates, M1/R2 is `no_candidates`, and M2/R1 is
    `unconstrained_candidate_set`. That covers all three statuses seen in the real run.
    """
    cfg_id = "deadbeefcafe"
    monkeypatch.setattr(audit_phase2, "config_id", lambda: cfg_id)

    reactions = pd.DataFrame({
        "model_id": ["M1", "M1", "M2"],
        "reaction_id": ["R1", "R2", "R1"],
        "included_in_eval": [True, True, True],
        "ground_truth_kegg_all": ["R00001", "R00002", "R00003"],
    })
    status = pd.DataFrame({
        "model_id": ["M1", "M1", "M2"],
        "reaction_id": ["R1", "R2", "R1"],
        "status": ["ok", "no_candidates", "unconstrained_candidate_set"],
        "num_candidates": [2, 0, 0],
        "degenerate_set_size": [0, 0, 0],
        "relaxation_level": [0, 0, 0],
        "relaxation_direction": ["exact", "exact", "exact"],
        "reaction_class": ["metabolic"] * 3,
        "filtered_species_count": [3, 0, 0],
        "config_id": [cfg_id] * 3,
    })
    candidates = pd.DataFrame({
        "model_id": ["M1", "M1"],
        "reaction_id": ["R1", "R1"],
        "candidate_kegg": ["R00001", "R00009"],
        "raw_rank": [1, 2],
        "heuristic_score": [1.0, 0.5],
        "relaxation_level": [0, 0],
        "relaxation_direction": ["exact", "exact"],
        "config_id": [cfg_id] * 2,
    })
    failures = pd.DataFrame(columns=[
        "model_id", "reaction_id", "scope", "failure_type", "message", "traceback_tail"])

    paths = {
        "reactions": tmp_path / "reactions.csv",
        "status": tmp_path / "candidate_status.csv",
        "candidates": tmp_path / "candidates.csv",
        "failures": tmp_path / "candidate_generation_failures.csv",
        "config": tmp_path / "candidate_generation_config.json",
    }
    _write_csv(reactions, paths["reactions"])
    _write_csv(status, paths["status"])
    _write_csv(candidates, paths["candidates"])
    _write_csv(failures, paths["failures"])

    cache_dir = tmp_path / "_candidates_cache"
    cache_dir.mkdir()
    for model_id in ("M1", "M2"):
        (cache_dir / f"{model_id}.json").write_text(
            json.dumps({"model_id": model_id, "config_id": cfg_id}), encoding="utf-8")

    monkeypatch.setattr(audit_phase2, "REACTIONS_CSV", paths["reactions"])
    monkeypatch.setattr(audit_phase2, "STATUS_CSV", paths["status"])
    monkeypatch.setattr(audit_phase2, "CANDIDATES_CSV", paths["candidates"])
    monkeypatch.setattr(audit_phase2, "FAILURES_CSV", paths["failures"])
    monkeypatch.setattr(audit_phase2, "CONFIG_JSON", paths["config"])
    monkeypatch.setattr(audit_phase2, "CACHE_DIR", cache_dir)

    def _refresh_config() -> None:
        """Rewrite the config json so its recorded digests match the files on disk."""
        paths["config"].write_text(json.dumps({
            "config_id": cfg_id,
            "outputs": {
                "candidates_csv_sha256": audit_phase2.sha256_file(paths["candidates"]),
                "candidate_status_csv_sha256": audit_phase2.sha256_file(paths["status"]),
                "candidate_generation_failures_csv_sha256":
                    audit_phase2.sha256_file(paths["failures"]),
            },
        }), encoding="utf-8")

    _refresh_config()
    paths["refresh_config"] = _refresh_config
    paths["config_id"] = cfg_id
    return paths


def _result(summary, name: str) -> dict:
    return next(c for c in summary["checks"] if c["check"] == name)


def _failed_names(summary) -> set:
    return {c["check"] for c in summary["checks"] if not c["passed"]}


# ----------------------------------------------------------------------------------
# The clean fixture must pass everything


def test_consistent_artifacts_pass_every_invariant(artifacts):
    summary = audit_phase2.audit(expect_config_id=artifacts["config_id"])
    assert summary["all_passed"], _failed_names(summary)
    assert summary["checks_failed"] == 0
    assert summary["evaluable_reactions"] == 3
    assert summary["candidate_rows"] == 2


# ----------------------------------------------------------------------------------
# Each invariant must actually fire when violated


def test_missing_cache_is_detected(artifacts):
    (artifacts["config"].parent / "_candidates_cache" / "M2.json").unlink()
    summary = audit_phase2.audit()
    assert not _result(summary, "all_included_models_have_compatible_cache")["passed"]
    assert "M2" in _result(
        summary, "all_included_models_have_compatible_cache")["detail"]["missing"]


def test_stale_cache_config_id_is_detected(artifacts):
    path = artifacts["config"].parent / "_candidates_cache" / "M2.json"
    path.write_text(json.dumps({"model_id": "M2", "config_id": "old000000000"}),
                    encoding="utf-8")
    summary = audit_phase2.audit()
    check = _result(summary, "all_included_models_have_compatible_cache")
    assert not check["passed"]
    assert any("old000000000" in d for d in check["detail"]["incompatible"])


def test_duplicate_status_row_is_detected(artifacts):
    status = pd.read_csv(artifacts["status"])
    _write_csv(pd.concat([status, status.iloc[[0]]]), artifacts["status"])
    artifacts["refresh_config"]()
    summary = audit_phase2.audit()
    assert not _result(
        summary, "exactly_one_status_row_per_evaluable_reaction")["passed"]


def test_missing_status_row_is_detected(artifacts):
    status = pd.read_csv(artifacts["status"])
    _write_csv(status.iloc[1:], artifacts["status"])
    artifacts["refresh_config"]()
    summary = audit_phase2.audit()
    assert not _result(
        summary, "exactly_one_status_row_per_evaluable_reaction")["passed"]
    assert not _result(summary, "status_counts_sum_to_evaluable")["passed"]


def test_ok_reaction_without_candidates_is_detected(artifacts):
    _write_csv(pd.read_csv(artifacts["candidates"]).iloc[0:0], artifacts["candidates"])
    artifacts["refresh_config"]()
    summary = audit_phase2.audit()
    assert not _result(summary, "every_ok_reaction_has_candidates")["passed"]


def test_num_candidates_disagreeing_with_rows_is_detected(artifacts):
    status = pd.read_csv(artifacts["status"])
    status.loc[0, "num_candidates"] = 99
    _write_csv(status, artifacts["status"])
    artifacts["refresh_config"]()
    summary = audit_phase2.audit()
    assert not _result(summary, "num_candidates_matches_stored_rows")["passed"]


def test_unconstrained_reaction_with_candidate_rows_is_detected(artifacts):
    """`unconstrained_candidate_set` means the whole database matched; storing rows for
    it would silently reintroduce the 12,312-candidate explosion."""
    cands = pd.read_csv(artifacts["candidates"])
    rogue = cands.iloc[[0]].copy()
    rogue["model_id"] = "M2"
    rogue["reaction_id"] = "R1"
    _write_csv(pd.concat([cands, rogue], ignore_index=True), artifacts["candidates"])
    artifacts["refresh_config"]()
    summary = audit_phase2.audit()
    assert not _result(summary, "empty_statuses_have_no_candidate_rows")["passed"]


def test_no_candidates_reaction_with_rows_is_detected(artifacts):
    cands = pd.read_csv(artifacts["candidates"])
    rogue = cands.iloc[[0]].copy()
    rogue["reaction_id"] = "R2"
    _write_csv(pd.concat([cands, rogue], ignore_index=True), artifacts["candidates"])
    artifacts["refresh_config"]()
    summary = audit_phase2.audit()
    assert not _result(summary, "empty_statuses_have_no_candidate_rows")["passed"]


@pytest.mark.parametrize("ranks", [[1, 3], [2, 3], [1, 1], [0, 1]])
def test_non_consecutive_ranks_are_detected(artifacts, ranks):
    cands = pd.read_csv(artifacts["candidates"])
    cands["raw_rank"] = ranks
    _write_csv(cands, artifacts["candidates"])
    artifacts["refresh_config"]()
    summary = audit_phase2.audit()
    assert not _result(summary, "candidate_ranks_unique_and_consecutive")["passed"]


def test_duplicate_candidate_within_reaction_is_detected(artifacts):
    cands = pd.read_csv(artifacts["candidates"])
    cands.loc[1, "candidate_kegg"] = cands.loc[0, "candidate_kegg"]
    _write_csv(cands, artifacts["candidates"])
    artifacts["refresh_config"]()
    summary = audit_phase2.audit()
    assert not _result(summary, "candidates_unique_within_reaction")["passed"]


@pytest.mark.parametrize("bad_id", ["R1", "R000001", "C00022", "r00001", "R0000A"])
def test_malformed_kegg_id_is_detected(artifacts, bad_id):
    cands = pd.read_csv(artifacts["candidates"])
    cands.loc[1, "candidate_kegg"] = bad_id
    _write_csv(cands, artifacts["candidates"])
    artifacts["refresh_config"]()
    summary = audit_phase2.audit()
    assert not _result(summary, "candidate_kegg_ids_well_formed")["passed"]


def test_wrong_config_id_in_candidates_is_detected(artifacts):
    cands = pd.read_csv(artifacts["candidates"])
    cands.loc[1, "config_id"] = "otherconfig1"
    _write_csv(cands, artifacts["candidates"])
    artifacts["refresh_config"]()
    summary = audit_phase2.audit()
    assert not _result(summary, "config_id_consistent_across_artifacts")["passed"]


def test_unexpected_config_id_is_detected(artifacts):
    """Auditing against a different expected config_id must fail loudly."""
    summary = audit_phase2.audit(expect_config_id="somethingelse")
    assert not _result(summary, "config_id_consistent_across_artifacts")["passed"]


def test_pipeline_failure_row_is_detected(artifacts):
    _write_csv(pd.DataFrame([{
        "model_id": "M1", "reaction_id": "R1", "scope": "reaction",
        "failure_type": "generation_failed", "message": "boom", "traceback_tail": "",
    }]), artifacts["failures"])
    artifacts["refresh_config"]()
    summary = audit_phase2.audit()
    assert not _result(summary, "no_pipeline_failures")["passed"]
    assert summary["pipeline_failures"] == 1


def test_edited_artifact_breaks_recorded_digest(artifacts):
    """A post-hoc edit must be caught even if the table stays internally consistent."""
    cands = pd.read_csv(artifacts["candidates"])
    _write_csv(cands, artifacts["candidates"])  # config still holds the old digest
    status = pd.read_csv(artifacts["status"])
    status.loc[0, "reaction_class"] = "tampered"
    _write_csv(status, artifacts["status"])
    summary = audit_phase2.audit()
    assert not _result(summary, "recorded_output_digests_match_files")["passed"]


def test_expected_counts_are_checked(artifacts):
    summary = audit_phase2.audit(expect_models=99, expect_reactions=12345)
    assert not _result(summary, "model_count_matches_expected")["passed"]
    assert not _result(summary, "reaction_count_matches_expected")["passed"]


# ----------------------------------------------------------------------------------
# Verification must not alter frozen artifacts


def test_default_audit_does_not_write_report(artifacts, monkeypatch, tmp_path):
    report = tmp_path / "phase2_audit.json"
    monkeypatch.setattr(audit_phase2, "OUT_JSON", report)
    monkeypatch.setattr(sys, "argv", ["audit_phase2.py"])
    assert audit_phase2.main() == 0
    assert not report.exists()


def test_failed_audit_does_not_write_report(artifacts, monkeypatch, tmp_path):
    report = tmp_path / "phase2_audit.json"
    monkeypatch.setattr(audit_phase2, "OUT_JSON", report)
    monkeypatch.setattr(audit_phase2, "CACHE_DIR", tmp_path / "no_caches")
    monkeypatch.setattr(sys, "argv", ["audit_phase2.py"])
    assert audit_phase2.main() == 1
    assert not report.exists()


def test_write_report_is_opt_in(artifacts, monkeypatch, tmp_path):
    report = tmp_path / "phase2_audit.json"
    monkeypatch.setattr(audit_phase2, "OUT_JSON", report)
    monkeypatch.setattr(sys, "argv", ["audit_phase2.py", "--write-report"])
    assert audit_phase2.main() == 0
    assert report.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["all_passed"] is True


def test_report_path_writes_elsewhere_not_frozen_location(artifacts, monkeypatch, tmp_path):
    frozen = tmp_path / "phase2_audit.json"
    custom = tmp_path / "elsewhere.json"
    monkeypatch.setattr(audit_phase2, "OUT_JSON", frozen)
    monkeypatch.setattr(sys, "argv", ["audit_phase2.py", "--report", str(custom)])
    assert audit_phase2.main() == 0
    assert custom.exists()
    assert not frozen.exists()


def test_check_reassembly_without_caches_does_not_wipe_tables(
        artifacts, monkeypatch, tmp_path):
    """A clone with no caches must not replace committed CSVs with empty tables."""
    from benchmark.scripts import generate_candidates as gc

    # Point the generator module at the fixture files so a broken redirect cannot
    # reach the real frozen artifacts.
    monkeypatch.setattr(gc, "CANDIDATES_CSV", artifacts["candidates"])
    monkeypatch.setattr(gc, "STATUS_CSV", artifacts["status"])
    monkeypatch.setattr(gc, "FAILURES_CSV", artifacts["failures"])
    monkeypatch.setattr(gc, "CONFIG_JSON", artifacts["config"])
    monkeypatch.setattr(gc, "CACHE_DIR", tmp_path / "no_caches")
    monkeypatch.setattr(audit_phase2, "CACHE_DIR", tmp_path / "no_caches")

    real_targets = [
        DATA_DIR / "candidates.csv",
        DATA_DIR / "candidate_status.csv",
        DATA_DIR / "candidate_generation_failures.csv",
        DATA_DIR / "candidate_generation_config.json",
        DATA_DIR / "phase2_audit.json",
        REPO_ROOT / "benchmark" / "PHASE2_MANIFEST.json",
    ]
    real_before = {p: p.read_bytes() for p in real_targets if p.exists()}
    fixture_before = {k: artifacts[k].read_bytes()
                      for k in ("candidates", "status", "failures", "config")}

    result = audit_phase2.check_reassembly()
    assert result["identical"] is False
    assert result["committed_artifacts_untouched"] is True
    for key, blob in fixture_before.items():
        assert artifacts[key].read_bytes() == blob
    for path, blob in real_before.items():
        assert path.read_bytes() == blob, f"rewrote {path}"


# ----------------------------------------------------------------------------------
# Failure-rate reporting: denominators must be explicit and distinct


def test_rate_carries_numerator_denominator_and_population():
    r = _rate(3, 12, "widgets")
    assert r == {"rate": 0.25, "pct": 25.0, "numerator": 3, "denominator": 12,
                 "population": "widgets"}


def test_rate_handles_zero_denominator():
    assert _rate(0, 0, "empty")["rate"] is None


def _decomposition_fixture():
    """10 nonempty sets out of 40 evaluable; 6 retrievable; ranker gets 4 first."""
    return pd.DataFrame({
        "hit_any_exact": [True] * 6 + [False] * 4,
        "hit_at_1_exact": [True] * 4 + [False] * 6,
    })


def test_failure_decomposition_uses_distinct_denominators():
    d = failure_decomposition(_decomposition_fixture(), n_evaluable=40, n_zero_candidate=30)

    # Conditional on a nonempty candidate set.
    assert d["conditional_retrieval_failure_rate_nonempty"]["numerator"] == 4
    assert d["conditional_retrieval_failure_rate_nonempty"]["denominator"] == 10
    assert d["conditional_retrieval_failure_rate_nonempty"]["rate"] == 0.4

    # Over the whole corpus: the 30 zero-candidate reactions are charged as failures.
    assert d["overall_retrieval_failure_rate"]["numerator"] == 34
    assert d["overall_retrieval_failure_rate"]["denominator"] == 40
    assert d["overall_retrieval_failure_rate"]["rate"] == 0.85

    # Reranking is conditional on the answer being present at all.
    assert d["conditional_reranking_failure_rate_retrievable"]["numerator"] == 2
    assert d["conditional_reranking_failure_rate_retrievable"]["denominator"] == 6

    assert d["zero_candidate_rate"]["numerator"] == 30
    assert d["zero_candidate_rate"]["denominator"] == 40

    assert d["overall_top1_accuracy"] == {
        "rate": 0.1, "pct": 10.0, "numerator": 4, "denominator": 40,
        "population": "all evaluable reactions"}
    assert d["conditional_top1_accuracy_nonempty"]["denominator"] == 10


def test_overall_and_conditional_retrieval_rates_are_not_interchangeable():
    """The bug being pinned: reporting one number where the other was meant."""
    d = failure_decomposition(_decomposition_fixture(), n_evaluable=40, n_zero_candidate=30)
    assert (d["overall_retrieval_failure_rate"]["rate"]
            != d["conditional_retrieval_failure_rate_nonempty"]["rate"])
    assert (d["overall_retrieval_failure_rate"]["denominator"]
            > d["conditional_retrieval_failure_rate_nonempty"]["denominator"])
    # Overall failure can never be lower than the conditional rate.
    assert (d["overall_retrieval_failure_rate"]["rate"]
            >= d["conditional_retrieval_failure_rate_nonempty"]["rate"])


def test_every_reported_rate_names_its_population():
    d = failure_decomposition(_decomposition_fixture(), n_evaluable=40, n_zero_candidate=30)
    for name, value in d.items():
        assert value["population"], f"{name} does not name its population"
        assert value["denominator"] > 0, f"{name} has no denominator"


def test_no_zero_candidate_reactions_collapses_the_two_rates():
    """With full candidate coverage the conditional and overall rates coincide."""
    d = failure_decomposition(_decomposition_fixture(), n_evaluable=10, n_zero_candidate=0)
    assert (d["overall_retrieval_failure_rate"]["rate"]
            == d["conditional_retrieval_failure_rate_nonempty"]["rate"])


def test_stratification_columns_are_unambiguously_named():
    """No bare `retrieval_failure_pct`: it silently read as a corpus-wide rate."""
    from benchmark.scripts.rank_baselines import failure_stratification

    rankings = pd.DataFrame({
        "ranker": ["heuristic"] * 4,
        "model_id": ["M1"] * 4,
        "reaction_id": ["R1", "R2", "R3", "R4"],
        "candidate_set_size": [1, 2, 3, 4],
        "is_genome_scale": [True] * 4,
        "species_annotation_source": ["chebi"] * 4,
        "any_missing_annotation": [False] * 4,
        "relaxation_required": [False] * 4,
        "complexity_bucket": ["small"] * 4,
        "hit_any_exact": [True, True, True, False],
        "hit_at_1_exact": [True, False, True, False],
        "hit_at_1_brite_orthology": [True, True, True, False],
        "retrieval_failure": [False, False, False, True],
        "reranking_failure": [False, True, False, False],
    })
    out = failure_stratification(rankings)
    assert "retrieval_failure_pct" not in out.columns
    assert "reranking_failure_pct" not in out.columns
    assert "conditional_retrieval_failure_pct_nonempty" in out.columns
    assert "conditional_reranking_failure_pct_retrievable" in out.columns
    assert "n_reactions_nonempty" in out.columns

    row = out[(out.stratum == "is_genome_scale")].iloc[0]
    assert row.n_reactions_nonempty == 4
    assert row.n_retrievable == 3
    # 1 of 3 retrievable was misranked, versus 1 of 4 nonempty sets.
    assert row.conditional_reranking_failure_pct_retrievable == pytest.approx(33.33, abs=0.01)
    assert row.conditional_reranking_failure_pct_nonempty == pytest.approx(25.0)


# ----------------------------------------------------------------------------------
# Diagnostics


def test_size_stats_report_percentiles_and_thresholds():
    stats = candidate_diagnostics._stats(pd.Series([0, 0, 1, 1, 20, 200, 2000]))
    assert stats["n"] == 7
    assert stats["median"] == 1
    assert stats["max"] == 2000
    assert stats["total_rows"] == 2222
    for p in (50, 75, 90, 95, 99):
        assert f"p{p}" in stats
    assert stats["count_gt_15"] == 3
    assert stats["count_gt_100"] == 2
    assert stats["count_gt_1000"] == 1
    assert stats["count_gt_10000"] == 0


def test_size_stats_on_empty_series():
    assert candidate_diagnostics._stats(pd.Series([], dtype=float)) == {"n": 0}


# ----------------------------------------------------------------------------------
# Live artifacts (skipped when the generation pass has not been run)

_LIVE_FILES = ["candidates.csv", "candidate_status.csv",
               "candidate_generation_failures.csv", "candidate_generation_config.json"]
live_only = pytest.mark.skipif(
    not all((DATA_DIR / f).exists() for f in _LIVE_FILES),
    reason="frozen Phase 2 artifacts not present; run generate_candidates.py first",
)

_CACHE_ZIP = REPO_ROOT / "benchmark" / "dist" / "aaaim-benchmark-phase2-v1-candidate-caches.zip"


def _committed_phase2_paths() -> list:
    from benchmark.scripts import freeze_phase2

    paths = [DATA_DIR / spec["name"] for spec in freeze_phase2.ARTIFACTS]
    paths.extend([
        REPO_ROOT / "benchmark" / "PHASE2_MANIFEST.json",
        REPO_ROOT / "benchmark" / "PHASE2_RESULTS.md",
        freeze_phase2.CACHE_REGISTRY,
    ])
    return [p for p in paths if p.exists()]


def _digests(paths) -> dict:
    return {str(p): audit_phase2.sha256_file(p) for p in paths}


@live_only
def test_successful_verification_leaves_committed_artifacts_byte_identical(monkeypatch):
    before = _digests(_committed_phase2_paths())
    monkeypatch.setattr(sys, "argv", [
        "audit_phase2.py",
        "--expect-config-id", EXPECTED_CONFIG_ID,
        "--expect-models", "74",
        "--expect-reactions", "5816",
        "--check-reassembly",
    ])
    assert audit_phase2.main() == 0
    assert _digests(_committed_phase2_paths()) == before


@live_only
def test_failed_verification_leaves_committed_artifacts_byte_identical(
        monkeypatch, tmp_path):
    before = _digests(_committed_phase2_paths())
    monkeypatch.setattr(audit_phase2, "CACHE_DIR", tmp_path / "no_caches")
    monkeypatch.setattr(sys, "argv", ["audit_phase2.py"])
    assert audit_phase2.main() != 0
    assert _digests(_committed_phase2_paths()) == before


@live_only
def test_freeze_verify_leaves_committed_artifacts_byte_identical(monkeypatch):
    from benchmark.scripts import freeze_phase2

    before = _digests(_committed_phase2_paths())
    monkeypatch.setattr(sys, "argv", ["freeze_phase2.py", "--verify"])
    freeze_phase2.main()
    assert _digests(_committed_phase2_paths()) == before


@live_only
def test_live_artifacts_pass_every_invariant():
    summary = audit_phase2.audit(
        expect_config_id=EXPECTED_CONFIG_ID, expect_models=74, expect_reactions=5816)
    assert summary["all_passed"], [c for c in summary["checks"] if not c["passed"]]


@live_only
def test_live_status_counts_are_frozen():
    summary = audit_phase2.audit()
    assert summary["status_counts"] == {
        "no_candidates": 2646, "ok": 2359, "unconstrained_candidate_set": 811}
    assert sum(summary["status_counts"].values()) == 5816
    assert summary["candidate_rows"] == 91802


@live_only
def test_live_zero_candidate_rate_is_reported_over_all_evaluable():
    status = pd.read_csv(DATA_DIR / "candidate_status.csv")
    zero = int((status.num_candidates == 0).sum())
    assert zero == 3457
    assert round(zero / len(status), 4) == 0.5944


@live_only
@pytest.mark.skipif(not (DATA_DIR / "baseline_summary.json").exists(),
                    reason="baselines not run")
def test_live_baseline_summary_separates_the_two_retrieval_rates():
    s = json.loads((DATA_DIR / "baseline_summary.json").read_text(encoding="utf-8"))
    d = s["headline"]["heuristic"]["failure_decomposition"]
    assert d["conditional_retrieval_failure_rate_nonempty"]["denominator"] == 2359
    assert d["overall_retrieval_failure_rate"]["denominator"] == 5816
    assert d["conditional_retrieval_failure_rate_nonempty"]["pct"] == 14.41
    assert d["overall_retrieval_failure_rate"]["pct"] == 65.29
    assert d["overall_top1_accuracy"]["pct"] == 33.25
    assert d["conditional_top1_accuracy_nonempty"]["pct"] == 81.98
    # The headline must not carry an unqualified retrieval/reranking failure figure.
    assert "retrieval_failure_pct" not in s["headline"]["heuristic"]
    assert "reranking_failure_pct" not in s["headline"]["heuristic"]


@live_only
@pytest.mark.skipif(not (DATA_DIR / "baseline_summary.json").exists(),
                    reason="baselines not run")
def test_live_perfect_reranking_gain_is_small():
    """The Phase 3 motivation: reranking the existing sets cannot buy much."""
    s = json.loads((DATA_DIR / "baseline_summary.json").read_text(encoding="utf-8"))
    heur = s["headline"]["heuristic"]["failure_decomposition"]["overall_top1_accuracy"]
    oracle = s["headline"]["oracle"]["failure_decomposition"]["overall_top1_accuracy"]
    gain_pp = oracle["pct"] - heur["pct"]
    assert 0 < gain_pp < 2.0, f"expected ~1.5pp headroom, got {gain_pp}"


@live_only
@pytest.mark.skipif(not (DATA_DIR / "candidate_diagnostics.json").exists(),
                    reason="diagnostics not run")
def test_live_candidate_rows_are_unique_and_concentrated():
    d = json.loads((DATA_DIR / "candidate_diagnostics.json").read_text(encoding="utf-8"))
    assert d["totals"]["duplicate_candidate_rows"] == 0
    top1 = d["model_concentration"]["top_1_models"]
    assert top1["models"] == ["BIOMD0000001063"]
    assert top1["candidate_rows"] == 77073
    # Weakly-constrained sets dominate storage while almost never being retrievable.
    weak = d["degeneracy"]["weakly_constrained"]
    assert weak["share_of_all_rows"] > 0.9
    assert weak["pct_retrievable_exact"] < 20.0


@live_only
def test_live_manifest_is_deterministic_and_verifies():
    """The manifest must be byte-reproducible and must catch a tampered artifact."""
    from benchmark.scripts import freeze_phase2

    first = freeze_phase2.build_manifest()
    second = freeze_phase2.build_manifest()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    assert first["config_id"] == EXPECTED_CONFIG_ID
    assert first["proposed_tag"] == "benchmark-phase2-v1"
    assert first["commits"]["candidate_generation"].startswith("dbf15d6")
    assert first["commits"]["analysis_and_artifacts"].startswith("b5065e0")
    assert first["commits"]["release_snapshot"] == "benchmark-phase2-v1"
    assert "source_commit" not in first
    assert first["counts"]["pipeline_failures"] == 0
    assert first["counts"]["candidate_rows"] == 91802
    assert first["audit"]["all_passed"] is True
    assert not first["artifacts_outstanding"]
    assert not freeze_phase2.verify(first)

    # A digest that no longer matches the file must be reported.
    tampered = json.loads(json.dumps(first))
    tampered["artifacts"][0]["sha256"] = "0" * 64
    assert freeze_phase2.verify(tampered)


@live_only
def test_live_manifest_records_embedding_baseline_as_outstanding():
    """Whether the optional baseline ran must be explicit, not inferred from silence."""
    from benchmark.scripts import freeze_phase2

    emb = freeze_phase2.build_manifest()["embedding_baseline"]
    assert emb["included"] is False
    assert emb["blocks_freeze"] is False
    assert emb["missing"], "must say what is missing"
    if not emb["model_asset_cached"]:
        assert any("model asset" in m for m in emb["missing"])


@live_only
def test_live_phase1_inputs_unchanged_by_phase2():
    """Phase 2 must not rewrite the frozen Phase 1 tables it reads."""
    version = json.loads((DATA_DIR / "VERSION.json").read_text(encoding="utf-8"))
    recorded = version["artifact_sha256"]
    assert version["benchmark_version"] == "phase1-v1"

    checked, mismatches = [], []
    for name, expected in recorded.items():
        path = DATA_DIR / name
        assert path.exists(), f"Phase 1 artifact {name} is missing"
        actual = audit_phase2.sha256_file(path)
        checked.append(name)
        if actual != expected:
            mismatches.append((name, expected, actual))

    assert not mismatches, mismatches
    # Guard against the check passing because it verified nothing.
    assert "reactions.csv" in checked
    assert len(checked) == len(recorded) >= 10


@live_only
@pytest.mark.skipif(not _CACHE_ZIP.exists(), reason="cache zip not packed yet")
def test_live_cache_archive_restores_and_reassembles():
    """Restoring the release zip must reproduce the committed aggregate tables."""
    from benchmark.scripts import freeze_phase2

    registry_path = freeze_phase2.CACHE_REGISTRY
    assert registry_path.exists(), "cache registry must be committed with the zip notes"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["n_files"] == 74
    assert registry["config_id"] == EXPECTED_CONFIG_ID

    before = _digests(_committed_phase2_paths())
    result = freeze_phase2.restore_and_reassemble(_CACHE_ZIP, registry)
    assert not result["registry_problems"], result["registry_problems"]
    assert result["reassembly_identical"]
    assert result["committed_artifacts_untouched"]
    assert _digests(_committed_phase2_paths()) == before
