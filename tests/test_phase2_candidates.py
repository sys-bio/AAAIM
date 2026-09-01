"""Regression tests for the Phase 2 candidate-generation and evaluation pipeline.

These are deliberately cheap (synthetic fixtures, no full model runs) because the real
generation pass takes days. Each test pins a bug that was found while building Phase 2
and that would otherwise corrupt a long run silently.

Run with::

    python -m pytest tests/test_phase2_candidates.py -q
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.scripts import kegg_equivalence
from benchmark.scripts.analyze_retrieval import three_way
from utils.constants import KEGG_COMPOUND_URI_PATTERNS, KEGG_REACTION_URI_PATTERNS


# ----------------------------------------------------------------------------------
# KEGG URI forms


@pytest.mark.parametrize("uri,expected", [
    ("https://identifiers.org/kegg.compound/C00022", "C00022"),
    ("http://identifiers.org/kegg.compound/C00022", "C00022"),
    ("https://identifiers.org/kegg.compound:C00022", "C00022"),
    ("urn:miriam:kegg.compound:C00022", "C00022"),
])
def test_kegg_compound_uri_forms(uri, expected):
    """BioModels uses the slash form; matching only the colon form hid 1,451 reactions."""
    found = [m for p in KEGG_COMPOUND_URI_PATTERNS for m in re.findall(p, uri)]
    assert expected in found, f"{uri} did not yield {expected}"


@pytest.mark.parametrize("uri,expected", [
    ("https://identifiers.org/kegg.reaction/R00024", "R00024"),
    ("https://identifiers.org/kegg.reaction:R00024", "R00024"),
    ("urn:miriam:kegg.reaction:R00024", "R00024"),
])
def test_kegg_reaction_uri_forms(uri, expected):
    found = [m for p in KEGG_REACTION_URI_PATTERNS for m in re.findall(p, uri)]
    assert expected in found


def test_compound_pattern_does_not_match_reaction_ids():
    for pattern in KEGG_COMPOUND_URI_PATTERNS:
        assert not re.findall(pattern, "https://identifiers.org/kegg.reaction/R00024")


# ----------------------------------------------------------------------------------
# Reaction id alignment
#
# `map_reactions_to_kegg` labels the i-th reaction string with reaction_ids[i]. The
# reaction list is filtered to reactions mentioning a mapped species, so ids must come
# from the same filtered pass, otherwise every candidate after the first omission is
# attributed to the wrong reaction.

SBML_THREE_REACTIONS = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
      <model id="align_test">
        <listOfCompartments>
          <compartment id="c" size="1"/>
        </listOfCompartments>
        <listOfSpecies>
          <species id="A" compartment="c" initialConcentration="1"/>
          <species id="B" compartment="c" initialConcentration="1"/>
          <species id="X" compartment="c" initialConcentration="1"/>
          <species id="Y" compartment="c" initialConcentration="1"/>
          <species id="P" compartment="c" initialConcentration="1"/>
          <species id="Q" compartment="c" initialConcentration="1"/>
        </listOfSpecies>
        <listOfReactions>
          <reaction id="RXN_FIRST" reversible="false">
            <listOfReactants><speciesReference species="A"/></listOfReactants>
            <listOfProducts><speciesReference species="B"/></listOfProducts>
          </reaction>
          <reaction id="RXN_UNMAPPED" reversible="false">
            <listOfReactants><speciesReference species="X"/></listOfReactants>
            <listOfProducts><speciesReference species="Y"/></listOfProducts>
          </reaction>
          <reaction id="RXN_LAST" reversible="false">
            <listOfReactants><speciesReference species="P"/></listOfReactants>
            <listOfProducts><speciesReference species="Q"/></listOfProducts>
          </reaction>
        </listOfReactions>
      </model>
    </sbml>
    """)


@pytest.fixture(scope="module")
def align_model(tmp_path_factory):
    path = tmp_path_factory.mktemp("align") / "align_test.xml"
    path.write_text(SBML_THREE_REACTIONS, encoding="utf-8")
    return path


def test_reaction_ids_are_aligned_when_reactions_are_filtered(align_model):
    from core.model_info import extract_reactions_with_ids_from_sbml

    # Only the first and last reactions mention a mapped species.
    ids, reactions, _ = extract_reactions_with_ids_from_sbml(
        str(align_model), ["A", "B", "P", "Q"]
    )
    assert len(ids) == len(reactions)
    assert ids == ["RXN_FIRST", "RXN_LAST"], ids
    # The surviving strings must describe the reactions their ids name.
    assert "A" in reactions[0] and "B" in reactions[0]
    assert "P" in reactions[1] and "Q" in reactions[1]


def test_positional_id_lookup_would_have_mislabelled(align_model):
    """Documents the bug: the unfiltered id list is not positionally usable."""
    from core.model_info import extract_reactions_with_ids_from_sbml, get_all_reaction_ids

    all_ids = [str(i) for i in get_all_reaction_ids(str(align_model))]
    kept_ids, kept_reactions, _ = extract_reactions_with_ids_from_sbml(
        str(align_model), ["A", "B", "P", "Q"]
    )
    assert len(kept_reactions) < len(all_ids)
    # Naive positional labelling maps the second kept reaction to RXN_UNMAPPED.
    naive = [all_ids[i] for i in range(len(kept_reactions))]
    assert naive != kept_ids
    assert naive[1] == "RXN_UNMAPPED" and kept_ids[1] == "RXN_LAST"


def test_backward_compatible_wrapper_matches(align_model):
    from core.model_info import (
        extract_reactions_from_sbml,
        extract_reactions_with_ids_from_sbml,
    )

    reactions, related = extract_reactions_from_sbml(str(align_model), ["A", "B"])
    _, reactions2, related2 = extract_reactions_with_ids_from_sbml(str(align_model), ["A", "B"])
    assert reactions == reactions2
    assert related == related2


# ----------------------------------------------------------------------------------
# Strict error propagation: an exception must never become an empty candidate list


def test_strict_errors_propagates_instead_of_returning_empty():
    from core.database_search import _get_kegg_recommendations_rulebased

    # A malformed normalized reaction record triggers an internal failure.
    broken = [{"id": "R1", "reaction_string": None, "substrates": None, "products": None}]

    with pytest.raises(Exception):
        _get_kegg_recommendations_rulebased(broken, evaluate_candidates=True, strict_errors=True)


def test_default_behaviour_still_swallows_errors():
    """The permissive default is preserved for existing callers."""
    from core.database_search import _get_kegg_recommendations_rulebased

    broken = [{"id": "R1", "reaction_string": None, "substrates": None, "products": None}]
    assert _get_kegg_recommendations_rulebased(broken, evaluate_candidates=True) == []


# ----------------------------------------------------------------------------------
# Unconstrained candidate sets
#
# filter_kegg_reactions tests `model_keys.issubset(kegg_keys)`, and the empty set is a
# subset of everything, so a reaction with no mapped participants "matches" all of KEGG.


def test_empty_constraints_match_everything_upstream():
    from core.database_search import filter_kegg_reactions

    everything = filter_kegg_reactions(set(), set())
    constrained = filter_kegg_reactions({"C00002"}, {"C00008"})
    assert len(everything) > len(constrained)
    # Confirms the degenerate case really is "the whole database".
    assert len(everything) > 10000


def test_unconstrained_reactions_are_recorded_not_stored():
    """A zero-constraint reaction is a retrieval failure, not a candidate set."""
    from benchmark.scripts.generate_candidates import (
        STATUS_UNCONSTRAINED, _relaxation_summary, _status_row,
    )

    metadata = {"reaction_class": "internal", "filtered_species_count": 0}
    level, direction = _relaxation_summary(metadata)
    row = _status_row(
        "M", "R1", STATUS_UNCONSTRAINED, "cfg",
        relaxation_level=level, relaxation_direction=direction,
        metadata=metadata, degenerate_set_size=12312,
    )
    assert row["status"] == STATUS_UNCONSTRAINED
    assert row["num_candidates"] == 0
    assert row["degenerate_set_size"] == 12312
    assert row["filtered_species_count"] == 0


# ----------------------------------------------------------------------------------
# Deterministic ranking


def test_rank_tie_break_is_deterministic():
    """Ties are broken by KEGG id so ranks do not depend on set iteration order."""
    scores = {"R00300": 0.5, "R00100": 0.5, "R00200": 0.9, "R00050": None}
    ordered = sorted(
        scores.items(), key=lambda kv: (-(kv[1] if kv[1] is not None else -1.0), kv[0])
    )
    assert [c for c, _ in ordered] == ["R00200", "R00100", "R00300", "R00050"]

    shuffled = {k: scores[k] for k in ["R00050", "R00200", "R00300", "R00100"]}
    ordered2 = sorted(
        shuffled.items(), key=lambda kv: (-(kv[1] if kv[1] is not None else -1.0), kv[0])
    )
    assert [c for c, _ in ordered2] == [c for c, _ in ordered]


def test_relaxation_summary_reports_max_distance():
    from benchmark.scripts.generate_candidates import _relaxation_summary

    assert _relaxation_summary(None) == (0, "exact")
    assert _relaxation_summary({"participant_relaxation": []}) == (0, "exact")

    meta = {"participant_relaxation": [
        {"distance": 0, "direction": "exact"},
        {"distance": 2, "direction": "up"},
        {"distance": 1, "direction": "down"},
    ]}
    level, direction = _relaxation_summary(meta)
    assert level == 2
    assert direction == "down+up"


# ----------------------------------------------------------------------------------
# Multiple ground-truth ids


def test_multiple_ground_truth_ids_all_count_as_hits():
    truth = {"R00024", "R00025"}
    assert kegg_equivalence.matches_exact("R00025", truth)
    assert kegg_equivalence.matches_exact("R00024", truth)
    assert not kegg_equivalence.matches_exact("R09999", truth)


# ----------------------------------------------------------------------------------
# Equivalence-aware matching


def test_equivalence_groups_are_parsed():
    idx = kegg_equivalence.equivalence_index()
    entry = idx.get("R00024")
    assert entry is not None
    assert "EC:4.1.1.39" in entry["ec"]
    assert "KO:K01601" in entry["ko"]
    assert entry["brite_orthology"] >= entry["ec"] | entry["ko"]


def test_exact_match_is_always_equivalent():
    verdict = kegg_equivalence.match_kinds("R00024", {"R00024"})
    assert verdict["exact"]
    assert all(verdict[k] for k in kegg_equivalence.EQUIVALENCE_KINDS)


def test_unannotated_candidate_cannot_be_equivalent():
    """Empty group sets must not match, or equivalence scores would be inflated."""
    assert not kegg_equivalence.is_equivalent("R99999", {"R00024"}, "ec")
    assert not kegg_equivalence.is_equivalent("R00024", {"R99999"}, "ec")


def test_equivalence_is_looser_than_exact():
    """A different reaction sharing an EC counts as equivalent but not exact."""
    idx = kegg_equivalence.equivalence_index()
    truth = "R00200"
    ec_groups = idx[truth]["ec"]
    sibling = next(
        (k for k, v in idx.items() if k != truth and v["ec"] & ec_groups), None
    )
    assert sibling, "expected at least one EC sibling in the KEGG feature table"
    assert not kegg_equivalence.matches_exact(sibling, {truth})
    assert kegg_equivalence.is_equivalent(sibling, {truth}, "ec")
    assert kegg_equivalence.match_kinds(sibling, {truth})["brite_orthology"]


# ----------------------------------------------------------------------------------
# Three-way averaging


def test_three_way_averaging_separates_model_and_cluster_weight():
    # One huge model with poor accuracy, two small models in one cluster with perfect
    # accuracy. Micro is dominated by the big model; macros are not.
    rows = (
        [{"model_id": "BIG", "cluster_id": "C1", "hit": 0} for _ in range(90)]
        + [{"model_id": "BIG", "cluster_id": "C1", "hit": 1} for _ in range(10)]
        + [{"model_id": "S1", "cluster_id": "C2", "hit": 1} for _ in range(5)]
        + [{"model_id": "S2", "cluster_id": "C2", "hit": 1} for _ in range(5)]
    )
    df = pd.DataFrame(rows)
    agg = three_way(df, "hit")

    assert agg["reaction_micro"] == pytest.approx(0.1818, abs=1e-3)
    assert agg["model_macro"] == pytest.approx((0.1 + 1.0 + 1.0) / 3, abs=1e-3)
    # Cluster macro: C1 = 0.1, C2 = 1.0
    assert agg["cluster_macro"] == pytest.approx(0.55, abs=1e-3)
    assert agg["n_models"] == 3 and agg["n_clusters"] == 2


def test_three_way_handles_empty_frame():
    empty = pd.DataFrame(columns=["model_id", "cluster_id", "hit"])
    agg = three_way(empty, "hit")
    assert agg["reaction_micro"] is None
    assert agg["n_reactions"] == 0


# ----------------------------------------------------------------------------------
# Retrieval vs reranking failure split


def test_retrieval_and_reranking_failures_are_mutually_exclusive():
    from benchmark.scripts.rank_baselines import CRITERIA  # noqa: F401  (import sanity)

    cases = [
        (None, True, False),   # answer absent -> retrieval failure
        (1, False, False),     # answer first -> success
        (4, False, True),      # answer present but not first -> reranking failure
    ]
    for first_hit, expect_retrieval, expect_rerank in cases:
        retrieval_failure = first_hit is None
        reranking_failure = first_hit is not None and first_hit > 1
        assert retrieval_failure is expect_retrieval
        assert reranking_failure is expect_rerank
        assert not (retrieval_failure and reranking_failure)


# ----------------------------------------------------------------------------------
# Missing-output classification must be reaction-aware.
#
# BIOMD0000000122/R1 and BIOMD0000000123/R1 are the same reaction:
#   NFAT_Pi_Nuc + Act_C_Nuc <=> Act_C_Nuc + NFAT_Nuc   (ground truth R00164)
# The only annotated species in those models are Ca_Cyt/Ca_Nuc (CHEBI:29108 / C00076),
# so the model *has* evidence while this reaction has none of it. The generator's
# species filter drops the reaction, which is a retrieval failure, not a fault.


def test_unannotated_reaction_in_annotated_model_is_unconstrained():
    from benchmark.scripts.generate_candidates import (
        STATUS_UNCONSTRAINED, classify_missing_reaction,
    )

    # "R2" is constrained (mentions the annotated calcium species); "R1" is not.
    status = classify_missing_reaction(
        "R1", is_ssx=False, had_evidence=True, constrained_ids={"R2"},
    )
    assert status == STATUS_UNCONSTRAINED


def test_reaction_with_constraints_that_vanishes_is_a_pipeline_failure():
    from benchmark.scripts.generate_candidates import (
        STATUS_ABSENT, classify_missing_reaction,
    )

    status = classify_missing_reaction(
        "R1", is_ssx=False, had_evidence=True, constrained_ids={"R1", "R2"},
    )
    assert status == STATUS_ABSENT


def test_missing_classification_precedence():
    from benchmark.scripts.generate_candidates import (
        STATUS_EXCHANGE_SKIPPED, STATUS_NO_SPECIES_EVIDENCE, classify_missing_reaction,
    )

    # SSX wins over everything else.
    assert classify_missing_reaction(
        "R1", is_ssx=True, had_evidence=True, constrained_ids={"R1"},
    ) == STATUS_EXCHANGE_SKIPPED
    # A model with no usable evidence at all is reported as such, not as unconstrained.
    assert classify_missing_reaction(
        "R1", is_ssx=False, had_evidence=False, constrained_ids=set(),
    ) == STATUS_NO_SPECIES_EVIDENCE


def test_unconstrained_reactions_are_not_pipeline_failures(monkeypatch, tmp_path):
    """End-to-end through assemble(): status recorded, failures table stays empty."""
    import benchmark.scripts.generate_candidates as gc

    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(gc, "CACHE_DIR", cache)
    monkeypatch.setattr(gc, "_cache_path", lambda m: cache / f"{m}.json")
    for name in ("CANDIDATES_CSV", "STATUS_CSV", "FAILURES_CSV", "CONFIG_JSON"):
        monkeypatch.setattr(gc, name, tmp_path / getattr(gc, name).name)

    for model_id in ("BIOMD0000000122", "BIOMD0000000123"):
        gc.write_json({
            "model_id": model_id,
            "config_id": gc.config_id(),
            "candidates": [],
            "status": [gc._status_row(
                model_id, "R1", gc.STATUS_UNCONSTRAINED, gc.config_id(),
            )],
            "failures": [],
            "elapsed_s": 1.0,
        }, cache / f"{model_id}.json")

    summary = gc.assemble(["BIOMD0000000122", "BIOMD0000000123"])

    assert summary["pipeline_failures"] == 0
    assert summary["status_counts"] == {gc.STATUS_UNCONSTRAINED: 2}

    status = pd.read_csv(gc.STATUS_CSV)
    assert set(status.status) == {gc.STATUS_UNCONSTRAINED}
    assert (status.filtered_species_count == 0).all()
    assert (status.num_candidates == 0).all()
    assert pd.read_csv(gc.CANDIDATES_CSV).empty
    assert pd.read_csv(gc.FAILURES_CSV).empty


# ----------------------------------------------------------------------------------
# Partial vs final assembly


@pytest.fixture
def cache_env(monkeypatch, tmp_path):
    """Isolate cache and output paths, seeded with one cached model."""
    import benchmark.scripts.generate_candidates as gc

    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(gc, "CACHE_DIR", cache)
    monkeypatch.setattr(gc, "_cache_path", lambda m: cache / f"{m}.json")
    for name in ("CANDIDATES_CSV", "STATUS_CSV", "FAILURES_CSV", "CONFIG_JSON"):
        monkeypatch.setattr(gc, name, tmp_path / getattr(gc, name).name)

    gc.write_json({
        "model_id": "M1",
        "config_id": gc.config_id(),
        "candidates": [],
        "status": [gc._status_row("M1", "R1", gc.STATUS_OK, gc.config_id(),
                                  num_candidates=0)],
        "failures": [],
        "elapsed_s": 1.0,
    }, cache / "M1.json")
    return gc


def test_partial_assembly_reports_pending_not_missing(cache_env):
    gc = cache_env
    summary = gc.assemble(["M1", "M2", "M3"], partial_ok=True)

    assert summary["partial_run"] is True
    assert summary["is_final"] is False
    assert summary["models_requested"] == 3
    assert summary["models_assembled"] == 1
    assert summary["models_pending"] == 2
    assert summary["models_pending_list"] == ["M2", "M3"]
    # Pending models must not masquerade as missing caches or failures.
    assert summary["models_missing_cache"] == []
    assert summary["pipeline_failures"] == 0


def test_final_assembly_treats_missing_caches_as_failure(cache_env):
    gc = cache_env
    summary = gc.assemble(["M1", "M2", "M3"], partial_ok=False)

    assert summary["partial_run"] is False
    assert summary["is_final"] is False
    assert summary["models_missing_cache"] == ["M2", "M3"]
    assert summary["models_pending"] == 0


def test_complete_assembly_is_final(cache_env):
    gc = cache_env
    summary = gc.assemble(["M1"], partial_ok=False)

    assert summary["is_final"] is True
    assert summary["partial_run"] is False
    assert summary["models_missing_cache"] == []


def _fake_reactions_reader(real_read_csv):
    """Serve a tiny reactions table so main() does not need the real benchmark."""
    import benchmark.scripts.generate_candidates as gc

    def reader(path, *args, **kwargs):
        if Path(path) == Path(gc.REACTIONS_CSV):
            return pd.DataFrame({
                "model_id": ["M1", "M2", "M3"],
                "reaction_id": ["R1", "R1", "R1"],
                "included_in_eval": [True, True, True],
                "is_exchange_ssx": [False, False, False],
                "ground_truth_kegg_all": ["R00164"] * 3,
            })
        return real_read_csv(path, *args, **kwargs)

    return reader


def _run_main(gc, monkeypatch, argv, worker_calls=None):
    """Invoke main() against the fake reactions table, recording _worker calls."""
    monkeypatch.setattr(sys, "argv", ["generate_candidates.py", *argv])
    monkeypatch.setattr(gc.pd, "read_csv", _fake_reactions_reader(gc.pd.read_csv))

    if worker_calls is not None:
        def fake_worker(model_id, scope):
            worker_calls.append(model_id)
            gc.write_json({
                "model_id": model_id,
                "config_id": gc.config_id(),
                "candidates": [],
                "status": [gc._status_row(model_id, "R1", gc.STATUS_OK, gc.config_id())],
                "failures": [],
                "elapsed_s": 0.0,
            }, gc._cache_path(model_id))
            return model_id

        monkeypatch.setattr(gc, "_worker", fake_worker)
    return gc.main()


def _write_cache(gc, model_id, *, failures=()):
    gc.write_json({
        "model_id": model_id,
        "config_id": gc.config_id(),
        "candidates": [],
        "status": [gc._status_row(model_id, "R1", gc.STATUS_OK, gc.config_id())],
        "failures": list(failures),
        "elapsed_s": 0.0,
    }, gc._cache_path(model_id))


GENUINE_FAILURE = {
    "model_id": "M1",
    "reaction_id": "R1",
    "scope": "reaction",
    "failure_type": "absent_from_generator_output",
    "message": "reaction present in frozen table but not returned by generator",
    "traceback_tail": "",
}


# --- exit codes ---------------------------------------------------------------------


def test_partial_run_with_pending_and_no_failures_exits_zero(cache_env, monkeypatch):
    gc = cache_env
    assert _run_main(gc, monkeypatch, ["--limit", "0", "--workers", "1"]) == 0


def test_partial_run_with_pipeline_failure_exits_nonzero(cache_env, monkeypatch):
    """A genuine failure must not hide behind an intentionally partial run."""
    gc = cache_env
    _write_cache(gc, "M1", failures=[GENUINE_FAILURE])
    assert _run_main(gc, monkeypatch, ["--limit", "0", "--workers", "1"]) != 0


def test_complete_coverage_with_pipeline_failure_exits_nonzero(cache_env, monkeypatch):
    gc = cache_env
    _write_cache(gc, "M1", failures=[GENUINE_FAILURE])
    _write_cache(gc, "M2")
    _write_cache(gc, "M3")
    assert _run_main(gc, monkeypatch, ["--assemble-only", "--workers", "1"]) != 0


def test_complete_coverage_without_failures_exits_zero(cache_env, monkeypatch):
    gc = cache_env
    _write_cache(gc, "M2")
    _write_cache(gc, "M3")
    assert _run_main(gc, monkeypatch, ["--assemble-only", "--workers", "1"]) == 0


def test_full_run_exits_nonzero_when_caches_missing(cache_env, monkeypatch):
    gc = cache_env
    assert _run_main(gc, monkeypatch, ["--assemble-only", "--workers", "1"]) == 1


def test_failed_complete_run_is_not_reported_as_final(cache_env, monkeypatch, caplog):
    gc = cache_env
    _write_cache(gc, "M1", failures=[GENUINE_FAILURE])
    _write_cache(gc, "M2")
    _write_cache(gc, "M3")
    with caplog.at_level("INFO"):
        assert _run_main(gc, monkeypatch, ["--assemble-only", "--workers", "1"]) != 0
    assert "final artifacts written" not in caplog.text


# --- generation selection -----------------------------------------------------------


def test_limit_zero_generates_nothing(cache_env, monkeypatch):
    """--limit 0 must not fall through to the full multi-day pass."""
    gc = cache_env
    calls: list = []
    assert _run_main(gc, monkeypatch, ["--limit", "0", "--workers", "1"], calls) == 0
    assert calls == []


def test_limit_one_generates_exactly_one_model(cache_env, monkeypatch):
    gc = cache_env
    calls: list = []
    assert _run_main(gc, monkeypatch, ["--limit", "1", "--workers", "1"], calls) == 0
    assert len(calls) == 1


def test_negative_limit_is_rejected(cache_env, monkeypatch):
    gc = cache_env
    calls: list = []
    assert _run_main(gc, monkeypatch, ["--limit", "-1", "--workers", "1"], calls) == 2
    assert calls == []


def test_no_limit_selects_every_pending_model(cache_env, monkeypatch):
    gc = cache_env  # M1 is already cached; M2 and M3 are pending.
    calls: list = []
    assert _run_main(gc, monkeypatch, ["--workers", "1"], calls) == 0
    assert sorted(calls) == ["M2", "M3"]


# ----------------------------------------------------------------------------------
# Cache schema invalidation


def test_cache_schema_bump_invalidates_old_payloads(cache_env, monkeypatch):
    gc = cache_env
    assert gc._load_cached("M1") is not None

    # A payload written under an earlier schema must not be reused.
    monkeypatch.setitem(gc.GENERATION_CONFIG, "cache_schema_version", 1)
    assert gc._load_cached("M1") is None


def test_cache_schema_version_is_in_config_id():
    from benchmark.scripts import generate_candidates as gc

    before = gc.config_id()
    original = gc.GENERATION_CONFIG["cache_schema_version"]
    try:
        gc.GENERATION_CONFIG["cache_schema_version"] = original + 1
        assert gc.config_id() != before
    finally:
        gc.GENERATION_CONFIG["cache_schema_version"] = original
    assert gc.config_id() == before


def test_config_id_changes_with_configuration():
    from benchmark.scripts import generate_candidates as gc

    before = gc.config_id()
    original = gc.GENERATION_CONFIG["reaction_scope"]
    try:
        gc.GENERATION_CONFIG["reaction_scope"] = "all" if original != "all" else "evaluable"
        assert gc.config_id() != before
    finally:
        gc.GENERATION_CONFIG["reaction_scope"] = original
    assert gc.config_id() == before
