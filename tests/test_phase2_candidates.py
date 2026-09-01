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
