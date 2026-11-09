from pathlib import Path

import pandas as pd

from benchmark.scripts.download_biomodels import portable_path
from core.annotation_workflow import _generate_recommendation_table
from core import database_search
from core.data_types import ReactionRecommendation
from core.reaction import annotation_workflow as reaction_workflow


def test_downloader_accepts_paths_outside_repository(tmp_path):
    external_model = tmp_path / "models" / "BIOMD0000000001.xml"
    assert Path(portable_path(external_model)).is_absolute()


def test_generation_only_reaction_candidates_allow_missing_scores():
    recommendation = ReactionRecommendation(
        id="R_model",
        synonyms=[],
        candidates=["R00001"],
        candidate_names=["example"],
        match_score=[],
    )

    result = _generate_recommendation_table(
        "model.xml",
        [recommendation],
        {},
        {"display_names": {"R_model": "Example reaction"}},
        entity_type="reaction",
        database="kegg",
    )

    assert result.loc[0, "annotation"] == "KEGG:R00001"
    assert result.loc[0, "match_score"] == 0.0


def test_reaction_ranker_accepts_public_n_return_api(monkeypatch, tmp_path):
    class FakeFeatures:
        def get_definition(self, _annotation):
            return "A <=> B"

        def get_iubmb_chains(self, _annotation):
            return ()

        def get_iubmb_ancestors(self, _annotation):
            return frozenset()

    monkeypatch.setattr(
        reaction_workflow.KEGGReactionFeatures,
        "load_from_file",
        lambda _path: FakeFeatures(),
    )
    monkeypatch.setattr(reaction_workflow, "get_all_reaction_ids", lambda _path: ["rxn1"])
    monkeypatch.setattr(
        reaction_workflow,
        "map_reaction_ids_to_stoichiometry_strings",
        lambda _path: {"rxn1": "rxn1: A -> B"},
    )
    query_options = []

    def fake_query(prompt, **kwargs):
        query_options.append(kwargs)
        if "You are evaluating whether a biochemical reaction network" in prompt:
            return "TRUE"
        return "R00001"

    monkeypatch.setattr(reaction_workflow, "query_llm", fake_query)

    recommendations = pd.DataFrame(
        [{"id": "rxn1", "annotation": "KEGG:R00001"}]
    )
    output_path = tmp_path / "recommendations.csv"

    ranked = reaction_workflow.rank_kegg_annotations_with_llm(
        "model.xml",
        recommendations,
        n_return=1,
        model_notes="test context",
        csv_path=str(output_path),
        only_rank_meaningful=True,
        max_completion_tokens=77,
    )

    assert ranked["annotation"].tolist() == ["KEGG:R00001"]
    assert len(query_options) == 2
    assert all(options["max_completion_tokens"] == 77 for options in query_options)
    assert Path(tmp_path / "recommendations_llm_ranked.csv").exists()


def test_reaction_candidate_generation_keeps_filtered_reaction_ids_aligned(monkeypatch):
    all_ids = ["rxn1", "rxn2", "rxn3"]
    extracted_ids = ["rxn1", "rxn3"]
    sent_to_matcher = {}

    monkeypatch.setattr(reaction_workflow, "check_environment", lambda _path: True)
    monkeypatch.setattr(reaction_workflow, "get_all_reaction_ids", lambda _path: all_ids)
    monkeypatch.setattr(
        reaction_workflow,
        "extract_model_info",
        lambda *_args: {"model_name": "fixture", "model_notes": ""},
    )
    monkeypatch.setattr(
        reaction_workflow,
        "map_chebi_to_kegg",
        lambda _df: (pd.DataFrame(), pd.DataFrame([{"id": "s1", "KEGG_ID": "C00001"}])),
    )
    monkeypatch.setattr(
        reaction_workflow,
        "extract_reactions_with_ids_from_sbml",
        lambda *_args: (extracted_ids, ["A -> B", "C -> D"], {"s1"}),
    )

    def fake_matcher(_rxns, ids, _species, **_kwargs):
        sent_to_matcher["ids"] = ids
        return [], [], {}

    monkeypatch.setattr(reaction_workflow, "map_reactions_to_kegg_with_relaxation", fake_matcher)
    monkeypatch.setattr(
        reaction_workflow,
        "_generate_recommendation_table",
        lambda *_args: pd.DataFrame(),
    )

    reaction_workflow.run_kegg_annotation_workflow_rulebased(
        "model.xml", pd.DataFrame([{"id": "s1", "annotation": "CHEBI:1"}]),
        evaluate_candidates=False,
    )

    assert all_ids != extracted_ids
    assert sent_to_matcher["ids"] == extracted_ids


def test_ontology_expansion_cache_preserves_empty_ambiguous_and_large_matches(
    monkeypatch,
):
    catalog = {
        "R_EMPTY": {"substrates": ["C999"], "products": ["C998"]},
        "R_AMBIG_A": {"substrates": ["C1"], "products": ["C2"]},
        "R_AMBIG_B": {"substrates": ["C1"], "products": ["C2"]},
    }
    catalog.update(
        {
            f"R_LARGE_{i:03d}": {"substrates": ["C1"], "products": ["C2"]}
            for i in range(80)
        }
    )
    monkeypatch.setattr(database_search, "load_kegg_parsed_reactions_dict", lambda: catalog)

    expansion_calls = []

    def identity_expansion(compounds, **_kwargs):
        expansion_calls.append(tuple(sorted(compounds)))
        return set(compounds)

    monkeypatch.setattr(
        database_search,
        "_expand_kegg_compound_set_by_chebi_ontology",
        identity_expansion,
    )
    context = {
        "chebi_to_kegg": {"CHEBI:1": {"C1"}, "CHEBI:2": {"C2"}},
        "parent_map": {"CHEBI:1": set(), "CHEBI:2": set()},
        "child_map": {},
        "reaction_ontology_max_up": 2,
        "reaction_ontology_max_down": 1,
    }

    baseline = database_search.filter_kegg_reactions({"C1"}, {"C2"}, **context)
    expected = ["R_AMBIG_A", "R_AMBIG_B"] + [f"R_LARGE_{i:03d}" for i in range(80)]
    assert baseline == expected

    cache = {}
    first_cached = database_search.filter_kegg_reactions(
        {"C1"}, {"C2"}, ontology_expansion_cache=cache, **context
    )
    second_cached = database_search.filter_kegg_reactions(
        {"C1"}, {"C2"}, ontology_expansion_cache=cache, **context
    )
    empty_baseline = database_search.filter_kegg_reactions({"C888"}, {"C887"}, **context)
    calls_before_empty_cached = len(expansion_calls)
    empty_cached = database_search.filter_kegg_reactions(
        {"C888"}, {"C887"}, ontology_expansion_cache=cache, **context
    )

    assert first_cached == baseline == second_cached
    assert empty_baseline == empty_cached == []
    assert len(expansion_calls) == calls_before_empty_cached


def test_candidate_scores_order_and_metadata_match_with_expansion_cache(monkeypatch):
    large_ids = [f"R_LARGE_{i:03d}" for i in range(80)]
    catalog = {
        "R_AMBIG_A": {"substrates": ["C1"], "products": ["C2"]},
        "R_AMBIG_B": {"substrates": ["C1"], "products": ["C2"]},
        **{
            kid: {"substrates": ["C1"], "products": ["C2"]}
            for kid in large_ids
        },
    }
    features = {kid: {"ORTHOLOGY": f"{kid} example reaction"} for kid in catalog}
    monkeypatch.setattr(database_search, "load_kegg_parsed_reactions_dict", lambda: catalog)
    monkeypatch.setattr(database_search, "load_kegg_reaction_features_dict", lambda: features)
    monkeypatch.setattr(
        database_search,
        "_expand_kegg_compound_set_by_chebi_ontology",
        lambda compounds, **_kwargs: set(compounds),
    )

    def normalized(rid, left, right):
        return {
            "id": rid,
            "reaction_string": f"{rid}: {left} -> {right}",
            "substrates": {
                left: {"coeff": 1, "candidates": [{"kegg_id": "C1" if left == "S1" else "C99"}]}
            },
            "products": {
                right: {"coeff": 1, "candidates": [{"kegg_id": "C2" if right == "P1" else "C98"}]}
            },
        }

    normalized_reactions = [
        normalized("ambiguous", "S1", "P1"),
        normalized("large", "S1", "P1"),
        normalized("empty", "S3", "P3"),
    ]
    common = {
        "evaluate_candidates": True,
        "include_exchange_reactions": False,
        "species_to_chebi": {"S1": "CHEBI:1", "P1": "CHEBI:2"},
        "chebi_to_kegg": {"CHEBI:1": {"C1"}, "CHEBI:2": {"C2"}},
        "parent_map": {"CHEBI:1": set(), "CHEBI:2": set()},
        "child_map": {},
    }
    baseline = database_search._get_kegg_recommendations_rulebased(
        normalized_reactions, **common
    )
    cache = {}
    cached = database_search._get_kegg_recommendations_rulebased(
        normalized_reactions, ontology_expansion_cache=cache, **common
    )

    def signature(rows):
        return [
            (
                row.id,
                row.candidates,
                row.candidate_names,
                row.match_score,
                row.substrates,
                row.products,
                row.metadata,
            )
            for row in rows
        ]

    assert signature(cached) == signature(baseline)
    candidate_counts = {}
    for row in cached:
        candidate_counts[row.id] = candidate_counts.get(row.id, 0) + len(row.candidates)
    assert candidate_counts == {
        "ambiguous": 82,
        "large": 82,
        "empty": 0,
    }
