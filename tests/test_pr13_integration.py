from pathlib import Path

import pandas as pd

from benchmark.scripts.download_biomodels import portable_path
from core.annotation_workflow import _generate_recommendation_table
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
    monkeypatch.setattr(reaction_workflow, "query_llm", lambda *args, **kwargs: "R00001")

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
        only_rank_meaningful=False,
    )

    assert ranked["annotation"].tolist() == ["KEGG:R00001"]
    assert Path(tmp_path / "recommendations_llm_ranked.csv").exists()
