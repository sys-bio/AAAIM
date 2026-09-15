"""Checks for annotate= species / reactions / both helpers."""

from pathlib import Path

import pandas as pd

from unittest.mock import patch

from core.annotation_workflow import (
    _apply_reason_comments,
    _chebi_rows,
    _extract_reason_comments,
    _load_species_recommendations,
    _output_csv,
    _parse_ranked_id_lines,
    _resolve_annotate,
    _species_recommendations_from_model,
    rank_species_annotations_with_llm,
)
from utils.constants import EntityType

ROOT = Path(__file__).resolve().parent.parent
MODEL_190 = ROOT / "tests" / "test_models" / "BIOMD0000000190.xml"
SPECIES_CSV = ROOT / "examples" / "glycolysis_part1-recommendations.csv"


def test_output_csv_prefix():
    assert _output_csv("modelA_gpt5", "foo.xml", "species") == "modelA_gpt5_species.csv"
    assert _output_csv("modelA_gpt5", "foo.xml", "reactions") == "modelA_gpt5_reactions.csv"
    assert _output_csv(None, "path/to/foo.xml", "species") == "foo.xml_species.csv"


def test_resolve_annotate():
    assert _resolve_annotate("species", EntityType.CHEMICAL, "direct") == "species"
    assert _resolve_annotate("reactions", EntityType.CHEMICAL, "direct") == "reactions"
    assert _resolve_annotate("reaction", EntityType.CHEMICAL, "direct") == "reactions"
    assert _resolve_annotate("both", EntityType.CHEMICAL, "direct") == "both"
    assert _resolve_annotate("species", EntityType.REACTION, "direct") == "reactions"
    assert _resolve_annotate("species", EntityType.CHEMICAL, "rulebased") == "reactions"
    assert _resolve_annotate("both", EntityType.REACTION, "rulebased") == "both"


def test_chebi_rows_filters_reason_and_non_chebi():
    df = pd.DataFrame([
        {"id": "Reason:", "annotation": "CHEBI:1"},
        {"id": "s1", "annotation": "CHEBI:4167"},
        {"id": "s2", "annotation": "UNIPROT:P12345"},
        {"id": "s3", "annotation": ""},
    ])
    out = _chebi_rows(df)
    assert list(out["id"]) == ["s1"]
    assert list(out["annotation"]) == ["CHEBI:4167"]


def test_species_recommendations_from_model_and_csv():
    from_model = _species_recommendations_from_model(str(MODEL_190))
    assert not from_model.empty
    assert {"id", "annotation"}.issubset(from_model.columns)
    assert from_model["annotation"].str.upper().str.startswith("CHEBI:").all()

    from_csv = _load_species_recommendations(str(MODEL_190), str(SPECIES_CSV))
    assert not from_csv.empty
    assert from_csv["annotation"].str.upper().str.startswith("CHEBI:").all()


def test_rank_species_annotations_keeps_n_return_in_llm_order():
    df = pd.DataFrame([
        {"id": "s1", "display_name": "glucose", "curated_name": "glucose",
         "annotation": "CHEBI:4167", "annotation_label": "D-glucopyranose"},
        {"id": "s1", "display_name": "glucose", "curated_name": "glucose",
         "annotation": "CHEBI:17234", "annotation_label": "glucose"},
        {"id": "s1", "display_name": "glucose", "curated_name": "glucose",
         "annotation": "CHEBI:42758", "annotation_label": "aldehyde-D-glucose"},
        {"id": "s2", "display_name": "ATP", "curated_name": "ATP",
         "annotation": "CHEBI:15422", "annotation_label": "ATP"},
    ])
    with patch("core.annotation_workflow.query_llm", return_value="s1: CHEBI:17234, CHEBI:4167") as mock_llm:
        out = rank_species_annotations_with_llm(
            "dummy.xml", df, n_return=2, model_notes="glycolysis model"
        )
    mock_llm.assert_called_once()
    prompt = mock_llm.call_args[0][0]
    assert "up to 2" in prompt
    assert "glycolysis model" in prompt
    assert "s1" in prompt and "s2" not in prompt.split("Instructions:")[0]
    s1 = out[out["id"] == "s1"]
    assert list(s1["annotation"]) == ["CHEBI:17234", "CHEBI:4167"]
    s2 = out[out["id"] == "s2"]
    assert list(s2["annotation"]) == ["CHEBI:15422"]


def test_parse_ranked_id_lines():
    assert _parse_ranked_id_lines("s1: CHEBI:17234, CHEBI:4167\nODC: UNK") == {
        "s1": ["CHEBI:17234", "CHEBI:4167"],
        "ODC": [],
    }


def test_rank_species_skips_llm_when_pool_fits_n_return():
    df = pd.DataFrame([
        {"id": "s1", "display_name": "glucose", "curated_name": "glucose",
         "annotation": "CHEBI:4167", "annotation_label": "D-glucopyranose"},
        {"id": "s1", "display_name": "glucose", "curated_name": "glucose",
         "annotation": "CHEBI:17234", "annotation_label": "glucose"},
        {"id": "s1", "display_name": "glucose", "curated_name": "glucose",
         "annotation": "CHEBI:42758", "annotation_label": "aldehyde-D-glucose"},
    ])
    with patch("core.annotation_workflow.query_llm") as mock_llm:
        out = rank_species_annotations_with_llm("dummy.xml", df, n_return=3)
    mock_llm.assert_not_called()
    assert list(out["annotation"]) == ["CHEBI:4167", "CHEBI:17234", "CHEBI:42758"]


def test_rank_species_uses_retrieval_top1_when_llm_response_is_empty():
    df = pd.DataFrame([
        {"id": "s1", "annotation": "CHEBI:1", "annotation_label": "first", "identity": "CHEBI:1"},
        {"id": "s1", "annotation": "CHEBI:2", "annotation_label": "second", "identity": "CHEBI:2"},
        {"id": "s1", "annotation": "CHEBI:3", "annotation_label": "third", "identity": "CHEBI:3"},
    ])
    with patch("core.annotation_workflow.query_llm", return_value=""):
        out = rank_species_annotations_with_llm("dummy.xml", df, n_return=1)
    assert list(out["annotation"]) == ["CHEBI:1"]


def test_reason_comment_once_per_chunk():
    df = pd.DataFrame([
        {"id": "s2", "annotation": "CHEBI:1"},
        {"id": "s2", "annotation": "CHEBI:2"},
        {"id": "s1", "annotation": "CHEBI:3"},
        {"id": "s3", "annotation": "CHEBI:4"},
    ])
    out = _apply_reason_comments(df, {
        "s1": "Chunk 1: glucose synonyms",
        "s3": "Chunk 2: ATP synonyms",
    })
    comments = list(out["comment"])
    assert comments[2] == "Chunk 1: glucose synonyms"
    assert comments[3] == "Chunk 2: ATP synonyms"
    assert comments[0] == "" and comments[1] == ""
    assert _extract_reason_comments(out) == {
        "s1": "Chunk 1: glucose synonyms",
        "s3": "Chunk 2: ATP synonyms",
    }


def test_rank_preserves_comment_on_first_row():
    df = pd.DataFrame([
        {"id": "s1", "display_name": "glucose", "curated_name": "glucose",
         "annotation": "CHEBI:4167", "annotation_label": "D-glucopyranose",
         "comment": "mapped from display names"},
        {"id": "s1", "display_name": "glucose", "curated_name": "glucose",
         "annotation": "CHEBI:17234", "annotation_label": "glucose",
         "comment": ""},
        {"id": "s1", "display_name": "glucose", "curated_name": "glucose",
         "annotation": "CHEBI:42758", "annotation_label": "aldehyde-D-glucose",
         "comment": ""},
    ])
    with patch("core.annotation_workflow.query_llm", return_value="s1: CHEBI:17234, CHEBI:4167"):
        out = rank_species_annotations_with_llm("dummy.xml", df, n_return=2)
    assert list(out["annotation"]) == ["CHEBI:17234", "CHEBI:4167"]
    assert list(out["comment"]) == ["mapped from display names", ""]


def test_format_prompt_includes_message():
    from core.model_info import format_prompt, get_all_species_ids
    ids = get_all_species_ids(str(MODEL_190))[:1]
    with_msg = format_prompt(str(MODEL_190), ids, message="Prefer KEGG names.")
    without = format_prompt(str(MODEL_190), ids)
    assert "// User message:" in with_msg
    assert "Prefer KEGG names." in with_msg
    assert "Prefer KEGG names." not in without


def test_fixed_gene_prompt_requires_typed_complex_components():
    from core.model_info import format_prompt, get_all_species_ids
    ids = get_all_species_ids(str(MODEL_190))[:1]
    prompt = format_prompt(str(MODEL_190), ids, entity_type="gene")
    assert "separate semicolon-delimited typed groups" in prompt
    assert "(gene)" in prompt


if __name__ == "__main__":
    test_output_csv_prefix()
    test_resolve_annotate()
    test_chebi_rows_filters_reason_and_non_chebi()
    test_species_recommendations_from_model_and_csv()
    test_parse_ranked_id_lines()
    test_rank_species_annotations_keeps_n_return_in_llm_order()
    test_rank_species_skips_llm_when_pool_fits_n_return()
    test_reason_comment_once_per_chunk()
    test_rank_preserves_comment_on_first_row()
    test_format_prompt_includes_message()
    print("ok")
