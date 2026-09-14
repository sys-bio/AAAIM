"""Regression checks for automatic annotation of SBML-qual models."""

from pathlib import Path

from core.model_info import extract_model_info, get_all_species_ids, get_species_display_names


ROOT = Path(__file__).resolve().parent.parent
QUAL_MODEL = ROOT / "tests" / "useCase" / "MODEL2506050001.xml"


def test_auto_uses_qualitative_species_and_transitions():
    display_names = get_species_display_names(str(QUAL_MODEL), "auto")
    species_ids = get_all_species_ids(str(QUAL_MODEL), "auto")

    assert species_ids
    assert species_ids == list(display_names)

    model_info = extract_model_info(str(QUAL_MODEL), species_ids[:5], "auto")
    assert model_info["reactions"]
    assert set(species_ids[:5]).issubset(model_info["display_names"])
