import json
from pathlib import Path

from core.annotation_workflow import (
    _filter_recommendations_by_chemistry,
    _formula_counter,
    _parse_structured_normalization,
    _parse_structured_ranking,
    _search_one_database,
    _search_with_entity_type_validation,
)
from core.model_info import get_species_chemical_properties
from utils.constants import EntityType


USE_CASE = Path(__file__).parent / "useCase" / "MODEL2507280001.xml"


def test_generic_formula_is_inconclusive():
    assert _formula_counter("C6H11O2SR") is None


def test_cross_database_validation_corrects_egf_route():
    events = []
    recommendations, database, entity_type = _search_with_entity_type_validation(
        "egf_component",
        {"egf_component": ["EGF", "epidermal growth factor"]},
        "chebi",
        ["chebi", "uniprot"],
        "direct",
        3,
        tax_id="9606",
        validation_events=events,
    )

    assert database == "uniprot"
    assert entity_type == "protein"
    assert recommendations[0].candidates == ["P01133"]
    assert events[0]["from_top"]["candidate"] == "140739"
    assert events[0]["to_top"]["candidate"] == "P01133"


def test_formula_and_charge_validation_selects_acetate():
    recommendations = _search_one_database(
        ["M_ac_c"],
        {"M_ac_c": ["acetate", "acetic acid"]},
        "chebi",
        "direct",
        3,
    )
    events = []
    _filter_recommendations_by_chemistry(
        recommendations,
        {"M_ac_c": "chebi"},
        {},
        get_species_chemical_properties(str(USE_CASE), ["M_ac_c"]),
        events,
    )

    assert recommendations[0].candidates == ["30089"]
    rejected = {event["candidate"]: event["conflicts"] for event in events}
    assert rejected["CHEBI:47622"] == ["charge"]  # its generic R-group formula is inconclusive
    assert rejected["CHEBI:15366"] == ["charge"]


def test_chemistry_conflict_without_exact_alternative_preserves_candidates():
    recommendations = _search_one_database(
        ["M_cobalt2_c"],
        {"M_cobalt2_c": ["cobalt2", "cobalt ion"]},
        "chebi",
        "direct",
        3,
    )
    original = recommendations[0].candidates.copy()
    events = []
    _filter_recommendations_by_chemistry(
        recommendations,
        {"M_cobalt2_c": "chebi"},
        {},
        get_species_chemical_properties(str(USE_CASE), ["M_cobalt2_c"]),
        events,
    )

    assert recommendations[0].candidates == original
    assert not events


def test_missing_chebi_chemistry_keeps_generic_formate_ester_for_review():
    recommendations = _search_one_database(
        ["M_for_c"],
        {"M_for_c": ["formate", "formic acid"]},
        "chebi",
        "direct",
        3,
    )
    _filter_recommendations_by_chemistry(
        recommendations,
        {"M_for_c": "chebi"},
        {},
        get_species_chemical_properties(str(USE_CASE), ["M_for_c"]),
    )

    assert recommendations[0].candidates[0] == "15740"  # exact formate wins retrieval order
    assert "52343" in recommendations[0].candidates  # no formula/structure for this class


def test_structured_normalization_requires_complete_ids():
    response = json.dumps(
        {
            "entities": [
                {
                    "id": "s1",
                    "entity_type": "complex",
                    "names": [],
                    "components": [
                        {"type": "protein", "names": ["EGF", "epidermal growth factor"]},
                        {"type": "protein", "names": ["EGFR"]},
                    ],
                }
            ],
            "reason": "",
        }
    )
    synonyms, entity_types, _reason, components = _parse_structured_normalization(
        response, ["s1"], EntityType.AUTO
    )
    assert synonyms["s1"] == ["EGF", "epidermal growth factor", "EGFR"]
    assert entity_types["s1"] == "complex"
    assert components["s1"][0] == ("protein", ["EGF", "epidermal growth factor"])


def test_structured_normalization_keeps_per_entity_abstention():
    response = json.dumps(
        {
            "entities": [
                {
                    "id": "s1",
                    "entity_type": "chemical",
                    "names": [],
                    "components": [],
                }
            ],
            "reason": "",
        }
    )
    synonyms, entity_types, _reason, _components = _parse_structured_normalization(
        response, ["s1"], EntityType.CHEMICAL
    )
    assert synonyms == {"s1": ["UNK"]}
    assert entity_types == {"s1": "chemical"}


def test_structured_ranking_rejects_unoffered_identifier():
    valid = json.dumps(
        {"units": [{"id": "s1", "selected_ids": ["CHEBI:30089"]}]}
    )
    assert _parse_structured_ranking(
        valid, ["s1"], {"s1": {"CHEBI:30089"}}, 1
    ) == {"s1": ["CHEBI:30089"]}

    invalid = json.dumps(
        {"units": [{"id": "s1", "selected_ids": ["CHEBI:47622"]}]}
    )
    try:
        _parse_structured_ranking(
            invalid, ["s1"], {"s1": {"CHEBI:30089"}}, 1
        )
    except ValueError as exc:
        assert "not present in candidate pool" in str(exc)
    else:
        raise AssertionError("unoffered identifier was accepted")
