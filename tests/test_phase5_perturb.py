"""Synthetic-only tests of deterministic identifier perturbations and accounting."""
import copy

import pytest

from benchmark.scripts import phase5_perturb as perturb


def fixture():
    record = {"model_id": "SYN-M", "reaction_id": "SYN-R", "sample_id": "SYN-1", "equation": "2 s1 + s2 => s3 + s4",
              "reactants": {"s1": 2, "s2": 1}, "products": {"s3": 1, "s4": 1}, "reversible": True,
              "participants": [{"species_id": f"s{i}", "name": f"Synthetic {i}", "chebi": [f"CHEBI:{i}"],
                                "kegg": [f"C{i:05}"]} for i in range(1, 5)]}
    ontology = perturb.Ontology({"CHEBI:1": ["CHEBI:10"], "CHEBI:2": ["CHEBI:10"], "CHEBI:3": ["CHEBI:10"], "CHEBI:4": ["CHEBI:10"]},
                               {f"CHEBI:{i}": [f"C{i:05}"] for i in range(1, 5)})
    return record, ontology


@pytest.mark.parametrize("fraction", [.1, .25, .5, .75])
def test_determinism_rounding_rate_and_original_preservation(fraction):
    record, ontology = fixture()
    original = copy.deepcopy(record)
    first = perturb.perturb(record, "remove", fraction, 123, ontology)
    assert first == perturb.perturb(record, "remove", fraction, 123, ontology)
    assert record == original
    assert first[1]["changed_species"] == int(fraction*4+.5)
    assert first[1]["links_removed"] == 2*first[1]["changed_species"]
    assert first[0]["equation"] == record["equation"]
    assert first[0]["reactants"] == record["reactants"]
    assert [p["name"] for p in first[0]["participants"]] == [p["name"] for p in record["participants"]]


def test_nested_masks_and_all_namespace_bundle_removal():
    record, ontology = fixture()
    previous = set()
    for fraction in [.1, .25, .5, .75]:
        altered, counts = perturb.perturb(record, "remove", fraction, 123, ontology)
        selected = {p["species_id"] for p in altered["participants"] if not p["chebi"] and not p["kegg"]}
        assert previous.issubset(selected)
        previous = selected


def test_one_sided_shared_species_and_ineligible_side():
    record, ontology = fixture()
    record["products"]["s1"] = 1
    changed, counts = perturb.perturb(record, "reactants", 1, 0, ontology)
    assert counts["changed_species"] == 2
    assert counts["selected_both_sides_species"] == 1
    assert changed["reversible"] is True
    record["products"] = {}
    unchanged, counts = perturb.perturb(record, "products", 1, 0, ontology)
    assert counts["zero_eligible_selected_side"] and counts["unchanged_query"]


def test_ancestor_distance_and_retained_direct_evidence():
    record, ontology = fixture()
    record["participants"][0]["chebi"].append("CHEBI:999")
    changed, counts = perturb.perturb(record, "ancestor", 1, 0, ontology)
    assert changed["participants"][0]["chebi"] == ["CHEBI:10", "CHEBI:999"]
    assert changed["participants"][0]["kegg"] == record["participants"][0]["kegg"]
    assert counts["events"][0]["ancestor_distances"][0]["distance"] == 1
    assert counts["events"][0]["ancestor_distances"][1]["distance"] == 0
    assert counts["ancestor_changes_with_direct_kegg_retained"] == 4


def test_wrong_mapping_is_sibling_and_disjoint_not_reference_derived():
    record, ontology = fixture()
    changed, counts = perturb.perturb(record, "wrong", 1, 123, ontology)
    assert counts["changed_species"] == 4
    for original, new in zip(record["participants"], changed["participants"]):
        assert not set(new["kegg"]).intersection(original["kegg"])
        assert set(ontology.parents[original["chebi"][0]]) & set(ontology.parents[new["chebi"][0]])
    isolated = perturb.Ontology({}, {})
    unchanged, counts = perturb.perturb(record, "wrong", 1, 123, isolated)
    assert counts["unchanged_selected_species"] == 4 and counts["unchanged_query"]


def test_lossless_query_roundtrip_cache_invalidation_and_leakage():
    record, ontology = fixture()
    query = perturb.render(record)
    assert perturb.render(perturb.parse_query(query)) == query
    changed, _ = perturb.perturb(record, "remove", .5, 123, ontology)
    assert perturb.digest(query) != perturb.digest(perturb.render(changed))
    record["equation"] = "R12345 synthetic => product"
    with pytest.raises(ValueError, match="leakage"):
        perturb.render(record)
    record["truth"] = ["R12345"]
    with pytest.raises(ValueError, match="metadata"):
        perturb.perturb(record, "remove", .5, 123, ontology)


def test_unannotated_species_not_sampled():
    record, ontology = fixture()
    for p in record["participants"]:
        p["chebi"], p["kegg"] = [], []
    _, counts = perturb.perturb(record, "remove", .75, 123, ontology)
    assert counts["zero_eligible"] and counts["eligible_species"] == 0
