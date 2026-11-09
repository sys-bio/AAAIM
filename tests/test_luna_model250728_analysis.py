import json

import pytest

from benchmark.scripts import luna_model250728_analysis as luna


def test_real_sbml_query_builder_keeps_all_reactions_and_avoids_ids(tmp_path):
    sbml = tmp_path / "tiny.xml"
    sbml.write_text('''<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
<model id="tiny"><listOfCompartments><compartment id="c" name="cytosol" size="1"/></listOfCompartments>
<listOfSpecies><species id="a_c" name="Alpha" compartment="c" boundaryCondition="false" hasOnlySubstanceUnits="false" initialConcentration="1"/><species id="b_c" name="Beta" compartment="c" boundaryCondition="false" hasOnlySubstanceUnits="false" initialConcentration="1"/></listOfSpecies>
<listOfReactions><reaction id="R_KEGG_R00001" name="hidden label" reversible="false"><listOfReactants><speciesReference species="a_c" stoichiometry="1"/></listOfReactants><listOfProducts><speciesReference species="b_c" stoichiometry="1"/></listOfProducts></reaction>
<reaction id="transport" reversible="true"><listOfReactants><speciesReference species="b_c" stoichiometry="2"/></listOfReactants><listOfProducts><speciesReference species="a_c" stoichiometry="2"/></listOfProducts></reaction></listOfReactions></model></sbml>''', encoding="utf-8")
    species = tmp_path / "species.csv"
    species.write_text("id,annotation\na_c,\nb_c,\n", encoding="utf-8")

    rows, summary = luna.build_queries(sbml, species)

    assert len(rows) == summary["reaction_count"] == 2
    assert summary["query_failures"] == 0
    assert all(row["query"].strip() for row in rows)
    assert all(row["reaction_id"] not in row["query"] for row in rows)
    assert all("R00001" not in row["query"] for row in rows)
    assert all("hidden label" not in row["query"] for row in rows)
    assert any("2 b_c" in row["reaction_equation"] for row in rows)


def _synthetic_evidence():
    return [{"evidence_id": "E01", "kegg_id": "R00001", "fused_rank": 1,
             "fused_score": 0.03, "bm25_rank": 2, "trained_biencoder_rank": 3,
             "name": "example", "definition": "example transformation", "equation": "A => B",
             "enzyme": "", "rclass": "", "brite": ""}]


def test_provider_payload_serializes_schema_settings_and_excludes_sidecar_provenance():
    body = luna.serialize_provider_body("A => B", _synthetic_evidence())
    assert body["model"] == "gpt-6.1-terra"
    assert body["store"] is False and body["tools"] == []
    assert body["service_tier"] == "default"
    assert body["max_output_tokens"] == 512
    assert body["text"]["format"]["strict"] is True
    assert "provenance" not in body["input"].lower()
    unit = {"model_id": luna.MODEL_ID, "reaction_id": "R1"}
    manifest = [{"unit": unit, "model": "gpt-6.1-terra", "provider_payload_sha256": luna.digest(body),
                 "unit_cache_key": luna.unit_cache_key({"model_id": luna.MODEL_ID, "reaction_id": "R1"}, body["model"], luna.digest(body)),
                 "reservation_nusd": luna.request_cost_bound(body)["reservation_nusd"]}]
    request = [{"unit": unit,
                "unit_cache_key": manifest[0]["unit_cache_key"], "provider_payload": body}]
    assert luna.verify_provider_plan(request, manifest)["provider_calls"] == 0


def test_cache_misses_for_changed_complete_provider_body(tmp_path):
    unit = {"model_id": luna.MODEL_ID, "reaction_id": "R1"}
    body_a = luna.serialize_provider_body("A => B", _synthetic_evidence())
    changed = _synthetic_evidence()
    changed[0]["definition"] = "changed transformation"
    body_b = luna.serialize_provider_body("A => B", changed)
    key_a = luna.unit_cache_key(unit, body_a["model"], luna.digest(body_a))
    key_b = luna.unit_cache_key(unit, body_b["model"], luna.digest(body_b))
    assert key_a != key_b
    cache = luna.UnitCache(tmp_path / "cache")
    cache.put(key_a, luna.digest(body_a), {"answer": "old"})
    assert cache.get(key_b, luna.digest(body_b)) is None
    with pytest.raises(ValueError, match="exact unit and provider body"):
        cache.get(key_a, luna.digest(body_b))


def test_complete_fusion_uses_full_catalog_and_deterministic_ties():
    components = {"bm25": {"r": {"ranked_ids": ["R2"]}},
                  "trained": {"r": {"ranked_ids": ["R3"]}},
                  "fused": {"r": {"ranked_ids": ["R2", "R3"]}}}
    first = luna.complete_fusion_rows(components, ["R3", "R1", "R2"])[0]
    second = luna.complete_fusion_rows(components, ["R3", "R1", "R2"])[0]
    assert first == second
    assert len(first["ranked_ids"]) == 3
    assert first["ranked_ids"][-1] == "R1"
    assert set(first["nonzero_scores"]) == {"R2", "R3"}


def test_attempt_ledger_retains_uncertain_reservation_and_blocks_retry(tmp_path):
    ledger = luna.AttemptLedger(tmp_path / "ledger.json", hard_cap_nusd=10)
    ledger.reserve("unit-key", 7)
    ledger.mark_uncertain("unit-key", "timeout")
    assert ledger.reserved_total_nusd() == 7
    with pytest.raises(ValueError, match="zero automatic retries"):
        ledger.reserve("unit-key", 1)


def test_verified_billing_releases_only_the_unused_reservation(tmp_path):
    ledger = luna.AttemptLedger(tmp_path / "ledger.json", hard_cap_nusd=10)
    ledger.reserve("unit-key", 7)
    ledger.settle_verified("unit-key", 3)
    assert ledger.reserved_total_nusd() == 3
