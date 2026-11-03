import json

import libsbml
import pandas as pd

from core.model_info import find_species_with_annotations_and_qualifiers
from core.qualifier_workflow import predict_species_qualifiers
from core.update_model import update_annotation


def _rows():
    return pd.DataFrame([
        {"id": "ATM_active", "display_name": "ATM active", "type": "protein",
         "annotation": "UNIPROT:Q13315", "annotation_label": "ATM_HUMAN",
         "status": "predicted only", "update_annotation": "add", "qualifier": "is"},
        {"id": "PUMA", "display_name": "PUMA", "type": "protein",
         "annotation": "UNIPROT:Q9BXH1", "annotation_label": "BBC3_HUMAN",
         "status": "predicted only", "update_annotation": "add", "qualifier": "is"},
        {"id": "NRF2_degraded", "display_name": "NRF2 degraded", "type": "protein",
         "annotation": "UNIPROT:Q16236", "annotation_label": "NFE2L2_HUMAN",
         "status": "predicted only", "update_annotation": "add", "qualifier": "is"},
        {"id": "ATM_existing", "display_name": "ATM existing", "type": "protein",
         "annotation": "UNIPROT:Q13315", "annotation_label": "ATM_HUMAN",
         "status": "original and predicted", "update_annotation": "keep", "qualifier": "isVersionOf"},
    ])


def test_low_effort_keeps_identity_and_limits_prompt(monkeypatch):
    prompts = []

    def fake_query(prompt, **kwargs):
        prompts.append(prompt)
        assert kwargs["response_format"]["type"] == "json_schema"
        return json.dumps({"rows": [{"row_id": "1", "qualifier": "is", "reason": "direct identity"}]})

    monkeypatch.setattr("core.qualifier_workflow.query_llm", fake_query)
    original = _rows()
    result = predict_species_qualifiers(
        original, effort="low",
        model_info={"model_notes": "SECRET_NOTES", "reactions": ["PUMA -> product"]},
    )

    assert len(prompts) == 1
    assert "SECRET_NOTES" not in prompts[0]
    assert "reactions" not in prompts[0]
    assert "http://identifiers.org/uniprot/Q9BXH1" in prompts[0]
    assert result["annotation"].tolist() == original["annotation"].tolist()
    assert result["qualifier"].tolist() == ["isVersionOf", "is", "is", "isVersionOf"]
    assert result["qualifier_rule"].tolist() == ["explicit_state", "llm", "explicit_degraded_state", "existing"]
    assert result.loc[2, "qualifier_status"] == "predicted"
    assert result.loc[2, "update_annotation"] == "add"


def test_high_effort_supplies_model_notes_and_matching_reactions(monkeypatch):
    prompts = []

    def fake_query(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps({"rows": [{"row_id": "1", "qualifier": "is", "reason": "identity"}]})

    monkeypatch.setattr("core.qualifier_workflow.query_llm", fake_query)
    result = predict_species_qualifiers(
        _rows(), effort="high",
        model_info={"model_notes": "DNA damage model",
                    "reactions": ["PUMA -> apoptosis", "ATP -> ADP", "PUMA2 -> sink"]},
    )
    assert "DNA damage model" in prompts[0]
    assert "PUMA -> apoptosis" in prompts[0]
    assert "ATP -> ADP" not in prompts[0]
    assert "PUMA2 -> sink" not in prompts[0]
    assert result.loc[1, "qualifier"] == "is"


def test_high_effort_can_use_explicit_model_side_state_evidence(monkeypatch):
    rows = _rows().iloc[[1]].copy()

    def fake_query(*args, **kwargs):
        return json.dumps({"rows": [
            {"row_id": "0", "qualifier": "isVersionOf", "reason": "PUMA phosphorylated in notes"},
        ]})

    monkeypatch.setattr("core.qualifier_workflow.query_llm", fake_query)
    result = predict_species_qualifiers(
        rows, effort="high",
        model_info={"model_notes": "The PUMA phosphorylated state is modeled.", "reactions": []},
    )
    assert result.loc[0, "qualifier"] == "isVersionOf"
    assert result.loc[0, "qualifier_rule"] == "llm_context_state"


def test_tied_accessions_receive_one_qualifier_decision(monkeypatch):
    rows = _rows().iloc[[1]].copy()
    rows["identity"] = "BBC3"
    tied = rows.copy()
    tied["annotation"] = "UNIPROT:Q96PG8"
    rows = pd.concat([rows, tied], ignore_index=True)
    prompts = []

    def fake_query(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps({"rows": [{"row_id": "0", "qualifier": "is", "reason": "BBC3"}]})

    monkeypatch.setattr("core.qualifier_workflow.query_llm", fake_query)
    result = predict_species_qualifiers(rows, effort="low")
    assert len(prompts) == 1
    assert "Q9BXH1" in prompts[0] and "Q96PG8" in prompts[0]
    assert result["annotation"].tolist() == rows["annotation"].tolist()
    assert result["qualifier"].tolist() == ["is", "is"]


def test_invalid_qualifier_defaults_to_is_and_add_all_writes_it(tmp_path, monkeypatch):
    monkeypatch.setattr("core.qualifier_workflow.query_llm", lambda *args, **kwargs: "{}")
    table = predict_species_qualifiers(_rows())
    assert table.loc[1, "qualifier_status"] == "predicted"
    assert table.loc[1, "qualifier_rule"] == "default_is"
    assert table.loc[1, "qualifier"] == "is"

    document = libsbml.SBMLDocument(3, 1)
    model = document.createModel()
    model.setId("m")
    compartment = model.createCompartment()
    compartment.setId("c")
    compartment.setConstant(True)
    for species_id in table["id"]:
        species = model.createSpecies()
        species.setId(species_id)
        species.setCompartment("c")
        species.setHasOnlySubstanceUnits(False)
        species.setBoundaryCondition(False)
        species.setConstant(False)
    source = tmp_path / "model.xml"
    target = tmp_path / "annotated.xml"
    assert libsbml.writeSBMLToFile(document, str(source))
    update_annotation(str(source), table, str(target), add_all=True)
    identifiers, qualifiers = find_species_with_annotations_and_qualifiers(str(target), "uniprot")
    assert identifiers.get("ATM_active") == ["Q13315"]
    assert qualifiers["ATM_active"]["Q13315"] == "isVersionOf"
    assert identifiers.get("PUMA") == ["Q9BXH1"]
    assert qualifiers["PUMA"]["Q9BXH1"] == "is"
    assert identifiers.get("NRF2_degraded") == ["Q16236"]
    assert qualifiers["NRF2_degraded"]["Q16236"] == "is"


def test_unsupported_llm_relationships_default_to_is(monkeypatch):
    rows = _rows().iloc[[1]].copy()
    rows.loc[rows.index[0], "annotation_label"] = "BBC3 isoforms 1/2"
    ck2 = rows.copy()
    ck2["id"] = "CK2"
    ck2["display_name"] = "CK2"
    ck2["annotation"] = "UNIPROT:P68400"
    ck2["annotation_label"] = "CSNK2A1"
    rows = pd.concat([rows, ck2], ignore_index=True)

    def fake_query(*args, **kwargs):
        return json.dumps({"rows": [
            {"row_id": "0", "qualifier": "isVersionOf", "reason": "UniProt isoform label"},
            {"row_id": "1", "qualifier": "hasPart", "reason": "CK2 subunit"},
        ]})

    monkeypatch.setattr("core.qualifier_workflow.query_llm", fake_query)
    result = predict_species_qualifiers(rows)
    assert result["qualifier"].tolist() == ["is", "is"]
    assert result["qualifier_rule"].tolist() == ["default_is", "default_is"]


def test_no_predicted_uri_has_no_qualifier(monkeypatch):
    rows = _rows().iloc[[1]].copy()
    rows["annotation"] = ""
    monkeypatch.setattr("core.qualifier_workflow.query_llm", lambda *args, **kwargs: None)
    result = predict_species_qualifiers(rows)
    assert result.loc[0, "qualifier"] == ""
    assert result.loc[0, "qualifier_status"] == "not_applicable"


def test_uri_without_existing_qualifier_defaults_to_is(monkeypatch):
    rows = _rows().iloc[[3]].copy()
    rows["qualifier"] = ""

    def fake_query(*args, **kwargs):
        return json.dumps({"rows": [
            {"row_id": "0", "qualifier": "is", "reason": "No modeled state"},
        ]})

    monkeypatch.setattr("core.qualifier_workflow.query_llm", fake_query)
    result = predict_species_qualifiers(rows)
    assert result.loc[0, "qualifier"] == "is"
