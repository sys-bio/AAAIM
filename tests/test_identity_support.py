import json
from pathlib import Path

import libsbml
import pandas as pd

from core.annotation_workflow import _calculate_metrics
from core.identity_support import assess_species_identity_support
from core.qualifier_workflow import predict_species_qualifiers
from core.update_model import update_annotation


MODEL = Path(__file__).parent / "useCase" / "MODEL2503190002.xml"


def _author_cases():
    rows = [
        ("ATM_active", "protein", "UNIPROT:Q13315", "ATM", "", ""),
        ("CK2", "protein", "UNIPROT:P68400", "CSNK2A1", "", ""),
        ("PP2A", "complex", "UNIPROT:P30153", "PPP2R1A", "PP2A_1", "PPP2R1A, PR65 alpha"),
        ("PP2A", "complex", "UNIPROT:P67775", "PPP2CA", "PP2A_2", "PPP2CA, PP2A catalytic subunit alpha"),
        ("PP2A_I2PP2A", "complex", "UNIPROT:P30153", "PPP2R1A", "PP2A_I2PP2A_1", "PPP2R1A, PR65 alpha"),
        ("PP2A_I2PP2A", "complex", "UNIPROT:Q01105", "SET", "PP2A_I2PP2A_3", "SET, I2PP2A"),
        ("PUMA", "protein", "UNIPROT:Q96PG8", "BBC3 isoforms 3/4", "", ""),
        ("PUMA", "protein", "UNIPROT:Q9BXH1", "BBC3 isoforms 1/2", "", ""),
        ("NRF2_degraded", "protein", "UNIPROT:Q16236", "NFE2L2", "", ""),
        ("Autophagy_active", "unknown", "", "", "", ""),
        ("DNA_damage", "unknown", "", "", "", ""),
    ]
    return pd.DataFrame([
        {"id": sid, "display_name": sid, "type": kind, "annotation": uri,
         "annotation_label": label, "component_id": cid, "component_name": cname,
         "status": "predicted only", "update_annotation": "add", "qualifier": "is"}
        for sid, kind, uri, label, cid, cname in rows
    ])


def test_author_cases_get_supported_or_unresolved_before_qualifier(monkeypatch):
    monkeypatch.setattr("core.identity_support.query_llm", lambda *args, **kwargs: None)
    table = assess_species_identity_support(_author_cases(), model_file=str(MODEL))
    by_id = {sid: group for sid, group in table.groupby("id")}
    assert by_id["ATM_active"].identity_status.tolist() == ["supported"]
    assert by_id["CK2"].identity_status.tolist() == ["unresolved"]
    assert by_id["PP2A"].identity_status.tolist() == ["unresolved", "unresolved"]
    assert by_id["PP2A_I2PP2A"].identity_status.tolist() == ["unresolved", "supported"]
    assert by_id["PUMA"].identity_status.tolist() == ["unresolved", "unresolved"]
    assert by_id["NRF2_degraded"].identity_status.tolist() == ["unresolved"]
    assert "NRF2_degradation_KEAP1" in by_id["NRF2_degraded"].Note.iloc[0]
    assert by_id["Autophagy_active"].identity_status.tolist() == ["unresolved"]
    assert by_id["DNA_damage"].identity_status.tolist() == ["unresolved"]
    assert all(table.loc[table.identity_status == "unresolved", "Note"].str.len() > 0)

    qualified = predict_species_qualifiers(table, effort="low")
    assert not qualified.loc[qualified.identity_status == "unresolved", "qualifier"].any()
    assert qualified.loc[qualified.id == "ATM_active", "qualifier"].iloc[0] == "isVersionOf"


def test_add_all_skips_unresolved_uris_and_writes_notes(tmp_path, monkeypatch):
    monkeypatch.setattr("core.identity_support.query_llm", lambda *args, **kwargs: None)
    table = assess_species_identity_support(_author_cases(), model_file=str(MODEL))
    table = predict_species_qualifiers(table)
    target = tmp_path / "annotated.xml"
    update_annotation(str(MODEL), table, str(target), add_all=True)
    document = libsbml.readSBMLFromFile(str(target))
    assert document.getNumErrors() == 0
    model = document.getModel()
    for sid in ("CK2", "PP2A", "PUMA", "NRF2_degraded"):
        species = model.getSpecies(sid)
        assert species.getNumCVTerms() == 0
        assert species.isSetNotes()
    complex_species = model.getSpecies("PP2A_I2PP2A")
    resources = [term.getResourceURI(i)
                 for term in (complex_species.getCVTerm(j) for j in range(complex_species.getNumCVTerms()))
                 for i in range(term.getNumResources())]
    assert any("Q01105" in uri for uri in resources)
    assert not any("P30153" in uri for uri in resources)
    assert "subunit" in complex_species.getNotesString().lower()
    assert model.getSpecies("ATM_active").getNumCVTerms() == 1


def test_ambiguous_alias_uses_reaction_context_and_can_be_supported(monkeypatch):
    prompts = []

    def fake_query(prompt, **kwargs):
        prompts.append(prompt)
        return json.dumps({"units": [{
            "unit_id": "NRF2_active", "status": "supported",
            "reason": "NFE2L2 is NRF2", "note": "",
        }]})

    monkeypatch.setattr("core.identity_support.query_llm", fake_query)
    rows = pd.DataFrame([{"id": "NRF2_active", "display_name": "NRF2_active",
                          "type": "protein", "annotation": "UNIPROT:Q16236",
                          "annotation_label": "NFE2L2", "status": "predicted only",
                          "qualifier": "is", "update_annotation": "add"}])
    result = assess_species_identity_support(rows, model_file=str(MODEL), effort="low")
    assert result.loc[0, "identity_status"] == "supported"
    assert "NRF2_activation" in prompts[0]
    assert '"role": "product"' in prompts[0]


def test_identity_support_failure_is_unresolved(monkeypatch):
    monkeypatch.setattr("core.identity_support.query_llm", lambda *args, **kwargs: "{}")
    rows = pd.DataFrame([{"id": "p21", "display_name": "p21", "type": "protein",
                          "annotation": "UNIPROT:P38936", "annotation_label": "CDKN1A",
                          "status": "predicted only", "qualifier": "is",
                          "update_annotation": "add"}])
    result = assess_species_identity_support(rows)
    assert result.loc[0, "identity_status"] == "unresolved"
    assert result.loc[0, "Note"]


def test_explicit_assembly_reaction_supports_complex_component(tmp_path, monkeypatch):
    document = libsbml.SBMLDocument(3, 1)
    model = document.createModel()
    model.setId("m")
    compartment = model.createCompartment()
    compartment.setId("c")
    compartment.setConstant(True)
    for sid in ("ProteinA", "ProteinB", "ComplexX"):
        species = model.createSpecies()
        species.setId(sid)
        species.setCompartment("c")
        species.setHasOnlySubstanceUnits(False)
        species.setBoundaryCondition(False)
        species.setConstant(False)
    reaction = model.createReaction()
    reaction.setId("assembly")
    reaction.setReversible(False)
    for sid in ("ProteinA", "ProteinB"):
        ref = reaction.createReactant()
        ref.setSpecies(sid)
        ref.setConstant(True)
    product = reaction.createProduct()
    product.setSpecies("ComplexX")
    product.setConstant(True)
    path = tmp_path / "assembly.xml"
    assert libsbml.writeSBMLToFile(document, str(path))
    rows = pd.DataFrame([{
        "id": "ComplexX", "display_name": "ComplexX", "type": "complex",
        "component_id": "ComplexX_1", "component_name": "ProteinA",
        "annotation": "UNIPROT:P00001", "annotation_label": "ProteinA",
        "status": "predicted only", "update_annotation": "add", "qualifier": "hasPart",
    }])
    monkeypatch.setattr("core.identity_support.query_llm", lambda *args, **kwargs: None)
    result = assess_species_identity_support(rows, model_file=str(path))
    assert result.loc[0, "identity_status"] == "supported"


def test_unresolved_candidates_are_not_counted_as_predictions(monkeypatch):
    monkeypatch.setattr("core.identity_support.query_llm", lambda *args, **kwargs: None)
    table = assess_species_identity_support(_author_cases(), model_file=str(MODEL))
    metrics = _calculate_metrics(table, {}, None, 8, 0.0, 0.0, 0.0)
    assert metrics["unresolved_entities"] == 7
    assert metrics["unresolved_candidates"] == 7
    assert metrics["total_predictions"] == 2
