import libsbml
import pandas as pd

from core.annotation_workflow import _generate_recommendation_table
from core.data_types import Recommendation
from core.model_info import find_species_with_annotations_and_qualifiers
from core.update_model import update_annotation


def test_complex_predictions_use_haspart_qualifier():
    complex_rec = Recommendation(
        id="E", synonyms=[], candidates=["P00533"],
        candidate_names=["EGFR"], match_score=[1.0],
    )
    protein_rec = Recommendation(
        id="Raf1", synonyms=[], candidates=["P04049"],
        candidate_names=["RAF1"], match_score=[1.0],
    )
    df = _generate_recommendation_table(
        "model.xml",
        [complex_rec, protein_rec],
        {},
        {"display_names": {}},
        entity_type="auto",
        database="kegg",
        entity_type_dict={"E": "complex", "Raf1": "protein"},
    )
    assert df.loc[df["id"] == "E", "qualifier"].item() == "hasPart"
    assert df.loc[df["id"] == "Raf1", "qualifier"].item() == "is"


def _model(path):
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("m")
    comp = model.createCompartment()
    comp.setId("c")
    comp.setConstant(True)
    for sid, meta in (("E", "meta_E"), ("Raf1", "meta_Raf1")):
        species = model.createSpecies()
        species.setId(sid)
        species.setMetaId(meta)
        species.setCompartment("c")
        species.setHasOnlySubstanceUnits(False)
        species.setBoundaryCondition(False)
        species.setConstant(False)
    kept = model.getSpecies("Raf1")
    assert kept.setAnnotation(
        '<annotation><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">'
        '<rdf:Description rdf:about="#meta_Raf1"><bqbiol:is><rdf:Bag>'
        '<rdf:li rdf:resource="http://identifiers.org/chebi/CHEBI:15377"/>'
        '</rdf:Bag></bqbiol:is></rdf:Description></rdf:RDF></annotation>'
    ) == 0
    assert libsbml.writeSBMLToFile(doc, str(path))


def test_add_all_sets_metaid_when_missing(tmp_path):
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("m")
    comp = model.createCompartment()
    comp.setId("c")
    comp.setConstant(True)
    species = model.createSpecies()
    species.setId("ATM_active")
    species.setCompartment("c")
    species.setHasOnlySubstanceUnits(False)
    species.setBoundaryCondition(False)
    species.setConstant(False)
    source = tmp_path / "model.xml"
    assert libsbml.writeSBMLToFile(doc, str(source))
    updated = tmp_path / "updated.xml"
    table = pd.DataFrame([
        {"id": "ATM_active", "annotation": "UNIPROT:Q13315", "update_annotation": "ignore", "qualifier": "is"},
    ])
    update_annotation(str(source), table, str(updated), add_all=True)
    uniprot, _ = find_species_with_annotations_and_qualifiers(str(updated), "uniprot")
    assert uniprot["ATM_active"] == ["Q13315"]


def test_add_all_ignores_update_column(tmp_path):
    source = tmp_path / "model.xml"
    updated = tmp_path / "updated.xml"
    _model(source)
    table = pd.DataFrame([
        {"id": "E", "annotation": "UNIPROT:P00533", "update_annotation": "ignore", "qualifier": "hasPart"},
    ])
    update_annotation(str(source), table, str(updated), add_all=True)
    uniprot, quals = find_species_with_annotations_and_qualifiers(str(updated), "uniprot")
    assert uniprot["E"] == ["P00533"]
    assert quals["E"]["P00533"] == "hasPart"


def test_update_annotation_writes_row_qualifier(tmp_path):
    source = tmp_path / "model.xml"
    updated = tmp_path / "updated.xml"
    _model(source)
    table = pd.DataFrame([
        {"id": "E", "annotation": "UNIPROT:P00533", "update_annotation": "add", "qualifier": "hasPart"},
        {"id": "E", "annotation": "CHEBI:15996", "update_annotation": "add", "qualifier": "hasPart"},
        {"id": "Raf1", "annotation": "UNIPROT:P04049", "update_annotation": "add", "qualifier": "is"},
        {"id": "Raf1", "annotation": "CHEBI:15377", "update_annotation": "keep", "qualifier": "is"},
    ])
    update_annotation(str(source), table, str(updated))

    uniprot, uniprot_qual = find_species_with_annotations_and_qualifiers(str(updated), "uniprot")
    chebi, chebi_qual = find_species_with_annotations_and_qualifiers(str(updated), "chebi")
    assert uniprot["E"] == ["P00533"]
    assert uniprot_qual["E"]["P00533"] == "hasPart"
    assert chebi["E"] == ["15996"]
    assert chebi_qual["E"]["15996"] == "hasPart"
    assert uniprot["Raf1"] == ["P04049"]
    assert uniprot_qual["Raf1"]["P04049"] == "is"
    assert chebi["Raf1"] == ["15377"]
    assert chebi_qual["Raf1"]["15377"] == "is"


def test_same_uri_can_have_distinct_qualifiers_without_duplicate_pair(tmp_path):
    source = tmp_path / "model.xml"
    updated = tmp_path / "updated.xml"
    _model(source)  # Raf1 already has CHEBI:15377 under bqbiol:is.
    table = pd.DataFrame([
        {"id": "Raf1", "annotation": "CHEBI:15377", "update_annotation": "add", "qualifier": "isVersionOf"},
        {"id": "Raf1", "annotation": "CHEBI:15377", "update_annotation": "add", "qualifier": "isVersionOf"},
    ])

    update_annotation(str(source), table, str(updated))

    document = libsbml.readSBML(str(updated))
    assert document.getNumErrors() == 0
    species = document.getModel().getSpecies("Raf1")
    annotation = species.getAnnotation().toXMLString()
    uri = 'rdf:resource="http://identifiers.org/chebi/CHEBI:15377"'
    assert annotation.count(uri) == 2
    assert "<bqbiol:is>" in annotation
    assert "<bqbiol:isVersionOf>" in annotation
