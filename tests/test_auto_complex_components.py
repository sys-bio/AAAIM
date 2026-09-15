"""Parser and per-component routing for auto-type complexes."""

from unittest.mock import patch

from core.annotation_workflow import _search_databases, rank_species_annotations_with_llm
from core.database_search import expand_lookup_synonyms, get_species_recommendations_direct
from core import database_search
from core.data_types import Recommendation
from core.llm_interface import parse_llm_response, parse_typed_components
from utils.constants import DatabaseID, EntityType
import pandas as pd


def test_parse_typed_complex_line():
    text = (
        'Ras_Raf1 (complex): "HRAS", "KRAS", "NRAS" (protein); "RAF1" (protein)\n'
        'Ras_GTP (complex): "HRAS", "KRAS", "NRAS" (protein); "GTP", "guanosine triphosphate" (chemical)\n'
        'A (chemical): "glucose", "D-glucose"\n'
        "Reason: mixed complex\n"
    )
    synonyms, types, reason, components = parse_llm_response(text, EntityType.AUTO)
    assert types["Ras_Raf1"] == "complex"
    assert types["Ras_GTP"] == "complex"
    assert types["A"] == "chemical"
    assert synonyms["Ras_Raf1"] == ["HRAS", "KRAS", "NRAS", "RAF1"]
    assert components["Ras_Raf1"] == [
        ("protein", ["HRAS", "KRAS", "NRAS"]),
        ("protein", ["RAF1"]),
    ]
    assert components["Ras_GTP"] == [
        ("protein", ["HRAS", "KRAS", "NRAS"]),
        ("chemical", ["GTP", "guanosine triphosphate"]),
    ]
    assert "A" not in components
    assert "mixed complex" in reason
    assert "(protein)" not in synonyms["Ras_Raf1"]


def test_parse_legacy_complex_has_no_components():
    text = 'D (complex): "glucose", "ATP", "Hexokinase-1"\nReason: old format\n'
    synonyms, types, _reason, components = parse_llm_response(text, EntityType.AUTO)
    assert types["D"] == "complex"
    assert synonyms["D"] == ["glucose", "ATP", "Hexokinase-1"]
    assert components == {}
    assert parse_typed_components('"glucose", "ATP", "Hexokinase-1"') == []


def test_fixed_gene_mode_preserves_typed_complex_components():
    text = 'C: "RELA", "p65" (gene); "NFKB1", "p50" (gene)\nReason: complex\n'
    synonyms, types, _reason, components = parse_llm_response(text, EntityType.GENE)
    assert types["C"] == "complex"
    assert components["C"] == [
        ("gene", ["RELA", "p65"]),
        ("gene", ["NFKB1", "p50"]),
    ]
    assert synonyms["C"] == ["RELA", "p65", "NFKB1", "p50"]


def _fake_search(species_list, synonyms_dict, database, method, top_k, tax_id=None, model_info=None, model_type=None):
    sid = species_list[0]
    names = list(synonyms_dict[sid])
    return [Recommendation(
        id=sid,
        synonyms=names,
        candidates=[f"{database}:{names[0]}"],
        candidate_names=[database],
        match_score=[1.0],
    )]


def test_complex_routes_each_component_to_one_db():
    with patch("core.annotation_workflow._search_one_database", side_effect=_fake_search) as mock_search:
        recs, _species_db, cand_dbs = _search_databases(
            ["X"],
            {"X": ["RAS", "GTP", "RAF1"]},
            EntityType.AUTO,
            [DatabaseID.CHEBI, DatabaseID.UNIPROT],
            "direct",
            3,
            entity_type_dict={"X": "complex"},
            component_dict={"X": [
                ("protein", ["RAS"]),
                ("chemical", ["GTP"]),
                ("protein", ["RAF1"]),
            ]},
        )
    dbs = [call.args[2] for call in mock_search.call_args_list]
    name_lists = [list(call.args[1]["X"]) for call in mock_search.call_args_list]
    assert dbs == ["uniprot", "chebi", "uniprot"]
    assert name_lists == [["RAS"], ["GTP"], ["RAF1"]]
    assert recs[0].candidates == ["uniprot:RAS", "chebi:GTP", "uniprot:RAF1"]
    assert recs[0].component_ids == ["component_1", "component_2", "component_3"]
    assert recs[0].component_names == ["RAS", "GTP", "RAF1"]
    assert recs[0].component_types == ["protein", "chemical", "protein"]
    assert cand_dbs[("X", "chebi:GTP")] == "chebi"
    assert cand_dbs[("X", "uniprot:RAS")] == "uniprot"


def test_untyped_complex_still_searches_all_dbs():
    with patch("core.annotation_workflow._search_one_database", side_effect=_fake_search) as mock_search:
        _search_databases(
            ["Ras_Raf1"],
            {"Ras_Raf1": ["RAS", "RAF1"]},
            EntityType.AUTO,
            [DatabaseID.CHEBI, DatabaseID.UNIPROT],
            "direct",
            3,
            entity_type_dict={"Ras_Raf1": "complex"},
        )
    dbs = [call.args[2] for call in mock_search.call_args_list]
    assert dbs == ["chebi", "uniprot"]


def test_rank_complex_species_per_component_and_keeps_taxon_accessions():
    df = pd.DataFrame([
        {"id": "c1", "type": "complex", "display_name": "Ras_Raf1",
         "component_id": "component_1", "component_name": "RAS", "component_type": "gene",
         "annotation": "NCBIGENE:1", "annotation_label": "HRAS", "identity": "HRAS",
         "identity_rank": 1, "tax_id": "9606"},
        {"id": "c1", "type": "complex", "display_name": "Ras_Raf1",
         "component_id": "component_1", "component_name": "RAS", "component_type": "gene",
         "annotation": "NCBIGENE:2", "annotation_label": "Hras", "identity": "HRAS",
         "identity_rank": 1, "tax_id": "10090"},
        {"id": "c1", "type": "complex", "display_name": "Ras_Raf1",
         "component_id": "component_1", "component_name": "RAS", "component_type": "gene",
         "annotation": "NCBIGENE:3", "annotation_label": "KRAS", "identity": "KRAS",
         "identity_rank": 2, "tax_id": "9606"},
        {"id": "c1", "type": "complex", "display_name": "Ras_Raf1",
         "component_id": "component_2", "component_name": "RAF1", "component_type": "gene",
         "annotation": "NCBIGENE:4", "annotation_label": "RAF1", "identity": "RAF1",
         "identity_rank": 1, "tax_id": "9606"},
        {"id": "c1", "type": "complex", "display_name": "Ras_Raf1",
         "component_id": "component_2", "component_name": "RAF1", "component_type": "gene",
         "annotation": "NCBIGENE:5", "annotation_label": "BRAF", "identity": "BRAF",
         "identity_rank": 2, "tax_id": "9606"},
    ])
    response = "c1|component_1: NCBIGENE:1\nc1|component_2: NCBIGENE:4"
    with patch("core.annotation_workflow.query_llm", return_value=response) as mock_llm:
        out = rank_species_annotations_with_llm("dummy.xml", df, n_return=1)
    mock_llm.assert_called_once()
    assert list(out["annotation"]) == ["NCBIGENE:1", "NCBIGENE:2", "NCBIGENE:4"]
    assert list(out["component_id"]) == ["component_1", "component_1", "component_2"]
    assert list(out["identity_rank"]) == [1, 1, 1]


def test_expand_composite_gene_synonyms():
    expanded = expand_lookup_synonyms(
        ["MAP2K1 (MEK1), MAP2K2 (MEK2)", "p21 mRNA (CDKN1A transcript)"],
        "ncbigene",
    )
    for expected in ("MAP2K1", "MEK1", "MAP2K2", "MEK2", "p21", "CDKN1A"):
        assert expected in expanded


def test_direct_top_k_limits_identity_before_taxon(monkeypatch):
    names = {
        "9606": {"raf1": ["h_raf", "h_wrong"]},
        "10090": {"raf1": ["m_raf", "m_wrong"]},
        "10116": {"raf1": ["r_raf"]},
    }
    labels = {
        "h_raf": "RAF1", "m_raf": "Raf1", "r_raf": "Raf1",
        "h_wrong": "RNASE3", "m_wrong": "Ear1",
    }
    monkeypatch.setattr(database_search, "load_ncbigene_names_dict", lambda tax_id=None: names[str(tax_id)])
    monkeypatch.setattr(database_search, "load_ncbigene_label_dict", lambda: labels)
    database_search._NORMALIZED_TAXON_NAMES_CACHE.clear()
    rec = get_species_recommendations_direct(
        ["x"], {"x": ["RAF1"]}, database="ncbigene",
        tax_id=["9606", "10090", "10116"], top_k=1,
    )[0]
    assert rec.candidates == ["h_raf", "m_raf", "r_raf"]
    assert rec.candidate_taxa == ["9606", "10090", "10116"]
    assert rec.candidate_identity_ranks == [1, 1, 1]


def test_similar_explicit_gene_symbols_are_not_merged(monkeypatch):
    names = {
        "9606": {"pdpk1": ["h_pdpk1"], "pdk1": ["h_pdk1"]},
        "10090": {"pdpk1": ["m_pdpk1"], "pdk1": ["m_pdk1"]},
    }
    labels = {
        "h_pdpk1": "PDPK1", "m_pdpk1": "Pdpk1",
        "h_pdk1": "PDK1", "m_pdk1": "Pdk1",
    }
    monkeypatch.setattr(database_search, "load_ncbigene_names_dict", lambda tax_id=None: names[str(tax_id)])
    monkeypatch.setattr(database_search, "load_ncbigene_label_dict", lambda: labels)
    database_search._NORMALIZED_TAXON_NAMES_CACHE.clear()
    rec = get_species_recommendations_direct(
        ["x"], {"x": ["PDPK1", "PDK1"]}, database="ncbigene",
        tax_id=["9606", "10090"], top_k=2,
    )[0]
    groups = {}
    for candidate, identity_rank in zip(rec.candidates, rec.candidate_identity_ranks):
        groups.setdefault(identity_rank, []).append(candidate)
    assert groups == {1: ["h_pdpk1", "m_pdpk1"], 2: ["h_pdk1", "m_pdk1"]}
