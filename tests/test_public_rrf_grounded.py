"""Offline public API contract tests; provider transport is always blocked/mocked."""
import json
from pathlib import Path

import libsbml
import numpy as np
import pandas as pd
import pytest
import httpx

import core.annotation_workflow as workflow
from core import AnnotationResult, annotate_model, write_reaction_annotations
from core.reaction import grounded, retrieval
from core.reaction.species_evidence import build_queries, compound_id, load_model, species_evidence

HTTPX_SEND = httpx.Client.send
EXECUTE_PAYLOAD = grounded.execute_payload


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import httpx
    import socket
    def blocked(*args, **kwargs):
        raise AssertionError("Network forbidden in offline integration tests")
    monkeypatch.setattr(httpx.Client, "send", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(grounded, "execute_payload", blocked)


@pytest.fixture
def model_file(tmp_path):
    doc = libsbml.SBMLDocument(3, 1)
    model = doc.createModel()
    model.setId("previously_unseen")
    c = model.createCompartment()
    c.setId("cell")
    c.setConstant(True)
    for sid, name in [("a", "ATP"), ("b", "ADP"), ("unmapped", "unknown mixture")]:
        s = model.createSpecies()
        s.setId(sid)
        s.setName(name)
        s.setCompartment("cell")
        s.setBoundaryCondition(False)
        s.setHasOnlySubstanceUnits(False)
        s.setConstant(False)
        s.setMetaId("meta_" + sid)
    for rid, reactants, products in [("first", [("a", 2)], [("b", 1)]),
                                      ("formerly_omitted", [("unmapped", 1)], []),
                                      ("boundary", [], [("a", 1)])]:
        r = model.createReaction()
        r.setId(rid)
        r.setName("Display only R12345")
        r.setReversible(False)
        r.setFast(False)
        for refs, create in [(reactants, r.createReactant), (products, r.createProduct)]:
            for sid, stoich in refs:
                ref = create()
                ref.setSpecies(sid)
                ref.setStoichiometry(stoich)
                ref.setConstant(True)
    path = tmp_path / "unseen.xml"
    libsbml.writeSBMLToFile(doc, str(path))
    return path


class TinyRetriever:
    """Actual runtime BM25/RRF/evidence construction over an arbitrary tiny catalog."""
    provenance = {"resource": "synthetic", "verified_selected_checkpoint": False}
    catalog = {f"R{i:05d}": {"NAME": "ATP" if i == 1 else f"name{i}",
                "DEFINITION": "ATP hydrolysis", "EQUATION": "ATP => ADP"} for i in range(1, 13)}

    def retrieve(self, query):
        docs = [{"kegg_id": kid, "text": retrieval.document_text(kid, fields)} for kid, fields in self.catalog.items()]
        return retrieval.evidence_for(self.catalog, retrieval.BM25(docs).rank(query), list(self.catalog)[::-1])


def valid_answer(abstain=False):
    evidence = TinyRetriever().retrieve("ATP")
    return {"abstain": abstain, "predicted_kegg_id": None if abstain else evidence[0]["kegg_id"],
            "selected_evidence_rank": None if abstain else 1,
            "supporting_evidence_ids": [] if abstain else ["E01"], "confidence": .5,
            "reasoning_summary": "Synthetic evidence judgment", "abstention_reason": "Insufficient evidence" if abstain else None}


def response(answer=None, **overrides):
    return {"status": "completed", "model": "gpt-6.1-terra", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": json.dumps(answer or valid_answer())}]}],
        "usage": {"input_tokens": 200, "output_tokens": 100, "output_tokens_details": {"reasoning_tokens": 50}}, **overrides}


def run(model_file, **kwargs):
    return annotate_model(str(model_file), annotate="reactions", reaction_method="rrf_grounded",
        llm_model="gpt-6.1-terra", reaction_retriever=TinyRetriever(), **kwargs)


def test_arbitrary_model_all_reactions_names_no_identifiers(model_file):
    queries = build_queries(model_file)
    assert [q["reaction_id"] for q in queries] == ["boundary", "first", "formerly_omitted"]
    assert "2 a => b" in queries[1]["query"]
    assert "unknown mixture" in queries[2]["query"]
    assert all("R12345" not in q["query"] for q in queries)
    assert all(len(TinyRetriever().retrieve(q["query"])) == 10 for q in queries)


@pytest.mark.parametrize("value,expected", [("CHEBI:123", "CHEBI:123"), ("https://identifiers.org/chebi/CHEBI:123", "CHEBI:123"),
    ("KEGG:C00001", "C00001"), ("https://identifiers.org/kegg.compound/C00001", "C00001"),
    ("KEGG:R00001", None), ("UNIPROT:P12345", None)])
def test_compound_identifier_formats(value, expected):
    assert compound_id(value) == expected


@pytest.mark.parametrize("status,qualifier,component,expected", [
    ("supported", "is", "", True), ("unresolved", "is", "", False),
    ("not_applicable", "is", "", False), ("", "is", "", False), ("unexpected", "is", "", False),
    ("supported", "hasPart", "", False), ("supported", "is", "component", False),
    ("supported", "isVersionOf", "", False)])
def test_species_status_qualifier_contract(model_file, status, qualifier, component, expected):
    frame = pd.DataFrame([{"id": "a", "annotation": "CHEBI:123", "identity_status": status,
                           "qualifier": qualifier, "component_id": component}])
    queries = build_queries(model_file, frame)
    assert ("ChEBI=CHEBI:123" in queries[1]["query"]) == expected
    assert "ATP" in queries[1]["query"]
    assert queries[1]["species_evidence"]["a"]["records"][0]["identity_status"] != "supported" or status == "supported"


def test_ties_alternatives_conflicts_legacy_and_direct_compound(model_file):
    frame = pd.DataFrame([{"id": "a", "annotation": x} for x in ["CHEBI:123", "CHEBI:123", "CHEBI:456", "KEGG:C00001"]])
    query = build_queries(model_file, frame)[1]
    assert query["query"].count("CHEBI:123") == 1
    assert "ChEBI=CHEBI:123,CHEBI:456" in query["query"] and "KEGG-compound=C00001" in query["query"]
    assert query["species_evidence"]["a"]["records"][0]["identity_status"] == "legacy_unassessed"
    frame["identity_status"] = ["supported", "unresolved", "supported", "supported"]
    query = build_queries(model_file, frame)[1]
    assert "CHEBI:123" not in query["query"]
    assert query["species_evidence"]["a"]["conflicting_identifiers"] == ["CHEBI:123"]


@pytest.mark.parametrize("qualifier,component", [("hasPart", ""), ("isVersionOf", ""),
    ("isDescribedBy", ""), ("is", "component")])
@pytest.mark.parametrize("legacy", [False, True])
def test_identity_survives_same_identifier_relationship_rows(model_file, qualifier, component, legacy):
    rows = [{"id": "a", "annotation": "CHEBI:15422", "qualifier": "is", "component_id": ""},
            {"id": "a", "annotation": "CHEBI:15422", "qualifier": qualifier, "component_id": component}]
    if not legacy:
        for row in rows:
            row["identity_status"] = "supported"
    queries = []
    for ordered in [rows, rows[::-1]]:
        query = build_queries(model_file, pd.DataFrame(ordered))[1]
        queries.append(query["query"])
        ev = query["species_evidence"]["a"]
        assert ev["chebi"] == ["CHEBI:15422"]
        assert ev["conflicting_identifiers"] == []
        identity = next(r for r in ev["records"] if r["qualifier"] == "is" and not r["component_id"])
        relation = next(r for r in ev["records"] if r is not identity)
        assert identity["eligible"] and identity["included"] and identity["usable"]
        assert identity["assertion_scope"] == "whole_species_identity"
        assert not relation["eligible"] and not relation["included"] and not relation["identity_veto"]
        assert relation["raw_annotation"] == "CHEBI:15422" and relation["exclusion_reason"]
        assert relation["identity_status"] == ("legacy_unassessed" if legacy else "supported")
    assert queries[0] == queries[1]


@pytest.mark.parametrize("status", ["unresolved", "", None, "not_applicable", "unknown"])
def test_actual_identity_conflicts_veto_independent_of_order(model_file, status):
    rows = [{"id": "a", "annotation": "CHEBI:15422", "qualifier": "is", "identity_status": "supported"},
            {"id": "a", "annotation": "CHEBI:15422", "qualifier": "", "identity_status": status}]
    for ordered in [rows, rows[::-1]]:
        ev = build_queries(model_file, pd.DataFrame(ordered))[1]["species_evidence"]["a"]
        assert ev["chebi"] == [] and ev["conflicting_identifiers"] == ["CHEBI:15422"]
        allowed = next(r for r in ev["records"] if r["identity_status"] == "supported")
        rejected = next(r for r in ev["records"] if r is not allowed)
        assert allowed["eligible"] and not allowed["included"] and not allowed["usable"]
        assert allowed["exclusion_reason"] == "identity_conflict"
        assert rejected["identity_veto"] and not rejected["eligible"]


def test_qualifier_abstention_on_whole_identity_still_vetoes(model_file):
    rows = [{"id": "a", "annotation": "CHEBI:15422", "identity_status": "supported", "qualifier": "is"},
            {"id": "a", "annotation": "CHEBI:15422", "identity_status": "supported", "qualifier": "is", "qualifier_status": "abstain"}]
    for ordered in [rows, rows[::-1]]:
        ev = build_queries(model_file, pd.DataFrame(ordered))[1]["species_evidence"]["a"]
        assert ev["chebi"] == [] and ev["conflicting_identifiers"] == ["CHEBI:15422"]
        assert any(r["identity_veto"] and r["exclusion_reason"] == "qualifier_abstention" for r in ev["records"])


@pytest.mark.parametrize("status", ["supported", "unresolved", "", "unknown"])
def test_component_and_relationship_only_rows_never_become_identity_or_veto(model_file, status):
    rows = [{"id": "a", "annotation": "CHEBI:15422", "identity_status": status, "qualifier": "hasPart"},
            {"id": "a", "annotation": "CHEBI:15422", "identity_status": status, "qualifier": "is", "component_id": "part"}]
    for ordered in [rows, rows[::-1]]:
        ev = build_queries(model_file, pd.DataFrame(ordered))[1]["species_evidence"]["a"]
        assert ev["chebi"] == [] and ev["conflicting_identifiers"] == []
        assert len(ev["records"]) == 2
        assert all(not r["eligible"] and not r["included"] and not r["identity_veto"] for r in ev["records"])
        # Even unresolved component statements cannot veto a separate whole identity.
        whole = {"id": "a", "annotation": "CHEBI:15422", "identity_status": "supported", "qualifier": "is"}
        assert build_queries(model_file, pd.DataFrame(ordered + [whole]))[1]["species_evidence"]["a"]["chebi"] == ["CHEBI:15422"]


def test_legacy_column_absence_differs_from_modern_missing_status(model_file):
    frame = pd.DataFrame([{"id": "a", "annotation": "CHEBI:15422", "qualifier": "is"}])
    legacy = build_queries(model_file, frame)[1]["species_evidence"]["a"]
    frame["identity_status"] = None
    modern = build_queries(model_file, frame)[1]["species_evidence"]["a"]
    assert legacy["chebi"] == ["CHEBI:15422"] and modern["chebi"] == []
    assert legacy["records"][0]["identity_status"] == "legacy_unassessed"
    assert modern["records"][0]["identity_status"] == "missing"
    assert modern["records"][0]["identity_veto"]


@pytest.mark.parametrize("qualifier", [libsbml.BQB_HAS_PART, libsbml.BQB_IS_VERSION_OF, libsbml.BQB_IS_DESCRIBED_BY])
@pytest.mark.parametrize("with_identity", [False, True])
def test_same_sbml_uri_under_multiple_qualifiers(model_file, qualifier, with_identity):
    doc = load_model(model_file)
    queries = []
    qualifiers = [libsbml.BQB_IS, qualifier] if with_identity else [qualifier]
    for order in [qualifiers, qualifiers[::-1]]:
        species = doc.getModel().getSpecies("a")
        # addCVTerm deduplicates the URI across qualifiers. Construct the RDF
        # directly, as upstream's updated annotation writer does, then reparse.
        blocks = "".join(
            f'<bqbiol:{name}><rdf:Bag><rdf:li rdf:resource="https://identifiers.org/chebi/CHEBI:15422"/>'
            f'</rdf:Bag></bqbiol:{name}>'
            for name in [libsbml.BiolQualifierType_toString(kind) for kind in order])
        assert species.setAnnotation(
            '<annotation><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
            'xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">'
            f'<rdf:Description rdf:about="#meta_a">{blocks}</rdf:Description>'
            '</rdf:RDF></annotation>') == 0
        assert libsbml.writeSBMLToFile(doc, str(model_file))
        query = build_queries(model_file)[1]
        queries.append(query["query"])
        ev = query["species_evidence"]["a"]
        assert ev["chebi"] == (["CHEBI:15422"] if with_identity else [])
        assert ev["conflicting_identifiers"] == []
        assert len(ev["records"]) == len(qualifiers)
        assert all(r["included"] == (r["qualifier"] == "is") for r in ev["records"])
        # An authoritative table still suppresses the old SBML identity.
        modern = pd.DataFrame([{"id": "a", "annotation": "CHEBI:15422", "identity_status": "unresolved"}])
        handoff = build_queries(model_file, modern)[1]["species_evidence"]["a"]
        assert handoff["chebi"] == [] and {r["source"] for r in handoff["records"]} == {"recommendation"}
    assert queries[0] == queries[1]


def test_sbml_evidence_qualifiers_and_unresolved_override(model_file):
    doc = load_model(model_file)
    species = doc.getModel().getSpecies("a")
    for qualifier, uri in [(libsbml.BQB_IS, "https://identifiers.org/kegg.compound/C00001"),
                           (libsbml.BQB_HAS_PART, "https://identifiers.org/chebi/CHEBI:123")]:
        term = libsbml.CVTerm(libsbml.BIOLOGICAL_QUALIFIER)
        term.setBiologicalQualifierType(qualifier)
        term.addResource(uri)
        species.addCVTerm(term)
    libsbml.writeSBMLToFile(doc, str(model_file))
    ev = build_queries(model_file)[1]["species_evidence"]["a"]
    assert ev["kegg_compound"] == ["C00001"] and ev["chebi"] == []
    frame = pd.DataFrame([{"id": "a", "annotation": "KEGG:C00001", "identity_status": "unresolved"}])
    assert build_queries(model_file, frame)[1]["species_evidence"]["a"]["kegg_compound"] == []


def test_csv_dataframe_equivalence(model_file, tmp_path):
    frame = pd.DataFrame([{"id": "a", "annotation": "CHEBI:123"}])
    path = tmp_path / "species.csv"
    frame.to_csv(path, index=False)
    assert build_queries(model_file, frame) == build_queries(model_file, path)
    result = run(model_file, species_recommendations_df=path, dry_run=True)
    pd.testing.assert_frame_equal(result.species_recommendations_df, frame)


@pytest.mark.parametrize("kind,expected", [("valid", "selected"), ("abstain", "abstained"),
    ("incomplete", "incomplete"), ("incomplete_message", "incomplete"), ("schema", "schema_invalid"), ("outside", "evidence_noncompliant"),
    ("wrong_rank", "evidence_noncompliant"), ("empty", "empty_response"), ("failed", "provider_error"),
    ("refusal", "refusal"), ("wrong_model", "provider_error"), ("exception", "provider_error")])
def test_response_outcomes_coverage_usage(model_file, kind, expected):
    attempts = []
    def transport(body):
        attempts.append(body)
        if kind == "exception":
            raise RuntimeError("synthetic provider failure")
        answer = valid_answer(kind == "abstain")
        if kind == "schema":
            answer["abstain"] = "false"
        if kind == "outside":
            answer["predicted_kegg_id"] = "R99999"
        if kind == "wrong_rank":
            answer["selected_evidence_rank"] = 10
        raw = response(answer)
        if kind in {"incomplete", "failed"}:
            raw["status"] = kind
        if kind == "incomplete_message":
            raw["output"][0]["status"] = "incomplete"
        if kind == "empty":
            raw["output"] = []
        if kind == "refusal":
            raw["output"][0]["content"] = [{"type": "refusal", "refusal": "no"}]
        if kind == "wrong_model":
            raw["model"] = "different-model"
        return raw
    result = run(model_file, reaction_transport=transport, max_completion_tokens=2048, reaction_reasoning_effort="medium")
    assert isinstance(result, AnnotationResult)
    assert len(result.recommendations_df) == len(attempts) == 3
    assert set(result.recommendations_df.status) == {expected}
    assert all(body["max_output_tokens"] == 2048 and body["reasoning"] == {"effort": "medium"} for body in attempts)
    if kind not in {"valid", "abstain"}:
        assert result.recommendations_df.biological_abstain.isna().all()
        assert result.recommendations_df.annotation.eq("").all()
    if kind != "exception":
        assert result.metrics["llm_usage"]["output_tokens"] == 300
        assert result.metrics["llm_usage"]["records"][0]["output_tokens_details"]["reasoning_tokens"] == 50
    assert json.loads(Path(str(model_file) + "_reaction_metrics.json").read_text())["coverage"]["total_reactions"] == 3
    assert len(Path(str(model_file) + "_reaction_evidence.jsonl").read_text().splitlines()) == 3


def test_dry_run_combined_no_provider_or_species_calls(model_file, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("species or provider inference attempted")
    monkeypatch.setattr(workflow, "get_species_recommendations_direct", forbidden)
    result = annotate_model(str(model_file), annotate="both", reaction_method="rrf_grounded", dry_run=True,
        llm_model="gpt-6.1-terra", reaction_retriever=TinyRetriever(), reaction_transport=forbidden)
    assert result.metrics["provider_attempts"] == 0
    assert result.metrics["species_stage"]["status"] == "skipped_dry_run"
    assert set(result.recommendations_df.status) == {"planned"}


def test_combined_passes_complete_species_table_not_chebi_filter(model_file, monkeypatch):
    real = workflow.annotate_single_model
    frame = pd.DataFrame([{"id": "a", "annotation": "KEGG:C00001", "identity_status": "supported"}])
    def species_stage(*args, **kwargs):
        assert kwargs["annotate"] == "species" and kwargs["predict_qualifiers"]
        result = AnnotationResult(frame, {"test": True}, model_file=str(model_file), conversation_history=[],
            entities_to_evaluate=["a"], entity_type="chemical", database="chebi", method="direct", llm_model="gpt-6.1-terra", top_k=3)
        result.species_recommendations_df = frame
        return result
    monkeypatch.setattr(workflow, "annotate_single_model", species_stage)
    result = real(str(model_file), annotate="both", reaction_method="rrf_grounded", predict_qualifiers=True,
        llm_model="gpt-6.1-terra", reaction_retriever=TinyRetriever(), reaction_transport=lambda body: response())
    assert result.species_recommendations_df is frame
    assert "KEGG-compound=C00001" in result.reaction_evidence[1]["query"]
    saved_metrics = json.loads(Path(str(model_file) + "_reaction_metrics.json").read_text())
    assert saved_metrics["species_stage"] == {"test": True}


def test_combined_missing_resources_prevents_species_spending(model_file, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("species stage must not run when reaction resources are missing")
    monkeypatch.setattr(workflow, "get_species_recommendations_direct", forbidden)
    result = annotate_model(str(model_file), annotate="both", reaction_method="rrf_grounded")
    assert set(result.recommendations_df.status) == {"resource_error"}
    assert result.metrics["species_stage"]["status"] == "skipped_resource_error"
    assert result.metrics["provider_attempts"] == 0


def test_resource_and_input_failure_accounting(model_file):
    result = annotate_model(str(model_file), annotate="reactions", reaction_method="rrf_grounded", dry_run=True)
    assert set(result.recommendations_df.status) == {"resource_error"}
    result = run(model_file, species_recommendations_df=pd.DataFrame([{"id": "absent", "annotation": "CHEBI:1"}]))
    assert set(result.recommendations_df.status) == {"input_error"}
    assert result.metrics["provider_attempts"] == 0


def test_rulebased_default_and_explicit_option_keep_existing_dispatch(model_file, monkeypatch):
    calls = []
    def old_loader(model, species):
        calls.append(model)
        return pd.DataFrame()
    monkeypatch.setattr(workflow, "_load_species_recommendations", old_loader)
    monkeypatch.setattr(grounded, "annotate_grounded", lambda *a, **k: pytest.fail("grounded path entered for baseline"))
    for options in [{}, {"reaction_method": "rulebased"}]:
        frame, metrics = annotate_model(str(model_file), annotate="reactions", **options)
        assert frame.empty and "No species ChEBI annotations" in metrics["error"]
    assert len(calls) == 2


def test_payload_identity_includes_every_provider_input_excludes_sidecar():
    evidence = TinyRetriever().retrieve("ATP")
    body = grounded.provider_payload("ATP", evidence)
    assert "provenance" not in body["input"]
    changed = [{**e, "provenance": {"path": "different", "sha256": "actual"}} for e in evidence]
    assert body == grounded.provider_payload("ATP", changed)
    for options in [{"max_output_tokens": 512}, {"model": "gpt-5.6"}, {"reasoning_effort": "high"}]:
        assert grounded.digest(body) != grounded.digest(grounded.provider_payload("ATP", evidence, **options))
    changed[0]["fused_score"] += 1e-12
    assert grounded.digest(body) != grounded.digest(grounded.provider_payload("ATP", changed))


def test_deterministic_fusion_matches_selected_implementation():
    from benchmark.scripts import phase3_retrieval, phase3b_release
    docs = [{"kegg_id": k, "text": retrieval.document_text(k, fields)} for k, fields in TinyRetriever.catalog.items()]
    for q in ["ATP", "2 ATP + water => ADP", "unseen", ""]:
        assert retrieval.BM25(docs).rank(q) == phase3_retrieval.BM25(docs).rank(q)
        assert retrieval.tokenise(q) == phase3_retrieval.tokenise(q)
    components = [list(TinyRetriever.catalog)[:5], list(TinyRetriever.catalog)[::-1][:7]]
    assert retrieval.fusion(components, list(TinyRetriever.catalog)) == phase3b_release.reciprocal_rank_fusion_complete(components, list(TinyRetriever.catalog))
    with pytest.raises(ValueError):
        retrieval.fusion([["R99999"]], list(TinyRetriever.catalog))


def test_missing_and_tampered_resources_never_download(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        retrieval.verify_model_dir(tmp_path)
    invalid = tmp_path / "bad.zip"
    invalid.write_bytes(b"not a selected checkpoint")
    with pytest.raises(ValueError):
        retrieval.install_resources(invalid, tmp_path / "install")
    assert not (tmp_path / "install").exists()


def test_gpu_unavailable_is_explicit_error(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA requested but unavailable"):
        retrieval.ReactionRetriever(retrieval.RetrievalResources("missing", "missing", device="cuda"))


def test_retriever_instance_cache(monkeypatch):
    retrieval.load_retriever.cache_clear()
    calls = []
    monkeypatch.setattr(retrieval, "ReactionRetriever", lambda resource: calls.append(resource) or object())
    resource = retrieval.RetrievalResources("catalog", "encoder")
    assert retrieval.load_retriever(resource) is retrieval.load_retriever(resource)
    assert len(calls) == 1
    retrieval.load_retriever.cache_clear()


def test_writer_preserves_species_and_reaction_qualifiers(model_file, tmp_path):
    before = model_file.read_bytes()
    result = run(model_file, reaction_transport=lambda body: response())
    assert model_file.read_bytes() == before
    doc = load_model(model_file)
    reaction = doc.getModel().getReaction("first")
    reaction.setMetaId("original_meta")
    term = libsbml.CVTerm(libsbml.BIOLOGICAL_QUALIFIER)
    term.setBiologicalQualifierType(libsbml.BQB_IS_DESCRIBED_BY)
    term.addResource("https://identifiers.org/pubmed/123")
    reaction.addCVTerm(term)
    term = libsbml.CVTerm(libsbml.BIOLOGICAL_QUALIFIER)
    term.setBiologicalQualifierType(libsbml.BQB_HAS_PART)
    term.addResource("https://identifiers.org/chebi/CHEBI:123")
    doc.getModel().getSpecies("a").addCVTerm(term)
    species_xml = doc.getModel().getSpecies("a").toSBML()
    libsbml.writeSBMLToFile(doc, str(model_file))
    rows = result.recommendations_df.copy()
    rows.loc[rows.id.eq("first"), "update_annotation"] = "add"
    output = tmp_path / "reviewed.xml"
    write_reaction_annotations(model_file, rows, output)
    parsed = load_model(output).getModel().clone()
    rxn = parsed.getReaction("first")
    terms = {term.getBiologicalQualifierType(): [term.getResourceURI(i) for i in range(term.getNumResources())] for term in rxn.getCVTerms()}
    assert libsbml.BQB_IS_DESCRIBED_BY in terms and libsbml.BQB_IS in terms
    assert parsed.getSpecies("a").toSBML() == species_xml
    assert not parsed.getSpecies("b").getNumCVTerms()
    assert not parsed.getReaction("boundary").getNumCVTerms()
    assert rxn.getMetaId() == "original_meta"
    with pytest.raises(ValueError):
        write_reaction_annotations(model_file, rows, model_file)
    with pytest.raises(ValueError):
        bad = rows.copy()
        bad.loc[bad.id.eq("first"), "id"] = "a"
        write_reaction_annotations(model_file, bad, tmp_path / "bad.xml")
    with pytest.raises(ValueError, match="new evidence-validated run"):
        result.revise("change reaction")


def test_runtime_has_no_benchmark_imports():
    import ast
    for path in Path(retrieval.__file__).parent.glob("*.py"):
        if path.name not in {"retrieval.py", "grounded.py", "species_evidence.py", "writer.py"}:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("benchmark")
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("benchmark") for alias in node.names)


def test_frozen_validation_components_fusion_parity():
    """Only frozen validation queries/rankings; no split, labels or test data."""
    import lzma
    from benchmark.scripts import phase3b_release as release
    bm = {(r["model_id"], r["reaction_id"]): r["ranked_ids"] for r in
          map(json.loads, release.BM25_RANKINGS.read_text().splitlines())}
    dense = {(r["model_id"], r["reaction_id"]): r["ranked_ids"] for r in
             map(json.loads, release.SELECTED_RANKINGS.read_text().splitlines())}
    with lzma.open(release.FUSION_OUT / release.FUSION_RANKING_NAME, "rt") as handle:
        frozen = [json.loads(line) for line in handle if line.strip()]
    assert len(frozen) == 969
    for row in frozen:
        key = row["model_id"], row["reaction_id"]
        assert row["split"] == "validation"
        assert retrieval.fusion([bm[key], dense[key]], row["ranked_ids"]) == row["ranked_ids"]


def test_deep_payload_diff_detects_order_settings_and_numerical_drift():
    from scripts.verify_rrf_grounded import differences
    before = {"model": "gpt-6.1-terra", "evidence": [{"score": .01}, {"score": .02}]}
    after = {"model": "gpt-5.6", "evidence": [{"score": .02}, {"score": .01}]}
    assert {r["field"] for r in differences(before, after)} == {"/model", "/evidence/0/score", "/evidence/1/score"}


def test_verified_encoder_cpu_fixture_when_installed():
    """Optional local resource test. No download; six tiny texts, not benchmark inference."""
    import os
    directory = os.environ.get("AAAIM_TEST_ENCODER_DIR")
    if not directory:
        pytest.skip("Set AAAIM_TEST_ENCODER_DIR to an explicitly installed selected archive")
    retrieval.verify_model_dir(directory)
    model, tokenizer = retrieval.load_encoder(directory, "cpu")
    obj = object.__new__(retrieval.ReactionRetriever)
    obj.resources = retrieval.RetrievalResources("unused", directory, device="cpu", batch_size=4)
    obj.model, obj.tokenizer = model, tokenizer
    fixture = json.loads((Path(directory) / "inference_fixture.json").read_text())
    q = obj.encode(fixture["queries"])
    d = obj.encode([x["text"] for x in fixture["documents"]])
    assert q.dtype == np.float32 and np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-6)
    ids = [x["id"] for x in fixture["documents"]]
    ranks = [[ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float((d @ query)[i]), ids[i]))] for query in q]
    assert ranks == fixture["expected_rankings"]
    assert retrieval.load_encoder(directory, "cpu")[0] is model


def test_actual_sdk_transport_sends_complete_planned_body_without_retries(monkeypatch):
    import openai
    evidence = TinyRetriever().retrieve("ATP")
    expected = grounded.provider_payload("ATP", evidence)
    sent = []
    def handler(request):
        assert request.method == "POST" and request.url.path == "/v1/responses"
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "resp_synthetic", "created_at": 0,
            "object": "response", **response()})
    monkeypatch.setattr(httpx.Client, "send", HTTPX_SEND)
    client = openai.OpenAI(api_key="synthetic-offline-key", max_retries=0,
                          http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    def factory(**kwargs):
        assert kwargs == {"max_retries": 0}
        return client
    monkeypatch.setattr(openai, "OpenAI", factory)
    try:
        actual = EXECUTE_PAYLOAD(expected)
        assert len(sent) == 1 and sent[0] == expected
        assert actual["usage"]["output_tokens"] == 100
    finally:
        client.close()


def test_full_catalog_cpu_public_dry_run_when_requested(model_file):
    """Opt-in actual selected retriever on synthetic unseen SBML; no benchmark data."""
    import os
    import torch
    if os.environ.get("AAAIM_TEST_FULL_CATALOG") != "1":
        pytest.skip("Explicit full-catalog CPU smoke test not requested")
    directory = os.environ["AAAIM_TEST_ENCODER_DIR"]
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        resources = retrieval.RetrievalResources(str(Path(__file__).resolve().parents[1] / "data/kegg/kegg_reaction_features.lzma"), directory)
        result = annotate_model(str(model_file), annotate="reactions", reaction_method="rrf_grounded",
                                reaction_resources=resources, dry_run=True)
        assert result.metrics["provider_attempts"] == 0
        assert result.metrics["coverage"]["status_counts"] == {"planned": 3}
        assert all(len(a["evidence"]) == 10 and a["retrieval_provenance"]["catalog_sha256"] == retrieval.CATALOG_SHA256
                   for a in result.reaction_evidence)
        assert len(retrieval.load_retriever(resources).catalog) == 12312
    finally:
        torch.set_num_threads(threads)
