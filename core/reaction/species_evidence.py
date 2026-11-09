"""Reaction-local SBML queries. See docs/reaction_species_contract.md."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import libsbml
import pandas as pd

CONTRACT_VERSION = "reaction-species-evidence-v1"


def text(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def compound_id(value):
    value = unquote(text(value)).upper()
    match = re.search(r"(?:^|[/#:])CHEBI[:_](\d+)$", value)
    if match:
        return "CHEBI:" + match[1]
    match = re.search(r"(?:^|[/#:])(?:KEGG(?:\.COMPOUND)?[:/])?(C\d{5})$", value)
    return match[1] if match else None


def load_model(path):
    document = libsbml.readSBMLFromFile(str(path))
    errors = [document.getError(i).getMessage() for i in range(document.getNumErrors())
              if document.getError(i).getSeverity() >= libsbml.LIBSBML_SEV_ERROR]
    if document.getModel() is None or errors:
        raise ValueError("Invalid/unsupported SBML: " + "; ".join(errors))
    return document


def species_evidence(model, recommendations=None):
    """Normalize alternatives once per species; never modify input tables/SBML."""
    species = {s.getId(): s for s in model.getListOfSpecies()}
    if isinstance(recommendations, (str, Path)):
        recommendations = pd.read_csv(recommendations, dtype=str).fillna("")
    if recommendations is not None and not {"id", "annotation"} <= set(recommendations):
        raise ValueError("Species table needs id and annotation columns")
    supplied = set() if recommendations is None else set(recommendations.id.map(text)) - {"Reason:", ""}
    if supplied - set(species):
        raise ValueError("Species table contains IDs absent from SBML: " + str(sorted(supplied - set(species))))
    records = {sid: [] for sid in species}
    if recommendations is not None:
        legacy = "identity_status" not in recommendations.columns
        for row in recommendations.to_dict("records"):
            sid = text(row.get("id"))
            if sid not in species:
                continue
            records[sid].append({"identifier": compound_id(row.get("annotation")),
                "raw_annotation": text(row.get("annotation")),
                "identity_status": "legacy_unassessed" if legacy else text(row.get("identity_status")).lower() or "missing",
                "qualifier": text(row.get("qualifier")).removeprefix("bqbiol:") or "is",
                "component_id": text(row.get("component_id")),
                "qualifier_status": text(row.get("qualifier_status")).lower(), "source": "recommendation"})
    for sid, item in species.items():
        if sid in supplied:
            continue
        for term in item.getCVTerms() or []:
            qualifier = libsbml.BiolQualifierType_toString(term.getBiologicalQualifierType())
            for i in range(term.getNumResources()):
                uri = term.getResourceURI(i)
                records[sid].append({"identifier": compound_id(uri), "raw_annotation": uri,
                    "identity_status": "legacy_unassessed", "qualifier": qualifier,
                    "component_id": "", "qualifier_status": "retained", "source": "SBML"})
    output = {}
    for sid, rows in records.items():
        allowed, veto = set(), set()
        for row in rows:
            identifier = row["identifier"]
            if row["component_id"] or row["qualifier"] == "hasPart":
                scope = "component_relationship"
            elif row["qualifier"] != "is":
                scope = "non_identity_relationship"
            else:
                scope = "whole_species_identity"
            row["assertion_scope"] = scope
            if scope != "whole_species_identity":
                reason = scope
            elif identifier is None:
                reason = "unrecognized_identifier"
            elif row["identity_status"] not in {"supported", "legacy_unassessed"}:
                reason = "unsupported_identity_status"
            elif row["qualifier_status"] == "abstain":
                reason = "qualifier_abstention"
            else:
                reason = None
            row["eligible"] = reason is None
            row["exclusion_reason"] = reason
            # Relationships describe a different assertion, not a negative
            # whole-species identity judgment about the same URI.
            row["identity_veto"] = scope == "whole_species_identity" and identifier is not None and not row["eligible"]
            if row["eligible"]:
                allowed.add(identifier)
            elif row["identity_veto"]:
                veto.add(identifier)
        usable = sorted(allowed - veto)
        for row in rows:
            row["included"] = row["eligible"] and row["identifier"] not in veto
            row["usable"] = row["included"]
            if row["eligible"] and not row["included"]:
                row["exclusion_reason"] = "identity_conflict"
        output[sid] = {"name": species[sid].getName() or sid,
            "chebi": [x for x in usable if x.startswith("CHEBI:")],
            "kegg_compound": [x for x in usable if x.startswith("C") and not x.startswith("CHEBI:")],
            "conflicting_identifiers": sorted(allowed & veto), "records": rows}
    return output


def build_queries(model_path, recommendations=None):
    from .retrieval import query_text
    document = load_model(model_path)
    model = document.getModel()
    species = species_evidence(model, recommendations)
    rows = []
    for reaction in model.getListOfReactions():
        def side(refs):
            return " + ".join(("" if abs(ref.getStoichiometry() - 1) < 1e-12
                               else f"{ref.getStoichiometry():g} ") + ref.getSpecies() for ref in refs)
        lhs, rhs = side(reaction.getListOfReactants()), side(reaction.getListOfProducts())
        equation = f"{lhs} {'<=>' if reaction.getReversible() else '=>'} {rhs}"
        ids = list(dict.fromkeys(ref.getSpecies() for ref in
            list(reaction.getListOfReactants()) + list(reaction.getListOfProducts())))
        blocks = []
        for sid in ids:
            ev = species[sid]
            parts = [f"species={sid}"]
            if ev["chebi"]:
                parts.append("ChEBI=" + ",".join(ev["chebi"]))
            if ev["kegg_compound"]:
                parts.append("KEGG-compound=" + ",".join(ev["kegg_compound"]))
            blocks.append(f"{ev['name']} [{'; '.join(parts)}]")
        rows.append({"reaction_id": reaction.getId(), "display_name": reaction.getName(),
            "reaction_equation": equation,
            "query": query_text({"reaction_equation": equation, "participant_evidence": "; ".join(blocks)}),
            "species_evidence": {sid: species[sid] for sid in ids}})
    return sorted(rows, key=lambda row: row["reaction_id"])
