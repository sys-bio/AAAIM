"""Build the deterministic offline human-review interface for Prompt 3.

Only frozen validation artifacts and the explicitly nominated non-test training
reaction are materialized.  Pass 1 is built from a strict source-side allowlist
and is physically separated from every system-output artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import lzma
import os
import pickle
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import libsbml

from benchmark.scripts.phase3_common import PHASE3_DIR, REPO_ROOT, atomic_write_json, sha256_portable, write_artifact_manifest
from benchmark.scripts.phase3_error_inventory import parse_multilabel_kegg_ids

ERROR_AUDIT = PHASE3_DIR / "error_audit"
OUT = ERROR_AUDIT / "reviewer"
ALGORITHM_VERSION = "phase3-error-audit-reviewer-v1"
REVIEW_SCHEMA_VERSION = "aaaim-human-review-v1"
PROMPT2_COMMIT = "3a28d3aaa88e6281e047ee7aef73b43c75a8efc2"
CATALOG_DIGEST = "00acb4de1bbfb1ddae0298a09d218f93eb22241844179439ff34a2912fba046c"

SOURCE_DIGESTS = {
    "benchmark/phase3/error_audit/all_validation_outcomes.csv": "14a70bc7144ed47d08e2e2dafe74ae286e792e00758a1424003c560de43c4eb6",
    "benchmark/phase3/error_audit/phase3c_noncorrect_review.csv": "a2fd3e4016112ebc8bc11e13903df49e9f49b6313695a9457d867b69bf846b7c",
    "benchmark/phase3/error_audit/formal_audit_sample.csv": "53622e877f65bae66bfaf58698a706963cd37322e07e8f7c865b392bc6e2ef89",
    "benchmark/phase3/error_audit/formal_audit_crosswalk.csv": "36cfc74244c100372022bb0889b416c80b8057ff46175811e9ab4c39fd0bdb7a",
    "benchmark/phase3/error_audit/formal_audit_blinded_order.csv": "a89c2f5bba19dbd4b1725a6bddd8013338dea9c217e96095ba2f4641b2c9dfea",
    "benchmark/phase3/error_audit/sampling_config.json": "6310381286139c21817c1756bba6af0ba301d514c31cfeec7d44c886ca0ead18",
    "benchmark/phase3/error_audit/sampling_distribution.json": "b345eb54bfc429f4518a8b638f59b1274c90d6e7bf5763632feaf1071539d147",
    "benchmark/phase3/phase3c_validation/frozen_evidence.jsonl": "b6045e9a77c22c884642b567dd8f011a76ad4f943b2fa3e33b96f87fcf490a53",
    "benchmark/phase3/retrieval_baselines/rankings_phase2_rule_based.jsonl": "e22954bb0b8cee75ff17511d96fc42b923217003388f3e4e4a7e03200ef31c0d",
    "benchmark/phase3/retrieval_baselines/rankings_bm25.jsonl": "f0b700248e388bd75fb7bca9c05a46ecbb6b5802861a8631eadb944e0b1572f0",
    "benchmark/phase3/phase3b_full/rankings_epoch_1.jsonl": "660c7ff55d787928050ba963cf4236788de21f661a523a5ebc1c47abbfc2f304",
    "benchmark/phase3/phase3b_fusion/rankings_bm25_trained_epoch1_rrf.jsonl.xz": "9601149e0297831ebfbfb4fde24b725a9b00287cf1b4d74270bf2921be138b31",
    "benchmark/manifest/model_registry.json": "8ba6de7b4171309828a043fa83835dcb1c47d202952152fa945119fa8cb1e367",
    "data/kegg/kegg_reaction_features.lzma": CATALOG_DIGEST,
}

PASS1_VERDICTS = [
    "label_correct", "label_correct_but_incomplete", "label_incorrect",
    "multiple_defensible_annotations", "not_kegg_mappable",
    "insufficient_model_information", "catalog_snapshot_issue", "unresolved",
]
PASS1_VERDICT_DESCRIPTIONS = {
    "label_correct": "The existing label accurately describes the modeled reaction.",
    "label_correct_but_incomplete": "The label is accurate but omits a material aspect of the modeled reaction.",
    "label_incorrect": "The existing label does not accurately describe the modeled reaction.",
    "multiple_defensible_annotations": "More than one KEGG reaction label is scientifically defensible.",
    "not_kegg_mappable": "The modeled event should not be represented by a KEGG reaction label.",
    "insufficient_model_information": "The local model does not provide enough information to decide.",
    "catalog_snapshot_issue": "The frozen catalog snapshot prevents a reliable determination.",
    "unresolved": "A decision cannot yet be made; explain what evidence is missing.",
}
PREDICTION_ASSESSMENTS = [
    "prediction_correct", "prediction_defensible_alternative", "prediction_related_but_incorrect",
    "prediction_incorrect", "abstention_appropriate", "abstention_inappropriate",
    "insufficient_information", "unresolved",
]
FAILURE_TAXONOMY = [
    "source_label_error", "source_label_incomplete", "catalog_coverage_failure",
    "query_information_failure", "retrieval_failure", "ranking_failure",
    "llm_selection_failure", "llm_overabstention", "appropriate_abstention",
    "ontology_equivalence", "nonchemical_event", "no_system_failure", "unresolved",
]
PASS1_REVIEW_FIELDS = [
    "audit_id", "verdict", "proposed_kegg_ids", "kegg_mappable", "confidence",
    "needs_second_review", "biological_rationale", "supporting_sources", "general_notes", "complete",
]
PASS2_REVIEW_FIELDS = [
    "audit_id", "prediction_assessment", "failure_taxonomy", "defensible_prediction_ids",
    "exact_matching_too_strict", "equivalence_relationship", "phase2_assessment",
    "bm25_assessment", "biencoder_assessment", "fusion_assessment", "direct_open_set_assessment",
    "grounded_llm_assessment", "recommended_label_action", "needs_adjudication", "confidence",
    "biological_rationale", "supporting_sources", "reviewer_notes", "complete",
]
PASS1_FORBIDDEN_KEYS = {
    "phase2", "bm25", "trained_biencoder", "fusion", "phase3a", "phase3c", "systems",
    "primary_category", "secondary_eligibility_reasons", "suspicion_rules", "failure_type",
    "selection_priority", "grounded_vs_fusion_transition", "retrieval_state", "cluster_id",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _canonical(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if value != value:
            return None
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def json_dumps_script(value: Any) -> str:
    return _canonical(value).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def verify_sources() -> None:
    for relative, expected in SOURCE_DIGESTS.items():
        path = REPO_ROOT / relative
        actual = sha256_portable(path) if path.is_file() else "missing"
        if actual != expected:
            raise ValueError(f"frozen reviewer source changed: {relative}: {actual} != {expected}")


def _resources(element: Any) -> list[str]:
    values: list[str] = []
    for index in range(element.getNumCVTerms()):
        term = element.getCVTerm(index)
        for resource_index in range(term.getNumResources()):
            value = str(term.getResourceURI(resource_index))
            if value and value not in values:
                values.append(value)
    return sorted(values)


def _identifiers(resources: Sequence[str], pattern: str) -> list[str]:
    values: list[str] = []
    regex = re.compile(pattern, re.IGNORECASE)
    for resource in resources:
        for value in regex.findall(resource):
            normalized = value.upper().replace("_", ":")
            if normalized not in values:
                values.append(normalized)
    return values


def _species_info(model: Any, species_reference: Any, role: str) -> dict[str, Any]:
    species_id = str(species_reference.getSpecies())
    species = model.getSpecies(species_id)
    if species is None:
        return {
            "role": role, "species_id": species_id, "display_name": None,
            "stoichiometry": None, "compartment_id": None, "compartment_name": None,
            "chebi_annotations": [], "kegg_compound_annotations": [], "annotation_resources": [],
        }
    resources = _resources(species)
    compartment_id = str(species.getCompartment() or "") or None
    compartment = model.getCompartment(compartment_id) if compartment_id else None
    stoichiometry = None
    if role != "modifier" and hasattr(species_reference, "getStoichiometry"):
        value = float(species_reference.getStoichiometry())
        stoichiometry = int(value) if value.is_integer() else value
    return {
        "role": role,
        "species_id": species_id,
        "display_name": str(species.getName() or "") or None,
        "stoichiometry": stoichiometry,
        "compartment_id": compartment_id,
        "compartment_name": (str(compartment.getName() or "") or None) if compartment is not None else None,
        "chebi_annotations": _identifiers(resources, r"(CHEBI[:_][0-9]+)"),
        "kegg_compound_annotations": _identifiers(resources, r"(?:compound[/:%]|kegg\.compound[:/])?(C[0-9]{5})"),
        "annotation_resources": resources,
    }


def _display_species(item: Mapping[str, Any]) -> str:
    name = item.get("display_name") or item.get("species_id") or "Unspecified species"
    coefficient = item.get("stoichiometry")
    prefix = "" if coefficient in (None, 1, 1.0) else f"{coefficient} "
    compartment = item.get("compartment_name") or item.get("compartment_id")
    suffix = f" [{compartment}]" if compartment else ""
    return f"{prefix}{name}{suffix}"


def _reaction_kegg_ids(reaction: Any) -> list[str]:
    return _identifiers(_resources(reaction), r"(?:reaction[/:%]|kegg\.reaction[:/])?(R[0-9]{5})")


def _model_registry() -> dict[str, dict[str, Any]]:
    payload = json.loads((REPO_ROOT / "benchmark/manifest/model_registry.json").read_text(encoding="utf-8"))
    return {str(item["model_id"]): item for item in payload["models"]}


def build_reaction_records(
    validation_rows: Sequence[Mapping[str, str]], supplemental_key: tuple[str, str],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Extract only validation reactions plus the explicitly authorized train key."""
    allowed: dict[str, set[str]] = defaultdict(set)
    normalized = {(row["model_id"], row["reaction_id"]): row["normalized_reaction_equation"] for row in validation_rows}
    for row in validation_rows:
        allowed[row["model_id"]].add(row["reaction_id"])
    allowed[supplemental_key[0]].add(supplemental_key[1])
    registry = _model_registry()
    records: dict[tuple[str, str], dict[str, Any]] = {}
    species_sets: dict[tuple[str, str], set[str]] = {}
    for model_id in sorted(allowed):
        metadata = registry[model_id]
        path = REPO_ROOT / metadata["local_path"]
        document = libsbml.readSBMLFromFile(str(path))
        model = document.getModel()
        if model is None:
            raise ValueError(f"unable to parse frozen SBML model: {model_id}")
        for reaction_id in sorted(allowed[model_id]):
            reaction = model.getReaction(reaction_id)
            if reaction is None:
                raise ValueError(f"frozen SBML reaction missing: {model_id}/{reaction_id}")
            reactants = [_species_info(model, reaction.getReactant(i), "reactant") for i in range(reaction.getNumReactants())]
            products = [_species_info(model, reaction.getProduct(i), "product") for i in range(reaction.getNumProducts())]
            modifiers = [_species_info(model, reaction.getModifier(i), "modifier") for i in range(reaction.getNumModifiers())]
            arrow = "⇌" if reaction.getReversible() else "→"
            readable = f"{' + '.join(map(_display_species, reactants)) or 'No reactants recorded'} {arrow} {' + '.join(map(_display_species, products)) or 'No products recorded'}"
            key = (model_id, reaction_id)
            records[key] = {
                "model_name": str(model.getName() or metadata.get("model_name") or "") or None,
                "model_id": model_id,
                "reaction_name": str(reaction.getName() or "") or None,
                "reaction_id": reaction_id,
                "readable_equation": readable,
                "normalized_equation": normalized.get(key),
                "reactants": reactants,
                "products": products,
                "modifiers": modifiers,
                "reversible": bool(reaction.getReversible()),
                "direction": "reversible" if reaction.getReversible() else "forward/irreversible",
                "kinetic_law_present": reaction.isSetKineticLaw(),
                "bounded_local_context": [],
                "source_annotation_provenance": {
                    "sbml_path": metadata["local_path"],
                    "sbml_sha256": metadata["local_sha256"],
                    "reaction_annotation_resources": _resources(reaction),
                    "sbml_level_version": metadata.get("sbml_format_version"),
                },
            }
            species_sets[key] = {item["species_id"] for item in [*reactants, *products, *modifiers]}
    for key, record in records.items():
        neighbors = []
        for other_key, other_species in species_sets.items():
            if other_key == key or other_key[0] != key[0]:
                continue
            overlap = sorted(species_sets[key] & other_species)
            if overlap:
                neighbors.append({
                    "reaction_id": other_key[1],
                    "reaction_name": records[other_key]["reaction_name"],
                    "shared_species_ids": overlap,
                })
        record["bounded_local_context"] = sorted(neighbors, key=lambda item: item["reaction_id"])[:6]
        if record["normalized_equation"] is None:
            record["normalized_equation"] = record["readable_equation"]
    return records


def _catalog() -> Mapping[str, Mapping[str, Any]]:
    return pickle.loads(lzma.open(REPO_ROOT / "data/kegg/kegg_reaction_features.lzma", "rb").read())


def catalog_card(identifier: str, catalog: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    raw = catalog.get(identifier)
    if raw is None:
        return {
            "kegg_id": identifier, "catalog_status": "absent_from_frozen_catalog",
            "name": None, "definition": None, "equation": None, "enzyme_ec": None,
            "rclass": None, "brite_or_orthology": None, "database_links": None,
            "catalog_sha256": CATALOG_DIGEST,
        }
    lowered = {str(key).lower(): value for key, value in raw.items()}
    return {
        "kegg_id": identifier,
        "catalog_status": "present_in_frozen_catalog",
        "name": lowered.get("name"),
        "definition": lowered.get("definition"),
        "equation": lowered.get("equation"),
        "enzyme_ec": lowered.get("enzyme") or lowered.get("ec"),
        "rclass": lowered.get("rclass"),
        "brite_or_orthology": lowered.get("brite") or lowered.get("orthology"),
        "database_links": lowered.get("dblinks"),
        "catalog_sha256": CATALOG_DIGEST,
    }


def _base_case(row: Mapping[str, str], reaction: Mapping[str, Any], catalog: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    labels = [catalog_card(identifier, catalog) for identifier in parse_multilabel_kegg_ids(row["ground_truth_kegg_ids"])]
    return {
        "audit_id": row["audit_id"],
        "sample_id": row["sample_id"],
        "model_id": row["model_id"],
        "reaction_id": row["reaction_id"],
        "model_reaction": reaction,
        "existing_labels": labels,
        "existing_label_provenance": {
            "source": "frozen BioModels reaction annotation represented in Prompt 1",
            "prompt1_inventory_sha256": SOURCE_DIGESTS["benchmark/phase3/error_audit/all_validation_outcomes.csv"],
            "catalog_sha256": CATALOG_DIGEST,
        },
    }


def _ranking_map(path: Path, wanted: set[tuple[str, str]]) -> dict[tuple[str, str], list[str]]:
    result: dict[tuple[str, str], list[str]] = {}
    for row in _read_jsonl(path):
        key = (str(row["model_id"]), str(row["reaction_id"]))
        if key in wanted:
            result[key] = [str(value) for value in row.get("ranked_ids") or []]
    if set(result) != wanted:
        missing = sorted(wanted - set(result))
        raise ValueError(f"validation ranking coverage missing {len(missing)} keys in {path.name}")
    return result


def _rank_cards(ids: Sequence[str], catalog: Mapping[str, Mapping[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    return [{"rank": rank, **catalog_card(identifier, catalog)} for rank, identifier in enumerate(ids[:limit], 1)]


def _bool(row: Mapping[str, str], field: str) -> bool:
    return row[field] == "true"


def _systems(
    row: Mapping[str, str], key: tuple[str, str], catalog: Mapping[str, Mapping[str, Any]],
    phase2: Mapping[tuple[str, str], list[str]], bm25: Mapping[tuple[str, str], list[str]],
    trained: Mapping[tuple[str, str], list[str]], fusion: Mapping[tuple[str, str], list[str]],
) -> dict[str, Any]:
    transition = {
        "grounded_only_correct": "grounded inference helped fusion",
        "fusion_only_correct": "grounded inference harmed fusion",
        "both_correct": "grounded inference preserved a correct fusion result",
        "neither_correct": "grounded inference did not recover the existing label",
    }.get(row["grounded_vs_fusion_transition"], row["grounded_vs_fusion_transition"])
    phase3a_prediction = row["phase3a_target_only_prediction"] or None
    phase3c_prediction = row["phase3c_prediction"] or None
    return {
        "phase2": {
            "status": row["phase2_status"], "candidate_set_size": int(row["phase2_candidate_set_size"] or 0),
            "top_candidates": _rank_cards(phase2[key], catalog),
            "existing_label_rank": int(row["phase2_ground_truth_rank"]) if row["phase2_ground_truth_rank"] else None,
            "exact_match": _bool(row, "phase2_exact"), "brite_orthology_match": _bool(row, "phase2_brite"),
        },
        "bm25": {
            "top_candidates": _rank_cards(bm25[key], catalog),
            "existing_label_rank": int(row["bm25_ground_truth_rank"]) if row["bm25_ground_truth_rank"] else None,
            "exact_match": _bool(row, "bm25_exact"), "brite_orthology_match": _bool(row, "bm25_brite"),
        },
        "trained_biencoder": {
            "checkpoint": "epoch_1", "top_candidates": _rank_cards(trained[key], catalog),
            "existing_label_rank": int(row["trained_epoch1_ground_truth_rank"]) if row["trained_epoch1_ground_truth_rank"] else None,
            "exact_match": _bool(row, "trained_epoch1_exact"), "brite_orthology_match": _bool(row, "trained_epoch1_brite"),
        },
        "fusion": {
            "method": "BM25 + trained epoch-1 bi-encoder RRF", "top_candidates": _rank_cards(fusion[key], catalog),
            "existing_label_rank": int(row["fusion_ground_truth_rank"]) if row["fusion_ground_truth_rank"] else None,
            "exact_match": _bool(row, "fusion_exact"), "brite_orthology_match": _bool(row, "fusion_brite"),
        },
        "phase3a": {
            "abstained": _bool(row, "phase3a_target_only_abstain"), "prediction": catalog_card(phase3a_prediction, catalog) if phase3a_prediction else None,
            "exact_match": _bool(row, "phase3a_target_only_exact"), "brite_orthology_match": _bool(row, "phase3a_target_only_brite"),
        },
        "phase3c": {
            "abstained": _bool(row, "phase3c_abstain"), "prediction": catalog_card(phase3c_prediction, catalog) if phase3c_prediction else None,
            "reasoning_summary": row["phase3c_reasoning_summary"] or None,
            "abstention_reason": row["phase3c_abstention_reason"] or None,
            "supporting_evidence_ids": [value for value in row["phase3c_supporting_evidence_ids"].split(";") if value],
            "evidence_compliant": _bool(row, "phase3c_evidence_compliant"),
            "compliance_problems": json.loads(row["phase3c_compliance_problems_json"] or "[]"),
            "exact_match": _bool(row, "phase3c_exact"), "brite_orthology_match": _bool(row, "phase3c_brite"),
        },
        "retrieval_state": row["retrieval_state"],
        "grounded_effect": transition,
    }


def _bundle(dataset_id: str, kind: str, cases: Sequence[Mapping[str, Any]], **extra: Any) -> dict[str, Any]:
    core = {
        "dataset_id": dataset_id,
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "kind": kind,
        "cases": list(cases),
        **extra,
    }
    return {**core, "dataset_digest": _digest(core)}


def build_data_bundles() -> dict[str, dict[str, Any]]:
    verify_sources()
    validation = _read_csv(ERROR_AUDIT / "all_validation_outcomes.csv")
    noncorrect = _read_csv(ERROR_AUDIT / "phase3c_noncorrect_review.csv")
    formal = _read_csv(ERROR_AUDIT / "formal_audit_sample.csv")
    crosswalk = {_key(row): row for row in _read_csv(ERROR_AUDIT / "formal_audit_crosswalk.csv")}
    blinded = _read_csv(ERROR_AUDIT / "formal_audit_blinded_order.csv")
    if len(validation) != 163 or len(noncorrect) != 61 or len(formal) != 60:
        raise ValueError("frozen Prompt 1/2 counts changed")
    if {row["split"] for row in validation + formal} != {"validation"}:
        raise ValueError("held-out row entered validation reviewer inputs")
    validation_by_key = {_key(row): row for row in validation}
    formal_keys = {_key(row) for row in formal}
    blinded_keys = [_key(row) for row in sorted(blinded, key=lambda row: int(row["blinded_order"]))]
    if set(blinded_keys) != formal_keys or len(blinded_keys) != 60:
        raise ValueError("Prompt 2 blinded order no longer aligns")
    supplemental_key = ("BIOMD0000000013", "E12")
    if supplemental_key in validation_by_key or supplemental_key in formal_keys:
        raise ValueError("supplemental training case entered validation")
    reactions = build_reaction_records(validation, supplemental_key)
    catalog = _catalog()

    pass1_cases = []
    for order, key in enumerate(blinded_keys, 1):
        row = validation_by_key[key]
        pass1_cases.append({"display_order": order, **_base_case(row, reactions[key], catalog)})
    pass1 = _bundle("aaaim-formal-pass1-20260910", "formal_pass1", pass1_cases, expected_case_count=60)
    assert_pass1_blinded(pass1)

    wanted = set(validation_by_key)
    phase2 = _ranking_map(PHASE3_DIR / "retrieval_baselines/rankings_phase2_rule_based.jsonl", wanted)
    bm25 = _ranking_map(PHASE3_DIR / "retrieval_baselines/rankings_bm25.jsonl", wanted)
    trained = _ranking_map(PHASE3_DIR / "phase3b_full/rankings_epoch_1.jsonl", wanted)
    fusion = _ranking_map(PHASE3_DIR / "phase3b_fusion/rankings_bm25_trained_epoch1_rrf.jsonl.xz", wanted)

    pass2_cases = []
    for order, key in enumerate(blinded_keys, 1):
        row = validation_by_key[key]
        selection = crosswalk[key]
        pass2_cases.append({
            "display_order": order,
            **_base_case(row, reactions[key], catalog),
            "systems": _systems(row, key, catalog, phase2, bm25, trained, fusion),
            "audit_crosswalk": {
                "primary_category": selection["primary_category"],
                "secondary_eligibility_reasons": json.loads(selection["secondary_eligibility_reasons"]),
                "selection_priority": selection["selection_priority"] or None,
                "suspicion_rules": json.loads(selection["suspicion_rules"]),
            },
        })
    pass2 = _bundle(
        "aaaim-formal-pass2-20260910", "formal_pass2", pass2_cases,
        expected_case_count=60, required_pass1_dataset_id=pass1["dataset_id"],
        required_pass1_dataset_digest=pass1["dataset_digest"],
        expected_audit_ids=[case["audit_id"] for case in pass1_cases],
    )

    formal_audits = {row["audit_id"] for row in formal}
    browse_cases = []
    for order, row in enumerate(sorted(noncorrect, key=lambda item: item["audit_id"]), 1):
        key = _key(row)
        browse_cases.append({
            "display_order": order,
            **_base_case(row, reactions[key], catalog),
            "systems": _systems(row, key, catalog, phase2, bm25, trained, fusion),
            "facets": {
                "outcome_kind": "abstention" if _bool(row, "phase3c_abstain") else "incorrect_selection",
                "retrieval_state": row["retrieval_state"],
                "grounded_effect": row["grounded_vs_fusion_transition"],
                "seen_unseen": "seen" if _bool(row, "target_seen_in_train") else "unseen",
                "corrected_stratum": row["corrected_phase2_stratum"],
                "model": row["model_id"], "cluster": row["cluster_id"],
                "evidence_compliance_issue": not _bool(row, "phase3c_evidence_compliant"),
                "formal_sample_inclusion": row["audit_id"] in formal_audits,
            },
        })
    browse = _bundle(
        "aaaim-phase3c-noncorrect-browse-20260910", "exploratory_noncorrect", browse_cases,
        expected_case_count=61, incorrect_selections=22, abstentions=39,
    )

    supplemental_reaction = reactions[supplemental_key]
    supplemental_ids = _reaction_kegg_ids(
        libsbml.readSBMLFromFile(str(REPO_ROOT / _model_registry()[supplemental_key[0]]["local_path"])).getModel().getReaction(supplemental_key[1])
    )
    if "R01429" not in supplemental_ids:
        supplemental_ids.append("R01429")
    supplemental_case = {
        "audit_id": "SUP-BIOMD0000000013-E12",
        "model_id": supplemental_key[0], "reaction_id": supplemental_key[1],
        "model_reaction": supplemental_reaction,
        "existing_labels": [catalog_card(identifier, catalog) for identifier in supplemental_ids],
        "manual_nomination": {
            "concern": "R01429 contains xylonolactone which is not reflected in reaction E12",
            "prior_source": "prior manual observation",
            "interpretation": "Nomination reason only; not a biological verdict.",
        },
        "split_status": "training",
        "formal_validation_audit_inclusion": False,
        "existing_method_results": None,
        "existing_method_results_status": "No frozen validation-method result exists for this training-only reaction; no new inference was run.",
    }
    supplemental = _bundle(
        "aaaim-supplemental-training-cases-20260910", "supplemental_training", [supplemental_case], expected_case_count=1,
    )
    return {"pass1": pass1, "pass2": pass2, "browse": browse, "supplemental": supplemental}


def _key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["model_id"]), str(row["reaction_id"])


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def assert_pass1_blinded(bundle: Mapping[str, Any]) -> None:
    found = set(_walk_keys(bundle)) & PASS1_FORBIDDEN_KEYS
    if found:
        raise ValueError(f"forbidden fields entered Pass 1: {sorted(found)}")
    if len(bundle["cases"]) != 60 or len({case["audit_id"] for case in bundle["cases"]}) != 60:
        raise ValueError("Pass 1 is not exactly 60 unique cases")


def blank_pass1_review(audit_id: str) -> dict[str, Any]:
    return {
        "audit_id": audit_id, "verdict": "", "proposed_kegg_ids": "", "kegg_mappable": "",
        "confidence": "", "needs_second_review": False, "biological_rationale": "",
        "supporting_sources": "", "general_notes": "", "complete": False,
    }


def validate_pass1_export(payload: Any, pass1_bundle: Mapping[str, Any], require_complete: bool = False) -> list[str]:
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["export must be a JSON object"]
    if payload.get("dataset_id") != pass1_bundle["dataset_id"] or payload.get("dataset_digest") != pass1_bundle["dataset_digest"]:
        problems.append("dataset identity or digest mismatch")
    if payload.get("review_schema_version") != REVIEW_SCHEMA_VERSION:
        problems.append("review schema mismatch")
    if payload.get("review_kind") != "formal_pass1":
        problems.append("review kind mismatch")
    digest_core = {
        key: value for key, value in payload.items()
        if key not in {"export_timestamp", "canonical_review_digest"}
    }
    if payload.get("canonical_review_digest") != _digest(digest_core):
        problems.append("canonical review digest mismatch")
    reviews = payload.get("reviews")
    if not isinstance(reviews, list):
        return [*problems, "reviews must be an array"]
    expected = [case["audit_id"] for case in pass1_bundle["cases"]]
    ids = [review.get("audit_id") for review in reviews if isinstance(review, dict)]
    if len(reviews) != 60 or len(ids) != 60 or len(set(ids)) != 60 or set(ids) != set(expected):
        problems.append("reviews must contain each of the expected 60 audit IDs exactly once")
    completed = 0
    for review in reviews:
        if not isinstance(review, dict):
            problems.append("each review must be an object")
            continue
        extra = set(review) - set(PASS1_REVIEW_FIELDS)
        if extra:
            problems.append(f"review contains forbidden or unknown fields: {sorted(extra)}")
        if review.get("verdict") and review.get("verdict") not in PASS1_VERDICTS:
            problems.append(f"invalid verdict for {review.get('audit_id')}")
        if review.get("kegg_mappable") not in {"", "yes", "no", "uncertain"}:
            problems.append(f"invalid KEGG-mappable value for {review.get('audit_id')}")
        if review.get("confidence") not in {"", "high", "medium", "low"}:
            problems.append(f"invalid confidence for {review.get('audit_id')}")
        if review.get("complete"):
            completed += 1
            if review.get("verdict") not in PASS1_VERDICTS or not str(review.get("biological_rationale") or "").strip():
                problems.append(f"completed review lacks verdict or rationale: {review.get('audit_id')}")
    if payload.get("completion_count") != completed:
        problems.append("completion count mismatch")
    all_complete = completed == 60 and not any("completed review" in problem for problem in problems)
    if bool(payload.get("complete")) != all_complete:
        problems.append("top-level completeness marker mismatch")
    if require_complete and not all_complete:
        problems.append("all 60 Pass 1 reviews must be complete")
    forbidden = set(_walk_keys(payload)) & PASS1_FORBIDDEN_KEYS
    if forbidden:
        problems.append(f"Pass 1 export contains system or sampling fields: {sorted(forbidden)}")
    return problems


REVIEW_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "aaaim-human-review-v1",
    "title": "AAAIM reaction annotation human review",
    "review_schema_version": REVIEW_SCHEMA_VERSION,
    "pass1": {
        "verdicts": PASS1_VERDICTS,
        "verdict_descriptions": PASS1_VERDICT_DESCRIPTIONS,
        "kegg_mappable": ["yes", "no", "uncertain"],
        "confidence": ["high", "medium", "low"],
        "required_to_complete": ["verdict", "biological_rationale"],
        "fields": PASS1_REVIEW_FIELDS,
    },
    "pass2": {
        "prediction_assessments": PREDICTION_ASSESSMENTS,
        "failure_taxonomy": FAILURE_TAXONOMY,
        "yes_no_uncertain": ["yes", "no", "uncertain"],
        "confidence": ["high", "medium", "low"],
        "required_to_complete": ["prediction_assessment", "biological_rationale"],
        "fields": PASS2_REVIEW_FIELDS,
    },
    "human_fields_prepopulated": False,
}


CSS = r"""
:root{--ink:#18212b;--muted:#526170;--paper:#f4f7f9;--card:#fff;--line:#c7d1da;--accent:#145c7d;--accent2:#7b3f00;--good:#176b42;--warn:#8a4b00;--focus:#ffd166;--shadow:0 2px 12px rgba(24,33,43,.10)}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}a{color:var(--accent)}header{background:#102a3a;color:#fff;padding:1rem 1.5rem}header h1{margin:.1rem 0;font-size:1.45rem}header p{margin:.25rem 0;color:#dcebf2}.shell{max-width:1200px;margin:auto;padding:1rem}.warning{border-left:5px solid #c56300;background:#fff4dc;padding:.8rem 1rem;margin:1rem 0}.success{border-left:5px solid var(--good);background:#e8f5ee;padding:.8rem 1rem}.toolbar{position:sticky;top:0;z-index:5;background:rgba(244,247,249,.97);border-bottom:1px solid var(--line);padding:.75rem 0;display:grid;gap:.65rem}.toolbar-row{display:flex;flex-wrap:wrap;gap:.55rem;align-items:center}.toolbar label{font-weight:650}.grow{flex:1;min-width:190px}button,.button,select,input,textarea{font:inherit}button,.button{border:1px solid #0f4964;border-radius:6px;background:var(--accent);color:#fff;padding:.55rem .8rem;cursor:pointer;text-decoration:none;display:inline-block}button.secondary,.button.secondary{background:#fff;color:var(--accent)}button:disabled{opacity:.5;cursor:not-allowed}input,select,textarea{border:1px solid #8594a2;border-radius:5px;background:#fff;padding:.45rem .55rem;max-width:100%}textarea{width:100%;min-height:95px;resize:vertical}:focus-visible{outline:3px solid var(--focus);outline-offset:2px}.progress-track{height:10px;background:#d7e0e7;border-radius:10px;overflow:hidden;min-width:170px}.progress-fill{height:100%;background:var(--good);width:0}.case-card,.section,.landing-card{background:var(--card);border:1px solid var(--line);border-radius:9px;box-shadow:var(--shadow);padding:1rem;margin:1rem 0;min-width:0}.case-title{display:flex;justify-content:space-between;gap:1rem;align-items:flex-start}.case-title h2{margin:0}.eyebrow{text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-size:.78rem;font-weight:750}.equation{font:600 1.12rem/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;background:#eef4f7;border-radius:6px;padding:.8rem;overflow-wrap:anywhere}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}.label-card,.proposal-card,.alternative-card{border:2px solid var(--line);border-radius:8px;padding:.85rem;overflow-wrap:anywhere}.label-card{border-color:var(--accent2)}.proposal-card{border-color:var(--accent)}.alternative-card{border-color:#738391}.role-note{font-size:.88rem;font-weight:700}.muted,.technical{color:var(--muted)}.missing{font-style:italic;color:#687480}.pill{display:inline-block;border:1px solid #6d7e8d;border-radius:999px;padding:.15rem .48rem;margin:.12rem;font-size:.82rem;background:#fff}.pill.good{border-color:var(--good)}.pill.warn{border-color:var(--warn)}table{border-collapse:collapse;width:100%;display:block;overflow-x:auto}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #d7dee4;padding:.45rem;overflow-wrap:anywhere}th{background:#edf2f5}details{border:1px solid #d4dce2;border-radius:6px;padding:.6rem;margin:.6rem 0}summary{cursor:pointer;font-weight:700}.review-form{border-top:5px solid var(--accent);background:#fdfefe}.field{margin:.8rem 0}.field>label,.fieldset-title{display:block;font-weight:700;margin-bottom:.25rem}.choices{display:grid;gap:.35rem}.choices label{font-weight:400}.status{font-weight:700}.locked{background:#eef2f5;border:1px solid #8795a2;padding:.8rem;border-radius:6px}.landing-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem}.landing-card h2{margin-top:0}.hidden{display:none!important}.filters{background:#e9f0f4;border:1px solid var(--line);padding:.7rem;border-radius:7px}.footer-note{font-size:.9rem;color:var(--muted);margin:2rem 0}@media(max-width:680px){header{padding:.8rem}.shell{padding:.6rem}.case-title{display:block}.toolbar{position:static}.grid{grid-template-columns:1fr}.equation{font-size:.98rem}}
""".strip() + "\n"


BASE_JS = r"""
"use strict";
const MISSING="Not provided in frozen local artifacts.";
const $=(id)=>document.getElementById(id);
const escCsv=(v)=>`"${String(v??"").replaceAll('"','""')}"`;
const canonical=(v)=>{if(Array.isArray(v))return`[${v.map(canonical).join(',')}]`;if(v&&typeof v==='object')return`{${Object.keys(v).sort().map(k=>JSON.stringify(k)+':'+canonical(v[k])).join(',')}}`;return JSON.stringify(v)};
function sha256(ascii){function rr(v,a){return(v>>>a)|(v<<(32-a))}let m=Math.pow,max=m(2,32),l='length',i,j,r='',w=[],al=ascii[l]*8,h=sha256.h=sha256.h||[],k=sha256.k=sha256.k||[],pc=k[l],is=[];for(let c=2;pc<64;c++){if(!is[c]){for(i=0;i<313;i+=c)is[i]=c;h[pc]=(m(c,.5)*max)|0;k[pc++]=(m(c,1/3)*max)|0}}ascii+='\x80';while(ascii[l]%64-56)ascii+='\x00';for(i=0;i<ascii[l];i++){j=ascii.charCodeAt(i);if(j>>8)return'';w[i>>2]|=j<<((3-i)%4)*8}w[w[l]]=((al/max)|0);w[w[l]]=al;for(j=0;j<w[l];){let wh=w.slice(j,j+=16),oh=h;h=h.slice(0,8);for(i=0;i<64;i++){let i2=i+j,a=h[0],e=h[4],t1=h[7]+(rr(e,6)^rr(e,11)^rr(e,25))+((e&h[5])^((~e)&h[6]))+k[i]+(wh[i]=i<16?wh[i]:(wh[i-16]+(rr(wh[i-15],7)^rr(wh[i-15],18)^(wh[i-15]>>>3))+wh[i-7]+(rr(wh[i-2],17)^rr(wh[i-2],19)^(wh[i-2]>>>10)))|0),t2=(rr(a,2)^rr(a,13)^rr(a,22))+((a&h[1])^(a&h[2])^(h[1]&h[2]));h=[(t1+t2)|0,a,h[1],h[2],(h[3]+t1)|0,e,h[5],h[6]]}for(i=0;i<8;i++)h[i]=(h[i]+oh[i])|0}for(i=0;i<8;i++)for(j=3;j+1;j--){let b=(h[i]>>(j*8))&255;r+=(b<16?'0':'')+b.toString(16)}return r}
const digestCanonical=(v)=>sha256(unescape(encodeURIComponent(canonical(v))));
function el(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined&&text!==null)n.textContent=String(text);return n}
function value(v){return(v===null||v===undefined||v===""||(Array.isArray(v)&&!v.length))?MISSING:String(v)}
function list(parent,label,items){const s=el('section','section'),h=el('h3','',label);s.append(h);if(!items||!items.length)s.append(el('p','missing',MISSING));else{const t=el('table'),head=el('tr');['Role / name','SBML ID','Stoichiometry','Compartment','ChEBI','KEGG compound'].forEach(x=>head.append(el('th','',x)));t.append(head);items.forEach(x=>{const tr=el('tr');[`${x.role}: ${x.display_name||MISSING}`,x.species_id,value(x.stoichiometry),x.compartment_name||x.compartment_id||MISSING,(x.chebi_annotations||[]).join(', ')||MISSING,(x.kegg_compound_annotations||[]).join(', ')||MISSING].forEach(v=>tr.append(el('td','',v)));t.append(tr)});s.append(t)}parent.append(s)}
function keggCard(card,role){const d=el('article',role==='Existing BioModels label'?'label-card':role==='AAAIM proposal'?'proposal-card':'alternative-card');d.append(el('div','eyebrow',role),el('h3','',card.kegg_id),el('p','role-note',card.catalog_status==='present_in_frozen_catalog'?'Present in frozen KEGG catalog':'Absent from frozen KEGG catalog'));[['Name',card.name],['Definition',card.definition],['Equation',card.equation],['Enzyme / EC',card.enzyme_ec],['RCLASS',card.rclass],['BRITE / orthology',card.brite_or_orthology]].forEach(([k,v])=>{d.append(el('div','eyebrow',k),el('p',v?'':'missing',value(v)))});const det=el('details'),sum=el('summary','', 'Technical provenance');det.append(sum,el('p','technical',`Catalog SHA-256: ${card.catalog_sha256}`),el('p','technical',`Database links: ${value(card.database_links)}`));d.append(det);return d}
function renderSource(container,c){const r=c.model_reaction;const top=el('article','case-card'),title=el('div','case-title');const left=el('div');left.append(el('div','eyebrow','Model reaction'),el('h2','',r.reaction_name||r.reaction_id),el('p','muted',`${r.model_name||MISSING} · ${r.model_id} / ${r.reaction_id}`));title.append(left,el('span','pill',c.audit_id));top.append(title,el('div','equation',r.readable_equation),el('p','technical',`Normalized source equation: ${value(r.normalized_equation)}`));const facts=el('div','grid');[['Reaction name',r.reaction_name||MISSING],['Direction / reversibility',r.direction],['Kinetic law',r.kinetic_law_present?'Present':'Not recorded'],['SBML source',r.source_annotation_provenance.sbml_path],['SBML digest',r.source_annotation_provenance.sbml_sha256]].forEach(([k,v])=>{const x=el('div','section');x.append(el('div','eyebrow',k),el('p',v===MISSING?'missing':'',v));facts.append(x)});top.append(facts);list(top,'Reactants',r.reactants);list(top,'Products',r.products);list(top,'Modifiers',r.modifiers);const context=el('details'),cs=el('summary','', 'Bounded local reaction context and source annotations');context.append(cs);if(r.bounded_local_context.length)r.bounded_local_context.forEach(x=>context.append(el('p','',`${x.reaction_id} — ${x.reaction_name||'unnamed'}; shared species: ${x.shared_species_ids.join(', ')}`)));else context.append(el('p','missing',MISSING));context.append(el('p','technical',`Reaction annotation resources: ${(r.source_annotation_provenance.reaction_annotation_resources||[]).join('; ')||MISSING}`));top.append(context);container.append(top);const labels=el('section','case-card');labels.append(el('div','eyebrow','Current source annotation'),el('h2','',c.existing_labels.length===1?'Existing BioModels KEGG label':'Existing BioModels KEGG labels'));const g=el('div','grid');c.existing_labels.forEach(x=>g.append(keggCard(x,'Existing BioModels label')));labels.append(g);container.append(labels)}
function download(name,text,type){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type}));a.download=name;document.body.append(a);a.click();setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove()},0)}
function setMessage(text,bad=false){const n=$('message');n.textContent=text;n.className=bad?'warning':'success'}
"""


PASS1_JS = BASE_JS + r"""
const STORE=`aaaim-review:formal-pass1:${DATA.review_schema_version}:${DATA.dataset_digest}`;
let index=0,state={reviewer_id:'',reviews:Object.fromEntries(DATA.cases.map(c=>[c.audit_id,{audit_id:c.audit_id,verdict:'',proposed_kegg_ids:'',kegg_mappable:'',confidence:'',needs_second_review:false,biological_rationale:'',supporting_sources:'',general_notes:'',complete:false}]))};
try{const saved=JSON.parse(localStorage.getItem(STORE));if(saved&&saved.reviews)state=saved}catch(_e){}
function save(){localStorage.setItem(STORE,JSON.stringify(state));$('saveStatus').textContent=`Saved locally ${new Date().toLocaleTimeString()}`;updateProgress()}
function field(parent,label,key,type='text',options=[]){const w=el('div','field'),lab=el('label','',label);lab.htmlFor=`f-${key}`;w.append(lab);let n;if(type==='select'){n=el('select');n.append(el('option','', 'Choose…'));options.forEach(v=>{const o=el('option','',v);o.value=v;o.title=SCHEMA.pass1.verdict_descriptions?.[v]||'';n.append(o)})}else if(type==='textarea')n=el('textarea');else{n=el('input');n.type=type}n.id=`f-${key}`;const review=state.reviews[DATA.cases[index].audit_id];if(type==='checkbox')n.checked=!!review[key];else n.value=review[key]??'';n.addEventListener('input',()=>{review[key]=type==='checkbox'?n.checked:n.value;if(key==='complete'&&n.checked&&!canComplete(review)){n.checked=false;review.complete=false;setMessage('A verdict and biological rationale are required before completion.',true)}save()});w.append(n);parent.append(w)}
function canComplete(r){return SCHEMA.pass1.verdicts.includes(r.verdict)&&String(r.biological_rationale||'').trim().length>0}
function render(){const c=DATA.cases[index],root=$('case');root.replaceChildren();$('caseNumber').textContent=`Case ${index+1} of ${DATA.cases.length}`;renderSource(root,c);const form=el('section','case-card review-form');form.append(el('div','eyebrow','Blank human review'),el('h2','','Source-label verdict'),el('p','muted','Choose whether the existing annotation describes the model reaction. “Unresolved” still requires an explanation of missing evidence.'));const guide=el('details');guide.append(el('summary','','Verdict descriptions'));Object.entries(SCHEMA.pass1.verdict_descriptions).forEach(([k,v])=>guide.append(el('p','',`${k}: ${v}`)));form.append(guide);field(form,'Required verdict','verdict','select',SCHEMA.pass1.verdicts);field(form,'Proposed KEGG IDs (optional)','proposed_kegg_ids');field(form,'Is this reaction KEGG-mappable?','kegg_mappable','select',SCHEMA.pass1.kegg_mappable);field(form,'Reviewer confidence','confidence','select',SCHEMA.pass1.confidence);field(form,'Needs second review','needs_second_review','checkbox');field(form,'Biological rationale / missing-evidence explanation','biological_rationale','textarea');field(form,'Supporting sources or notes','supporting_sources','textarea');field(form,'General reviewer notes','general_notes','textarea');field(form,'Mark this case complete','complete','checkbox');root.append(form);$('navigator').value=String(index);$('prev').disabled=index===0;$('next').disabled=index===DATA.cases.length-1;updateProgress()}
function updateProgress(){const values=Object.values(state.reviews),done=values.filter(x=>x.complete).length,unresolved=values.filter(x=>x.verdict==='unresolved').length;$('progressText').textContent=`${done} completed · ${unresolved} unresolved`;$('progressFill').style.width=`${done/60*100}%`}
function exportObject(){const reviews=DATA.cases.map(c=>state.reviews[c.audit_id]),count=reviews.filter(x=>x.complete).length,core={dataset_id:DATA.dataset_id,dataset_digest:DATA.dataset_digest,review_schema_version:DATA.review_schema_version,review_kind:'formal_pass1',reviewer_id:state.reviewer_id,reviews,completion_count:count,complete:count===60};return{...core,export_timestamp:new Date().toISOString(),canonical_review_digest:digestCanonical(core)}}
function exportJson(){const x=exportObject();download(`pass1_review_${state.reviewer_id||'reviewer'}.json`,JSON.stringify(x,null,2)+'\n','application/json');setMessage(x.complete?'Complete Pass 1 exported.':'Incomplete checkpoint exported; it cannot unlock Pass 2.')}
function exportCsv(){const x=exportObject(),heads=['dataset_id','dataset_digest','review_schema_version','reviewer_id','audit_id','verdict','proposed_kegg_ids','kegg_mappable','confidence','needs_second_review','biological_rationale','supporting_sources','general_notes','complete'];const lines=[heads.map(escCsv).join(',')];x.reviews.forEach(r=>lines.push(heads.map(h=>escCsv(r[h]??x[h]??'')).join(',')));download(`pass1_review_${state.reviewer_id||'reviewer'}.csv`,lines.join('\n')+'\n','text/csv')}
function validate(x){const issues=[];if(!x||x.dataset_id!==DATA.dataset_id||x.dataset_digest!==DATA.dataset_digest)issues.push('Dataset identity or digest mismatch.');if(!x||x.review_schema_version!==DATA.review_schema_version||x.review_kind!=='formal_pass1')issues.push('Review schema or kind mismatch.');if(!x||!Array.isArray(x.reviews)||x.reviews.length!==60)issues.push('Exactly 60 review rows are required.');else{const ids=x.reviews.map(r=>r.audit_id),expected=new Set(DATA.cases.map(c=>c.audit_id)),allowed=new Set(['audit_id','verdict','proposed_kegg_ids','kegg_mappable','confidence','needs_second_review','biological_rationale','supporting_sources','general_notes','complete']);if(new Set(ids).size!==60||ids.some(id=>!expected.has(id)))issues.push('Audit IDs are duplicated or mismatched.');x.reviews.forEach(r=>{const extra=Object.keys(r).filter(k=>!allowed.has(k));if(extra.length)issues.push(`Unknown or forbidden fields: ${extra.join(', ')}`);if(r.verdict&&!SCHEMA.pass1.verdicts.includes(r.verdict))issues.push(`Invalid verdict: ${r.audit_id}`);if(r.complete&&!canComplete(r))issues.push(`Incomplete required fields: ${r.audit_id}`)});const done=x.reviews.filter(r=>r.complete).length;if(x.completion_count!==done||Boolean(x.complete)!==(done===60))issues.push('Completion markers do not match review rows.')}if(x){const core={...x};delete core.export_timestamp;delete core.canonical_review_digest;if(!x.canonical_review_digest||digestCanonical(core)!==x.canonical_review_digest)issues.push('Canonical review digest mismatch.')}return issues}
function importFile(file){const reader=new FileReader();reader.onload=()=>{try{const x=JSON.parse(reader.result),issues=validate(x);if(issues.length)throw new Error(issues.join(' '));state.reviewer_id=x.reviewer_id||'';state.reviews=Object.fromEntries(x.reviews.map(r=>[r.audit_id,r]));$('reviewer').value=state.reviewer_id;save();render();setMessage('Pass 1 checkpoint imported and resumed.')}catch(e){setMessage(`Import rejected: ${e.message}`,true)}};reader.readAsText(file)}
DATA.cases.forEach((c,i)=>{const o=el('option','',`${i+1}. ${c.audit_id} · ${c.model_id}/${c.reaction_id}`);o.value=String(i);$('navigator').append(o)});$('reviewer').value=state.reviewer_id;$('reviewer').addEventListener('input',e=>{state.reviewer_id=e.target.value;save()});$('navigator').addEventListener('change',e=>{index=Number(e.target.value);render()});$('prev').onclick=()=>{if(index){index--;render()}};$('next').onclick=()=>{if(index<59){index++;render()}};$('search').addEventListener('input',e=>{const q=e.target.value.trim().toLowerCase();if(!q)return;const found=DATA.cases.findIndex(c=>[c.audit_id,c.model_id,c.reaction_id].some(v=>v.toLowerCase().includes(q)));if(found>=0){index=found;render()}});$('exportJson').onclick=exportJson;$('exportCsv').onclick=exportCsv;$('import').addEventListener('change',e=>{if(e.target.files[0])importFile(e.target.files[0])});document.addEventListener('keydown',e=>{if(e.altKey&&e.key==='ArrowLeft')$('prev').click();if(e.altKey&&e.key==='ArrowRight')$('next').click()});render();save();
"""


REVEALED_JS = BASE_JS + r"""
let index=0,visible=DATA.cases.slice(),lockedPass1=null;
const reviewFields=SCHEMA.pass2;
function blank(c){return{audit_id:c.audit_id,prediction_assessment:'',failure_taxonomy:[],defensible_prediction_ids:'',exact_matching_too_strict:'',equivalence_relationship:'',phase2_assessment:'',bm25_assessment:'',biencoder_assessment:'',fusion_assessment:'',direct_open_set_assessment:'',grounded_llm_assessment:'',recommended_label_action:'',needs_adjudication:false,confidence:'',biological_rationale:'',supporting_sources:'',reviewer_notes:'',complete:false}}
const STORE=`aaaim-review:${MODE}:${DATA.review_schema_version}:${DATA.dataset_digest}`;let state={reviewer_id:'',reviews:Object.fromEntries(DATA.cases.map(c=>[c.audit_id,blank(c)]))};try{const s=JSON.parse(localStorage.getItem(STORE));if(s&&s.reviews)state=s}catch(_e){}
function save(){localStorage.setItem(STORE,JSON.stringify(state));if($('saveStatus'))$('saveStatus').textContent=`Saved locally ${new Date().toLocaleTimeString()}`;progress()}
function renderRanking(parent,title,data){const d=el('details');d.append(el('summary','',`${title} — Top ${data.top_candidates.length}`));d.append(el('p','technical',`Existing-label rank: ${value(data.existing_label_rank)} · Exact match: ${data.exact_match?'yes':'no'} · BRITE/orthology match: ${data.brite_orthology_match?'yes':'no'}`));const g=el('div','grid');data.top_candidates.forEach(x=>g.append(keggCard(x,`Retrieved alternative · rank ${x.rank}`)));d.append(g);parent.append(d)}
function renderSystems(root,c){const sys=c.systems,s=el('section','case-card');s.append(el('div','eyebrow','System-disagreement evidence'),el('h2','','Frozen method outputs'),el('p','muted','Roles do not imply biological correctness. Compare every proposal to the model reaction and existing label.'));const summary=el('div','grid');[['Retrieval state',sys.retrieval_state],['Grounded effect',sys.grounded_effect],['Phase 2 status',sys.phase2.status],['Phase 2 candidate count',sys.phase2.candidate_set_size]].forEach(([k,v])=>{const x=el('div','section');x.append(el('div','eyebrow',k),el('p','',value(v)));summary.append(x)});s.append(summary);renderRanking(s,'Phase 2 candidates',sys.phase2);renderRanking(s,'BM25 results',sys.bm25);renderRanking(s,'Trained epoch-1 bi-encoder results',sys.trained_biencoder);renderRanking(s,'BM25 + trained bi-encoder RRF evidence',sys.fusion);const open=el('div','grid');[['Phase 3A target-only',sys.phase3a],['Phase 3C grounded',sys.phase3c]].forEach(([name,x])=>{const d=el('article','section');d.append(el('div','eyebrow',name),el('p','role-note',x.abstained?'Abstained':'Selected a proposal'),el('p','technical',`Exact match: ${x.exact_match?'yes':'no'} · BRITE/orthology: ${x.brite_orthology_match?'yes':'no'}`));if(x.prediction)d.append(keggCard(x.prediction,'AAAIM proposal'));if(x.reasoning_summary)d.append(el('h4','','Frozen reasoning summary'),el('p','',x.reasoning_summary));if(x.abstention_reason)d.append(el('h4','','Abstention reason'),el('p','',x.abstention_reason));if(x.supporting_evidence_ids)d.append(el('p','technical',`Supporting evidence IDs: ${x.supporting_evidence_ids.join(', ')||MISSING}`),el('p','technical',`Evidence compliant: ${x.evidence_compliant?'yes':'no'}`));open.append(d)});s.append(open);if(c.audit_crosswalk){const d=el('details');d.append(el('summary','','Private sampling crosswalk (Pass 2 only)'),el('p','technical',`Primary category: ${c.audit_crosswalk.primary_category}`),el('p','technical',`Secondary reasons: ${c.audit_crosswalk.secondary_eligibility_reasons.join('; ')||MISSING}`),el('p','technical',`Mechanical suspicion rules: ${c.audit_crosswalk.suspicion_rules.join('; ')||MISSING}`));s.append(d)}root.append(s)}
function formField(parent,label,key,type='text',options=[]){const w=el('div','field'),lab=el('label','',label);lab.htmlFor=`f-${key}`;w.append(lab);let n;if(type==='select'){n=el('select');n.append(el('option','', 'Choose…'));options.forEach(v=>{const o=el('option','',v);o.value=v;n.append(o)})}else if(type==='textarea')n=el('textarea');else{n=el('input');n.type=type}n.id=`f-${key}`;const r=state.reviews[visible[index].audit_id];if(type==='checkbox')n.checked=!!r[key];else n.value=r[key]??'';n.addEventListener('input',()=>{r[key]=type==='checkbox'?n.checked:n.value;if(key==='complete'&&n.checked&&!(reviewFields.prediction_assessments.includes(r.prediction_assessment)&&String(r.biological_rationale||'').trim())){n.checked=false;r.complete=false;setMessage('An assessment and biological rationale are required before completion.',true)}save()});w.append(n);parent.append(w)}
function renderForm(root,c){const f=el('section','case-card review-form');f.append(el('div','eyebrow',MODE==='browse'?'Blank exploratory review':'Blank Pass 2 human review'),el('h2','','Prediction assessment'));formField(f,'Prediction assessment','prediction_assessment','select',reviewFields.prediction_assessments);const tax=el('fieldset','field'),legend=el('legend','fieldset-title','Failure taxonomy (select all that apply)');tax.append(legend);reviewFields.failure_taxonomy.forEach(v=>{const l=el('label'),n=el('input');n.type='checkbox';n.checked=(state.reviews[c.audit_id].failure_taxonomy||[]).includes(v);n.addEventListener('input',()=>{const a=new Set(state.reviews[c.audit_id].failure_taxonomy||[]);n.checked?a.add(v):a.delete(v);state.reviews[c.audit_id].failure_taxonomy=[...a].sort();save()});l.append(n,document.createTextNode(` ${v}`));tax.append(l)});f.append(tax);formField(f,'Scientifically defensible prediction IDs','defensible_prediction_ids');formField(f,'Is exact matching too strict?','exact_matching_too_strict','select',reviewFields.yes_no_uncertain);formField(f,'Relevant equivalence relationship','equivalence_relationship');[['Phase 2 assessment','phase2_assessment'],['BM25 assessment','bm25_assessment'],['Bi-encoder assessment','biencoder_assessment'],['Fusion assessment','fusion_assessment'],['Direct open-set assessment','direct_open_set_assessment'],['Grounded LLM assessment','grounded_llm_assessment'],['Recommended label action','recommended_label_action']].forEach(([l,k])=>formField(f,l,k,'textarea'));formField(f,'Needs adjudication','needs_adjudication','checkbox');formField(f,'Reviewer confidence','confidence','select',reviewFields.confidence);formField(f,'Biological rationale','biological_rationale','textarea');formField(f,'Supporting sources','supporting_sources','textarea');formField(f,'Reviewer notes','reviewer_notes','textarea');formField(f,'Mark this case complete','complete','checkbox');root.append(f)}
function render(){if(MODE==='pass2'&&!lockedPass1)return;const c=visible[index],root=$('case');root.replaceChildren();$('caseNumber').textContent=`Case ${index+1} of ${visible.length}`;if(MODE==='pass2'){const locked=el('section','case-card locked');locked.append(el('div','eyebrow','Pass 1 locked'),el('p','',`Imported reviewer: ${lockedPass1.reviewer_id||'not specified'}`));const p1=lockedPass1.reviews.find(x=>x.audit_id===c.audit_id);locked.append(el('p','',`Verdict: ${p1.verdict}`),el('p','',`Rationale: ${p1.biological_rationale}`));root.append(locked)}renderSource(root,c);renderSystems(root,c);renderForm(root,c);$('navigator').value=c.audit_id;$('prev').disabled=index===0;$('next').disabled=index===visible.length-1;progress()}
function progress(){if(!$('progressText'))return;const ids=new Set(visible.map(c=>c.audit_id)),v=Object.values(state.reviews).filter(r=>ids.has(r.audit_id)),done=v.filter(r=>r.complete).length,un=v.filter(r=>r.prediction_assessment==='unresolved').length;$('progressText').textContent=`${done} completed · ${un} unresolved`;$('progressFill').style.width=`${v.length?done/v.length*100:0}%`}
function validatePass1Export(x,complete=true){const issues=[],allowedTop=new Set(['dataset_id','dataset_digest','review_schema_version','review_kind','reviewer_id','reviews','completion_count','complete','export_timestamp','canonical_review_digest']),allowedReview=new Set(['audit_id','verdict','proposed_kegg_ids','kegg_mappable','confidence','needs_second_review','biological_rationale','supporting_sources','general_notes','complete']);if(!x||x.dataset_id!==DATA.required_pass1_dataset_id||x.dataset_digest!==DATA.required_pass1_dataset_digest)issues.push('Dataset identity or digest mismatch.');if(x&&x.review_schema_version!==DATA.review_schema_version)issues.push('Review schema mismatch.');if(x&&x.review_kind!=='formal_pass1')issues.push('Review kind mismatch.');if(x){const extra=Object.keys(x).filter(k=>!allowedTop.has(k));if(extra.length)issues.push(`Unknown or forbidden export fields: ${extra.join(', ')}`)}if(!x||!Array.isArray(x.reviews)||x.reviews.length!==60)issues.push('Exactly 60 Pass 1 reviews are required.');else{const ids=x.reviews.map(r=>r.audit_id),expected=new Set(DATA.expected_audit_ids);if(new Set(ids).size!==60||ids.some(id=>!expected.has(id)))issues.push('Duplicate or mismatched audit IDs.');x.reviews.forEach(r=>{if(!SCHEMA.pass1.verdicts.includes(r.verdict)||!String(r.biological_rationale||'').trim()||!r.complete)issues.push(`Incomplete Pass 1 case: ${r.audit_id}`);const forbidden=Object.keys(r).filter(k=>!allowedReview.has(k));if(forbidden.length)issues.push(`Forbidden Pass 1 fields: ${forbidden.join(', ')}`)})}if(complete&&x&&(!x.complete||x.completion_count!==60))issues.push('Pass 1 export is marked incomplete.');if(x){const core={...x};delete core.export_timestamp;delete core.canonical_review_digest;if(!x.canonical_review_digest||digestCanonical(core)!==x.canonical_review_digest)issues.push('Canonical Pass 1 digest mismatch.')}return issues}
window.validatePass1Export=validatePass1Export;
function validateRevealedExport(x){const issues=[];const expectedKind=MODE==='browse'?'exploratory_noncorrect':'formal_pass2';if(!x||x.dataset_id!==DATA.dataset_id||x.dataset_digest!==DATA.dataset_digest)issues.push('Dataset identity or digest mismatch.');if(!x||x.review_schema_version!==DATA.review_schema_version||x.review_kind!==expectedKind)issues.push('Review schema or kind mismatch.');if(!x||!Array.isArray(x.reviews)||x.reviews.length!==DATA.cases.length)issues.push(`Exactly ${DATA.cases.length} review rows are required.`);else{const expected=new Set(DATA.cases.map(c=>c.audit_id)),ids=x.reviews.map(r=>r.audit_id),allowed=new Set(Object.keys(blank(DATA.cases[0])));if(new Set(ids).size!==DATA.cases.length||ids.some(id=>!expected.has(id)))issues.push('Review rows are missing, duplicated, or mismatched.');x.reviews.forEach(r=>{const extra=Object.keys(r).filter(k=>!allowed.has(k));if(extra.length)issues.push(`Unknown review fields: ${extra.join(', ')}`);if(r.prediction_assessment&&!reviewFields.prediction_assessments.includes(r.prediction_assessment))issues.push(`Invalid prediction assessment: ${r.audit_id}`);if(!Array.isArray(r.failure_taxonomy)||(r.failure_taxonomy||[]).some(v=>!reviewFields.failure_taxonomy.includes(v)))issues.push(`Invalid failure taxonomy: ${r.audit_id}`);if(r.complete&&!(reviewFields.prediction_assessments.includes(r.prediction_assessment)&&String(r.biological_rationale||'').trim()))issues.push(`Incomplete required fields: ${r.audit_id}`)});const done=x.reviews.filter(r=>r.complete).length;if(x.completion_count!==done||Boolean(x.complete)!==(done===DATA.cases.length))issues.push('Completion markers do not match review rows.')}if(x){const core={...x};delete core.export_timestamp;delete core.canonical_review_digest;if(!x.canonical_review_digest||digestCanonical(core)!==x.canonical_review_digest)issues.push('Canonical review digest mismatch.')}return issues}
window.validateRevealedExport=validateRevealedExport;
function exportObject(){const reviews=DATA.cases.map(c=>state.reviews[c.audit_id]),count=reviews.filter(x=>x.complete).length,core={dataset_id:DATA.dataset_id,dataset_digest:DATA.dataset_digest,review_schema_version:DATA.review_schema_version,review_kind:MODE==='browse'?'exploratory_noncorrect':'formal_pass2',reviewer_id:state.reviewer_id,reviews,completion_count:count,complete:count===DATA.cases.length};if(MODE==='pass2')core.locked_pass1_export=lockedPass1;return{...core,export_timestamp:new Date().toISOString(),canonical_review_digest:digestCanonical(core)}}
function exportJson(){const x=exportObject(),prefix=MODE==='browse'?'exploratory_review':'pass2_review';download(`${prefix}_${state.reviewer_id||'reviewer'}.json`,JSON.stringify(x,null,2)+'\n','application/json');setMessage(x.complete?'Complete review exported.':'Incomplete checkpoint exported.')}
function exportCsv(){const x=exportObject(),heads=['dataset_id','dataset_digest','review_schema_version','reviewer_id',...Object.keys(blank(DATA.cases[0]))];const lines=[heads.map(escCsv).join(',')];x.reviews.forEach(r=>lines.push(heads.map(h=>escCsv(Array.isArray(r[h])?r[h].join(';'):r[h]??x[h]??'')).join(',')));const prefix=MODE==='browse'?'exploratory_review':'pass2_review';download(`${prefix}_${state.reviewer_id||'reviewer'}.csv`,lines.join('\n')+'\n','text/csv')}
function unlock(file){const reader=new FileReader();reader.onload=()=>{try{const x=JSON.parse(reader.result),issues=validatePass1Export(x,true);if(issues.length)throw new Error(issues.join(' '));lockedPass1=x;$('gate').classList.add('hidden');$('workspace').classList.remove('hidden');setMessage('Valid complete Pass 1 imported. Pass 1 is locked and system evidence is unlocked.');setupNavigator();render()}catch(e){setMessage(`Pass 2 remains locked: ${e.message}`,true)}};reader.readAsText(file)}
function resumeReview(file){const reader=new FileReader();reader.onload=()=>{try{const x=JSON.parse(reader.result),reviewIssues=validateRevealedExport(x);if(reviewIssues.length)throw new Error(reviewIssues.join(' '));if(MODE==='pass2'){const issues=validatePass1Export(x.locked_pass1_export,true);if(issues.length)throw new Error(`Locked Pass 1 is invalid: ${issues.join(' ')}`);lockedPass1=x.locked_pass1_export;$('gate').classList.add('hidden');$('workspace').classList.remove('hidden')}state.reviewer_id=x.reviewer_id||'';state.reviews=Object.fromEntries(x.reviews.map(r=>[r.audit_id,r]));$('reviewer').value=state.reviewer_id;visible=DATA.cases.slice();index=0;setupNavigator();save();render();setMessage('Review checkpoint imported and resumed.')}catch(e){setMessage(`Review import rejected: ${e.message}`,true)}};reader.readAsText(file)}
function setupNavigator(){const nav=$('navigator');nav.replaceChildren();visible.forEach((c,i)=>{const o=el('option','',`${i+1}. ${c.audit_id} · ${c.model_id}/${c.reaction_id}`);o.value=c.audit_id;nav.append(o)});nav.onchange=e=>{index=visible.findIndex(c=>c.audit_id===e.target.value);render()}}
function applyFilters(){if(MODE!=='browse')return;const pick=id=>$(id).value,truth=pick('f-truth'),outcome=pick('f-outcome'),effect=pick('f-effect'),seen=pick('f-seen'),stratum=pick('f-stratum'),model=pick('f-model'),cluster=pick('f-cluster'),formal=pick('f-formal'),compliance=pick('f-compliance');visible=DATA.cases.filter(c=>{const f=c.facets;return(!truth||f.retrieval_state===truth)&&(!outcome||f.outcome_kind===outcome)&&(!effect||f.grounded_effect===effect)&&(!seen||f.seen_unseen===seen)&&(!stratum||f.corrected_stratum===stratum)&&(!model||f.model===model)&&(!cluster||f.cluster===cluster)&&(!formal||String(f.formal_sample_inclusion)===formal)&&(!compliance||String(f.evidence_compliance_issue)===compliance)});index=0;setupNavigator();if(visible.length)render();else{$('case').replaceChildren(el('p','warning','No cases match these filters.'));$('caseNumber').textContent='0 cases'} }
$('reviewer').value=state.reviewer_id;$('reviewer').addEventListener('input',e=>{state.reviewer_id=e.target.value;save()});$('prev').onclick=()=>{if(index){index--;render()}};$('next').onclick=()=>{if(index<visible.length-1){index++;render()}};$('search').addEventListener('input',e=>{const q=e.target.value.trim().toLowerCase();if(!q)return;const found=visible.findIndex(c=>[c.audit_id,c.model_id,c.reaction_id].some(v=>v.toLowerCase().includes(q)));if(found>=0){index=found;render()}});$('exportJson').onclick=exportJson;$('exportCsv').onclick=exportCsv;document.addEventListener('keydown',e=>{if(e.altKey&&e.key==='ArrowLeft')$('prev').click();if(e.altKey&&e.key==='ArrowRight')$('next').click()});if($('reviewImport'))$('reviewImport').addEventListener('change',e=>{if(e.target.files[0])resumeReview(e.target.files[0])});if(MODE==='pass2'){$('pass1Import').addEventListener('change',e=>{if(e.target.files[0])unlock(e.target.files[0])});$('pass2Resume').addEventListener('change',e=>{if(e.target.files[0])resumeReview(e.target.files[0])})}else{setupNavigator();render();save();if(MODE==='browse')document.querySelectorAll('.filters select').forEach(x=>x.addEventListener('change',applyFilters))}
"""


SUPPLEMENTAL_JS = BASE_JS + r"""
const c=DATA.cases[0],STORE=`aaaim-review:supplemental:${DATA.review_schema_version}:${DATA.dataset_digest}`,REVIEWER_STORE=`${STORE}:reviewer`;let review={audit_id:c.audit_id,verdict:'',proposed_kegg_ids:'',kegg_mappable:'',confidence:'',needs_second_review:false,biological_rationale:'',supporting_sources:'',general_notes:'',complete:false};try{review={...review,...JSON.parse(localStorage.getItem(STORE)||'{}')}}catch(_e){}
function save(){localStorage.setItem(STORE,JSON.stringify(review));localStorage.setItem(REVIEWER_STORE,$('reviewer').value);$('saveStatus').textContent=`Saved locally ${new Date().toLocaleTimeString()}`;$('progressText').textContent=`${review.complete?1:0} completed · ${review.verdict==='unresolved'?1:0} unresolved`;$('progressFill').style.width=review.complete?'100%':'0%'}
function field(parent,label,key,type='text',options=[]){const w=el('div','field'),l=el('label','',label);l.htmlFor=`f-${key}`;w.append(l);let n;if(type==='select'){n=el('select');n.append(el('option','', 'Choose…'));options.forEach(v=>{const o=el('option','',v);o.value=v;o.title=SCHEMA.pass1.verdict_descriptions?.[v]||'';n.append(o)})}else if(type==='textarea')n=el('textarea');else{n=el('input');n.type=type}n.id=`f-${key}`;if(type==='checkbox')n.checked=!!review[key];else n.value=review[key]||'';n.oninput=()=>{review[key]=type==='checkbox'?n.checked:n.value;if(key==='complete'&&n.checked&&!(SCHEMA.pass1.verdicts.includes(review.verdict)&&String(review.biological_rationale||'').trim())){n.checked=false;review.complete=false;setMessage('A verdict and biological rationale are required before completion.',true)}save()};w.append(n);parent.append(w)}
const root=$('case'),nom=el('section','case-card warning');nom.append(el('h2','','Manual nomination context'),el('p','',c.manual_nomination.concern),el('p','technical',`Prior source: ${c.manual_nomination.prior_source}`),el('p','',c.manual_nomination.interpretation));root.append(nom);renderSource(root,c);const status=el('section','case-card');status.append(el('h2','','Already-existing method results'),el('p','missing',c.existing_method_results_status));root.append(status);const f=el('section','case-card review-form');f.append(el('h2','','Blank supplemental source-label review'));field(f,'Verdict','verdict','select',SCHEMA.pass1.verdicts);field(f,'Proposed KEGG IDs','proposed_kegg_ids');field(f,'KEGG-mappable?','kegg_mappable','select',SCHEMA.pass1.kegg_mappable);field(f,'Confidence','confidence','select',SCHEMA.pass1.confidence);field(f,'Needs second review','needs_second_review','checkbox');field(f,'Biological rationale','biological_rationale','textarea');field(f,'Supporting sources','supporting_sources','textarea');field(f,'General notes','general_notes','textarea');field(f,'Complete','complete','checkbox');root.append(f);
function exportObject(){const core={dataset_id:DATA.dataset_id,dataset_digest:DATA.dataset_digest,review_schema_version:DATA.review_schema_version,review_kind:'supplemental_training',reviewer_id:$('reviewer').value,reviews:[review],completion_count:review.complete?1:0,complete:!!review.complete};return{...core,export_timestamp:new Date().toISOString(),canonical_review_digest:digestCanonical(core)}}
$('reviewer').value=localStorage.getItem(REVIEWER_STORE)||'';$('reviewer').addEventListener('input',save);$('exportJson').onclick=()=>download(`supplemental_review_${$('reviewer').value||'reviewer'}.json`,JSON.stringify(exportObject(),null,2)+'\n','application/json');$('exportCsv').onclick=()=>{const x=exportObject(),heads=['dataset_id','dataset_digest','review_schema_version','reviewer_id',...Object.keys(review)];download(`supplemental_review_${$('reviewer').value||'reviewer'}.csv`,heads.map(escCsv).join(',')+'\n'+heads.map(h=>escCsv(review[h]??x[h]??'')).join(',')+'\n','text/csv')};$('import').addEventListener('change',e=>{const file=e.target.files[0];if(!file)return;const reader=new FileReader();reader.onload=()=>{try{const x=JSON.parse(reader.result),core={...x};delete core.export_timestamp;delete core.canonical_review_digest;if(x.dataset_id!==DATA.dataset_id||x.dataset_digest!==DATA.dataset_digest||x.review_schema_version!==DATA.review_schema_version||x.review_kind!=='supplemental_training'||!x.canonical_review_digest||digestCanonical(core)!==x.canonical_review_digest||!Array.isArray(x.reviews)||x.reviews.length!==1||x.reviews[0].audit_id!==c.audit_id)throw new Error('Dataset, digest, schema, or case mismatch.');const imported=x.reviews[0];if(imported.verdict&&!SCHEMA.pass1.verdicts.includes(imported.verdict))throw new Error('Invalid verdict.');if(imported.complete&&!(SCHEMA.pass1.verdicts.includes(imported.verdict)&&String(imported.biological_rationale||'').trim()))throw new Error('Completed review lacks required fields.');if(x.completion_count!==(imported.complete?1:0)||Boolean(x.complete)!==Boolean(imported.complete))throw new Error('Completion markers do not match.');review={...review,...imported};localStorage.setItem(STORE,JSON.stringify(review));localStorage.setItem(REVIEWER_STORE,x.reviewer_id||'');location.reload()}catch(err){setMessage(`Import rejected: ${err.message}`,true)}};reader.readAsText(file)});$('prev').disabled=true;$('next').disabled=true;$('caseNumber').textContent='Case 1 of 1';$('search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();$('case').classList.toggle('hidden',!!q&&![c.audit_id,c.model_id,c.reaction_id].some(v=>v.toLowerCase().includes(q)))});save();
"""


def _csp() -> str:
    return "default-src 'self' 'unsafe-inline' data: blob:; connect-src 'none'; img-src 'self' data:; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'"


def _page(
    title: str, subtitle: str, body: str, bundle: Mapping[str, Any], script: str,
    mode: str | None = None, page_schema: Mapping[str, Any] = REVIEW_SCHEMA,
) -> str:
    mode_line = f"const MODE={json.dumps(mode)};" if mode else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="{html.escape(_csp(), quote=True)}"><title>{html.escape(title)}</title><link rel="stylesheet" href="reviewer.css"></head>
<body><header><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></header><main class="shell">{body}</main>
<script>const DATA={json_dumps_script(bundle)};const SCHEMA={json_dumps_script(page_schema)};{mode_line}\n{script}</script></body></html>
"""


def _toolbar(import_control: str = "", filters: str = "", include_message: bool = True) -> str:
    message = '<div id="message" aria-live="polite"></div>' if include_message else ""
    return f"""<div class="toolbar"><div class="toolbar-row"><label for="reviewer">Reviewer ID</label><input id="reviewer" class="grow" autocomplete="off"><span id="saveStatus" class="status">Not yet saved</span></div>{filters}<div class="toolbar-row"><button id="prev" class="secondary">Previous</button><span id="caseNumber" aria-live="polite"></span><select id="navigator" class="grow" aria-label="Case navigator"></select><button id="next">Next</button><label for="search">Search</label><input id="search" placeholder="Audit, model, reaction ID">{import_control}<button id="exportJson">Export JSON</button><button id="exportCsv" class="secondary">Export CSV</button></div><div class="toolbar-row"><div class="progress-track" aria-label="Completion progress"><div id="progressFill" class="progress-fill"></div></div><span id="progressText"></span></div></div>{message}"""


def _select_filter(identifier: str, label: str, values: Sequence[str]) -> str:
    options = "".join(f'<option value="{html.escape(value, quote=True)}">{html.escape(value)}</option>' for value in values)
    return f'<label>{html.escape(label)} <select id="{identifier}"><option value="">All</option>{options}</select></label>'


def _filters(bundle: Mapping[str, Any]) -> str:
    cases = bundle["cases"]
    facets = [case["facets"] for case in cases]
    values = lambda key: sorted({str(item[key]) for item in facets})
    controls = [
        _select_filter("f-outcome", "Outcome", values("outcome_kind")),
        _select_filter("f-truth", "Truth location", values("retrieval_state")),
        _select_filter("f-effect", "Grounded effect", values("grounded_effect")),
        _select_filter("f-seen", "Train seen", values("seen_unseen")),
        _select_filter("f-stratum", "Corrected stratum", values("corrected_stratum")),
        _select_filter("f-model", "Model", values("model")),
        _select_filter("f-cluster", "Cluster", values("cluster")),
        _select_filter("f-compliance", "Compliance issue", ["true", "false"]),
        _select_filter("f-formal", "Formal sample", ["true", "false"]),
    ]
    return '<div class="toolbar-row filters">' + "".join(controls) + "</div>"


def _index_html() -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="{html.escape(_csp(), quote=True)}"><title>AAAIM human reaction-label review</title><link rel="stylesheet" href="reviewer.css"></head><body><header><h1>AAAIM human reaction-label review</h1><p>Offline, blank review interfaces built from frozen local artifacts</p></header><main class="shell"><section class="case-card"><h2>Required order of operations</h2><ol><li>Complete Formal Pass 1 first.</li><li>Export the completed Pass 1 JSON review file.</li><li>Open Formal Pass 2 and import that completed export.</li><li>Do not open browse-all until Pass 1 is complete.</li><li>Use the supplemental training case separately.</li></ol><p class="warning">Export regularly. Browser storage is convenient autosave, not a durable scientific archive.</p></section><div class="landing-grid"><section class="landing-card"><div class="eyebrow">Step 1</div><h2>Formal Pass 1</h2><p>Physically blinded source-label review. No system outputs are present in the file.</p><a class="button" href="formal_pass1.html">Open Formal Pass 1</a></section><section class="landing-card"><div class="eyebrow">Step 2</div><h2>Formal Pass 2</h2><p>Locked until a valid, complete Pass 1 JSON export is imported.</p><a class="button" href="formal_pass2.html">Open Formal Pass 2</a></section><section class="landing-card"><div class="eyebrow">After Pass 1</div><h2>Browse all 61</h2><p class="warning">Opening this page before completing Formal Pass 1 may unblind the formal audit.</p><a class="button secondary" href="browse_phase3c_noncorrect.html">Open browse-all queue</a></section><section class="landing-card"><div class="eyebrow">Separate</div><h2>Supplemental E12</h2><p>Training-label review; excluded from the formal validation audit.</p><a class="button secondary" href="supplemental_cases.html">Open supplemental case</a></section></div><p class="footer-note">No page makes network requests. All data, styles, and scripts are local. See <a href="README.md">reviewer instructions</a>.</p></main></body></html>\n"""


def _readme(bundles: Mapping[str, Mapping[str, Any]]) -> str:
    return f"""# Offline reaction-label reviewer

Open `index.html` first, normally by double-clicking it. No server, installation, login, network connection, or remote asset is required.

1. Enter a stable reviewer identifier in Formal Pass 1.
2. Review the source reaction and existing BioModels label. Drafts autosave in browser local storage under a namespace containing schema `{REVIEW_SCHEMA_VERSION}` and dataset digest `{bundles['pass1']['dataset_digest']}`.
3. Export JSON regularly; JSON is authoritative and CSV is a readable convenience copy. Browser storage is not a durable scientific archive.
4. Put exports in `benchmark/phase3/error_audit/review_work/`, which is intentionally ignored by Git. Suggested names are `pass1_review_<reviewer>.json`, `pass1_review_<reviewer>.csv`, `pass2_review_<reviewer>.json`, `pass2_review_<reviewer>.csv`, and `exploratory_review_<reviewer>.json`.
5. Pass 1 must precede Pass 2 to prevent system suggestions from influencing the source-label judgment. Pass 2 stays locked until it validates a complete 60-case Pass 1 JSON export.
6. Do not open browse-all until Formal Pass 1 is complete; browse-all exposes system outputs for overlapping cases.
7. Send the authoritative exported JSON file to LunaStarr using your normal approved project file-sharing channel. Do not paste reviews into source CSVs or commit them.
8. Two independent reviews will be compared and adjudicated in a later milestone. Nothing entered here is automatically treated as a corrected label.

The supplemental page is a separate training-label review for `BIOMD0000000013/E12`; it is not part of validation or prevalence estimates. Browse-all is an error-enriched exploratory queue and also cannot estimate prevalence.

If direct double-clicking is restricted by a particular browser policy, run `python -m http.server 8000 --directory benchmark/phase3/error_audit/reviewer` and open `http://127.0.0.1:8000/`. This fallback still uses only local files.
"""


def _build_config(bundles: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "phase3-error-audit-reviewer-build-v1", "algorithm_version": ALGORITHM_VERSION,
        "source_prompt2_commit": PROMPT2_COMMIT, "source_digests": SOURCE_DIGESTS,
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "datasets": {name: {"dataset_id": bundle["dataset_id"], "dataset_digest": bundle["dataset_digest"], "case_count": len(bundle["cases"])} for name, bundle in bundles.items()},
        "offline": True, "network_requests": 0, "external_assets": [], "api_calls": 0,
        "new_inference": False, "test_rows_loaded": 0, "test_labels_loaded": 0,
        "human_judgments_prepopulated": False, "tracked_builds_required": 2,
        "pass1_forbidden_keys": sorted(PASS1_FORBIDDEN_KEYS),
        "pass1_data_allowlist": ["display_order", "audit_id", "sample_id", "model_id", "reaction_id", "model_reaction", "existing_labels", "existing_label_provenance"],
    }


TRACKED_OUTPUTS = [
    "index.html", "formal_pass1.html", "formal_pass2.html", "browse_phase3c_noncorrect.html", "supplemental_cases.html",
    "reviewer.css", "README.md", "review_schema.json", "reviewer_build_config.json",
    "formal_pass1_data.json", "formal_pass2_data.json", "browse_phase3c_noncorrect_data.json", "supplemental_cases_data.json",
]


def build_bundle(out: Path = OUT) -> dict[str, dict[str, Any]]:
    bundles = build_data_bundles()
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(bundles["pass1"], out / "formal_pass1_data.json")
    atomic_write_json(bundles["pass2"], out / "formal_pass2_data.json")
    atomic_write_json(bundles["browse"], out / "browse_phase3c_noncorrect_data.json")
    atomic_write_json(bundles["supplemental"], out / "supplemental_cases_data.json")
    atomic_write_json(REVIEW_SCHEMA, out / "review_schema.json")
    atomic_write_json(_build_config(bundles), out / "reviewer_build_config.json")
    (out / "reviewer.css").write_text(CSS, encoding="utf-8", newline="\n")
    (out / "README.md").write_text(_readme(bundles), encoding="utf-8", newline="\n")
    (out / "index.html").write_text(_index_html(), encoding="utf-8", newline="\n")
    pass1_body = _toolbar('<label class="button secondary" for="import">Import/resume JSON</label><input id="import" type="file" accept="application/json" class="hidden">') + '<p class="warning">This file is physically blinded: only source reaction and existing-label information are present.</p><div id="case"></div><p class="footer-note">Alt+Left / Alt+Right moves between cases. Export regularly.</p>'
    pass1_schema = {"review_schema_version": REVIEW_SCHEMA_VERSION, "pass1": REVIEW_SCHEMA["pass1"]}
    (out / "formal_pass1.html").write_text(_page("Formal Pass 1 — source-label review", "Does the existing BioModels KEGG annotation accurately describe the SBML reaction?", pass1_body, bundles["pass1"], PASS1_JS, page_schema=pass1_schema), encoding="utf-8", newline="\n")
    pass2_import = '<label class="button secondary" for="reviewImport">Import/resume Pass 2</label><input id="reviewImport" type="file" accept="application/json" class="hidden">'
    pass2_body = '<section id="gate" class="case-card"><h2>Pass 2 is locked</h2><p>Import a valid, complete 60-case Pass 1 JSON export. No case-level system result is displayed before validation.</p><label class="button" for="pass1Import">Import completed Pass 1 JSON</label><input id="pass1Import" type="file" accept="application/json" class="hidden"> <label class="button secondary" for="pass2Resume">Resume Pass 2 JSON</label><input id="pass2Resume" type="file" accept="application/json" class="hidden"></section><div id="message" aria-live="polite"></div><div id="workspace" class="hidden">' + _toolbar(pass2_import, include_message=False) + '<div id="case"></div></div>'
    (out / "formal_pass2.html").write_text(_page("Formal Pass 2 — system-disagreement review", "After seeing frozen alternatives, is AAAIM’s output scientifically defensible?", pass2_body, bundles["pass2"], REVEALED_JS, "pass2"), encoding="utf-8", newline="\n")
    browse_import = '<label class="button secondary" for="reviewImport">Import/resume JSON</label><input id="reviewImport" type="file" accept="application/json" class="hidden">'
    browse_body = '<p class="warning">Opening this page before completing Formal Pass 1 may unblind the formal audit.</p><p class="warning">This is an error-enriched exploratory review queue and cannot estimate population label-error prevalence.</p>' + _toolbar(browse_import, filters=_filters(bundles["browse"])) + '<div id="case"></div>'
    (out / "browse_phase3c_noncorrect.html").write_text(_page("Browse all Phase 3C non-exact outcomes", "Exploratory queue: 22 incorrect selections and 39 abstentions", browse_body, bundles["browse"], REVEALED_JS, "browse"), encoding="utf-8", newline="\n")
    supplemental_body = '<p class="warning">Supplemental training-label review; not part of the formal 60-case validation audit and not included in validation prevalence estimates.</p><div class="toolbar"><div class="toolbar-row"><label for="reviewer">Reviewer ID</label><input id="reviewer" class="grow"><span id="saveStatus"></span></div><div class="toolbar-row"><button id="prev" class="secondary">Previous</button><span id="caseNumber"></span><button id="next" disabled>Next</button><label for="search">Search</label><input id="search" placeholder="Audit, model, reaction ID"><label class="button secondary" for="import">Import/resume JSON</label><input id="import" type="file" accept="application/json" class="hidden"><button id="exportJson">Export JSON</button><button id="exportCsv" class="secondary">Export CSV</button></div><div class="toolbar-row"><span id="progressText">0 completed · 0 unresolved</span><div class="progress-track" aria-hidden="true"><div id="progressFill" class="progress-fill"></div></div></div></div><div id="message" aria-live="polite"></div><div id="case"></div>'
    (out / "supplemental_cases.html").write_text(_page("Supplemental training-label review", "BIOMD0000000013 / E12 — nomination context, not a verdict", supplemental_body, bundles["supplemental"], SUPPLEMENTAL_JS), encoding="utf-8", newline="\n")
    return bundles


def build_twice(out: Path = OUT) -> dict[str, dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="_reviewer_a_", dir=ERROR_AUDIT) as first_name, tempfile.TemporaryDirectory(prefix="_reviewer_b_", dir=ERROR_AUDIT) as second_name:
        first, second = Path(first_name), Path(second_name)
        build_bundle(first); build_bundle(second)
        a = {name: (first / name).read_bytes() for name in TRACKED_OUTPUTS}
        b = {name: (second / name).read_bytes() for name in TRACKED_OUTPUTS}
        if a != b:
            raise ValueError("reviewer rebuilds are not byte-identical")
    bundles = build_bundle(out)
    write_artifact_manifest(out, [out / name for name in TRACKED_OUTPUTS])
    return bundles


def verify_manifest(out: Path = OUT) -> list[str]:
    manifest_path = out / "artifact_manifest.json"
    before = manifest_path.read_bytes()
    manifest = json.loads(before)
    problems: list[str] = []
    for item in manifest.get("files") or []:
        path = REPO_ROOT / item["path"]
        if not path.is_file() or sha256_portable(path) != item["sha256"]:
            problems.append(f"digest mismatch: {item['path']}")
    try:
        pass1 = json.loads((out / "formal_pass1_data.json").read_text(encoding="utf-8"))
        assert_pass1_blinded(pass1)
    except (OSError, ValueError) as exc:
        problems.append(f"Pass 1 verification failed: {exc}")
    if manifest_path.read_bytes() != before:
        problems.append("read-only verification mutated manifest")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        problems = verify_manifest(args.out)
        print(json.dumps({"n_problems": len(problems), "problems": problems}, sort_keys=True))
        return int(bool(problems))
    bundles = build_twice(args.out)
    print(json.dumps({name: {"cases": len(bundle["cases"]), "digest": bundle["dataset_digest"]} for name, bundle in bundles.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
