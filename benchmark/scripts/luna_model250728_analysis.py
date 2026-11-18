"""Rebuild and prepare a no-call grounded RRF analysis for MODEL2507280001.

This reuses the archived, label-free 541-reaction Phase 3B rankings only after
rebuilding all queries from the recorded SBML and species input and checking
exact equality. It never opens the publication supplement or any answer key.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Mapping, Sequence

import libsbml
import pandas as pd

from benchmark.scripts import phase3_retrieval as retrieval
from benchmark.scripts import phase3b_release as release
from benchmark.scripts.phase3c_grounded import (
    GroundedAnnotation,
    SYSTEM_INSTRUCTIONS,
    validate_grounded_annotation,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "MODEL2507280001"
ARCHIVE_COMMIT = "27cfe4f243d658bb5fc9391a95ada90c11e42bbf"
ARCHIVE_ROOT = f"tests/results/{MODEL_ID}_reaction_review"
ARCHIVE_SHA256 = "3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53"
CHECKPOINT_SHA256 = "8773b04f09916889b74c956e708044fe2c653fa764fe73c422ecc376ecae81c1"
FUSION_TOP100_ROWS = "phase3b_fusion_rankings_top100.jsonl"
BM25_TOP100_ROWS = "phase3b_bm25_rankings_top100.jsonl"
TRAINED_TOP100_ROWS = "phase3b_trained_epoch1_rankings_top100.jsonl"
TOP10_SOURCE = "phase3b_fusion_top10_candidates.csv"
REVIEW_SOURCE = "phase3b_fusion_reaction_review.csv"
SPECIES_SOURCE = "species_input.csv"
SUMMARY_SOURCE = "phase3b_full_summary.json"
PROVENANCE_SOURCE = "phase3b_artifact_provenance.json"
MAX_OUTPUT_TOKENS = 512
REASONING_EFFORT = "low"
PROMPT_VERSION = "luna-model250728-grounded-rrf-no-provenance-v1"
SCHEMA_VERSION = "phase3c-grounded-output-v1"
PROVIDER_WRAPPER_ALLOWANCE_TOKENS = 4096
PRICES = {
    "pricing_date": "2026-11-07",
    "source": "https://developers.openai.com/api/docs/pricing",
    "model_page": "https://developers.openai.com/api/docs/models/gpt-6.1-terra",
    "standard_input_per_million": Decimal("1.00"),
    "standard_output_per_million": Decimal("5.00"),
    "implicit_30m_cache_write_per_million": Decimal("1.25"),
    "long_context_threshold": 272_000,
    "long_context_input_multiplier": Decimal("2"),
    "long_context_output_multiplier": Decimal("1.5"),
}
ARCHIVED_FILES = (
    "phase3b_bm25_rankings_top100.jsonl",
    "phase3b_trained_epoch1_rankings_top100.jsonl",
    "phase3b_fusion_rankings_top100.jsonl",
    TOP10_SOURCE,
    REVIEW_SOURCE,
    SPECIES_SOURCE,
    SUMMARY_SOURCE,
    PROVENANCE_SOURCE,
    "reaction_review.csv",
    "run_phase3_review.py",
    "run_phase3b_full_review.py",
)
VISIBLE_EVIDENCE_FIELDS = (
    "evidence_id", "kegg_id", "fused_rank", "fused_score", "bm25_rank",
    "trained_biencoder_rank", "name", "definition", "equation", "enzyme",
    "rclass", "brite",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)).decode("utf-8") + "\n")
    os.replace(tmp, path)


def git_blob(commit: str, relpath: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relpath}"], cwd=REPO_ROOT)


def export_archived_sources(out: Path) -> dict[str, dict[str, Any]]:
    source_dir = out / "reused_source_artifacts"
    source_dir.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name in ARCHIVED_FILES:
        data = git_blob(ARCHIVE_COMMIT, f"{ARCHIVE_ROOT}/{name}")
        target = source_dir / name
        target.write_bytes(data)
        entries[name] = {"path": target.relative_to(out).as_posix(), "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest(),
                          "source_commit": ARCHIVE_COMMIT,
                          "source_path": f"{ARCHIVE_ROOT}/{name}"}
    return entries


def _stoich(ref: Any) -> str:
    coefficient = float(ref.getStoichiometry())
    return ("" if abs(coefficient - 1.0) < 1e-12 else f"{coefficient:g} ") + str(ref.getSpecies())


def build_queries(model_path: Path, species_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    document = libsbml.readSBMLFromFile(str(model_path))
    model = document.getModel()
    if model is None or document.getNumErrors() != 0:
        raise ValueError("SBML must parse without errors")
    species_frame = pd.read_csv(species_path, dtype=str).fillna("")
    species_annotations = {
        str(row.id): str(row.annotation)
        for row in species_frame.itertuples()
        if str(row.annotation).upper().startswith("CHEBI:")
    }
    species_names = {str(x.getId()): str(x.getName() or x.getId()) for x in model.getListOfSpecies()}
    model_species = {str(x.getId()) for x in model.getListOfSpecies()}
    unknown_species = set(species_annotations) - model_species
    if unknown_species:
        raise ValueError(f"species recommendations include IDs absent from the SBML: {sorted(unknown_species)[:5]}")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for reaction in model.getListOfReactions():
        reaction_id = str(reaction.getId())
        lhs = " + ".join(_stoich(ref) for ref in reaction.getListOfReactants())
        rhs = " + ".join(_stoich(ref) for ref in reaction.getListOfProducts())
        equation = f"{lhs} {'<=>' if reaction.getReversible() else '=>'} {rhs}"
        references = list(reaction.getListOfReactants()) + list(reaction.getListOfProducts())
        participant_ids = list(dict.fromkeys(str(ref.getSpecies()) for ref in references))
        evidence = []
        for species_id in participant_ids:
            parts = [f"species={species_id}"]
            if species_id in species_annotations:
                parts.append("ChEBI=" + species_annotations[species_id])
            evidence.append(f"{species_names.get(species_id, species_id)} [{'; '.join(parts)}]")
        query = retrieval.query_text({"reaction_equation": equation,
                                      "participant_evidence": "; ".join(evidence)})
        if not query.strip():
            failures.append({"reaction_id": reaction_id, "reason": "empty query"})
        retrieval.assert_no_kegg_leakage(query, where=f"{reaction_id} query")
        if reaction_id in query:
            failures.append({"reaction_id": reaction_id, "reason": "source reaction ID appears in query"})
        occurrence_ids = [str(ref.getSpecies()) for ref in references]
        chebi_occurrences = sum(species_id in species_annotations for species_id in occurrence_ids)
        rows.append({
            "model_id": MODEL_ID,
            "reaction_id": reaction_id,
            "query": query,
            "query_sha256": digest(query.encode("utf-8")),
            "reaction_equation": equation,
            "reactant_count": len(list(reaction.getListOfReactants())),
            "product_count": len(list(reaction.getListOfProducts())),
            "participant_occurrences": len(occurrence_ids),
            "participant_occurrences_with_chebi": chebi_occurrences,
            "participants_without_chebi": len(occurrence_ids) - chebi_occurrences,
            "reaction_name_for_display_only": str(reaction.getName() or ""),
        })
    if failures:
        raise ValueError(f"query-construction failures: {failures[:10]}")
    rows.sort(key=lambda row: row["reaction_id"])
    summary = {"model_id": str(model.getId()), "species_count": model.getNumSpecies(),
               "reaction_count": model.getNumReactions(), "recommendation_rows": len(species_frame),
               "species_with_chebi_recommendation": len(species_annotations),
               "model_species_without_chebi_recommendation": len(model_species - set(species_annotations)),
               "query_count": len(rows), "query_failures": len(failures),
               "reactions_with_no_chebi_participant_occurrence": sum(
                   row["participant_occurrences_with_chebi"] == 0 for row in rows)}
    return rows, summary


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _row_map(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("model_id") != MODEL_ID:
            raise ValueError(f"unexpected model ID in {label}")
        rid = str(row["reaction_id"])
        if rid in result:
            raise ValueError(f"duplicate reaction ID in {label}: {rid}")
        result[rid] = dict(row)
    return result


def _read_features() -> dict[str, dict[str, Any]]:
    import lzma
    import pickle
    return pickle.loads(lzma.open(retrieval.CATALOG, "rb").read())


def verify_reused_retrieval(source_dir: Path, queries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    q_by_id = {str(row["reaction_id"]): row for row in queries}
    old_review = pd.read_csv(source_dir / REVIEW_SOURCE, dtype=str).fillna("")
    old_query_by_id = {str(row.reaction_id): str(row.query_text) for row in old_review.itertuples()}
    if set(old_query_by_id) != set(q_by_id):
        raise ValueError("archived query population does not align with the current 541-reaction SBML")
    changed = [rid for rid, query in q_by_id.items() if old_query_by_id[rid] != query["query"]]
    if changed:
        raise ValueError(f"archived query text differs for {len(changed)} reactions: {changed[:5]}")

    bm_rows = _row_map(_jsonl(source_dir / "phase3b_bm25_rankings_top100.jsonl"), "BM25")
    trained_rows = _row_map(_jsonl(source_dir / "phase3b_trained_epoch1_rankings_top100.jsonl"), "trained bi-encoder")
    fused_rows = _row_map(_jsonl(source_dir / FUSION_TOP100_ROWS), "RRF")
    if any(set(rows) != set(q_by_id) for rows in (bm_rows, trained_rows, fused_rows)):
        raise ValueError("a retrieval component does not contain exactly the 541 target reactions")
    for label, rows in (("BM25", bm_rows), ("trained bi-encoder", trained_rows), ("RRF", fused_rows)):
        for rid, row in rows.items():
            ids = list(row.get("ranked_ids", []))
            if len(ids) != 100 or len(ids) != len(set(ids)):
                raise ValueError(f"{label} ranking is not 100 unique candidates for {rid}")

    fusion = pd.read_csv(source_dir / TOP10_SOURCE, dtype=str).fillna("")
    if len(fusion) != 5410:
        raise ValueError(f"archived Top-10 evidence row count changed: {len(fusion)} != 5410")
    if fusion.duplicated(["reaction_id", "candidate_rank"]).any():
        raise ValueError("duplicate Top-10 evidence positions")
    top10_by_id: dict[str, list[dict[str, Any]]] = {}
    for rid, group in fusion.groupby("reaction_id", sort=False):
        group = group.sort_values("candidate_rank", key=lambda s: s.astype(int))
        rows = group.to_dict("records")
        if len(rows) != 10:
            raise ValueError(f"wrong evidence count for {rid}: {len(rows)}")
        if [str(row["candidate_kegg"]) for row in rows] != fused_rows[str(rid)]["ranked_ids"][:10]:
            raise ValueError(f"Top-10 evidence order does not match fused ranking for {rid}")
        top10_by_id[str(rid)] = rows

    catalog = _read_features()
    catalog_ids = [str(row["kegg_id"]) for row in retrieval.load_catalog()]
    if len(catalog) != 12_312 or len(catalog_ids) != 12_312 or set(catalog) != set(catalog_ids):
        raise ValueError("frozen KEGG catalog identifiers or size mismatch")

    out_rows: list[dict[str, Any]] = []
    for rid in sorted(q_by_id):
        bm = bm_rows[rid]["ranked_ids"]
        tr = trained_rows[rid]["ranked_ids"]
        bm_rank = {kid: rank for rank, kid in enumerate(bm, 1)}
        tr_rank = {kid: rank for rank, kid in enumerate(tr, 1)}
        for source in top10_by_id[rid]:
            kid = str(source["candidate_kegg"])
            score = round((1 / (60 + bm_rank[kid]) if kid in bm_rank else 0.0)
                          + (1 / (60 + tr_rank[kid]) if kid in tr_rank else 0.0), 12)
            if abs(float(source["fusion_score"]) - score) > 1e-12:
                raise ValueError(f"RRF score mismatch for {rid}/{kid}")
            fields = catalog.get(kid)
            if fields is None:
                raise ValueError(f"Top-10 candidate missing from frozen catalog: {kid}")
            out_rows.append({
                "model_id": MODEL_ID, "reaction_id": rid,
                "evidence_id": f"E{int(source['candidate_rank']):02d}",
                "fused_rank": int(source["candidate_rank"]), "kegg_id": kid,
                "fused_score": score, "bm25_rank": bm_rank.get(kid),
                "trained_biencoder_rank": tr_rank.get(kid),
                "name": str(fields.get("NAME") or ""),
                "definition": str(fields.get("DEFINITION") or ""),
                "equation": str(fields.get("EQUATION") or ""),
                "enzyme": str(fields.get("ENZYME") or ""),
                "rclass": str(fields.get("RCLASS") or ""),
                "brite": str(fields.get("BRITE") or ""),
            })
    return {"bm25": bm_rows, "trained": trained_rows, "fused": fused_rows,
            "top10_by_id": top10_by_id, "evidence_rows": out_rows,
            "counts": {"reaction_count": len(q_by_id), "bm25_top100_rows": len(bm_rows),
                       "trained_top100_rows": len(trained_rows), "fused_top100_rows": len(fused_rows),
                       "top10_evidence_rows": len(out_rows), "catalog_size": len(catalog)}}


def complete_fusion_rows(components: Mapping[str, Any], catalog_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Reconstruct the frozen full permutation and all nonzero RRF scores from Top-100 inputs."""
    rows = []
    for rid in sorted(components["fused"]):
        bm_ids = components["bm25"][rid]["ranked_ids"]
        tr_ids = components["trained"][rid]["ranked_ids"]
        full = release.reciprocal_rank_fusion_complete([bm_ids, tr_ids], catalog_ids, k=60)
        expected_top = components["fused"][rid]["ranked_ids"]
        if full[:len(expected_top)] != expected_top:
            raise ValueError(f"reconstructed full-catalog RRF top 100 differs from frozen ranking for {rid}")
        bm_rank = {kid: i for i, kid in enumerate(bm_ids, 1)}
        tr_rank = {kid: i for i, kid in enumerate(tr_ids, 1)}
        score_by_id = {
            kid: round((1 / (60 + bm_rank[kid]) if kid in bm_rank else 0.0)
                       + (1 / (60 + tr_rank[kid]) if kid in tr_rank else 0.0), 12)
            for kid in set(bm_ids) | set(tr_ids)
        }
        rows.append({"model_id": MODEL_ID, "reaction_id": rid,
                     "ranked_ids": full, "nonzero_scores": score_by_id,
                     "zero_score_tail_rule": "all omitted component IDs score 0 and are sorted ascending by KEGG ID"})
    return rows


def render_grounded_input(query: str, evidence: Sequence[Mapping[str, Any]]) -> str:
    visible = [{field: row[field] for field in VISIBLE_EVIDENCE_FIELDS} for row in evidence]
    return "TARGET\n" + json.dumps({"query": query}, sort_keys=True, ensure_ascii=False) + "\nEVIDENCE\n" + json.dumps(visible, sort_keys=True, ensure_ascii=False)


def serialize_provider_body(query: str, evidence: Sequence[Mapping[str, Any]], *, model: str = "gpt-6.1-terra") -> dict[str, Any]:
    """Use the installed SDK's Responses.parse serializer without a client or network."""
    from openai.resources.responses import Responses
    resource = object.__new__(Responses)
    def capture(url: str, *, body: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        # The SDK carries parameters unknown to this installed version in
        # request options; merge them exactly as the HTTP transport does.
        options = kwargs.get("options", {})
        body.update(options.get("extra_json") or {})
        return body
    resource._post = capture
    return resource.parse(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=render_grounded_input(query, evidence),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning={"effort": REASONING_EFFORT},
        store=False,
        tools=[],
        text_format=GroundedAnnotation,
        extra_body={"service_tier": "default"},
    )


def request_cost_bound(body: Mapping[str, Any]) -> dict[str, Any]:
    body_bytes = len(canonical_json(dict(body)))
    input_bound = body_bytes + PROVIDER_WRAPPER_ALLOWANCE_TOKENS
    long = input_bound > PRICES["long_context_threshold"]
    input_rate = PRICES["implicit_30m_cache_write_per_million"] * (PRICES["long_context_input_multiplier"] if long else 1)
    output_rate = PRICES["standard_output_per_million"] * (PRICES["long_context_output_multiplier"] if long else 1)
    usd = (Decimal(input_bound) * input_rate + Decimal(MAX_OUTPUT_TOKENS) * output_rate) / Decimal(1_000_000)
    nanos = int((usd * Decimal(1_000_000_000)).to_integral_value(rounding=ROUND_CEILING))
    return {"canonical_provider_body_bytes": body_bytes,
            "input_token_upper_bound": input_bound,
            "max_output_tokens_including_reasoning": MAX_OUTPUT_TOKENS,
            "applied_input_rate_per_million_usd": str(input_rate),
            "applied_output_rate_per_million_usd": str(output_rate),
            "long_context_pricing": long,
            "reservation_nusd": nanos,
            "reservation_usd": f"{Decimal(nanos) / Decimal(1_000_000_000):.9f}"}


def unit_cache_key(unit: Mapping[str, str], model: str, body_digest: str) -> str:
    return digest({"cache_schema": "luna-model250728-unit-v1", "unit": dict(unit),
                   "model": model, "complete_provider_body_sha256": body_digest})


def compare_public_rulebased(
    selected: Path, rrf_top10: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    public_file = selected.parent / "public_reaction_audit" / "public_candidate_recommendations.csv"
    if not public_file.is_file():
        raise FileNotFoundError(f"separately named public-workflow baseline missing: {public_file}")
    frame = pd.read_csv(public_file, dtype=str).fillna("")
    baseline: dict[str, list[str]] = {}
    for rid, group in frame.groupby("id", sort=False):
        candidates = [(int(row["candidate_rank"] or 0), row["annotation"].removeprefix("KEGG:"))
                      for row in group.to_dict("records") if row["annotation"].strip()]
        baseline[str(rid)] = [kid for _, kid in sorted(candidates) if kid]
    rows = []
    for rid in sorted(rrf_top10):
        public_ids = baseline.get(rid, [])
        fused_ids = [str(row["candidate_kegg"]) for row in rrf_top10[rid]]
        intersection = set(fused_ids) & set(public_ids)
        union = set(fused_ids) | set(public_ids)
        rows.append({"reaction_id": rid,
                     "rrf_top10_count": len(fused_ids),
                     "public_rulebased_candidate_count": len(public_ids),
                     "comparison_status": "aligned candidate sets" if public_ids else "public method had no candidate; not a disagreement",
                     "top1_agreement": bool(public_ids and fused_ids[0] == public_ids[0]),
                     "top10_intersection_count": len(intersection),
                     "top10_jaccard": (len(intersection) / len(union)) if union else None})
    aligned = [row for row in rows if row["public_rulebased_candidate_count"]]
    summary = {"reaction_population": len(rows), "rrf_top10_available": sum(row["rrf_top10_count"] == 10 for row in rows),
               "public_rulebased_candidate_bearing": len(aligned),
               "public_rulebased_no_candidate": len(rows) - len(aligned),
               "top1_agreement_count": sum(row["top1_agreement"] for row in aligned),
               "top1_agreement_denominator": len(aligned),
               "mean_top10_intersection_aligned": (sum(row["top10_intersection_count"] for row in aligned) / len(aligned)) if aligned else None,
               "mean_top10_jaccard_aligned": (sum(row["top10_jaccard"] for row in aligned) / len(aligned)) if aligned else None,
               "interpretation": "method overlap only; candidate agreement is not correctness"}
    return rows, summary


def representative_review_queue(
    source_review: pd.DataFrame, top10: Mapping[str, Sequence[Mapping[str, Any]]], queries: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    desired = {
        "top1_transformation_supported": 2,
        "plausible_candidate_misranked": 2,
        "weak_candidate_only": 2,
        "wrong_transformation_or_retrieval_miss": 2,
        "insufficient_species_mapping": 2,
        "not_applicable_transport": 1,
        "not_applicable_biomass": 1,
        "not_applicable_model_pseudo_reaction": 1,
    }
    review = source_review.fillna("").set_index("reaction_id", drop=False)
    output = []
    for status, count in desired.items():
        ids = sorted(str(x) for x in review.index if str(review.at[x, "assessment"]) == status)
        for rid in ids[:count]:
            prior = review.loc[rid]
            output.append({
                "reaction_id": rid,
                "reaction_equation": queries[rid]["reaction_equation"],
                "target_query": queries[rid]["query"],
                "top10_evidence": json.dumps(list(top10[rid]), ensure_ascii=False, sort_keys=True),
                "grounded_prediction": "",
                "grounded_abstention": "pending inference",
                "current_biological_review_status": "not reviewed in this analysis",
                "prior_archived_review_assessment": status,
                "prior_review_source": f"archived 541-reaction Phase 3B case study, commit {ARCHIVE_COMMIT}; reviewer identity/independence not established",
                "review_question": "Does any supplied Top-10 record represent this reaction at the right specificity? If yes, which evidence ID supports the transformation? If none, is abstention appropriate, and what participant or stoichiometric evidence is missing?",
            })
    return output


def build_analysis(*, archive: Path, output: Path | None = None) -> Path:
    model_path = REPO_ROOT / "tests/useCase/MODEL2507280001.xml"
    species_path = REPO_ROOT / "tests/useCase/results/MODEL2507280001/full_source_fidelity_species.csv"
    if output is None:
        output = REPO_ROOT / "tests/useCase/results/MODEL2507280001/selected_rrf_analysis"
    archive = archive.resolve()
    if not archive.is_file() or sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("selected epoch-1 inference archive is missing or has an unexpected checksum")
    with zipfile.ZipFile(archive) as bundle:
        inference_config = json.loads(bundle.read("inference_config.json"))
        archive_manifest = json.loads(bundle.read("archive_manifest.json"))
        payload_hashes = {item["path"]: item["sha256"] for item in archive_manifest["files"]}
        if inference_config["source_checkpoint"]["sha256"] != CHECKPOINT_SHA256:
            raise ValueError("selected epoch-1 checkpoint provenance mismatch")
        if any(hashlib.sha256(bundle.read(name)).hexdigest() != expected for name, expected in payload_hashes.items()):
            raise ValueError("selected inference archive has a payload checksum mismatch")
    fusion_config_path = REPO_ROOT / "benchmark/phase3/phase3b_fusion/config.json"
    fusion_config = json.loads(fusion_config_path.read_text(encoding="utf-8"))
    if fusion_config["method"] != "bm25_trained_epoch1_rrf" or fusion_config["rrf"]["k"] != 60:
        raise ValueError("selected Phase 3B fusion configuration mismatch")
    output.mkdir(parents=True, exist_ok=False)
    source_manifest = export_archived_sources(output)
    source_dir = output / "reused_source_artifacts"

    historical_model = git_blob(ARCHIVE_COMMIT, "tests/useCase/MODEL2507280001.xml")
    current_model = model_path.read_bytes()
    if historical_model.replace(b"\r\n", b"\n") != current_model.replace(b"\r\n", b"\n"):
        raise ValueError("current SBML differs from historical model beyond CRLF serialization")
    model_hash = sha256_file(model_path)
    species_hash = sha256_file(source_dir / SPECIES_SOURCE)
    catalog_hash = sha256_file(retrieval.CATALOG)
    if catalog_hash != "00acb4de1bbfb1ddae0298a09d218f93eb22241844179439ff34a2912fba046c":
        raise ValueError("frozen catalog does not match archived selected-method catalog")

    query_rows, query_summary = build_queries(model_path, source_dir / SPECIES_SOURCE)
    components = verify_reused_retrieval(source_dir, query_rows)
    evidence_rows = components["evidence_rows"]
    evidence_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in evidence_rows:
        evidence_by_id.setdefault(row["reaction_id"], []).append({k: row[k] for k in VISIBLE_EVIDENCE_FIELDS})
    for values in evidence_by_id.values():
        values.sort(key=lambda row: row["fused_rank"])

    # Reconstruct the selected full-catalog RRF permutation from the frozen Top-100 components.
    catalog_ids = [row["kegg_id"] for row in retrieval.load_catalog()]
    full_fused = complete_fusion_rows(components, catalog_ids)
    if any(len(row["ranked_ids"]) != 12_312 or len(set(row["ranked_ids"])) != 12_312 for row in full_fused):
        raise ValueError("complete fusion reconstruction is not a unique full-catalog permutation")

    previous_review = pd.read_csv(source_dir / REVIEW_SOURCE, dtype=str).fillna("")
    review_by_id = {str(row["reaction_id"]): row for row in previous_review.to_dict("records")}
    if set(review_by_id) != {row["reaction_id"] for row in query_rows}:
        raise ValueError("prior review table does not cover the full SBML reaction population")

    # Preserve exact reused rows with source hashes; add newly reconstructed query digests and provenance.
    atomic_jsonl(output / "queries.jsonl", query_rows)
    atomic_jsonl(output / "rankings_bm25_top100.jsonl", [components["bm25"][rid] for rid in sorted(components["bm25"])])
    atomic_jsonl(output / "rankings_trained_biencoder_epoch1_top100.jsonl", [components["trained"][rid] for rid in sorted(components["trained"])])
    atomic_jsonl(output / "rankings_fused_top100.jsonl", [components["fused"][rid] for rid in sorted(components["fused"])])
    # Write the full permutation as compressed canonical JSONL.
    import lzma
    temp_full = output / "rankings_fused_complete.jsonl.xz.tmp"
    with lzma.open(temp_full, "wt", encoding="utf-8", newline="\n") as handle:
        for row in full_fused:
            handle.write(canonical_json(row).decode("utf-8") + "\n")
    os.replace(temp_full, output / "rankings_fused_complete.jsonl.xz")
    (output / "rankings_fused_complete.jsonl.xz.tmp").unlink(missing_ok=True)
    top10_rows = [{k: row[k] for k in ("model_id", "reaction_id", "evidence_id", "fused_rank", "kegg_id", "fused_score", "bm25_rank", "trained_biencoder_rank", "name", "definition", "equation", "enzyme", "rclass", "brite")} for row in evidence_rows]
    atomic_jsonl(output / "top10_evidence.jsonl", top10_rows)

    query_by_id = {row["reaction_id"]: row for row in query_rows}
    coverage = []
    for rid in sorted(query_by_id):
        prior = review_by_id[rid]
        q = query_by_id[rid]
        top10 = evidence_by_id[rid]
        coverage.append({
            "model_id": MODEL_ID,
            "reaction_id": rid,
            "query_constructed": True,
            "query_sha256": q["query_sha256"],
            "participant_occurrences": q["participant_occurrences"],
            "participant_occurrences_with_chebi": q["participant_occurrences_with_chebi"],
            "participants_without_chebi": q["participants_without_chebi"],
            "retrieval_eligible": True,
            "fused_candidate_count": len(top10),
            "grounded_request_eligible": len(top10) == 10,
            "special_scope_flag_from_prior_review": prior["assessment"] if str(prior["assessment"]).startswith("not_applicable") else "",
            "prior_mapped_evidence_flag": prior["assessment"] if prior["assessment"] == "insufficient_species_mapping" else "",
            "biological_evidence_sufficiency": "not adjudicated in this analysis",
            "prediction_status": "not run; paid calls unauthorized",
            "abstention_status": "not run",
        })
    pd.DataFrame(coverage).to_csv(output / "coverage_541.csv", index=False, encoding="utf-8")

    # Exact SDK serialization happens locally; this function does not construct a client or send a request.
    request_rows = []
    manifest_rows = []
    sidecars = []
    for query in query_rows:
        rid = query["reaction_id"]
        body = serialize_provider_body(query["query"], evidence_by_id[rid])
        body_digest = digest(body)
        unit = {"model_id": MODEL_ID, "reaction_id": rid}
        cache_key = unit_cache_key(unit, "gpt-6.1-terra", body_digest)
        billing = request_cost_bound(body)
        request_rows.append({"unit": unit, "unit_cache_key": cache_key, "provider_payload": body})
        manifest_rows.append({"unit": unit, "model": "gpt-6.1-terra", "api": "responses.parse",
                              "prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION,
                              "provider_payload_sha256": body_digest, "unit_cache_key": cache_key,
                              "compatible_cached_response": False, "max_attempts": 1,
                              "automatic_retries": 0, **billing})
        sidecars.append({"unit": unit, "query_sha256": query["query_sha256"],
                         "species_recommendation_path": "reused_source_artifacts/species_input.csv",
                         "species_recommendation_sha256": species_hash,
                         "sbml_path": "tests/useCase/MODEL2507280001.xml", "sbml_sha256": model_hash,
                         "catalog_path": "data/kegg/kegg_reaction_features.lzma", "catalog_sha256": catalog_hash,
                         "retrieval_method": "bm25_trained_epoch1_rrf", "rrf_k": 60,
                         "fusion_ranking_source_sha256": source_manifest[FUSION_TOP100_ROWS]["sha256"],
                         "bm25_ranking_source_sha256": source_manifest[BM25_TOP100_ROWS]["sha256"],
                         "trained_ranking_source_sha256": source_manifest[TRAINED_TOP100_ROWS]["sha256"],
                         "checkpoint_archive_sha256": ARCHIVE_SHA256,
                         "source_case_study_commit": ARCHIVE_COMMIT})
    atomic_jsonl(output / "requests.jsonl", request_rows)
    atomic_jsonl(output / "request_manifest.jsonl", manifest_rows)
    atomic_jsonl(output / "provenance_sidecars.jsonl", sidecars)
    dry_run = verify_provider_plan(request_rows, manifest_rows)
    atomic_json(output / "dry_run_report.json", {
        **dry_run, "provider_transport_invoked": False,
        "schema_serialization": "OpenAI Python SDK Responses.parse; strict GroundedAnnotation JSON schema",
        "service_tier": "default", "paid_calls_authorized": False,
    })

    overlap_rows, overlap_summary = compare_public_rulebased(output, components["top10_by_id"])
    pd.DataFrame(overlap_rows).to_csv(output / "method_overlap_541.csv", index=False, encoding="utf-8")
    atomic_json(output / "method_overlap_summary.json", overlap_summary)
    queue = representative_review_queue(previous_review, components["top10_by_id"], query_by_id)
    atomic_jsonl(output / "representative_human_review_queue.jsonl", queue)

    total_reservation = sum(int(row["reservation_nusd"]) for row in manifest_rows)
    proposed_cap_nusd = 12_000_000_000
    if total_reservation > proposed_cap_nusd:
        cap_fits = False
    else:
        cap_fits = True
    summary = {
        "schema": "luna-model250728-selected-rrf-analysis-v1",
        "status": "retrieval reused and verified; grounded requests prepared; no API calls",
        "population": query_summary,
        "retrieval": {**components["counts"], "full_catalog_rankings_reconstructed": len(full_fused),
                      "full_catalog_size": len(catalog_ids), "top10_evidence_rows": len(top10_rows),
                      "complete_fused_order_rows": len(full_fused),
                      "full_fusion_formula": "sum 1/(60+rank) over BM25 and trained epoch-1 Top-100; omitted IDs score 0; score ties by KEGG ID ascending",
                      "component_depth": 100, "rrf_k": 60, "weights": "equal"},
        "reuse_validation": {"current_sbml_crlf_normalized_identical_to_archived": True,
                             "reconstructed_query_text_equal_for_all_541": True,
                             "retrieval_rankings_541_aligned": True,
                             "top10_evidence_matches_fused_order_and_scores": True,
                             "catalog_sha256": catalog_hash,
                             "species_recommendation_file_sha256": species_hash,
                             "historical_archive_commit": ARCHIVE_COMMIT,
                             "retrieval_recomputed": False},
        "grounded_plan": {"eligible_reactions": len(query_rows), "request_count": len(request_rows),
                          "compatible_cached_responses": 0, "new_calls_required_if_authorized": len(request_rows),
                          "actual_calls": 0, "model": "gpt-6.1-terra", "api": "Responses API",
                          "reasoning_effort": REASONING_EFFORT, "max_output_tokens": MAX_OUTPUT_TOKENS,
                          "strict_schema": "GroundedAnnotation / phase3c-grounded-output-v1",
                          "automatic_retries": 0, "max_attempts": len(request_rows),
                          "estimated_reserved_usd": f"{Decimal(total_reservation) / Decimal(1_000_000_000):.9f}",
                          "proposed_hard_cap_usd": "12.00", "proposed_cap_covers_all_reservations": cap_fits,
                          "account_balance_verified": False,
                          "execution_state": "prepared only; no response cache or attempt ledger entries created"},
        "prior_review_context": {"assessment_counts": previous_review["assessment"].value_counts().sort_index().to_dict(),
                                 "archived_review_population": len(previous_review),
                                 "eligible_transformation_review_n": 264,
                                 "strict_top1_supported": 182,
                                 "strict_top10_supported": 215,
                                 "attribution": "archived single-team rubric report; reviewer identity/independence not established; descriptive prior judgments only"},
        "artifacts": {"source_artifacts": source_manifest},
        "no_call_boundary": {"heldout_access": False, "external_biological_queries": False,
                             "paid_calls": False, "new_retrieval_inference": False,
                             "source_answer_keys_used_for_query_or_ranking": False},
    }
    atomic_json(output / "analysis_summary.json", summary)
    atomic_json(output / "retrieval_method.json", {
        "method": "BM25 + trained epoch-1 bi-encoder reciprocal-rank fusion",
        "fusion_config_path": "benchmark/phase3/phase3b_fusion/config.json",
        "fusion_config_sha256": sha256_file(fusion_config_path),
        "catalog_path": "data/kegg/kegg_reaction_features.lzma",
        "catalog_sha256": catalog_hash,
        "catalog_size": len(catalog_ids),
        "bm25": {"depth": 100, "weight": 1, "implementation": "benchmark/scripts/phase3_retrieval.py",
                 "tokenizer": retrieval.TOKENIZER_VERSION, "k1": 1.2, "b": 0.75,
                 "query_template": retrieval.QUERY_TEMPLATE_VERSION,
                 "document_template": retrieval.DOCUMENT_TEMPLATE_VERSION},
        "trained_biencoder": {"depth": 100, "weight": 1, "model": inference_config["model"],
                 "initializer_revision": inference_config["initializer_revision"], "selected_epoch": 1,
                 "source_checkpoint_sha256": inference_config["source_checkpoint"]["sha256"],
                 "archive_sha256": sha256_file(archive), "tokenizer_files_sha256": {
                     name: payload_hashes[name] for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json")},
                 "pooling": inference_config["pooling"], "normalization": inference_config["normalization"],
                 "similarity": inference_config["similarity"], "max_sequence_length": inference_config["max_sequence_length"]},
        "rrf": fusion_config["rrf"], "query_template": inference_config["query_template"],
        "historical_rankings_reused": True, "new_gpu_or_encoder_inference": False,
        "archived_ranking_code_sha256": {name: source_manifest[name]["sha256"]
             for name in ("run_phase3b_full_review.py", "run_phase3_review.py")},
        "selected_method_code_sha256": {
            name: hashlib.sha256(git_blob(ARCHIVE_COMMIT, name)).hexdigest()
            for name in ("benchmark/scripts/phase3_retrieval.py", "benchmark/scripts/phase3_biencoder.py", "benchmark/scripts/phase3b_release.py")},
    })
    atomic_json(output / "validation_report.json", {
        "status": "passed", "reaction_count": 541, "query_failures": 0,
        "exact_sbml_content_after_crlf_normalization": True,
        "exact_rebuilt_query_match_to_archived_query_for_all_reactions": True,
        "component_ranking_alignment": {"bm25": 541, "trained_biencoder_epoch1": 541, "rrf": 541},
        "each_component_has_100_unique_ids": True,
        "reconstructed_fused_permutation_catalog_size": 12312,
        "top10_order_and_rrf_scores_verified": True,
        "source_answer_keys_or_supplement_used_for_query_or_ranking": False,
        "archive": {"path": str(archive), "sha256": sha256_file(archive),
                    "selected_checkpoint_sha256": inference_config["source_checkpoint"]["sha256"],
                    "payload_files_verified": len(payload_hashes)},
        "inputs": {"current_sbml": {"path": "tests/useCase/MODEL2507280001.xml", "sha256": model_hash},
                   "archived_species_recommendations": {"path": f"{ARCHIVE_ROOT}/{SPECIES_SOURCE}", "sha256": species_hash},
                   "frozen_catalog": {"path": "data/kegg/kegg_reaction_features.lzma", "sha256": catalog_hash}},
        "archived_rankings": {name: source_manifest[name] for name in
             (BM25_TOP100_ROWS, TRAINED_TOP100_ROWS, FUSION_TOP100_ROWS, TOP10_SOURCE, REVIEW_SOURCE)},
    })
    (output / "REPRODUCE.md").write_text(
        "# Reproduce the selected RRF case analysis\n\n"
        "From the repository/worktree root, with the Phase 3B inference archive available at "
        "`benchmark/dist/aaaim-phase3b-selected-epoch1-inference.zip`, run:\n\n"
        "```powershell\npython -c \"from pathlib import Path; from benchmark.scripts.luna_model250728_analysis import build_analysis; "+
        "build_analysis(archive=Path('benchmark/dist/aaaim-phase3b-selected-epoch1-inference.zip'))\"\n```\n\n"
        "The command performs checksum/query/ranking validation, reuses archived BM25, trained-encoder and RRF outputs, "
        "and serializes a no-call GPT-6.1 Terra request plan. It does not access the held-out split, run retrieval inference, "
        "query biological services, or call an API. Generated case-level artifacts are ignored by Git.\n",
        encoding="utf-8")
    overlap_line = (f"Public rule-based overlap is reported for {overlap_summary['public_rulebased_candidate_bearing']} aligned candidate-bearing cases; "
                    f"Top-1 ID agreement is {overlap_summary['top1_agreement_count']}/{overlap_summary['top1_agreement_denominator']}. "
                    f"The public method's {overlap_summary['public_rulebased_no_candidate']} no-candidate cases are retained as a separate method outcome, not treated as retrieval exclusions.")
    (output / "REPORT.md").write_text(
        "# MODEL2507280001 selected RRF analysis\n\n"
        "## Scope and method\n\n"
        "This no-call analysis uses the selected BM25 plus trained epoch-1 bi-encoder equal-weight RRF method (`k=60`) over the full frozen 12,312-reaction KEGG catalog. Both component lists are depth 100; ties use ascending KEGG reaction ID. The bi-encoder is BAAI/bge-small-en-v1.5 with the validation-selected epoch-1 checkpoint, CLS pooling, L2 normalization, inner-product similarity, and max sequence length 256. BM25 uses the project-native biochemical regex tokenizer, `k1=1.2`, and `b=0.75`.\n\n"
        "Historical component rankings and Top-10 evidence were reused after validating their checksums and query identity; no GPU/encoder inference was rerun. Every one of 541 SBML reactions has a reconstructed, nonempty query and Top-10 evidence. The SBML agrees byte-for-byte with historical input after CRLF normalization, and all 541 rebuilt target query strings exactly match archived query strings. No reactions were excluded for rule-based candidate absence or missing ChEBI; the archived source review's transport/biomass/pseudo and insufficient-mapping labels remain contextual status fields, not retrieval exclusions.\n\n"
        "## Results available now\n\n"
        "- Query construction: 541/541; failures: 0. Reactions with no mapped ChEBI participant occurrence: "
        f"{query_summary['reactions_with_no_chebi_participant_occurrence']}.\n"
        "- Full-catalog retrieval and grounded request eligibility: 541/541; Top-10 records: 5,410.\n"
        "- Grounded predictions, abstentions, and evidence-compliance outcomes: none yet; paid inference is not authorized. Candidate availability is not annotation correctness.\n"
        f"- {overlap_line}\n\n"
        "The archived single-team source review reported 182/264 strict Top-1 and 215/264 strict Top-10 support for RRF. These are prior descriptive review judgments; assessor independence is not established, and they are not gold-standard accuracy. No labels or supplement were used to construct or rank this current analysis.\n\n"
        "## Proposed inference plan (not authorized)\n\n"
        f"The exact Responses API request plan contains {len(request_rows)} requests for `gpt-6.1-terra` (low reasoning, strict `GroundedAnnotation` schema, max output 512 tokens including reasoning, `store=false`, no tools, default service tier). No compatible model response cache exists. The pessimistic reserved bound is USD {Decimal(total_reservation) / Decimal(1_000_000_000):.9f}; a USD 12 cap covers the plan, conditional on the account dashboard confirming at least that balance. User-reported approximate credit is USD 20 but is unverified. One attempt per unit; no retries. Failed or uncertain attempts retain reservations; verified completed usage settles to billed cost. No calls have been made.\n\n"
        "Hosted-model version and time drift remain possible even for identical payloads.\n\n"
        "## Reproduction and artifacts\n\n"
        "See `REPRODUCE.md`, `validation_report.json`, `retrieval_method.json`, `analysis_summary.json`, `coverage_541.csv`, `top10_evidence.jsonl`, `requests.jsonl`, `request_manifest.jsonl`, `provenance_sidecars.jsonl`, and `dry_run_report.json`. These case-level outputs are ignored by Git. The public rule-based diagnostic remains in the separately named `public_reaction_audit/` directory.\n",
        encoding="utf-8")
    atomic_json(output / "pricing_snapshot.json", {
        "checked": PRICES["pricing_date"], "source": PRICES["source"], "model_page": PRICES["model_page"],
        "standard_input_per_million_usd": str(PRICES["standard_input_per_million"]),
        "standard_output_per_million_usd": str(PRICES["standard_output_per_million"]),
        "implicit_30m_cache_write_per_million_usd": str(PRICES["implicit_30m_cache_write_per_million"]),
        "assumption": "reserve all bounded prompt input at 30-minute cache-write price, no cache-read discount, plus full output cap; explicit Standard default tier; zero retries; provider framing allowance 4,096 tokens/request",
        "approximate_credit_reported_by_user_usd": 20,
        "balance_verified": False,
        "proposed_cap_usd": 12,
        "proposed_cap_condition": "proceed only after the account dashboard confirms at least USD 12.00 available; no paid authorization is implied"})
    atomic_json(output / "execution_state_template.json", {
        "schema": "luna-unit-attempt-ledger-v1", "model": "gpt-6.1-terra",
        "plan_manifest_sha256": sha256_file(output / "request_manifest.jsonl"),
        "hard_cap_nusd": proposed_cap_nusd, "maximum_attempts_per_unit": 1,
        "automatic_retries": 0, "attempts": [],
        "billing_policy": "reserve before send; verified completed usage replaces that unit reservation; failed or uncertain attempts retain the full reservation and cannot be retried"})
    return output


class UnitCache:
    """Atomic persistent response cache whose keys are experimental-unit specific."""
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, expected_body_digest: str) -> dict[str, Any] | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("unit_cache_key") != key or row.get("provider_payload_sha256") != expected_body_digest:
            raise ValueError("cache entry does not match its exact unit and provider body")
        return row

    def put(self, key: str, body_digest: str, response: Mapping[str, Any]) -> None:
        atomic_json(self.directory / f"{key}.json", {"unit_cache_key": key,
                    "provider_payload_sha256": body_digest, "response": dict(response)})


class AttemptLedger:
    """Atomic single-attempt ledger; uncertain reservations are never released."""
    def __init__(self, path: Path, hard_cap_nusd: int):
        self.path = path
        self.hard_cap_nusd = int(hard_cap_nusd)
        if path.exists():
            self.state = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.state = {"schema": "luna-unit-attempt-ledger-v1", "attempts": []}
        if self.state.get("schema") != "luna-unit-attempt-ledger-v1":
            raise ValueError("unexpected attempt ledger schema")

    def _save(self) -> None:
        atomic_json(self.path, self.state)

    def reserved_total_nusd(self) -> int:
        return sum(int(item["actual_cost_nusd"] if item.get("actual_cost_nusd") is not None
                       else item["reservation_nusd"]) for item in self.state["attempts"])

    def reserve(self, unit_cache_key_value: str, reservation_nusd: int) -> None:
        attempts = self.state["attempts"]
        if any(item["unit_cache_key"] == unit_cache_key_value for item in attempts):
            raise ValueError("zero automatic retries: this experimental unit already has an attempt")
        if self.reserved_total_nusd() + int(reservation_nusd) > self.hard_cap_nusd:
            raise ValueError("hard cost cap would be exceeded")
        attempts.append({"unit_cache_key": unit_cache_key_value,
                         "reservation_nusd": int(reservation_nusd),
                         "actual_cost_nusd": None, "status": "reserved-before-send"})
        self._save()

    def settle_verified(self, unit_cache_key_value: str, actual_cost_nusd: int) -> None:
        item = self._find(unit_cache_key_value)
        if item["status"] != "reserved-before-send":
            raise ValueError("only a reserved attempt can be settled")
        if not 0 <= int(actual_cost_nusd) <= int(item["reservation_nusd"]):
            raise ValueError("verified cost is outside the reservation")
        item.update({"actual_cost_nusd": int(actual_cost_nusd), "status": "completed-usage-verified"})
        self._save()

    def mark_uncertain(self, unit_cache_key_value: str, detail: str) -> None:
        item = self._find(unit_cache_key_value)
        if item["status"] != "reserved-before-send":
            raise ValueError("only a reserved attempt can be marked uncertain")
        item.update({"status": "uncertain-reservation-retained", "detail": str(detail)})
        self._save()

    def _find(self, unit_cache_key_value: str) -> dict[str, Any]:
        for item in self.state["attempts"]:
            if item["unit_cache_key"] == unit_cache_key_value:
                return item
        raise KeyError(unit_cache_key_value)


def verify_provider_plan(request_rows: Sequence[Mapping[str, Any]], manifest_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """No-network structural verification for all prepared requests."""
    if len(request_rows) != len(manifest_rows):
        raise ValueError("request and manifest counts differ")
    body_digests = []
    keys = []
    total = 0
    for request, metadata in zip(request_rows, manifest_rows):
        body = request["provider_payload"]
        unit = request["unit"]
        if metadata.get("model") != "gpt-6.1-terra" or metadata.get("unit") != unit:
            raise ValueError("request and manifest experimental unit/model mismatch")
        if body.get("model") != "gpt-6.1-terra" or body.get("store") is not False or body.get("tools") != []:
            raise ValueError("request model/storage/tool settings drifted")
        if body.get("reasoning") != {"effort": "low"} or body.get("max_output_tokens") != MAX_OUTPUT_TOKENS:
            raise ValueError("request reasoning/output settings drifted")
        if "provenance" in body["input"].lower():
            raise ValueError("provenance text must remain in sidecars")
        target_text = body["input"].split("\nEVIDENCE\n", 1)[0]
        if retrieval.KEGG_REACTION_RE.search(target_text):
            raise ValueError("KEGG reaction ID leaked into target query")
        body_hash = digest(body)
        if body_hash != metadata["provider_payload_sha256"]:
            raise ValueError("provider payload digest mismatch")
        if request["unit_cache_key"] != metadata["unit_cache_key"]:
            raise ValueError("unit cache key mismatch")
        if int(metadata["reservation_nusd"]) != request_cost_bound(body)["reservation_nusd"]:
            raise ValueError("request reservation does not match current conservative pricing formula")
        total += int(metadata["reservation_nusd"])
        body_digests.append(body_hash)
        keys.append(request["unit_cache_key"])
    if len(set(keys)) != len(keys):
        raise ValueError("unit-specific cache keys must be unique")
    return {"request_count": len(request_rows), "unique_unit_cache_keys": len(set(keys)),
            "unique_provider_payloads": len(set(body_digests)),
            "reserved_total_nusd": total, "provider_calls": 0}
