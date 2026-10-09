"""Run the selected Phase 3B retriever/fusion on MODEL2507280001.

This is an ignored evaluation artifact.  It does not alter the model or any
production/benchmark pipeline code.  Chemical review labels are joined only after
the label-free Phase 3 query, dense ranking, BM25 ranking, and RRF ranking are frozen.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import lzma
import os
import pickle
import platform
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import torch

# The exact Phase 3B environment intentionally contains only retrieval dependencies.
# The normal project interpreter provides libSBML for the post-ranking chemistry
# review.  Import retrieval packages first, then append (never prepend) the base
# interpreter's site-packages when this script is run in the isolated environment.
try:
    import libsbml
except ModuleNotFoundError:
    sys.path.append(str(Path(sys.base_prefix) / "Lib" / "site-packages"))
    import libsbml


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(OUT) not in sys.path:
    sys.path.insert(0, str(OUT))

import run_phase3_review as baseline
from benchmark.scripts import phase3_retrieval as retrieval
from benchmark.scripts import phase3_biencoder as biencoder
from benchmark.scripts import phase3b_release as release
from core.database_search import (
    load_kegg_parsed_reactions_dict,
    score_model_against_kegg_reaction,
)
from core.reaction.amendment_config import CofactorConfig


EXPECTED_ARCHIVE_SHA256 = "3412a3fa546347d8209ab62ca7ef55fb490e148b5f90c25cf33616328a0d0f53"
EXPECTED_CHECKPOINT_SHA256 = "8773b04f09916889b74c956e708044fe2c653fa764fe73c422ecc376ecae81c1"
MODEL_ID = "MODEL2507280001"
TOP_DEPTH = 100
REVIEW_TOP_K = 10
SUPPORTED = {
    "top1_transformation_supported",
    "top1_partial_plausible",
    "top1_supported_with_partial_mapping",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(value: object, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _features() -> dict:
    return pickle.loads(lzma.open(baseline.FEATURES, "rb").read())


def _review_contexts() -> tuple[libsbml.Model, dict[str, dict], dict[str, str]]:
    supplement_rows = baseline._xlsx_rows(baseline.SUPPLEMENT, "Metabolites")[1:]
    gold = {
        str(row[0]): str(row[5])
        for row in supplement_rows
        if len(row) > 5 and row[0] and row[5] not in (None, "", "NA")
    }
    document = libsbml.readSBMLFromFile(str(baseline.MODEL))
    model = document.getModel()
    if model is None or document.getNumErrors() != 0:
        raise RuntimeError("SBML did not parse cleanly")
    contexts: dict[str, dict] = {}
    for reaction in model.getListOfReactions():
        lhs, lm, lt = baseline._compound_side(reaction.getListOfReactants(), gold)
        rhs, rm, rt = baseline._compound_side(reaction.getListOfProducts(), gold)
        lhs, rhs = baseline._cancel(lhs, rhs)
        contexts[str(reaction.getId())] = {
            "lhs": lhs,
            "rhs": rhs,
            "review_lhs": Counter({kid: 1 for kid in lhs}),
            "review_rhs": Counter({kid: 1 for kid in rhs}),
            "mapped": lm + rm,
            "total": lt + rt,
        }
    return model, contexts, gold


def _score_candidates(
    reaction_id: str,
    ranked: list[str],
    bm25_ids: list[str],
    dense_ids: list[str],
    bm25_raw: dict[str, float],
    dense_raw: dict[str, float],
    context: dict,
    features: dict,
    parsed: dict,
    pipeline_cofactors: set[str],
) -> list[dict]:
    bm_rank = {kid: rank for rank, kid in enumerate(bm25_ids, 1)}
    dense_rank = {kid: rank for rank, kid in enumerate(dense_ids, 1)}
    rows: list[dict] = []
    for fused_rank, kid in enumerate(ranked[:REVIEW_TOP_K], 1):
        feat = features.get(kid, {})
        enzyme_text = baseline._text(feat.get("ENZYME"))
        enzymes = set(baseline.re.findall(r"\b\d+\.\d+\.\d+\.(?:\d+|-)\b", enzyme_text))
        base = BASELINE_BY_REACTION[reaction_id]
        model_ec = set(filter(None, str(base["existing_ec"]).split(";")))
        ec_match = bool(model_ec & enzymes)
        if context["lhs"] and context["rhs"]:
            pipeline_score, _, _ = score_model_against_kegg_reaction(
                context["lhs"], context["rhs"], kid,
                kegg_parsed_reactions_dict=parsed,
                cofactors_to_ignore=pipeline_cofactors,
                spectators=False,
            )
            strict_score, forward, reverse = score_model_against_kegg_reaction(
                context["review_lhs"], context["review_rhs"], kid,
                kegg_parsed_reactions_dict=parsed,
                cofactors_to_ignore=baseline.BOOKKEEPING_KEGG,
                spectators=False,
            )
        else:
            pipeline_score = strict_score = forward = reverse = float("nan")
        br = bm_rank.get(kid)
        dr = dense_rank.get(kid)
        fused_score = (1.0 / (release.RRF_K + br) if br else 0.0) + (
            1.0 / (release.RRF_K + dr) if dr else 0.0
        )
        rows.append({
            "model_id": MODEL_ID,
            "reaction_id": reaction_id,
            "candidate_rank": fused_rank,
            "candidate_kegg": kid,
            "fusion_score": fused_score,
            "bm25_rank_top100": br or "",
            "trained_biencoder_rank_top100": dr or "",
            "bm25_score": bm25_raw.get(kid, float("nan")),
            "trained_biencoder_cosine": dense_raw.get(kid, float("nan")),
            "pipeline_chemical_score": pipeline_score,
            "strict_chemical_score": strict_score,
            "strict_forward_score": forward,
            "strict_reverse_score": reverse,
            "model_ec": ";".join(sorted(model_ec)),
            "candidate_ec": ";".join(sorted(enzymes)),
            "ec_match": ec_match,
            "kegg_definition": baseline._text(feat.get("DEFINITION")),
            "kegg_equation": baseline._text(feat.get("EQUATION")),
            "kegg_names": baseline._text(feat.get("NAME")),
        })
    return rows


def _assessment(base: dict, scored: list[dict], context: dict) -> tuple[str, str, dict | None]:
    top = scored[0]
    numeric = [row for row in scored if not np.isnan(row["strict_chemical_score"])]
    best = max(
        numeric,
        key=lambda row: (row["strict_chemical_score"], row["ec_match"], -row["candidate_rank"]),
        default=None,
    )
    if str(base["assessment"]).startswith("not_applicable"):
        return str(base["assessment"]), str(base["failure_stage"]), best
    mapped_fraction = context["mapped"] / context["total"] if context["total"] else 0.0
    if mapped_fraction < 0.75 and top["ec_match"] and top["strict_chemical_score"] >= 0.75:
        return "top1_supported_with_partial_mapping", "", best
    if mapped_fraction < 0.75 or not context["lhs"] or not context["rhs"]:
        return "insufficient_species_mapping", "metabolite_mapping", best
    if top["strict_chemical_score"] >= 0.75:
        return "top1_transformation_supported", "", best
    if top["pipeline_chemical_score"] >= 0.75 and top["strict_chemical_score"] < 0.5:
        return "cofactor_dependent_false_positive", "candidate_ranking", best
    if best is not None and best["strict_chemical_score"] >= 0.75:
        return "plausible_candidate_misranked", "candidate_ranking", best
    if top["strict_chemical_score"] >= 0.5 and top["ec_match"]:
        return "top1_partial_plausible", "", best
    if best is not None and best["strict_chemical_score"] >= 0.5:
        return "weak_candidate_only", "candidate_retrieval", best
    return "wrong_transformation_or_retrieval_miss", "candidate_retrieval", best


def _transition_examples(comparison: pd.DataFrame, label: str, n: int = 8) -> list[dict]:
    subset = comparison.loc[comparison["transition"].eq(label)].copy()
    cols = [
        "reaction_id", "reaction_name", "reaction_equation", "bm25_top1",
        "phase3b_top1", "bm25_assessment", "phase3b_assessment",
        "phase3b_top1_strict_score", "phase3b_best_top10", "phase3b_best_top10_score",
    ]
    return subset.sort_values("reaction_id").head(n)[cols].to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    archive = args.archive.resolve()
    archive_sha = sha256_file(archive)
    if archive_sha != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"selected inference archive digest mismatch: {archive_sha}")
    problems = release.verify_archive_contents(archive)
    if problems:
        raise ValueError(f"selected inference archive failed verification: {problems}")

    baseline_df = pd.read_csv(OUT / "reaction_review.csv").fillna("")
    global BASELINE_BY_REACTION
    BASELINE_BY_REACTION = {
        str(row["reaction_id"]): row for row in baseline_df.to_dict("records")
    }
    queries = baseline_df["query_text"].astype(str).tolist()
    reaction_ids = baseline_df["reaction_id"].astype(str).tolist()
    docs = retrieval.load_catalog()
    catalog_ids = [row["kegg_id"] for row in docs]
    bm25 = retrieval.BM25(docs)
    bm25_ranked_with_scores = [baseline._bm25_rank_scores(bm25, query, TOP_DEPTH) for query in queries]
    bm25_rankings = [[kid for kid, _ in rows] for rows in bm25_ranked_with_scores]

    with tempfile.TemporaryDirectory(prefix="_phase3b_selected_", dir=OUT) as tmp:
        model_dir = Path(tmp)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(model_dir)
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        model, tokenizer, device = release._load_archive_model(model_dir, device_name=device_name)
        document_embeddings = release._encode_local(
            model, tokenizer, [row["text"] for row in docs], device, batch_size=args.batch
        )
        query_embeddings = release._encode_local(
            model, tokenizer, queries, device, batch_size=args.batch
        )

    dense_rankings = retrieval.dense_rank(
        query_embeddings, document_embeddings, catalog_ids, topn=TOP_DEPTH
    )
    fusion_rankings = [
        release.reciprocal_rank_fusion_complete([bm, dense], catalog_ids)[:TOP_DEPTH]
        for bm, dense in zip(bm25_rankings, dense_rankings)
    ]

    bm25_rows = []
    dense_rows = []
    fusion_rows = []
    for rid, bm_ids, dense_ids, fused_ids in zip(
        reaction_ids, bm25_rankings, dense_rankings, fusion_rankings
    ):
        bm25_rows.append({"model_id": MODEL_ID, "reaction_id": rid, "method": "bm25", "ranked_ids": bm_ids})
        dense_rows.append({"model_id": MODEL_ID, "reaction_id": rid, "method": "trained_biencoder_epoch1", "ranked_ids": dense_ids})
        fusion_rows.append({"schema": release.FUSION_SCHEMA, "model_id": MODEL_ID, "reaction_id": rid, "method": release.FUSION_METHOD, "ranked_ids": fused_ids})
    write_jsonl(bm25_rows, OUT / "phase3b_bm25_rankings_top100.jsonl")
    write_jsonl(dense_rows, OUT / "phase3b_trained_epoch1_rankings_top100.jsonl")
    write_jsonl(fusion_rows, OUT / "phase3b_fusion_rankings_top100.jsonl")

    _, contexts, _ = _review_contexts()
    features = _features()
    parsed = load_kegg_parsed_reactions_dict()
    pipeline_cofactors = set(CofactorConfig().kegg_ids)
    doc_index = {kid: i for i, kid in enumerate(catalog_ids)}
    candidate_rows: list[dict] = []
    review_rows: list[dict] = []
    for position, rid in enumerate(reaction_ids):
        bm_ids = bm25_rankings[position]
        dense_ids = dense_rankings[position]
        fused_ids = fusion_rankings[position]
        bm_raw = dict(bm25_ranked_with_scores[position])
        dense_raw = {
            kid: float(document_embeddings[doc_index[kid]] @ query_embeddings[position])
            for kid in dense_ids
        }
        scored = _score_candidates(
            rid, fused_ids, bm_ids, dense_ids, bm_raw, dense_raw,
            contexts[rid], features, parsed, pipeline_cofactors,
        )
        candidate_rows.extend(scored)
        base = BASELINE_BY_REACTION[rid]
        assessment, failure_stage, best = _assessment(base, scored, contexts[rid])
        top = scored[0]
        mapped_fraction = contexts[rid]["mapped"] / contexts[rid]["total"] if contexts[rid]["total"] else 0.0
        warnings = list(filter(None, str(base["warnings"]).split("; ")))
        if top["bm25_rank_top100"] == "":
            warnings.append("fusion Top-1 absent from BM25 Top-100")
        if top["trained_biencoder_rank_top100"] == "":
            warnings.append("fusion Top-1 absent from trained-bi-encoder Top-100")
        review_rows.append({
            **base,
            "candidate_generation_succeeded": True,
            "phase3_component": release.FUSION_METHOD,
            "predicted_kegg_top1": top["candidate_kegg"],
            "predicted_kegg_top10": ";".join(row["candidate_kegg"] for row in scored),
            "fusion_top1_score": top["fusion_score"],
            "fusion_top1_bm25_rank": top["bm25_rank_top100"],
            "fusion_top1_trained_rank": top["trained_biencoder_rank_top100"],
            "fusion_top1_dense_cosine": top["trained_biencoder_cosine"],
            "top1_pipeline_chemical_score": top["pipeline_chemical_score"],
            "top1_strict_chemical_score": top["strict_chemical_score"],
            "top1_ec_match": top["ec_match"],
            "top1_kegg_definition": top["kegg_definition"],
            "top1_kegg_equation": top["kegg_equation"],
            "best_chemical_candidate_top10": best["candidate_kegg"] if best else "",
            "best_chemical_candidate_rank": best["candidate_rank"] if best else "",
            "best_strict_chemical_score": best["strict_chemical_score"] if best else float("nan"),
            "mapped_participant_fraction": mapped_fraction,
            "assessment": assessment,
            "failure_stage": failure_stage,
            "warnings": "; ".join(dict.fromkeys(warnings)),
            "phase3c_status": "not_run_missing_OPENAI_API_KEY" if not os.environ.get("OPENAI_API_KEY") else "not_run_requires_explicit_execute",
        })

    candidates = pd.DataFrame(candidate_rows)
    review = pd.DataFrame(review_rows)
    candidates.to_csv(OUT / "phase3b_fusion_top10_candidates.csv", index=False)
    review.to_csv(OUT / "phase3b_fusion_reaction_review.csv", index=False)

    adequate = (~baseline_df["assessment"].str.startswith("not_applicable")) & (
        baseline_df["mapped_participant_fraction"].astype(float) >= 0.75
    )
    base_eval = baseline_df.loc[adequate].set_index("reaction_id")
    phase_eval = review.loc[review["reaction_id"].isin(base_eval.index)].set_index("reaction_id")
    cand_eval = candidates.loc[candidates["reaction_id"].isin(base_eval.index)].copy()
    at_k = {}
    for k in (1, 3, 5, 10):
        hit_ids = set(
            cand_eval.loc[
                (cand_eval["candidate_rank"] <= k) & (cand_eval["strict_chemical_score"] >= 0.75),
                "reaction_id",
            ]
        )
        at_k[str(k)] = {"count": len(hit_ids), "fraction": len(hit_ids) / len(base_eval)}
    baseline_candidates = pd.read_csv(OUT / "phase3_bm25_top10_candidates.csv")
    baseline_candidates = baseline_candidates.loc[baseline_candidates["reaction_id"].isin(base_eval.index)]
    bm_at_k = {}
    for k in (1, 3, 5, 10):
        hit_ids = set(
            baseline_candidates.loc[
                (baseline_candidates["candidate_rank"] <= k)
                & (baseline_candidates["strict_chemical_score"] >= 0.75),
                "reaction_id",
            ]
        )
        bm_at_k[str(k)] = {"count": len(hit_ids), "fraction": len(hit_ids) / len(base_eval)}

    ordered_ids = list(base_eval.index)
    comparison = pd.DataFrame({
        "reaction_id": ordered_ids,
        "reaction_name": [base_eval.at[rid, "reaction_name"] for rid in ordered_ids],
        "reaction_equation": [base_eval.at[rid, "reaction_equation"] for rid in ordered_ids],
        "bm25_top1": [base_eval.at[rid, "predicted_kegg_top1"] for rid in ordered_ids],
        "phase3b_top1": [phase_eval.at[rid, "predicted_kegg_top1"] for rid in ordered_ids],
        "bm25_assessment": [base_eval.at[rid, "assessment"] for rid in ordered_ids],
        "phase3b_assessment": [phase_eval.at[rid, "assessment"] for rid in ordered_ids],
        "bm25_top1_strict_score": [float(base_eval.at[rid, "top1_strict_chemical_score"]) for rid in ordered_ids],
        "phase3b_top1_strict_score": [float(phase_eval.at[rid, "top1_strict_chemical_score"]) for rid in ordered_ids],
        "bm25_best_top10": [base_eval.at[rid, "best_chemical_candidate_top10"] for rid in ordered_ids],
        "bm25_best_top10_score": [float(base_eval.at[rid, "best_strict_chemical_score"]) for rid in ordered_ids],
        "phase3b_best_top10": [phase_eval.at[rid, "best_chemical_candidate_top10"] for rid in ordered_ids],
        "phase3b_best_top10_score": [float(phase_eval.at[rid, "best_strict_chemical_score"]) for rid in ordered_ids],
    })
    bm_supported = comparison["bm25_assessment"].isin(SUPPORTED)
    ph_supported = comparison["phase3b_assessment"].isin(SUPPORTED)
    comparison["transition"] = np.select(
        [~bm_supported & ph_supported, bm_supported & ~ph_supported, bm_supported & ph_supported],
        ["corrected", "degraded", "remained_correct"],
        default="remained_incorrect",
    )
    comparison.to_csv(OUT / "phase3b_vs_bm25_comparison_264.csv", index=False)

    phase_best = comparison["phase3b_best_top10_score"]
    summary = {
        "model_id": MODEL_ID,
        "pipeline": {
            "phase3a": "project-native BM25, full 12,312-reaction catalog, depth 100",
            "phase3b": "validation-selected BAAI/bge-small-en-v1.5 epoch-1 fine-tuned bi-encoder, depth 100",
            "fusion": "equal-weight reciprocal rank fusion; k=60; one-indexed; missing contribution zero; KEGG-ID ascending tie-break",
            "phase3c": "not executed",
        },
        "artifact": {
            "source_path": str(archive),
            "archive_sha256": archive_sha,
            "expected_archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "source_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "contents_verified": True,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "device": device_name,
            "batch_size": args.batch,
            "elapsed_seconds": round(time.time() - started, 3),
        },
        "population": {
            "all_model_reactions": len(review),
            "adequately_mapped_internal_reactions": len(base_eval),
            "no_call_scope": int(baseline_df["assessment"].str.startswith("not_applicable").sum()),
            "insufficient_mapping": int((baseline_df["assessment"] == "insufficient_species_mapping").sum()),
        },
        "phase3b_metrics_264": {
            "top1_transformation_match_strict": at_k["1"],
            "top_k_transformation_recall_strict": at_k,
            "top1_supported_including_ec_backed_partial": {
                "count": int(ph_supported.sum()),
                "fraction": float(ph_supported.mean()),
            },
            "strong_candidate_absent_top10": int((phase_best < 0.75).sum()),
            "plausible_candidate_absent_top10": int((phase_best < 0.5).sum()),
            "assessment_counts": phase_eval["assessment"].value_counts().sort_index().to_dict(),
            "failure_stage_counts": phase_eval["failure_stage"].replace("", "none").value_counts().sort_index().to_dict(),
        },
        "bm25_baseline_264": {
            "top_k_transformation_recall_strict": bm_at_k,
            "top1_supported_including_ec_backed_partial": {
                "count": int(bm_supported.sum()),
                "fraction": float(bm_supported.mean()),
            },
            "top1_strict": int((comparison["bm25_top1_strict_score"] >= 0.75).sum()),
            "top10_strict": int((comparison["bm25_best_top10_score"] >= 0.75).sum()),
        },
        "comparison": {
            "bm25_failures_corrected_by_phase3b": int((comparison["transition"] == "corrected").sum()),
            "bm25_correct_predictions_degraded_by_phase3b": int((comparison["transition"] == "degraded").sum()),
            "remained_correct": int((comparison["transition"] == "remained_correct").sum()),
            "remained_incorrect": int((comparison["transition"] == "remained_incorrect").sum()),
            "strong_top10_gained": int(((comparison["bm25_best_top10_score"] < 0.75) & (comparison["phase3b_best_top10_score"] >= 0.75)).sum()),
            "strong_top10_lost": int(((comparison["bm25_best_top10_score"] >= 0.75) & (comparison["phase3b_best_top10_score"] < 0.75)).sum()),
        },
        "representative_cases": {
            "corrected": _transition_examples(comparison, "corrected"),
            "degraded": _transition_examples(comparison, "degraded"),
            "remaining": _transition_examples(comparison, "remained_incorrect"),
        },
        "credential": {"environment_variable": "OPENAI_API_KEY", "available": bool(os.environ.get("OPENAI_API_KEY"))},
    }
    write_json(summary, OUT / "phase3b_full_summary.json")
    write_json(summary["artifact"], OUT / "phase3b_artifact_provenance.json")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
