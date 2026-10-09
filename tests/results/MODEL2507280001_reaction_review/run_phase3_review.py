"""Reproduce the MODEL2507280001 Phase 3 reaction review.

This is an ignored evaluation artifact, not a production-pipeline change.  Prediction
uses only the SBML plus the existing source-fidelity species recommendations.  The
publication supplement is loaded only after ranking, to review whether retrieved KEGG
reactions represent the modeled chemistry.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import lzma
import pickle
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile

import libsbml
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.scripts.phase3_retrieval import BM25, load_catalog, query_text, tokenise
from core.database_search import (
    load_kegg_parsed_reactions_dict,
    score_model_against_kegg_reaction,
)
from core.reaction.amendment_config import CofactorConfig


MODEL = ROOT / "tests" / "useCase" / "MODEL2507280001.xml"
SPECIES = (
    ROOT
    / "tests"
    / "useCase"
    / "results"
    / "MODEL2507280001"
    / "full_source_fidelity_species.csv"
)
SUPPLEMENT = (
    ROOT
    / "tests"
    / "useCase"
    / "context"
    / "publications"
    / "MODEL2507280001"
    / "supplement.xlsx"
)
FEATURES = ROOT / "data" / "kegg" / "kegg_reaction_features.lzma"
OUT = Path(__file__).resolve().parent
TOP_K = 10
BIOMODELS_ID = "MODEL2507280001"

# These are bookkeeping participants explicitly allowed by this review.  The normal
# pipeline score is also retained with CofactorConfig.default for comparison.
BOOKKEEPING_KEGG = {"C00001", "C00080"}  # water, proton
TRANSPORT_RE = re.compile(
    r"transport|export|import|uptake|efflux|influx|symport|antiport|porin|channel|diffusion",
    re.I,
)
PSEUDO_RE = re.compile(r"providing\s+aa|biomass objective", re.I)


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(map(str, value))
    return str(value)


def _xlsx_rows(path: Path, sheet_name: str) -> list[list[str]]:
    """Read values from one XLSX sheet without adding an optional dependency."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    office_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", ns):
                shared.append("".join(node.text or "" for node in item.findall(".//m:t", ns)))
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rel_id = ""
        for sheet in workbook.findall(".//m:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                rel_id = sheet.attrib.get(f"{{{office_rel}}}id", "")
                break
        if not rel_id:
            raise KeyError(f"XLSX sheet not found: {sheet_name}")
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(
            rel.attrib["Target"]
            for rel in rels.findall("r:Relationship", rel_ns)
            if rel.attrib.get("Id") == rel_id
        )
        sheet_path = "xl/" + target.lstrip("/")
        sheet = ET.fromstring(archive.read(sheet_path))
        output: list[list[str]] = []
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            values: dict[int, str] = {}
            for cell in row.findall("m:c", ns):
                ref = cell.attrib.get("r", "A1")
                letters = re.match(r"[A-Z]+", ref).group(0)
                col = 0
                for letter in letters:
                    col = col * 26 + ord(letter) - 64
                kind = cell.attrib.get("t", "")
                if kind == "inlineStr":
                    value = "".join(
                        node.text or "" for node in cell.findall(".//m:t", ns)
                    )
                else:
                    node = cell.find("m:v", ns)
                    value = "" if node is None else (node.text or "")
                    if kind == "s" and value:
                        value = shared[int(value)]
                values[col - 1] = value
            width = max(values, default=-1) + 1
            output.append([values.get(i, "") for i in range(width)])
        return output


def _bm25_rank_scores(index: BM25, query: str, topn: int) -> list[tuple[str, float]]:
    """Same ranking as Phase 3 BM25.rank, with its raw score retained."""
    scores = np.zeros(len(index.docs))
    terms = Counter(tokenise(query))
    for term, qfreq in terms.items():
        for i, freq in index.postings.get(term, []):
            denom = freq + index.k1 * (
                1 - index.b + index.b * index.lengths[i] / index.avgdl
            )
            scores[i] += (
                index.idf[term]
                * qfreq
                * freq
                * (index.k1 + 1)
                / denom
            )
    order = sorted(
        range(len(index.docs)),
        key=lambda i: (-scores[i], index.docs[i]["kegg_id"]),
    )[:topn]
    return [(index.docs[i]["kegg_id"], float(scores[i])) for i in order]


def _stoich(ref) -> str:
    coeff = float(ref.getStoichiometry())
    return ("" if abs(coeff - 1.0) < 1e-12 else f"{coeff:g} ") + str(ref.getSpecies())


def _equation(reaction) -> str:
    lhs = " + ".join(_stoich(x) for x in reaction.getListOfReactants())
    rhs = " + ".join(_stoich(x) for x in reaction.getListOfProducts())
    return f"{lhs} {'<=>' if reaction.getReversible() else '=>'} {rhs}"


def _resources(reaction) -> list[str]:
    values: list[str] = []
    for i in range(reaction.getNumCVTerms()):
        term = reaction.getCVTerm(i)
        values.extend(term.getResourceURI(j) for j in range(term.getNumResources()))
    return values


def _compound_side(references, gold: dict[str, str]) -> tuple[Counter, int, int]:
    counter: Counter = Counter()
    mapped = 0
    total = 0
    for ref in references:
        total += 1
        kid = gold.get(str(ref.getSpecies()), "")
        if kid:
            mapped += 1
            counter[kid] += float(ref.getStoichiometry())
    return counter, mapped, total


def _cancel(lhs: Counter, rhs: Counter) -> tuple[Counter, Counter]:
    common = lhs & rhs
    return lhs - common, rhs - common


def _score(
    lhs: Counter,
    rhs: Counter,
    kegg_id: str,
    parsed: dict,
    ignored: set[str],
) -> tuple[float, float, float]:
    return score_model_against_kegg_reaction(
        lhs,
        rhs,
        kegg_id,
        kegg_parsed_reactions_dict=parsed,
        cofactors_to_ignore=ignored,
        spectators=False,
    )


def main() -> None:
    started = time.time()
    species_df = pd.read_csv(SPECIES, dtype=str).fillna("")
    species_ann = {
        str(row.id): str(row.annotation)
        for row in species_df.itertuples()
        if str(row.annotation).upper().startswith("CHEBI:")
    }

    supplement_rows = _xlsx_rows(SUPPLEMENT, "Metabolites")[1:]
    gold = {
        str(row[0]): str(row[5])
        for row in supplement_rows
        if len(row) > 5 and row[0] and row[5] not in (None, "", "NA")
    }

    document = libsbml.readSBMLFromFile(str(MODEL))
    model = document.getModel()
    if model is None or document.getNumErrors() != 0:
        raise RuntimeError("SBML did not parse cleanly")
    species_names = {
        str(item.getId()): str(item.getName() or item.getId())
        for item in model.getListOfSpecies()
    }

    docs = load_catalog()
    index = BM25(docs)
    parsed = load_kegg_parsed_reactions_dict()
    features = pickle.loads(lzma.open(FEATURES, "rb").read())
    pipeline_cofactors = set(CofactorConfig().kegg_ids)

    rows: list[dict] = []
    candidate_rows: list[dict] = []
    for reaction in model.getListOfReactions():
        rid = str(reaction.getId())
        rname = str(reaction.getName() or "")
        equation = _equation(reaction)
        refs = list(reaction.getListOfReactants()) + list(reaction.getListOfProducts())
        participant_ids = list(dict.fromkeys(str(ref.getSpecies()) for ref in refs))
        evidence_parts = []
        for sid in participant_ids:
            detail = [f"species={sid}"]
            if sid in species_ann:
                detail.append(f"ChEBI={species_ann[sid]}")
            evidence_parts.append(f"{species_names.get(sid, sid)} [{'; '.join(detail)}]")
        query = query_text(
            {
                "reaction_equation": equation,
                "participant_evidence": "; ".join(evidence_parts),
            }
        )
        ranked = _bm25_rank_scores(index, query, TOP_K)

        lhs, lm, lt = _compound_side(reaction.getListOfReactants(), gold)
        rhs, rm, rt = _compound_side(reaction.getListOfProducts(), gold)
        lhs, rhs = _cancel(lhs, rhs)
        # KEGG's installed parsed-reaction cache stores participant presence but not
        # coefficients (the raw equation retains them).  Use presence on both sides for
        # the independent review score so valid 4:1/8:1 reactions are not penalized
        # asymmetrically.  Raw equations remain in the report for stoichiometry review.
        review_lhs = Counter({kid: 1 for kid in lhs})
        review_rhs = Counter({kid: 1 for kid in rhs})
        mapped_fraction = (lm + rm) / (lt + rt) if lt + rt else 0.0

        resources = _resources(reaction)
        model_ec = sorted(
            {
                uri.rsplit("/", 1)[-1]
                for uri in resources
                if "/ec-code/" in uri.lower()
            }
        )
        scored = []
        for rank, (kid, bm25_score) in enumerate(ranked, start=1):
            feat = features.get(kid, {})
            enzymes = set(re.findall(r"\b\d+\.\d+\.\d+\.(?:\d+|-)\b", _text(feat.get("ENZYME"))))
            ec_match = bool(set(model_ec) & enzymes)
            if lhs and rhs:
                pipeline_score, _, _ = _score(lhs, rhs, kid, parsed, pipeline_cofactors)
                strict_score, strict_forward, strict_reverse = _score(
                    review_lhs, review_rhs, kid, parsed, BOOKKEEPING_KEGG
                )
            else:
                pipeline_score = strict_score = strict_forward = strict_reverse = float("nan")
            item = {
                "model_id": BIOMODELS_ID,
                "sbml_model_id": str(model.getId()),
                "reaction_id": rid,
                "candidate_rank": rank,
                "candidate_kegg": kid,
                "bm25_score": bm25_score,
                "pipeline_chemical_score": pipeline_score,
                "strict_chemical_score": strict_score,
                "strict_forward_score": strict_forward,
                "strict_reverse_score": strict_reverse,
                "model_ec": ";".join(model_ec),
                "candidate_ec": ";".join(sorted(enzymes)),
                "ec_match": ec_match,
                "kegg_definition": _text(feat.get("DEFINITION")),
                "kegg_equation": _text(feat.get("EQUATION")),
                "kegg_names": _text(feat.get("NAME")),
            }
            scored.append(item)
            candidate_rows.append(item)

        top = scored[0]
        numeric = [x for x in scored if not np.isnan(x["strict_chemical_score"])]
        best = max(
            numeric,
            key=lambda x: (x["strict_chemical_score"], x["ec_match"], -x["candidate_rank"]),
            default=None,
        )
        is_exchange = reaction.getNumReactants() == 0 or reaction.getNumProducts() == 0
        is_transport = bool(TRANSPORT_RE.search(f"{rid} {rname}")) or (
            bool(lhs or rhs)
            and Counter(gold.get(str(x.getSpecies()), "") for x in reaction.getListOfReactants())
            == Counter(gold.get(str(x.getSpecies()), "") for x in reaction.getListOfProducts())
        )
        is_biomass = "biomass" in f"{rid} {rname}".lower()
        is_pseudo = bool(PSEUDO_RE.search(f"{rid} {rname}"))
        best_score = best["strict_chemical_score"] if best else float("nan")
        if is_exchange:
            assessment = "not_applicable_exchange"
            failure_stage = "candidate_generation_scope"
        elif is_transport:
            assessment = "not_applicable_transport"
            failure_stage = "database_scope"
        elif is_biomass:
            assessment = "not_applicable_biomass"
            failure_stage = "database_scope"
        elif is_pseudo:
            assessment = "not_applicable_model_pseudo_reaction"
            failure_stage = "database_scope"
        elif mapped_fraction < 0.75 and top["ec_match"] and top["strict_chemical_score"] >= 0.75:
            assessment = "top1_supported_with_partial_mapping"
            failure_stage = ""
        elif mapped_fraction < 0.75 or not lhs or not rhs:
            assessment = "insufficient_species_mapping"
            failure_stage = "metabolite_mapping"
        elif top["strict_chemical_score"] >= 0.75:
            assessment = "top1_transformation_supported"
            failure_stage = ""
        elif top["pipeline_chemical_score"] >= 0.75 and top["strict_chemical_score"] < 0.5:
            assessment = "cofactor_dependent_false_positive"
            failure_stage = "candidate_ranking"
        elif best is not None and best_score >= 0.75:
            assessment = "plausible_candidate_misranked"
            failure_stage = "candidate_ranking"
        elif top["strict_chemical_score"] >= 0.5 and top["ec_match"]:
            assessment = "top1_partial_plausible"
            failure_stage = ""
        elif best is not None and best_score >= 0.5:
            assessment = "weak_candidate_only"
            failure_stage = "candidate_retrieval"
        else:
            assessment = "wrong_transformation_or_retrieval_miss"
            failure_stage = "candidate_retrieval"

        warnings = []
        if not model_ec:
            warnings.append("no reaction-level EC annotation")
        if mapped_fraction < 1:
            warnings.append(f"supplement KEGG mapping coverage {lm + rm}/{lt + rt}")
        if is_transport:
            warnings.append("transport has no ordinary KEGG reaction transformation")
        if not species_ann or any(sid not in species_ann for sid in participant_ids):
            missing = sum(sid not in species_ann for sid in participant_ids)
            if missing:
                warnings.append(f"{missing}/{len(participant_ids)} participants lack AAAIM ChEBI evidence")
        if not np.isnan(top["pipeline_chemical_score"]) and top["pipeline_chemical_score"] - top["strict_chemical_score"] >= 0.4:
            warnings.append("default cofactor removal materially inflates chemical similarity")

        rows.append(
            {
                "model_id": BIOMODELS_ID,
                "sbml_model_id": str(model.getId()),
                "reaction_id": rid,
                "reaction_name": rname,
                "reaction_equation": equation,
                "reactants": "; ".join(
                    f"{_stoich(x)} [{species_names.get(str(x.getSpecies()), x.getSpecies())}]"
                    for x in reaction.getListOfReactants()
                ),
                "products": "; ".join(
                    f"{_stoich(x)} [{species_names.get(str(x.getSpecies()), x.getSpecies())}]"
                    for x in reaction.getListOfProducts()
                ),
                "existing_annotation": ";".join(resources),
                "existing_kegg_reaction": "",
                "existing_ec": ";".join(model_ec),
                "candidate_generation_succeeded": True,
                "phase3_component": "bm25_full_catalog",
                "predicted_kegg_top1": top["candidate_kegg"],
                "predicted_kegg_top10": ";".join(x["candidate_kegg"] for x in scored),
                "bm25_top1_score": top["bm25_score"],
                "bm25_top2_gap": top["bm25_score"] - scored[1]["bm25_score"],
                "top1_pipeline_chemical_score": top["pipeline_chemical_score"],
                "top1_strict_chemical_score": top["strict_chemical_score"],
                "top1_ec_match": top["ec_match"],
                "top1_kegg_definition": top["kegg_definition"],
                "top1_kegg_equation": top["kegg_equation"],
                "best_chemical_candidate_top10": best["candidate_kegg"] if best else "",
                "best_chemical_candidate_rank": best["candidate_rank"] if best else "",
                "best_strict_chemical_score": best_score,
                "mapped_participant_occurrences": lm + rm,
                "total_participant_occurrences": lt + rt,
                "mapped_participant_fraction": mapped_fraction,
                "is_transport": is_transport,
                "is_exchange": is_exchange,
                "is_biomass": is_biomass,
                "is_pseudo_reaction": is_pseudo,
                "assessment": assessment,
                "failure_stage": failure_stage,
                "warnings": "; ".join(warnings),
                "query_text": query,
            }
        )

    detail = pd.DataFrame(rows)
    candidates = pd.DataFrame(candidate_rows)
    detail.to_csv(OUT / "reaction_review.csv", index=False)
    candidates.to_csv(OUT / "phase3_bm25_top10_candidates.csv", index=False)

    counts = detail["assessment"].value_counts().sort_index().to_dict()
    stage_counts = detail["failure_stage"].replace("", "none").value_counts().sort_index().to_dict()
    summary = {
        "branch_base": "debc52978603532fe8cc25a1c9e1e4b3070df836",
        "model_id": BIOMODELS_ID,
        "sbml_model_id": str(model.getId()),
        "model_name": str(model.getName()),
        "sbml_errors": int(document.getNumErrors()),
        "species": int(model.getNumSpecies()),
        "reactions": int(model.getNumReactions()),
        "reactions_with_existing_kegg": 0,
        "reactions_with_existing_ec": int(detail["existing_ec"].ne("").sum()),
        "prediction_component": "Phase 3 deterministic BM25 full-catalog retriever",
        "candidate_generation_success": int(detail["candidate_generation_succeeded"].sum()),
        "assessment_counts": counts,
        "failure_stage_counts": stage_counts,
        "top1_strict_supported_non_scope": int(
            detail["assessment"].isin(
                [
                    "top1_transformation_supported",
                    "top1_partial_plausible",
                    "top1_supported_with_partial_mapping",
                ]
            ).sum()
        ),
        "top10_contains_supported_non_scope": int(
            (
                (detail["best_strict_chemical_score"] >= 0.75)
                & ~detail["assessment"].str.startswith("not_applicable")
            ).sum()
        ),
        "species_prediction_rows": int(len(species_df)),
        "species_with_aaaim_chebi": int(len(species_ann)),
        "species_with_heldout_review_kegg": int(len(gold)),
        "phase3_limitations": [
            "The selected Phase 3B trained checkpoint/inference archive is absent locally, so trained-neural plus BM25 RRF could not run.",
            "AAAIM_PHASE3C_OPENAI_API_KEY is unset, so the grounded Phase 3C final selector could not run.",
            "Reported predictions are BM25 component Top-1, not full fusion/grounded final outputs.",
        ],
        "review_scoring": {
            "pipeline_score_ignored": sorted(pipeline_cofactors),
            "strict_review_ignored": sorted(BOOKKEEPING_KEGG),
            "strict_review_stoichiometry": "participant-presence comparison; raw model and KEGG equations retained for multiplicity review",
            "strict_supported_threshold": 0.75,
            "heldout_supplement_used_only_after_prediction": True,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
