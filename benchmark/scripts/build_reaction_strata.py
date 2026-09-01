"""Phase 2 step 2: per-reaction stratification variables for the ceiling and failure analyses.

Emits ``reaction_strata.csv`` with one row per reaction in the frozen benchmark, carrying
the covariates every Phase 2 result is broken down by:

* species annotation source and coverage (ChEBI, direct KEGG compound, unannotated)
* reaction complexity (participant counts)
* genome-scale vs smaller model
* cluster id, for cluster-macro averaging

Usage::

    python benchmark/scripts/build_reaction_strata.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import libsbml
import pandas as pd

DATA_DIR = REPO_ROOT / "benchmark" / "data"
MODELS_DIR = REPO_ROOT / "benchmark" / "models"

REACTIONS_CSV = DATA_DIR / "reactions.csv"
SPECIES_EVIDENCE_CSV = DATA_DIR / "species_evidence.csv"
CLUSTERS_CSV = DATA_DIR / "model_clusters.csv"
OUT_CSV = DATA_DIR / "reaction_strata.csv"
OUT_JSON = DATA_DIR / "reaction_strata_summary.json"

# The benchmark is dominated by a handful of genome-scale reconstructions; anything at
# or above this reaction count is treated as genome-scale for macro reporting.
GENOME_SCALE_MIN_REACTIONS = 300

logger = logging.getLogger("build_reaction_strata")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")


def write_json(obj: Any, path: Path) -> None:
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def complexity_bucket(n: int) -> str:
    if n <= 2:
        return "1-2"
    if n <= 4:
        return "3-4"
    if n <= 6:
        return "5-6"
    return "7+"


def reaction_participants(model_file: Path) -> Dict[str, Dict[str, List[str]]]:
    """Map reaction id -> {'substrates': [...], 'products': [...]} using libsbml."""
    document = libsbml.SBMLReader().readSBML(str(model_file))
    model = document.getModel()
    out: Dict[str, Dict[str, List[str]]] = {}
    if model is None:
        return out
    for rxn in model.getListOfReactions():
        subs = [rxn.getReactant(i).getSpecies() for i in range(rxn.getNumReactants())]
        prods = [rxn.getProduct(i).getSpecies() for i in range(rxn.getNumProducts())]
        out[str(rxn.getId())] = {
            "substrates": [str(s) for s in subs if s],
            "products": [str(p) for p in prods if p],
        }
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    reactions = pd.read_csv(REACTIONS_CSV)
    evidence = pd.read_csv(SPECIES_EVIDENCE_CSV)
    clusters = pd.read_csv(CLUSTERS_CSV)
    cluster_of = dict(zip(clusters.model_id.astype(str), clusters.cluster_id.astype(str)))

    chebi_species: Dict[str, Set[str]] = {}
    kegg_species: Dict[str, Set[str]] = {}
    for (model_id, ann_type), sub in evidence.groupby(["model_id", "annotation_type"]):
        target = chebi_species if ann_type == "chebi" else kegg_species
        target.setdefault(str(model_id), set()).update(sub.species_id.astype(str))

    reactions_per_model = reactions.groupby("model_id").size().to_dict()

    rows: List[Dict[str, Any]] = []
    for model_id, model_rows in reactions.groupby("model_id"):
        model_id = str(model_id)
        model_file = MODELS_DIR / f"{model_id}.xml"
        participants = reaction_participants(model_file) if model_file.exists() else {}
        if not participants:
            logger.warning("no participants parsed for %s", model_id)

        chebi_set = chebi_species.get(model_id, set())
        kegg_set = kegg_species.get(model_id, set())
        n_model_reactions = int(reactions_per_model.get(model_id, 0))
        is_genome_scale = n_model_reactions >= GENOME_SCALE_MIN_REACTIONS

        for r in model_rows.itertuples():
            rid = str(r.reaction_id)
            parts = participants.get(rid, {"substrates": [], "products": []})
            all_parts = list(parts["substrates"]) + list(parts["products"])
            distinct = sorted(set(all_parts))

            n_chebi = sum(1 for s in distinct if s in chebi_set)
            n_kegg = sum(1 for s in distinct if s in kegg_set)
            n_any = sum(1 for s in distinct if s in chebi_set or s in kegg_set)
            n_unannotated = len(distinct) - n_any

            if n_any == 0:
                source = "none"
            elif n_kegg and n_chebi:
                source = "chebi+kegg_compound"
            elif n_kegg:
                source = "kegg_compound"
            else:
                source = "chebi"

            coverage = (n_any / len(distinct)) if distinct else 0.0
            rows.append({
                "model_id": model_id,
                "reaction_id": rid,
                "cluster_id": cluster_of.get(model_id, f"CLU_{model_id}"),
                "included_in_eval": bool(r.included_in_eval),
                "num_participants": len(distinct),
                "num_substrates": len(set(parts["substrates"])),
                "num_products": len(set(parts["products"])),
                "complexity_bucket": complexity_bucket(len(distinct)),
                "species_annotation_source": source,
                "num_participants_chebi": n_chebi,
                "num_participants_kegg_compound": n_kegg,
                "num_participants_annotated": n_any,
                "num_participants_unannotated": n_unannotated,
                "participant_annotation_coverage": round(coverage, 4),
                "fully_annotated": n_unannotated == 0 and len(distinct) > 0,
                "any_missing_annotation": n_unannotated > 0,
                "model_reaction_count": n_model_reactions,
                "is_genome_scale": is_genome_scale,
            })

    df = pd.DataFrame(rows).sort_values(["model_id", "reaction_id"]).reset_index(drop=True)
    write_csv(df, OUT_CSV)

    ev = df[df.included_in_eval]
    summary = {
        "reactions": int(len(df)),
        "evaluable_reactions": int(len(ev)),
        "genome_scale_models": sorted(df[df.is_genome_scale].model_id.unique().tolist()),
        "genome_scale_reaction_share": round(float(ev.is_genome_scale.mean()), 4),
        "species_annotation_source_counts": ev.species_annotation_source.value_counts().to_dict(),
        "complexity_bucket_counts": ev.complexity_bucket.value_counts().sort_index().to_dict(),
        "reactions_with_missing_annotations": int(ev.any_missing_annotation.sum()),
        "mean_participant_annotation_coverage": round(
            float(ev.participant_annotation_coverage.mean()), 4
        ),
        "clusters": int(df.cluster_id.nunique()),
        "inputs": {
            "reactions_csv_sha256": sha256_file(REACTIONS_CSV),
            "species_evidence_csv_sha256": sha256_file(SPECIES_EVIDENCE_CSV),
        },
        "outputs": {"reaction_strata_csv_sha256": sha256_file(OUT_CSV)},
    }
    write_json(summary, OUT_JSON)

    logger.info("wrote %d strata rows (%d evaluable)", len(df), len(ev))
    logger.info("genome-scale models: %s (%.1f%% of evaluable reactions)",
                len(summary["genome_scale_models"]),
                100 * summary["genome_scale_reaction_share"])
    logger.info("species annotation source: %s", summary["species_annotation_source_counts"])
    logger.info("complexity: %s", summary["complexity_bucket_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
