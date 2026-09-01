"""Phase 2 step 0: assemble the species-annotation evidence used for candidate generation.

Phase 1 froze species annotations from ChEBI only, which made three models look like
they had no species annotations at all:

    BIOMD0000000725, BIOMD0000001090, BIOMD0000001091   (1,451 reactions, 25% of the benchmark)

Those models in fact annotate species with KEGG compound ids directly, which is the
most directly usable evidence available since it needs no ChEBI->KEGG mapping. Two
issues hid them:

1. Phase 1 extracted only ChEBI species annotations.
2. ``KEGG_COMPOUND_URI_PATTERNS`` matched only the ``kegg.compound:C#####`` colon form,
   while BioModels uses the ``kegg.compound/C#####`` slash form (the same bug class
   fixed for KEGG reaction URIs in Phase 1).

This script writes ``species_evidence.csv``, the union of ChEBI and direct KEGG
compound species annotations, leaving the frozen Phase 1 artifacts untouched.

Usage::

    python benchmark/scripts/build_species_evidence.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from core.model_info import find_species_with_annotations_and_qualifiers
from utils.constants import KEGG_COMPOUND_URI_PATTERNS

DATA_DIR = REPO_ROOT / "benchmark" / "data"
MODELS_DIR = REPO_ROOT / "benchmark" / "models"
REACTIONS_CSV = DATA_DIR / "reactions.csv"
PHASE1_SPECIES_CSV = DATA_DIR / "species_annotations.csv"
OUT_CSV = DATA_DIR / "species_evidence.csv"
OUT_JSON = DATA_DIR / "species_evidence_summary.json"

CHEBI_RE = re.compile(r"^CHEBI:\d+$", re.IGNORECASE)
KEGG_COMPOUND_RE = re.compile(r"^C\d+$")

logger = logging.getLogger("build_species_evidence")


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


def kegg_compound_annotations(model_file: Path) -> Dict[str, List[str]]:
    """Extract direct KEGG compound species annotations from an SBML file."""
    annotations, _ = find_species_with_annotations_and_qualifiers(str(model_file), "kegg")
    out: Dict[str, List[str]] = {}
    for species_id, values in (annotations or {}).items():
        keep = sorted({str(v).strip() for v in values if KEGG_COMPOUND_RE.match(str(v).strip())})
        if keep:
            out[str(species_id)] = keep
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    reactions = pd.read_csv(REACTIONS_CSV)
    model_ids = sorted(reactions["model_id"].astype(str).unique())
    phase1 = pd.read_csv(PHASE1_SPECIES_CSV)

    rows: List[Dict[str, str]] = []

    for _, r in phase1.iterrows():
        annotation = str(r["annotation"]).strip()
        if not CHEBI_RE.match(annotation):
            continue
        rows.append({
            "model_id": str(r["model_id"]),
            "species_id": str(r["species_id"]),
            "annotation": annotation.upper(),
            "annotation_type": "chebi",
            "evidence_source": "phase1_frozen_species_annotations",
        })

    per_model_kegg: Dict[str, int] = {}
    for model_id in model_ids:
        model_file = MODELS_DIR / f"{model_id}.xml"
        if not model_file.exists():
            logger.error("missing model file: %s", model_file)
            continue
        kegg = kegg_compound_annotations(model_file)
        n = 0
        for species_id, values in sorted(kegg.items()):
            for value in values:
                rows.append({
                    "model_id": model_id,
                    "species_id": species_id,
                    "annotation": value,
                    "annotation_type": "kegg_compound",
                    "evidence_source": "sbml_kegg_compound_uri",
                })
                n += 1
        per_model_kegg[model_id] = n
        if n:
            logger.info("%s: %d direct KEGG compound species annotations", model_id, n)

    df = pd.DataFrame(rows, columns=[
        "model_id", "species_id", "annotation", "annotation_type", "evidence_source",
    ])
    df = df.drop_duplicates(subset=["model_id", "species_id", "annotation"])
    df = df.sort_values(["model_id", "species_id", "annotation_type", "annotation"])
    write_csv(df.reset_index(drop=True), OUT_CSV)

    # Reaction-level evidence classes, to compare against Phase 1's species_source.
    species_by_model: Dict[str, Dict[str, set]] = {}
    for model_id, sub in df.groupby("model_id"):
        species_by_model[str(model_id)] = {
            "chebi": set(sub[sub.annotation_type == "chebi"]["species_id"]),
            "kegg_compound": set(sub[sub.annotation_type == "kegg_compound"]["species_id"]),
        }

    counts = {"chebi_only": 0, "kegg_only": 0, "both": 0, "none": 0}
    per_model_class: Dict[str, str] = {}
    for model_id in model_ids:
        ev = species_by_model.get(model_id, {"chebi": set(), "kegg_compound": set()})
        has_chebi, has_kegg = bool(ev["chebi"]), bool(ev["kegg_compound"])
        cls = ("both" if has_chebi and has_kegg else
               "chebi_only" if has_chebi else
               "kegg_only" if has_kegg else "none")
        per_model_class[model_id] = cls
        n_rxn = int((reactions.model_id == model_id).sum())
        counts[cls] += n_rxn

    recovered = [m for m, c in per_model_class.items() if c == "kegg_only"]
    recovered_reactions = int(reactions[reactions.model_id.isin(recovered)].shape[0])

    summary = {
        "models": len(model_ids),
        "evidence_rows": int(len(df)),
        "chebi_rows": int((df.annotation_type == "chebi").sum()),
        "kegg_compound_rows": int((df.annotation_type == "kegg_compound").sum()),
        "models_with_kegg_compound_species": sorted(
            m for m, n in per_model_kegg.items() if n > 0
        ),
        "reactions_by_evidence_class": counts,
        "models_recovered_by_kegg_compound_path": sorted(recovered),
        "reactions_recovered_by_kegg_compound_path": recovered_reactions,
        "per_model_evidence_class": per_model_class,
        "inputs": {
            "reactions_csv_sha256": sha256_file(REACTIONS_CSV),
            "phase1_species_annotations_sha256": sha256_file(PHASE1_SPECIES_CSV),
        },
        "outputs": {"species_evidence_csv_sha256": sha256_file(OUT_CSV)},
    }
    write_json(summary, OUT_JSON)

    logger.info("evidence rows: %d (chebi=%d, kegg_compound=%d)",
                len(df), summary["chebi_rows"], summary["kegg_compound_rows"])
    logger.info("reactions by evidence class: %s", counts)
    logger.info("recovered by direct KEGG compound path: %d reactions in %s",
                recovered_reactions, recovered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
