"""Build Phase 3 lookup tables from local SBML and the frozen KEGG catalog.

No downloads and no API calls. Species names are joined later by
``(model_id, species_id)``. KEGG reaction identifiers are the keys of the
frozen Phase 2 features pickle.

Usage::

    python benchmark/scripts/build_phase3_lookups.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import libsbml
import pandas as pd

from benchmark.scripts.build_reaction_text import MODELS_DIR, _name_of
from benchmark.scripts.phase3_common import (
    OUT_KEGG_CATALOG_IDS,
    OUT_SPECIES_NAMES,
    PHASE2_COMMIT,
    PHASE2_TAG,
    write_csv,
)

logger = logging.getLogger("build_phase3_lookups")


def extract_species_names(model_file: Path) -> List[Dict[str, str]]:
    """Return one row per SBML species: model_id, species_id, species_name."""
    document = libsbml.SBMLReader().readSBML(str(model_file))
    model = document.getModel()
    if model is None:
        return []
    model_id = model_file.stem
    rows = []
    for species in model.getListOfSpecies():
        sid = str(species.getId())
        rows.append({
            "model_id": model_id,
            "species_id": sid,
            "species_name": _name_of(species),
        })
    return rows


def build_species_names(models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    files = sorted(models_dir.glob("*.xml"))
    if not files:
        raise FileNotFoundError(
            f"No SBML files in {models_dir}; restore local models before building lookups."
        )
    rows: List[Dict[str, str]] = []
    for path in files:
        rows.extend(extract_species_names(path))
    frame = pd.DataFrame(rows, columns=["model_id", "species_id", "species_name"])
    if frame.duplicated(["model_id", "species_id"]).any():
        raise RuntimeError("duplicate (model_id, species_id) in species name table")
    return frame.sort_values(["model_id", "species_id"]).reset_index(drop=True)


def dump_kegg_catalog_ids() -> Dict[str, Any]:
    from core.database_search import load_kegg_reaction_features_dict

    features = load_kegg_reaction_features_dict()
    ids = sorted(str(k) for k in features.keys())
    return {
        "phase2_tag": PHASE2_TAG,
        "phase2_commit": PHASE2_COMMIT,
        "source": "data/kegg/kegg_reaction_features.lzma",
        "n": len(ids),
        "ids": ids,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    names = build_species_names()
    write_csv(names, OUT_SPECIES_NAMES)
    logger.info("species names: %d rows, %d models -> %s",
                len(names), names.model_id.nunique(), OUT_SPECIES_NAMES)

    catalog = dump_kegg_catalog_ids()
    OUT_KEGG_CATALOG_IDS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_KEGG_CATALOG_IDS, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(catalog, fh, separators=(",", ":"), sort_keys=True)
        fh.write("\n")
    logger.info("kegg catalog ids: n=%d -> %s", catalog["n"], OUT_KEGG_CATALOG_IDS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
