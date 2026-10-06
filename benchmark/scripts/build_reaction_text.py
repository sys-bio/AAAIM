"""Phase 2 step 4: natural-language context for each benchmark reaction.

The lexical, embedding and LLM rankers all need text, not ids. This extracts, per
reaction, the model context a curator would actually see: the reaction's own name, the
display names of its participants, and the enclosing model's name.

Usage::

    python benchmark/scripts/build_reaction_text.py
"""

from __future__ import annotations

import hashlib
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

DATA_DIR = REPO_ROOT / "benchmark" / "data"
MODELS_DIR = REPO_ROOT / "benchmark" / "models"
REACTIONS_CSV = DATA_DIR / "reactions.csv"
OUT_CSV = DATA_DIR / "reaction_text.csv"
OUT_JSON = DATA_DIR / "reaction_text_summary.json"

logger = logging.getLogger("build_reaction_text")


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


def _name_of(element) -> str:
    name = element.getName() if element.isSetName() else ""
    return str(name).strip() or str(element.getId())


def extract_model_text(model_file: Path) -> Dict[str, Dict[str, Any]]:
    document = libsbml.SBMLReader().readSBML(str(model_file))
    model = document.getModel()
    if model is None:
        return {}

    model_name = (model.getName() if model.isSetName() else "") or model.getId() or ""
    species_names = {
        str(s.getId()): _name_of(s) for s in model.getListOfSpecies()
    }

    out: Dict[str, Dict[str, Any]] = {}
    for rxn in model.getListOfReactions():
        rid = str(rxn.getId())
        rxn_name = rxn.getName().strip() if rxn.isSetName() else ""
        subs = [species_names.get(str(rxn.getReactant(i).getSpecies()), "")
                for i in range(rxn.getNumReactants())]
        prods = [species_names.get(str(rxn.getProduct(i).getSpecies()), "")
                 for i in range(rxn.getNumProducts())]
        mods = [species_names.get(str(rxn.getModifier(i).getSpecies()), "")
                for i in range(rxn.getNumModifiers())]
        out[rid] = {
            "model_name": str(model_name).strip(),
            "reaction_name": rxn_name,
            "substrate_names": "; ".join([s for s in subs if s]),
            "product_names": "; ".join([p for p in prods if p]),
            "modifier_names": "; ".join([m for m in mods if m]),
        }
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    reactions = pd.read_csv(REACTIONS_CSV)
    rows: List[Dict[str, Any]] = []

    for model_id, model_rows in reactions.groupby("model_id"):
        model_id = str(model_id)
        model_file = MODELS_DIR / f"{model_id}.xml"
        text = extract_model_text(model_file) if model_file.exists() else {}
        if not text:
            logger.warning("no reaction text parsed for %s", model_id)
        for r in model_rows.itertuples():
            rid = str(r.reaction_id)
            t = text.get(rid, {})
            name = t.get("reaction_name", "")
            subs = t.get("substrate_names", "")
            prods = t.get("product_names", "")
            # Query text as a curator would read it: what the reaction is called, then
            # the transformation in words.
            query = " | ".join(part for part in [
                name,
                f"{subs} -> {prods}" if (subs or prods) else "",
                t.get("modifier_names", ""),
            ] if part)
            rows.append({
                "model_id": model_id,
                "reaction_id": rid,
                "model_name": t.get("model_name", ""),
                "reaction_name": name,
                "has_reaction_name": bool(name),
                "substrate_names": subs,
                "product_names": prods,
                "modifier_names": t.get("modifier_names", ""),
                "reaction_equation": str(r.reaction_equation),
                "query_text": query,
            })

    df = pd.DataFrame(rows).sort_values(["model_id", "reaction_id"]).reset_index(drop=True)
    write_csv(df, OUT_CSV)

    summary = {
        "reactions": int(len(df)),
        "with_reaction_name": int(df.has_reaction_name.sum()),
        "with_reaction_name_pct": round(100.0 * float(df.has_reaction_name.mean()), 2),
        "empty_query_text": int((df.query_text.fillna("") == "").sum()),
        "mean_query_chars": round(float(df.query_text.fillna("").str.len().mean()), 1),
        "inputs": {"reactions_csv_sha256": sha256_file(REACTIONS_CSV)},
        "outputs": {"reaction_text_csv_sha256": sha256_file(OUT_CSV)},
    }
    write_json(summary, OUT_JSON)
    logger.info("wrote %d rows; %.1f%% have a reaction name; %d empty queries",
                len(df), summary["with_reaction_name_pct"], summary["empty_query_text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
