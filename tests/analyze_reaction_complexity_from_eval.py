#!/usr/bin/env python3
"""Join evaluation CSV rows with SBML reaction complexity (substrate/product counts).

Reads ``per_reaction_results.csv`` from a reaction-evaluation run (e.g.
``tests/reaction_evaluation_results/curated_species/<run>``), loads each
``model_id`` SBML from ``tests/BioModels_251106`` (or ``--models-dir``), and
adds per-reaction counts from libSBML:

- ``n_reactant_species``: distinct species on the left (reactants)
- ``n_product_species``: distinct species on the right (products)
- ``n_unique_metabolites``: distinct species across both sides

Also writes ``complexity_impact_summary.csv`` with aggregate ``found`` /
``top1`` / ``top3`` / ``top5`` rates per complexity bin (unique metabolite
counts: ``1``, ``2``, ``3``, ``4``, ``5+``).

Example::

    python tests/analyze_reaction_complexity_from_eval.py \\
        --per-reaction-csv tests/reaction_evaluation_results/curated_species/Llama-3.3-70B-Instruct-20260430_120414/per_reaction_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import libsbml  # noqa: E402

# Bins for ``n_unique_metabolites`` (distinct species across reactants and products).
COMPLEXITY_BIN_1 = "1"
COMPLEXITY_BIN_2 = "2"
COMPLEXITY_BIN_3 = "3"
COMPLEXITY_BIN_4 = "4"
COMPLEXITY_BIN_5_PLUS = "5+"
COMPLEXITY_BIN_ORDER = (
    COMPLEXITY_BIN_1,
    COMPLEXITY_BIN_2,
    COMPLEXITY_BIN_3,
    COMPLEXITY_BIN_4,
    COMPLEXITY_BIN_5_PLUS,
)


def _complexity_bin_label(n_unique: int) -> str:
    if n_unique <= 1:
        return COMPLEXITY_BIN_1
    if n_unique == 2:
        return COMPLEXITY_BIN_2
    if n_unique == 3:
        return COMPLEXITY_BIN_3
    if n_unique == 4:
        return COMPLEXITY_BIN_4
    return COMPLEXITY_BIN_5_PLUS


def _reaction_complexity(
    model_path: Path, reaction_id: str
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return (n_reactants, n_products, n_unique) or (None, None, None) if unavailable."""
    reader = libsbml.SBMLReader()
    doc = reader.readSBML(str(model_path))
    model = doc.getModel()
    if model is None:
        return None, None, None
    rxn = model.getReaction(reaction_id)
    if rxn is None:
        return None, None, None
    lhs: set[str] = set()
    for i in range(rxn.getNumReactants()):
        sp = rxn.getReactant(i).getSpecies()
        if sp:
            lhs.add(sp)
    rhs: set[str] = set()
    for i in range(rxn.getNumProducts()):
        sp = rxn.getProduct(i).getSpecies()
        if sp:
            rhs.add(sp)
    uniq = lhs | rhs
    return len(lhs), len(rhs), len(uniq)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-reaction-csv",
        type=Path,
        required=True,
        help="Path to per_reaction_results.csv from an evaluation run.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "BioModels_251106",
        help="Directory containing <model_id>.xml SBML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs (default: same folder as --per-reaction-csv).",
    )
    args = parser.parse_args()

    per_path = args.per_reaction_csv
    if not per_path.is_absolute():
        per_path = REPO_ROOT / per_path
    if not per_path.exists():
        print(f"Missing per-reaction CSV: {per_path}", file=sys.stderr)
        return 1

    out_dir = args.output_dir or per_path.parent
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models_dir = args.models_dir
    if not models_dir.is_absolute():
        models_dir = REPO_ROOT / models_dir

    df = pd.read_csv(per_path, encoding="utf-8-sig")
    required = {"model_id", "reaction_id"}
    if not required.issubset(df.columns):
        print(f"CSV must contain columns {required}, got {list(df.columns)}", file=sys.stderr)
        return 1

    # Cache SBML paths and complexity to avoid re-reading the same model.
    model_cache: Dict[str, Optional[Path]] = {}
    complexity_cache: Dict[Tuple[str, str], Tuple[Optional[int], Optional[int], Optional[int]]] = {}

    def _model_path(mid: str) -> Optional[Path]:
        if mid not in model_cache:
            p = models_dir / f"{mid}.xml"
            model_cache[mid] = p if p.exists() else None
        return model_cache[mid]

    def _complexity(mid: str, rid: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        key = (mid, rid)
        if key not in complexity_cache:
            mp = _model_path(mid)
            if mp is None:
                complexity_cache[key] = (None, None, None)
            else:
                complexity_cache[key] = _reaction_complexity(mp, rid)
        return complexity_cache[key]

    n_lhs: list[Optional[int]] = []
    n_rhs: list[Optional[int]] = []
    n_uni: list[Optional[int]] = []
    for _, row in df.iterrows():
        a, b, u = _complexity(str(row["model_id"]), str(row["reaction_id"]))
        n_lhs.append(a)
        n_rhs.append(b)
        n_uni.append(u)

    df = df.copy()
    df["n_reactant_species"] = n_lhs
    df["n_product_species"] = n_rhs
    df["n_unique_metabolites"] = n_uni

    merged_path = out_dir / "per_reaction_with_complexity.csv"
    df.to_csv(merged_path, index=False)
    print(f"Wrote {merged_path}")

    # Bins by total unique metabolites (primary "complexity" scalar).
    work = df.copy()
    work = work[work["n_unique_metabolites"].notna()].copy()
    work["n_unique_metabolites"] = work["n_unique_metabolites"].astype(int)

    if not work.empty:
        work["complexity_bin"] = work["n_unique_metabolites"].map(_complexity_bin_label)
        summary = (
            work.groupby("complexity_bin", dropna=False)
            .agg(
                n_reactions=("reaction_id", "count"),
                coverage=("found", "mean"),
                top1=("top1", "mean"),
                top3=("top3", "mean"),
                top5=("top5", "mean"),
                mean_candidates=("num_candidates", "mean"),
            )
            .reset_index()
        )
        summary["complexity_bin"] = pd.Categorical(
            summary["complexity_bin"],
            categories=list(COMPLEXITY_BIN_ORDER),
            ordered=True,
        )
        summary = summary.sort_values("complexity_bin").round(3)
        summary_path = out_dir / "complexity_impact_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Wrote {summary_path}")
        print(summary.to_string(index=False))
    else:
        print("No rows with parsable reaction complexity; skipped summary.")

    missing_models = sorted({m for m, p in model_cache.items() if p is None})
    if missing_models:
        print(f"Warning: {len(missing_models)} model_id(s) had no {models_dir}/<id>.xml")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
