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

**CLI**

::

    python tests/analyze_reaction_complexity_from_eval.py \\
        --per-reaction-csv tests/reaction_evaluation_results/curated_species/Llama-3.3-70B-Instruct-20260430_120414/per_reaction_results.csv

**Notebook**

::

    from tests.analyze_reaction_complexity_from_eval import run_complexity_analysis

    result = run_complexity_analysis("path/to/per_reaction_results.csv")
    display(result.per_reaction)
    display(result.summary)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, NamedTuple, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import libsbml  # noqa: E402

__all__ = [
    "COMPLEXITY_BIN_1",
    "COMPLEXITY_BIN_2",
    "COMPLEXITY_BIN_3",
    "COMPLEXITY_BIN_4",
    "COMPLEXITY_BIN_5_PLUS",
    "COMPLEXITY_BIN_ORDER",
    "ComplexityAnalysisResult",
    "complexity_bin_label",
    "reaction_complexity",
    "run_complexity_analysis",
]

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


def complexity_bin_label(n_unique: int) -> str:
    if n_unique <= 1:
        return COMPLEXITY_BIN_1
    if n_unique == 2:
        return COMPLEXITY_BIN_2
    if n_unique == 3:
        return COMPLEXITY_BIN_3
    if n_unique == 4:
        return COMPLEXITY_BIN_4
    return COMPLEXITY_BIN_5_PLUS


def reaction_complexity(
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


def _resolve_path(p: Path | str, *, default_relative_to: Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = default_relative_to / path
    return path


class ComplexityAnalysisResult(NamedTuple):
    """Return value of :func:`run_complexity_analysis`."""

    per_reaction: pd.DataFrame
    summary: Optional[pd.DataFrame]
    merged_csv_path: Optional[Path]
    summary_csv_path: Optional[Path]
    missing_models: tuple[str, ...]


def run_complexity_analysis(
    per_reaction_csv: Path | str,
    *,
    models_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    write_csv: bool = True,
    quiet: bool = False,
) -> ComplexityAnalysisResult:
    """Load eval CSV, attach SBML complexity columns, optionally aggregate and write CSVs.

    Parameters
    ----------
    per_reaction_csv
        Path to ``per_reaction_results.csv`` (relative paths are under ``REPO_ROOT``).
    models_dir
        Directory of ``<model_id>.xml``. Default: ``tests/BioModels_251106`` under repo root.
    output_dir
        Where to write outputs. Default: directory of ``per_reaction_csv``.
    write_csv
        If True, writes ``per_reaction_with_complexity.csv`` and
        ``complexity_impact_summary.csv`` when applicable.
    quiet
        If False, print paths and summary table (CLI-style).

    Returns
    -------
    ComplexityAnalysisResult
        DataFrames always populated for ``per_reaction``; ``summary`` is None if no
        rows had parsable complexity. ``*_csv_path`` fields are set only when
        ``write_csv`` is True and that file was written.
    """
    per_path = _resolve_path(per_reaction_csv, default_relative_to=REPO_ROOT)
    if not per_path.exists():
        raise FileNotFoundError(f"Missing per-reaction CSV: {per_path}")

    mdir = models_dir if models_dir is not None else REPO_ROOT / "tests" / "BioModels_251106"
    models_path = _resolve_path(mdir, default_relative_to=REPO_ROOT)

    out: Path
    if output_dir is None:
        out = per_path.parent
    else:
        out = _resolve_path(output_dir, default_relative_to=REPO_ROOT)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_path, encoding="utf-8-sig")
    required = {"model_id", "reaction_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}, got {list(df.columns)}")

    model_cache: Dict[str, Optional[Path]] = {}
    complexity_cache: Dict[Tuple[str, str], Tuple[Optional[int], Optional[int], Optional[int]]] = {}

    def _model_path(mid: str) -> Optional[Path]:
        if mid not in model_cache:
            p = models_path / f"{mid}.xml"
            model_cache[mid] = p if p.exists() else None
        return model_cache[mid]

    def _cached_counts(mid: str, rid: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        key = (mid, rid)
        if key not in complexity_cache:
            mp = _model_path(mid)
            if mp is None:
                complexity_cache[key] = (None, None, None)
            else:
                complexity_cache[key] = reaction_complexity(mp, rid)
        return complexity_cache[key]

    n_lhs: list[Optional[int]] = []
    n_rhs: list[Optional[int]] = []
    n_uni: list[Optional[int]] = []
    for _, row in df.iterrows():
        a, b, u = _cached_counts(str(row["model_id"]), str(row["reaction_id"]))
        n_lhs.append(a)
        n_rhs.append(b)
        n_uni.append(u)

    merged = df.copy()
    merged["n_reactant_species"] = n_lhs
    merged["n_product_species"] = n_rhs
    merged["n_unique_metabolites"] = n_uni

    merged_csv_path: Optional[Path] = None
    if write_csv:
        merged_csv_path = out / "per_reaction_with_complexity.csv"
        merged.to_csv(merged_csv_path, index=False)
        if not quiet:
            print(f"Wrote {merged_csv_path}")

    work = merged[merged["n_unique_metabolites"].notna()].copy()
    summary_df: Optional[pd.DataFrame] = None
    summary_csv_path: Optional[Path] = None

    if not work.empty:
        work = work.copy()
        work["n_unique_metabolites"] = work["n_unique_metabolites"].astype(int)
        work["complexity_bin"] = work["n_unique_metabolites"].map(complexity_bin_label)
        summary_df = (
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
        summary_df["complexity_bin"] = pd.Categorical(
            summary_df["complexity_bin"],
            categories=list(COMPLEXITY_BIN_ORDER),
            ordered=True,
        )
        summary_df = summary_df.sort_values("complexity_bin").round(3)
        if write_csv:
            summary_csv_path = out / "complexity_impact_summary.csv"
            summary_df.to_csv(summary_csv_path, index=False)
            if not quiet:
                print(f"Wrote {summary_csv_path}")
                print(summary_df.to_string(index=False))
    elif not quiet:
        print("No rows with parsable reaction complexity; skipped summary.")

    missing_models = tuple(sorted(m for m, p in model_cache.items() if p is None))
    if missing_models and not quiet:
        print(f"Warning: {len(missing_models)} model_id(s) had no {models_path}/<id>.xml")

    return ComplexityAnalysisResult(
        per_reaction=merged,
        summary=summary_df,
        merged_csv_path=merged_csv_path,
        summary_csv_path=summary_csv_path,
        missing_models=missing_models,
    )


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

    try:
        run_complexity_analysis(
            args.per_reaction_csv,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            write_csv=True,
            quiet=False,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
