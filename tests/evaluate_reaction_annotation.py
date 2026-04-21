#!/usr/bin/env python3
"""Evaluate reaction annotation performance on curated BioModels.

For every model listed in ``tests/kegg_annotated_files.txt`` this script:

1. Extracts ground-truth KEGG reaction IDs from the model (first one per
   reaction; reactions without a KEGG annotation are skipped).
2. Builds a species-annotation table (ChEBI preferred, KEGG-compound as a
   fallback) in the ``pd.read_csv``-compatible shape expected by
   ``annotate_model``.
3. Runs ``annotate_model(method="rulebased", entity_type="reaction",
   database="kegg")`` to produce candidate KEGG reaction IDs per reaction.
4. Normalizes candidate ranks by KEGG-Orthology (BRITE) group — candidates
   that share at least one K-number are collapsed to the best (lowest) rank
   in their group.
5. Scores each reaction (found?, rank, top1/3/5) and writes per-reaction
   results + aggregate summary to ``tests/``.

Run: ``python tests/evaluate_reaction_annotation.py``
(or ``conda run -n aaaim python tests/evaluate_reaction_annotation.py``).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

# Make repo root importable when running from tests/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import annotate_model  # noqa: E402
from core.database_search import load_kegg_reaction_features_dict  # noqa: E402
from core.model_info import (  # noqa: E402
    find_reactions_with_kegg_annotations,
    find_species_with_annotations_and_qualifiers,
    find_species_with_chebi_annotations,
)
from utils.constants import DatabaseID  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_LIST = REPO_ROOT / "tests" / "kegg_annotated_files.txt"
DEFAULT_PER_REACTION_OUT = REPO_ROOT / "tests" / "reaction_eval_per_reaction.csv"
DEFAULT_SUMMARY_OUT = REPO_ROOT / "tests" / "reaction_eval_summary.csv"

# Rule-based generation-only is fast; set True for the slower scored/EM pipeline.
EVALUATE_CANDIDATES = False
INCLUDE_EXCHANGE_REACTIONS = False
LLM_MODEL = "Llama-3.3-70B-Instruct"  # ignored in rulebased/generation-only

_K_NUMBER_RE = re.compile(r"\bK\d{5,}\b")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("evaluate_reaction_annotation")


# ---------------------------------------------------------------------------
# 1. Load model list
# ---------------------------------------------------------------------------

def load_model_paths(list_file: Path) -> List[Path]:
    """Read non-blank, non-comment lines from the model-list file."""
    with list_file.open("r", encoding="utf-8") as fh:
        paths: List[Path] = []
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if not p.is_absolute():
                p = REPO_ROOT / p
            paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# 2. Ground-truth reaction annotations
# ---------------------------------------------------------------------------

def extract_ground_truth_reactions(model_file: Path) -> Dict[str, str]:
    """Return ``{reaction_id: first_kegg_reaction_id}`` (drops reactions w/o KEGG)."""
    existing, _ = find_reactions_with_kegg_annotations(str(model_file))
    out: Dict[str, str] = {}
    for rxn_id, kegg_ids in existing.items():
        for kid in kegg_ids:
            k = str(kid).strip()
            if k:
                out[str(rxn_id)] = k
                break  # first one only
    return out


# ---------------------------------------------------------------------------
# 3. Species annotations -> annotate_model-compatible DataFrame
# ---------------------------------------------------------------------------

def build_species_recommendations_df(model_file: Path) -> Tuple[Optional[pd.DataFrame], str]:
    """Build the species->annotation table expected by the rulebased workflow.

    Prefers ChEBI. Falls back to KEGG-compound IDs if no ChEBI annotations
    exist. Returns ``(df, source)`` where ``source`` is ``"chebi"``,
    ``"kegg_compound"``, or ``"none"``. A ``None`` DataFrame means the model
    has no usable species annotations.

    The returned DataFrame has columns ``id, annotation, match_score`` and
    round-trips through ``pd.read_csv`` cleanly.
    """
    chebi = find_species_with_chebi_annotations(str(model_file))
    if chebi:
        rows = [
            {"id": str(sid), "annotation": f"CHEBI:{str(cid).split(':', 1)[-1]}", "match_score": 1.0}
            for sid, ids in chebi.items()
            for cid in ids
            if str(cid).strip()
        ]
        if rows:
            return pd.DataFrame(rows), "chebi"

    kegg, _ = find_species_with_annotations_and_qualifiers(
        str(model_file), DatabaseID.KEGG.value
    )
    if kegg:
        rows = [
            {"id": str(sid), "annotation": str(kid).strip(), "match_score": 1.0}
            for sid, ids in kegg.items()
            for kid in ids
            if str(kid).strip()
        ]
        if rows:
            return pd.DataFrame(rows), "kegg_compound"

    return None, "none"


# ---------------------------------------------------------------------------
# 4. Annotation pipeline
# ---------------------------------------------------------------------------

def run_annotation(
    model_file: Path, species_df: pd.DataFrame, work_dir: Path
) -> Optional[pd.DataFrame]:
    """Run ``annotate_model`` and return the resulting recommendations DataFrame.

    The species table is written to a CSV in ``work_dir`` so the call matches
    the ``pd.read_csv``-based contract described in the task. ``annotate_model``
    itself writes ``<modelname>_recommendations.csv`` in the current working
    directory; we work in ``work_dir`` to keep those artifacts out of the repo
    root.
    """
    species_csv = work_dir / f"{model_file.stem}__species.csv"
    species_df.to_csv(species_csv, index=False)

    # Round-trip through read_csv to satisfy the CSV-compatible contract.
    reloaded = pd.read_csv(species_csv)

    cwd = Path.cwd()
    try:
        import os

        os.chdir(work_dir)
        result = annotate_model(
            model_file=str(model_file),
            llm_model=LLM_MODEL,
            method="rulebased",
            entity_type="reaction",
            database="kegg",
            species_recommendations_df=reloaded,
            evaluate_candidates=EVALUATE_CANDIDATES,
            include_exchange_reactions=INCLUDE_EXCHANGE_REACTIONS,
        )
    finally:
        import os

        os.chdir(cwd)

    # ``result`` supports ``df, metrics = result`` unpacking.
    df, _metrics = result
    if df is None or df.empty:
        return pd.DataFrame(columns=["id", "annotation"])
    return df


# ---------------------------------------------------------------------------
# 5. Rank normalization by BRITE / KEGG-Orthology groups
# ---------------------------------------------------------------------------

def _strip_kegg_prefix(raw: object) -> str:
    """Normalize candidate values like ``"KEGG:R00024"`` -> ``"R00024"``."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    if s.upper().startswith("KEGG:"):
        s = s[5:].strip()
    return s


def _k_numbers_for(kegg_id: str, features: Dict[str, Dict]) -> Set[str]:
    """KEGG Orthology K-numbers for a reaction, parsed from the ORTHOLOGY field."""
    feat = features.get(kegg_id)
    if not feat:
        return set()
    orth = str(feat.get("ORTHOLOGY", "") or "")
    return set(_K_NUMBER_RE.findall(orth))


def normalize_ranks_by_brite(
    ordered_candidates: List[str], features: Dict[str, Dict]
) -> Dict[str, int]:
    """Assign each candidate a rank, then collapse candidates sharing any KEGG
    Orthology K-number into one group and give every member the group's best
    (lowest) rank. Candidates without any K-numbers keep their own rank.
    """
    # Deduplicate while preserving first-seen order (best rank wins).
    seen: Dict[str, int] = {}
    for idx, cand in enumerate(ordered_candidates, start=1):
        if cand and cand not in seen:
            seen[cand] = idx
    ranks: Dict[str, int] = dict(seen)
    if not ranks:
        return ranks

    # Union-find over candidates that share any K-number.
    parent: Dict[str, str] = {c: c for c in ranks}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Keep the one with the better (lower) rank as root.
            if ranks[ra] <= ranks[rb]:
                parent[rb] = ra
            else:
                parent[ra] = rb

    k_to_candidates: Dict[str, List[str]] = {}
    for cand in ranks:
        for k in _k_numbers_for(cand, features):
            k_to_candidates.setdefault(k, []).append(cand)
    for cands in k_to_candidates.values():
        for other in cands[1:]:
            union(cands[0], other)

    # Collapse to root rank.
    return {c: ranks[find(c)] for c in ranks}


# ---------------------------------------------------------------------------
# 6. Per-reaction scoring
# ---------------------------------------------------------------------------

def evaluate_model(
    model_file: Path,
    ground_truth: Dict[str, str],
    recommendations_df: pd.DataFrame,
    features: Dict[str, Dict],
    species_source: str,
) -> List[Dict]:
    """Produce one evaluation row per ground-truth reaction.

    ``species_source == "none"`` means the model had no usable species
    annotations, so every reaction is scored as a mapping failure (no
    candidates). Otherwise reactions missing from ``recommendations_df`` are
    counted as having zero candidates.
    """
    model_id = model_file.stem
    by_reaction: Dict[str, List[str]] = {}
    if not recommendations_df.empty and {"id", "annotation"}.issubset(recommendations_df.columns):
        for rxn_id, group in recommendations_df.groupby("id", sort=False):
            cands: List[str] = []
            for ann in group["annotation"].tolist():
                bare = _strip_kegg_prefix(ann)
                if bare and bare not in cands:
                    cands.append(bare)
            by_reaction[str(rxn_id)] = cands

    rows: List[Dict] = []
    for rxn_id, truth in ground_truth.items():
        cands = by_reaction.get(rxn_id, [])
        num_candidates = len(cands)

        if species_source == "none":
            failure_reason = "no_species_annotations"
        elif num_candidates == 0:
            failure_reason = "no_candidates"
        else:
            failure_reason = ""

        if num_candidates and truth in cands:
            normalized = normalize_ranks_by_brite(cands, features)
            rank = normalized.get(truth)
            found = rank is not None
        else:
            rank = None
            found = False

        rows.append(
            {
                "model_id": model_id,
                "reaction_id": rxn_id,
                "ground_truth_kegg": truth,
                "found": bool(found),
                "rank": int(rank) if rank is not None else None,
                "top1": bool(found and rank == 1),
                "top3": bool(found and rank is not None and rank <= 3),
                "top5": bool(found and rank is not None and rank <= 5),
                "num_candidates": num_candidates,
                "species_source": species_source,
                "failure_reason": failure_reason if not found else "",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 7. Aggregate reporting
# ---------------------------------------------------------------------------

def summarize(per_reaction: pd.DataFrame) -> pd.DataFrame:
    total = len(per_reaction)
    if total == 0:
        return pd.DataFrame(
            [{"total_reactions": 0, "coverage": 0.0, "top1": 0.0, "top3": 0.0, "top5": 0.0}]
        )
    return pd.DataFrame(
        [
            {
                "total_reactions": int(total),
                "coverage": float(per_reaction["found"].mean()),
                "top1": float(per_reaction["top1"].mean()),
                "top3": float(per_reaction["top3"].mean()),
                "top5": float(per_reaction["top5"].mean()),
            }
        ]
    )


# ---------------------------------------------------------------------------
# 8. Driver
# ---------------------------------------------------------------------------

def process_model(
    model_file: Path, features: Dict[str, Dict], work_dir: Path
) -> List[Dict]:
    if not model_file.exists():
        logger.warning("Model not found, skipping: %s", model_file)
        return []

    logger.info("==> %s", model_file)

    ground_truth = extract_ground_truth_reactions(model_file)
    if not ground_truth:
        logger.info("  no KEGG reaction ground truth; skipping model")
        return []
    logger.info("  %d ground-truth reactions", len(ground_truth))

    species_df, source = build_species_recommendations_df(model_file)
    if species_df is None:
        logger.warning("  no species annotations; all reactions -> mapping failure")
        return evaluate_model(model_file, ground_truth, pd.DataFrame(), features, "none")
    logger.info("  species annotations: %d rows (source=%s)", len(species_df), source)

    try:
        rec_df = run_annotation(model_file, species_df, work_dir)
    except Exception as exc:  # pragma: no cover — pipeline failures surface here
        logger.exception("  annotate_model failed: %s", exc)
        rec_df = pd.DataFrame(columns=["id", "annotation"])

    return evaluate_model(model_file, ground_truth, rec_df, features, source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-list", type=Path, default=DEFAULT_MODEL_LIST,
        help="Text file listing model paths (default: tests/kegg_annotated_files.txt).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N models (for smoke testing).",
    )
    parser.add_argument(
        "--per-reaction-out", type=Path, default=DEFAULT_PER_REACTION_OUT,
        help="Output CSV path for per-reaction results.",
    )
    parser.add_argument(
        "--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT,
        help="Output CSV path for the aggregate summary.",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=REPO_ROOT / "tests" / "_eval_work",
        help="Scratch directory for intermediate CSVs.",
    )
    args = parser.parse_args()

    model_paths = load_model_paths(args.model_list)
    if args.limit is not None:
        model_paths = model_paths[: args.limit]
    logger.info("Evaluating %d models", len(model_paths))

    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading KEGG reaction features (for BRITE/orthology grouping)...")
    features = load_kegg_reaction_features_dict()

    all_rows: List[Dict] = []
    for i, model_file in enumerate(model_paths, start=1):
        t0 = time.time()
        try:
            rows = process_model(model_file, features, work_dir)
        except Exception as exc:  # pragma: no cover
            logger.exception("  model %s failed: %s", model_file, exc)
            rows = []
        all_rows.extend(rows)
        # Incremental checkpoint so long runs don't lose progress.
        args.per_reaction_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(args.per_reaction_out, index=False)
        logger.info(
            "  [%d/%d] %s -> %d rows (%.1fs)",
            i, len(model_paths), model_file.name, len(rows), time.time() - t0,
        )

    per_reaction_df = pd.DataFrame(all_rows)
    per_reaction_df.to_csv(args.per_reaction_out, index=False)
    logger.info("Per-reaction results written to %s (%d rows)",
                args.per_reaction_out, len(per_reaction_df))

    summary_df = summarize(per_reaction_df)
    summary_df.to_csv(args.summary_out, index=False)
    logger.info("Summary written to %s", args.summary_out)

    print()
    print("=== Aggregate reaction-annotation evaluation ===")
    if len(per_reaction_df) == 0:
        print("No reactions evaluated.")
        return 0
    s = summary_df.iloc[0]
    print(f"Total reactions evaluated : {int(s['total_reactions'])}")
    print(f"Coverage (found)          : {s['coverage']:.1%}")
    print(f"Top-1 accuracy            : {s['top1']:.1%}")
    print(f"Top-3 accuracy            : {s['top3']:.1%}")
    print(f"Top-5 accuracy            : {s['top5']:.1%}")

    failures = per_reaction_df[~per_reaction_df["found"]]
    if not failures.empty:
        reason_counts = failures["failure_reason"].replace("", "not_in_candidates").value_counts()
        print("\nFailure reasons:")
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
