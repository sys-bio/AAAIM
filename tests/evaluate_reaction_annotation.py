#!/usr/bin/env python3
"""Evaluate reaction annotation performance on curated BioModels.

For every model listed in ``tests/kegg_annotated_files.txt`` this script:

1. Extracts ground-truth KEGG reaction IDs from the model (first one per
   reaction; reactions without a KEGG annotation are skipped).
2. Builds a species-annotation table (ChEBI preferred, KEGG-compound as a
   fallback) in the ``pd.read_csv``-compatible shape expected by
   ``annotate_model``.
3. Runs ``annotate_model(method="rulebased", entity_type="reaction",
   database="kegg")`` to produce candidate KEGG reaction IDs per reaction,
   then (unless ``--skip-llm-ranking``) calls
   ``rank_kegg_annotations_with_llm`` so ``<model>_recommendations_llm_ranked.csv``
   is written alongside the rule-based CSV under the run ``_work/`` folder.
4. Normalizes candidate ranks by KEGG-Orthology (BRITE) group — candidates
   that share at least one K-number are collapsed to the best (lowest) rank
   in their group.
5. Scores each reaction (found?, rank, top1/3/5) and writes per-reaction
   results + aggregate summary under the run directory.
6. Records per-model wall time and ``wall_seconds / num_evaluated_reactions`` in
   ``per_model_timing.csv`` (same run folder as ``per_reaction_results.csv``).

Run: ``python tests/evaluate_reaction_annotation.py``
(or ``conda run -n aaaim python tests/evaluate_reaction_annotation.py``).

python tests/evaluate_reaction_annotation.py --skip-cofactor-removal
python tests/evaluate_reaction_annotation.py --skip-brite-normalization


"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
import pandas as pd

# Make repo root importable when running from tests/.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# API keys (OPENAI_API_KEY, OPENROUTER_API_KEY, etc.) for LLM ranking.
load_dotenv(REPO_ROOT / ".env")

from core import annotate_model  # noqa: E402
from core.database_search import load_kegg_reaction_features_dict  # noqa: E402
from core.reaction.amendment_config import CofactorConfig  # noqa: E402
from core.reaction.annotation_workflow import rank_kegg_annotations_with_llm  # noqa: E402
from core.model_info import (  # noqa: E402
    exchange_constraint_skipped_reaction_ids,
    find_reactions_with_kegg_annotations,
    find_species_with_annotations_and_qualifiers,
    find_species_with_chebi_annotations,
)
from utils.constants import DatabaseID  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_LIST = REPO_ROOT / "tests" / "kegg_annotated_files-test.txt"
DEFAULT_RESULTS_DIR = (
    REPO_ROOT / "tests" / "reaction_evaluation_results" / "curated_species"
)

# Rule-based generation-only is fast; set True for the slower scored/EM pipeline.
EVALUATE_CANDIDATES = False
INCLUDE_EXCHANGE_REACTIONS = False
# LLM_MODEL = "Llama-3.3-70B-Instruct"
# LLM_MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"
LLM_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_LLM_TOP_K = 10
# Default ``kegg_reaction_features.lzma`` is resolved under ``data/kegg/`` by the loader.
DEFAULT_KEGG_FEATURES_FILE = "kegg_reaction_features.lzma"

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

    Per **species**, if ChEBI annotations exist they are used; otherwise KEGG
    compound IDs are used for that species. A model may therefore mix rows
    where some metabolites came from ChEBI and others from bare KEGG compounds.

    The rule-based pipeline maps ChEBI→KEGG via reference tables and treats bare
    KEGG compound ids (``C#####``) as identity mappings (see
    :func:`core.reaction.utils.map_chebi_to_kegg` and
    :mod:`core.reaction.kegg_compound_ids`).

    Returns ``(df, source)`` where ``source`` is ``"chebi"`` (all rows from
    ChEBI), ``"kegg_compound"`` (all from KEGG compounds), ``"mixed"``
    (both conventions appear), or ``"none"``. A ``None`` DataFrame means no
    species had usable annotations.

    The returned DataFrame has columns ``id, annotation, match_score`` and
    round-trips through ``pd.read_csv`` cleanly.
    """
    chebi = find_species_with_chebi_annotations(str(model_file))
    kegg, _ = find_species_with_annotations_and_qualifiers(
        str(model_file), DatabaseID.KEGG.value
    )

    rows: List[Dict[str, object]] = []

    all_ids = sorted(set(chebi.keys()) | set(kegg.keys()))
    for sid in all_ids:
        sid_str = str(sid)
        ch_list = chebi.get(sid) if chebi else None
        if ch_list:
            for cid in ch_list:
                if str(cid).strip():
                    rows.append(
                        {
                            "id": sid_str,
                            "annotation": f"CHEBI:{str(cid).split(':', 1)[-1]}",
                            "match_score": 1.0,
                        }
                    )
            continue

        kg_list = kegg.get(sid) if kegg else None
        if kg_list:
            for kid in kg_list:
                ks = str(kid).strip()
                if ks:
                    rows.append(
                        {"id": sid_str, "annotation": ks, "match_score": 1.0}
                    )

    if not rows:
        return None, "none"

    species_with_chebi = {s for s in all_ids if chebi and chebi.get(s)}
    species_kegg_only = {
        s for s in all_ids if (not (chebi and chebi.get(s))) and kegg and kegg.get(s)
    }
    if species_with_chebi and species_kegg_only:
        source = "mixed"
    elif species_kegg_only and not species_with_chebi:
        source = "kegg_compound"
    else:
        source = "chebi"

    return pd.DataFrame(rows), source


# ---------------------------------------------------------------------------
# 4. Annotation pipeline
# ---------------------------------------------------------------------------

def run_annotation(
    model_file: Path,
    species_df: pd.DataFrame,
    work_dir: Path,
    *,
    rank_with_llm: bool = True,
    disable_cofactors: bool = False,
    disable_ontology_relaxation: bool = False,
    llm_model: str = LLM_MODEL,
    llm_top_k: int = DEFAULT_LLM_TOP_K,
    kegg_features_file: Optional[str] = None,
) -> pd.DataFrame:
    """Run ``annotate_model``, optionally LLM-re-rank candidates, return recommendations.

    ``annotate_model`` writes ``<model_basename>_recommendations.csv`` under
    ``work_dir`` (cwd is switched there during the call).

    When ``rank_with_llm`` is True, ``rank_kegg_annotations_with_llm`` reads that
    CSV (enriched with definitions), queries the LLM per reaction, and writes
    ``<same_stem>_llm_ranked.csv`` next to it — evaluation uses the ranked table.

    When ``disable_cofactors`` is True, an empty ``CofactorConfig`` is passed so
    H2O, ATP, NAD+, etc. are **not** excluded from reaction matching.

    When ``disable_ontology_relaxation`` is True, ChEBI ancestor traversal is
    skipped entirely (``max_relax_level=0``).
    """
    species_csv = work_dir / f"{model_file.stem}__species.csv"
    species_df.to_csv(species_csv, index=False)

    # Round-trip through read_csv to satisfy the CSV-compatible contract.
    # utf-8-sig strips a UTF-8 BOM so the first column stays ``id``, not ``\ufeffid``.
    reloaded = pd.read_csv(species_csv, encoding="utf-8-sig")

    recommendations_csv = work_dir / f"{model_file.name}_recommendations.csv"

    cofactor_cfg = CofactorConfig(cofactors_dict={}) if disable_cofactors else None

    cwd = Path.cwd()
    try:
        import os

        os.chdir(work_dir)
        result = annotate_model(
            model_file=str(model_file),
            llm_model=llm_model,
            method="rulebased",
            entity_type="reaction",
            database="kegg",
            species_recommendations_df=reloaded,
            evaluate_candidates=EVALUATE_CANDIDATES,
            include_exchange_reactions=INCLUDE_EXCHANGE_REACTIONS,
            cofactor_config=cofactor_cfg,
            disable_ontology_relaxation=disable_ontology_relaxation,
        )
    finally:
        import os

        os.chdir(cwd)

    # ``result`` supports ``df, metrics = result`` unpacking.
    df, _metrics = result
    if df is None or df.empty:
        return pd.DataFrame(columns=["id", "annotation"])

    kff = kegg_features_file or DEFAULT_KEGG_FEATURES_FILE

    if rank_with_llm:
        try:
            df = rank_kegg_annotations_with_llm(
                model_file=str(model_file),
                recommendations_df=df,
                llm_model=llm_model,
                kegg_features_file=kff,
                top_k=llm_top_k,
                csv_path=str(recommendations_csv),
            )
            ranked_path = recommendations_csv.with_name(
                recommendations_csv.stem + "_llm_ranked.csv"
            )
            logger.info("LLM-ranked table: %s", ranked_path)
        except Exception as exc:
            logger.warning(
                "LLM ranking failed (%s); evaluating with rule-based candidate order.",
                exc,
            )

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


def build_normalized_rank_rows(
    model_id: str,
    reaction_id: str,
    ordered_candidates: List[str],
    features: Dict[str, Dict],
) -> List[Dict[str, object]]:
    """Per-candidate raw vs BRITE-normalized ranks for one reaction."""
    normalized = normalize_ranks_by_brite(ordered_candidates, features)
    # normalized only includes first-seen candidates (deduped).
    rows: List[Dict[str, object]] = []
    for cand, norm_rank in normalized.items():
        # raw rank is the first-seen rank in ordered_candidates (1-indexed)
        raw_rank = None
        for i, c in enumerate(ordered_candidates, start=1):
            if c == cand:
                raw_rank = i
                break
        k_numbers = sorted(_k_numbers_for(cand, features))
        rows.append(
            {
                "model_id": model_id,
                "reaction_id": reaction_id,
                "candidate_kegg": cand,
                "raw_rank": int(raw_rank) if raw_rank is not None else None,
                "normalized_rank": int(norm_rank),
                "k_numbers": ";".join(k_numbers),
            }
        )
    # Stable output ordering by raw rank then candidate id.
    rows.sort(key=lambda r: ((r.get("raw_rank") or 10**9), str(r.get("candidate_kegg") or "")))
    return rows


# ---------------------------------------------------------------------------
# 6. Per-reaction scoring
# ---------------------------------------------------------------------------

def evaluate_model(
    model_file: Path,
    ground_truth: Dict[str, str],
    recommendations_df: pd.DataFrame,
    features: Dict[str, Dict],
    species_source: str,
    ssx_reaction_ids: Optional[Set[str]] = None,
    *,
    normalize_brite: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """Produce one evaluation row per ground-truth reaction.

    ``species_source == "none"`` means the model had no usable species
    annotations, so every reaction is scored as a mapping failure (no
    candidates). Otherwise reactions missing from ``recommendations_df`` are
    counted as having zero candidates.

    ``failure_reason`` when ``found`` is False (empty when ``found`` is True):

    - ``no_species_annotations`` — no usable species annotations for the model.
    - ``SSX`` — source/sink/exchange-style stoichiometry (empty LHS or RHS) while
      ``INCLUDE_EXCHANGE_REACTIONS`` is False; no rule-based candidates and not
      sent to the LLM ranker, same as the annotation pipeline.
    - ``no_candidates`` — zero candidates for other reasons (e.g. no KEGG match).
    - ``""`` — candidates existed but the ground-truth KEGG id was not among them
      (aggregate reporting labels these as ``not_in_candidates``).

    When ``normalize_brite`` is False, candidate ranks are used as-is (raw list
    order) without collapsing KEGG-Orthology co-members into one group.
    """
    model_id = model_file.stem
    by_reaction: Dict[str, List[str]] = {}
    has_brite_col = (
        not recommendations_df.empty
        and "brite_group_members" in recommendations_df.columns
    )
    if not recommendations_df.empty and {"id", "annotation"}.issubset(recommendations_df.columns):
        for rxn_id, group in recommendations_df.groupby("id", sort=False):
            cands: List[str] = []
            for _, row in group.iterrows():
                bare = _strip_kegg_prefix(row.get("annotation", ""))
                if bare and bare not in cands:
                    cands.append(bare)
                # Pre-LLM filtering shows the LLM only the BRITE-orthology
                # representative; the file lists co-members in
                # `brite_group_members`. Expand them here at the same rank slot
                # so a ground-truth that is a non-representative member is still
                # counted as found (normalize_ranks_by_brite below collapses
                # them anyway, but this also handles candidates whose K-numbers
                # are absent from the loaded features).
                if has_brite_col:
                    raw = row.get("brite_group_members", "")
                    if isinstance(raw, str) and raw.strip():
                        for m in raw.split(";"):
                            m = m.strip()
                            if m and m not in cands:
                                cands.append(m)
            by_reaction[str(rxn_id)] = cands

    ssx_ids = ssx_reaction_ids or set()

    rows: List[Dict] = []
    rank_rows: List[Dict] = []
    for rxn_id, truth in ground_truth.items():
        cands = by_reaction.get(rxn_id, [])
        num_candidates = len(cands)

        if species_source == "none":
            failure_reason = "no_species_annotations"
        elif num_candidates == 0 and rxn_id in ssx_ids:
            failure_reason = "SSX"
        elif num_candidates == 0:
            failure_reason = "no_candidates"
        else:
            failure_reason = ""

        if num_candidates:
            rank_rows.extend(build_normalized_rank_rows(model_id, rxn_id, cands, features))

        if num_candidates and truth in cands:
            if normalize_brite:
                normalized = normalize_ranks_by_brite(cands, features)
                rank = normalized.get(truth)
            else:
                rank = cands.index(truth) + 1
            found = True
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
    return rows, rank_rows


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
    model_file: Path,
    features: Dict[str, Dict],
    work_dir: Path,
    *,
    rank_with_llm: bool = True,
    disable_cofactors: bool = False,
    disable_ontology_relaxation: bool = False,
    normalize_brite: bool = True,
    llm_top_k: int = DEFAULT_LLM_TOP_K,
    kegg_features_file: Optional[str] = None,
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

    ssx_ids: Set[str] = set()
    if not INCLUDE_EXCHANGE_REACTIONS:
        try:
            ssx_ids = exchange_constraint_skipped_reaction_ids(str(model_file))
        except Exception as exc:  # pragma: no cover — antimony / SBML edge cases
            logger.warning("  could not classify SSX reactions: %s", exc)

    species_df, source = build_species_recommendations_df(model_file)
    if species_df is None:
        logger.warning("  no species annotations; all reactions -> mapping failure")
        rxn_rows, rank_rows = evaluate_model(
            model_file, ground_truth, pd.DataFrame(), features, "none", ssx_ids,
            normalize_brite=normalize_brite,
        )
        process_model._last_rank_rows = rank_rows  # type: ignore[attr-defined]
        return rxn_rows
    logger.info("  species annotations: %d rows (source=%s)", len(species_df), source)

    try:
        rec_df = run_annotation(
            model_file,
            species_df,
            work_dir,
            rank_with_llm=rank_with_llm,
            disable_cofactors=disable_cofactors,
            disable_ontology_relaxation=disable_ontology_relaxation,
            llm_top_k=llm_top_k,
            kegg_features_file=kegg_features_file,
        )
    except Exception as exc:  # pragma: no cover — pipeline failures surface here
        logger.exception("  annotate_model failed: %s", exc)
        rec_df = pd.DataFrame(columns=["id", "annotation"])

    rxn_rows, rank_rows = evaluate_model(
        model_file, ground_truth, rec_df, features, source, ssx_ids,
        normalize_brite=normalize_brite,
    )
    process_model._last_rank_rows = rank_rows  # type: ignore[attr-defined]
    return rxn_rows


def main() -> int:
    wall_start = time.time()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-list", type=Path, default=DEFAULT_MODEL_LIST,
        help="Text file listing model paths (default: tests/kegg_annotated_files.txt).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Base output directory. Each run writes into a new timestamped "
            "subdirectory under this path."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N models (for smoke testing).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Optional run identifier (subdirectory name). Defaults to a timestamp, "
            "e.g. 20260421_134455."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Optional scratch directory for intermediate CSVs. Defaults to "
            "<run_dir>/_work."
        ),
    )
    parser.add_argument(
        "--skip-llm-ranking",
        action="store_true",
        help=(
            "Do not call rank_kegg_annotations_with_llm; evaluate using only the "
            "rule-based <model>_recommendations.csv order (faster, no LLM API)."
        ),
    )
    parser.add_argument(
        "--skip-cofactor-removal",
        action="store_true",
        help=(
            "Disable cofactor filtering during reaction matching (H2O, H+, ATP, "
            "NAD+, etc. are included as regular participants). By default cofactors "
            "are removed before candidate scoring."
        ),
    )
    parser.add_argument(
        "--skip-brite-normalization",
        action="store_true",
        help=(
            "Use raw candidate list order for scoring instead of collapsing "
            "KEGG-Orthology co-members (BRITE groups) to their best shared rank."
        ),
    )
    parser.add_argument(
        "--skip-ontology-relaxation",
        action="store_true",
        help=(
            "Disable ChEBI ontology relaxation during reaction matching. Species "
            "are matched only at their exact annotated ChEBI level; no ancestor "
            "traversal is attempted."
        ),
    )
    parser.add_argument(
        "--llm-top-k",
        type=int,
        default=DEFAULT_LLM_TOP_K,
        help="Max KEGG reaction ids to keep per reaction from the LLM (default: 10).",
    )
    parser.add_argument(
        "--kegg-features-file",
        type=str,
        default=None,
        help=(
            "Path or filename for kegg_reaction_features.lzma (default: "
            f"{DEFAULT_KEGG_FEATURES_FILE}, resolved under data/kegg/)."
        ),
    )
    args = parser.parse_args()

    def _safe_slug(s: str) -> str:
        # Windows-safe-ish: avoid characters that break paths.
        return (
            str(s)
            .strip()
            .replace(" ", "_")
            .replace("/", "-")
            .replace("\\", "-")
            .replace(":", "-")
            .replace("|", "-")
            .replace("*", "-")
            .replace("?", "")
            .replace("\"", "")
            .replace("<", "(")
            .replace(">", ")")
        )

    model_paths = load_model_paths(args.model_list)
    if args.limit is not None:
        model_paths = model_paths[: args.limit]
    logger.info("Evaluating %d models", len(model_paths))

    rank_with_llm = not bool(args.skip_llm_ranking)
    disable_cofactors = bool(args.skip_cofactor_removal)
    disable_ontology_relaxation = bool(args.skip_ontology_relaxation)
    normalize_brite = not bool(args.skip_brite_normalization)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    _run_suffixes = (
        ("-no-cofactors" if disable_cofactors else "")
        + ("-no-relaxation" if disable_ontology_relaxation else "")
        + ("-no-brite" if not normalize_brite else "")
    )
    run_id = args.run_id or f"{_safe_slug(LLM_MODEL)}-{timestamp}{_run_suffixes}"
    run_dir = args.results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    per_reaction_out = run_dir / "per_reaction_results.csv"
    summary_out = run_dir / "results_summary.csv"
    normalized_ranks_out = run_dir / "normalized_candidate_ranks.csv"
    per_model_timing_out = run_dir / "per_model_timing.csv"

    work_dir = args.work_dir or (run_dir / "_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run directory: %s", run_dir)

    logger.info("Loading KEGG reaction features (for BRITE/orthology grouping)...")
    features = load_kegg_reaction_features_dict()
    if not rank_with_llm:
        logger.info("LLM re-ranking disabled (--skip-llm-ranking).")
    if disable_cofactors:
        logger.info("Cofactor removal disabled (--skip-cofactor-removal).")
    if disable_ontology_relaxation:
        logger.info("Ontology relaxation disabled (--skip-ontology-relaxation).")
    if not normalize_brite:
        logger.info("BRITE normalization disabled (--skip-brite-normalization).")

    all_rows: List[Dict] = []
    all_rank_rows: List[Dict] = []
    timing_rows: List[Dict[str, object]] = []
    for i, model_file in enumerate(model_paths, start=1):
        t0 = time.time()
        try:
            rows = process_model(
                model_file,
                features,
                work_dir,
                rank_with_llm=rank_with_llm,
                disable_cofactors=disable_cofactors,
                disable_ontology_relaxation=disable_ontology_relaxation,
                normalize_brite=normalize_brite,
                llm_top_k=int(args.llm_top_k),
                kegg_features_file=args.kegg_features_file,
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("  model %s failed: %s", model_file, exc)
            rows = []
        elapsed = time.time() - t0
        n_eval = len(rows)
        spd = (elapsed / n_eval) if n_eval else float("nan")
        timing_rows.append(
            {
                "model_id": model_file.stem,
                "model_path": str(model_file),
                "wall_seconds": round(elapsed, 3),
                "num_evaluated_reactions": int(n_eval),
                "seconds_per_reaction": spd if n_eval else float("nan"),
            }
        )
        pd.DataFrame(timing_rows).to_csv(per_model_timing_out, index=False)
        all_rows.extend(rows)
        # Collected inside process_model via evaluate_model.
        all_rank_rows.extend(getattr(process_model, "_last_rank_rows", []) or [])
        # Incremental checkpoint so long runs don't lose progress.
        pd.DataFrame(all_rows).to_csv(per_reaction_out, index=False)
        spd_msg = f"{spd:.3f}s/rxn" if n_eval else "n/a"
        logger.info(
            "  [%d/%d] %s -> %d rows (%.1fs, %s)",
            i, len(model_paths), model_file.name, len(rows), elapsed, spd_msg,
        )

    per_reaction_df = pd.DataFrame(all_rows)
    per_reaction_df.to_csv(per_reaction_out, index=False)
    logger.info("Per-reaction results written to %s (%d rows)",
                per_reaction_out, len(per_reaction_df))

    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(per_model_timing_out, index=False)
    logger.info("Per-model timing written to %s (%d models)", per_model_timing_out, len(timing_df))

    rank_df = pd.DataFrame(all_rank_rows)
    rank_df.to_csv(normalized_ranks_out, index=False)
    logger.info(
        "Normalized candidate ranks written to %s (%d rows)",
        normalized_ranks_out,
        len(rank_df),
    )

    summary_df = summarize(per_reaction_df)
    summary_df.to_csv(summary_out, index=False)
    logger.info("Summary written to %s", summary_out)

    total_s = time.time() - wall_start
    logger.info("Total runtime: %.1fs", total_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
