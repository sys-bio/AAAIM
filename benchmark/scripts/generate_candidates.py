"""Phase 2 step 1: generate rule-based KEGG candidates for the frozen benchmark.

Reads the frozen Phase 1 artifacts (``reactions.csv``, ``species_annotations.csv``)
and runs AAAIM's rule-based candidate generator over every evaluable reaction.

Design constraints driven by the benchmark contract:

* Candidates live in their own table, keyed by (model, reaction, candidate); they are
  never written back into ``reactions.csv``.
* A candidate-generation exception is an explicit pipeline failure. The core generator
  is invoked with ``strict_errors=True`` so it raises instead of returning an empty
  list; when a model raises, it is retried reaction-by-reaction to isolate exactly
  which reactions failed, so one bad reaction cannot silently zero out a whole model.
* "No candidates found" and "generation crashed" are recorded as different statuses.
* Ranks are re-derived deterministically as ``(-score, candidate_kegg)``. The upstream
  ordering is a stable sort over set-derived iteration order, which is not stable
  across processes under string hash randomisation.

Usage::

    python benchmark/scripts/generate_candidates.py                 # all models
    python benchmark/scripts/generate_candidates.py --models BIOMD0000000013
    python benchmark/scripts/generate_candidates.py --workers 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

DATA_DIR = REPO_ROOT / "benchmark" / "data"
MODELS_DIR = REPO_ROOT / "benchmark" / "models"
CACHE_DIR = DATA_DIR / "_candidates_cache"

REACTIONS_CSV = DATA_DIR / "reactions.csv"
# Phase 2 evidence table: frozen Phase 1 ChEBI annotations plus direct KEGG compound
# annotations (see build_species_evidence.py).
SPECIES_CSV = DATA_DIR / "species_evidence.csv"

CANDIDATES_CSV = DATA_DIR / "candidates.csv"
STATUS_CSV = DATA_DIR / "candidate_status.csv"
FAILURES_CSV = DATA_DIR / "candidate_generation_failures.csv"
CONFIG_JSON = DATA_DIR / "candidate_generation_config.json"

# Generation configuration. Any change here changes ``config_id`` and therefore
# invalidates cached per-model results.
GENERATION_CONFIG: Dict[str, Any] = {
    "generator": "aaaim_rulebased_kegg",
    "evaluate_candidates": True,
    "include_exchange_reactions": False,
    "max_relax_level": 2,
    "max_ancestor_depth": 2,
    "max_descendant_depth": 1,
    "penalty_lam": 0.0,
    "top_k": None,
    "spectators": False,
    "cofactors_to_ignore": "CofactorConfig.default",
    "species_evidence": "phase2_species_evidence_chebi_plus_kegg_compound",
    "species_match_score": 1.0,
    # Reaction ids come from the same filtered pass as the reaction strings; taking them
    # from get_all_reaction_ids() misattributes candidates whenever a reaction is
    # filtered out for mentioning none of the mapped species.
    "reaction_id_source": "extract_reactions_with_ids_from_sbml",
    # See --scope. "evaluable" only generates for reactions that carry ground truth,
    # which is a large saving because many models pass far more reactions to the
    # generator than the benchmark scores (e.g. BIOMD0000000244: 50 vs 2).
    "reaction_scope": "evaluable",
    "rank_tie_break": "candidate_kegg_ascending",
    "strict_errors": True,
    # A reaction whose participants yield no mapped KEGG constraints produces an
    # unconstrained candidate set: upstream `filter_kegg_reactions` tests
    # `model_keys.issubset(kegg_keys)`, and the empty set is a subset of everything,
    # so every KEGG reaction "matches". That is the absence of retrieval, not a
    # candidate set, so no candidate rows are stored for those reactions.
    "drop_unconstrained_candidate_sets": True,
}

STATUS_OK = "ok"
STATUS_NO_CANDIDATES = "no_candidates"
STATUS_NO_SPECIES_EVIDENCE = "no_species_evidence"
STATUS_UNCONSTRAINED = "unconstrained_candidate_set"
STATUS_EXCHANGE_SKIPPED = "exchange_skipped"
STATUS_FAILED = "generation_failed"
STATUS_ABSENT = "absent_from_generator_output"

logger = logging.getLogger("generate_candidates")


def config_id() -> str:
    payload = json.dumps(GENERATION_CONFIG, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write LF-terminated CSV so recorded digests verify on any platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        df.to_csv(fh, index=False, lineterminator="\n")


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


@dataclass
class ModelOutcome:
    """Per-model generation result, cached to disk so runs are resumable."""

    model_id: str
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    status: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "config_id": config_id(),
            "candidates": self.candidates,
            "status": self.status,
            "failures": self.failures,
            "elapsed_s": round(self.elapsed_s, 2),
        }


def _species_recommendations(species_df: pd.DataFrame, model_id: str) -> pd.DataFrame:
    """Rebuild the species ChEBI evidence frame from the frozen Phase 1 artifact."""
    sub = species_df[species_df["model_id"] == model_id]
    out = sub.rename(columns={"species_id": "id"})[["id", "annotation"]].copy()
    out["id"] = out["id"].astype(str)
    out["annotation"] = out["annotation"].astype(str)
    out["match_score"] = float(GENERATION_CONFIG["species_match_score"])
    out = out.drop_duplicates(subset=["id", "annotation"]).sort_values(["id", "annotation"])
    return out.reset_index(drop=True)


def _relaxation_summary(metadata: Optional[Dict[str, Any]]) -> Tuple[int, str]:
    """Collapse per-participant relaxation records into (level, direction) for a reaction."""
    if not metadata:
        return 0, "exact"
    records = metadata.get("participant_relaxation") or []
    level = 0
    directions = set()
    for rec in records:
        try:
            distance = int(rec.get("distance", 0) or 0)
        except (TypeError, ValueError):
            distance = 0
        if distance > level:
            level = distance
        if distance > 0:
            directions.add(str(rec.get("direction", "") or ""))
    if level == 0:
        return 0, "exact"
    return level, "+".join(sorted(d for d in directions if d)) or "unknown"


def _run_generator(
    model_file: Path,
    recs: pd.DataFrame,
    reaction_ids: List[str],
    restrict_to: Optional[Set[str]] = None,
) -> Tuple[List[Any], bool]:
    """Invoke the rule-based generator in strict mode.

    Returns ``(match_results, had_species_evidence)``. An empty result with
    ``had_species_evidence=False`` means the generator could not run for lack of
    usable species evidence, which is a retrieval limitation rather than a crash.

    ``restrict_to`` limits generation to specific reaction ids, used to isolate which
    reaction caused a batch failure.
    """
    from core.model_info import extract_reactions_with_ids_from_sbml
    from core.reaction.amendment_config import CofactorConfig
    from core.reaction.relaxation_workflow import map_reactions_to_kegg_with_relaxation
    from core.reaction.utils import map_chebi_to_kegg

    _, high_score = map_chebi_to_kegg(recs)
    if high_score.empty or "id" not in high_score.columns:
        return [], False

    mapped_species_ids = list(high_score["id"].astype(str).unique())
    # Reactions mentioning none of the mapped species are filtered out, so the ids must
    # come from the same pass; indexing the full id list would mislabel every reaction
    # after the first omission.
    aligned_ids, reactions, _ = extract_reactions_with_ids_from_sbml(
        str(model_file), mapped_species_ids
    )
    if restrict_to is not None:
        keep = [i for i, rid in enumerate(aligned_ids) if rid in restrict_to]
        aligned_ids = [aligned_ids[i] for i in keep]
        reactions = [reactions[i] for i in keep]

    _, match_results, _ = map_reactions_to_kegg_with_relaxation(
        reactions,
        aligned_ids,
        high_score,
        spectators=bool(GENERATION_CONFIG["spectators"]),
        cofactors_to_ignore=CofactorConfig().kegg_ids,
        top_k=GENERATION_CONFIG["top_k"],
        evaluate_candidates=bool(GENERATION_CONFIG["evaluate_candidates"]),
        include_exchange_reactions=bool(GENERATION_CONFIG["include_exchange_reactions"]),
        max_relax_level=int(GENERATION_CONFIG["max_relax_level"]),
        max_ancestor_depth=int(GENERATION_CONFIG["max_ancestor_depth"]),
        max_descendant_depth=int(GENERATION_CONFIG["max_descendant_depth"]),
        penalty_lam=float(GENERATION_CONFIG["penalty_lam"]),
        strict_errors=True,
    )
    return match_results, True


def _collapse_match_results(match_results: List[Any]) -> Dict[str, Dict[str, Any]]:
    """Group per-candidate recommendation rows back into one record per reaction."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for rec in match_results:
        rid = str(rec.id)
        entry = grouped.setdefault(
            rid, {"scores": {}, "metadata": rec.metadata or {}}
        )
        if not entry["metadata"] and rec.metadata:
            entry["metadata"] = rec.metadata
        for i, cand in enumerate(rec.candidates or []):
            score = None
            if rec.match_score and i < len(rec.match_score):
                try:
                    score = float(rec.match_score[i])
                except (TypeError, ValueError):
                    score = None
            prev = entry["scores"].get(cand)
            # A candidate can appear in several split records; keep the best score.
            if prev is None or (score is not None and score > prev):
                entry["scores"][str(cand)] = score
    return grouped


def process_model(model_id: str, scope: Optional[str] = None) -> Dict[str, Any]:
    """Generate candidates for one model. Safe to call in a worker process.

    ``scope`` must be passed explicitly: spawned workers re-import this module, so a
    CLI override applied in the parent would not otherwise reach them (and would make
    the child compute a different ``config_id``).
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logging.getLogger("core").setLevel(logging.ERROR)
    if scope:
        GENERATION_CONFIG["reaction_scope"] = scope

    started = time.time()
    outcome = ModelOutcome(model_id=model_id)
    cid = config_id()

    reactions_df = pd.read_csv(REACTIONS_CSV)
    species_df = pd.read_csv(SPECIES_CSV)

    model_rows = reactions_df[reactions_df["model_id"] == model_id]
    evaluable = model_rows[model_rows["included_in_eval"].astype(bool)]
    target_ids = [str(r) for r in evaluable["reaction_id"].tolist()]
    ssx_ids = {
        str(r)
        for r in model_rows[model_rows["is_exchange_ssx"].astype(bool)]["reaction_id"].tolist()
    }

    model_file = MODELS_DIR / f"{model_id}.xml"
    if not model_file.exists():
        outcome.failures.append({
            "model_id": model_id,
            "reaction_id": "",
            "scope": "model",
            "failure_type": "model_file_missing",
            "message": f"{model_file} not found",
            "traceback_tail": "",
        })
        for rid in target_ids:
            outcome.status.append(_status_row(model_id, rid, STATUS_FAILED, cid))
        outcome.elapsed_s = time.time() - started
        return outcome.to_dict()

    from core.model_info import get_all_reaction_ids

    recs = _species_recommendations(species_df, model_id)
    reaction_ids = [str(r) for r in get_all_reaction_ids(str(model_file))]

    grouped: Dict[str, Dict[str, Any]] = {}
    isolated_failures: Dict[str, str] = {}
    had_evidence = True
    scope = str(GENERATION_CONFIG.get("reaction_scope", "evaluable"))
    restrict = set(target_ids) if scope == "evaluable" else None

    try:
        match_results, had_evidence = _run_generator(
            model_file, recs, reaction_ids, restrict_to=restrict
        )
        grouped = _collapse_match_results(match_results)
    except Exception as exc:  # noqa: BLE001 - recorded as an explicit pipeline failure
        outcome.failures.append({
            "model_id": model_id,
            "reaction_id": "",
            "scope": "model",
            "failure_type": f"model_batch_{type(exc).__name__}",
            "message": str(exc)[:500],
            "traceback_tail": _tb_tail(),
        })
        # Isolate: retry one reaction at a time so a single bad reaction does not
        # discard the whole model's candidates.
        grouped, isolated_failures = _isolate_per_reaction(
            model_file, recs, reaction_ids, sorted(restrict) if restrict else reaction_ids
        )
        for rid, msg in isolated_failures.items():
            outcome.failures.append({
                "model_id": model_id,
                "reaction_id": rid,
                "scope": "reaction",
                "failure_type": "reaction_generation_error",
                "message": msg[:500],
                "traceback_tail": "",
            })

    for rid in target_ids:
        entry = grouped.get(rid)
        if rid in isolated_failures:
            outcome.status.append(_status_row(model_id, rid, STATUS_FAILED, cid))
            continue
        if entry is None:
            if rid in ssx_ids:
                status = STATUS_EXCHANGE_SKIPPED
            elif not had_evidence:
                # The generator never ran for this model: no usable species evidence.
                status = STATUS_NO_SPECIES_EVIDENCE
            else:
                status = STATUS_ABSENT
            if status == STATUS_ABSENT:
                outcome.failures.append({
                    "model_id": model_id,
                    "reaction_id": rid,
                    "scope": "reaction",
                    "failure_type": "absent_from_generator_output",
                    "message": "reaction present in frozen table but not returned by generator",
                    "traceback_tail": "",
                })
            outcome.status.append(_status_row(model_id, rid, status, cid))
            continue

        metadata = entry["metadata"] or {}
        level, direction = _relaxation_summary(metadata)
        scores = entry["scores"]

        if metadata.get("exchange_skipped"):
            outcome.status.append(_status_row(
                model_id, rid, STATUS_EXCHANGE_SKIPPED, cid,
                relaxation_level=level, relaxation_direction=direction, metadata=metadata,
            ))
            continue

        if not scores:
            outcome.status.append(_status_row(
                model_id, rid, STATUS_NO_CANDIDATES, cid,
                relaxation_level=level, relaxation_direction=direction, metadata=metadata,
            ))
            continue

        # Zero mapped constraints => the "candidate set" is the whole KEGG database.
        if int(metadata.get("filtered_species_count", 0) or 0) == 0:
            outcome.status.append(_status_row(
                model_id, rid, STATUS_UNCONSTRAINED, cid,
                relaxation_level=level, relaxation_direction=direction, metadata=metadata,
                degenerate_set_size=len(scores),
            ))
            continue

        # Deterministic rank: score descending, KEGG id ascending as tie-break.
        ordered = sorted(
            scores.items(),
            key=lambda kv: (-(kv[1] if kv[1] is not None else -1.0), kv[0]),
        )
        for rank, (cand, score) in enumerate(ordered, start=1):
            outcome.candidates.append({
                "model_id": model_id,
                "reaction_id": rid,
                "candidate_kegg": cand,
                "raw_rank": rank,
                "heuristic_score": None if score is None else round(float(score), 6),
                "relaxation_level": level,
                "relaxation_direction": direction,
                "config_id": cid,
            })
        outcome.status.append(_status_row(
            model_id, rid, STATUS_OK, cid, num_candidates=len(ordered),
            relaxation_level=level, relaxation_direction=direction, metadata=metadata,
        ))

    outcome.elapsed_s = time.time() - started
    return outcome.to_dict()


def _status_row(
    model_id: str,
    reaction_id: str,
    status: str,
    cid: str,
    *,
    num_candidates: int = 0,
    relaxation_level: int = 0,
    relaxation_direction: str = "exact",
    metadata: Optional[Dict[str, Any]] = None,
    degenerate_set_size: int = 0,
) -> Dict[str, Any]:
    metadata = metadata or {}
    return {
        "model_id": model_id,
        "reaction_id": reaction_id,
        "status": status,
        "num_candidates": int(num_candidates),
        "degenerate_set_size": int(degenerate_set_size),
        "relaxation_level": int(relaxation_level),
        "relaxation_direction": relaxation_direction,
        "reaction_class": str(metadata.get("reaction_class", "") or ""),
        "filtered_species_count": int(metadata.get("filtered_species_count", 0) or 0),
        "config_id": cid,
    }


def _tb_tail(limit: int = 600) -> str:
    return traceback.format_exc()[-limit:].replace("\n", " | ")


def _isolate_per_reaction(
    model_file: Path,
    recs: pd.DataFrame,
    reaction_ids: List[str],
    targets: Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Re-run generation one reaction at a time to attribute a batch failure."""
    grouped: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}
    for rid in (targets if targets is not None else reaction_ids):
        try:
            results, _ = _run_generator(model_file, recs, reaction_ids, restrict_to={rid})
            grouped.update(_collapse_match_results(results))
        except Exception as exc:  # noqa: BLE001
            failures[rid] = f"{type(exc).__name__}: {exc}"
    return grouped, failures


def _cache_path(model_id: str) -> Path:
    return CACHE_DIR / f"{model_id}.json"


def _load_cached(model_id: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(model_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if payload.get("config_id") != config_id():
        return None
    return payload


def _worker(model_id: str, scope: str) -> str:
    """Process one model and persist its cache file. Returns the model id."""
    payload = process_model(model_id, scope=scope)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(payload, _cache_path(model_id))
    return model_id


def assemble(model_ids: List[str]) -> Dict[str, Any]:
    """Collect per-model caches into the frozen Phase 2 candidate artifacts."""
    cand_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    failure_rows: List[Dict[str, Any]] = []
    timings: Dict[str, float] = {}
    missing: List[str] = []

    for model_id in model_ids:
        payload = _load_cached(model_id)
        if payload is None:
            missing.append(model_id)
            continue
        cand_rows.extend(payload.get("candidates", []))
        status_rows.extend(payload.get("status", []))
        failure_rows.extend(payload.get("failures", []))
        timings[model_id] = float(payload.get("elapsed_s", 0.0))

    cand_cols = [
        "model_id", "reaction_id", "candidate_kegg", "raw_rank", "heuristic_score",
        "relaxation_level", "relaxation_direction", "config_id",
    ]
    status_cols = [
        "model_id", "reaction_id", "status", "num_candidates", "degenerate_set_size",
        "relaxation_level", "relaxation_direction", "reaction_class",
        "filtered_species_count", "config_id",
    ]
    failure_cols = [
        "model_id", "reaction_id", "scope", "failure_type", "message", "traceback_tail",
    ]

    cand_df = pd.DataFrame(cand_rows, columns=cand_cols)
    status_df = pd.DataFrame(status_rows, columns=status_cols)
    fail_df = pd.DataFrame(failure_rows, columns=failure_cols)

    if not cand_df.empty:
        cand_df = cand_df.sort_values(["model_id", "reaction_id", "raw_rank", "candidate_kegg"])
    if not status_df.empty:
        status_df = status_df.sort_values(["model_id", "reaction_id"])
    if not fail_df.empty:
        fail_df = fail_df.sort_values(["model_id", "reaction_id", "failure_type"])

    write_csv(cand_df.reset_index(drop=True), CANDIDATES_CSV)
    write_csv(status_df.reset_index(drop=True), STATUS_CSV)
    write_csv(fail_df.reset_index(drop=True), FAILURES_CSV)

    summary = {
        "config_id": config_id(),
        "generation_config": GENERATION_CONFIG,
        "models_requested": len(model_ids),
        "models_assembled": len(model_ids) - len(missing),
        "models_missing_cache": missing,
        "reactions_with_status": int(len(status_df)),
        "status_counts": (
            status_df["status"].value_counts().sort_index().to_dict() if not status_df.empty else {}
        ),
        "candidate_rows": int(len(cand_df)),
        "distinct_candidates": int(cand_df["candidate_kegg"].nunique()) if not cand_df.empty else 0,
        "pipeline_failures": int(len(fail_df)),
        "failure_types": (
            fail_df["failure_type"].value_counts().sort_index().to_dict() if not fail_df.empty else {}
        ),
        "total_generation_seconds": round(sum(timings.values()), 1),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
        },
        "inputs": {
            "reactions_csv_sha256": sha256_file(REACTIONS_CSV),
            "species_annotations_csv_sha256": sha256_file(SPECIES_CSV),
        },
        "outputs": {
            "candidates_csv_sha256": sha256_file(CANDIDATES_CSV),
            "candidate_status_csv_sha256": sha256_file(STATUS_CSV),
            "candidate_generation_failures_csv_sha256": sha256_file(FAILURES_CSV),
        },
    }
    write_json(summary, CONFIG_JSON)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=None, help="Model ids (default: all)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--force", action="store_true", help="Ignore cached per-model results")
    parser.add_argument("--assemble-only", action="store_true", help="Only rebuild the tables")
    parser.add_argument(
        "--scope", choices=["evaluable", "all"], default=GENERATION_CONFIG["reaction_scope"],
        help="Generate for reactions carrying ground truth (default) or every model reaction. "
             "'all' reproduces production behaviour but is substantially slower.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the N smallest pending models (smoke test)",
    )
    args = parser.parse_args()

    GENERATION_CONFIG["reaction_scope"] = args.scope

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    reactions_df = pd.read_csv(REACTIONS_CSV)
    all_models = sorted(reactions_df["model_id"].astype(str).unique())
    model_ids = args.models or all_models
    unknown = [m for m in model_ids if m not in set(all_models)]
    if unknown:
        logger.error("Unknown model ids: %s", unknown)
        return 2

    if not args.assemble_only:
        todo = [m for m in model_ids if args.force or _load_cached(m) is None]
        logger.info(
            "config_id=%s | %d models requested, %d need generation, %d cached",
            config_id(), len(model_ids), len(todo), len(model_ids) - len(todo),
        )
        if todo:
            sizes = reactions_df.groupby("model_id").size()
            if args.limit:
                # Smallest first, so a smoke test finishes quickly.
                todo.sort(key=lambda m: int(sizes.get(m, 0)))
                todo = todo[: args.limit]
            else:
                # Largest models first so the long tail does not straggle at the end.
                todo.sort(key=lambda m: -int(sizes.get(m, 0)))
            logger.info(
                "scope=%s | pending reactions in scope: %d",
                args.scope, int(sum(int(sizes.get(m, 0)) for m in todo)),
            )
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            done = 0
            if args.workers <= 1:
                for m in todo:
                    started = time.time()
                    _worker(m, args.scope)
                    done += 1
                    payload = _load_cached(m) or {}
                    logger.info(
                        "[%d/%d] %s done in %.0fs (%d candidate rows)",
                        done, len(todo), m, time.time() - started,
                        len(payload.get("candidates", [])),
                    )
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(_worker, m, args.scope): m for m in todo}
                    for fut in as_completed(futures):
                        m = futures[fut]
                        done += 1
                        try:
                            fut.result()
                            payload = _load_cached(m) or {}
                            logger.info(
                                "[%d/%d] %s done in %.0fs (%d candidate rows)",
                                done, len(todo), m, payload.get("elapsed_s", 0.0),
                                len(payload.get("candidates", [])),
                            )
                        except Exception as exc:  # noqa: BLE001
                            logger.error("[%d/%d] %s WORKER CRASH: %s", done, len(todo), m, exc)

    summary = assemble(model_ids)
    logger.info("Status counts: %s", summary["status_counts"])
    logger.info("Candidate rows: %s", summary["candidate_rows"])
    logger.info("Pipeline failures: %s %s", summary["pipeline_failures"], summary["failure_types"])
    if summary["models_missing_cache"]:
        logger.error("Missing caches: %s", summary["models_missing_cache"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
