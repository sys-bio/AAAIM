"""Phase 2 audit: check the frozen candidate artifacts against explicit invariants.

This is a read-only consistency check over the artifacts already produced by
``generate_candidates.py``. It never regenerates candidates, so it is cheap to run
before a freeze and safe to run repeatedly.

Each invariant is reported as an independent pass/fail record with the observed and
expected values, so a failure names the specific violated property and a sample of the
offending rows rather than only a boolean.

Usage::

    python benchmark/scripts/audit_phase2.py
    python benchmark/scripts/audit_phase2.py --expect-config-id 86938b48ab88
    python benchmark/scripts/audit_phase2.py --check-reassembly
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.scripts.generate_candidates import (
    CANDIDATES_CSV,
    CONFIG_JSON,
    DATA_DIR,
    FAILURES_CSV,
    REACTIONS_CSV,
    STATUS_CSV,
    STATUS_NO_CANDIDATES,
    STATUS_OK,
    STATUS_UNCONSTRAINED,
    config_id,
    sha256_file,
    write_json,
)

CACHE_DIR = DATA_DIR / "_candidates_cache"
OUT_JSON = DATA_DIR / "phase2_audit.json"

# Statuses that must never carry candidate rows. `no_candidates` is empty by
# definition; `unconstrained_candidate_set` is deliberately dropped because an empty
# constraint set matches all of KEGG (see GENERATION_CONFIG notes).
EMPTY_STATUSES = (STATUS_NO_CANDIDATES, STATUS_UNCONSTRAINED)

KEGG_REACTION_RE = re.compile(r"^R\d{5}$")

# Phase 1 artifacts that Phase 2 consumes but must not modify.
PHASE1_ARTIFACTS = (
    "reactions.csv",
    "model_summary.csv",
    "exclusions.csv",
    "model_clusters.csv",
    "duplicate_groups.csv",
    "species_annotations.csv",
    "invariants.json",
    "VERSION.json",
)

logger = logging.getLogger("audit_phase2")


class Audit:
    """Accumulates invariant results."""

    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def record(
        self,
        name: str,
        passed: bool,
        *,
        expected: Any = None,
        observed: Any = None,
        detail: Any = None,
    ) -> bool:
        self.checks.append({
            "check": name,
            "passed": bool(passed),
            "expected": expected,
            "observed": observed,
            "detail": detail,
        })
        return bool(passed)

    @property
    def failed(self) -> List[Dict[str, Any]]:
        return [c for c in self.checks if not c["passed"]]


def _sample(values, limit: int = 10) -> List[str]:
    return [str(v) for v in list(values)[:limit]]


def audit(
    expect_config_id: Optional[str] = None,
    expect_models: Optional[int] = None,
    expect_reactions: Optional[int] = None,
) -> Dict[str, Any]:
    a = Audit()

    reactions = pd.read_csv(REACTIONS_CSV)
    evaluable = reactions[reactions.included_in_eval.astype(bool)].copy()
    evaluable["model_id"] = evaluable.model_id.astype(str)
    evaluable["reaction_id"] = evaluable.reaction_id.astype(str)

    status = pd.read_csv(STATUS_CSV)
    status["model_id"] = status.model_id.astype(str)
    status["reaction_id"] = status.reaction_id.astype(str)

    candidates = pd.read_csv(CANDIDATES_CSV)
    candidates["model_id"] = candidates.model_id.astype(str)
    candidates["reaction_id"] = candidates.reaction_id.astype(str)
    candidates["candidate_kegg"] = candidates.candidate_kegg.astype(str)

    failures = pd.read_csv(FAILURES_CSV)

    expected_models = sorted(evaluable.model_id.unique())
    live_config = config_id()

    # 1. Every included model has a cache compatible with the live config.
    cache_ids = set()
    incompatible: List[str] = []
    for path in sorted(CACHE_DIR.glob("*.json")) if CACHE_DIR.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            incompatible.append(f"{path.stem}:unreadable")
            continue
        if payload.get("config_id") != live_config:
            incompatible.append(f"{path.stem}:{payload.get('config_id')}")
            continue
        cache_ids.add(str(payload.get("model_id", path.stem)))
    missing_caches = [m for m in expected_models if m not in cache_ids]
    a.record(
        "all_included_models_have_compatible_cache",
        not missing_caches and not incompatible,
        expected=f"{len(expected_models)} compatible caches",
        observed=f"{len(cache_ids)} compatible, {len(missing_caches)} missing, "
                 f"{len(incompatible)} incompatible",
        detail={"missing": _sample(missing_caches), "incompatible": _sample(incompatible)},
    )

    if expect_models is not None:
        a.record("model_count_matches_expected", len(expected_models) == expect_models,
                 expected=expect_models, observed=len(expected_models))

    # 2. Exactly one status row per evaluable reaction, and no extras.
    eval_keys = set(zip(evaluable.model_id, evaluable.reaction_id))
    status_keys = list(zip(status.model_id, status.reaction_id))
    dupes = [k for k, n in pd.Series(status_keys).value_counts().items() if n > 1]
    missing_status = sorted(eval_keys - set(status_keys))
    extra_status = sorted(set(status_keys) - eval_keys)
    a.record(
        "exactly_one_status_row_per_evaluable_reaction",
        not dupes and not missing_status and not extra_status,
        expected=f"{len(eval_keys)} unique keys",
        observed=f"{len(status_keys)} rows, {len(set(status_keys))} unique",
        detail={
            "duplicated": _sample(dupes),
            "missing_status": _sample(missing_status),
            "status_without_evaluable_reaction": _sample(extra_status),
        },
    )

    # 3. Status counts sum to the evaluable corpus.
    counts = status.status.value_counts().sort_index().to_dict()
    a.record("status_counts_sum_to_evaluable", int(sum(counts.values())) == len(eval_keys),
             expected=len(eval_keys), observed=int(sum(counts.values())), detail=counts)

    if expect_reactions is not None:
        a.record("reaction_count_matches_expected", len(eval_keys) == expect_reactions,
                 expected=expect_reactions, observed=len(eval_keys))

    # 4. Every `ok` reaction has at least one candidate row.
    rows_per_key = candidates.groupby(["model_id", "reaction_id"]).size()
    ok = status[status.status == STATUS_OK]
    ok_keys = set(zip(ok.model_id, ok.reaction_id))
    ok_without_rows = sorted(k for k in ok_keys if rows_per_key.get(k, 0) == 0)
    a.record("every_ok_reaction_has_candidates", not ok_without_rows,
             expected=0, observed=len(ok_without_rows), detail=_sample(ok_without_rows))

    # `num_candidates` must agree with the rows actually stored.
    mismatched = [
        {"model_id": r.model_id, "reaction_id": r.reaction_id,
         "num_candidates": int(r.num_candidates),
         "stored_rows": int(rows_per_key.get((r.model_id, r.reaction_id), 0))}
        for r in ok.itertuples()
        if int(r.num_candidates) != int(rows_per_key.get((r.model_id, r.reaction_id), 0))
    ]
    a.record("num_candidates_matches_stored_rows", not mismatched,
             expected=0, observed=len(mismatched), detail=mismatched[:10])

    # 5. Statuses that mean "no usable candidate set" store zero rows.
    empty = status[status.status.isin(EMPTY_STATUSES)]
    empty_with_rows = sorted(
        k for k in set(zip(empty.model_id, empty.reaction_id)) if rows_per_key.get(k, 0) > 0
    )
    a.record(
        "empty_statuses_have_no_candidate_rows",
        not empty_with_rows,
        expected=0,
        observed=len(empty_with_rows),
        detail={"statuses": list(EMPTY_STATUSES), "offenders": _sample(empty_with_rows)},
    )

    # 6. Ranks are unique and consecutive from 1 within each reaction.
    bad_ranks: List[Dict[str, Any]] = []
    for key, sub in candidates.groupby(["model_id", "reaction_id"], sort=False):
        ranks = sorted(int(x) for x in sub.raw_rank)
        if ranks != list(range(1, len(ranks) + 1)):
            bad_ranks.append({
                "model_id": key[0], "reaction_id": key[1],
                "n": len(ranks), "min": ranks[0], "max": ranks[-1],
                "unique": len(set(ranks)),
            })
    a.record("candidate_ranks_unique_and_consecutive", not bad_ranks,
             expected=0, observed=len(bad_ranks), detail=bad_ranks[:10])

    # Candidates must also be unique within a reaction: a duplicated KEGG id would
    # inflate set sizes and silently distort recall denominators.
    dup_cands = candidates[candidates.duplicated(
        subset=["model_id", "reaction_id", "candidate_kegg"], keep=False)]
    a.record("candidates_unique_within_reaction", dup_cands.empty,
             expected=0, observed=int(len(dup_cands)),
             detail=_sample(zip(dup_cands.model_id, dup_cands.reaction_id)))

    # 7. KEGG ids are well formed.
    bad_ids = sorted({c for c in candidates.candidate_kegg.unique()
                      if not KEGG_REACTION_RE.match(str(c))})
    a.record("candidate_kegg_ids_well_formed", not bad_ids,
             expected="all match ^R[0-9]{5}$", observed=f"{len(bad_ids)} malformed",
             detail=_sample(bad_ids))

    # 8. Every artifact carries the expected config_id.
    target = expect_config_id or live_config
    cfg = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    stamped = {
        "live_config_id": live_config,
        "config_json": cfg.get("config_id"),
        "candidates_csv": sorted({str(v) for v in candidates.config_id.unique()}),
        "candidate_status_csv": sorted({str(v) for v in status.config_id.unique()}),
    }
    consistent = (
        live_config == target
        and cfg.get("config_id") == target
        and stamped["candidates_csv"] == [target]
        and stamped["candidate_status_csv"] == [target]
    )
    a.record("config_id_consistent_across_artifacts", consistent,
             expected=target, observed=stamped)

    # 9. No pipeline failures.
    a.record("no_pipeline_failures", failures.empty,
             expected=0, observed=int(len(failures)),
             detail=failures.head(10).to_dict("records") if not failures.empty else None)

    # Cross-check the recorded digests in the config against the files on disk, so a
    # post-hoc edit of an artifact cannot pass unnoticed.
    digest_mismatches = []
    for key, path in (
        ("candidates_csv_sha256", CANDIDATES_CSV),
        ("candidate_status_csv_sha256", STATUS_CSV),
        ("candidate_generation_failures_csv_sha256", FAILURES_CSV),
    ):
        recorded = (cfg.get("outputs") or {}).get(key)
        actual = sha256_file(path)
        if recorded != actual:
            digest_mismatches.append({"artifact": path.name, "recorded": recorded,
                                      "actual": actual})
    a.record("recorded_output_digests_match_files", not digest_mismatches,
             expected=0, observed=len(digest_mismatches), detail=digest_mismatches)

    summary = {
        "config_id": live_config,
        "expected_config_id": target,
        "models": len(expected_models),
        "evaluable_reactions": len(eval_keys),
        "status_counts": counts,
        "candidate_rows": int(len(candidates)),
        "distinct_candidates": int(candidates.candidate_kegg.nunique()),
        "pipeline_failures": int(len(failures)),
        "checks": a.checks,
        "checks_passed": int(sum(1 for c in a.checks if c["passed"])),
        "checks_failed": int(len(a.failed)),
        "all_passed": not a.failed,
        "artifact_digests": {
            p.name: sha256_file(p)
            for p in (CANDIDATES_CSV, STATUS_CSV, FAILURES_CSV, CONFIG_JSON)
        },
        "phase1_artifact_digests": {
            name: sha256_file(DATA_DIR / name)
            for name in PHASE1_ARTIFACTS if (DATA_DIR / name).exists()
        },
    }
    return summary


def check_reassembly() -> Dict[str, Any]:
    """Re-run assembly from the existing caches and confirm byte-identical outputs.

    Only reads the per-model caches; it never invokes the generator.
    """
    from benchmark.scripts.generate_candidates import assemble

    targets = (CANDIDATES_CSV, STATUS_CSV, FAILURES_CSV)
    before = {p.name: sha256_file(p) for p in targets}

    reactions = pd.read_csv(REACTIONS_CSV)
    evaluable = reactions[reactions.included_in_eval.astype(bool)]
    model_ids = sorted(evaluable.model_id.astype(str).unique())
    assemble(model_ids, partial_ok=False)

    after = {p.name: sha256_file(p) for p in targets}
    changed = [name for name in before if before[name] != after[name]]
    return {
        "identical": not changed,
        "changed": changed,
        "before": before,
        "after": after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-config-id", default=None,
                        help="Fail unless every artifact carries this config_id")
    parser.add_argument("--expect-models", type=int, default=None)
    parser.add_argument("--expect-reactions", type=int, default=None)
    parser.add_argument("--check-reassembly", action="store_true",
                        help="Re-assemble from caches and require byte-identical outputs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    summary = audit(
        expect_config_id=args.expect_config_id,
        expect_models=args.expect_models,
        expect_reactions=args.expect_reactions,
    )

    if args.check_reassembly:
        summary["reassembly"] = check_reassembly()

    write_json(summary, OUT_JSON)

    for check in summary["checks"]:
        logger.info("[%s] %s (expected=%s observed=%s)",
                    "PASS" if check["passed"] else "FAIL", check["check"],
                    check["expected"], check["observed"])
    if "reassembly" in summary:
        logger.info("[%s] reassembly_byte_identical",
                    "PASS" if summary["reassembly"]["identical"] else "FAIL")

    logger.info("%d/%d checks passed; config_id=%s; %d models; %d evaluable reactions",
                summary["checks_passed"], len(summary["checks"]),
                summary["config_id"], summary["models"], summary["evaluable_reactions"])

    ok = summary["all_passed"] and summary.get("reassembly", {"identical": True})["identical"]
    if not ok:
        for check in summary["checks"]:
            if not check["passed"]:
                logger.error("FAILED: %s -> %s", check["check"], check["detail"])
        return 1
    logger.info("Phase 2 artifacts pass every invariant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
