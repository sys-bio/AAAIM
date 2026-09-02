"""Write the Phase 2 release manifest.

Records everything needed to identify, verify and reproduce the frozen Phase 2
artifacts: configuration, source commit, per-artifact digests and row counts, the exact
commands, environment versions, and which optional baselines were included.

The manifest carries no timestamp on purpose, so re-running on unchanged inputs is
byte-identical and the file can be diffed meaningfully. The git commit is the temporal
anchor.

Usage::

    python benchmark/scripts/freeze_phase2.py
    python benchmark/scripts/freeze_phase2.py --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from benchmark.scripts.generate_candidates import (
    DATA_DIR,
    GENERATION_CONFIG,
    config_id,
    sha256_file,
    write_json,
)

OUT_JSON = REPO_ROOT / "benchmark" / "PHASE2_MANIFEST.json"

BENCHMARK_VERSION = "phase2-v1"
PROPOSED_TAG = "benchmark-phase2-v1"
PHASE1_TAG = "benchmark-phase1-v1"

# Frozen Phase 2 artifacts, in dependency order. `optional` entries are recorded when
# present and reported as outstanding when absent.
ARTIFACTS: List[Dict[str, Any]] = [
    {"name": "candidates.csv", "step": "generate_candidates"},
    {"name": "candidate_status.csv", "step": "generate_candidates"},
    {"name": "candidate_generation_failures.csv", "step": "generate_candidates"},
    {"name": "candidate_generation_config.json", "step": "generate_candidates"},
    {"name": "reaction_retrieval.csv", "step": "analyze_retrieval"},
    {"name": "retrieval_ceiling.json", "step": "analyze_retrieval"},
    {"name": "retrieval_ceiling_by_stratum.csv", "step": "analyze_retrieval"},
    {"name": "baseline_rankings.csv", "step": "rank_baselines"},
    {"name": "baseline_table.csv", "step": "rank_baselines"},
    {"name": "baseline_summary.json", "step": "rank_baselines"},
    {"name": "failure_stratification.csv", "step": "rank_baselines"},
    {"name": "candidate_diagnostics.json", "step": "candidate_diagnostics"},
    {"name": "candidate_size_by_stratum.csv", "step": "candidate_diagnostics"},
    {"name": "candidate_largest_sets.csv", "step": "candidate_diagnostics"},
    {"name": "phase2_audit.json", "step": "audit_phase2"},
]

# Phase 1 artifacts Phase 2 consumes read-only. Their digests are recorded so a future
# drift in the inputs is detectable.
PHASE1_INPUTS = ("reactions.csv", "species_annotations.csv", "model_clusters.csv",
                 "VERSION.json")
# Phase 2 inputs built on the phase-2 branch before candidate generation.
PHASE2_INPUTS = ("species_evidence.csv", "reaction_strata.csv", "reaction_text.csv")

COMMANDS = {
    "build_inputs": [
        "python benchmark/scripts/build_species_evidence.py",
        "python benchmark/scripts/build_reaction_strata.py",
        "python benchmark/scripts/build_reaction_text.py",
    ],
    "generation": [
        "# ~3 days wall clock, resumable; caches land in benchmark/data/_candidates_cache/",
        "python benchmark/scripts/generate_candidates.py --workers 4 --scope evaluable",
    ],
    "analysis": [
        "python benchmark/scripts/generate_candidates.py --assemble-only",
        "python benchmark/scripts/analyze_retrieval.py",
        "python benchmark/scripts/rank_baselines.py "
        "--rankers heuristic lexical random oracle",
        "python benchmark/scripts/candidate_diagnostics.py",
        "python benchmark/scripts/audit_phase2.py --expect-config-id 86938b48ab88 "
        "--expect-models 74 --expect-reactions 5816 --check-reassembly",
    ],
    "freeze": ["python benchmark/scripts/freeze_phase2.py"],
    "tests": ["python -m pytest tests/test_phase2_candidates.py "
              "tests/test_phase2_audit.py tests/test_benchmark_build.py -q"],
}

KNOWN_LIMITATIONS = [
    "Embedding baseline (MiniLM) not run: chromadb 1.0.11 and onnxruntime are installed "
    "and the model is SHA-256 pinned, but the ~80 MB asset is not in the local cache and "
    "was not downloaded. Explicitly outstanding and optional.",
    "LLM reranking baseline not run: requires paid API calls.",
    "131 reactions hold candidate sets larger than 100, contributing 94% of all candidate "
    "rows while containing the exact answer only 13.7% of the time and never ranking it "
    "first. Cause is documented in PHASE2_RESULTS.md section 5; nothing is capped.",
    "BIOMD0000001063 alone contributes 83.96% of candidate rows. Diagnosed as sparse "
    "species annotation degenerating subset-containment retrieval, not a bookkeeping bug; "
    "effect on corpus metrics is about 0.2 pp.",
    "Only one generation configuration was run; no sensitivity sweep over "
    "max_relax_level, cofactor handling or top_k.",
    "Cluster-macro averages use the conservative Phase 1 clustering that keeps the seven "
    "yeast models together; the provenance-based sensitivity analysis is outstanding.",
    "Per-model resumability caches (benchmark/data/_candidates_cache/) are git-ignored. "
    "The committed aggregate artifacts are the frozen deliverable, and reassembly from "
    "caches is verified byte-identical, but regenerating caches from scratch is a ~3-day "
    "pass.",
]


def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _row_count(path: Path) -> Optional[int]:
    if path.suffix != ".csv":
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return max(0, sum(1 for _ in fh) - 1)


def _package_versions() -> Dict[str, Optional[str]]:
    from importlib.metadata import PackageNotFoundError, version

    names = ["pandas", "numpy", "scikit-learn", "python-libsbml", "chromadb",
             "onnxruntime", "requests", "pytest"]
    out: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _embedding_status() -> Dict[str, Any]:
    """Whether the embedding baseline ran, and if not, exactly what is missing."""
    import importlib.util

    summary_path = DATA_DIR / "baseline_summary.json"
    rankers: List[str] = []
    if summary_path.exists():
        rankers = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "rankers_run", [])

    model_dir = (Path.home() / ".cache" / "chroma" / "onnx_models"
                 / "all-MiniLM-L6-v2" / "onnx")
    deps = {name: importlib.util.find_spec(name) is not None
            for name in ("chromadb", "onnxruntime")}
    asset_present = (model_dir / "model.onnx").exists()

    missing: List[str] = [f"python package: {n}" for n, ok in deps.items() if not ok]
    if not asset_present:
        missing.append(
            "model asset: all-MiniLM-L6-v2 onnx.tar.gz (~80 MB) from "
            "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz, "
            "sha256 913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3, "
            f"expected under {model_dir}"
        )
    return {
        "included": "embedding" in rankers,
        "dependencies_installed": deps,
        "model_asset_cached": asset_present,
        "model_asset_pinned_by_sha256": True,
        "missing": missing,
        "blocks_freeze": False,
        "note": "Optional. Bounded by the same oracle ceiling as every other reranker "
                "(+1.46 pp overall), so it cannot change the retrieval-first conclusion.",
    }


def build_manifest() -> Dict[str, Any]:
    artifacts: List[Dict[str, Any]] = []
    outstanding: List[str] = []
    for spec in ARTIFACTS:
        path = DATA_DIR / spec["name"]
        if not path.exists():
            outstanding.append(spec["name"])
            continue
        entry = {
            "name": spec["name"],
            "step": spec["step"],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        rows = _row_count(path)
        if rows is not None:
            entry["rows"] = rows
        artifacts.append(entry)

    audit_path = DATA_DIR / "phase2_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}

    status_path = DATA_DIR / "candidate_status.csv"
    counts: Dict[str, Any] = {}
    if status_path.exists():
        status = pd.read_csv(status_path)
        counts = {
            "models": int(status.model_id.nunique()),
            "evaluable_reactions": int(len(status)),
            "status_counts": status.status.value_counts().sort_index().to_dict(),
            "candidate_rows": _row_count(DATA_DIR / "candidates.csv"),
            "pipeline_failures": _row_count(
                DATA_DIR / "candidate_generation_failures.csv"),
        }

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "proposed_tag": PROPOSED_TAG,
        "builds_on": PHASE1_TAG,
        "source_commit": _git("rev-parse", "HEAD"),
        "source_commit_note": "HEAD at freeze time, i.e. the code and caches that produced "
                              "these artifacts. The commit that adds this manifest and the "
                              "artifacts themselves is its child; tag that child.",
        "source_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "config_id": config_id(),
        "generation_config": GENERATION_CONFIG,
        "counts": counts,
        "audit": {
            "checks_passed": audit.get("checks_passed"),
            "checks_failed": audit.get("checks_failed"),
            "all_passed": audit.get("all_passed"),
            "reassembly_byte_identical": (audit.get("reassembly") or {}).get("identical"),
        },
        "artifacts": artifacts,
        "artifacts_outstanding": outstanding,
        "phase1_inputs": {
            name: sha256_file(DATA_DIR / name)
            for name in PHASE1_INPUTS if (DATA_DIR / name).exists()
        },
        "phase2_inputs": {
            name: sha256_file(DATA_DIR / name)
            for name in PHASE2_INPUTS if (DATA_DIR / name).exists()
        },
        "commands": COMMANDS,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "determinism": {
                "PYTHONHASHSEED": "0",
                "random_seed": 20260901,
                "notes": "CSV/JSON writers force LF line endings so digests verify on "
                         "any platform; candidate ranks break ties on ascending KEGG id.",
            },
        },
        "embedding_baseline": _embedding_status(),
        "git_tracking": {
            "committed": [a["name"] for a in artifacts],
            "ignored": ["benchmark/data/_candidates_cache/ (per-model resumability "
                        "caches, 25.7 MB, regenerable only by the ~3-day pass)",
                        "benchmark/data/_*.log (transient run logs)",
                        "benchmark/models/ (SBML inputs; see benchmark/dist/)"],
            "rationale": "Aggregate artifacts are the frozen deliverable and are small "
                         "enough to version (9.5 MB total), matching the Phase 1 "
                         "precedent of committing reactions.csv and "
                         "species_annotations.csv. The underscore-prefixed caches are "
                         "intermediates: assembly from them is verified byte-identical, "
                         "so they add no information the committed artifacts lack.",
        },
        "known_limitations": KNOWN_LIMITATIONS,
    }


def verify(manifest: Dict[str, Any]) -> List[str]:
    """Re-hash every recorded artifact and report mismatches."""
    problems: List[str] = []
    for entry in manifest["artifacts"]:
        path = DATA_DIR / entry["name"]
        if not path.exists():
            problems.append(f"{entry['name']}: missing")
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            problems.append(f"{entry['name']}: sha256 {actual} != {entry['sha256']}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="Check the existing manifest against files on disk")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("freeze_phase2")

    if args.verify:
        if not OUT_JSON.exists():
            logger.error("no manifest at %s; run without --verify first", OUT_JSON)
            return 1
        manifest = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        problems = verify(manifest)
        for p in problems:
            logger.error("MISMATCH %s", p)
        logger.info("verified %d artifacts; %d problems",
                    len(manifest["artifacts"]), len(problems))
        return 1 if problems else 0

    manifest = build_manifest()
    write_json(manifest, OUT_JSON)

    logger.info("%s: %d artifacts, config_id=%s, commit=%s",
                manifest["benchmark_version"], len(manifest["artifacts"]),
                manifest["config_id"], manifest["source_commit"])
    if manifest["artifacts_outstanding"]:
        logger.warning("outstanding artifacts: %s", manifest["artifacts_outstanding"])
    emb = manifest["embedding_baseline"]
    logger.info("embedding baseline included: %s%s", emb["included"],
                "" if emb["included"] else f" (missing: {len(emb['missing'])} item(s))")
    if not manifest["audit"]["all_passed"]:
        logger.warning("audit did not fully pass; the freeze is not clean")
    logger.info("proposed tag: %s (not created here)", manifest["proposed_tag"])
    logger.info("wrote %s", OUT_JSON.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
