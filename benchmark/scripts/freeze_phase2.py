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
    python benchmark/scripts/freeze_phase2.py --pack-caches
    python benchmark/scripts/freeze_phase2.py --verify-caches
    python benchmark/scripts/freeze_phase2.py --verify-cache-archive
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import tempfile
import zipfile
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

CACHE_DIR = DATA_DIR / "_candidates_cache"
# Per-file digests for the resumability caches. Committed so a restored archive can be
# verified file-by-file; zip archives embed timestamps, so the zip's own digest is not
# reproducible and the contents are what must be checked.
CACHE_REGISTRY = REPO_ROOT / "benchmark" / "manifest" / "candidate_cache_registry.json"

BENCHMARK_VERSION = "phase2-v1"
PROPOSED_TAG = "benchmark-phase2-v1"
PHASE1_TAG = "benchmark-phase1-v1"

# The generation pass ran against dbf15d6. The aggregate artifacts and analysis scripts
# landed in b5065e0. The release tag is the snapshot; it is not created here. These are
# recorded as constants rather than `git rev-parse HEAD` so the manifest cannot point at
# a parent that does not contain the files it describes.
CANDIDATE_GENERATION_COMMIT = "dbf15d6db3370a11f4c0889336af220705bf75b1"
ANALYSIS_ARTIFACT_COMMIT = "b5065e042fe7cc31dcaad7d72f196f879bd39ce3"

CACHE_ARCHIVE_NAME = f"aaaim-benchmark-{BENCHMARK_VERSION}-candidate-caches.zip"
CACHE_ARCHIVE = REPO_ROOT / "benchmark" / "dist" / CACHE_ARCHIVE_NAME
CACHE_ARC_PREFIX = "benchmark/data/_candidates_cache"

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
        "--expect-models 74 --expect-reactions 5816 --check-reassembly --write-report",
    ],
    "freeze": [
        "python benchmark/scripts/freeze_phase2.py --pack-caches",
        "python benchmark/scripts/freeze_phase2.py",
        "python benchmark/scripts/freeze_phase2.py --verify",
        "python benchmark/scripts/freeze_phase2.py --verify-cache-archive",
    ],
    "tests": ["python -m pytest tests -q"],
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
    "Per-model resumability caches (benchmark/data/_candidates_cache/) are git-ignored "
    "and archived as benchmark/dist/aaaim-benchmark-phase2-v1-candidate-caches.zip, "
    "verified file-by-file via benchmark/manifest/candidate_cache_registry.json. "
    "The zip is a release asset, not a git object; regenerating caches from scratch "
    "is a ~3-day pass.",
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

    registry = None
    if CACHE_REGISTRY.exists():
        registry = json.loads(CACHE_REGISTRY.read_text(encoding="utf-8"))

    archive_info: Dict[str, Any] = {
        "registry": str(CACHE_REGISTRY.relative_to(REPO_ROOT)).replace("\\", "/"),
        "asset": CACHE_ARCHIVE_NAME,
        "path": str(CACHE_ARCHIVE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "release_notes": "benchmark/dist/RELEASE_phase2-v1.md",
        "verify_command": "python benchmark/scripts/freeze_phase2.py --verify-cache-archive",
        "present": CACHE_ARCHIVE.exists(),
        "gitignored": True,
    }
    if registry:
        archive_info["n_files"] = registry.get("n_files")
        archive_info["config_id"] = registry.get("config_id")
        archive_info["uncompressed_bytes"] = registry.get("total_bytes")
    elif CACHE_DIR.exists():
        archive_info["n_files"] = len(list(CACHE_DIR.glob("*.json")))
    if CACHE_ARCHIVE.exists():
        archive_info["sha256"] = sha256_file(CACHE_ARCHIVE)
        archive_info["bytes"] = CACHE_ARCHIVE.stat().st_size
        archive_info["note"] = (
            "Zip archives embed timestamps, so this digest identifies the uploaded "
            "asset; verify restored contents against the registry, not the zip bytes."
        )

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "proposed_tag": PROPOSED_TAG,
        "builds_on": PHASE1_TAG,
        "commits": {
            "candidate_generation": CANDIDATE_GENERATION_COMMIT,
            "analysis_and_artifacts": ANALYSIS_ARTIFACT_COMMIT,
            "release_snapshot": PROPOSED_TAG,
        },
        "commits_note": (
            "candidate_generation (dbf15d6) is the last change to generate_candidates.py "
            "before the full run. analysis_and_artifacts (b5065e0) added the frozen "
            "aggregate tables and analysis scripts. release_snapshot is the proposed tag; "
            "it is not created by this script. HEAD is not recorded: a freeze commit "
            "cannot name itself."
        ),
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
                        "caches; archived as a release asset, see "
                        "benchmark/dist/RELEASE_phase2-v1.md)",
                        "benchmark/data/_*.log (transient run logs)",
                        "benchmark/models/ (SBML inputs; see benchmark/dist/)"],
            "rationale": "Aggregate artifacts are the frozen deliverable and are small "
                         "enough to version (9.5 MB total), matching the Phase 1 "
                         "precedent of committing reactions.csv and "
                         "species_annotations.csv. The underscore-prefixed caches are "
                         "intermediates: assembly from them is verified byte-identical, "
                         "so they add no information the committed artifacts lack. They "
                         "are archived rather than committed because regenerating them "
                         "costs a ~3-day pass, so resumability stays recoverable without "
                         "putting 25.7 MB of intermediates in git history.",
        },
        "cache_archive": archive_info,
        "known_limitations": KNOWN_LIMITATIONS,
    }


def build_cache_registry() -> Dict[str, Any]:
    """Digest every per-model cache so an archived copy can be verified after restore."""
    files: Dict[str, Any] = {}
    for path in sorted(CACHE_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        files[path.name] = {
            "model_id": payload.get("model_id"),
            "config_id": payload.get("config_id"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "candidate_rows": len(payload.get("candidates", [])),
            "status_rows": len(payload.get("status", [])),
            "failures": len(payload.get("failures", [])),
            "elapsed_s": payload.get("elapsed_s"),
        }
    total_bytes = sum(f["bytes"] for f in files.values())
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "config_id": config_id(),
        "purpose": "Per-model candidate-generation caches. Regenerating them costs a "
                   "~3-day pass, so they are archived as a release asset rather than "
                   "committed. Assembly from them into the committed aggregate artifacts "
                   "is verified byte-identical by audit_phase2.py --check-reassembly.",
        "n_files": len(files),
        "total_bytes": total_bytes,
        "total_generation_seconds": round(
            sum(f["elapsed_s"] or 0.0 for f in files.values()), 1),
        "files": files,
    }


def verify_caches(
    registry: Dict[str, Any], cache_dir: Optional[Path] = None,
) -> List[str]:
    """Check a cache directory against the committed registry."""
    cache_dir = cache_dir or CACHE_DIR
    problems: List[str] = []
    for name, entry in registry["files"].items():
        path = cache_dir / name
        if not path.exists():
            problems.append(f"{name}: missing")
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            problems.append(f"{name}: sha256 {actual} != {entry['sha256']}")
    extra = sorted({p.name for p in cache_dir.glob("*.json")} - set(registry["files"]))
    problems.extend(f"{name}: not in registry" for name in extra)
    return problems


def pack_caches(dest: Optional[Path] = None) -> Dict[str, Any]:
    """Zip the per-model caches. The zip is a release asset, not a git object."""
    dest = dest or CACHE_ARCHIVE
    if not CACHE_DIR.exists():
        raise FileNotFoundError(f"no cache directory at {CACHE_DIR}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(CACHE_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no cache files in {CACHE_DIR}")
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            zf.write(path, f"{CACHE_ARC_PREFIX}/{path.name}")
    return {
        "path": str(dest.relative_to(REPO_ROOT)).replace("\\", "/"),
        "n_files": len(files),
        "bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "config_id": config_id(),
    }


def restore_and_reassemble(
    archive: Path, registry: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract the archive into a temp tree, verify files, reassemble, compare CSVs.

    Never writes into the working copy's cache directory or aggregate artifacts.
    """
    from benchmark.scripts import generate_candidates as gc
    from benchmark.scripts.generate_candidates import (
        CANDIDATES_CSV, FAILURES_CSV, REACTIONS_CSV, STATUS_CSV,
    )

    committed = {
        "candidates.csv": sha256_file(CANDIDATES_CSV),
        "candidate_status.csv": sha256_file(STATUS_CSV),
        "candidate_generation_failures.csv": sha256_file(FAILURES_CSV),
    }
    redirected = ("CACHE_DIR", "CANDIDATES_CSV", "STATUS_CSV", "FAILURES_CSV", "CONFIG_JSON")
    original = {name: getattr(gc, name) for name in redirected}

    with tempfile.TemporaryDirectory(prefix="aaaim-cache-restore-") as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp_path)
        restored_dir = tmp_path / "benchmark" / "data" / "_candidates_cache"
        problems = verify_caches(registry, cache_dir=restored_dir)

        out_dir = tmp_path / "assembled"
        out_dir.mkdir()
        gc.CACHE_DIR = restored_dir
        gc.CANDIDATES_CSV = out_dir / "candidates.csv"
        gc.STATUS_CSV = out_dir / "candidate_status.csv"
        gc.FAILURES_CSV = out_dir / "candidate_generation_failures.csv"
        gc.CONFIG_JSON = out_dir / "candidate_generation_config.json"
        try:
            reactions = pd.read_csv(REACTIONS_CSV)
            evaluable = reactions[reactions.included_in_eval.astype(bool)]
            model_ids = sorted(evaluable.model_id.astype(str).unique())
            gc.assemble(model_ids, partial_ok=False)
            assembled = {
                name: sha256_file(out_dir / name) for name in committed
            }
        finally:
            for name, value in original.items():
                setattr(gc, name, value)

    after_committed = {
        "candidates.csv": sha256_file(CANDIDATES_CSV),
        "candidate_status.csv": sha256_file(STATUS_CSV),
        "candidate_generation_failures.csv": sha256_file(FAILURES_CSV),
    }
    mismatches = [name for name in committed if assembled.get(name) != committed[name]]
    return {
        "registry_problems": problems,
        "reassembly_identical": not mismatches,
        "mismatches": mismatches,
        "committed": committed,
        "assembled": assembled,
        "committed_artifacts_untouched": after_committed == committed,
        "n_files_restored": registry["n_files"],
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
    parser.add_argument("--verify-caches", action="store_true",
                        help="Check the working-copy caches against the committed registry")
    parser.add_argument("--pack-caches", action="store_true",
                        help="Write the gitignored cache zip and the committed registry")
    parser.add_argument("--verify-cache-archive", nargs="?", const="",
                        metavar="ZIP",
                        help="Restore the cache zip into a temp dir, verify files, and "
                             "reassemble; never writes into the working copy")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("freeze_phase2")

    if args.verify_caches:
        if not CACHE_REGISTRY.exists():
            logger.error("no cache registry at %s; run --pack-caches first",
                         CACHE_REGISTRY)
            return 1
        registry = json.loads(CACHE_REGISTRY.read_text(encoding="utf-8"))
        problems = verify_caches(registry)
        for p in problems:
            logger.error("MISMATCH %s", p)
        logger.info("verified %d caches against %s; %d problems",
                    registry["n_files"], CACHE_REGISTRY.name, len(problems))
        return 1 if problems else 0

    if args.verify_cache_archive is not None:
        archive = Path(args.verify_cache_archive) if args.verify_cache_archive else CACHE_ARCHIVE
        if not archive.exists():
            logger.error("no cache archive at %s; run --pack-caches first", archive)
            return 1
        if not CACHE_REGISTRY.exists():
            logger.error("no cache registry at %s", CACHE_REGISTRY)
            return 1
        registry = json.loads(CACHE_REGISTRY.read_text(encoding="utf-8"))
        result = restore_and_reassemble(archive, registry)
        for p in result["registry_problems"]:
            logger.error("REGISTRY %s", p)
        if result["reassembly_identical"] and not result["registry_problems"]:
            logger.info(
                "restored %d caches from %s; reassembly byte-identical; "
                "committed artifacts untouched=%s",
                result["n_files_restored"], archive.name,
                result["committed_artifacts_untouched"],
            )
            return 0 if result["committed_artifacts_untouched"] else 1
        logger.error("reassembly mismatches: %s", result["mismatches"])
        return 1

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

    if args.pack_caches:
        registry = build_cache_registry()
        write_json(registry, CACHE_REGISTRY)
        packed = pack_caches()
        logger.info("cache registry: %d files, %.1f MB uncompressed, %s",
                    registry["n_files"], registry["total_bytes"] / 1e6,
                    CACHE_REGISTRY.relative_to(REPO_ROOT))
        logger.info("packed %s (%d files, %.1f MB, sha256=%s, config_id=%s)",
                    packed["path"], packed["n_files"], packed["bytes"] / 1e6,
                    packed["sha256"], packed["config_id"])
        return 0

    # Default: rewrite the manifest. The registry is written only by --pack-caches so
    # a freeze without local caches cannot replace it with an empty one.
    manifest = build_manifest()
    write_json(manifest, OUT_JSON)

    logger.info("%s: %d artifacts, config_id=%s, generation=%s artifacts=%s",
                manifest["benchmark_version"], len(manifest["artifacts"]),
                manifest["config_id"],
                manifest["commits"]["candidate_generation"][:7],
                manifest["commits"]["analysis_and_artifacts"][:7])
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
