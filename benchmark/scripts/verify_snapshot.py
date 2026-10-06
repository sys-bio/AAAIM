"""Verify SBML inputs against the Phase 1 provenance registry.

Checks either the working copy in ``benchmark/models/`` or a restored release archive,
so a future checkout can prove it is using the exact bytes Phase 1 was built from even
if upstream BioModels files have since changed.

Usage::

    python benchmark/scripts/verify_snapshot.py
    python benchmark/scripts/verify_snapshot.py --archive benchmark/dist/....zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "benchmark" / "models"
REGISTRY = REPO_ROOT / "benchmark" / "manifest" / "model_registry.json"

logger = logging.getLogger("verify_snapshot")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


DIGEST_FIELDS = ("local_sha256", "sha256", "upstream_sha256")


def expected_digests() -> Dict[str, str]:
    """Map local filename -> expected sha256 from the provenance registry."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = registry.get("models", []) if isinstance(registry, dict) else registry
    out: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("model_id")
        if not model_id:
            continue
        digest = next((entry[f] for f in DIGEST_FIELDS if entry.get(f)), None)
        if not digest:
            continue
        local_path = entry.get("local_path") or f"benchmark/models/{model_id}.xml"
        out[Path(str(local_path)).name] = str(digest).lower()
    return out


def verify(archive: Optional[Path]) -> int:
    expected = expected_digests()
    if not expected:
        logger.error("no digests found in %s", REGISTRY)
        return 2

    actual: Dict[str, str] = {}
    if archive:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                actual[name] = sha256_bytes(zf.read(info))
        source = str(archive)
    else:
        for path in sorted(MODELS_DIR.glob("*.xml")):
            actual[path.name] = sha256_bytes(path.read_bytes())
        source = str(MODELS_DIR)

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(
        name for name in set(expected) & set(actual) if expected[name] != actual[name]
    )

    logger.info("source: %s", source)
    logger.info("expected %d files, found %d", len(expected), len(actual))
    for name in missing:
        logger.error("MISSING: %s", name)
    for name in extra:
        logger.warning("UNEXPECTED: %s", name)
    for name in mismatched:
        logger.error("DIGEST MISMATCH: %s expected=%s actual=%s",
                     name, expected[name], actual[name])

    if missing or mismatched:
        logger.error("verification FAILED")
        return 1
    logger.info("verification PASSED: all %d files match Phase 1 provenance", len(expected))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=None,
                        help="Verify a release zip instead of benchmark/models/")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return verify(args.archive)


if __name__ == "__main__":
    raise SystemExit(main())
