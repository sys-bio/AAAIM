#!/usr/bin/env python3
"""Download and pin BioModels SBML files for the reaction-annotation benchmark.

Reads accession IDs from ``benchmark/manifest/models.txt``, resolves each
model's *main* SBML file via the BioModels files API, downloads it, verifies the
bytes against the upstream SHA-256, and writes a provenance registry to
``benchmark/manifest/model_registry.json``.

Provenance recorded per model: accession, upstream filename, download URL,
upstream SHA-256/MD5/size, locally computed SHA-256/size, checksum agreement,
SBML format version, curation status, submission ID, publication ID, and the
latest upstream revision.

The main filename is *not* predictable from the accession (e.g.
``BIOMD0000000013_url.xml`` vs ``iJN1463.xml``), so it must be resolved from the
API rather than guessed.

Run from the repo root::

    python benchmark/scripts/download_biomodels.py
    python benchmark/scripts/download_biomodels.py --limit 3   # smoke test
    python benchmark/scripts/download_biomodels.py --force     # ignore cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "benchmark" / "manifest" / "models.txt"
DEFAULT_MODELS_DIR = REPO_ROOT / "benchmark" / "models"
DEFAULT_REGISTRY = REPO_ROOT / "benchmark" / "manifest" / "model_registry.json"

API_BASE = "https://www.ebi.ac.uk/biomodels"
FILES_URL = API_BASE + "/model/files/{model_id}?format=json"
METADATA_URL = API_BASE + "/{model_id}?format=json"
DOWNLOAD_URL = API_BASE + "/model/download/{model_id}?filename={filename}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("download_biomodels")


def load_model_ids(manifest_path: Path) -> List[str]:
    ids: List[str] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            ids.append(line)
    return ids


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def portable_path(path: Path) -> str:
    """Use a repository-relative path when possible, otherwise an absolute path."""
    resolved = path.resolve()
    try:
        resolved = resolved.relative_to(REPO_ROOT)
    except ValueError:
        pass
    return str(resolved).replace("\\", "/")


def pick_main_sbml(main_files: List[Dict[str, object]], model_id: str) -> Optional[Dict[str, object]]:
    """Choose the primary SBML file from the API's ``main`` file list.

    Prefers the conventional ``<accession>_url.xml``, then any XML file. Models
    imported from external reconstructions use their original filename.
    """
    if not main_files:
        return None
    preferred = f"{model_id}_url.xml"
    for entry in main_files:
        if str(entry.get("name", "")) == preferred:
            return entry
    xml_files = [
        e
        for e in main_files
        if str(e.get("name", "")).lower().endswith(".xml")
        or str(e.get("mimeType", "")) == "application/xml"
    ]
    if xml_files:
        # Largest XML is the model itself when auxiliary XML files are present.
        return max(xml_files, key=lambda e: int(str(e.get("fileSize", "0")) or 0))
    return None


def fetch_metadata(model_id: str, session: requests.Session) -> Dict[str, object]:
    url = METADATA_URL.format(model_id=model_id)
    out: Dict[str, object] = {"metadata_url": url}
    try:
        resp = session.get(url, timeout=90)
        if resp.status_code != 200:
            out["metadata_error"] = f"HTTP {resp.status_code}"
            return out
        data = resp.json()
        fmt = data.get("format", {}) if isinstance(data, dict) else {}
        out["model_name"] = data.get("name")
        out["sbml_format"] = fmt.get("identifier")
        out["sbml_format_version"] = fmt.get("version")
        out["curation_status"] = data.get("curationStatus")
        out["submission_id"] = data.get("submissionId")
        out["publication_id"] = data.get("publicationId")
        out["first_published"] = data.get("firstPublished")
        revisions = ((data.get("history") or {}).get("revisions") or [])
        if revisions:
            latest = revisions[-1]
            out["latest_revision"] = latest.get("version")
            out["latest_revision_submitted"] = latest.get("submitted")
            out["num_revisions"] = len(revisions)
    except Exception as exc:
        out["metadata_error"] = str(exc)
    return out


def download_model(
    model_id: str,
    models_dir: Path,
    session: requests.Session,
    *,
    force: bool = False,
) -> Dict[str, object]:
    entry: Dict[str, object] = {
        "model_id": model_id,
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }

    files_url = FILES_URL.format(model_id=model_id)
    entry["files_url"] = files_url
    try:
        resp = session.get(files_url, timeout=90)
        resp.raise_for_status()
        files_payload = resp.json()
    except Exception as exc:
        entry["status"] = "failed"
        entry["error"] = f"files API: {exc}"
        logger.error("%s: files API failed: %s", model_id, exc)
        return entry

    main_files = files_payload.get("main") or []
    entry["main_files_available"] = [str(f.get("name", "")) for f in main_files]
    chosen = pick_main_sbml(main_files, model_id)
    if chosen is None:
        entry["status"] = "failed"
        entry["error"] = "no SBML file in 'main' file list"
        logger.error("%s: no main SBML file", model_id)
        return entry

    upstream_name = str(chosen.get("name", ""))
    entry["upstream_filename"] = upstream_name
    entry["upstream_sha256"] = chosen.get("sha256sum")
    entry["upstream_md5"] = chosen.get("md5sum")
    entry["upstream_bytes"] = int(str(chosen.get("fileSize", "0")) or 0)
    entry["upstream_description"] = chosen.get("description")

    download_url = DOWNLOAD_URL.format(model_id=model_id, filename=upstream_name)
    entry["download_url"] = download_url

    dest = models_dir / f"{model_id}.xml"
    entry["local_path"] = portable_path(dest)

    entry.update(fetch_metadata(model_id, session))

    if dest.exists() and not force:
        local_hash = sha256_file(dest)
        if local_hash == entry.get("upstream_sha256"):
            entry["local_sha256"] = local_hash
            entry["local_bytes"] = dest.stat().st_size
            entry["checksum_verified"] = True
            entry["status"] = "cached"
            entry["downloaded_at"] = None
            logger.info("%s cached and checksum-verified", model_id)
            return entry
        logger.warning(
            "%s cached file does not match upstream checksum; re-downloading", model_id
        )

    try:
        resp = session.get(download_url, timeout=300)
        resp.raise_for_status()
        data = resp.content
        local_hash = sha256_bytes(data)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        entry["local_sha256"] = local_hash
        entry["local_bytes"] = len(data)
        entry["downloaded_at"] = datetime.now(timezone.utc).isoformat()
        entry["final_url"] = resp.url
        upstream_hash = entry.get("upstream_sha256")
        entry["checksum_verified"] = bool(upstream_hash) and local_hash == upstream_hash
        entry["status"] = "downloaded" if entry["checksum_verified"] else "checksum_mismatch"
        if entry["checksum_verified"]:
            logger.info("%s downloaded (%d bytes, verified)", model_id, len(data))
        else:
            logger.error(
                "%s checksum mismatch: upstream=%s local=%s",
                model_id,
                upstream_hash,
                local_hash,
            )
    except Exception as exc:
        entry["status"] = "failed"
        entry["error"] = str(exc)
        logger.error("%s download failed: %s", model_id, exc)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--limit", type=int, default=None, help="Only first N models.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    parser.add_argument("--sleep", type=float, default=0.3, help="Seconds between models.")
    args = parser.parse_args()

    model_ids = load_model_ids(args.manifest)
    duplicates = sorted({m for m in model_ids if model_ids.count(m) > 1})
    unique_ids = sorted(set(model_ids))
    if duplicates:
        logger.error("Manifest contains duplicate accessions: %s", duplicates)

    if args.limit is not None:
        model_ids = model_ids[: args.limit]
    logger.info("Processing %d models (%d unique in manifest)", len(model_ids), len(unique_ids))

    session = requests.Session()
    session.headers["User-Agent"] = "AAAIM-benchmark/1.0 (research reproducibility)"

    args.models_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, object]] = []
    for i, model_id in enumerate(model_ids, start=1):
        logger.info("[%d/%d] %s", i, len(model_ids), model_id)
        entries.append(download_model(model_id, args.models_dir, session, force=args.force))
        if i < len(model_ids) and args.sleep > 0:
            time.sleep(args.sleep)

    n_ok = sum(1 for e in entries if e.get("status") in ("downloaded", "cached"))
    n_mismatch = sum(1 for e in entries if e.get("status") == "checksum_mismatch")
    n_failed = sum(1 for e in entries if e.get("status") == "failed")

    registry = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_base": API_BASE,
        "manifest": portable_path(args.manifest),
        "models_dir": portable_path(args.models_dir),
        "manifest_entries": len(load_model_ids(args.manifest)),
        "manifest_unique_accessions": len(unique_ids),
        "manifest_duplicate_accessions": duplicates,
        "summary": {
            "processed": len(entries),
            "ok": n_ok,
            "checksum_mismatch": n_mismatch,
            "failed": n_failed,
        },
        "models": entries,
    }

    args.registry.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" so the registry bytes (and thus its digest) are identical on
    # every platform and match the LF-normalised blob stored by git.
    with args.registry.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(registry, indent=2) + "\n")
    logger.info(
        "Registry written to %s (ok=%d, mismatch=%d, failed=%d)",
        args.registry,
        n_ok,
        n_mismatch,
        n_failed,
    )
    return 1 if (n_failed or n_mismatch) else 0


if __name__ == "__main__":
    raise SystemExit(main())
