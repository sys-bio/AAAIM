from __future__ import annotations

import argparse
from pathlib import Path


SEARCH_TERM = b"kegg.reaction"


def file_contains_kegg(path: Path) -> bool:
    try:
        # Read as bytes to avoid decode issues; search ASCII substring case-insensitively.
        data = path.read_bytes()
    except OSError:
        return False
    return SEARCH_TERM in data.lower()


def iter_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find files containing a search term (case-insensitive) under a directory."
    )
    parser.add_argument(
        "--root",
        default="tests/BioModels_251106",
        help="Directory to scan (default: tests/BioModels_251106)",
    )
    parser.add_argument(
        "--out",
        default="kegg_files.txt",
        help="Output file listing matches (default: kegg_files.txt)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root directory not found or not a directory: {root}")

    matches: list[Path] = []
    for p in iter_files(root):
        if file_contains_kegg(p):
            matches.append(p)

    out_path = Path(args.out)
    out_path.write_text(
        "\n".join(str(p.as_posix()) for p in sorted(matches)) + ("\n" if matches else ""),
        encoding="utf-8",
    )

    print(f"Scanned: {root}")
    print(f"Matches: {len(matches)}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

