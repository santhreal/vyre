#!/usr/bin/env python3
"""Bench corpus duplication checker.

Scans benches/competition/corpora/ for duplicate benchmark programs
(by wire-format hash) and reports any collisions. Used by the CI
release gate to ensure corpus quality.

Exit 0 if no duplicates; exit 1 if duplicates found.
"""

import hashlib
import sys
from pathlib import Path


def main() -> int:
    corpora_dir = Path(__file__).resolve().parent.parent / "corpora"
    if not corpora_dir.is_dir():
        print(f"corpora dir not found: {corpora_dir}", file=sys.stderr)
        return 0  # No corpora is not a failure.

    seen: dict[str, Path] = {}
    duplicates: list[tuple[Path, Path]] = []

    for path in sorted(corpora_dir.rglob("*.vir0")):
        digest = hashlib.blake2b(path.read_bytes(), digest_size=32).hexdigest()
        if digest in seen:
            duplicates.append((seen[digest], path))
        else:
            seen[digest] = path

    if duplicates:
        print("Duplicate bench corpus entries:", file=sys.stderr)
        for original, dup in duplicates:
            print(f"  {dup} duplicates {original}", file=sys.stderr)
        return 1

    print(f"OK: {len(seen)} unique corpus entries, 0 duplicates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
