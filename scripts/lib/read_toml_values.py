#!/usr/bin/env python3
"""Print scalar TOML values, one per line, for the release shell helpers.

Usage: read_toml_values.py MANIFEST LABEL KEY [KEY ...]

Each KEY is a dotted path. Every key must exist and must resolve to a scalar; a
missing key or a non-scalar value exits 2 with a `Fix:` message rather than
printing a blank line, because a silently empty value would let a release script
continue with an unset version or tag.

This lives in its own file rather than a shell heredoc: the release-hygiene scan
blocks heredocs in release tooling, since a heredoc hides an entire second
language from review, lint, and syntax checking.
"""

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    print("Fix: python3 tomllib is required; use Python 3.11+.", file=sys.stderr)
    sys.exit(2)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(
            "Fix: read_toml_values.py requires MANIFEST, LABEL, and at least one key.",
            file=sys.stderr,
        )
        return 2
    path = Path(argv[0])
    label = argv[1]
    keys = argv[2:]
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except Exception as error:  # noqa: BLE001 - the message is the operator's fix
        print(f"Fix: failed to read {path}: {error}", file=sys.stderr)
        return 2

    for key in keys:
        current = data
        for part in key.split("."):
            if not isinstance(current, dict) or part not in current:
                print(
                    f"Fix: {path} is missing required {label} key {key}.",
                    file=sys.stderr,
                )
                return 2
            current = current[part]
        if isinstance(current, bool):
            print("true" if current else "false")
        elif isinstance(current, (str, int, float)):
            print(current)
        else:
            print(
                f"Fix: {path} {label} key {key} must be a scalar value.",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
