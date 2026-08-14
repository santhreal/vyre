#!/usr/bin/env python3
"""Derive the `sweep_*` oracle-matrix test roster from the tree, for the sweep runners.

Usage: sweep_targets.py ROOT KIND

KIND selects a partition of the roster:

  matrix   every tracked `sweep_*` test whose name does not contain `volume`
  volume   every tracked `sweep_*_volume_*` test (the 16k-case waves)
  all      both, which is what the partition must add up to

Each line of output is `CRATE<TAB>TARGET<TAB>FEATURES`, where FEATURES is the
comma-joined `required-features` the crate's own `[[test]]` entry declares for
that target, or empty when it declares none.

This exists because both runners previously carried a hardcoded target list and a
hardcoded per-crate feature union. A list of test binaries in a shell array goes
stale the moment someone adds a sweep, and a stale roster is the same failure as
no runner at all: the new sweep is never executed and nothing says so. `ci.yml`
runs the workspace suite with default features only, so a feature-gated sweep is
skipped there too, and these runners are the only thing that executes it.

Refuses to print a roster it cannot vouch for, because a runner that silently
runs nothing reports success forever:

  * an empty partition
  * a `[[test]]` entry naming a sweep with no tracked source file
  * a `required-features` entry naming a feature the crate does not define
  * a sweep source file outside the declared workspace members
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

SWEEP_SOURCE = re.compile(r"^(?P<crate>[^/]+)/tests/(?P<target>sweep_[A-Za-z0-9_]*)\.rs$")


def fail(message: str) -> int:
    print(f"Fix: {message}", file=sys.stderr)
    return 2


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed in {root}: {result.stderr.strip()}")
    return [entry for entry in result.stdout.split("\0") if entry]


def workspace_members(root: Path) -> set[str]:
    with (root / "Cargo.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    members = manifest.get("workspace", {}).get("members", [])
    if not members:
        raise RuntimeError(
            "the root Cargo.toml declares no [workspace.members]; the sweep roster "
            "cannot be derived from an empty workspace"
        )
    return set(members)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        return fail("sweep_targets.py requires ROOT and KIND (matrix, volume, or all).")
    root = Path(argv[0])
    kind = argv[1]
    if kind not in ("matrix", "volume", "all"):
        return fail(f"unknown sweep kind {kind!r}; use matrix, volume, or all.")

    try:
        members = workspace_members(root)
        tracked = tracked_files(root)
    except (OSError, RuntimeError, tomllib.TOMLDecodeError) as error:
        return fail(str(error))

    sources: dict[str, set[str]] = {}
    for path in tracked:
        match = SWEEP_SOURCE.match(path)
        if not match:
            continue
        crate = match.group("crate")
        if crate not in members:
            return fail(
                f"{path} is a sweep test in {crate}, which is not a [workspace.members] "
                "entry, so no cargo invocation can reach it."
            )
        sources.setdefault(crate, set()).add(match.group("target"))

    if not sources:
        return fail(
            "no tracked <crate>/tests/sweep_*.rs files exist; the sweep runners would "
            "report success without executing anything."
        )

    roster: list[tuple[str, str, list[str]]] = []
    for crate in sorted(sources):
        manifest_path = root / crate / "Cargo.toml"
        try:
            with manifest_path.open("rb") as handle:
                manifest = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as error:
            return fail(f"failed to read {manifest_path}: {error}")
        defined = set(manifest.get("features", {}))
        declared: dict[str, list[str]] = {}
        for entry in manifest.get("test", []):
            name = entry.get("name")
            if not isinstance(name, str) or not name.startswith("sweep_"):
                continue
            declared[name] = list(entry.get("required-features", []))
        for name in sorted(declared):
            if name not in sources[crate]:
                return fail(
                    f"{crate}/Cargo.toml declares [[test]] {name} but no tracked "
                    f"{crate}/tests/{name}.rs exists; the entry reserves features for a "
                    "target that cannot build."
                )
        for target in sorted(sources[crate]):
            features = declared.get(target, [])
            unknown = [feature for feature in features if feature not in defined]
            if unknown:
                return fail(
                    f"{crate}/Cargo.toml gives [[test]] {target} required-features "
                    f"{unknown}, which {crate} does not define in [features]; cargo "
                    "would refuse the target."
                )
            roster.append((crate, target, features))

    selected = [
        row
        for row in roster
        if kind == "all" or (("volume" in row[1]) == (kind == "volume"))
    ]
    if not selected:
        return fail(
            f"the {kind} partition of the sweep roster is empty out of {len(roster)} "
            "tracked sweep target(s); the runner would execute nothing."
        )

    for crate, target, features in selected:
        print(f"{crate}\t{target}\t{','.join(features)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
