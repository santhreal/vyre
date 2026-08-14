#!/usr/bin/env bash
# Every path dependency and workspace member in a tracked Cargo.toml must point
# at a tracked Cargo.toml inside this repository.
#
# The class this closes: 5826591fad deleted the vyre-intrinsics/ tree and left
# vyre-libs/Cargo.toml depending on it. Every cargo command failed from a clean
# checkout for the next several commits, and nothing said so, because the working
# tree carried the fix uncommitted the whole time.
#
# This gate deliberately does NOT invoke cargo. The failure it detects is
# precisely a workspace cargo cannot load, so a cargo-based test cannot run when
# the defect is present. It reads tracked manifests with git and tomllib only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - "$ROOT" <<'PY'
import sys
import tomllib
from pathlib import Path
from subprocess import run

root = Path(sys.argv[1]).resolve()


def tracked(*args: str) -> list[str]:
    done = run(["git", "ls-files", "-z", "--", *args], cwd=root, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"check-path-deps-resolve: git ls-files failed: {done.stderr.strip()}")
    return [entry for entry in done.stdout.split("\0") if entry]


manifests = tracked("*Cargo.toml")
if not manifests:
    sys.exit("check-path-deps-resolve: no tracked Cargo.toml found; the scan would pass vacuously")

tracked_manifests = {root / entry for entry in manifests}
DEP_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")
problems: list[str] = []
edges = 0

_root_workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8")).get(
    "workspace", {}
)
WORKSPACE_DEPS = set(_root_workspace.get("dependencies") or {})
WORKSPACE_PACKAGE = set(_root_workspace.get("package") or {})
if not WORKSPACE_DEPS:
    sys.exit(
        "check-path-deps-resolve: root workspace.dependencies is empty; "
        "the inheritance scan would pass vacuously"
    )


def check_inherited(manifest: Path, label: str, name: str, table: set[str], origin: str) -> None:
    """Record a problem unless `workspace = true` has something to inherit from."""
    global edges
    edges += 1
    if name not in table:
        problems.append(
            f"`{manifest.relative_to(root)}` {label} declares `workspace = true` but "
            f"`{origin}.{name}` does not exist\n"
            f"    Fix: add the entry to the root manifest or drop the inheritance. "
            f"cargo cannot load the workspace at all while this dangles."
        )


def check(manifest: Path, label: str, raw: str) -> None:
    """Record a problem unless raw resolves to a tracked Cargo.toml under root."""
    global edges
    edges += 1
    where = manifest.relative_to(root)
    try:
        resolved = (manifest.parent / raw).resolve()
    except OSError as exc:
        problems.append(f"`{where}` {label} path `{raw}` is unresolvable: {exc}")
        return
    if root not in resolved.parents and resolved != root:
        problems.append(
            f"`{where}` {label} path `{raw}` escapes the repository "
            f"(resolves to `{resolved}`)\n"
            f"    Fix: express the path relative to the manifest, inside this repository."
        )
        return
    target = resolved / "Cargo.toml"
    if target not in tracked_manifests:
        state = "exists but is untracked" if target.is_file() else "does not exist"
        problems.append(
            f"`{where}` {label} path `{raw}` names `{target.relative_to(root)}`, which {state}\n"
            f"    Fix: delete the entry, or restore the member it names. A manifest that "
            f"names a missing member makes every cargo command fail from a clean checkout."
        )


def walk_dep_tables(manifest: Path, table: dict, prefix: str) -> None:
    for kind in DEP_TABLES:
        for name, spec in (table.get(kind) or {}).items():
            if not isinstance(spec, dict):
                continue
            if "path" in spec:
                check(manifest, f"{prefix}{kind}.{name}", spec["path"])
            if spec.get("workspace") is True:
                check_inherited(
                    manifest,
                    f"{prefix}{kind}.{name}",
                    name,
                    WORKSPACE_DEPS,
                    "workspace.dependencies",
                )


for entry in manifests:
    manifest = root / entry
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        problems.append(f"`{entry}` is not readable as TOML: {exc}")
        continue

    walk_dep_tables(manifest, data, "")
    for triple, spec in (data.get("target") or {}).items():
        if isinstance(spec, dict):
            walk_dep_tables(manifest, spec, f"target.{triple}.")

    for field, spec in (data.get("package") or {}).items():
        if isinstance(spec, dict) and spec.get("workspace") is True:
            check_inherited(
                manifest, f"package.{field}", field, WORKSPACE_PACKAGE, "workspace.package"
            )

    workspace = data.get("workspace") or {}
    walk_dep_tables(manifest, workspace, "workspace.")
    for name, spec in (workspace.get("dependencies") or {}).items():
        if isinstance(spec, dict) and "path" in spec:
            check(manifest, f"workspace.dependencies.{name}", spec["path"])
    for member in workspace.get("members") or []:
        if any(ch in member for ch in "*?["):
            if not list(manifest.parent.glob(f"{member}/Cargo.toml")):
                problems.append(
                    f"`{entry}` workspace.members pattern `{member}` matches no Cargo.toml\n"
                    f"    Fix: delete the pattern or restore the members it was written for."
                )
            edges += 1
            continue
        check(manifest, "workspace.members", member)

    for registry, spec in (data.get("patch") or {}).items():
        if isinstance(spec, dict):
            for name, entry_spec in spec.items():
                if isinstance(entry_spec, dict) and "path" in entry_spec:
                    check(manifest, f"patch.{registry}.{name}", entry_spec["path"])

if problems:
    print(f"check-path-deps-resolve: {len(problems)} unresolvable manifest edge(s)")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)

print(
    f"check-path-deps-resolve: {edges} path edge(s) across "
    f"{len(manifests)} tracked manifest(s) all resolve to tracked members."
)
PY
