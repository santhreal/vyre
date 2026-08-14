#!/usr/bin/env bash
# Both halves of the internal-dependency version rule, for every publishable
# member of the workspace.
#
#   1. A publishable crate's normal or build dependency on a PUBLISHED member
#      must carry a version. Path-only breaks `cargo publish`: the published
#      crate cannot resolve its sibling from the registry.
#   2. A publishable crate's dependency on a `publish = false` member must NOT
#      carry a version, in any table including dev-dependencies. A version
#      requirement on an unpublishable crate is one no registry can ever
#      satisfy, so `cargo package` on the depender fails. Cargo strips
#      path-only dev-dependencies at package time, which is what makes the
#      path-only form the correct one here.
#
# The same two rules apply to `[workspace.dependencies]` entries that name a
# member, because `<crate>.workspace = true` inherits whatever that table says.
# That table is where the live defect sat: three `publish = false` members
# carried `version = "0.7.2"` and a published crate inherited one of them.
#
# Both crate sets are derived from the tracked manifests at run time. Earlier
# revisions hardcoded 13 publishable crates and a 15-name alternation of
# internal crates, so 13 publishable crates and 18 members were unchecked and
# adding either kind kept the gate green.
#
# Run before any `./cargo_full publish`. Wired into release signoff.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - "$ROOT" <<'PY'
import sys
import tomllib
from pathlib import Path
from subprocess import run

root = Path(sys.argv[1]).resolve()
NAME = "internal-deps-have-versions"
DEP_TABLES = ("dependencies", "build-dependencies", "dev-dependencies")


def tracked(*args: str) -> set[str]:
    done = run(["git", "ls-files", "-z", "--", *args], cwd=root, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"{NAME}: git ls-files failed: {done.stderr.strip()}")
    return {entry for entry in done.stdout.split("\0") if entry}


def read_manifest(rel: str) -> dict:
    try:
        return tomllib.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        sys.exit(f"{NAME}: {rel} is not readable as TOML: {exc}")


def dep_lines(rel: str) -> dict[tuple[str, str], int]:
    """Map (table header, dependency key) to its 1-based line number."""
    located: dict[tuple[str, str], int] = {}
    table = ""
    for number, line in enumerate((root / rel).read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("[") and not stripped.startswith("[["):
            table = stripped[1 : stripped.index("]")].strip()
            continue
        if "=" not in stripped or stripped.startswith("#"):
            continue
        key = stripped.split("=", 1)[0].strip().strip('"').split(".", 1)[0].strip('"')
        located.setdefault((table, key), number)
    return located


tracked_manifests = tracked("*Cargo.toml")
if "Cargo.toml" not in tracked_manifests:
    sys.exit(f"{NAME}: the root Cargo.toml is not tracked; the scan would pass vacuously")

workspace = read_manifest("Cargo.toml").get("workspace") or {}
ws_deps = workspace.get("dependencies") or {}
member_dirs = workspace.get("members") or []
if not member_dirs:
    sys.exit(f"{NAME}: [workspace.members] is empty; the scan would pass vacuously")

# Derived roster: package name -> (manifest path, publishable).
members: dict[str, tuple[str, bool]] = {}
for member in member_dirs:
    rel = f"{member}/Cargo.toml"
    if rel not in tracked_manifests:
        sys.exit(f"{NAME}: workspace member manifest {rel} is not tracked")
    package = read_manifest(rel).get("package") or {}
    name = package.get("name")
    if not name:
        sys.exit(f"{NAME}: {rel} has no [package].name")
    publish = package.get("publish", True)
    publishable = not (publish is False or (isinstance(publish, list) and not publish))
    members[name] = (rel, publishable)

publishable = sorted(name for name, (_, ok) in members.items() if ok)
unpublished = sorted(name for name, (_, ok) in members.items() if not ok)
if not publishable or not unpublished:
    sys.exit(
        f"{NAME}: derived {len(publishable)} publishable and {len(unpublished)} "
        "unpublished members; one half of the rule would scan nothing"
    )


def target_package(key: str, spec: object) -> str:
    """The real package a dependency key names, following `package =` renames."""
    if isinstance(spec, dict):
        if "package" in spec:
            return str(spec["package"])
        if spec.get("workspace") is True:
            inherited = ws_deps.get(key)
            if isinstance(inherited, dict) and "package" in inherited:
                return str(inherited["package"])
    return key


def has_version(key: str, spec: object) -> bool | None:
    """True/False, or None when `workspace = true` names no table entry."""
    if isinstance(spec, str):
        return True
    if not isinstance(spec, dict):
        return False
    if spec.get("workspace") is True:
        inherited = ws_deps.get(key)
        if inherited is None:
            return None
        if isinstance(inherited, str):
            return True
        return "version" in inherited
    return "version" in spec


problems: list[str] = []
edges = 0


def judge(rel: str, table: str, key: str, spec: object, lines: dict) -> None:
    global edges
    package = target_package(key, spec)
    if package not in members:
        return
    edges += 1
    where = f"{rel}:{lines.get((table, key), 0)}"
    named = key if package == key else f"{key} (package = {package})"
    versioned = has_version(key, spec)
    inherited = isinstance(spec, dict) and spec.get("workspace") is True
    source = "[workspace.dependencies]" if inherited else f"[{table}]"
    if versioned is None:
        problems.append(
            f"{where}: {named} sets `workspace = true` but [workspace.dependencies] "
            f"has no `{key}` entry\n"
            f"    Fix: add the entry, or declare the dependency locally."
        )
        return
    if not members[package][1]:
        if versioned:
            problems.append(
                f"{where}: {named} carries a version through {source}, but {package} is "
                f"`publish = false`\n"
                f"    Fix: make it path-only. No registry can satisfy a version "
                f"requirement on an unpublishable crate, so `cargo package` fails here."
            )
        return
    if not versioned and table != "dev-dependencies" and not table.endswith(".dev-dependencies"):
        problems.append(
            f"{where}: {named} is path-only through {source}, but {package} is published\n"
            f"    Fix: give it both `version` and `path`, or `{key}.workspace = true` "
            f"against a versioned table entry. Path-only blocks `cargo publish` from "
            f"resolving siblings on the registry."
        )


for name in publishable:
    rel = members[name][0]
    data = read_manifest(rel)
    lines = dep_lines(rel)
    for kind in DEP_TABLES:
        for key, spec in (data.get(kind) or {}).items():
            judge(rel, kind, key, spec, lines)
    for triple, table in (data.get("target") or {}).items():
        if not isinstance(table, dict):
            continue
        for kind in DEP_TABLES:
            for key, spec in (table.get(kind) or {}).items():
                judge(rel, f"target.{triple}.{kind}", key, spec, lines)

# The workspace table itself: an entry naming a member must obey the same rule,
# since every `workspace = true` site inherits it.
root_lines = dep_lines("Cargo.toml")
for key, spec in ws_deps.items():
    package = target_package(key, spec)
    if package not in members:
        continue
    edges += 1
    where = f"Cargo.toml:{root_lines.get(('workspace.dependencies', key), 0)}"
    named = key if package == key else f"{key} (package = {package})"
    versioned = has_version(key, spec)
    if members[package][1] and not versioned:
        problems.append(
            f"{where}: [workspace.dependencies] {named} is path-only, but {package} is "
            f"published\n"
            f"    Fix: add `version`. Every member writing `{key}.workspace = true` "
            f"inherits this entry and would publish unresolvable."
        )
    elif not members[package][1] and versioned:
        problems.append(
            f"{where}: [workspace.dependencies] {named} carries a version, but {package} "
            f"is `publish = false`\n"
            f"    Fix: make the entry path-only. Every member writing "
            f"`{key}.workspace = true` inherits a requirement no registry can satisfy."
        )

if problems:
    print(f"{NAME}: {len(problems)} violation(s) across {len(publishable)} publishable crate(s)")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)

print(
    f"{NAME}: {edges} internal dependency edge(s) from {len(publishable)} publishable "
    f"crate(s) obey both halves of the rule ({len(unpublished)} members are "
    f"`publish = false` and must stay path-only)."
)
PY
