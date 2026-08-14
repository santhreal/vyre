#!/usr/bin/env bash
# Transitive dependency direction, checked against docs/CRATE_OWNERSHIP.toml.
#
# The registry declares, per crate, the internal crates it is allowed to depend
# on. This gate resolves the real dependency graph and holds every workspace
# member to the transitive closure of its declared edges, plus one external
# rule: a crate in a substrate-neutral layer must not reach a backend API crate
# at any depth.
#
# What nothing else catches: depth. scripts/check_architectural_invariants.sh
# and `xtask check-tier-deps` read manifest text, so they see direct edges only.
# A neutral crate that reaches a concrete backend through one intermediate is
# invisible to both and compiles clean.
#
# Two defects this replaces. The gate checked three hardcoded crates against
# hardcoded forbidden lists, so the other thirty members were unchecked and
# adding a member kept it green. And it ran `cargo tree ... 2>/dev/null` with
# a failure treated as "skipping missing workspace crate", so any cargo failure
# silently skipped every rule and the gate printed that all layers were green:
# `VYRE_CARGO_RUNNER=false scripts/check_layering.sh` exited 0.

set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/lib/cargo_runner.sh
vyre_select_cargo_runner

python3 - "$PWD" "$CARGO_RUNNER" <<'PY'
import sys
import tomllib
from pathlib import Path
from subprocess import run

root = Path(sys.argv[1]).resolve()
cargo = sys.argv[2]
NAME = "check-layering"

# Every layer name used by a workspace member needs a decision here. A member
# whose layer is missing, and a decision no member uses, are both fatal: the
# first is an unreviewed crate, the second an allowance nothing needs.
NEUTRAL_LAYERS = {
    "backend-neutral": True,
    "compiler-boundary": True,
    "conformance": False,
    "concrete-backend": False,
    "emitter": False,
    "facade": True,
    "foundation": True,
    "frontend": True,
    "libraries": True,
    "lowering": True,
    "packaging": True,
    "primitives": True,
    "runtime": True,
    "scheduler": True,
    "semantics": True,
    "test-tooling": False,
    "tooling": False,
}

# Backend API crates. A neutral crate reaching one of these has crossed the
# substrate boundary whatever the intermediate was.
BACKEND_APIS = ("ash", "cudarc", "metal", "naga", "wgpu")


def fatal(message: str) -> None:
    sys.exit(f"{NAME}: {message}")


def read_toml(rel: str) -> dict:
    try:
        return tomllib.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        fatal(f"{rel} is not readable as TOML: {exc}")


def cargo_tree(*args: str) -> list[str]:
    done = run([cargo, "tree", *args], cwd=root, capture_output=True, text=True)
    if done.returncode != 0:
        fatal(
            f"`{cargo} tree {' '.join(args)}` failed with status {done.returncode}. "
            f"A layer rule that cannot resolve the graph is unmeasured, not clean.\n"
            f"    {done.stderr.strip()}"
        )
    return [line.split()[0] for line in done.stdout.splitlines() if line.strip()]


workspace = read_toml("Cargo.toml").get("workspace") or {}
member_dirs = workspace.get("members") or []
if not member_dirs:
    fatal("[workspace.members] is empty; every rule would scan nothing")

members: dict[str, str] = {}
for member in member_dirs:
    package = read_toml(f"{member}/Cargo.toml").get("package") or {}
    name = package.get("name")
    if not name:
        fatal(f"{member}/Cargo.toml has no [package].name")
    members[name] = member

registry = read_toml("docs/CRATE_OWNERSHIP.toml").get("crate") or []
if not registry:
    fatal("docs/CRATE_OWNERSHIP.toml declares no [[crate]]; every rule would scan nothing")
declared: dict[str, set[str]] = {}
layers: dict[str, str] = {}
for entry in registry:
    package = entry.get("package")
    if not package:
        fatal("docs/CRATE_OWNERSHIP.toml has a [[crate]] with no package name")
    declared[package] = {dep["package"] for dep in entry.get("dependency") or []}
    layers[package] = entry.get("layer") or ""

unregistered = sorted(name for name in members if name not in declared)
if unregistered:
    fatal(
        "workspace member(s) with no docs/CRATE_OWNERSHIP.toml entry: "
        + ", ".join(unregistered)
        + "\n    Fix: record the crate's layer and allowed internal edges in the "
        "registry, then run `python3 scripts/crate_ownership.py --write`. An "
        "unregistered crate has no declared layer, so no layer rule constrains it."
    )

used_layers = {layers[name] for name in members}
missing_policy = sorted(layer for layer in used_layers if layer not in NEUTRAL_LAYERS)
if missing_policy:
    fatal(
        "layer(s) used by a workspace member with no neutrality decision in this "
        "gate: " + ", ".join(missing_policy) + "\n    Fix: add the layer to "
        "NEUTRAL_LAYERS as neutral (true) or substrate-bound (false)."
    )
stale_policy = sorted(layer for layer in NEUTRAL_LAYERS if layer not in used_layers)
if stale_policy:
    fatal(
        "NEUTRAL_LAYERS decision(s) that no workspace member uses: "
        + ", ".join(stale_policy)
        + "\n    Fix: delete the entry. A neutrality decision for a layer nobody "
        "is in constrains nothing."
    )

third_party = (workspace.get("dependencies") or {}).keys()
absent = [api for api in BACKEND_APIS if api not in third_party]
if absent:
    fatal(
        "BACKEND_APIS name(s) absent from [workspace.dependencies]: "
        + ", ".join(absent)
        + "\n    Fix: update the list to the crate the workspace actually pins. A "
        "name no manifest uses can never match, so the rule it carries is dead."
    )


def closure(package: str) -> set[str]:
    reachable: set[str] = set()
    pending = list(declared.get(package, ()))
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(declared.get(current, ()))
    return reachable


def why(member: str, package: str) -> str:
    done = run(
        [cargo, "tree", "-p", member, "--edges=normal", "--invert", package],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        return f"      (cargo tree --invert {package} failed: {done.stderr.strip()})"
    lines = done.stdout.strip().splitlines()[:8]
    return "\n".join(f"      {line}" for line in lines)


problems: list[str] = []
edges = 0
for name in sorted(members):
    tree = set(cargo_tree("-p", name, "--edges=normal", "--prefix=none"))
    internal = (tree & set(members)) - {name}
    edges += len(internal)
    for package in sorted(internal - closure(name)):
        problems.append(
            f"LAYER VIOLATION: {name} reaches {package}, which its "
            f"docs/CRATE_OWNERSHIP.toml entry does not allow, directly or through "
            f"a declared edge\n" + why(name, package)
        )
    if NEUTRAL_LAYERS[layers[name]]:
        for api in sorted(tree.intersection(BACKEND_APIS)):
            problems.append(
                f"LAYER VIOLATION: {name} is in the substrate-neutral layer "
                f"`{layers[name]}` and reaches the backend API crate {api}\n"
                + why(name, api)
            )

if problems:
    for problem in problems:
        print(problem, file=sys.stderr)
    print(
        f"\n{NAME}: {len(problems)} layer violation(s). Cross-layer edges go DOWN "
        f"only; see docs/CRATE_OWNERSHIP.toml.",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"{NAME}: {edges} resolved internal edge(s) across {len(members)} workspace "
    f"member(s) stay inside their declared closure, and no crate in the "
    f"{sum(1 for name in members if NEUTRAL_LAYERS[layers[name]])} neutral-layer "
    f"crates reaches {', '.join(BACKEND_APIS)}."
)
PY
