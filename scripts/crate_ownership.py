#!/usr/bin/env python3
"""Generate and validate Vyre workspace ownership and dependency documentation."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MAX_TOML_BYTES = 1_048_576
REGISTRY_PATH = Path("docs/CRATE_OWNERSHIP.toml")
GRAPH_PATH = Path("docs/CRATE_GRAPH.md")
OWNERSHIP_PATH = Path("docs/OWNERSHIP.md")


class ContractError(Exception):
    """A workspace ownership contract is incomplete or stale."""


@dataclass(frozen=True)
class CrateRecord:
    package: str
    path: str
    owner: str
    layer: str
    responsibility: str
    allowed_dependencies: tuple[str, ...]


@dataclass(frozen=True)
class PlannedRecord:
    package: str
    path: str
    owner: str
    layer: str
    responsibility: str
    allowed_dependencies: tuple[str, ...]


def read_toml(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ContractError(f"could not inspect `{path}`: {error}") from error
    if size > MAX_TOML_BYTES:
        raise ContractError(
            f"`{path}` is {size} bytes, above the {MAX_TOML_BYTES}-byte TOML limit"
        )
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ContractError(f"could not read TOML `{path}`: {error}") from error


def require_text(row: dict[str, Any], field: str, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{context} must define non-empty `{field}`")
    return value.strip()


def require_dependencies(
    row: dict[str, Any], field: str, context: str
) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ContractError(f"{context} must define string array `{field}`")
    normalized = tuple(sorted(item.strip() for item in value))
    if len(normalized) != len(set(normalized)):
        raise ContractError(f"{context} `{field}` contains duplicate packages")
    return normalized


def load_registry(root: Path) -> tuple[list[CrateRecord], list[PlannedRecord]]:
    registry = read_toml(root / REGISTRY_PATH)
    if registry.get("schema_version") != 1:
        raise ContractError(f"`{REGISTRY_PATH}` must declare schema_version = 1")

    rows = registry.get("crate")
    if not isinstance(rows, list):
        raise ContractError(f"`{REGISTRY_PATH}` must define one [[crate]] row per member")
    records: list[CrateRecord] = []
    for index, row in enumerate(rows):
        context = f"{REGISTRY_PATH} [[crate]] row {index + 1}"
        if not isinstance(row, dict):
            raise ContractError(f"{context} must be a table")
        records.append(
            CrateRecord(
                package=require_text(row, "package", context),
                path=require_text(row, "path", context),
                owner=require_text(row, "owner", context),
                layer=require_text(row, "layer", context),
                responsibility=require_text(row, "responsibility", context),
                allowed_dependencies=require_dependencies(
                    row, "allowed_dependencies", context
                ),
            )
        )

    planned_table = registry.get("planned", {})
    if not isinstance(planned_table, dict):
        raise ContractError(f"`{REGISTRY_PATH}` [planned] value must be a table")
    planned: list[PlannedRecord] = []
    for package, row in sorted(planned_table.items()):
        context = f"{REGISTRY_PATH} [planned.{package}]"
        if not isinstance(row, dict):
            raise ContractError(f"{context} must be a table")
        if row.get("present") is not False:
            raise ContractError(
                f"{context} must declare present = false until it is a workspace member"
            )
        planned.append(
            PlannedRecord(
                package=package,
                path=require_text(row, "path", context),
                owner=require_text(row, "owner", context),
                layer=require_text(row, "layer", context),
                responsibility=require_text(row, "responsibility", context),
                allowed_dependencies=require_dependencies(
                    row, "allowed_dependencies", context
                ),
            )
        )
    return records, planned


def dependency_tables(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for name in ("dependencies", "build-dependencies"):
        table = manifest.get(name, {})
        if isinstance(table, dict):
            yield table
    targets = manifest.get("target", {})
    if isinstance(targets, dict):
        for target in targets.values():
            if not isinstance(target, dict):
                continue
            for name in ("dependencies", "build-dependencies"):
                table = target.get(name, {})
                if isinstance(table, dict):
                    yield table


def dependency_package(
    alias: str, specification: Any, workspace_dependencies: dict[str, Any]
) -> str:
    if isinstance(specification, dict):
        package = specification.get("package")
        if isinstance(package, str):
            return package
        if specification.get("workspace") is True:
            workspace_specification = workspace_dependencies.get(alias)
            if isinstance(workspace_specification, dict):
                workspace_package = workspace_specification.get("package")
                if isinstance(workspace_package, str):
                    return workspace_package
    return alias


def workspace_state(
    root: Path,
) -> tuple[list[str], dict[str, tuple[str, dict[str, Any]]], dict[str, tuple[str, ...]]]:
    workspace = read_toml(root / "Cargo.toml")
    workspace_table = workspace.get("workspace")
    if not isinstance(workspace_table, dict):
        raise ContractError("workspace Cargo.toml has no [workspace] table")
    members = workspace_table.get("members")
    if not isinstance(members, list) or not all(isinstance(item, str) for item in members):
        raise ContractError("workspace.members must be a string array")
    if len(members) != len(set(members)):
        raise ContractError("workspace.members contains duplicate paths")
    workspace_dependencies = workspace_table.get("dependencies", {})
    if not isinstance(workspace_dependencies, dict):
        raise ContractError("workspace.dependencies must be a table")

    manifests: dict[str, tuple[str, dict[str, Any]]] = {}
    for member in members:
        manifest = read_toml(root / member / "Cargo.toml")
        package = manifest.get("package")
        if not isinstance(package, dict):
            raise ContractError(f"`{member}/Cargo.toml` has no [package] table")
        name = package.get("name")
        if not isinstance(name, str) or not name:
            raise ContractError(f"`{member}/Cargo.toml` has no package.name")
        if name in manifests:
            raise ContractError(f"workspace contains duplicate package `{name}`")
        manifests[name] = (member, manifest)

    package_names = set(manifests)
    edges: dict[str, tuple[str, ...]] = {}
    for package, (_, manifest) in manifests.items():
        dependencies = {
            dependency_package(alias, specification, workspace_dependencies)
            for table in dependency_tables(manifest)
            for alias, specification in table.items()
        }
        edges[package] = tuple(sorted(dependencies & package_names))
    return list(members), manifests, edges


def validate(
    members: list[str],
    manifests: dict[str, tuple[str, dict[str, Any]]],
    edges: dict[str, tuple[str, ...]],
    records: list[CrateRecord],
    planned: list[PlannedRecord],
) -> None:
    by_package: dict[str, CrateRecord] = {}
    by_path: dict[str, CrateRecord] = {}
    for record in records:
        if record.package in by_package:
            raise ContractError(f"ownership registry duplicates package `{record.package}`")
        if record.path in by_path:
            raise ContractError(f"ownership registry duplicates path `{record.path}`")
        by_package[record.package] = record
        by_path[record.path] = record

    member_set = set(members)
    registry_paths = set(by_path)
    if member_set != registry_paths:
        missing = sorted(member_set - registry_paths)
        extra = sorted(registry_paths - member_set)
        raise ContractError(
            "ownership registry path set differs from workspace.members: "
            f"missing={missing}, extra={extra}"
        )
    if set(manifests) != set(by_package):
        missing = sorted(set(manifests) - set(by_package))
        extra = sorted(set(by_package) - set(manifests))
        raise ContractError(
            "ownership registry package set differs from workspace packages: "
            f"missing={missing}, extra={extra}"
        )

    for package, (path, _) in manifests.items():
        record = by_package[package]
        if record.path != path:
            raise ContractError(
                f"package `{package}` registry path `{record.path}` does not match `{path}`"
            )
        actual = edges[package]
        if actual != record.allowed_dependencies:
            undeclared = sorted(set(actual) - set(record.allowed_dependencies))
            stale = sorted(set(record.allowed_dependencies) - set(actual))
            raise ContractError(
                f"package `{package}` production dependency contract differs from manifests: "
                f"undeclared={undeclared}, stale={stale}"
            )

    planned_packages = {record.package for record in planned}
    if planned_packages & set(manifests):
        overlap = sorted(planned_packages & set(manifests))
        raise ContractError(
            f"planned ownership entries already exist as workspace packages: {overlap}"
        )
    planned_paths = {record.path for record in planned}
    if planned_paths & member_set:
        overlap = sorted(planned_paths & member_set)
        raise ContractError(
            f"planned ownership paths already exist in workspace.members: {overlap}"
        )
    for record in planned:
        unknown = sorted(set(record.allowed_dependencies) - set(manifests))
        if unknown:
            raise ContractError(
                f"planned package `{record.package}` names unknown dependencies: {unknown}"
            )


def mermaid_id(index: int) -> str:
    return f"C{index}"


def render_graph(
    records: list[CrateRecord], planned: list[PlannedRecord]
) -> str:
    ordered = sorted(records, key=lambda record: record.package)
    ids = {record.package: mermaid_id(index) for index, record in enumerate(ordered)}
    lines = [
        "# Vyre Crate Graph",
        "",
        "This file is generated by `python3 scripts/crate_ownership.py --write` from",
        "the workspace manifests and `docs/CRATE_OWNERSHIP.toml`. Edit the registry or",
        "a manifest, then regenerate this file. `check-tier-deps` rejects drift.",
        "",
        "## Current workspace",
        "",
        f"The workspace contains {len(ordered)} crates. An arrow points from a crate to",
        "an internal production dependency. Development dependencies are excluded because",
        "they do not define the shipped dependency DAG.",
        "",
        "```mermaid",
        "graph TD",
    ]
    for record in ordered:
        lines.append(f'  {ids[record.package]}["{record.package}"]')
    for record in ordered:
        for dependency in record.allowed_dependencies:
            lines.append(f"  {ids[record.package]} --> {ids[dependency]}")
    lines.extend(["```", "", "## Current ownership and edges", ""])
    lines.append("| Crate | Path | Owner | Layer | Internal production dependencies |")
    lines.append("| --- | --- | --- | --- | --- |")
    for record in ordered:
        dependencies = ", ".join(f"`{item}`" for item in record.allowed_dependencies)
        lines.append(
            f"| `{record.package}` | `{record.path}` | `{record.owner}` | "
            f"`{record.layer}` | {dependencies or "None"} |"
        )

    lines.extend(
        [
            "",
            "## Planned compiler boundary",
            "",
            "The entries in this section are plans, not workspace members. The generator",
            "fails if a planned entry is presented as current before its manifest exists.",
            "",
        ]
    )
    for record in planned:
        dependencies = ", ".join(f"`{item}`" for item in record.allowed_dependencies)
        lines.extend(
            [
                f"### `{record.package}` (planned, not a workspace member)",
                "",
                record.responsibility,
                "",
                f"- Intended path: `{record.path}`",
                f"- Owner: `{record.owner}`",
                f"- Layer: `{record.layer}`",
                f"- Intended dependencies: {dependencies or 'None'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Cross-crate promotion patch contract",
            "",
            "When you change a production dependency, update the manifest and",
            "`docs/CRATE_OWNERSHIP.toml` in the same patch. Regenerate both ownership",
            "documents. Add an import-path migration test when a public path moves.",
            "`check-tier-deps` rejects undeclared edges, and `lego-audit` rejects",
            "cross-tier composition that bypasses the canonical owner.",
            "",
        ]
    )
    return "\n".join(lines)


def render_ownership(
    records: list[CrateRecord], planned: list[PlannedRecord]
) -> str:
    ordered = sorted(records, key=lambda record: record.package)
    lines = [
        "# Vyre Crate Ownership",
        "",
        "This file is generated by `python3 scripts/crate_ownership.py --write` from",
        "`docs/CRATE_OWNERSHIP.toml` and the workspace manifests. The registry is the",
        "single source for each crate's owner, responsibility, layer, path, and allowed",
        "internal production dependencies.",
        "",
        "## Boundary rule",
        "",
        "A crate may use only the internal production dependencies listed below. Any",
        "other normal or build dependency is rejected by `check-tier-deps`. Development",
        "dependencies are test wiring and do not expand the shipped boundary.",
        "",
        "Concrete backend APIs stay in their owning driver or emitter crate. Shared",
        "foundation, runtime, library, primitive, and conformance code uses neutral",
        "backend contracts. If shared code needs a capability, add the neutral contract",
        "to its canonical owner before implementing it in a concrete backend.",
        "",
        "## Per-crate ownership",
        "",
    ]
    for record in ordered:
        dependencies = ", ".join(f"`{item}`" for item in record.allowed_dependencies)
        lines.extend(
            [
                f"### `{record.package}`",
                "",
                record.responsibility,
                "",
                f"- Path: `{record.path}`",
                f"- Owner: `{record.owner}`",
                f"- Layer: `{record.layer}`",
                f"- Allowed internal production dependencies: {dependencies or 'None'}",
                "",
            ]
        )

    lines.extend(["## Planned ownership", ""])
    for record in planned:
        dependencies = ", ".join(f"`{item}`" for item in record.allowed_dependencies)
        lines.extend(
            [
                f"### `{record.package}` (planned, not a workspace member)",
                "",
                record.responsibility,
                "",
                f"- Intended path: `{record.path}`",
                f"- Owner: `{record.owner}`",
                f"- Layer: `{record.layer}`",
                f"- Intended dependencies: {dependencies or 'None'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Changing a boundary",
            "",
            "1. Change the manifest and `docs/CRATE_OWNERSHIP.toml` together.",
            "2. Run `python3 scripts/crate_ownership.py --write`.",
            "3. Add an import-path migration test for a public move.",
            "4. Run `cargo_full run --bin xtask -- check-tier-deps` and `lego-audit`.",
            "",
        ]
    )
    return "\n".join(lines)


def check_or_write(path: Path, expected: str, write: bool) -> None:
    if write:
        path.write_text(expected, encoding="utf-8")
        return
    try:
        actual = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ContractError(f"could not read generated document `{path}`: {error}") from error
    if actual != expected:
        raise ContractError(
            f"generated document `{path}` is stale; run "
            "`python3 scripts/crate_ownership.py --write`"
        )


def run(root: Path, write: bool) -> None:
    root = root.resolve()
    records, planned = load_registry(root)
    members, manifests, edges = workspace_state(root)
    validate(members, manifests, edges, records, planned)
    check_or_write(root / GRAPH_PATH, render_graph(records, planned), write)
    check_or_write(root / OWNERSHIP_PATH, render_ownership(records, planned), write)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="workspace root, defaults to the repository containing this script",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="reject registry or document drift")
    mode.add_argument("--write", action="store_true", help="regenerate the two Markdown documents")
    args = parser.parse_args()
    try:
        run(args.root, args.write)
    except ContractError as error:
        print(f"crate-ownership: {error}", file=sys.stderr)
        return 1
    action = "wrote" if args.write else "verified"
    print(f"crate-ownership: {action} {GRAPH_PATH} and {OWNERSHIP_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
