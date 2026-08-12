#!/usr/bin/env python3
"""Generate and validate Vyre workspace ownership and dependency documentation."""

from __future__ import annotations

import argparse
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
class DependencyRecord:
    package: str
    purpose: str
    features: tuple[str, ...]
    conditions: tuple[str, ...]
    kinds: tuple[str, ...]
    optional: bool
    default_features: bool
    boundary: str
    seam: str


@dataclass(frozen=True)
class CrateRecord:
    package: str
    path: str
    owner: str
    layer: str
    responsibility: str
    dependencies: tuple[DependencyRecord, ...]

    @property
    def allowed_dependencies(self) -> tuple[str, ...]:
        """Return declared package names for README generators."""
        return tuple(sorted(dependency.package for dependency in self.dependencies))


@dataclass(frozen=True)
class DependencyUse:
    package: str
    features: tuple[str, ...]
    conditions: tuple[str, ...]
    kinds: tuple[str, ...]
    optional: bool
    default_features: bool


@dataclass(frozen=True)
class WorkspaceState:
    members: tuple[str, ...]
    manifests: dict[str, tuple[str, dict[str, Any]]]
    dependencies: dict[str, tuple[DependencyUse, ...]]


@dataclass
class DependencyAccumulator:
    features: set[str]
    conditions: set[str]
    kinds: set[str]
    optional: bool
    default_features: bool


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


def require_strings(row: dict[str, Any], field: str, context: str) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ContractError(f"{context} must define string array `{field}`")
    normalized = tuple(sorted(item.strip() for item in value))
    if len(normalized) != len(set(normalized)):
        raise ContractError(f"{context} `{field}` contains duplicate values")
    return normalized


def require_bool(row: dict[str, Any], field: str, context: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise ContractError(f"{context} must define boolean `{field}`")
    return value


def load_dependency(row: dict[str, Any], context: str) -> DependencyRecord:
    boundary = require_text(row, "boundary", context)
    if boundary not in {"public", "private"}:
        raise ContractError(f"{context} `boundary` must be `public` or `private`")
    kinds = require_strings(row, "kinds", context)
    if not kinds or set(kinds) - {"normal", "build"}:
        raise ContractError(f"{context} `kinds` must contain only `normal` or `build`")
    conditions = require_strings(row, "conditions", context)
    if not conditions:
        raise ContractError(f"{context} must declare at least one dependency condition")
    return DependencyRecord(
        package=require_text(row, "package", context),
        purpose=require_text(row, "purpose", context),
        features=require_strings(row, "features", context),
        conditions=conditions,
        kinds=kinds,
        optional=require_bool(row, "optional", context),
        default_features=require_bool(row, "default_features", context),
        boundary=boundary,
        seam=require_text(row, "seam", context),
    )


def load_registry(root: Path) -> list[CrateRecord]:
    registry = read_toml(root / REGISTRY_PATH)
    if registry.get("schema_version") != 2:
        raise ContractError(f"`{REGISTRY_PATH}` must declare schema_version = 2")
    if "planned" in registry:
        raise ContractError(
            f"`{REGISTRY_PATH}` cannot describe planned crates; add only current workspace owners"
        )

    rows = registry.get("crate")
    if not isinstance(rows, list):
        raise ContractError(f"`{REGISTRY_PATH}` must define one [[crate]] row per member")
    records: list[CrateRecord] = []
    for index, row in enumerate(rows):
        context = f"{REGISTRY_PATH} [[crate]] row {index + 1}"
        if not isinstance(row, dict):
            raise ContractError(f"{context} must be a table")
        if "allowed_dependencies" in row:
            raise ContractError(
                f"{context} uses removed `allowed_dependencies`; declare complete [[crate.dependency]] records"
            )
        dependency_rows = row.get("dependency", [])
        if not isinstance(dependency_rows, list):
            raise ContractError(f"{context} `dependency` must be an array of tables")
        dependencies = tuple(
            load_dependency(dependency, f"{context} dependency {dependency_index + 1}")
            for dependency_index, dependency in enumerate(dependency_rows)
            if isinstance(dependency, dict)
        )
        if len(dependencies) != len(dependency_rows):
            raise ContractError(f"{context} contains a non-table dependency record")
        names = [dependency.package for dependency in dependencies]
        if len(names) != len(set(names)):
            raise ContractError(f"{context} contains duplicate dependency packages")
        records.append(
            CrateRecord(
                package=require_text(row, "package", context),
                path=require_text(row, "path", context),
                owner=require_text(row, "owner", context),
                layer=require_text(row, "layer", context),
                responsibility=require_text(row, "responsibility", context),
                dependencies=tuple(sorted(dependencies, key=lambda dependency: dependency.package)),
            )
        )
    return records


def dependency_tables(
    manifest: dict[str, Any],
) -> Iterable[tuple[dict[str, Any], str, str]]:
    for name, kind in (("dependencies", "normal"), ("build-dependencies", "build")):
        table = manifest.get(name, {})
        if isinstance(table, dict):
            yield table, kind, "always"
    targets = manifest.get("target", {})
    if isinstance(targets, dict):
        for condition, target in targets.items():
            if not isinstance(target, dict):
                continue
            for name, kind in (("dependencies", "normal"), ("build-dependencies", "build")):
                table = target.get(name, {})
                if isinstance(table, dict):
                    yield table, kind, condition


def merged_specification(
    alias: str, specification: Any, workspace_dependencies: dict[str, Any]
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if isinstance(specification, dict) and specification.get("workspace") is True:
        inherited = workspace_dependencies.get(alias, {})
        if isinstance(inherited, dict):
            merged.update(inherited)
        elif isinstance(inherited, str):
            merged["version"] = inherited
    elif isinstance(specification, dict):
        merged.update(specification)
    elif isinstance(specification, str):
        merged["version"] = specification
    if isinstance(specification, dict):
        inherited_features = merged.get("features", [])
        local_features = specification.get("features", [])
        merged.update({key: value for key, value in specification.items() if key != "workspace"})
        if isinstance(inherited_features, list) and isinstance(local_features, list):
            merged["features"] = sorted(set(inherited_features) | set(local_features))
    return merged


def dependency_use(
    alias: str,
    specification: Any,
    workspace_dependencies: dict[str, Any],
    kind: str,
    condition: str,
) -> DependencyUse:
    merged = merged_specification(alias, specification, workspace_dependencies)
    package = merged.get("package", alias)
    if not isinstance(package, str) or not package:
        raise ContractError(f"dependency alias `{alias}` has invalid package metadata")
    features = merged.get("features", [])
    if not isinstance(features, list) or not all(isinstance(item, str) for item in features):
        raise ContractError(f"dependency `{package}` has invalid feature metadata")
    optional = merged.get("optional", False)
    default_features = merged.get("default-features", True)
    if not isinstance(optional, bool) or not isinstance(default_features, bool):
        raise ContractError(f"dependency `{package}` has invalid boolean metadata")
    return DependencyUse(
        package=package,
        features=tuple(sorted(set(features))),
        conditions=(condition,),
        kinds=(kind,),
        optional=optional,
        default_features=default_features,
    )


def workspace_state(root: Path) -> WorkspaceState:
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
    dependencies: dict[str, tuple[DependencyUse, ...]] = {}
    for package, (_, manifest) in manifests.items():
        accumulators: dict[str, DependencyAccumulator] = {}
        for table, kind, condition in dependency_tables(manifest):
            for alias, specification in table.items():
                use = dependency_use(alias, specification, workspace_dependencies, kind, condition)
                if use.package not in package_names:
                    continue
                accumulator = accumulators.setdefault(
                    use.package,
                    DependencyAccumulator(set(), set(), set(), False, True),
                )
                accumulator.features.update(use.features)
                accumulator.conditions.update(use.conditions)
                accumulator.kinds.update(use.kinds)
                accumulator.optional = accumulator.optional or use.optional
                accumulator.default_features = (
                    accumulator.default_features and use.default_features
                )
        dependencies[package] = tuple(
            DependencyUse(
                package=dependency,
                features=tuple(sorted(accumulator.features)),
                conditions=tuple(sorted(accumulator.conditions)),
                kinds=tuple(sorted(accumulator.kinds)),
                optional=accumulator.optional,
                default_features=accumulator.default_features,
            )
            for dependency, accumulator in sorted(accumulators.items())
        )
    return WorkspaceState(tuple(members), manifests, dependencies)


def validate(state: WorkspaceState, records: list[CrateRecord]) -> None:
    by_package: dict[str, CrateRecord] = {}
    by_path: dict[str, CrateRecord] = {}
    for record in records:
        if record.package in by_package:
            raise ContractError(f"ownership registry duplicates package `{record.package}`")
        if record.path in by_path:
            raise ContractError(f"ownership registry duplicates path `{record.path}`")
        by_package[record.package] = record
        by_path[record.path] = record

    member_set = set(state.members)
    registry_paths = set(by_path)
    if member_set != registry_paths:
        missing = sorted(member_set - registry_paths)
        extra = sorted(registry_paths - member_set)
        raise ContractError(
            "ownership registry path set differs from workspace.members: "
            f"missing={missing}, extra={extra}"
        )
    if set(state.manifests) != set(by_package):
        missing = sorted(set(state.manifests) - set(by_package))
        extra = sorted(set(by_package) - set(state.manifests))
        raise ContractError(
            "ownership registry package set differs from workspace packages: "
            f"missing={missing}, extra={extra}"
        )

    for package, (path, _) in state.manifests.items():
        record = by_package[package]
        if record.path != path:
            raise ContractError(
                f"package `{package}` registry path `{record.path}` does not match `{path}`"
            )
        actual = {dependency.package: dependency for dependency in state.dependencies[package]}
        declared = {dependency.package: dependency for dependency in record.dependencies}
        if actual.keys() != declared.keys():
            undeclared = sorted(actual.keys() - declared.keys())
            stale = sorted(declared.keys() - actual.keys())
            owners = {
                dependency: by_package[dependency].owner
                for dependency in undeclared
                if dependency in by_package
            }
            raise ContractError(
                f"package `{package}` production dependency contract differs from manifests: "
                f"undeclared={undeclared}, stale={stale}, owning_boundaries={owners}; "
                f"declare each required destination under `{REGISTRY_PATH}`"
            )
        for dependency, expected in declared.items():
            observed = actual[dependency]
            fields = {
                "features": (expected.features, observed.features),
                "conditions": (expected.conditions, observed.conditions),
                "kinds": (expected.kinds, observed.kinds),
                "optional": (expected.optional, observed.optional),
                "default_features": (
                    expected.default_features,
                    observed.default_features,
                ),
            }
            mismatches = {
                name: {"declared": declared_value, "actual": actual_value}
                for name, (declared_value, actual_value) in fields.items()
                if declared_value != actual_value
            }
            if mismatches:
                raise ContractError(
                    f"dependency `{package}` -> `{dependency}` metadata differs from Cargo: "
                    f"{mismatches}"
                )
            required_seam = by_package[dependency].owner
            if expected.seam != required_seam:
                raise ContractError(
                    f"dependency `{package}` -> `{dependency}` declares seam `{expected.seam}`; "
                    f"required destination owner is `{required_seam}`"
                )


def mermaid_id(index: int) -> str:
    return f"C{index}"


def format_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"`{value}`" for value in values) or "None"


def render_graph(records: list[CrateRecord]) -> str:
    ordered = sorted(records, key=lambda record: record.package)
    ids = {record.package: mermaid_id(index) for index, record in enumerate(ordered)}
    lines = [
        "# Vyre Crate Graph",
        "",
        "This file is generated by `python3 scripts/crate_ownership.py --write` from",
        "the workspace manifests and `docs/CRATE_OWNERSHIP.toml`. Edit those authorities",
        "together, then regenerate this file.",
        "",
        "## Workspace dependency graph",
        "",
        f"The workspace contains {len(ordered)} crates. An arrow points from a crate to",
        "an internal normal or build dependency. Development dependencies are excluded.",
        "",
        "```mermaid",
        "graph TD",
    ]
    for record in ordered:
        lines.append(f'  {ids[record.package]}["{record.package}"]')
    for record in ordered:
        for dependency in record.dependencies:
            lines.append(f"  {ids[record.package]} --> {ids[dependency.package]}")
    lines.extend(
        [
            "```",
            "",
            "## Dependency contracts",
            "",
            "| Consumer | Dependency | Purpose | Features | Conditions | Kinds | Optional | Default features | Boundary | Owning seam |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for record in ordered:
        for dependency in record.dependencies:
            lines.append(
                f"| `{record.package}` | `{dependency.package}` | {dependency.purpose} | "
                f"{format_list(dependency.features)} | {format_list(dependency.conditions)} | "
                f"{format_list(dependency.kinds)} | `{str(dependency.optional).lower()}` | "
                f"`{str(dependency.default_features).lower()}` | `{dependency.boundary}` | "
                f"`{dependency.seam}` |"
            )
    lines.extend(
        [
            "",
            "## Changing a dependency",
            "",
            "Change the Cargo manifest and its complete `[[crate.dependency]]` record in",
            "the same patch. The registry rejects undeclared packages, feature drift, target",
            "condition drift, stale seams, and missing visibility declarations.",
            "",
        ]
    )
    return "\n".join(lines)


def render_ownership(records: list[CrateRecord]) -> str:
    ordered = sorted(records, key=lambda record: record.package)
    lines = [
        "# Vyre Crate Ownership",
        "",
        "This file is generated by `python3 scripts/crate_ownership.py --write` from",
        "`docs/CRATE_OWNERSHIP.toml` and the workspace manifests.",
        "",
        "## Boundary rule",
        "",
        "Each workspace crate has one owner and responsibility. Each internal production",
        "edge declares why it exists, its Cargo feature and target conditions, whether it",
        "crosses the public API, and the destination seam that owns the contract.",
        "",
        "## Per-crate ownership",
        "",
    ]
    for record in ordered:
        lines.extend(
            [
                f"### `{record.package}`",
                "",
                record.responsibility,
                "",
                f"- Path: `{record.path}`",
                f"- Owner: `{record.owner}`",
                f"- Layer: `{record.layer}`",
                f"- Internal production dependencies: {format_list(record.allowed_dependencies)}",
                "",
            ]
        )
        if record.dependencies:
            lines.extend(
                [
                    "| Dependency | Purpose | Boundary | Owning seam |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for dependency in record.dependencies:
                lines.append(
                    f"| `{dependency.package}` | {dependency.purpose} | "
                    f"`{dependency.boundary}` | `{dependency.seam}` |"
                )
            lines.append("")
    lines.extend(
        [
            "## Changing a boundary",
            "",
            "1. Change the manifest and `docs/CRATE_OWNERSHIP.toml` together.",
            "2. Run `python3 scripts/crate_ownership.py --write`.",
            "3. Add a public import migration test when a public edge changes.",
            "4. Run `cargo_full run --bin xtask -- check-tier-deps` and `lego-audit`.",
            "",
        ]
    )
    return "\n".join(lines)


def check_or_write(path: Path, expected: str, write_output: bool) -> None:
    if write_output:
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


def run(root: Path, write_output: bool) -> None:
    root = root.resolve()
    records = load_registry(root)
    state = workspace_state(root)
    validate(state, records)
    check_or_write(root / GRAPH_PATH, render_graph(records), write_output)
    check_or_write(root / OWNERSHIP_PATH, render_ownership(records), write_output)


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
