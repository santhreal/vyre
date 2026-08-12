#!/usr/bin/env python3
"""Generate one crate-specific testing guide for every Vyre workspace member."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

MAX_TOML_BYTES = 1_048_576
OWNERSHIP_PATH = Path("docs/CRATE_OWNERSHIP.toml")
METADATA_PATH = Path("docs/testing/TESTING.toml")
GUIDE_DIRECTORY = Path("docs/testing")


class ContractError(Exception):
    """Testing guide inputs or generated outputs violate the contract."""


@dataclass(frozen=True)
class Ownership:
    package: str
    path: str
    owner: str
    layer: str
    responsibility: str


@dataclass(frozen=True, order=True)
class Target:
    kind: str
    name: str
    source: str
    required_features: tuple[str, ...]


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

def gitignored(root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0



def require_text(table: dict[str, Any], field: str, context: str) -> str:
    value = table.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{context} must define non-empty `{field}`")
    return value.strip()


def require_text_list(table: dict[str, Any], field: str, context: str) -> list[str]:
    value = table.get(field)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ContractError(f"{context} must define string array `{field}`")
    return [item.strip() for item in value]


def ownership_rows(root: Path) -> list[Ownership]:
    value = read_toml(root / OWNERSHIP_PATH)
    rows = value.get("crate")
    if not isinstance(rows, list):
        raise ContractError(f"`{OWNERSHIP_PATH}` must define [[crate]] rows")
    ownership: list[Ownership] = []
    for index, row in enumerate(rows):
        context = f"{OWNERSHIP_PATH} [[crate]] row {index + 1}"
        if not isinstance(row, dict):
            raise ContractError(f"{context} must be a table")
        ownership.append(
            Ownership(
                package=require_text(row, "package", context),
                path=require_text(row, "path", context),
                owner=require_text(row, "owner", context),
                layer=require_text(row, "layer", context),
                responsibility=require_text(row, "responsibility", context),
            )
        )
    return ownership


def workspace_members(root: Path) -> list[str]:
    value = read_toml(root / "Cargo.toml")
    workspace = value.get("workspace")
    if not isinstance(workspace, dict):
        raise ContractError("workspace Cargo.toml has no [workspace] table")
    members = workspace.get("members")
    if not isinstance(members, list) or not all(isinstance(item, str) for item in members):
        raise ContractError("workspace.members must be a string array")
    return list(members)


def target_name(row: dict[str, Any], fallback: str, context: str) -> str:
    value = row.get("name", fallback)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{context} target name must be a non-empty string")
    return value


def required_features(row: dict[str, Any], context: str) -> tuple[str, ...]:
    value = row.get("required-features", [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractError(f"{context} required-features must be a string array")
    return tuple(sorted(value))


def explicit_targets(manifest: dict[str, Any], kind: str, default_dir: str) -> list[Target]:
    value = manifest.get(kind, [])
    if not isinstance(value, list):
        raise ContractError(f"Cargo [[{kind}]] targets must be an array of tables")
    targets: list[Target] = []
    for index, row in enumerate(value):
        context = f"Cargo [[{kind}]] row {index + 1}"
        if not isinstance(row, dict):
            raise ContractError(f"{context} must be a table")
        name = target_name(row, f"{kind}-{index + 1}", context)
        source = row.get("path", f"{default_dir}/{name}.rs")
        if not isinstance(source, str) or not source:
            raise ContractError(f"{context} path must be a non-empty string")
        targets.append(Target(kind, name, source, required_features(row, context)))
    return targets


def implicit_file_targets(
    crate_root: Path, directory: Path, kind: str, enabled: bool
) -> Iterable[Target]:
    if not enabled or not directory.is_dir():
        return []
    return [
        Target(kind, path.stem, path.relative_to(crate_root).as_posix(), ())
        for path in sorted(directory.glob("*.rs"))
    ]


def cargo_targets(root: Path, record: Ownership, manifest: dict[str, Any]) -> list[Target]:
    crate_root = root / record.path
    package = manifest.get("package", {})
    if not isinstance(package, dict):
        raise ContractError(f"`{record.path}/Cargo.toml` has no [package] table")
    targets: list[Target] = []
    library = manifest.get("lib")
    if isinstance(library, dict):
        name = target_name(library, record.package.replace("-", "_"), "Cargo [lib]")
        source = library.get("path", "src/lib.rs")
        if not isinstance(source, str):
            raise ContractError("Cargo [lib] path must be a string")
        targets.append(Target("lib", name, source, required_features(library, "Cargo [lib]")))
    elif (crate_root / "src/lib.rs").is_file():
        targets.append(Target("lib", record.package.replace("-", "_"), "src/lib.rs", ()))

    targets.extend(explicit_targets(manifest, "bin", "src/bin"))
    if package.get("autobins", True) is not False and (crate_root / "src/main.rs").is_file():
        targets.append(Target("bin", record.package, "src/main.rs", ()))
    targets.extend(
        implicit_file_targets(
            crate_root,
            crate_root / "src/bin",
            "bin",
            package.get("autobins", True) is not False,
        )
    )

    for kind, directory, auto_field in [
        ("test", "tests", "autotests"),
        ("bench", "benches", "autobenches"),
        ("example", "examples", "autoexamples"),
    ]:
        targets.extend(explicit_targets(manifest, kind, directory))
        targets.extend(
            implicit_file_targets(
                crate_root,
                crate_root / directory,
                kind,
                package.get(auto_field, True) is not False,
            )
        )
    return sorted(
        target
        for target in set(targets)
        if not gitignored(root, f"{record.path}/{target.source}")
    )


def merge_metadata(
    defaults: dict[str, Any], profile: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(profile)
    merged.update(override)
    return merged


def command_for_target(package: str, target: Target) -> str:
    flag = {"test": "--test", "bin": "--bin", "example": "--example", "bench": "--bench"}.get(
        target.kind
    )
    base = f"CARGO_BUILD_JOBS=1 ./cargo_full test -p {package}"
    if flag is None:
        return base
    return f"{base} {flag} {target.name}"


def markdown_list(items: Iterable[str]) -> list[str]:
    return [f"- {item}" for item in items]


def render_guide(
    record: Ownership,
    manifest: dict[str, Any],
    targets: list[Target],
    metadata: dict[str, Any],
) -> str:
    features = manifest.get("features", {})
    if not isinstance(features, dict):
        raise ContractError(f"`{record.path}/Cargo.toml` [features] must be a table")
    feature_names = sorted(features)
    default_features = features.get("default", [])
    if not isinstance(default_features, list) or not all(
        isinstance(item, str) for item in default_features
    ):
        raise ContractError(f"`{record.path}/Cargo.toml` features.default must be a string array")

    context = f"testing metadata for `{record.package}`"
    hardware = require_text(metadata, "hardware", context)
    expected_skips = require_text(metadata, "expected_skips", context)
    failure_behavior = require_text(metadata, "failure_behavior", context)
    test_classes = require_text_list(metadata, "test_classes", context)
    evidence_outputs = require_text_list(metadata, "evidence_outputs", context)
    extra_commands = metadata.get("commands", [])
    if not isinstance(extra_commands, list) or not all(
        isinstance(item, str) and item.strip() for item in extra_commands
    ):
        raise ContractError(f"{context} commands must be a string array")

    commands = [f"CARGO_BUILD_JOBS=1 ./cargo_full test -p {record.package}"]
    if feature_names:
        commands.append(
            f"CARGO_BUILD_JOBS=1 ./cargo_full test -p {record.package} --all-features"
        )
    commands.extend(item.strip() for item in extra_commands)
    commands = list(dict.fromkeys(commands))

    lines = [
        f"# Testing `{record.package}`",
        "",
        "Run the default crate suite from the workspace root:",
        "",
        "```console",
        commands[0],
        "```",
        "",
        record.responsibility,
        "",
        f"The crate lives at `{record.path}`. The `{record.owner}` owner maintains its",
        f"`{record.layer}` testing contract.",
        "",
        "## Commands",
        "",
    ]
    for command in commands:
        lines.extend(["```console", command, "```", ""])

    lines.extend(["## Feature sets", ""])
    if feature_names:
        defaults = ", ".join(f"`{item}`" for item in default_features) or "None"
        available = ", ".join(f"`{item}`" for item in feature_names)
        lines.extend(
            [
                f"- Default feature members: {defaults}",
                f"- Available manifest features: {available}",
                "- Use the all-features command above to compile every declared feature together.",
            ]
        )
    else:
        lines.append("This crate declares no Cargo features.")

    lines.extend(["", "## Cargo targets", ""])
    if targets:
        lines.extend(
            [
                "| Kind | Target | Source | Required features | Focused command |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for target in targets:
            required = ", ".join(f"`{item}`" for item in target.required_features) or "None"
            lines.append(
                f"| `{target.kind}` | `{target.name}` | `{record.path}/{target.source}` | {required} | "
                f"`{command_for_target(record.package, target)}` |"
            )
    else:
        lines.append("Cargo declares no executable, library, test, example, or benchmark target.")

    lines.extend(["", "## Test classes", ""])
    lines.extend(markdown_list(test_classes))
    lines.extend(["", "## Hardware requirements", "", hardware])
    lines.extend(["", "## Evidence outputs", ""])
    lines.extend(markdown_list(f"`{item}`" if "/" in item else item for item in evidence_outputs))
    lines.extend(
        [
            "",
            "## Skips and failures",
            "",
            expected_skips,
            "",
            failure_behavior,
            "",
        ]
    )
    return "\n".join(lines)


def guide_name(record: Ownership) -> str:
    return f"{Path(record.path).name}.md"


def run(root: Path, write: bool) -> None:
    root = root.resolve()
    records = ownership_rows(root)
    members = workspace_members(root)
    if {record.path for record in records} != set(members):
        raise ContractError(
            "crate ownership paths must match workspace.members before testing guides can generate"
        )

    config = read_toml(root / METADATA_PATH)
    if config.get("schema_version") != 1:
        raise ContractError(f"`{METADATA_PATH}` must declare schema_version = 1")
    defaults = config.get("defaults")
    profiles = config.get("profile", {})
    overrides = config.get("package", {})
    if not isinstance(defaults, dict):
        raise ContractError(f"`{METADATA_PATH}` must define [defaults]")
    if not isinstance(profiles, dict):
        raise ContractError(f"`{METADATA_PATH}` [profile] value must contain tables")
    if not isinstance(overrides, dict):
        raise ContractError(f"`{METADATA_PATH}` [package] overrides must be tables")
    unknown_overrides = sorted(set(overrides) - {record.package for record in records})
    if unknown_overrides:
        raise ContractError(f"testing metadata has unknown package overrides: {unknown_overrides}")

    expected: dict[Path, str] = {}
    for record in records:
        profile = profiles.get(record.layer)
        if not isinstance(profile, dict):
            raise ContractError(
                f"testing metadata has no profile for ownership layer `{record.layer}` used by `{record.package}`"
            )
        override = overrides.get(record.package, {})
        if not isinstance(override, dict):
            raise ContractError(f"testing override for `{record.package}` must be a table")
        manifest = read_toml(root / record.path / "Cargo.toml")
        package = manifest.get("package", {})
        if not isinstance(package, dict) or package.get("name") != record.package:
            raise ContractError(
                f"`{record.path}/Cargo.toml` package name does not match `{record.package}`"
            )
        targets = cargo_targets(root, record, manifest)
        destination = root / GUIDE_DIRECTORY / guide_name(record)
        if destination in expected:
            raise ContractError(f"testing guide filename collision at `{destination}`")
        expected[destination] = render_guide(
            record, manifest, targets, merge_metadata(defaults, profile, override)
        )

    actual_guides = set((root / GUIDE_DIRECTORY).glob("*.md"))
    expected_guides = set(expected)
    extras = sorted(path.name for path in actual_guides - expected_guides)
    if extras:
        raise ContractError(f"testing guide directory contains non-member guides: {extras}")

    stale: list[str] = []
    for path, content in sorted(expected.items()):
        if write:
            path.write_text(content, encoding="utf-8")
            continue
        try:
            actual = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            stale.append(path.name)
            continue
        if actual != content:
            stale.append(path.name)
    if stale:
        raise ContractError(
            f"missing or stale testing guides: {stale}; run "
            "`python3 scripts/testing_guides.py --write`"
        )


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
    mode.add_argument("--check", action="store_true", help="reject metadata or guide drift")
    mode.add_argument("--write", action="store_true", help="regenerate every testing guide")
    args = parser.parse_args()
    try:
        run(args.root, args.write)
    except ContractError as error:
        print(f"testing-guides: {error}", file=sys.stderr)
        return 1
    action = "wrote" if args.write else "verified"
    print(f"testing-guides: {action} {len(ownership_rows(args.root.resolve()))} crate guides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
