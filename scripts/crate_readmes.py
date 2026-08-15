#!/usr/bin/env python3
"""Generate and check manifest-backed contract sections in every crate README."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from crate_ownership import ContractError, load_registry, read_toml, validate, workspace_state

MAX_README_BYTES = 2_097_152
METADATA_PATH = Path("docs/CRATE_GUIDES.toml")
RELEASE_TRAIN_PATH = Path("release/release-train.toml")
BEGIN_MARKER = "<!-- BEGIN GENERATED CRATE CONTRACT -->"
END_MARKER = "<!-- END GENERATED CRATE CONTRACT -->"
RETIRED_VERSION = re.compile(r"\b0\.4\.\d+\b")


def require_text(table: dict[str, Any], field: str, context: str) -> str:
    value = table.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{context} must define non-empty `{field}`")
    return value.strip()


def crate_manifest(root: Path, relative: str) -> dict[str, Any]:
    return read_toml(root / relative / "Cargo.toml")


def feature_contract(manifest: dict[str, Any], context: str) -> tuple[list[str], list[str]]:
    features = manifest.get("features", {})
    if not isinstance(features, dict):
        raise ContractError(f"{context} [features] must be a table")
    names = sorted(features)
    defaults = features.get("default", [])
    if not isinstance(defaults, list) or not all(isinstance(item, str) for item in defaults):
        raise ContractError(f"{context} features.default must be a string array")
    return names, list(defaults)


def gitignored(root: Path, path: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def explicit_example(manifest: dict[str, Any]) -> tuple[str, list[str]] | None:
    rows = manifest.get("example", [])
    if not isinstance(rows, list):
        raise ContractError("Cargo [[example]] targets must be an array")
    candidates: list[tuple[str, list[str]]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ContractError(f"Cargo [[example]] row {index + 1} must be a table")
        name = row.get("name")
        required = row.get("required-features", [])
        if not isinstance(name, str) or not name:
            raise ContractError(f"Cargo [[example]] row {index + 1} has no name")
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ContractError(
                f"Cargo [[example]] `{name}` required-features must be a string array"
            )
        candidates.append((name, sorted(required)))
    return sorted(candidates)[0] if candidates else None


def runnable_example(
    root: Path, package: str, relative: str, manifest: dict[str, Any]
) -> tuple[str, str]:
    explicit = explicit_example(manifest)
    if explicit is not None:
        name, features = explicit
        feature_args = f" --features {','.join(features)}" if features else ""
        return (
            f"`{relative}/examples/{name}.rs`",
            f"./cargo_full run -p {package} --example {name}{feature_args}",
        )
    examples = sorted(
        path
        for path in (root / relative / "examples").glob("*.rs")
        if not gitignored(root, path)
    )
    if examples:
        example = examples[0]
        return (
            f"`{example.relative_to(root).as_posix()}`",
            f"./cargo_full run -p {package} --example {example.stem}",
        )

    package_table = manifest.get("package", {})
    autobins = not isinstance(package_table, dict) or package_table.get("autobins", True) is not False
    if autobins and (root / relative / "src/main.rs").is_file():
        return (
            f"`{relative}/src/main.rs`",
            f"./cargo_full run -p {package} -- --help",
        )
    bins = manifest.get("bin", [])
    if isinstance(bins, list) and bins:
        names = sorted(
            row.get("name")
            for row in bins
            if isinstance(row, dict) and isinstance(row.get("name"), str)
        )
        if names:
            return (
                f"the `{names[0]}` binary target",
                f"./cargo_full run -p {package} --bin {names[0]} -- --help",
            )

    tests = sorted(
        path
        for path in (root / relative / "tests").glob("*.rs")
        if not gitignored(root, path)
    )
    if tests:
        test = tests[0]
        return (
            f"`{test.relative_to(root).as_posix()}`",
            f"./cargo_full test -p {package} --test {test.stem}",
        )
    return (
        f"the `{package}` library target",
        f"./cargo_full test -p {package} --lib",
    )


def release_versions(root: Path) -> dict[str, str]:
    value = read_toml(root / RELEASE_TRAIN_PATH)
    table = value.get("versions")
    if not isinstance(table, dict):
        raise ContractError(f"`{RELEASE_TRAIN_PATH}` must define [versions]")
    versions = {
        f"{key}_version": version
        for key, version in table.items()
        if isinstance(version, str)
    }
    if "vyre_version" not in versions:
        raise ContractError(f"`{RELEASE_TRAIN_PATH}` must define versions.vyre")
    return versions


def default_release_status(package: str, version: str, publishable: bool) -> str:
    if publishable:
        return (
            f"`{package}@{version}` is a publishable crate on the current Vyre release train. "
            "Publication still requires the release evidence and user-approval gates."
        )
    return (
        f"`{package}@{version}` is workspace-internal on the current Vyre release train "
        "and is not published as a standalone crate."
    )


def publishable(manifest: dict[str, Any]) -> bool:
    package = manifest.get("package", {})
    if not isinstance(package, dict):
        return False
    value = package.get("publish", True)
    return value is not False and value != []


def render_contract(
    root: Path,
    record: Any,
    manifest: dict[str, Any],
    profile: dict[str, Any],
    override: dict[str, Any],
    versions: dict[str, str],
) -> str:
    package_table = manifest.get("package")
    if not isinstance(package_table, dict):
        raise ContractError(f"`{record.path}/Cargo.toml` has no [package] table")
    version = package_table.get("version")
    if not isinstance(version, str) or not version:
        raise ContractError(f"`{record.path}/Cargo.toml` has no package.version")
    features, default_features = feature_contract(
        manifest, f"`{record.path}/Cargo.toml`"
    )
    error_behavior = require_text(
        override if "error_behavior" in override else profile,
        "error_behavior",
        f"crate guide metadata for `{record.package}`",
    )
    status_template = override.get("release_status")
    if status_template is None:
        release_status = default_release_status(
            record.package, version, publishable(manifest)
        )
    elif isinstance(status_template, str) and status_template.strip():
        try:
            release_status = status_template.format(**versions)
        except KeyError as error:
            raise ContractError(
                f"crate guide status for `{record.package}` uses unknown release placeholder {error}"
            ) from error
    else:
        raise ContractError(
            f"crate guide release_status for `{record.package}` must be non-empty text"
        )

    example_source, example_command = runnable_example(
        root, record.package, record.path, manifest
    )
    allowed = ", ".join(f"`{item}`" for item in record.allowed_dependencies) or "None"
    available = ", ".join(f"`{item}`" for item in features) or "None"
    defaults = ", ".join(f"`{item}`" for item in default_features) or "None"
    testing_toml = "docs/testing/TESTING.toml"
    testing_link = f"{'../' * len(Path(record.path).parts)}{testing_toml}"
    ownership_toml = "docs/CRATE_OWNERSHIP.toml"
    ownership_link = f"{'../' * len(Path(record.path).parts)}{ownership_toml}"
    return "\n".join(
        [
            BEGIN_MARKER,
            "## Crate contract",
            "",
            "This section is generated by `python3 scripts/crate_readmes.py --write` from",
            "the crate manifest, release train, ownership registry, and crate-guide metadata.",
            "",
            "### Purpose",
            "",
            record.responsibility,
            "",
            "### Boundaries",
            "",
            f"The `{record.owner}` owner maintains this `{record.layer}` crate at `{record.path}`.",
            f"Its allowed internal production dependencies are: {allowed}.",
            "Any other normal or build dependency requires an ownership-registry change.",
            "",
            "### Minimal real example",
            "",
            f"Run the checked-in behavior from {example_source}:",
            "",
            "```console",
            example_command,
            "```",
            "",
            "### Features",
            "",
            f"- Manifest features: {available}",
            f"- Default feature members: {defaults}",
            "",
            "### Errors and unsupported behavior",
            "",
            error_behavior,
            "",
            "### Testing",
            "",
            f"See [`{testing_toml}`]({testing_link}) for the crate's test command,",
            "hardware contract, expected skips, and failure semantics.",
            "",
            "### Release status",
            "",
            release_status,
            "",
            "### Ownership",
            "",
            f"[`{ownership_toml}`]({ownership_link}) is authoritative for this crate's",
            "responsibility and allowed internal edges.",
            "",
            "### License",
            "",
            "Licensed under either of",
            "",
            "- Apache License, Version 2.0, or",
            "- MIT license",
            "",
            "at your option. See the workspace `LICENSE-APACHE` and `LICENSE-MIT` files.",
            "",
            END_MARKER,
            "",
        ]
    )


def strip_generated_contract(text: str, path: Path) -> str:
    begin_count = text.count(BEGIN_MARKER)
    end_count = text.count(END_MARKER)
    if begin_count != end_count or begin_count > 1:
        raise ContractError(
            f"`{path}` has unbalanced or duplicate generated crate contract markers"
        )
    if begin_count == 0:
        return text.rstrip()
    before, remainder = text.split(BEGIN_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    return f"{before.rstrip()}\n{after.strip()}".strip()


def read_readme(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ContractError(f"could not inspect `{path}`: {error}") from error
    if size > MAX_README_BYTES:
        raise ContractError(
            f"`{path}` is {size} bytes, above the {MAX_README_BYTES}-byte README limit"
        )
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ContractError(f"could not read `{path}`: {error}") from error


def composed_readme(existing: str, package: str, contract: str, path: Path) -> str:
    body = strip_generated_contract(existing, path)
    if not body:
        body = "\n".join(
            [
                f"# `{package}`",
                "",
                "Use this crate through the contract and checked-in example below.",
            ]
        )
    return f"{body.rstrip()}\n\n{contract}"


def run(root: Path, write: bool) -> int:
    root = root.resolve()
    records = load_registry(root)
    state = workspace_state(root)
    validate(state, records)
    config = read_toml(root / METADATA_PATH)
    if config.get("schema_version") != 1:
        raise ContractError(f"`{METADATA_PATH}` must declare schema_version = 1")
    profiles = config.get("profile", {})
    overrides = config.get("package", {})
    if not isinstance(profiles, dict) or not isinstance(overrides, dict):
        raise ContractError(f"`{METADATA_PATH}` must define profile and package tables")
    known = {record.package for record in records}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ContractError(f"crate guide metadata has unknown package overrides: {unknown}")
    # A profile for a layer no crate occupies is a promise about an architecture
    # that no longer exists: it survives a layer rename or a crate absorption and
    # reads as coverage while describing nothing.
    layers_in_use = {record.layer for record in records}
    orphaned = sorted(set(profiles) - layers_in_use)
    if orphaned:
        raise ContractError(
            f"crate guide metadata has error profiles for layers no crate uses: {orphaned}"
        )
    versions = release_versions(root)

    stale: list[str] = []
    retired: list[str] = []
    for record in records:
        profile = profiles.get(record.layer)
        if not isinstance(profile, dict):
            raise ContractError(
                f"crate guide metadata has no error profile for layer `{record.layer}` used by `{record.package}`"
            )
        override = overrides.get(record.package, {})
        if not isinstance(override, dict):
            raise ContractError(f"crate guide override for `{record.package}` must be a table")
        manifest = crate_manifest(root, record.path)
        path = root / record.path / "README.md"
        existing = read_readme(path)
        contract = render_contract(
            root, record, manifest, profile, override, versions
        )
        expected = composed_readme(existing, record.package, contract, path)
        if RETIRED_VERSION.search(expected):
            retired.append(path.relative_to(root).as_posix())
        if write:
            path.write_text(expected, encoding="utf-8")
        elif existing != expected:
            stale.append(path.relative_to(root).as_posix())

    if retired:
        raise ContractError(
            f"crate READMEs contain retired 0.4.x release claims: {sorted(retired)}"
        )
    if stale:
        raise ContractError(
            f"missing or stale crate README contracts: {sorted(stale)}; run "
            "`python3 scripts/crate_readmes.py --write`"
        )
    return len(records)


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
    mode.add_argument("--check", action="store_true", help="reject README contract drift")
    mode.add_argument("--write", action="store_true", help="update every crate README contract")
    args = parser.parse_args()
    try:
        count = run(args.root, args.write)
    except ContractError as error:
        print(f"crate-readmes: {error}", file=sys.stderr)
        return 1
    action = "wrote" if args.write else "verified"
    print(f"crate-readmes: {action} {count} crate README contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
