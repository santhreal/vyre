#!/usr/bin/env python3
"""Execute the command-line contract of every Cargo binary and generate the CLI section of the crate READMEs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from cargo_runner import cargo_runner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs/CLI.toml"
BEGIN = "<!-- BEGIN GENERATED CLI CONTRACT -->"
END = "<!-- END GENERATED CLI CONTRACT -->"
MAX_HELP_BYTES = 1_048_576


def fail(message: str) -> None:
    raise ValueError(message)


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open("rb") as handle:
        data = tomllib.load(handle)
    if data.get("schema_version") != 1:
        fail("docs/CLI.toml: schema_version must be 1")
    entries = data.get("binary")
    if not isinstance(entries, list) or not entries:
        fail("docs/CLI.toml: at least one [[binary]] entry is required")
    required = {
        "package",
        "name",
        "readme",
        "audience",
        "hardware",
        "environment",
        "config",
        "failure",
        "exit_codes",
    }
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries, start=1):
        missing = sorted(required - entry.keys())
        if missing:
            fail(f"docs/CLI.toml binary {index}: missing fields {missing}")
        key = (entry["package"], entry["name"])
        if key in seen:
            fail(f"docs/CLI.toml: duplicate binary {key[0]}/{key[1]}")
        seen.add(key)
        if entry["audience"] not in {"public", "internal"}:
            fail(f"docs/CLI.toml: {key[1]} audience must be public or internal")
        readme = ROOT / entry["readme"]
        if not readme.is_file():
            fail(f"docs/CLI.toml: README does not exist: {entry['readme']}")
    return entries


def cargo_metadata() -> dict:
    output = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if output.returncode != 0:
        fail(f"cargo metadata failed: {output.stderr.decode(errors='replace')}")
    return json.loads(output.stdout)


def inventory_bins(metadata: dict) -> set[tuple[str, str]]:
    return {
        (package["name"], target["name"])
        for package in metadata["packages"]
        for target in package["targets"]
        if "bin" in target["kind"]
    }


def build_bins() -> None:
    runner = cargo_runner(ROOT)
    result = subprocess.run(
        [runner, "build", "--workspace", "--bins"],
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        fail(
            f"`{runner} build --workspace --bins` failed; "
            "repair the CLI build before documenting it"
        )


def run_help(executable: Path, args: list[str]) -> str:
    result = subprocess.run(
        [str(executable), *args],
        cwd=ROOT,
        capture_output=True,
        check=False,
        timeout=60,
    )
    output = result.stdout + result.stderr
    if len(output) > MAX_HELP_BYTES:
        fail(f"{executable.name} {' '.join(args)} help exceeds {MAX_HELP_BYTES} bytes")
    text = output.decode("utf-8", errors="strict")
    if result.returncode != 0:
        fail(
            f"{executable.name} {' '.join(args)} returned {result.returncode}; "
            f"every help route must exit 0 without executing the command:\n{text}"
        )
    if not text.strip():
        fail(f"{executable.name} {' '.join(args)} returned empty help")
    return text.rstrip() + "\n"


def commands_from_help(binary: str, help_text: str) -> list[str]:
    commands: list[str] = []
    in_commands = False
    for line in help_text.splitlines():
        stripped = line.strip()
        if stripped in {"Commands:", "SUBCOMMANDS:"}:
            in_commands = True
            continue
        if in_commands and stripped.endswith(":"):
            in_commands = False
        if in_commands and stripped:
            token = stripped.split()[0]
            if re.fullmatch(r"[a-z][a-z0-9-]*", token) and token != "help":
                commands.append(token)
    if binary == "vyre-conform":
        commands.extend(re.findall(r"vyre-conform (dispatch|plan|merge|prove)\b", help_text))
    if binary == "vyre_new_op" and "vyre new-op" in help_text:
        commands.append("new-op")
    return sorted(set(commands))


def validate_xtask_dispatch(commands: list[str]) -> None:
    """Check the generated help against the registered subcommand table.

    Dispatch used to be a `match` in `xtask/src/main.rs` and this read its arms.
    The table in `xtask/src/subcommands.rs` is now the single registry that help,
    dispatch and CI wiring are all built from, so read that instead.
    """
    source = (ROOT / "xtask/src/subcommands.rs").read_text(encoding="utf-8")
    table = source.split("pub const SUBCOMMANDS", 1)
    if len(table) != 2:
        fail("xtask/src/subcommands.rs no longer defines SUBCOMMANDS")
    dispatch = set(re.findall(r'^\s+name: "([^"]+)",$', table[1], re.M))
    if not dispatch:
        fail("xtask/src/subcommands.rs defines no subcommand rows")
    documented = set(commands)
    if dispatch != documented:
        missing = sorted(dispatch - documented)
        stale = sorted(documented - dispatch)
        fail(f"xtask help/dispatch mismatch: missing={missing}, stale={stale}")



def render_readme_block(entries: list[dict[str, str]], commands: dict) -> str:
    lines = [
        BEGIN,
        "## Command-line interface",
        "",
        "This section is generated from `docs/CLI.toml` and executable help output.",
    ]
    for entry in entries:
        key = (entry["package"], entry["name"])
        lines.extend(
            [
                "",
                f"### `{entry['name']}`",
                "",
                "```console",
                f"./cargo_full run -p {entry['package']} --bin {entry['name']} -- --help",
                "```",
            ]
        )
        if entry["name"] == "vyre-wgpu":
            lines.extend(
                [
                    "",
                    "Run the real GPU smoke path:",
                    "",
                    "```console",
                    "./cargo_full run -p vyre-driver-wgpu --bin vyre-wgpu -- demo",
                    "# vyre demo gpu_u32=42",
                    "```",
                ]
            )
        command_text = ", ".join(f"`{value}`" for value in commands[key]) or "none"
        lines.extend(
            [
                "",
                f"Commands: {command_text}.",
                "",
                f"Hardware: {entry['hardware']}",
                "",
                f"Environment: {entry['environment']}",
                "",
                f"Configuration: {entry['config']}",
                "",
                f"Failure behavior: {entry['failure']}",
                "",
                f"Exit codes: {entry['exit_codes']}",
            ]
        )
    lines.extend([END, ""])
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    start = text.find(BEGIN)
    end = text.find(END)
    if (start < 0) != (end < 0):
        fail("README has only one generated CLI marker")
    if start >= 0:
        end += len(END)
        return (
            text[:start].rstrip()
            + "\n\n"
            + block.rstrip()
            + "\n\n"
            + text[end:].lstrip("\n")
        )
    crate_contract = text.find("<!-- BEGIN GENERATED CRATE CONTRACT -->")
    if crate_contract >= 0:
        return text[:crate_contract].rstrip() + "\n\n" + block + "\n" + text[crate_contract:]
    return text.rstrip() + "\n\n" + block


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        entries = load_manifest()
        metadata = cargo_metadata()
        declared = {(entry["package"], entry["name"]) for entry in entries}
        actual = inventory_bins(metadata)
        if declared != actual:
            fail(f"CLI binary inventory mismatch: missing={sorted(actual-declared)}, stale={sorted(declared-actual)}")
        build_bins()
        target = Path(metadata["target_directory"]) / "debug"
        commands: dict = {}
        for entry in entries:
            key = (entry["package"], entry["name"])
            executable = target / entry["name"]
            top = run_help(executable, ["--help"])
            discovered = commands_from_help(entry["name"], top)
            commands[key] = discovered
            if entry["name"] == "xtask":
                validate_xtask_dispatch(discovered)
            if entry["audience"] == "public":
                for command in discovered:
                    run_help(executable, [command, "--help"])
        outputs: dict[Path, str] = {}
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for entry in entries:
            grouped[entry["readme"]].append(entry)
        for readme, readme_entries in grouped.items():
            path = ROOT / readme
            outputs[path] = replace_block(
                path.read_text(encoding="utf-8"),
                render_readme_block(readme_entries, commands),
            )
        if args.write:
            for path, content in outputs.items():
                path.write_text(content, encoding="utf-8")
            print(f"cli-docs: wrote {len(entries)} binary contracts")
            return 0
        stale = [path.relative_to(ROOT) for path, expected in outputs.items() if path.read_text(encoding="utf-8") != expected]
        if stale:
            fail(f"generated CLI documentation is stale: {stale}; run `python3 scripts/cli_docs.py --write`")
        print(f"cli-docs: verified {len(entries)} binaries and {sum(map(len, commands.values()))} subcommands")
        return 0
    except (OSError, UnicodeDecodeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"cli-docs: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
