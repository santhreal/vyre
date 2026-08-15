#!/usr/bin/env python3
"""Assert every feature combination builds against the declared MSRV.

Usage: check_feature_msrv.py ROOT CARGO_RUNNER [TOOLCHAIN]

This lives in its own file rather than a shell heredoc: the release-hygiene scan
blocks heredocs in release tooling, since a heredoc hides an entire second
language from review, lint, and syntax checking.
"""

import sys
import tomllib
from pathlib import Path
from subprocess import run

root = Path(sys.argv[1]).resolve()
cargo = sys.argv[2]
mode = sys.argv[3]
NAME = "feature-MSRV"

if mode not in ("", "--list"):
    sys.exit(f"{NAME}: unknown argument `{mode}`; use --list or no argument.")


def fatal(message: str) -> None:
    sys.exit(f"{NAME}: {message}")


def read_manifest(rel: str) -> dict:
    try:
        return tomllib.loads((root / rel).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        fatal(f"{rel} is not readable as TOML: {exc}")


root_manifest = read_manifest("Cargo.toml")
workspace = root_manifest.get("workspace") or {}
msrv = ((workspace.get("package") or {}).get("rust-version") or "").strip()
if not msrv:
    fatal("[workspace.package].rust-version is missing; the MSRV is the whole point")

try:
    installed = run(["rustup", "toolchain", "list"], capture_output=True, text=True)
except OSError as exc:
    fatal(
        f"`rustup` is not runnable: {exc}\n"
        f"    Fix: install rustup. The sweep pins a toolchain by name, so without "
        f"rustup there is no way to compile on {msrv} rather than on whatever "
        f"compiler happens to be first on PATH."
    )
if installed.returncode != 0:
    fatal(
        f"`rustup toolchain list` failed: {installed.stderr.strip()}\n"
        f"    A sweep that cannot confirm the toolchain would check some other "
        f"compiler and report MSRV."
    )
if not any(line.startswith(msrv) for line in installed.stdout.splitlines()):
    fatal(
        f"the MSRV toolchain {msrv} is not installed.\n"
        f"    Fix: rustup toolchain install {msrv}. Falling back to the default "
        f"toolchain would report a pass for a compiler this crate does not claim."
    )

members = workspace.get("members") or []
if not members:
    fatal("[workspace.members] is empty; the sweep would check nothing")

matrix: list[tuple[str, list[str], str]] = []
for member in members:
    manifest = read_manifest(f"{member}/Cargo.toml")
    package = manifest.get("package") or {}
    name = package.get("name")
    if not name:
        fatal(f"{member}/Cargo.toml has no [package].name")
    publish = package.get("publish", True)
    if publish is False or (isinstance(publish, list) and not publish):
        continue
    features = sorted(key for key in (manifest.get("features") or {}) if key != "default")
    if not features:
        continue
    matrix.append((name, [], "default features"))
    matrix.append((name, ["--no-default-features"], "no default features"))
    for feature in features:
        matrix.append(
            (name, ["--no-default-features", "--features", feature], f"only `{feature}`")
        )

if not matrix:
    fatal("no publishable member declares a feature; the sweep would check nothing")

if mode == "--list":
    print(f"{NAME}: MSRV {msrv}, {len(matrix)} derived matrix entries")
    for name, flags, label in matrix:
        print(f"  - {name}: {label}")
    print("Run without --list to compile each entry. --list checks nothing.")
    sys.exit(0)

failed: list[tuple[str, str]] = []
for index, (name, flags, label) in enumerate(matrix, 1):
    argv = [cargo, f"+{msrv}", "check", "--locked", "-p", name, *flags]
    print(f"[{index}/{len(matrix)}] {name}: {label}", flush=True)
    done = run(argv, cwd=root, capture_output=True, text=True)
    if done.returncode != 0:
        tail = "\n".join(
            f"      {line}" for line in done.stderr.strip().splitlines()[-6:]
        )
        failed.append((f"{name} ({label})", f"      $ {' '.join(argv)}\n{tail}"))

if failed:
    print(
        f"{NAME}: {len(failed)} of {len(matrix)} matrix entries fail on {msrv}.",
        file=sys.stderr,
    )
    for label, detail in failed:
        print(f"  - {label}\n{detail}", file=sys.stderr)
    sys.exit(1)

print(f"{NAME}: all {len(matrix)} derived matrix entries compile on {msrv}.")
