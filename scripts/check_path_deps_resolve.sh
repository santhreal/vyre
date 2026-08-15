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

python3 "$ROOT/scripts/lib/check_path_deps_resolve.py" "$ROOT"
