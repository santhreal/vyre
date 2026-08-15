#!/usr/bin/env bash
# Every advertised feature of every publishable crate compiles alone, on the
# workspace MSRV.
#
# Two classes, neither covered elsewhere:
#
#   - Feature isolation. strict.yml builds `--all-features`, which is a union:
#     a feature whose prerequisites are turned on by some other feature passes
#     there and breaks for the consumer who enables it alone. ci.yml builds
#     default features only, so it never sees a granular feature at all.
#   - MSRV. `[workspace.package].rust-version` is a published claim, and
#     ci.yml's toolchain matrix is `stable` and `nightly`. Nothing compiles this
#     workspace on the version it advertises.
#
# The matrix is derived from the tracked manifests at run time: every
# publishable member that declares features contributes default,
# `--no-default-features`, and each declared feature alone. A new feature or a
# new member joins the matrix and turns this red until it compiles. The previous
# revision hardcoded 19 entries, four of which named features that no longer
# exist (`vyre-aot --features spirv` among them), so the sweep could only fail;
# and its default mode printed the matrix and exited 0, which is a gate that
# reports success without checking anything.
#
# Usage:
#   scripts/check_feature_msrv.sh          # run the derived sweep on the MSRV
#   scripts/check_feature_msrv.sh --list   # print the derived matrix, check nothing

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/cargo_runner.sh
vyre_select_cargo_runner

python3 "$ROOT/scripts/lib/check_feature_msrv.py" "$ROOT" "$CARGO_RUNNER" "${1:-}"
