#!/usr/bin/env bash
# Every tracked .rs file must be compiled by some declared cargo target.
#
# The class this closes: a source file nothing declares is not code, it is a
# claim. `vyre-libs/src/visual/glyph_grid/mod.rs` shipped an op registration and
# eight contracts while `docs/catalog/visual.md`, `docs/generated/OP_SCHEMA.json`
# and `docs/optimization/OP_MATRIX.toml` all listed the op as supported; no `mod`
# declaration ever named it, so none of it compiled and none of it ran. The four
# `vyre-driver-cuda/tests/resident_dispatch_contracts/*_contracts.rs` chunks lost
# their parent test file to a deletion and took 15 contracts with them, while
# OP_MATRIX.toml still cited the file as the proving test for elementwise_add.
# Both read as coverage from any distance.
#
# The target set is derived from the tracked manifests at run time, not listed
# here, so a new crate, bin, test, bench or example is picked up by adding it and
# a new orphan cannot be added without turning this gate red.
#
# It reports four failures:
#   1. a tracked .rs file no target reaches,
#   2. a declared target `path` that names no tracked file,
#   3. a `mod` declaration in a reached file that resolves to no tracked file,
#   4. a stale exemption: a template root with no tracked .rs, or a trybuild
#      fixture path matching nothing.
#
# This gate deliberately does NOT invoke cargo. `cargo build` cannot see this
# defect at all: an undeclared file is not part of any target, so a green build
# is exactly what an orphan produces. It reads tracked files with git, tomllib
# and a module-graph walk.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 "$ROOT/scripts/lib/check_every_source_file_is_reachable.py" "$ROOT"
