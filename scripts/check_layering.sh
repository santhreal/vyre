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

python3 scripts/lib/check_layering.py "$PWD" "$CARGO_RUNNER"
