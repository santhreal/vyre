#!/usr/bin/env bash
# Public API compatibility gate.
#
# The Rust gate owns the crate inventory, simplified snapshot format, bounded
# reads, and non-breaking update policy. This shell entry point only selects the
# repository Cargo wrapper and delegates so CI and local checks cannot drift.

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v cargo-public-api >/dev/null 2>&1; then
    echo "public API check requires cargo-public-api" >&2
    echo "  install: cargo install cargo-public-api --locked" >&2
    exit 1
fi

source scripts/lib/cargo_runner.sh
vyre_select_cargo_runner
VYRE_CARGO_RUNNER="$CARGO_RUNNER" "$CARGO_RUNNER" run --bin public_api_check
