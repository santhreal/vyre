#!/usr/bin/env bash
# Local simulation of .github/workflows/wire-ci.yml.
# Run as a pre-commit / pre-push hook:
#
#   ln -sf "$(realpath scripts/wire_ci_local.sh)" \
#       /media/mukund-thiru/SanthData/Santh/.git/hooks/pre-push
#
# Exits non-zero on the first failed step so the hook blocks the push.
# Time budget mirrors the CI workflow target: under 10 min wall.

set -euo pipefail

# Run from the vyre root regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Same env-var as the workflow so proptest cases stay CI-sized (1k, not 10k).
export PROPTEST_CASES="${PROPTEST_CASES:-1000}"
export RUST_BACKTRACE=1
# Cargo is incremental locally; CI sets CARGO_INCREMENTAL=0 — mirror it
# so the local + CI outputs are bit-comparable.
export CARGO_INCREMENTAL=0

log() { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }

log "fmt — wire surface"
cargo fmt -p vyre-primitives -- --check vyre-primitives/src/wire.rs

log "clippy — wire crates (--no-deps keeps the gate scoped to our code)"
cargo clippy -p vyre-primitives --no-deps \
    --features "matching cpu-parity hash inventory-registry" -- -D warnings
cargo clippy -p vyre-libs --no-deps -- -D warnings

log "check — wire and consumers"
cargo check -p vyre-primitives
cargo check -p vyre-libs
cargo check -p vyre-frontend-c
cargo check -p vyre-intrinsics
cargo check -p vyre-self-substrate
cargo check -p vyre-bench
cargo check -p vyre-driver

log "test — wire contracts (positive + negative + property + differential)"
cargo test -p vyre-primitives --test wire_pack_into_contracts --features matching
cargo test -p vyre-primitives --test wire_differential_std_io --features matching
cargo test -p vyre-primitives --test proptest_wire_roundtrip --features matching

log "test — cross-crate compat"
cargo test -p vyre-libs --test wire_cross_crate_compat

log "harness — build + run the agent-harness smoke binary"
cargo build --release --example wire_harness_smoke -p vyre-primitives
cargo test -p vyre-primitives --test wire_harness_smoke_test --features matching

log "doc-build — wire module doctests"
cargo test --doc -p vyre-primitives wire

log "determinism — run the contract suite twice; outputs must match"
TMP1="$(mktemp)"
TMP2="$(mktemp)"
trap 'rm -f "$TMP1" "$TMP2"' EXIT
cargo test -p vyre-primitives --test wire_pack_into_contracts --features matching \
    -- --nocapture --test-threads=1 > "$TMP1" 2>&1 || true
cargo test -p vyre-primitives --test wire_pack_into_contracts --features matching \
    -- --nocapture --test-threads=1 > "$TMP2" 2>&1 || true
diff <(grep -E '^test ' "$TMP1" | sort) <(grep -E '^test ' "$TMP2" | sort)

printf '\n\033[1;32m✓ wire CI passed (pre-commit-hook ready)\033[0m\n'
