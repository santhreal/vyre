#!/usr/bin/env bash
# Legendary bar — enforcement gates + 1M+ test execution ledger.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
COORD="$(cd "$ROOT/../../../../coordination/vyre-legendary-sweep" && pwd)"

echo "=== legendary_gate: check_expect_has_fix ==="
bash scripts/check_expect_has_fix.sh

echo "=== legendary_gate: test execution ledger (>=1M) ==="
bash "$COORD/scripts/test_execution_ledger.sh"

echo "=== legendary_gate: cargo check --workspace ==="
cargo check --workspace

echo "=== legendary_gate: xtask check-tier-deps ==="
cargo run -p xtask --bin xtask -- check-tier-deps

echo "=== legendary_gate: xtask platform-boundary ==="
cargo run -p xtask --bin xtask -- platform-boundary

echo "=== legendary_gate: xtask catalog --check ==="
cargo run -p xtask --bin xtask -- catalog --check

echo "=== legendary_gate: lint-shape-tests ==="
cargo run -p xtask --bin xtask -- lint-shape-tests

echo "=== legendary_gate: contract_workspace ==="
cargo test -p vyre-foundation --test contract_workspace

echo "=== legendary_gate: sweep oracle matrix (original 23) ==="
bash scripts/run_sweep_oracle_matrix.sh

echo "=== legendary_gate: volume oracle sample ==="
cargo test -p vyre-primitives --features 'hash,bitset,cpu-parity' \
  --test sweep_hash_volume_oracle_matrix \
  --test sweep_bitset_and_not_volume_oracle_matrix -q
cargo test -p vyre-foundation --test sweep_validation_rejection_volume_oracle_matrix -q

echo "=== legendary_gate: vyre-primitives lib (graph) ==="
cargo test -p vyre-primitives --features graph --lib -q

echo ""
echo "LEGENDARY GATE: ALL CHECKS PASSED (incl. >=1M test execution ledger)"
