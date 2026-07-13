#!/usr/bin/env bash
# quality bar (enforcement gates + 1M+ test execution ledger).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
COORD="$(cd "$ROOT/../../../../coordination/vyre-quality-sweep" && pwd)"
source scripts/lib/cargo_runner.sh
vyre_select_cargo_runner

echo "=== quality_gate: check_expect_has_fix ==="
bash scripts/check_expect_has_fix.sh

echo "=== quality_gate: test execution ledger (>=1M) ==="
bash "$COORD/scripts/test_execution_ledger.sh"

echo "=== quality_gate: cargo check --workspace ==="
"$CARGO_RUNNER" check --workspace

echo "=== quality_gate: xtask check-tier-deps ==="
"$CARGO_RUNNER" run -p xtask --bin xtask -- check-tier-deps

echo "=== quality_gate: xtask platform-boundary ==="
"$CARGO_RUNNER" run -p xtask --bin xtask -- platform-boundary

echo "=== quality_gate: xtask catalog --check ==="
"$CARGO_RUNNER" run -p xtask --bin xtask -- catalog --check

echo "=== quality_gate: lint-shape-tests ==="
"$CARGO_RUNNER" run -p xtask --bin xtask -- lint-shape-tests

echo "=== quality_gate: contract_workspace ==="
"$CARGO_RUNNER" test -p vyre-foundation --test contract_workspace

echo "=== quality_gate: sweep oracle matrix (original 23) ==="
bash scripts/run_sweep_oracle_matrix.sh

echo "=== quality_gate: volume oracle sample ==="
"$CARGO_RUNNER" test -p vyre-primitives --features 'hash,bitset,cpu-parity' \
  --test sweep_hash_volume_oracle_matrix \
  --test sweep_bitset_and_not_volume_oracle_matrix -q
"$CARGO_RUNNER" test -p vyre-foundation --test sweep_validation_rejection_volume_oracle_matrix -q

echo "=== quality_gate: vyre-primitives lib (graph) ==="
"$CARGO_RUNNER" test -p vyre-primitives --features graph --lib -q

echo ""
echo "quality GATE: ALL CHECKS PASSED (incl. >=1M test execution ledger)"
