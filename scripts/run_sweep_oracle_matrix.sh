#!/usr/bin/env bash
# Run every tracked `sweep_*` oracle-matrix integration test that is not a volume wave.
#
# The class this closes: a feature-gated integration test that no workflow executes.
# `ci.yml` runs `./cargo_full test --workspace` with default features, so a test whose
# `[[test]] required-features` name a non-default feature is silently skipped there, and
# `strict.yml` builds `--all-features --all-targets` without running anything. These
# sweeps are the oracle-parity matrices, so a skipped one is unproven parity.
#
# The roster and the per-crate feature set are derived by scripts/lib/sweep_targets.py
# from tracked sources and each crate's own manifest. Cargo's `--test` takes exact
# binary names and no globs, which is why this script exists at all; it is not why the
# names may be written down, and a hardcoded list stops running new sweeps in silence.
#
# Volume waves belong to scripts/run_volume_sweep_shard.sh. The two partitions come from
# the same lister, so every tracked sweep is claimed by exactly one runner.
#
# Usage:
#   scripts/run_sweep_oracle_matrix.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/cargo_runner.sh
vyre_select_cargo_runner

mapfile -t ROSTER < <(python3 scripts/lib/sweep_targets.py "$ROOT" matrix)
if ((${#ROSTER[@]} == 0)); then
    echo "Fix: the sweep roster is empty; scripts/lib/sweep_targets.py found no matrix targets." >&2
    exit 1
fi

# One cargo invocation per crate, with the union of the required-features its own sweep
# targets declare. Cargo refuses a `--test` whose required-features are unmet, so the
# union is the smallest feature set that can build the crate's whole partition at once.
CRATES=()
declare -A CRATE_TARGETS=()
declare -A CRATE_FEATURES=()
for row in "${ROSTER[@]}"; do
    IFS=$'\t' read -r crate target features <<< "$row"
    if [[ -z "${CRATE_TARGETS[$crate]+set}" ]]; then
        CRATES+=("$crate")
        CRATE_TARGETS[$crate]=""
        CRATE_FEATURES[$crate]=""
    fi
    CRATE_TARGETS[$crate]+="$target "
    [[ -n "$features" ]] && CRATE_FEATURES[$crate]+="$features,"
done

echo "sweep oracle matrices: ${#ROSTER[@]} target(s) across ${#CRATES[@]} crate(s)"

total=0
for crate in "${CRATES[@]}"; do
    args=()
    count=0
    for target in ${CRATE_TARGETS[$crate]}; do
        args+=(--test "$target")
        count=$((count + 1))
    done
    features="$(tr ',' '\n' <<< "${CRATE_FEATURES[$crate]}" | sed '/^$/d' | sort -u | paste -sd,)"
    echo
    if [[ -n "$features" ]]; then
        echo "▶ $crate: $count target(s), --features $features"
        "$CARGO_RUNNER" test -p "$crate" --features "$features" "${args[@]}"
    else
        echo "▶ $crate: $count target(s), default features"
        "$CARGO_RUNNER" test -p "$crate" "${args[@]}"
    fi
    total=$((total + count))
done

echo
echo "All $total tracked sweep_* oracle matrix integration test(s) passed."
