#!/usr/bin/env bash
# Run a shard of volume-wave oracle matrices (16k cases each) for CI/runtime validation.
#
# Usage:
#   scripts/run_volume_sweep_shard.sh [shard_index] [shard_count]
#   VYRE_VOLUME_SHARD=0 VYRE_VOLUME_SHARDS=8 scripts/run_volume_sweep_shard.sh
#
# Default: shard 0 of 4 (quarter of all sweep_*_volume_oracle_matrix targets).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/cargo_runner.sh
vyre_select_cargo_runner

SHARD="${1:-${VYRE_VOLUME_SHARD:-0}}"
SHARDS="${2:-${VYRE_VOLUME_SHARDS:-4}}"

# Targets live in three crates. Keep each one with its owning crate: passing a
# vyre-reference target to `-p vyre-primitives` fails with "no test target".
CRATE_FEATURES_vyre_primitives='cpu-parity,bitset,graph,reduce,hash,predicate,text'

ALL_TARGETS=()
shopt -s nullglob
for crate in vyre-foundation vyre-primitives vyre-reference; do
    for path in "$crate"/tests/*volume_oracle_matrix*.rs; do
        target="${path##*/}"
        ALL_TARGETS+=("$crate:${target%.rs}")
    done
done
shopt -u nullglob

if ((${#ALL_TARGETS[@]} == 0)); then
    echo "no volume oracle matrix targets found" >&2
    exit 1
fi

SELECTED=()
for i in "${!ALL_TARGETS[@]}"; do
    if (( i % SHARDS == SHARD )); then
        SELECTED+=("${ALL_TARGETS[$i]}")
    fi
done

echo "volume shard ${SHARD}/${SHARDS}: ${#SELECTED[@]} of ${#ALL_TARGETS[@]} targets"

for crate in vyre-foundation vyre-primitives vyre-reference; do
    args=()
    for entry in "${SELECTED[@]}"; do
        [[ "$entry" == "$crate:"* ]] && args+=(--test "${entry#*:}")
    done
    ((${#args[@]} == 0)) && continue
    features_var="CRATE_FEATURES_${crate//-/_}"
    features="${!features_var:-}"
    echo "  $crate: $(( ${#args[@]} / 2 )) target(s)"
    if [[ -n "$features" ]]; then
        "$CARGO_RUNNER" test -p "$crate" --features "$features" "${args[@]}" -q
    else
        "$CARGO_RUNNER" test -p "$crate" "${args[@]}" -q
    fi
done
