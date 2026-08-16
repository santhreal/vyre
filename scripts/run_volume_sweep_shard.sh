#!/usr/bin/env bash
# Run one shard of the volume-wave oracle matrices (16k cases each).
#
# The class this closes: an unrun volume wave. `ci.yml` runs the workspace suite with
# default features, so a wave whose `[[test]] required-features` name a non-default
# feature never executes there. The waves are sharded because one host running all of
# them serially is longer than any gate budget, not because any wave is optional.
#
# The roster comes from scripts/lib/sweep_targets.py, which derives it from tracked
# sources and each crate's manifest. This script used to iterate a hardcoded three-crate
# list, so the tracked wave in a fourth crate was in no shard and ran nowhere, and it
# used a hardcoded feature string for one crate. A shard index outside the shard count
# also selected nothing and exited 0, which is the same silence.
#
# Usage:
#   scripts/run_volume_sweep_shard.sh [shard_index] [shard_count]
#   VYRE_VOLUME_SHARD=0 VYRE_VOLUME_SHARDS=8 scripts/run_volume_sweep_shard.sh
#
# Default: shard 0 of 4. Running indices 0 through count-1 covers every wave exactly once.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SHARD="${1:-${VYRE_VOLUME_SHARD:-0}}"
SHARDS="${2:-${VYRE_VOLUME_SHARDS:-4}}"

for value in "$SHARD" "$SHARDS"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "Fix: shard index and shard count must be non-negative integers, got '$SHARD' of '$SHARDS'." >&2
        exit 1
    fi
done
if ((SHARDS == 0)); then
    echo "Fix: shard count must be at least 1." >&2
    exit 1
fi
if ((SHARD >= SHARDS)); then
    echo "Fix: shard index $SHARD is outside shard count $SHARDS; use 0 through $((SHARDS - 1)). A shard that selects no target proves nothing." >&2
    exit 1
fi

mapfile -t ROSTER < <(python3 scripts/lib/sweep_targets.py "$ROOT" volume)
if ((${#ROSTER[@]} == 0)); then
    echo "Fix: the volume roster is empty; scripts/lib/sweep_targets.py found no volume targets." >&2
    exit 1
fi
if ((SHARDS > ${#ROSTER[@]})); then
    echo "Fix: shard count $SHARDS exceeds the ${#ROSTER[@]} volume target(s); the highest shards would run nothing." >&2
    exit 1
fi

# Round-robin over the roster, then group by crate so each cargo invocation gets the
# union of the required-features its selected waves declare. Passing a wave to the wrong
# crate fails with "no test target", so the crate stays attached to its target.
CRATES=()
declare -A CRATE_TARGETS=()
declare -A CRATE_FEATURES=()
selected=0
for index in "${!ROSTER[@]}"; do
    ((index % SHARDS == SHARD)) || continue
    IFS=$'\t' read -r crate target features <<< "${ROSTER[$index]}"
    if [[ -z "${CRATE_TARGETS[$crate]+set}" ]]; then
        CRATES+=("$crate")
        CRATE_TARGETS[$crate]=""
        CRATE_FEATURES[$crate]=""
    fi
    CRATE_TARGETS[$crate]+="$target "
    [[ -n "$features" ]] && CRATE_FEATURES[$crate]+="$features,"
    selected=$((selected + 1))
done

if ((selected == 0)); then
    echo "Fix: shard $SHARD of $SHARDS selected none of the ${#ROSTER[@]} volume target(s)." >&2
    exit 1
fi

echo "volume shard ${SHARD}/${SHARDS}: $selected of ${#ROSTER[@]} target(s) across ${#CRATES[@]} crate(s)"

for crate in "${CRATES[@]}"; do
    args=()
    count=0
    for target in ${CRATE_TARGETS[$crate]}; do
        args+=(--test "$target")
        count=$((count + 1))
    done
    features="$(tr ',' '\n' <<< "${CRATE_FEATURES[$crate]}" | sed '/^$/d' | sort -u | paste -sd,)"
    if [[ -n "$features" ]]; then
        echo "  $crate: $count target(s), --features $features"
        ./cargo_full test -p "$crate" --features "$features" "${args[@]}" -q
    else
        echo "  $crate: $count target(s), default features"
        ./cargo_full test -p "$crate" "${args[@]}" -q
    fi
done

echo
echo "Volume shard ${SHARD}/${SHARDS} passed $selected target(s)."
