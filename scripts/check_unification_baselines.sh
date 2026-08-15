#!/usr/bin/env bash
# P-DELETE-10 + P-DELETE-1 + P-UNIFY-2 + P-UNIFY-4 + P-UNIFY-1b  - 
# unification baselines.
#
# Each audit row asks for a cross-crate refactor: drop a duplicate,
# unify a planning surface, lift a substrate from a backend crate up
# into the driver tier. Each one is a multi-day refactor that has to
# land alongside cross-crate API changes; doing them in a single
# session would cascade through the build.
#
# This gate locks the current state by counting the offending sites
# per audit row and ratcheting downward. Adding a new
# match-on-Node validator (P-DELETE-1), a new BufferAccess auto-
# inference helper (P-DELETE-10), a new cpu_references parallel impl
# (P-UNIFY-2), or a new fusion-planning surface (P-UNIFY-4) is a
# regression. Removing one decreases the floor.
#
# The architectural targets live in `docs/MIGRATION.md` under the
# "Future migrations" section.
#
# Usage:
#   scripts/check_unification_baselines.sh           # enforce
#   scripts/check_unification_baselines.sh --report  # print every count

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mode="${1:-enforce}"

# Each row: name@@pattern@@search_paths@@floor
#
# Every declared path MUST exist. A path that has been moved or deleted used to
# be dropped silently, so the row scanned less than it claimed and scored 0,
# which is at or below every floor. Three of these five rows were in exactly
# that state: `vyre-driver-wgpu/src/lowering`, `vyre-foundation/src/
# cpu_references.rs` and the driver-owned substrate tree had all been moved,
# and the rows reading them passed by measuring nothing.
ROWS=(
    # P-DELETE-1 used to live here as a count of `match node {` occurrences in
    # validate/ and transform/, pinned at 18 against an actual 22. It is gone
    # because it measured the wrong thing: 22 distinct traversals over one enum
    # is not duplication, and `match node {}` is the only idiom Rust offers for
    # dispatching on a variant. The property it tracked by coincidence was that
    # exactly 4 of those blocks carried a catch-all `_ =>` arm, and 22 - 4 = 18.
    #
    # Child enumeration now has one public owner,
    # vyre-foundation::visit::child_bodies, with no catch-all, so
    # adding a Node variant fails to compile there. This row asserts that
    # nothing re-implements it: a second exhaustive child match is a duplicate.
    # The other half of the property is a test, not a count, because `Node` is
    # #[non_exhaustive] and no crate outside vyre-foundation can match it
    # exhaustively: vyre-foundation/tests/node_variant_traversal_closure.rs and
    # vyre_test_support::ir_variants enumerate NODE_VARIANT_NAMES at run time and
    # fail until every declared variant has a fixture and a traversal decision.
    "P-DELETE-1__child_bodies_owner@@fn child_bodies\\b@@vyre-foundation/src@@1"
    "P-DELETE-10__buffer_access_auto@@BufferAccess::(infer|auto|derive_from)@@vyre-foundation/src/lower vyre-driver-wgpu/src vyre-runtime/src/resident_work_queue@@0"
    "P-UNIFY-2__cpu_references@@fn cpu_reference\\b@@vyre-foundation/src vyre-reference/src@@0"
    # Floor 1, not 0: the unification this row tracks is ACHIEVED. There is
    # exactly one fusion-planning entry point and it lives in
    # vyre-foundation/src/execution_plan/fusion/fuse.rs. The previous 0 was
    # never measured, because all three declared paths had moved; the row
    # scanned nothing. A second entry point anywhere fails this row.
    "P-UNIFY-4__fusion_planning@@fn (plan_fusion|fuse_programs|tensor_network_fusion_order)\\b@@vyre-foundation/src/execution_plan vyre-pass-engine/src vyre-runtime/src/resident_work_queue@@1"
    "P-UNIFY-1b__cache_in_wgpu@@impl PipelineCacheStore for@@vyre-driver-wgpu/src@@0"
)

errors=()
report=()

for row in "${ROWS[@]}"; do
    name=$(printf '%s' "$row" | awk -F'@@' '{print $1}')
    pattern=$(printf '%s' "$row" | awk -F'@@' '{print $2}')
    paths=$(printf '%s' "$row" | awk -F'@@' '{print $3}')
    floor=$(printf '%s' "$row" | awk -F'@@' '{print $4}')
    # shellcheck disable=SC2206
    path_arr=( $paths )
    missing=()
    for p in "${path_arr[@]}"; do
        [[ -e "$p" ]] || missing+=("$p")
    done
    if (( ${#missing[@]} > 0 )); then
        errors+=("$name: declared path(s) do not exist: ${missing[*]}")
        report+=("$name: UNMEASURABLE, missing ${missing[*]}")
        continue
    fi
    count=0
    if hits=$(grep -rnE "$pattern" --include='*.rs' "${path_arr[@]}" 2>/dev/null | grep -vE '/tests/|_tests\.rs:|test_fixtures' || true); then
        if [[ -n "$hits" ]]; then
            count=$(printf '%s\n' "$hits" | wc -l | tr -d ' ')
        fi
    fi
    report+=("$name: $count (floor=$floor)")
    if (( count > floor )); then
        errors+=("$name: $count exceeds floor $floor  -  ratchet violated")
    fi
done

if [[ "$mode" == "--report" ]]; then
    for r in "${report[@]}"; do echo "  $r"; done
    exit 0
fi

if (( ${#errors[@]} > 0 )); then
    echo "unification-baselines gate: ${#errors[@]} ratchets violated." >&2
    for e in "${errors[@]}"; do echo "  $e" >&2; done
    echo >&2
    echo "Fix: bring the count back to or below the floor. Each audit row" >&2
    echo "tracks a cross-crate refactor; new sites are a regression." >&2
    echo "Lowering the floor follows a real refactor  -  update the floor in" >&2
    echo "scripts/check_unification_baselines.sh in the same patch." >&2
    exit 1
fi

echo "unification-baselines gate: every ratchet at or below floor."
exit 0
