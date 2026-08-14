#!/usr/bin/env bash
# Direct dispatch must stage ordinary outputs through the size-classed readback
# ring, not through fresh per-output MAP_READ buffers.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/source_scan.sh

record="vyre-driver-wgpu/src/engine/record_and_readback.rs"
record_modules="vyre-driver-wgpu/src/engine/record_and_readback"
arena="vyre-driver-wgpu/src/lib.rs"

# Each of these is load-bearing for routing an ordinary output through a ring
# slot. Absence means the routing was removed or renamed.
required_patterns=(
    "readback_rings:"
    "SubmittedReadback::Ring"
    ".record_copy("
    ".arm_ticket("
    ".with_mapped_ticket("
)

for pattern in "${required_patterns[@]}"; do
    if [[ -z "$(vyre_scan_tracked_fixed "$pattern" "" "$record" "$record_modules")" ]]; then
        echo "direct readback ring gate: missing '$pattern' in the record/readback modules" >&2
        echo "Fix: route ordinary output readbacks through ReadbackRing slots before" >&2
        echo "falling back to pooled staging." >&2
        exit 1
    fi
done

if ! vyre_file_has_fixed "ReadbackRingSet::new()" "$arena"; then
    echo "direct readback ring gate: the dispatch arena does not own a ReadbackRingSet" >&2
    echo "Fix: keep readback rings in the backend dispatch arena so hot dispatches" >&2
    echo "reuse staging slots." >&2
    exit 1
fi

# A third check used to live here, written as
# `for output in request.output_bindings[\s\S]*pool.*acquire`. It could never
# fire: ripgrep matches within a line unless asked otherwise, so a pattern
# spanning a loop body matched nothing on any tree it was run against.
#
# It is not replaced, because what it wanted is not a line property. The pooled
# per-output MAP_READ loop in record_and_readback/staging.rs is legitimate: it is
# the fallback branch taken when no readback ring set is supplied, and the trap
# equivalent below it is guarded the same way. The invariant is which branch the
# code sits in, and no line-based scan can see that.
#
# Two GPU tests in vyre-driver-wgpu/src/pipeline/tests/readback_ring_contracts.rs
# own it by exercising both branches:
# direct_record_and_readback_trap_uses_readback_rings_only, with rings supplied,
# and direct_record_and_readback_trap_without_readback_rings_allocates_full_sidecar_copy
# with rings absent. That is a behavioural proof, which is what this needed.

echo "direct readback ring gate: ordinary outputs stage through ring slots."
