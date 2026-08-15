#!/usr/bin/env bash
# Law: inventory::iter is forbidden on the dispatch hot path.
#
# Inventory registrations are link-time metadata. Consuming them per dispatch
# means walking a linked list of static items, which blows the hot path's
# allocation and cache invariants. Every registry has a frozen-after-init
# `OnceLock<FrozenIndex>` that serves lookups in sub-ns. If this script fails,
# the hot path just regressed.
#
# See docs/inventory-contract.md §"Hot-path prohibition".

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
source scripts/lib/source_scan.sh

# Production trees where a per-dispatch registry walk would land. Four of these
# were single files that later became directories during the module splits, and
# `vyre-driver-wgpu/src/pipeline{,.rs}` needs both spellings because a pathspec
# naming the directory does not match the sibling file.
forbidden_paths=(
    "vyre-driver/src/backend"
    "vyre-driver/src/pipeline"
    "vyre-driver-wgpu/src/async_dispatch.rs"
    "vyre-driver-wgpu/src/engine"
    "vyre-driver-wgpu/src/lib.rs"
    "vyre-driver-wgpu/src/pipeline/mod.rs"
    "vyre-driver-wgpu/src/pipeline"
    "vyre-driver-wgpu/src/runtime"
    "vyre-driver-cuda/src"
    "vyre-driver-spirv/src"
    "vyre-runtime/src"
)

# Files that are legitimately init-only, exempt from the hot-path ban. Each must
# document in-file why inventory::iter is acceptable there.
allowlist_regex='vyre-driver/src/registry/(registry|migration)\.rs|vyre-driver/src/backend/(dialect_supported_ops|registry|registry/inventory_streams|registry/acquire)\.rs|vyre-foundation/src/optimizer\.rs'

# Inline test modules under these trees are not the dispatch path.
exclude_paths='/tests?/'

# The real call syntax only, and not a line that begins as a comment: prose and
# doc comments reference the symbol legitimately.
needle='^[[:space:]]*[^/]*inventory::iter::<'

hits="$(vyre_scan_tracked "$needle" "$exclude_paths" "${forbidden_paths[@]}")"

exit_code=0
if [[ -n "$hits" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        if grep -qE "$allowlist_regex" <<< "$line"; then
            continue
        fi
        exit_code=1
        echo "Hot-path inventory::iter detected:" >&2
        echo "  $line" >&2
    done <<< "$hits"
fi

if [[ "$exit_code" -ne 0 ]]; then
    echo "" >&2
    echo "Fix: route the lookup through the registry's frozen OnceLock." >&2
    echo "If the site is init-only, add it to the allowlist in this script AND" >&2
    echo "document the invariant in a nearby // HOT-PATH-OK: comment." >&2
    exit 1
fi

echo "hot-path inventory gate: no per-dispatch inventory::iter across ${#forbidden_paths[@]} production trees."
exit 0
