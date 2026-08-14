#!/usr/bin/env bash
# Backend extension contract: a new backend is one crate plus inventory submits.
#
# The core driver must expose inventory collections and frozen registry views;
# concrete backend crates must own their implementation, depend on
# `vyre-driver`, and submit BackendRegistration, BackendPrecedence, and
# BackendCapability from their own crate. The core registry must not contain a
# hand-maintained list of concrete backend ids.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/source_scan.sh

failures=0

fail() {
    echo "backend-extension-contract: $1" >&2
    failures=$((failures + 1))
}

require_grep() {
    local pattern="$1"
    local path="$2"
    local message="$3"
    if ! grep -Eq "$pattern" "$path"; then
        fail "$message"
    fi
}

# Directory-recursive presence check. This used `rg -q`, which made every one of
# the 35 per-crate assertions below depend on an optional binary: absent ripgrep
# failed all of them at once, for the wrong reason.
require_tree() {
    local pattern="$1"
    local path="$2"
    local message="$3"
    if [[ -z "$(vyre_scan_tracked "$pattern" "" "$path")" ]]; then
        fail "$message"
    fi
}

inventory_file="vyre-driver/src/backend/registry/inventory_streams.rs"
acquire_file="vyre-driver/src/backend/registry/acquire.rs"

require_grep 'inventory::collect!\(BackendRegistration\);' "$inventory_file" \
    "BackendRegistration must be an inventory collection in $inventory_file"
require_grep 'inventory::collect!\(BackendPrecedence\);' "$inventory_file" \
    "BackendPrecedence must be an inventory collection in $inventory_file"
require_grep 'inventory::collect!\(BackendCapability\);' "$inventory_file" \
    "BackendCapability must be an inventory collection in $inventory_file"
require_grep 'LazyLock<Result<BackendRegistry, BackendError>>' "$inventory_file" \
    "registered_backends must freeze inventory through one fallible process-wide registry"
require_grep 'inventory::iter::<BackendRegistration>' "$inventory_file" \
    "registered_backends must be populated from inventory::iter::<BackendRegistration>"
require_grep 'registered_backends_by_precedence_slice' "$acquire_file" \
    "backend acquisition must route through the precedence-sorted frozen slice"
require_grep 'backend_dispatches' "$acquire_file" \
    "preferred backend acquisition must consult BackendCapability dispatch metadata"

# A failed search here used to mean "no hardcoded ids found", because the result
# was read as the `if` condition. It also wrote its findings to /tmp, so a gate
# left a file outside the repository behind.
hardcoded="$(vyre_scan_tracked '"(cuda|wgpu|spirv|metal|dxil)"' "" vyre-driver/src/backend/registry)"
if [[ -n "$hardcoded" ]]; then
    fail "core backend registry contains concrete backend id literals; a new backend must not require editing vyre-driver/src/backend/registry"
    printf '%s\n' "$hardcoded" | head -n 20 >&2
fi

for crate in vyre-driver-cuda vyre-driver-wgpu vyre-driver-metal vyre-driver-spirv vyre-driver-reference; do
    if [[ ! -f "$crate/Cargo.toml" ]]; then
        fail "$crate/Cargo.toml missing"
        continue
    fi
    if [[ ! -d "$crate/src" ]]; then
        fail "$crate/src missing"
        continue
    fi
    require_grep 'vyre-driver' "$crate/Cargo.toml" \
        "$crate must depend on vyre-driver instead of editing core registry code"
    require_grep 'inventory\.workspace|inventory[[:space:]]*=' "$crate/Cargo.toml" \
        "$crate must depend on inventory for link-time backend registration"
    require_tree 'impl .*VyreBackend for' "$crate/src" \
        "$crate must implement VyreBackend in its own crate"
    require_tree 'inventory::submit![[:space:]]*\{' "$crate/src" \
        "$crate must submit backend metadata through inventory::submit!"
    require_tree 'BackendRegistration[[:space:]]*\{' "$crate/src" \
        "$crate must submit BackendRegistration"
    require_tree 'BackendPrecedence[[:space:]]*\{' "$crate/src" \
        "$crate must submit BackendPrecedence"
    require_tree 'BackendCapability[[:space:]]*\{' "$crate/src" \
        "$crate must submit BackendCapability so dispatch ownership is explicit"
    require_tree 'supported_ops[[:space:]]*:' "$crate/src" \
        "$crate BackendRegistration must advertise supported_ops"
done

if (( failures > 0 )); then
    echo "backend-extension-contract gate failed with $failures violation(s)." >&2
    echo "Fix: keep backend addition as one concrete crate implementing VyreBackend and registering via inventory::submit!." >&2
    exit 1
fi

echo "backend-extension-contract gate: backend addition remains one crate + inventory::submit!."
