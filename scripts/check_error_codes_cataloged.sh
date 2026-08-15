#!/usr/bin/env bash
# Every stable V### validation code emitted by vyre source must appear in
# docs/generated/error-codes.toml, which is rendered from the rule table in
# vyre-foundation/src/validate/catalog.rs. Prose drifts; codes don't. Catching
# a code that's been added to source without a registry entry lets tooling keep
# up.
#
# Scans for V### (3-digit) tokens inside Rust string literals. Codes in the
# E-*, W-*, B-* and C-* families are owned by the crate that emits them and are
# checked by that crate's own catalog test, not here.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CATALOG="docs/generated/error-codes.toml"
if [[ ! -f "$CATALOG" ]]; then
    echo "Missing catalog: $CATALOG. Fix: run VYRE_WRITE_ERROR_CATALOG=1 ./cargo_full test -p vyre-foundation --test validator_error_docs." >&2
    exit 1
fi

# Extract codes from source (inside "..." strings only  -  grepping the
# full file would false-positive on module names like `V013.rs`).
SEARCH_DIRS=(
    vyre-foundation/src
    vyre-spec/src
    vyre-driver/src
    vyre-reference/src
    vyre-driver-wgpu/src
    vyre-driver-spirv/src
)

codes_in_source="$(
    grep -rEho '"V[0-9]{3}' \
        --include='*.rs' \
        "${SEARCH_DIRS[@]}" \
        2>/dev/null \
    | grep -oE 'V[0-9]{3}' \
    | sort -u
)"

missing=0
while IFS= read -r code; do
    if [[ -z "$code" ]]; then continue; fi
    if ! grep -Fq "code = \"${code}\"" "$CATALOG"; then
        echo "Uncataloged validation code: ${code}. Fix: add a ValidationRule to vyre-foundation/src/validate/catalog.rs and regenerate ${CATALOG}." >&2
        missing=1
    fi
done <<< "$codes_in_source"

if [[ "$missing" -ne 0 ]]; then exit 1; fi
count="$(echo "$codes_in_source" | grep -c . || true)"
echo "Error codes cataloged: ${count} validation codes verified against ${CATALOG}."
