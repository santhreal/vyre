#!/usr/bin/env bash
# Production callers must prefer borrowed dispatch.
#
# `VyreBackend::dispatch(&[Vec<u8>])` stays as compatibility surface, but hot
# production and conformance paths call `dispatch_borrowed` so a backend with
# clone-free staging is not forced through owned row APIs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/source_scan.sh

SCAN_PATHS=(
    "vyre-libs/src"
    "vyre-runtime/src"
    "conform/vyre-conform/src"
)
EXCLUDE='/tests?/'
PATTERN='\.dispatch\('

hits="$(vyre_scan_tracked "$PATTERN" "$EXCLUDE" "${SCAN_PATHS[@]}")"

if [[ -n "$hits" ]]; then
    echo "owned dispatch on a hot path:" >&2
    printf '%s\n' "$hits" >&2
    echo "" >&2
    echo "Fix: build borrowed rows with inputs.iter().map(Vec::as_slice) and call" >&2
    echo "dispatch_borrowed." >&2
    exit 1
fi

echo "owned dispatch gate: no .dispatch( call on the scanned production paths."
exit 0
