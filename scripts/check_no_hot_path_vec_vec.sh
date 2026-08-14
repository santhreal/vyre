#!/usr/bin/env bash
# `Vec<Vec<u8>>` output and IO handles on hot dispatch surfaces.
#
# Byte rows belong in contiguous buffers, scratch arenas, borrowed slices
# (`&[&[u8]]`), or pools. This gate prevents backsliding: new occurrences in
# vyre-driver-wgpu production sources cannot appear without an audit.
#
# Default mode ratchets a ceiling. `--strict` demands every match be doc-only.

set -euo pipefail

STRICT=false
[[ "${1:-}" == "--strict" ]] && STRICT=true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/source_scan.sh

SCAN_PATHS=( "vyre-driver-wgpu/src" )
EXCLUDE='/tests?/'
PATTERN='Vec<Vec<u8>>'

# Measured on the tree that introduced this owner. A ceiling far above the real
# count permits a silent regression up to the slack, which is how the previous
# value of 35 sat above an actual 24.
CEILING=24

hits="$(vyre_scan_tracked "$PATTERN" "$EXCLUDE" "${SCAN_PATHS[@]}")"
count="$(vyre_scan_count "$hits")"

echo "Vec<Vec<u8>> scan: $count occurrences (ceiling=$CEILING, strict=$STRICT)."

if [[ "$STRICT" == true ]]; then
  exit_code=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    content="${line#*:}"
    content="${content#*:}"
    # A doc comment naming the type is fine; live code is a release blocker.
    if grep -qE '^[[:space:]]*(//|/\*)' <<< "$content"; then
      continue
    fi
    exit_code=1
    echo "Vec<Vec<u8>> outside doc-only allowance:" >&2
    echo "  $line" >&2
  done <<< "$hits"

  if [[ "$exit_code" -ne 0 ]]; then
    echo "" >&2
    echo "Fix: migrate to borrowed row handles, one flat buffer plus offsets, or arena-backed rows." >&2
    exit 1
  fi
  exit 0
fi

if [[ "$count" -gt "$CEILING" ]]; then
  echo "(ratchet) Regression: $count occurrences exceed ceiling=$CEILING." >&2
  printf '%s\n' "$hits" >&2
  echo "Fix: remove the new nested-Vec output paths, or raise the ceiling with an audit." >&2
  exit 1
fi

# A ceiling above the measured count is slack a regression can hide in.
if [[ "$count" -lt "$CEILING" ]]; then
  echo "(ratchet) $count occurrences is below ceiling=$CEILING." >&2
  echo "Fix: lower CEILING to $count in this script so the next regression is caught." >&2
  exit 1
fi

exit 0
