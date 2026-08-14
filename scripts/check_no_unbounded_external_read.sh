#!/usr/bin/env bash
# Unbounded synchronous external reads (`read_to_end` over an arbitrary file) must
# not appear on dispatch-critical paths outside the approved cache modules.
#
# Artifact and tiered caches expose byte caps, truncation, and checksum length
# proofs. A bare `File::open` into `read_to_end` is a DoS amplifier once it is
# wired into a synchronous dispatch loop.
#
# This is an allow-prefix gate, not a ratchet: a new site is either one of the
# reviewed cache modules or a defect. Each allowed entry documents its cap policy
# in-module.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/source_scan.sh

SCAN_PATHS=( "vyre-driver-wgpu/src" )
EXCLUDE='/tests?/'
PATTERN='read_to_end'

ALLOW_PREFIX=(
  '^vyre-driver-wgpu/src/pipeline/disk_cache\.rs:'
)

hits="$(vyre_scan_tracked "$PATTERN" "$EXCLUDE" "${SCAN_PATHS[@]}")"

if [[ -z "$hits" ]]; then
  echo "unbounded-external-read gate: no read_to_end on the scanned surface."
  exit 0
fi

exit_code=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  allowed=false
  for pref in "${ALLOW_PREFIX[@]}"; do
    if grep -qE "$pref" <<< "$line"; then
      allowed=true
      break
    fi
  done
  if [[ "$allowed" != true ]]; then
    exit_code=1
    echo "Disallowed unbounded synchronous read-all:" >&2
    echo "  $line" >&2
  fi
done <<< "$hits"

if [[ "$exit_code" -ne 0 ]]; then
  echo "" >&2
  echo "Fix: read behind a bound (explicit max bytes, chunked read, capped mmap)," >&2
  echo "or add the module to ALLOW_PREFIX here once it documents its cap." >&2
  exit 1
fi

# An allow-prefix that matches nothing has stopped describing the tree.
for pref in "${ALLOW_PREFIX[@]}"; do
  if ! grep -qE "$pref" <<< "$hits"; then
    echo "unbounded-external-read gate: allow-prefix matches nothing: $pref" >&2
    echo "Fix: delete the stale entry. It reserves an exemption nothing uses." >&2
    exit 1
  fi
done

echo "unbounded-external-read gate: every read_to_end site is a reviewed cache module."
exit 0
