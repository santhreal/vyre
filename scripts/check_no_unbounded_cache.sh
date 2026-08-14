#!/usr/bin/env bash
# No unbounded `HashMap::new()` / `VecDeque::new()` growth footguns in dispatcher
# and tenant runtime wiring.
#
# Maps and deques reachable from compilation or megakernel hot paths need an
# explicit capacity budget, tier eviction, or a pool. A bare `new()` is the lint
# signal for none of those.
#
# Default mode ratchets a ceiling. `--strict` demands every hit be one of the
# reviewed exceptions.

set -euo pipefail

STRICT=false
[[ "${1:-}" == "--strict" ]] && STRICT=true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/source_scan.sh

SCAN_PATHS=(
  "vyre-driver-wgpu/src"
  "vyre-runtime/src"
)
EXCLUDE='/tests?/|/benches/|/fuzz/'
PATTERN='\b(HashMap|VecDeque)::new\(\)'

# The two reviewed sites, each documenting its own bound in-module.
ALLOWED='^(vyre-driver-wgpu/src/buffer/handle\.rs|vyre-runtime/src/uring/io_loop\.rs):'

CEILING=2

hits="$(vyre_scan_tracked "$PATTERN" "$EXCLUDE" "${SCAN_PATHS[@]}")"
count="$(vyre_scan_count "$hits")"

echo "unbounded-cache scan: $count occurrences (ceiling=$CEILING, strict=$STRICT)."

if [[ "$STRICT" == true ]]; then
  exit_code=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    if ! grep -qE "$ALLOWED" <<< "$line"; then
      echo "Forbidden unbounded associative container construction:" >&2
      echo "  $line" >&2
      exit_code=1
    fi
  done <<< "$hits"

  if [[ "$exit_code" -ne 0 ]]; then
    echo "" >&2
    echo "Fix: construct with a bound (capacity, eviction, pool) or move off the hot tier." >&2
    exit 1
  fi
  exit 0
fi

if [[ "$count" -gt "$CEILING" ]]; then
  echo "(ratchet) Regression: $count occurrences exceed ceiling=$CEILING." >&2
  printf '%s\n' "$hits" >&2
  echo "Fix: remove the new bare ::new sites, or raise the ceiling with rationale here." >&2
  exit 1
fi

# Slack above the measured count is room a regression hides in.
if [[ "$count" -lt "$CEILING" ]]; then
  echo "(ratchet) $count occurrences is below ceiling=$CEILING." >&2
  echo "Fix: lower CEILING to $count in this script so the next regression is caught." >&2
  exit 1
fi

exit 0
