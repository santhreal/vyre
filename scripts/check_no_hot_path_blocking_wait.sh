#!/usr/bin/env bash
# Block-on, busy-waits, and polled blocking maintenance on throughput paths.
#
# Default mode ratchets a ceiling on line matches in vyre-driver-wgpu production
# sources. `--strict` fails on any match outside the reviewed prefixes, which
# permit one-shot device initialization waits and the single shared backoff helper
# but not waits scattered through the hot path.

set -euo pipefail

STRICT=false
[[ "${1:-}" == "--strict" ]] && STRICT=true

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/source_scan.sh

SCAN_PATHS=( "vyre-driver-wgpu/src" )

# wait_backoff.rs is the one sanctioned home for adaptive backoff, so it is not a
# finding. Tests and benches are not the dispatch path.
EXCLUDE='/tests?/|/benches/|vyre-driver-wgpu/src/wait_backoff\.rs'

# One alternation, all single-line.
PATTERN='Maintain::Wait|pollster::block_on|std::thread::sleep\(|std::thread::yield_now\(\)|thread::park\(\)|park_timeout'

# Measured. The two remaining sites are one-shot device acquisition waits, in
# runtime/device/device.rs and runtime/device/selector.rs.
CEILING=2

hits="$(vyre_scan_tracked "$PATTERN" "$EXCLUDE" "${SCAN_PATHS[@]}")"
count="$(vyre_scan_count "$hits")"

echo "blocking-wait scan: $count occurrences (ceiling=$CEILING, strict=$STRICT)."

if [[ "$STRICT" == true ]]; then
  # Deliberate one-shot init and teardown polling. Extend only with review.
  STRICT_ALLOW_PREFIX=(
    '^vyre-driver-wgpu/src/lib\.rs:'
    '^vyre-driver-wgpu/src/backend_impl\.rs:'
    '^vyre-driver-wgpu/src/runtime/device/device\.rs:'
    '^vyre-driver-wgpu/src/runtime/device/selector\.rs:'
  )
  exit_code=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    allowed=false
    for pref in "${STRICT_ALLOW_PREFIX[@]}"; do
      if grep -qE "$pref" <<< "$line"; then
        allowed=true
        break
      fi
    done
    if [[ "$allowed" != true ]]; then
      exit_code=1
      echo "Blocking wait on a hot-scope path:" >&2
      echo "  $line" >&2
    fi
  done <<< "$hits"

  if [[ "$exit_code" -ne 0 ]]; then
    echo "" >&2
    echo "Fix: prefer Poll, fence callbacks, or Maintain::Poll; consolidate waits; if a" >&2
    echo "site is genuinely init-only, move it under the allow-prefix list." >&2
    exit 1
  fi
  exit 0
fi

if [[ "$count" -gt "$CEILING" ]]; then
  echo "(ratchet) Regression: $count occurrences exceed ceiling=$CEILING." >&2
  printf '%s\n' "$hits" >&2
  echo "Fix: remove the blocking wait, or raise the ceiling with rationale here." >&2
  exit 1
fi

# Slack above the measured count is room a regression hides in.
if [[ "$count" -lt "$CEILING" ]]; then
  echo "(ratchet) $count occurrences is below ceiling=$CEILING." >&2
  echo "Fix: lower CEILING to $count in this script so the next regression is caught." >&2
  exit 1
fi

exit 0
