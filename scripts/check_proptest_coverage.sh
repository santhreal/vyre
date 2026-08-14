#!/usr/bin/env bash
# Proptest coverage ratchet.
#
# Counts tracked `*.rs` files that import proptest and enforces a floor.
# Property tests are first-class regression coverage: they are the cheapest
# way to expose IR, wire-format and optimizer invariants at scale, and the
# count must not silently shrink.
#
# The gate passes when coverage holds or grows, and fails only when it
# shrinks. An earlier version also failed when `count > FLOOR`, demanding a
# manual floor bump, which made the gate punish the improvement it exists to
# encourage. That is why it was never wired into CI. A ratchet is a lower
# bound, not an equality.
#
# The floor is raised deliberately, in a commit that says why. It is never
# lowered to match a deletion: restore the test instead.
#
# Usage:
#   scripts/check_proptest_coverage.sh           # enforce
#   scripts/check_proptest_coverage.sh --report  # print current count

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Measured floor. 181 tracked files on 2026-08-12.
FLOOR=181
# Stretch target tracked for the 0.7 release.
TARGET=200

mode="${1:-enforce}"

# Tracked files only. A build artifact under target/ is not coverage, and
# neither is a sibling worktree that happens to sit inside the tree.
# `grep -l` exits non-zero for a file with no match, so xargs exits 123; the
# pipeline status is deliberately discarded rather than tripping `set -e`.
count=$( { git ls-files '*.rs' \
    | xargs -r grep -lE 'proptest!|use proptest|proptest::|extern crate proptest' 2>/dev/null \
    || true; } | wc -l | tr -d ' ')

if [[ "$mode" == "--report" ]]; then
    echo "proptest-coverage: $count files import proptest (floor=$FLOOR, target=$TARGET)"
    exit 0
fi

if (( count < FLOOR )); then
    echo "proptest-coverage gate: $count files against a floor of $FLOOR." >&2
    echo "Fix: a property test was deleted. Restore it. Lower FLOOR only" >&2
    echo "with a stated reason for why the coverage is no longer needed." >&2
    exit 1
fi

if (( count > FLOOR )); then
    echo "proptest-coverage: $count files, $(( count - FLOOR )) above the floor of $FLOOR (target=$TARGET)."
    echo "Raise FLOOR to $count to lock the gain."
    exit 0
fi

echo "proptest-coverage: $count files at the floor (target=$TARGET)."
exit 0
