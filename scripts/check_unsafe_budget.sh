#!/usr/bin/env bash
# Every file permitted to contain unsafe code is on a reviewed list.
#
# `[workspace.lints.rust]` sets `unsafe_code = "deny"` and every member inherits
# it, so a file cannot contain unsafe without an explicit `allow(unsafe_code)`
# override. The set of files carrying that override is therefore the COMPLETE
# unsafe surface of the workspace, and rustc is the thing enforcing it.
#
# That makes the override set the budget. The earlier version of this gate
# instead grepped for `unsafe\s+(impl|fn|\{)` against a whitelist of path
# fragments, which was weaker in both directions: it matched the word in prose
# and in string literals, it missed unsafe reachable through a macro, and its
# whitelist admitted whole directories rather than files. Three of its nine
# entries named `/vyre-pipeline/`, a crate that no longer exists, so the gate
# reserved budget for nothing and nobody noticed.
#
# Additions fail because new unsafe needs a security review. Removals fail too:
# a list that still names a file which no longer carries the override overstates
# the audited surface, which is the same defect as the vyre-pipeline entries.

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

budget_file="scripts/unsafe_budget.txt"

if [[ ! -f "$budget_file" ]]; then
    echo "unsafe-budget gate: $budget_file is missing." >&2
    exit 1
fi

expected=$(grep -vE '^\s*(#|$)' "$budget_file" | sort)

# Tracked files only: an untracked scratch file is not workspace surface.
actual=$( { git ls-files '*.rs' \
    | xargs -r grep -ln 'allow(unsafe_code)' 2>/dev/null \
    || true; } | sort)

added=$(comm -13 <(printf '%s\n' "$expected") <(printf '%s\n' "$actual"))
removed=$(comm -23 <(printf '%s\n' "$expected") <(printf '%s\n' "$actual"))

if [[ -z "$added" && -z "$removed" ]]; then
    count=$(printf '%s\n' "$expected" | grep -c . || true)
    echo "unsafe-budget gate: $count files permitted to contain unsafe, all reviewed."
    exit 0
fi

echo "unsafe-budget gate: the unsafe surface no longer matches the reviewed list." >&2

if [[ -n "$added" ]]; then
    echo >&2
    echo "New unsafe surface, not yet reviewed:" >&2
    printf '  %s\n' $added >&2
    echo >&2
    echo "Fix: remove the unsafe, wrap it in a safe abstraction inside a file" >&2
    echo "     already on the list, or add the path to $budget_file after a" >&2
    echo "     security review. Every site needs a SAFETY comment naming the" >&2
    echo "     invariant its caller relies on." >&2
fi

if [[ -n "$removed" ]]; then
    echo >&2
    echo "Listed but no longer carrying allow(unsafe_code):" >&2
    printf '  %s\n' $removed >&2
    echo >&2
    echo "Fix: delete these lines from $budget_file. A stale entry reserves" >&2
    echo "     audited budget for a file that does not use it." >&2
fi

exit 1
