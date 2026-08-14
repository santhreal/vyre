#!/usr/bin/env bash
# One owner for the question "which tracked Rust sources match this pattern".
#
# Five gates each carried their own ripgrep invocation ending in
# `2>/dev/null || true`. That construct reports an empty result whenever ripgrep
# is absent or lacks a compiled-in feature, and an empty result is exactly what
# these gates read as a clean tree. check_no_hot_path_inventory.sh passed on
# every possible tree for that reason: it asked for -P, this ripgrep build has no
# PCRE2, every invocation errored, and the error went to /dev/null.
#
# git and grep are present wherever the repository is checked out, so neither can
# fail open the way an optional binary does. A search that genuinely fails is now
# fatal instead of being indistinguishable from success.
#
# Tracked files only. A count taken over whatever is on disk can be moved by
# untracked scratch, which makes a ratchet disagree between a dev tree and CI.

# vyre_scan_tracked <ere> <exclude-path-ere|""> <path>...
#
# Prints one `path:line:content` row per match and nothing when there are none.
# Returns 2, which is fatal under `set -e`, when a scan path does not exist or
# the search itself fails. A missing scan path is a defect in the rule: a rule
# whose subject has moved measures nothing while still reporting success.
vyre_scan_tracked() {
    local pattern="$1" exclude="$2"
    shift 2

    local p
    for p in "$@"; do
        if [[ ! -e "$p" ]]; then
            printf 'source-scan: scan path does not exist: %s\n' "$p" >&2
            printf 'Fix: repoint the rule at the path the code moved to, or delete\n' >&2
            printf '     the rule. A rule scanning nothing reports success forever.\n' >&2
            return 2
        fi
    done

    local files status=0
    files="$(git ls-files -- "$@" | grep -E '\.rs$')" || status=$?
    if (( status > 1 )); then
        printf 'source-scan: listing tracked files failed (status %d)\n' "$status" >&2
        return 2
    fi
    [[ -n "$files" ]] || return 0

    if [[ -n "$exclude" ]]; then
        files="$(printf '%s\n' "$files" | grep -Ev "$exclude")" || true
        [[ -n "$files" ]] || return 0
    fi
    # A tracked file can be absent from disk while a parallel edit is in flight.
    # A fresh CI checkout never has one, so this is a dev-tree condition rather
    # than a rule failure, but it is reported: a file dropped from the scan
    # without a word is how a gate quietly stops covering what it names.
    local present absent=0 f
    present=""
    while IFS= read -r f; do
        if [[ -f "$f" ]]; then
            present+="$f"$'\n'
        else
            absent=$(( absent + 1 ))
        fi
    done <<< "$files"
    if (( absent > 0 )); then
        printf 'source-scan: %d tracked file(s) absent from the working tree, not scanned.\n' "$absent" >&2
    fi
    files="${present%$'\n'}"
    [[ -n "$files" ]] || return 0


    local out
    status=0
    out="$(printf '%s\n' "$files" | xargs -d '\n' -r grep -EnH -- "$pattern")" || status=$?
    # grep exits 1 when a file has no match, and xargs exits 123 when one of its
    # grep invocations did. Every other status is a real failure and must not be
    # mistaken for a clean tree, which is the bug this file exists to remove.
    if (( status != 0 && status != 1 && status != 123 )); then
        printf 'source-scan: search failed (status %d)\n' "$status" >&2
        return 2
    fi

    [[ -n "$out" ]] || return 0
    printf '%s\n' "$out"
}

# Rows produced by vyre_scan_tracked, counted. Empty input counts as zero rather
# than as the one line `wc -l` sees in an empty string.
vyre_scan_count() {
    local rows="$1"
    if [[ -z "$rows" ]]; then
        printf '0'
    else
        printf '%s\n' "$rows" | wc -l | tr -d ' '
    fi
}
