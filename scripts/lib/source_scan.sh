#!/usr/bin/env bash
# One owner for the question "which tracked files match this pattern".
#
# Nine gates each carried their own ripgrep invocation. Most ended in
# `2>/dev/null || true`, which turns a failed search into an empty result, and an
# empty result is exactly what those gates read as a clean tree.
# check_no_hot_path_inventory.sh passed on every possible tree for that reason: it
# asked for -P, this ripgrep build has no PCRE2, every invocation errored, and the
# error went to /dev/null. Others used `if ! rg -q ...; then continue`, where a
# failed search skips the file instead of scanning it.
#
# git and grep exist wherever the repository is checked out, so neither can fail
# open the way an optional binary does. A search that genuinely fails is fatal.
#
# Tracked files only. A count taken over whatever is on disk can be moved by
# untracked scratch, which makes a ratchet disagree between a dev tree and CI.

# _vyre_scan <grep-mode> <pattern> <exclude-path-ere|""> <path>...
# grep-mode is -E for a regex or -F for a literal.
_vyre_scan() {
    local mode="$1" pattern="$2" exclude="$3"
    shift 3

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
    local present="" absent=0 f
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
    out="$(printf '%s\n' "$files" | xargs -d '\n' -r grep "$mode" -nH -- "$pattern")" || status=$?
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

# vyre_scan_tracked <ere> <exclude-path-ere|""> <path>...
# Prints one `path:line:content` row per match, nothing when there are none.
# Returns 2, fatal under `set -e`, when a scan path is missing or the search fails.
vyre_scan_tracked() { _vyre_scan -E "$@"; }

# vyre_scan_tracked_fixed <literal> <exclude-path-ere|""> <path>...
# As above, for a literal needle containing regex metacharacters.
vyre_scan_tracked_fixed() { _vyre_scan -F "$@"; }

# Rows produced by a scan, counted. Empty input counts as zero rather than as the
# one line `wc -l` sees in an empty string.
vyre_scan_count() {
    local rows="$1"
    if [[ -z "$rows" ]]; then
        printf '0'
    else
        printf '%s\n' "$rows" | wc -l | tr -d ' '
    fi
}

# vyre_file_has <ere> <file>  /  vyre_file_has_fixed <literal> <file>
# Returns 0 on a match and 1 on none, for any file type, tracked or not.
#
# A missing file or a failed search EXITS the calling script rather than
# returning, because `set -e` does not fire inside an `if` condition. Returning a
# status here would let `if ! vyre_file_has ...` read a broken search as "no
# match", which is the precise shape of the bug this file removes.
_vyre_file_has() {
    local mode="$1" pattern="$2" file="$3"
    if [[ ! -f "$file" ]]; then
        printf 'source-scan: file does not exist: %s\n' "$file" >&2
        printf 'Fix: repoint the rule, or delete it if its subject is gone.\n' >&2
        exit 2
    fi
    local status=0
    grep -q "$mode" -- "$pattern" "$file" || status=$?
    if (( status > 1 )); then
        printf 'source-scan: search failed on %s (status %d)\n' "$file" "$status" >&2
        exit 2
    fi
    return "$status"
}

vyre_file_has() { _vyre_file_has -E "$@"; }
vyre_file_has_fixed() { _vyre_file_has -F "$@"; }
