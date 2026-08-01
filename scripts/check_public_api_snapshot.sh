#!/usr/bin/env bash
# Public-API stability gate.
#
# Extracts every `pub` item from each published crate's src/ tree and
# diffs against docs/public-api/<crate>.txt. Any drift requires
# --refresh + a matching CHANGELOG entry.
#
# HAZARD, READ BEFORE REFRESHING IN A SHARED WORKTREE. A refresh reads the
# tree as it exists at that instant, so an unscoped refresh installs EVERY
# crate's current surface, including surface from other people's in-flight
# work that nobody has reviewed. That turns the gate into a rubber stamp
# whose scope is "whatever happened to be on disk". Two things guard it:
# a refresh always prints the diff it is about to install, per crate, and
# you can scope it to one crate. Name your crate, and read what it prints
# before you accept it. A crate you did not touch showing a diff means you
# are about to bless somebody else's surface.
#
# Usage:
#   scripts/check_public_api_snapshot.sh                       # verify
#   scripts/check_public_api_snapshot.sh --refresh <crate>     # regenerate one
#   scripts/check_public_api_snapshot.sh --refresh             # regenerate all

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Guarded: --refresh WRITES snapshots, so a silent cd failure would rewrite
# whatever tree the caller happened to be standing in.
cd "$ROOT" || exit 2

SNAPSHOT_DIR="docs/public-api"
mkdir -p "$SNAPSHOT_DIR"

PUBLISHED_CRATES=(
    "vyre-core:vyre"
    "vyre-driver:vyre-driver"
    "vyre-driver-wgpu:vyre-driver-wgpu"
    "vyre-foundation:vyre-foundation"
    "vyre-primitives:vyre-primitives"
    "vyre-spec:vyre-spec"
)

extract_api() {
    local src_dir="$1"
    # LC_ALL=C is LOAD-BEARING, not tidiness. Without it `sort` collates under
    # the caller's locale, so the snapshot's line order becomes a function of
    # the environment rather than of the tree. The orders genuinely differ:
    # byte order puts `:` (0x3A) before `_` (0x5F), so C collation emits
    # `HOT_PATH_COST_SCALE:` before `HOT_PATH_COST_SCALE_BPS` while a
    # locale-aware collation weights punctuation differently and swaps them.
    # Unpinned, the gate reported drift on a pure reordering with zero surface
    # change, and it failed in opposite environments depending on which locale
    # last refreshed it. Pinning makes the snapshot a function of the tree
    # alone. Changing this line rewrites all six snapshots.
    grep -rhE '^[[:space:]]*pub[[:space:]]+(fn|struct|enum|trait|const|static|type|mod|use)[[:space:]]' \
        "$src_dir" 2>/dev/null \
        | grep -vE '^[[:space:]]*pub[[:space:]]+use[[:space:]]' \
        | sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' \
        | LC_ALL=C sort -u
}

refresh=0
only_crate=""
if [[ "${1:-}" == "--refresh" ]]; then
    refresh=1
    only_crate="${2:-}"
    if [[ "$only_crate" == -* ]]; then
        echo "Fix: --refresh takes an optional crate name, not '$only_crate'." >&2
        exit 2
    fi
elif [[ -n "${1:-}" ]]; then
    echo "Fix: unknown argument '$1'. Usage: $0 [--refresh [crate]]" >&2
    exit 2
fi

# Crate names accepted by --refresh: either the directory or the snapshot name,
# since those differ for vyre-core (snapshot `vyre`).
if [[ -n "$only_crate" ]]; then
    matched=0
    for entry in "${PUBLISHED_CRATES[@]}"; do
        if [[ "$only_crate" == "${entry%:*}" || "$only_crate" == "${entry#*:}" ]]; then
            matched=1
            break
        fi
    done
    if [[ "$matched" -eq 0 ]]; then
        echo "Fix: '$only_crate' is not a snapshotted crate. Known: ${PUBLISHED_CRATES[*]}" >&2
        exit 2
    fi
fi

refreshed_any=0
failed=0
for entry in "${PUBLISHED_CRATES[@]}"; do
    crate_dir="${entry%:*}"
    crate_name="${entry#*:}"
    src="$crate_dir/src"
    snap="$SNAPSHOT_DIR/${crate_name}.txt"

    [[ ! -d "$src" ]] && continue

    current="$(extract_api "$src")"
    [[ -z "$current" ]] && continue

    if [[ "$refresh" -eq 1 ]]; then
        if [[ -n "$only_crate" && "$only_crate" != "$crate_dir" && "$only_crate" != "$crate_name" ]]; then
            continue
        fi

        # Show the change before installing it. An unreviewed diff here is the
        # whole hazard: printing it is what makes an unintended bless visible.
        if [[ -f "$snap" ]]; then
            pending="$(mktemp)"
            printf '%s\n' "$current" > "$pending"
            added="$(diff "$snap" "$pending" | grep -c '^>')"
            removed="$(diff "$snap" "$pending" | grep -c '^<')"
            if [[ "$added" -eq 0 && "$removed" -eq 0 ]]; then
                echo "unchanged: $crate_name"
                rm -f "$pending"
                continue
            fi
            echo "refreshing: $crate_name  +$added -$removed"
            diff "$snap" "$pending" | sed -n 's/^\([<>]\) /  \1 /p' | head -40
            if [[ $((added + removed)) -gt 40 ]]; then
                echo "  ... $((added + removed - 40)) more changed lines"
            fi
            rm -f "$pending"
        else
            echo "refreshing: $crate_name  (new snapshot, $(printf '%s\n' "$current" | wc -l) items)"
        fi

        printf '%s\n' "$current" > "$snap"
        refreshed_any=1
        continue
    fi

    if [[ ! -f "$snap" ]]; then
        echo "MISSING SNAPSHOT: $snap. Fix: run --refresh AND bump the crate version." >&2
        failed=1
        continue
    fi

    expected="$(cat "$snap")"
    if [[ "$current" != "$expected" ]]; then
        echo "PUBLIC-API DRIFT: $crate_name" >&2
        diff <(echo "$expected") <(echo "$current") | head -20 >&2
        echo "Fix: refresh snapshot AND add CHANGELOG entry in the same commit." >&2
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    exit 1
fi
if [[ "$refresh" -eq 0 ]]; then
    echo "Public API: all crates byte-stable."
elif [[ "$refreshed_any" -eq 0 ]]; then
    echo "Public API: nothing to refresh."
elif [[ -z "$only_crate" ]]; then
    echo "Refreshed every crate above. If one you did not touch changed, you just blessed someone else's surface: scope it with --refresh <crate>."
fi
exit 0
