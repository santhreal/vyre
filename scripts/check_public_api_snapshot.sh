#!/usr/bin/env bash
# Public-API stability gate.
#
# Uses rustdoc through `cargo public-api` to snapshot the externally reachable
# API of every publishable workspace crate, including modules and reexports,
# then diffs it against docs/public-api/<package>.txt.
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

if ! inventory_output="$(
    PYTHONDONTWRITEBYTECODE=1 python3 scripts/public_api_snapshot_inventory.py "$ROOT"
)"; then
    exit 2
fi
mapfile -t PUBLISHED_CRATES <<< "$inventory_output"
if [[ "${#PUBLISHED_CRATES[@]}" -eq 0 ]]; then
    echo "Fix: public API inventory found no publishable workspace crates." >&2
    exit 2
fi

extract_api() {
    local crate_name="$1"
    local current

    if ! current="$(cargo public-api -sss -p "$crate_name")"; then
        echo "Fix: cargo public-api could not extract the $crate_name surface." >&2
        return 1
    fi

    # LC_ALL=C is LOAD-BEARING, not tidiness. Without it `sort` collates under
    # the caller's locale, so the snapshot's line order becomes a function of
    # the environment rather than of rustdoc's public surface. Pinning makes
    # the snapshot a function of the tree alone.
    printf '%s\n' "$current" | LC_ALL=C sort -u
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

# The snapshot directory and publishable manifest inventory are one set. A
# stale snapshot is as misleading as a missing one: it implies a stability
# promise for a package that no longer participates in the release train.
declare -A expected_snapshots=()
for entry in "${PUBLISHED_CRATES[@]}"; do
    expected_snapshots["${entry#*:}"]=1
done
for snap in "$SNAPSHOT_DIR"/*.txt; do
    [[ -e "$snap" ]] || continue
    snapshot_name="$(basename "$snap" .txt)"
    if [[ -z "${expected_snapshots[$snapshot_name]+x}" ]]; then
        echo "UNOWNED SNAPSHOT: $snap. Fix: remove it or restore a publishable workspace package with package.name '$snapshot_name'." >&2
        failed=1
    fi
done
for entry in "${PUBLISHED_CRATES[@]}"; do
    crate_dir="${entry%:*}"
    crate_name="${entry#*:}"
    src="$crate_dir/src"
    snap="$SNAPSHOT_DIR/${crate_name}.txt"

    [[ ! -d "$src" ]] && continue
    if [[ "$refresh" -eq 1 && -n "$only_crate" && "$only_crate" != "$crate_dir" && "$only_crate" != "$crate_name" ]]; then
        continue
    fi


    if ! current="$(extract_api "$crate_name")"; then
        failed=1
        continue
    fi
    [[ -z "$current" ]] && continue

    if [[ "$refresh" -eq 1 ]]; then

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
