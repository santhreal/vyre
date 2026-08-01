#!/usr/bin/env bash
# Enforce that public documentation never links a target a reader cannot open.
#
# docs/INDEX.md is not the only place a broken link hurts. scripts/check_docs_index.sh
# checks the routing table; this checks every outbound markdown link in every
# public document, because the defect that motivated it lived one directory down:
# docs/archive/README.md listed fourteen sibling documents that are gitignored,
# so the file reads as a complete directory listing to us and as fourteen dead
# links to everyone else.
#
# Three violation classes, in descending severity:
#
#   OUTSIDE-REPO  the link escapes the repository root, so it cannot resolve for
#                 any clone. docs/TESTING_PROGRAM.md pointed five levels above
#                 the root into a private operator tree.
#   MISSING       the target does not exist on disk, for anyone.
#   GITIGNORED    the target exists here and is excluded from the repository, so
#                 the link works for the author and fails for every other reader.
#                 This is the same defect check_docs_index.sh catches in the
#                 index, generalised to all documents.
#
# Anchor fragments are deliberately out of scope. Checking whether #some-heading
# exists inside a target is a much weaker signal than whether the FILE exists,
# and folding it in would bury the two classes that matter under heading churn.
#
# SCOPE IS A RULE, NOT A LIST. An allowlist of exempt paths is a deferral: it
# rots the moment someone adds the next file. The rule is that a HISTORICAL
# RECORD is not gated. A document under docs/archive/ or docs/legacy/ is a
# snapshot of what was true on the date it was written, so rewriting its links to
# point at today's documents would falsify the record, and it is honest for a
# snapshot to reference documents that have since moved or become private. The
# one exception is the directory's own README.md: that is not a snapshot, it is
# the current signpost telling a reader the directory is stale and where to go
# instead, so its links are live claims and are gated like any other live
# document. This is the same carve-out check_docs_index.sh makes for the
# per-directory status rule, for the same reason.

set -euo pipefail

# sort and comm must agree on collation, so pin the whole script to byte order.
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

sites="$(mktemp)"
candidates="$(mktemp)"
ignored="$(mktemp)"
trap 'rm -f "$sites" "$candidates" "$ignored"' EXIT

# Collapse . and .. textually. realpath is not usable here: the whole point is to
# classify targets that do NOT exist, and realpath cannot normalise those.
normalize_path() {
  local input="$1" segment
  local -a parts=() out=()
  IFS='/' read -ra parts <<< "$input"
  for segment in "${parts[@]}"; do
    case "$segment" in
      '' | '.')
        continue
        ;;
      '..')
        if (( ${#out[@]} > 0 )) && [[ "${out[${#out[@]} - 1]}" != ".." ]]; then
          out=("${out[@]:0:${#out[@]} - 1}")
        else
          out+=("..")
        fi
        ;;
      *)
        out+=("$segment")
        ;;
    esac
  done
  local joined
  joined="$(IFS='/'; printf '%s' "${out[*]}")"
  printf '%s' "$joined"
}

# A historical record is a dated snapshot, not a live claim. See the header.
is_historical_record() {
  local doc="$1"
  case "$doc" in
    docs/archive/* | docs/legacy/*)
      [[ "$(basename "$doc")" == "README.md" ]] && return 1
      return 0
      ;;
  esac
  return 1
}

: > "$sites"
while IFS= read -r doc; do
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    && git check-ignore -q "$doc"; then
    continue
  fi
  is_historical_record "$doc" && continue
  base="$(dirname "$doc")"
  while IFS= read -r raw; do
    case "$raw" in
      http://* | https://* | mailto:* | '#'*)
        continue
        ;;
    esac
    target="${raw%%#*}"
    [[ -z "$target" ]] && continue
    if [[ "$target" == /* ]]; then
      # A leading slash means repository root, the way a forge renders it.
      resolved="$(normalize_path "${target#/}")"
    else
      resolved="$(normalize_path "$base/$target")"
    fi
    [[ -z "$resolved" ]] && continue
    printf '%s\t%s\t%s\n' "$doc" "$raw" "$resolved" >> "$sites"
  done < <(grep -oE '\[[^][]*\]\([^()[:space:]]+\)' "$doc" \
    | sed -E 's/^\[[^][]*\]\(//; s/\)$//')
done < <(find docs -type f -name '*.md' -print | sort)

# One batched ignore query. Targets that escape the root would make git fatal,
# so they are excluded here and reported as OUTSIDE-REPO instead.
: > "$ignored"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  cut -f3 "$sites" | grep -v '^\.\.' | sort -u > "$candidates" || true
  if [[ -s "$candidates" ]]; then
    status=0
    git check-ignore --stdin < "$candidates" > "$ignored" || status=$?
    if (( status > 1 )); then
      printf 'documentation link contract: git check-ignore failed with status %d.\n' "$status" >&2
      exit 1
    fi
  fi
fi

violations=()
while IFS=$'\t' read -r doc raw resolved; do
  if [[ "$resolved" == ..* ]]; then
    violations+=("OUTSIDE-REPO  $doc  [$raw]  escapes the repository root")
  elif [[ ! -e "$resolved" ]]; then
    violations+=("MISSING       $doc  [$raw]  no such path: $resolved")
  elif grep -qxF "$resolved" "$ignored"; then
    violations+=("GITIGNORED    $doc  [$raw]  $resolved is excluded from the repository")
  fi
done < "$sites"

if (( ${#violations[@]} > 0 )); then
  printf 'documentation link contract failed.\n' >&2
  printf '%s\n' "${violations[@]}" >&2
  printf '\nFix: repoint the link at a published document, or drop the whole claim, or state inline that the referenced document is not published. Deleting the link syntax and leaving the sentence promising it is not a fix.\n' >&2
  exit 1
fi

printf 'documentation link contract: %d links in public documents all resolve to published paths.\n' "$(wc -l < "$sites")"
