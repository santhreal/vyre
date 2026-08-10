#!/usr/bin/env bash
# Enforce that public documentation never links a target a reader cannot open.
#
# `docs/DOCS.toml` defines the published active set. This gate checks every
# outbound Markdown link from that set plus generated navigation. Archived and
# superseded pages are lifecycle records and are not current claims.
#
# Three violation classes, in descending severity:
#
#   OUTSIDE-REPO  the link escapes the repository root, so it cannot resolve for
#                 any clone. docs/TESTING_PROGRAM.md pointed five levels above
#                 the root into a private operator tree.
#   MISSING       the target does not exist on disk, for anyone.
#   GITIGNORED    the target exists here and is excluded from the repository, so
#                 the link works for the author and fails for every other reader.
#                 This is the same defect the documentation manifest prevents
#                 for generated and navigable pages.
#
# Anchor fragments are deliberately out of scope. Checking whether #some-heading
# exists inside a target is a much weaker signal than whether the FILE exists,
# and folding it in would bury the two classes that matter under heading churn.
#
# SCOPE IS DATA. `scripts/docs_manifest.py --list-active` emits the exact
# current/generated navigation set from `docs/DOCS.toml`. The link gate does not
# keep a second archive carve-out or filesystem-derived publication list.

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


: > "$sites"
while IFS= read -r doc; do
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
done < <(python3 scripts/docs_manifest.py --list-active)

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
