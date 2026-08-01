#!/usr/bin/env bash
# Enforce docs/INDEX.md as the complete routing table for the public docs set.
#
# ORACLE: the filesystem plus git ignore status. NOT git's index.
#
# "This link resolves" is a statement about a file existing where the link
# points, so the question is answered by stat(2), not by whether the file has
# been committed yet. An earlier revision of this gate enumerated `git ls-files`
# and got both directions wrong at once:
#
#   * A document created but not yet committed was reported as "INDEX.md points
#     at missing files" while sitting on disk at full size. A present file is
#     not a missing file.
#   * A document deleted from disk stayed in the tracked set until the deletion
#     was committed, so the gate demanded that INDEX.md keep indexing a file
#     nobody could open, and reported it as "missing from docs/INDEX.md".
#
# Tracking state is a transient fact about the staging area. Existence and
# publishability are the facts a documentation index is about, so those are what
# this gate reads.
#
# Ignore status still comes from git, and that is deliberate rather than a
# leftover: `git check-ignore` answers "will this file ever reach another
# reader", which is a different question from "is it committed right now".
# .gitignore excludes **/*PLAN*.md, **/*STATUS*.md, **/*ROADMAP*.md,
# **/*AUDIT*.md, **/*BACKLOG*.md and **/AGENT_*.md by policy, so a link to one
# of those is a link every reader outside this working copy will find broken.
# Indexing it is worse than omitting it. check-ignore is run index-aware on
# purpose: a file that matches an ignore pattern but is already tracked is
# already in public history and stays public.
#
# The resulting contract, also stated on the Rust gate
# docs_index_covers_every_public_markdown_document:
#
#   INDEX.md must list every *.md under docs/ that exists on disk and is not
#   gitignored, and must link nothing else. Not a nonexistent file, and not a
#   gitignored one.

set -euo pipefail

# sort and comm must agree on collation, so pin the whole script to byte order.
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INDEX="docs/INDEX.md"
if [[ ! -f "$INDEX" ]]; then
  printf 'docs index missing: %s\nFix: create docs/INDEX.md with status and last-verified metadata.\n' "$INDEX" >&2
  exit 1
fi

present="$(mktemp)"
ignored="$(mktemp)"
public="$(mktemp)"
indexed="$(mktemp)"
missing="$(mktemp)"
dead="$(mktemp)"
local_only="$(mktemp)"
trap 'rm -f "$present" "$ignored" "$public" "$indexed" "$missing" "$dead" "$local_only"' EXIT

# Every markdown document that actually exists under docs/, INDEX.md included so
# that a self-link is not mistaken for a dead link.
find docs -type f -name '*.md' -print | LC_ALL=C sort > "$present"

# Documents git would refuse to publish. Outside a git checkout, such as a
# packaged crate, ignore status is not knowable, so nothing is treated as
# local-only and the gate degrades to pure existence checking.
: > "$ignored"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  check_ignore_status=0
  git check-ignore --stdin < "$present" > "$ignored" || check_ignore_status=$?
  # 0: some paths ignored. 1: none ignored. Anything else is a real failure.
  if (( check_ignore_status > 1 )); then
    printf 'docs index contract: git check-ignore failed with status %d.\n' "$check_ignore_status" >&2
    exit 1
  fi
  LC_ALL=C sort -o "$ignored" "$ignored"
else
  printf 'docs index contract: not a git checkout, ignore status unavailable, checking existence only.\n'
fi

comm -23 "$present" "$ignored" | grep -v "^${INDEX}$" > "$public" || true

grep -Eo '\((docs/)?[^)]*\.md\)' "$INDEX" \
  | tr -d '()' \
  | awk '{ if ($0 ~ /^docs\//) print $0; else print "docs/" $0 }' \
  | LC_ALL=C sort -u > "$indexed"

comm -23 "$public" "$indexed" > "$missing"
comm -23 "$indexed" "$present" > "$dead"
comm -12 "$indexed" "$ignored" > "$local_only"

violations=()
if [[ -s "$missing" ]]; then
  violations+=("public docs on disk missing from docs/INDEX.md:")
  while IFS= read -r path; do
    violations+=("  $path")
  done < "$missing"
fi

if [[ -s "$dead" ]]; then
  violations+=("docs/INDEX.md links files that do not exist on disk:")
  while IFS= read -r path; do
    violations+=("  $path")
  done < "$dead"
fi

if [[ -s "$local_only" ]]; then
  violations+=("docs/INDEX.md links gitignored documents, which no other reader can open:")
  while IFS= read -r path; do
    violations+=("  $path")
  done < "$local_only"
fi

while IFS= read -r path; do
  relative_path="${path#docs/}"
  line="$(grep -F "]($path)" "$INDEX" || grep -F "]($relative_path)" "$INDEX" || true)"
  if [[ -z "$line" ]]; then
    continue
  fi
  # A directory README is the signpost telling a reader what the directory holds
  # and where to go instead. That guidance is current even when everything it
  # points at is not, so it is exempt from the per-directory status rule.
  if [[ "$(basename "$path")" == "README.md" ]]; then
    continue
  fi
  case "$path" in
    docs/archive/*)
      [[ "$line" == \|*\`archived\`* ]] || violations+=("$path must be indexed with status archived")
      ;;
    docs/generated/*)
      [[ "$line" == \|*\`generated\`* ]] || violations+=("$path must be indexed with status generated")
      ;;
    docs/legacy/*)
      if [[ "$line" != \|*\`archived\`* && "$line" != \|*\`superseded\`* ]]; then
        violations+=("$path must be indexed with status archived or superseded")
      fi
      ;;
  esac
done < "$public"

if ! grep -Eq '^Last verified: [0-9]{4}-[0-9]{2}-[0-9]{2}$' "$INDEX"; then
  violations+=("docs/INDEX.md must declare Last verified: YYYY-MM-DD")
fi

if (( ${#violations[@]} > 0 )); then
  printf 'documentation index contract failed.\n' >&2
  printf '%s\n' "${violations[@]}" >&2
  printf '\nFix: index every existing, non-gitignored docs/*.md with current/generated/archived/superseded status, and drop rows for files that were deleted or are gitignored.\n' >&2
  exit 1
fi

printf 'documentation index contract: docs/INDEX.md covers every public markdown document.\n'
