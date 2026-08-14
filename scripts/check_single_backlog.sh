#!/usr/bin/env bash
# Enforce one execution queue and no parallel Markdown plans.
#
# The queue itself is a gitignored local file, so its ABSENCE is not a
# violation: a clean CI checkout legitimately has none. The rule is an upper
# bound, "no second execution-plan surface", and it is measured over TRACKED
# files only. An untracked local scratch plan confuses nobody, because nobody
# else can see it; a committed one does.
#
# An earlier version asserted `-f BACKLOG.md` first, which meant the gate
# could never pass in CI, which is why it was never wired.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

violations=()

[[ -f CHANGELOG.md ]] || violations+=("CHANGELOG.md is missing")

if [[ -f BACKLOG.md ]]; then
  # The documented contract spells these headings in lower case; match either.
  if ! grep -qiF '| number | affected files | problem | acceptance criteria |' BACKLOG.md; then
    violations+=("BACKLOG.md must use the four-column project backlog contract")
  fi
  if grep -Fq '| ID | Axis | Local evidence | Research basis | Work | Proof gate | Dedup seam |' BACKLOG.md; then
    violations+=("BACKLOG.md still contains the superseded seven-column plan table")
  fi
fi

while IFS= read -r path; do
  name="${path##*/}"
  case "$name" in
    *PLAN*.md|*ROADMAP*.md|*BACKLOG*.md|*STATUS*.md|*HANDOFF*.md|*TASKS*.md|*BUILDOUT*.md|*PRD*.md|*BRIEF*.md|*TRAJECTORY*.md|*SEGMENTATION*.md|*GENERALIZATION*.md)
      violations+=("$path is a committed parallel execution-plan surface; migrate it into the backlog and delete it")
      ;;
  esac
done < <(git ls-files '*.md')

if (( ${#violations[@]} > 0 )); then
  printf 'single-backlog contract failed.\n' >&2
  printf '%s\n' "${violations[@]}" >&2
  printf '\nFix: keep active work only in BACKLOG.md; keep public docs as contracts, procedures, or evidence.\n' >&2
  exit 1
fi

printf 'single-backlog contract: BACKLOG.md is the only execution queue.\n'
