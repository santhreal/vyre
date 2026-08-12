#!/usr/bin/env bash
# Enforce one private execution queue and no parallel Markdown plans.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

violations=()

[[ -f BACKLOG.md ]] || violations+=("BACKLOG.md is missing")
[[ -f CHANGELOG.md ]] || violations+=("CHANGELOG.md is missing")

if [[ -f BACKLOG.md ]]; then
  if ! grep -Fq '| Number | Affected files | Problem | Acceptance criteria |' BACKLOG.md; then
    violations+=("BACKLOG.md must use the four-column project backlog contract")
  fi
  if grep -Fq '| ID | Axis | Local evidence | Research basis | Work | Proof gate | Dedup seam |' BACKLOG.md; then
    violations+=("BACKLOG.md still contains the superseded seven-column plan table")
  fi
fi

shopt -s globstar nullglob
for path in *.md **/*.md; do
  [[ "$path" == "BACKLOG.md" ]] && continue
  name="${path##*/}"
  case "$name" in
    *PLAN*.md|*ROADMAP*.md|*BACKLOG*.md|*STATUS*.md|*HANDOFF*.md|*TASKS*.md|*BUILDOUT*.md|*PRD*.md|*BRIEF*.md|*TRAJECTORY*.md|*SEGMENTATION*.md|*GENERALIZATION*.md)
      violations+=("$path is a parallel execution-plan surface; migrate it into BACKLOG.md and delete it")
      ;;
  esac
done

if (( ${#violations[@]} > 0 )); then
  printf 'single-backlog contract failed.\n' >&2
  printf '%s\n' "${violations[@]}" >&2
  printf '\nFix: keep active work only in BACKLOG.md; keep public docs as contracts, procedures, or evidence.\n' >&2
  exit 1
fi

printf 'single-backlog contract: BACKLOG.md is the only execution queue.\n'
