#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/.github/CI_REQUIRED.md"
BRANCH="${1:-main}"

if [ ! -f "$DOC" ]; then
  printf "Fix: required CI document is missing: %s\n" "$DOC" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  printf "Fix: GitHub CLI gh is required to apply branch protection.\n" >&2
  exit 1
fi

REPO="${GITHUB_REPOSITORY:-}"
if [ -z "$REPO" ]; then
  REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
fi
if [ "$REPO" != "santhreal/vyre" ]; then
  printf "Fix: branch protection applies only to santhreal/vyre, got %s.\n" "${REPO:-<empty>}" >&2
  exit 1
fi

mapfile -t CONTEXTS < <(
  awk '
    /^## Scheduled or Manual Deep Gates/ { stop=1 }
    stop { next }
    /^- `[^`]+`/ {
      line=$0
      sub(/^- `/, "", line)
      sub(/`.*/, "", line)
      print line
    }
  ' "$DOC" \
    | sort -u
)

if [ "${#CONTEXTS[@]}" -eq 0 ]; then
  printf "Fix: no required status contexts parsed from %s\n" "$DOC" >&2
  exit 1
fi

missing_contexts=()
for context in "${CONTEXTS[@]}"; do
  if ! grep -R -F "name: $context" "$ROOT/.github/workflows" >/dev/null 2>&1 \
    && ! grep -R -F "  $context:" "$ROOT/.github/workflows" >/dev/null 2>&1; then
    missing_contexts+=("$context")
  fi
done
if [ "${#missing_contexts[@]}" -ne 0 ]; then
  printf "Fix: CI_REQUIRED.md lists status context(s) that no workflow defines:\n" >&2
  printf "  - %s\n" "${missing_contexts[@]}" >&2
  exit 1
fi

path_filtered_workflows=()
missing_required_triggers=()
for workflow in \
  "$ROOT/.github/workflows/ci.yml" \
  "$ROOT/.github/workflows/bench.yml" \
  "$ROOT/.github/workflows/architectural-invariants.yml" \
  "$ROOT/.github/workflows/conform.yml" \
  "$ROOT/.github/workflows/gpu-parity.yml"; do
  if [ ! -f "$workflow" ]; then
    missing_required_triggers+=("$workflow (missing)")
    continue
  fi
  if ! grep -E '^[[:space:]]*pull_request:' "$workflow" >/dev/null \
    || ! grep -E '^[[:space:]]*push:' "$workflow" >/dev/null \
    || ! grep -E '^[[:space:]]*branches:[[:space:]]*\[.*main.*\]' "$workflow" >/dev/null; then
    missing_required_triggers+=("$workflow")
  fi
  if awk '
    /^jobs:/ { exit }
    /^[[:space:]]*paths(-ignore)?:/ { found=1 }
    END { exit found ? 0 : 1 }
  ' "$workflow"; then
    path_filtered_workflows+=("$workflow")
  fi
done
if [ "${#missing_required_triggers[@]}" -ne 0 ]; then
  printf "Fix: required workflow(s) must run on pull_request and push to main:\n" >&2
  printf "  - %s\n" "${missing_required_triggers[@]}" >&2
  exit 1
fi
if [ "${#path_filtered_workflows[@]}" -ne 0 ]; then
  printf "Fix: required workflow(s) still use path filters:\n" >&2
  printf "  - %s\n" "${path_filtered_workflows[@]}" >&2
  exit 1
fi

missing_fail_closed=()
for pair in \
  "$ROOT/.github/workflows/ci.yml::CI release gate" \
  "$ROOT/.github/workflows/conform.yml::Conform release gate" \
  "$ROOT/.github/workflows/gpu-parity.yml::GPU release gate"; do
  workflow="${pair%%::*}"
  job="${pair##*::}"
  if [ ! -f "$workflow" ] \
    || ! grep -F "name: $job" "$workflow" >/dev/null \
    || ! grep -F 'if: ${{ always() }}' "$workflow" >/dev/null \
    || ! grep -F ".result" "$workflow" >/dev/null \
    || ! grep -F "exit 1" "$workflow" >/dev/null; then
    missing_fail_closed+=("$pair")
  fi
done
if [ "${#missing_fail_closed[@]}" -ne 0 ]; then
  printf "Fix: required fan-in job(s) are missing fail-closed dependency checks:\n" >&2
  printf "  - %s\n" "${missing_fail_closed[@]}" >&2
  exit 1
fi

args=(
  api
  --method
  PUT
  "repos/$REPO/branches/$BRANCH/protection"
  -H
  "Accept: application/vnd.github+json"
  -F
  "required_status_checks[strict]=true"
  -F
  "enforce_admins=true"
  -F
  "required_pull_request_reviews=null"
  -F
  "restrictions=null"
)

for context in "${CONTEXTS[@]}"; do
  args+=(-f "required_status_checks[contexts][]=$context")
done

printf "Applying branch protection to %s@%s with %s required status context(s).\n" "$REPO" "$BRANCH" "${#CONTEXTS[@]}"
gh "${args[@]}"
