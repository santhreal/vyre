#!/usr/bin/env bash
# Apply the required status contexts in .github/CI_REQUIRED.md to a branch.
#
# This is an operator action against the GitHub API, not a rule: whether the
# document is coherent with the workflows is the `ci-required` gate's question,
# and this script refuses to apply anything until that gate passes. The context
# list is read here because it is the payload of the API call.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="$ROOT/.github/CI_REQUIRED.md"
BRANCH="${1:-main}"

if ! command -v gh >/dev/null 2>&1; then
  printf "Fix: install the GitHub CLI; branch protection is applied through it.\n" >&2
  exit 1
fi

REPO="${GITHUB_REPOSITORY:-}"
if [ -z "$REPO" ]; then
  REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
fi
if [ "$REPO" != "santhreal/vyre" ]; then
  printf "Fix: run this against santhreal/vyre; the repository resolved to %s.\n" "$REPO" >&2
  exit 1
fi

(cd "$ROOT" && ./cargo_full run -q -p xtask --bin xtask -- ci-required)

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
