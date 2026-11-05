#!/usr/bin/env bash
# Apply the required status contexts in .github/CI_REQUIRED.md to a branch.
#
# This is an operator action against the GitHub API, not a rule: whether the
# document is coherent with the workflows is the `ci-required` gate's question,
# and this script refuses to apply anything until that gate passes. The context
# list is read here because it is the payload of the API call.
#
# The payload also requires the code-owner review. Every line of
# .github/CODEOWNERS is advisory until it does, so the `codeowners` gate reads
# this file and fails when the requirement is dropped. The approval count is
# zero on purpose: an owner review is required for an owned path and no extra
# approval is demanded elsewhere.
#
# The gate also refuses to run if this script stops naming the branch that
# docs/PROTECTED_BOUNDARIES.toml declares protected.

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

CONTEXTS=()
while IFS= read -r context; do
  CONTEXTS+=("$context")
done < <(
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
  "required_pull_request_reviews[require_code_owner_reviews]=true"
  -F
  "required_pull_request_reviews[required_approving_review_count]=0"
  -F
  "required_pull_request_reviews[dismiss_stale_reviews]=true"
  -F
  "restrictions=null"
)

for context in "${CONTEXTS[@]}"; do
  args+=(-f "required_status_checks[contexts][]=$context")
done

printf "Applying branch protection to %s@%s with %s required status context(s).\n" "$REPO" "$BRANCH" "${#CONTEXTS[@]}"
gh "${args[@]}"
