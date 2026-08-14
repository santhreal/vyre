#!/usr/bin/env bash
# Inventory #112  -  status-managed audit files must tag every finding row.
#
# A status-managed audit is any audits/*.md file that declares a "Status legend".
# Within those files, numbered findings must begin with one of the canonical
# status tags so open work cannot hide as plain prose.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/source_scan.sh

VALID_STATUS='`(open|in_progress|fixed)`'
violations=0
managed=0

while IFS= read -r file; do
  # A failed search must not read as "this file is not status-managed", which is
  # what `if ! rg -q` did: an unavailable search skipped every file, `managed`
  # stayed 0, and the gate then blamed the tree for having no managed audits.
  if ! vyre_file_has '^Status legend:' "$file"; then
    continue
  fi
  managed=$((managed + 1))

  while IFS=: read -r line_no text; do
    if ! grep -qE "^[[:space:]]*[0-9]+\\. ${VALID_STATUS}[[:space:]]" <<< "$text"; then
      echo "audit status violation: $file:$line_no" >&2
      echo "  $text" >&2
      violations=$((violations + 1))
    fi
  done < <(
    awk '
      /^## Highest Leverage Execution Order/ { exit }
      /^[[:space:]]*[0-9]+\. / { print FNR ":" $0 }
    ' "$file"
  )
done < <(git ls-files -- 'audits/*.md' | sort)

if [[ "$managed" -eq 0 ]]; then
  echo "audit status check: no status-managed audit files found." >&2
  exit 1
fi

if [[ "$violations" -gt 0 ]]; then
  echo "Fix: prefix every numbered finding in status-managed audits with \`open\`, \`in_progress\`, or \`fixed\`." >&2
  exit 1
fi

echo "audit status check: $managed status-managed audit file(s), all finding rows tagged."
