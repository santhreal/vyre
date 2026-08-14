#!/usr/bin/env bash
# Enforce that every filesystem path cited inside release/evidence resolves.
#
# ORACLE: the filesystem plus git ignore status. NOT git's index, and NOT the
# artifact's own internal consistency.
#
# WHY THIS EXISTS. Before this gate, nothing validated that the paths named
# inside release evidence still existed. The release hygiene consumer only
# compares findings.len() against the summed finding_summary counts. Nothing
# reads findings[].path or stats it. release_evidence/artifact_status.rs
# does stat files, but only the artifact files themselves from a hardcoded
# expected list, never paths parsed out of their contents.
#
# So the only semantic check on an artifact was INTERNAL SELF-CONSISTENCY, which
# a stale artifact passes trivially: deleting a source file changes neither the
# array nor the summary, so the counts still agree. The artifact stays perfectly
# self-validating while citing code that no longer exists. That is a dead link
# wearing a JSON schema, and it is the same defect shape as an index that asks
# git what is on disk: the check runs, goes green, and carries no information
# about the property it appears to establish.
#
# Release evidence is the worst place for it, because its entire purpose is to
# be trusted at release time. An artifact that authoritatively cites deleted
# symbols certifies against fiction.
#
# SCOPE: every string leaf, at any depth, in every JSON under release/evidence
# whose value is shaped like a filesystem path. Shape is three conditions: no
# whitespace, a last component carrying a file extension the tree actually uses,
# and either a `/` or a nearest enclosing key named `path`, because bare sibling
# filenames are cited that way.
#
# Deliberately NOT just `findings`, and deliberately not just objects sitting
# directly inside a top-level array. When this gate was written the tree carried
# 185 stale citations across 16 artifacts, and only 8 of those were in a
# findings array. The largest single block was a stale path prefix: a renamed
# analysis component left two artifacts pointing at its previous source
# directory. The evidence described a real capability at the wrong path, which is
# the failure mode a path oracle catches and a self-consistency check cannot.
# Gating findings alone would have covered 4 percent of the defect.
#
# The depth rule is not decoration either. Restricting discovery to members of a
# top-level array left 81 of 634 citations unread, and the one dead citation
# among them sat on an artifact's own root object: an unexpanded ${SANTH_ROOT}
# template naming a README in another repository. A gate that reads most of a
# document reports a clean tree it did not measure.
#
# Reading only the key `path` was the same mistake one level up. That key names
# 629 citations; the shape rule names 3404. The rest sit under `manifest`,
# `artifact`, `evidence_link`, `source_artifact`, `workflow` and bare array
# members, and nine of them were dead: four generated manifest inventories still
# listed a crate that had been folded into another, at a Cargo.toml that no
# longer exists. A key allowlist is a hardcoded member table, so it stops
# covering the next schema that cites a file under a name nobody added here.
#
# THE EXTENSION VOCABULARY IS DERIVED FROM THE TREE AT RUN TIME, from the
# extensions that occur among its own files, never from a literal list. The
# first `.cu` file added to the workspace extends this gate in the same commit,
# with nobody editing this script. It is also what keeps the shape rule off
# version strings, op ids and schema ids: `1.2.0`, `vyre-primitives::hardware`
# and `vyre-conform-input-envelope-v1` end in nothing the tree uses as an
# extension.
#
# PATH RESOLUTION, three conventions in use, all of them live:
#   absolute            taken as-is (the hygiene scanners emit absolute paths)
#   relative            resolved against the workspace root
#   relative            else resolved against the artifact's own directory
#                       (release-gate-log.json cites sibling artifacts this way)
# The artifact-directory fallback is not decoration: without it this gate emits
# three false positives on conformance/release-gate-log.json. A gate with false
# positives gets muted, so it is worth the extra stat.
#
# IGNORE STATUS comes from `git check-ignore`, never `git ls-files`. Existence
# is a question for stat(2): a generated-but-untracked file is present on disk
# and must not read as missing, and a file deleted from disk stays in the
# tracked set until the deletion is committed. Tracking state is a transient
# fact about the staging area. check-ignore answers a different and legitimate
# question, "will this path ever reach another reader", so a cited path that is
# present but gitignored is reported separately: evidence that cites a
# local-only file is unverifiable by anyone else. check-ignore is run
# index-aware, so a file matching an ignore pattern but already tracked stays
# public. Findings span more than one repository, so it is run per repository.
#
# Outside a git checkout, ignore status is not knowable, so the gate degrades to
# pure existence checking rather than guessing.

set -euo pipefail

# sort and comm must agree on collation, so pin the whole script to byte order.
export LC_ALL=C

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Overridable so the gate can be exercised against a fixture tree. Defaults are
# the real thing; nothing in CI passes these.
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$ROOT}"
EVIDENCE_DIR="${EVIDENCE_DIR:-release/evidence}"

if [[ ! -d "$EVIDENCE_DIR" ]]; then
  printf 'evidence path contract: %s does not exist.\n' "$EVIDENCE_DIR" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  printf 'evidence path contract: jq is required to read evidence artifacts.\n' >&2
  exit 1
fi

cited="$(mktemp)"
present="$(mktemp)"
ignored="$(mktemp)"
trap 'rm -f "$cited" "$present" "$ignored"' EXIT

# The extension vocabulary comes from the tree itself. Tracked files answer it
# inside a checkout; a fixture tree need not be one, so fall back to a walk that
# skips build output.
if git -C "$WORKSPACE_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  tree_files() { git -C "$WORKSPACE_ROOT" ls-files; }
else
  tree_files() {
    find "$WORKSPACE_ROOT" -type f -not -path '*/.git/*' -not -path '*/target/*' -print
  }
fi

extensions="$(
  tree_files \
    | sed -n 's|.*/||; s|^.*\.\([A-Za-z0-9][A-Za-z0-9]*\)$|\1|p' \
    | tr 'A-Z' 'a-z' \
    | LC_ALL=C sort -u \
    | jq -R -s 'split("\n") | map(select(length > 0))'
)"

if [[ "$extensions" == "[]" ]]; then
  printf 'evidence path contract: no file extension occurs under %s, so the citation vocabulary cannot be derived. Fix: point WORKSPACE_ROOT at the workspace being certified.\n' \
    "$WORKSPACE_ROOT" >&2
  exit 1
fi

# Every (artifact, location, path) citation in the evidence tree. One line each,
# tab separated. The location is the full jq-style route to the string itself,
# including the key that carries it, so two citations on one object stay
# distinguishable and a citation stays addressable however deeply a schema nests
# it.
: > "$cited"
while IFS= read -r artifact; do
  jq -r --arg artifact "$artifact" --argjson exts "$extensions" '
    . as $root
    | [paths(type == "string")]
    | .[]
    | . as $route
    | ($root | getpath($route)) as $value
    | select($value != "")
    | select($value | test("[[:space:]]") | not)
    | ($route | map(select(type == "string")) | last) as $key
    | select(($value | test("/")) or $key == "path")
    | ($value | sub("/+$"; "") | split("/") | last | split(".")) as $parts
    | select(($parts | length) > 1)
    | select($exts | index($parts | last | ascii_downcase))
    | ($route
        | map(if type == "number" then "[\(.)]" else ".\(.)" end)
        | join("")
        | sub("^\\."; "")) as $location
    | "\($artifact)\t\($location)\t\($value)"
  ' "$artifact" >> "$cited"
done < <(find "$EVIDENCE_DIR" -type f -name '*.json' -print | LC_ALL=C sort)

missing_report=()
: > "$present"

while IFS=$'\t' read -r artifact location path; do
  [[ -n "$path" ]] || continue
  resolved=""
  if [[ "$path" = /* ]]; then
    [[ -e "$path" ]] && resolved="$path"
  else
    if [[ -e "$WORKSPACE_ROOT/$path" ]]; then
      resolved="$WORKSPACE_ROOT/$path"
    elif [[ -e "$(dirname "$artifact")/$path" ]]; then
      resolved="$(dirname "$artifact")/$path"
    fi
  fi

  if [[ -z "$resolved" ]]; then
    missing_report+=("  ${artifact} ${location} cites a path that does not exist: ${path}")
  else
    printf '%s\n' "$resolved" >> "$present"
  fi
done < "$cited"

# Present-but-unpublishable. Grouped by repository, because cited paths span
# more than one checkout and .gitignore is per repository.
: > "$ignored"
if [[ -s "$present" ]]; then
  LC_ALL=C sort -u -o "$present" "$present"
  declare -A repo_of=()
  while IFS= read -r path; do
    dir="$(dirname "$path")"
    if [[ -z "${repo_of[$dir]+set}" ]]; then
      repo_of[$dir]="$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null || true)"
    fi
    repo="${repo_of[$dir]}"
    [[ -n "$repo" ]] || continue
    printf '%s\n' "$path" >> "${ignored}.repo.$(printf '%s' "$repo" | tr -c 'A-Za-z0-9' '_')"
  done < "$present"

  for bucket in "${ignored}".repo.*; do
    [[ -e "$bucket" ]] || continue
    sample="$(head -n 1 "$bucket")"
    repo="$(git -C "$(dirname "$sample")" rev-parse --show-toplevel 2>/dev/null || true)"
    [[ -n "$repo" ]] || { rm -f "$bucket"; continue; }
    status=0
    git -C "$repo" check-ignore --stdin < "$bucket" >> "$ignored" || status=$?
    # 0: some paths ignored. 1: none ignored. Anything else is a real failure.
    if (( status > 1 )); then
      printf 'evidence path contract: git check-ignore failed with status %d in %s.\n' "$status" "$repo" >&2
      rm -f "${ignored}".repo.*
      exit 1
    fi
    rm -f "$bucket"
  done
fi

local_only_report=()
if [[ -s "$ignored" ]]; then
  LC_ALL=C sort -u -o "$ignored" "$ignored"
  while IFS= read -r path; do
    local_only_report+=("  ${path}")
  done < "$ignored"
fi

failed=0

if (( ${#missing_report[@]} > 0 )); then
  failed=1
  printf 'evidence path contract: %d citation(s) name a path that is not on disk.\n' \
    "${#missing_report[@]}" >&2
  printf '%s\n' "${missing_report[@]}" >&2
  printf 'Fix: regenerate the artifact from the current tree with its owning release-evidence command, or delete the citation if the evidence is genuinely obsolete. Do not hand-edit generated artifacts.\n' >&2
fi

if (( ${#local_only_report[@]} > 0 )); then
  failed=1
  printf 'evidence path contract: %d cited path(s) exist but are gitignored, so no other reader can verify them.\n' \
    "${#local_only_report[@]}" >&2
  printf '%s\n' "${local_only_report[@]}" >&2
  printf 'Fix: evidence must cite paths that reach other readers. Either commit the path or stop citing it.\n' >&2
fi

if (( failed )); then
  exit 1
fi

total="$(wc -l < "$cited" | tr -d ' ')"
printf 'evidence path contract: all %s cited path(s) under %s resolve on disk.\n' \
  "$total" "$EVIDENCE_DIR"
