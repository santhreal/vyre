#!/usr/bin/env bash
#
# Guarded public launch launcher for Vyre 0.4.2 / Weir 0.1.0.
#
# This script intentionally refuses to run unless the maintainer sets:
#   VYRE_RELEASE_APPROVED=launch-vyre-0.4.2-weir-0.1.0
#
# It performs the three approval-gated actions that complete
# release/plans/paradigm-shift-100-concrete.md:
#   1. cargo publish in audited dependency order.
#   2. make approved GitHub repositories public.
#   3. push the release branch and product-scoped tags.

set -euo pipefail

VYRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VYRE_ROOT"
source scripts/lib/cargo_runner.sh
vyre_select_cargo_runner

APPROVAL_TOKEN="launch-vyre-0.4.2-weir-0.1.0"
if [[ "${VYRE_RELEASE_APPROVED:-}" != "$APPROVAL_TOKEN" ]]; then
    printf 'Fix: refusing final launch without explicit approval.\n' >&2
    printf 'Set VYRE_RELEASE_APPROVED=%s only after maintainer approval for publish, public visibility, and git push.\n' "$APPROVAL_TOKEN" >&2
    exit 2
fi

if [[ -z "${VYRE_RELEASE_REPOS:-}" ]]; then
    printf 'Fix: VYRE_RELEASE_REPOS must list approved GitHub repos to make public, separated by spaces.\n' >&2
    printf 'Example: VYRE_RELEASE_REPOS="santhsecurity/vyre santhsecurity/weir"\n' >&2
    exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
    printf 'Fix: jq is required to write launch completion evidence.\n' >&2
    exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
    printf 'Fix: GitHub CLI `gh` is required for repository visibility changes.\n' >&2
    exit 2
fi

if ! gh auth status >/dev/null 2>&1; then
    printf 'Fix: GitHub CLI is not authenticated; run gh auth login before final launch.\n' >&2
    exit 2
fi

if ! git remote get-url origin >/dev/null 2>&1; then
    printf 'Fix: git remote `origin` is missing; refusing final launch.\n' >&2
    exit 2
fi

if ! release_branch="$(git symbolic-ref --quiet --short HEAD)"; then
    printf 'Fix: refusing final launch from a detached HEAD.\n' >&2
    exit 2
fi

if [[ -n "$(git status --porcelain)" ]]; then
    printf 'Fix: working tree has uncommitted or untracked changes; commit or intentionally clear them before final launch.\n' >&2
    exit 2
fi

RELEASE_TAGS=(vyre-v0.4.2 weir-v0.1.0 vyre-0.4.2-weir-0.1.0)
for tag in "${RELEASE_TAGS[@]}"; do
    if git rev-parse --verify "refs/tags/${tag}" >/dev/null 2>&1; then
        printf 'Fix: release tag %s already exists locally; refusing to risk stale tag target.\n' "$tag" >&2
        exit 2
    fi
    if git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null 2>&1; then
        printf 'Fix: release tag %s already exists on origin; refusing to overwrite public release tags.\n' "$tag" >&2
        exit 2
    fi
done

for repo in ${VYRE_RELEASE_REPOS}; do
    if ! gh repo view "$repo" >/dev/null 2>&1; then
        printf 'Fix: GitHub repository %s is not visible to gh; refusing final launch before publish.\n' "$repo" >&2
        exit 2
    fi
done

export VYRE_RELEASE_BACKEND="${VYRE_RELEASE_BACKEND:-all}"
export VYRE_RELEASE_SHARDS="${VYRE_RELEASE_SHARDS:-64}"
export VYRE_RELEASE_FEATURES="${VYRE_RELEASE_FEATURES:-gpu}"
export VYRE_RELEASE_CERT_DIR="${VYRE_RELEASE_CERT_DIR:-.internals/certs/release-shards}"
release_conformance_certificate="$(scripts/prove-release-shards.sh)"
release_conformance_evidence="release/evidence/conformance/release-all-backends-certificate.json"
mkdir -p "$(dirname "$release_conformance_evidence")"
cp "$release_conformance_certificate" "$release_conformance_evidence"
if [[ ! -s "$release_conformance_evidence" ]]; then
    printf 'Fix: release conformance certificate evidence was not written: %s\n' "$release_conformance_evidence" >&2
    exit 1
fi

VYRE_RELEASE_APPROVED=publish-vyre-0.4.2-weir-0.1.0 bash scripts/publish-release.sh

for repo in ${VYRE_RELEASE_REPOS}; do
    printf 'making GitHub repository public: %s\n' "$repo"
    gh repo edit "$repo" --visibility public --accept-visibility-change-consequences
done

mkdir -p release/evidence/final
jq -n \
    --arg repos "$VYRE_RELEASE_REPOS" \
    --arg branch "$release_branch" \
    --arg conformance "$release_conformance_evidence" \
    '{
        schema_version: 1,
        release_train: {
            vyre: "0.4.2",
            weir: "0.1.0"
        },
        git: {
            branch: $branch,
            tags: [
                "vyre-v0.4.2",
                "weir-v0.1.0",
                "vyre-0.4.2-weir-0.1.0"
            ]
        },
        repositories_public: ($repos | split(" ") | map(select(length > 0))),
        external_actions: [
            {
                action: "prove sharded all-backend conformance certificate",
                status: "complete",
                evidence: $conformance
            },
            {
                action: "cargo publish approved crates in dependency order",
                status: "complete",
                evidence: "scripts/publish-release.sh"
            },
            {
                action: "make repositories public",
                status: "complete",
                evidence: $repos
            },
            {
                action: "git push release branch and tags",
                status: "complete",
                evidence: "git push origin release branch && git push origin vyre-v0.4.2 weir-v0.1.0 vyre-0.4.2-weir-0.1.0"
            }
        ],
        completion_status: "complete"
    }' > release/evidence/final/public-launch-completion.json

"$CARGO_RUNNER" run -j1 --manifest-path xtask/Cargo.toml --bin xtask -- launch-state --output release/evidence/final/public-launch-state.json
"$CARGO_RUNNER" run -j1 --manifest-path xtask/Cargo.toml --bin xtask -- release-completion-audit --output release/evidence/final/completion-audit.json
"$CARGO_RUNNER" run -j1 --manifest-path xtask/Cargo.toml --bin xtask -- vyre-weir-release-gate

git add \
    release/evidence/package/publish-readiness.json \
    release/evidence/conformance/release-all-backends-certificate.json \
    release/evidence/final/public-launch-completion.json \
    release/evidence/final/public-launch-state.json \
    release/evidence/final/completion-audit.json
git commit -m "Record Vyre 0.4.2 and Weir 0.1.0 public launch"

for tag in "${RELEASE_TAGS[@]}"; do
    git tag -a "$tag" -m "$tag"
done

printf 'pushing release branch and product-scoped tags\n'
git push origin "$release_branch"
git push origin "${RELEASE_TAGS[@]}"

printf 'Vyre 0.4.2 / Weir 0.1.0 public launch actions completed.\n'
