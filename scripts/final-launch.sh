#!/usr/bin/env bash
#
# Guarded public launch launcher for the configured Vyre release.
#
# This script intentionally refuses to run unless the maintainer sets:
#   VYRE_RELEASE_APPROVED=<token derived by scripts/lib/release_train.sh>
#
# It performs the approval-gated publish and push and records launch
# verification:
#   1. cargo_full publish in audited dependency order.
#   2. verify the configured public repository without changing the private repository.
#   3. push the release branch and product-scoped tags.

set -euo pipefail

PREFLIGHT=0
if [[ "${1:-}" == "--preflight" ]]; then
    PREFLIGHT=1
    shift
fi
if [[ "$#" -ne 0 ]]; then
    printf 'Fix: unknown final-launch argument(s): %s\n' "$*" >&2
    exit 2
fi

VYRE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$VYRE_ROOT"
source scripts/lib/cargo_runner.sh
source scripts/lib/repo_boundary.sh
source scripts/lib/release_train.sh
vyre_select_cargo_runner
vyre_load_repo_boundary
vyre_load_release_train

vyre_tags=(
    "$VYRE_RELEASE_TAG_VYRE_RC"
    "$VYRE_RELEASE_TAG_VYRE"
)

APPROVAL_TOKEN="$VYRE_RELEASE_LAUNCH_APPROVAL_TOKEN"
if [[ "$PREFLIGHT" != "1" && "${VYRE_RELEASE_APPROVED:-}" != "$APPROVAL_TOKEN" ]]; then
    printf 'Fix: refusing final launch without explicit approval.\n' >&2
    printf 'Set VYRE_RELEASE_APPROVED=%s only after maintainer approval for publish and git push. This script verifies %s is already public and does not change %s visibility.\n' "$APPROVAL_TOKEN" "$VYRE_RELEASE_PUBLIC_REPO" "$VYRE_RELEASE_PRIVATE_REPO" >&2
    exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
    printf 'Fix: jq is required to write launch completion evidence.\n' >&2
    exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
    printf 'Fix: GitHub CLI `gh` is required for repository visibility verification.\n' >&2
    exit 2
fi

if ! gh auth status >/dev/null 2>&1; then
    printf 'Fix: GitHub CLI is not authenticated; run gh auth login before final launch.\n' >&2
    exit 2
fi

vyre_origin="$(git remote get-url origin 2>/dev/null || true)"
if [[ "$vyre_origin" != *"$VYRE_RELEASE_VYRE_REPOSITORY"* ]]; then
    printf 'Fix: Vyre origin %s does not match release repository %s.\n' "$vyre_origin" "$VYRE_RELEASE_VYRE_REPOSITORY" >&2
    exit 2
fi

if ! release_branch="$(git symbolic-ref --quiet --short HEAD)"; then
    printf 'Fix: refusing final launch from a detached Vyre HEAD.\n' >&2
    exit 2
fi
if [[ "$release_branch" != "main" ]]; then
    printf 'Fix: final launch requires main; found %s.\n' "$release_branch" >&2
    exit 2
fi

vyre_dirty="$(git status --porcelain)"
if [[ -n "$vyre_dirty" && "$PREFLIGHT" != "1" ]]; then
    printf 'Fix: the release repository must be clean before final launch; commit or intentionally clear the reported work.\n' >&2
    exit 2
fi
if [[ "$PREFLIGHT" == "1" && -n "$vyre_dirty" ]]; then
    printf 'final-launch preflight note: the release repository is dirty; real launch will refuse until it is clean.\n'
fi

for tag in "${vyre_tags[@]}"; do
    if git rev-parse --verify "refs/tags/${tag}" >/dev/null 2>&1; then
        printf 'Fix: Vyre release tag %s already exists locally; refusing to risk a stale target.\n' "$tag" >&2
        exit 2
    fi
    if git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null 2>&1; then
        printf 'Fix: Vyre release tag %s already exists on origin; refusing to overwrite it.\n' "$tag" >&2
        exit 2
    fi
done

repo_visibility="$(gh repo view "$VYRE_RELEASE_PUBLIC_REPO" --json visibility --jq '.visibility' 2>/dev/null || true)"
if [[ -z "$repo_visibility" ]]; then
    printf 'Fix: GitHub repository %s is not visible to gh; refusing final launch before publish.\n' "$VYRE_RELEASE_PUBLIC_REPO" >&2
    exit 2
fi
if [[ "${repo_visibility^^}" != "PUBLIC" ]]; then
    printf 'Fix: GitHub repository %s visibility is %s, expected PUBLIC. %s visibility is intentionally untouched.\n' "$VYRE_RELEASE_PUBLIC_REPO" "$repo_visibility" "$VYRE_RELEASE_PRIVATE_REPO" >&2
    exit 2
fi

if [[ "$PREFLIGHT" == "1" ]]; then
    bash scripts/publish-release.sh --preflight
    printf 'final-launch preflight passed; no publish, evidence commit, tag, or push performed.\n'
    exit 0
fi

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


git tag -a "$VYRE_RELEASE_TAG_VYRE_RC" -m "$VYRE_RELEASE_TAG_VYRE_RC"
git push origin "$release_branch"
git push origin "$VYRE_RELEASE_TAG_VYRE_RC"


"$CARGO_RUNNER" run -j1 --manifest-path xtask/Cargo.toml --bin xtask -- vyre-release-gate
VYRE_RELEASE_APPROVED="$VYRE_RELEASE_PUBLISH_APPROVAL_TOKEN" bash scripts/publish-release.sh
printf 'verified GitHub repository is public: %s\n' "$VYRE_RELEASE_PUBLIC_REPO"

git tag -a "$VYRE_RELEASE_TAG_VYRE" -m "$VYRE_RELEASE_TAG_VYRE"
git push origin "$VYRE_RELEASE_TAG_VYRE"


release_notes="release/evidence/docs/release-notes-body.md"
gh release create "$VYRE_RELEASE_TAG_VYRE" \
    --repo "$VYRE_RELEASE_VYRE_REPOSITORY" \
    --title "$VYRE_RELEASE_DISPLAY" \
    --notes-file "$release_notes"

mkdir -p release/evidence/final
jq -n \
    --arg public_repo "$VYRE_RELEASE_PUBLIC_REPO" \
    --arg branch "$release_branch" \
    --arg conformance "$release_conformance_evidence" \
    --arg vyre_version "$VYRE_RELEASE_VYRE_VERSION" \
    --arg verify_public_repo_action "$VYRE_RELEASE_VERIFY_PUBLIC_REPO_ACTION" \
    --arg vyre_rc_tag "$VYRE_RELEASE_TAG_VYRE_RC" \
    --arg vyre_tag "$VYRE_RELEASE_TAG_VYRE" \
    '{
        schema_version: 2,
        release_train: {
            vyre: $vyre_version
        },
        git: {
            branch: $branch,
            tags: [
                $vyre_rc_tag,
                $vyre_tag
            ]
        },
        public_repository: $public_repo,
        external_actions: [
            {
                action: "prove sharded all-backend conformance certificate",
                status: "complete",
                evidence: $conformance
            },
            {
                action: "cargo_full publish approved crates in dependency order",
                status: "complete",
                evidence: "scripts/publish-release.sh"
            },
            {
                action: $verify_public_repo_action,
                status: "complete",
                evidence: ("gh repo view " + $public_repo + " --json visibility")
            },
            {
                action: "git push release branch and tags",
                status: "complete",
                evidence: ("git push origin " + $branch + " " + $vyre_rc_tag + " " + $vyre_tag + " && gh release create " + $vyre_tag)
            }
        ],
        completion_status: "complete"
    }' > release/evidence/final/public-launch-completion.json

"$CARGO_RUNNER" run -j1 --manifest-path xtask/Cargo.toml --bin xtask -- launch-state --output release/evidence/final/public-launch-state.json
"$CARGO_RUNNER" run -j1 --manifest-path xtask/Cargo.toml --bin xtask -- vyre-release-gate --launch-complete

git add \
    release/evidence/package/publish-readiness.json \
    release/evidence/conformance/release-all-backends-certificate.json \
    release/evidence/final/public-launch-completion.json \
    release/evidence/final/public-launch-state.json
git commit -m "Record ${VYRE_RELEASE_DISPLAY} public launch"
git push origin "$release_branch"

printf '%s public launch actions completed.\n' "$VYRE_RELEASE_DISPLAY"
