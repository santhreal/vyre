#!/usr/bin/env bash
# Both halves of the internal-dependency version rule, for every publishable
# member of the workspace.
#
#   1. A publishable crate's normal or build dependency on a PUBLISHED member
#      must carry a version. Path-only breaks `cargo publish`: the published
#      crate cannot resolve its sibling from the registry.
#   2. A publishable crate's dependency on a `publish = false` member must NOT
#      carry a version, in any table including dev-dependencies. A version
#      requirement on an unpublishable crate is one no registry can ever
#      satisfy, so `cargo package` on the depender fails. Cargo strips
#      path-only dev-dependencies at package time, which is what makes the
#      path-only form the correct one here.
#
# The same two rules apply to `[workspace.dependencies]` entries that name a
# member, because `<crate>.workspace = true` inherits whatever that table says.
# That table is where the live defect sat: three `publish = false` members
# carried `version = "0.7.2"` and a published crate inherited one of them.
#
# Both crate sets are derived from the tracked manifests at run time. Earlier
# revisions hardcoded 13 publishable crates and a 15-name alternation of
# internal crates, so 13 publishable crates and 18 members were unchecked and
# adding either kind kept the gate green.
#
# Run before any `./cargo_full publish`. Wired into release signoff.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 "$ROOT/scripts/lib/check_internal_deps_have_versions.py" "$ROOT"
