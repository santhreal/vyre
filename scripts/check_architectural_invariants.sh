#!/usr/bin/env bash
# Substrate-neutral dependency guard.
#
# The ownership contract in docs/ARCHITECTURE.md and
# docs/CRATE_OWNERSHIP.toml keeps semantic IR, compiler contracts, and
# backend-neutral interfaces independent from concrete drivers. This check
# rejects non-development dependency edges from neutral crates to concrete
# targets or runtime products.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Crates that MUST stay substrate-neutral. A violation in any of these is
# a rebuild regression, not a style nit.
PURE_CRATES=(
  "vyre"
  "vyre-foundation"
  "vyre-primitives"
  "vyre-reference"
  "vyre-spec"
  "vyre-driver"
)

# Crates that the pure crates must never depend on (outside dev-dependencies).
FORBIDDEN_DEPS=(
  "vyre-driver-wgpu"
  "vyre-driver-cuda"
  "vyre-driver-spirv"
  "vyre-runtime"
  "vyre-aot"
  "wgpu"
  "naga"
)

violations=0

for crate in "${PURE_CRATES[@]}"; do
  manifest="$REPO_ROOT/$crate/Cargo.toml"
  if [[ ! -f "$manifest" ]]; then
    echo "ARCH VIOLATION: required neutral crate manifest is missing: $manifest" >&2
    violations=$((violations + 1))
    continue
  fi

  # Extract the [dependencies] and [build-dependencies] sections only.
  # [dev-dependencies] are intentionally permitted.
  pure_deps="$(awk '
    /^\[dependencies\]/          { inside=1; next }
    /^\[build-dependencies\]/    { inside=1; next }
    /^\[dev-dependencies\]/      { inside=0; next }
    /^\[target\.[^]]+\.dev-dependencies\]/ { inside=0; next }
    /^\[target\.[^]]+\.dependencies\]/     { inside=1; next }
    /^\[target\.[^]]+\.build-dependencies\]/ { inside=1; next }
    /^\[/                        { inside=0; next }
    inside && NF > 0             { print }
  ' "$manifest" | grep -v 'optional[[:space:]]*=[[:space:]]*true')"

  for forbidden in "${FORBIDDEN_DEPS[@]}"; do
    if echo "$pure_deps" | grep -qE "^[[:space:]]*\"?${forbidden}\"?[[:space:]]*="; then
      echo "ARCH VIOLATION: $crate depends on $forbidden outside [dev-dependencies]." >&2
      echo "  Manifest: $manifest" >&2
      echo "  Neutral crates must follow docs/CRATE_OWNERSHIP.toml." >&2
      echo "  Fix: move the dependency under [dev-dependencies] or relocate" >&2
      echo "  production code to the owning downstream crate." >&2
      violations=$((violations + 1))
    fi
  done
done

# A second invariant: backend crates must not depend on non-existent legacy
# crate names. Stale references make this gate pass or fail for the wrong
# architecture.
if rg -n '^[[:space:]]*"?vyre-(ir|wgpu)"?[[:space:]]*=' --glob Cargo.toml "$REPO_ROOT" >/tmp/vyre_arch_legacy_hits.$$ 2>/dev/null; then
  echo "ARCH VIOLATION: stale legacy crate names in manifests:" >&2
  cat /tmp/vyre_arch_legacy_hits.$$ >&2
  rm -f /tmp/vyre_arch_legacy_hits.$$
  violations=$((violations + 1))
else
  rm -f /tmp/vyre_arch_legacy_hits.$$
fi

if [[ "$violations" -gt 0 ]]; then
  echo "" >&2
  echo "Architectural invariants failed: $violations violation(s)." >&2
  echo "See docs/ARCHITECTURE.md for the substrate-neutrality contract." >&2
  exit 1
fi

echo "Architectural invariants: all $(printf '%s\n' "${PURE_CRATES[@]}" | wc -l | tr -d ' ') substrate-neutral crates clean."
