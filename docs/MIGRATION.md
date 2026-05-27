# vyre  -  Migration Path

vyre's public API is frozen across minor versions. Anything that
breaks compatibility is recorded here with a migration path, deprecation
window, and the gate that proves the migration is honored. The
`scripts/check_trait_freeze.sh` and
`scripts/check_public_api_snapshot.sh` gates fail any change that
breaks the contract without a migration entry below.

## Doctrine

1. **Add, don't remove.** New surface is introduced as new types or new
   methods. Removing is the last resort.
2. **Extend, don't replace.** When a method's contract changes, ship
   the new version under a new name and deprecate the old one for at
   least one minor release.
3. **Document the why.** Every entry below names the audit item or
   incident that drove the change.
4. **Cite the gate.** Every entry below names the gate that prevents
   the migration from regressing once it is implemented.

## Active compatibility migrations

### `Match` → `ByteRange`
- **Why:** non-matching dialects (decode, future scanners) need a
  byte-range type that does not carry matching-specific semantics
  (`audits/VISION_ALIGNMENT_2026-04-23.md` V1).
- **Path:** new code uses `vyre::ByteRange`; existing code keeps
  `vyre::match_result::Match`. Zero-cost conversions both ways.
- **Window:** `Match` stays for the entire 0.6.x line. Removal is
  blocked until every in-workspace consumer has migrated and the
  public API snapshot records the deletion as a semver-major change.
- **Gate:** `scripts/check_public_api_snapshot.sh` keeps both surfaces
  green; `audits/V7_api.toml` records the contract.

### `vyre-ops` → `vyre-intrinsics`
- **Why:** Cat-C hardware intrinsics moved out of the catch-all
  `vyre-ops` crate into a focused `vyre-intrinsics` so the registry
  can be hand-audited (`docs/migration-vyre-ops-to-intrinsics.md`).
- **Path:** consumers depending on `vyre-ops::Atomic*` import from
  `vyre-intrinsics::*` instead. Cat-A ops moved to `vyre-libs`.
- **Status:** Complete in 0.6. The `vyre-ops` name is permanently
  dead.
- **Gate:** `scripts/check_layering.sh` blocks any new `vyre-ops`
  reference; `scripts/check_consumers.sh` ensures consumer / pyrograph /
  warpscan still build against the new surface.

### `dispatch(&[Vec<u8>])` → `dispatch_borrowed_into(&[&[u8]], &mut OutputBuffers)`
- **Why:** the owned-bytes API forces a heap allocation per output.
  Caller-owned output storage removes the alloc class entirely
  (audit item #4 + #10  -  `dispatch_borrowed_into` + allocation-count
  contract test).
- **Path:** new code calls `dispatch_borrowed_into`. Existing code
  keeps the legacy `dispatch` shim, which now wraps the borrowed
  surface internally.
- **Window:** `dispatch(&[Vec<u8>])` stays for the entire 0.6.x line;
  audit item #5 tracks the deletion across every backend.
- **Gate:** `scripts/check_no_hot_path_vec_vec.sh` blocks any NEW
  `Vec<Vec<u8>>` API on hot paths; the allocation-count contract test
  ratchets the steady-state alloc budget.

### `vyre-runtime --features remote` MSRV restoration
- **Why:** the remote-cache feature stack pulled `icu` 2.2 which
  required rustc 1.86  -  above the workspace MSRV.
- **Path:** lockfile pinned to `url 2.5.0` / `idna 0.5.0`, avoiding
  the icu transitive (audit item #27).
- **Status:** Fixed; gate verifies via
  `cargo check -p vyre-runtime --features remote`.

### Pipeline-cache config: hardcoded constants → Tier-A env vars
- **Why:** the in-memory pipeline-cache budget was a hardcoded
  constant; production deployments need to tune it without a recompile
  (audit item #18).
- **Path:** new `VYRE_PIPELINE_CACHE_ENTRIES` and
  `VYRE_PIPELINE_CACHE_BYTES` env vars surface through
  `WgpuBackendStats`. Compiled defaults remain.
- **Status:** Fixed; `vyre-driver-wgpu/CONFIG.md` documents Tier-A
  knobs as the canonical surface.

## Open source-change migrations

### `[lints] missing_docs` enforcement on `vyre-driver`
- **Why:** vyre-driver currently has `#![allow(missing_docs)]`
  overriding the workspace deny. Removing it surfaces real
  documentation gaps that block publish-readiness.
- **Path:** sweep the public surface, document each exported item, then
  drop the override in the same source patch.
- **Status:** open source work. This document is not marking the
  migration complete.

### `vyre-runtime` backend-neutralization
- **Why:** the runtime currently transitions through
  `vyre-driver-wgpu`. The audit (item #65) flags this as a layer
  inversion: runtime should be backend-neutral with an optional adapter
  crate.
- **Path:** introduce `vyre-runtime-wgpu-adapter` (or feature gate),
  remove the direct dep.
- **Status:** open source work. The direct dependency remains until the
  adapter boundary is implemented.

### Exposed-internal crate dependencies → workspace versions
- **Why:** some internal `path = "../foo"` deps lack a `version`,
  blocking a publishable workspace member from resolving its sibling
  from crates.io after a publish (audit item #72).
- **Path:** sweep all `path = "..."` deps, ensure each has the matching
  `version = "0.4.1"` clause.
- **Status:** open until the dependency gate reports zero publishable
  path-only internal deps.

## How to add a migration

1. Open an audit item describing the breaking change and the migration
   surface.
2. Add a section here documenting why, path, window, and gate.
3. Update `audits/V7_api.toml` if the public API signature changes.
4. Update the relevant gate script (`check_trait_freeze.sh`,
   `check_public_api_snapshot.sh`, `check_consumers.sh`) so the
   migration cannot regress silently.
5. Land all of the above in the same patch.

If a contributor's PR breaks the contract without an entry here, the
`scripts/check_release_signoff.sh` composite gate catches it and
fails the build.

## What "complete" looks like

A migration is complete when:
- The legacy surface has been deleted from the public API.
- Every consumer (vyre-libs, consumer, pyrograph, warpscan, dataflow consumer) builds
  cleanly without referencing the legacy surface.
- The gate that blocked the legacy pattern has been retained as a
  permanent invariant (not deleted).
- The entry above moves to a "Completed migrations" section with the
  release tag in which the deletion landed.
