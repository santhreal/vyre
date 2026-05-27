# Crate-extraction migration plan

P-MIGRATE-3 — deferred crate splits and what blocks them.

## Why split

The current vyre topology has four large crates (`vyre-foundation`,
`vyre-driver`, `vyre-libs`, `vyre-runtime`) plus the per-backend
drivers. Three categories of code inside those large crates have
grown to the point where extraction would pay for itself:

1. Category-theoretic substrate (Cat). Sits inside
   `vyre-foundation/src/cat_substrate/` and
   `vyre-driver/src/self_substrate/cat/`. Used by callers who don't
   want the rest of vyre.
2. Type-level wire format (Types). Frozen wire enums + opaque
   payload helpers in `vyre-foundation/src/ir_inner/model/types/`.
   Pure data-contract crate; consumers like `vyre-spec`
   re-export it but everyone takes the whole foundation today.
3. Provenance closure (Prov). Self-substrate consumer in
   `vyre-driver/src/self_substrate/scallop_provenance/`. Has its
   own paper-implementation surface that makes sense as a
   standalone Rust crate.
4. Scaling primitives (Scale). Heterophilic-cluster + sheaf
   diffusion lives in `vyre-runtime/src/megakernel/scaling.rs` +
   `vyre-runtime`. Re-exportable as `vyre-scale` for
   callers building their own megakernels.

## Deferred crates

### `vyre-cat` (P-CRATE-2)

**Contents**: `vyre-foundation/src/cat_substrate/`,
`vyre-driver/src/self_substrate/cat/`, region-chain composition
helpers from `vyre-libs/src/region.rs`.

**Blocks**: the cat substrate currently shares an inventory
registry with vyre-driver's main backend list. Splitting requires
the inventory to be surfaced as an injectable trait so cat
consumers can register without pulling vyre-driver.

### `vyre-types` (P-CRATE-3)

**Contents**: `vyre-foundation/src/ir_inner/model/types/`,
`vyre-spec`'s re-exports of those types.

**Blocks**: nothing structural — vyre-types could ship today as a
new crate in the workspace. The reason it hasn't is downstream:
every consumer of `vyre::ir::types` would re-import from
`vyre_types::*`. That's a 200-site rewrite. Hold until the rest of
the foundation reorg lands so we do it in one sweep.

### `vyre-prov` (P-CRATE-4)

**Contents**:
`vyre-driver/src/self_substrate/scallop_provenance/`,
`vyre-driver/src/provenance/` plus the closure-evaluation harness
in `vyre-harness/tests/provenance_closure.rs`.

**Blocks**: provenance is a hot consumer of vyre-driver internals
(routing table, validation cache). Need to expose those via stable
trait hooks before it can stand alone.

### `vyre-scale` (P-CRATE-5)

**Contents**: `vyre-runtime/src/megakernel/scaling.rs`,
heterophilic-cluster detection, sheaf diffusion wiring,
megakernel autotuner.

**Blocks**: scaling currently calls into `vyre-runtime`
private modules. Make those `pub` (or expose a stable scaling
API) before extracting.

## Sequencing

1. `vyre-types` first — pure re-export, lowest risk, unlocks
   independent versioning of the wire format.
2. `vyre-prov` second — has a self-contained test surface in
   `vyre-harness`; easiest to assert correctness offline.
3. `vyre-cat` third — needs the inventory injection refactor
   first.
4. `vyre-scale` last — depends on the megakernel API
   stabilising post-launch.

## Cross-cutting work

- Each extraction must add a `#[cfg(feature = "in_<crate>")]` shim
  in the parent crate so existing import paths keep resolving for
  one release cycle.
- Each extraction adds a `cargo publish --dry-run` step to
  `scripts/publish_dryrun.sh`.
- Each extraction adds an entry to `RECURSION_THESIS.md` recording
  which substrate consumers moved out.
