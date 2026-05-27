# vyre architecture clarity audit (2026-05-01)

What's confusing today, what will tangle as the codebase grows, and where the boundaries need teeth (not just docs). **Lego-block enforcement is the headline.**

Companion to `CLEANUP_PLAN_2026-05-01.md` (org dups) and `PERF_ROADMAP_2026-05-01.md` (speed). This doc is about *architectural clarity*  -  the kind of stuff that doesn't fail a test today but will rot the codebase by next quarter if untouched.

Each section ends with a **proposed enforcement mechanism**, because docs alone don't hold the line.

---

## S0. Lego-block enforcement (TOP priority)

### The principle (from CLAUDE.md "crates of crates")

Three tiers compose:
- **Tier 1  -  IR primitives** (`vyre-ir`): `Node::Store`, `Node::let_bind`, `Expr::add`, raw IR atoms. Anyone can read; only Tier-2.5 should construct.
- **Tier 2.5  -  LEGO ops** (`vyre-primitives`): hand-built composites of Tier-1 atoms (e.g. `workgroup_tree_reduce`, `bitwise_and`, `attention_max_pass`). Exposed as builders. Tier-3 dialects compose Tier-2.5 LEGO blocks; they should never reach into Tier-1.
- **Tier 3  -  Dialects** (`vyre-libs::nn`, `vyre-libs::visual`, `vyre-libs::parsing`, `vyre-libs::security`, `vyre-libs::matching`): domain-shaped APIs (e.g. `gqa_attention`, `box_shadow`). Built by composing Tier-2.5.

### What's broken today

The principle is undefined in code. **Nothing prevents `vyre-libs::nn::gqa_attention` from emitting raw `Node::Store` instead of going through `vyre-primitives::storage::store_strided`.** Concretely:

- `vyre-libs/src/nn/attention/gqa_attention.rs:240` constructs `Node::Store { ... }` directly. Should go through a Tier-2.5 builder.
- `vyre-libs/src/visual/shadow/mod.rs` (the box_shadow op caught earlier today) constructs `Node::let_bind`, `Node::if_then`, raw `Expr::abs_diff`, etc. These are Tier-1 atoms.
- This is what the user calls "lego block isn't enforced and Codex is fixing now."

The cost of *not* enforcing:
- Two ops doing the same thing two different ways. Optimizer can't recognize either as a known shape.
- Tier-2.5 primitives become decorative  -  no one is forced to use them, so they bitrot.
- Bug fixes to Tier-2.5 (like the Region-scoping fix this session) don't propagate to ops that bypassed.
- Egglog rule LHS patterns become brittle: a rule matching the canonical `attention_max_pass` shape misses the four hand-rolled variants.

### Enforcement options (pick one or stack)

**Option A  -  Visibility-based (compile-time, strongest).**
- Make raw `Node::Store`, `Node::let_bind`, etc. constructors `pub(crate)` in `vyre-ir`. External crates can read these enum variants for matching but cannot construct.
- Move all "I am building IR" entry points into `vyre-primitives`. vyre-libs depends on vyre-primitives, not vyre-ir.
- **Cost**: every existing constructor call site in vyre-libs must be rewritten. Probably 2-4 weeks of agent work. consumer needs the same treatment.
- **Benefit**: violations become compiler errors. Hardest to evade.

**Option B  -  Provenance tags + validator gate (runtime, weaker but easier to land).**
- Every `Node` carries a `built_by: BuiltBy` enum tag (`PrimitiveBuilder | DialectComposition | RawConstructor`).
- Validator rejects any program containing `BuiltBy::RawConstructor` nodes that came from a `vyre-libs::*` source location.
- **Cost**: ~200 LOC validator change + tag plumbing.
- **Benefit**: catches violations at first build. Bypassable in principle, but each bypass is a visible ugly cast.

**Option C  -  CI lint via cargo-deny / cargo-vet / custom clippy lint.**
- Custom clippy lint: in any file under `vyre-libs/src/`, calls to `Node::Store`, `Node::let_bind`, `Expr::*` are forbidden.
- Same for `vyre-primitives/src/`: those ARE the LEGO blocks, so they can construct Tier-1, but they must not construct Tier-3 things.
- **Cost**: ~400 LOC clippy lint.
- **Benefit**: lint runs in CI; violations block merge; easy to whitelist for migration.

**Recommendation**: stack A + C. A is the structural fix; C catches drift before it's invisible.

### Proposed enforcement: **Option A + Option C**, with C landing first (faster, unblocks visibility).

---

## S1. Optimizer separation of concerns

### What's broken today

Inside `vyre-foundation/src/optimizer/`, passes don't have a stated input/output contract. `ConstFold` and `StrengthReduce` both walk `Expr` trees, both rewrite, neither declares "I assume the input has been canonicalized" or "my output guarantees no double-negation." The phase-4 ConstFold sweep is the smoking gun: it exists because a previous canonicalize moved a literal that ConstFold could now fold. Pass ordering is folklore.

When egglog (A6) lands, this dissolves  -  rules are orthogonal by construction. But the legacy passes will coexist with the egglog engine for some transition window, and the contract is still missing.

### Enforcement options

- **Pass invariants as types**: `Pass<Input = NormalizedIR, Output = CanonicalIR>` with phantom-type tags on Program. Costly, idiomatic Rust.
- **Pass invariants as runtime assertions** (debug-only): each pass declares `requires(p)` and `ensures(p)` predicates checked at boundaries.
- **Pass scheduler with explicit topology**: scheduler refuses to run pass B after pass A unless A.ensures ⊇ B.requires.

### Proposed enforcement: runtime `requires`/`ensures` predicates in debug builds, panic on violation.

Plus: **archive the legacy pass list** as soon as A6 ships. Don't let it become parallel-truth.

---

## S2. Three OpEntry registries  -  collapse or document?

### What's broken today

- `vyre-harness::OpEntry`             (Cat-A, general ops)
- `vyre-intrinsics::harness::OpEntry` (Cat-C, hardware intrinsics)
- `vyre-primitives::harness::OpEntry` (Tier-2.5, LEGO primitives)

Three different types. Three separate `inventory::collect!` slots. Three places to fix any cross-cutting concern (e.g. the box_shadow stub `expected_output` issue from earlier today  -  if box_shadow was a Tier-2.5 primitive, the same fix would need to be done in *its* harness file, not in Cat-A).

### Options

- **Collapse to one** `OpEntry` with `category: Category` field. Simpler.
- **Keep three but unify behind a `trait HarnessEntry`**  -  Cat-A, Cat-C, Tier-2.5 each implement; the universal_harness walks the union via the trait.
- **Document the split** (cleanup-O5) and live with it.

### Proposed enforcement: **collapse to one `OpEntry` with a `category: Category` field.**

Reason: every cross-cutting fix today is a 3x cost. The "categories are different concerns" defense doesn't hold up  -  all three want the same fixture-truth invariants, the same harness logic, the same telemetry.

---

## S3. Lowering responsibility boundary

### What's broken today

`vyre-driver-cuda` and `vyre-driver-wgpu` each do **lowering + emit + dispatch**. There's no clear contract between "lower IR to backend-neutral kernel descriptor" and "emit kernel descriptor to backend artifact (PTX / naga IR / SPIRV)."

This means:
- Common lowering code is duplicated or drifts between drivers.
- Substrate-aware-but-driver-agnostic optimizations (B14 memory coalescing, B12 shared-memory promotion) have no home  -  they want to live "between lower and emit" but that boundary doesn't exist.
- Adding a new backend (e.g. Metal, ROCm) means re-implementing lowering from scratch.

### Proposed boundary

```
vyre-ir → vyre-opt → vyre-lower → KernelDescriptor → vyre-emit-{naga,ptx,spirv} → driver
                                                       ↑
                                       backend-specific emit-time analyses
                                       (vec4 packing, coalescing, tensor cores)
                                       live HERE, not in the driver
```

`KernelDescriptor` is a substrate-neutral kernel IR (binding layout, dispatch shape, kernel body in lowered form)  -  *not* the same as `Program`. Drivers stay thin: take a backend artifact + bind buffers + dispatch.

### Proposed enforcement

- New `vyre-lower` crate with pub `KernelDescriptor` type.
- New `vyre-emit-*` crates per backend.
- Drivers depend on both: `vyre-driver-wgpu` = `vyre-emit-naga` + wgpu-specific dispatch.
- vyre-driver (the shared layer) holds boilerplate that ALL drivers reimplement (next section).

---

## S4. `vyre-driver` is shared driver code, not just a facade

### What I had wrong earlier

I described `vyre-driver` as "the facade that selects backend at runtime." That's part of it, but the more important role: **everything that CAN be shared between drivers lives here so each driver doesn't reimplement.**

Concrete examples of stuff that belongs in vyre-driver (and is currently duplicated or could become duplicated):
- Binding-set construction boilerplate.
- Command-buffer / command-encoder recording boilerplate.
- Push-constant packing.
- Dispatch-shape computation (workgroup count from total threads).
- Error normalization (driver-specific errors → vyre `DriverError`).
- Buffer-handle traits (`trait Buffer { fn size() -> usize; ... }`)  -  drivers implement with their concrete handle.
- Pipeline cache abstractions (vyre-driver-wgpu has a disk_cache; CUDA may want one too).
- Telemetry / event-recording API.

### Is the move mechanical or hard?

**Mostly mechanical, with one wrinkle.**

The mechanical 80%:
1. `git grep` each shared concern across drivers (`fn create_bind_group`, `fn record_dispatch`, `fn pack_push_constants`, etc.).
2. Identify the common shape; extract to vyre-driver as a `pub` module.
3. Rewrite each driver to call the shared helper.
4. Build gate.

The wrinkle (20%): **each driver has its own typed handles** (cuda::Buffer ≠ wgpu::Buffer). Solution is the standard Rust pattern: define generic-over-handle traits in vyre-driver, each driver implements the trait with its concrete handle type. Helpers in vyre-driver are generic over the handle.

```rust
// vyre-driver/src/buffer.rs
pub trait DeviceBuffer {
    fn size_bytes(&self) -> usize;
    fn write(&mut self, offset: usize, data: &[u8]) -> Result<(), DriverError>;
}

// vyre-driver/src/binding.rs
pub fn build_bind_group<B: DeviceBuffer>(layout: &BindLayout, buffers: &[B]) -> BindGroupDesc {
    /* shared logic */
}

// vyre-driver-cuda implements DeviceBuffer for its CUDA buffer
// vyre-driver-wgpu implements DeviceBuffer for wgpu::Buffer
```

**Effort estimate**: 1-2 weeks of agent work once the `vyre-emit-*` crates exist. Without those, the shared-driver-code extraction has to coexist with the lowering duplication  -  uglier, but still doable.

### Proposed sequence

1. Land `vyre-emit-naga` + `vyre-emit-ptx` (extracts emit out of drivers).
2. Land `vyre-driver` shared traits + helpers (extracts boilerplate out of drivers).
3. Drivers shrink to ~200-400 LOC each: just dispatch + buffer-handle + pipeline-handle.

---

## S5. Region nodes are still IR, should be sidecar

### What's broken today

`Node::Region { body, .. }` is an IR node. We just spent a session figuring out that it's a tracing/grouping marker, not a scoping construct. Validator now treats it correctly. But:

- Optimizer still walks Region nodes (extra work).
- Two crates (vyre-libs/region, vyre-intrinsics/region, vyre-harness/region) re-export wrap_anonymous/wrap_child.
- Future contributors will hit the "is Region a scope?" confusion again because the node is still in the enum.

### Proposed enforcement

- Region disappears from `Node` entirely.
- Tracing/grouping is a side-table: `Program` carries an optional `regions: Vec<RegionSpan>` keyed by node-index ranges.
- `wrap_anonymous` / `wrap_child` push to the side-table; nothing in the IR enum changes.
- Optimizer never sees Regions; diagnostics consult the side-table.

**Cost**: ~600 LOC across vyre-ir + the four region.rs shims + every consumer that pattern-matches on `Node::Region`. Probably 3-5 days of agent work.

**Benefit**: an entire bug class disappears.

---

## S6. `vyre-runtime` vs `vyre-driver-megakernel` overlap

### What's broken today

Megakernel composition logic lives partly in vyre-runtime/src/megakernel/ and partly in vyre-driver-megakernel. The boundary leaks: composition decisions get made in both places, and `fuse_programs` is called from runtime but barrier insertion logic lives in the driver.

### Proposed: fold `vyre-driver-megakernel` into `vyre-runtime`

- vyre-runtime owns: composition (fuse_programs), barrier insertion, megakernel materialization, dispatch ordering, caching.
- The "driver" suffix implies a hardware backend; megakernel is logical, not hardware. Misnamed.

**Cost**: medium. A bunch of imports break. ~3 days of agent work.

---

## S7. `matching` appears as a sub-module twice

`vyre-libs/src/matching/` AND `vyre-primitives/src/matching/`. Two different concerns:
- `vyre-libs/matching/`      -  Tier-3 dialect: high-level pattern-matching API (DFA scan, regex, literal scan).
- `vyre-primitives/matching/`  -  Tier-2.5 LEGO: low-level matching primitives (region.rs lives here).

The name overlap is confusing. Possibilities:
- Rename `vyre-primitives/matching/` → `vyre-primitives/match_atoms/` or `vyre-primitives/regions/`.
- Rename `vyre-libs/matching/` → `vyre-libs/scan/` (matches scan_dfa/scan_literal naming used in vyre-reference).

### Proposed: rename `vyre-libs/matching/` to `vyre-libs/scan/`

Less invasive (libs is the dialect layer, naming there is more flexible). Aligns with existing `scan_dfa.rs`, `scan_literal.rs`, `scan.rs` in vyre-reference/primitives.

---

## S8. Validator error codes (V001, V008, V032, etc.) are undocumented

A new contributor seeing a `V032 duplicate sibling` error has nowhere to look up what V032 means or how to fix it. They'll grep the source and read code.

### Proposed: `vyre-ir/VALIDATOR_ERRORS.md`

Table of every Vxxx code with: short description, common cause, recommended fix. Owned by vyre-ir crate, lives next to the validator source.

**Cost**: 1 day of agent work (read the validator, list the codes, write the doc).

---

## S9. Tag-bit families split across two u64 words

`pg_node_tags` (low) + `pg_node_tags_high` (high). Bits get assigned to whichever word has space. The choice is invisible to readers  -  bit 30 is in low, bit 36 is in high (yes, that's the REASSIGN tag mismatch we hit on 2026-04-30).

### Proposed enforcement

- **Single `pg_node_tags: u128` field.** No high/low. Bits never move.
- Or: per-Program tag bitset (item A4 in the perf roadmap), which makes the question moot.

**Cost**: medium  -  every consumer of `pg_node_tags_high` needs migration. ~500 LOC.

---

## S10. Inventory-based OpEntry registration is invisible to the type system

If a contributor adds a new op but forgets the `inventory::submit!` block at the bottom, the op silently isn't tested. There's no compile-time check.

### Proposed enforcement

- **Derive macro `#[vyre_op]`** that auto-generates the `inventory::submit!`. Forgetting the macro is visible (function builds the op but isn't registered → easy lint).
- **CI gate**: `cargo test -p vyre-libs --test universal_harness -- --list` enumerates registered ops. A grep for `pub fn` in `nn/`, `visual/`, etc. minus the registered list = unregistered ops. Fail CI.

---

## S11. vyre-driver feature flags inconsistency

`vyre-driver-cuda` is gated behind `cuda` feature in some places, behind a runtime flag in others. Same for wgpu. Mixed patterns:
- vyre's facade selects at runtime.
- But some downstream Cargo.toml gates vyre-driver-cuda behind a feature.
- And some test files conditionally compile.

**Proposed**: pick one  -  runtime selection (current vyre-driver design)  -  and remove all build-time feature gating around drivers. Drivers are always compiled; runtime selects.

---

## S12. Examples use path deps; can't catch publish-shape regressions

`examples/external_ir_extension/Cargo.toml` uses `vyre = { path = "../../vyre-core" }`. If we publish vyre crates and someone follows the example, the path dep won't work. Worse, the example currently can't catch regressions in the published-crate API surface.

**Proposed**: examples consume **published versions** of vyre crates (after first publish), with a `[patch.crates-io]` block at workspace root for local-development convenience. Examples then double as integration tests of the published surface.

---

## S13. Workspace member listing

Root `Cargo.toml` either lists every crate explicitly or uses `members = ["*"]`. Quick audit needed; whichever is currently used, document the choice and stick to it.

---

## Summary  -  what to enforce, in what order

| # | Item | Mechanism | Priority | Effort |
|---|------|-----------|----------|--------|
| S0 | Lego-block tier visibility | clippy lint (CI) + visibility (`pub(crate)`) | **TOP** | 2-4 weeks |
| S1 | Optimizer pass invariants | runtime `requires`/`ensures` (debug) | high | ~500 LOC |
| S2 | Single `OpEntry` with category tag | refactor | medium | ~600 LOC |
| S3 | `KernelDescriptor` between lower and emit | new crate | high (enables S4) | ~3000 LOC |
| S4 | vyre-driver as shared driver code | extract boilerplate, generic-over-handle | high (after S3) | 1-2 weeks |
| S5 | Region as sidecar, not IR node | refactor | medium | ~600 LOC |
| S6 | Fold vyre-driver-megakernel into vyre-runtime | rename + import | low | ~3 days |
| S7 | Rename `vyre-libs/matching/` → `vyre-libs/scan/` | rename | low | ~1 day |
| S8 | Validator error code documentation | docs | low | ~1 day |
| S9 | Single u128 tag field | refactor | medium | ~500 LOC |
| S10 | `#[vyre_op]` derive macro + CI gate | macro + lint | medium | ~400 LOC |
| S11 | Drop build-time driver feature gates | refactor | low | ~200 LOC |
| S12 | Examples consume published crates | refactor | low | ~150 LOC |
| S13 | Workspace member list discipline | docs | trivial | <1 hour |

---

## What this prevents from rotting

- Dialect ops bypassing primitives → optimizer can't see them as known shapes → egglog rules become brittle (S0).
- Pass ordering becoming folklore again as the egglog/legacy boundary shifts (S1).
- Three-place fixture fixes for cross-cutting concerns (S2).
- New backends being a from-scratch reimplementation (S3, S4).
- Region-style "what does this node mean again?" bugs (S5).
- Tag bit confusion (S9).
- Silently-unregistered ops (S10).
