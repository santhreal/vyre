# CC-owned backlog (2026-05-01)  -  SUPERSEDED 2026-05-02

> **This document is evidence-only.** The canonical executable roadmap is now
> [`docs/optimization/ROADMAP.md`](docs/optimization/ROADMAP.md); active claims
> live in [`docs/optimization/CLAIMS.toml`](docs/optimization/CLAIMS.toml);
> lane assignment is in [`docs/optimization/OWNERSHIP.toml`](docs/optimization/OWNERSHIP.toml);
> patch contract is in [`docs/optimization/AGENT_CONTRACT.md`](docs/optimization/AGENT_CONTRACT.md).
> Read [`docs/optimization/START_HERE.md`](docs/optimization/START_HERE.md) first.
>
> The structural seeds (SEED-1..6) below remain useful as historical context for
> sequencing decisions but are not authoritative against the lane-based
> ownership model. The "Active claims" table below is frozen  -  new claims go
> in `CLAIMS.toml`.

Per the 2026-05-01 reframe, Jules touches only tests/fixtures/data. Everything that modifies source-code semantics is mine. This file is the master index so context compaction doesn't lose it.

Source-of-truth docs (read these for full context on each item):
- `PERF_ROADMAP_2026-05-01.md`  -  speed work, items M0..L4
- `SEPARATION_AUDIT_2026-05-01.md`  -  architecture clarity, items S0..S13
- `CLEANUP_PLAN_2026-05-01.md`  -  org dups (cleanup-plan agent owns; not me)

## Sequencing

1. **Five structural seeds first** (the foundation everything else builds on).
2. **Folder-structure refactor** in parallel (mostly mechanical renames).
3. **Optimizer items, lowering/emit items, runtime items** in parallel waves once seeds land.
4. **GPU-specific perf craters** (B12, B13, B14) as soon as KernelDescriptor exists.

## Active claims (CC, live)

Always update before/while working an item  -  peers and future-CC read this on cold-start to avoid double-claiming.

| Started | Item | File(s) | Headline goal |
|---------|------|---------|---------------|
| 2026-05-02 | **cat_a_gpu_differential 9 failing entries**  -  Real test output (run 2026-05-02 10:21). Standalone `vyre-libs::hash::fnv1a64` PASSES (prior "loop body broken" hypothesis was wrong). Failing: (1) `vfs::resolve consumer_a/b`  -  "unknown local `file_hash`" IR builder bug, variable read before bind; (2) `nn::linear`  -  CPU=1024 / GPU=16 buffer length, 64× under-dispatch; (3) `optim::ema_apply`  -  1-ULP f32 divergence at one lane (0x42206666 vs 0x42206667), likely FMA/operand-reorder; (4) `logit_softcap`, `newton_schulz_poly5_f32 consumer_a/b`, `catalog::hash::fnv1a64 consumer_a/b`  -  caught panics, specific divergence not printed yet. Working order: vfs::resolve first (cleanest, IR-construction), then linear (dispatch math), then ema_apply (f32 ULP), then drill into the unprinted catalog-consumer divergences. | `vyre-libs/src/catalog/vfs/resolve.rs`, `vyre-libs/src/nn/linear.rs`, `vyre-libs/src/optim/ema_apply.rs`, `vyre-libs/src/catalog/`, `vyre-driver-wgpu/tests/cat_a_gpu_differential.rs` | Closes 9 of the open cat_a_gpu_differential failures. Headline: turn `diff_universal_registry` from FAILED → ok. |
| 2026-05-02 | **PERF M5 added**  -  Tier B device signature TOML (`vyre/devices/*.toml`). Spec landed; loader + first device file (sm_120 / Blackwell) not yet built. | `PERF_ROADMAP_2026-05-01.md` §M | Unblocks A6 egraph cost vector once built. |

vyre-libs HOLD lifted by user 2026-05-02  -  CC may now touch `vyre-libs/`. `vyre-primitives/` and `vyre-reference/` HOLD status: assume open unless user re-flags. `vyre-driver/`, `vyre-driver-cuda/`, `vyre-driver-wgpu/` were already open during the May-1 session and remain so.

## Five structural seeds (priority order  -  must land first)

| # | Item | Source | Priority | Status |
|---|------|--------|----------|--------|
| SEED-1 | Lego-block clippy lint (`vyre-lints` crate, forbids raw `Node::*` / `Expr::*` in `vyre-libs/src/`) | SEPARATION S0 | P0 | not started |
| SEED-2 | Hash-cons Expr (slab + 32-bit ids; `Expr::clone()` becomes `Copy`) | PERF A1 | P0 | not started |
| SEED-3 | SoA columnar Program (opcode column + operand-id columns) | PERF A2 | P0 | not started, **depends on SEED-2** |
| SEED-4 | Egglog engine (schema, indexing, saturation budget, extraction, applicability tags) | PERF A6 | P0 | not started, **depends on SEED-2 + SEED-3** |
| SEED-5 | Lowering boundary + `KernelDescriptor` type + wgpu reference conversion | SEPARATION S3 | P0 | not started |
| SEED-6 | vyre-driver shared-code traits (`DeviceBuffer`, `DevicePipeline`) + one driver converted as reference | SEPARATION S4 | P0 | not started, **depends on SEED-5** |

(I called it "five seeds" earlier but vyre-driver shared traits was always implicit; calling it out explicitly as SEED-6.)

## Optimizer items  -  section A of PERF_ROADMAP

### A.1 Structural rewrites

- A1  -  hash-cons Expr **(SEED-2)**
- A2  -  SoA Program **(SEED-3)**
- A3  -  strip Region pre-optimize, restore as side-table for diagnostics
- A4  -  tags as per-Program bitsets
- A5  -  validator skip-cache (also K1)

### A.2 Egglog rewrite

- A6  -  egglog engine **(SEED-4)**
- A7  -  cost-aware extraction (part of SEED-4 deliverable)
- A8  -  saturation budget per rule family
- A9  -  applicability predicates (rule tags)
- A10  -  GPU-resident e-graph (deferred until measurement says CPU saturation is the bottleneck)

### A.3 Wire existing analyses into optimizer

- A11  -  reaching-defs → ConstFold across CFG
- A12  -  points-to → memory-side optimization
- A13  -  escape analysis → buffer-storage reuse across megakernel arms (also C2)
- A14  -  live-range + register-pressure → rematerialization
- A15  -  buffer aliasing → load elision
- A16  -  range analysis → cast/branch elision

### A.4 Classical passes we lack

- A17  -  LICM
- A18  -  GVN (subsumed by SEED-4 if egglog lands first)
- A19  -  predicate hoisting (subsumed by SEED-4)
- A20  -  dead-store elimination
- A21  -  dead-load elimination
- A22  -  store-to-load forwarding (depends on A12)
- A23  -  branch coalescing
- A24  -  phi/select coalescing
- A25  -  boolean simplification
- A26  -  loop fusion
- A27  -  loop fission (enables A39 vectorization)
- A28  -  loop peeling
- A29  -  loop strip-mining (enables A39)
- A30  -  polyhedral transformations (long-term)
- A31  -  software pipelining
- A32  -  tail duplication
- A33  -  algebraic identity expansion (subsumed by SEED-4 rule DB)
- A34  -  strength reduction expansion (Horner, shifts, FMA)
- A35  -  range-based folding (depends on A16)
- A36  -  atomic minimization (verify NormalizeAtomicsPass completeness)

## Lowering / emit  -  section B of PERF_ROADMAP

### B.1 Naga / wgpu emit

- B1  -  vec2/vec4 packing (depends on SEED-3)
- B2  -  naga IR caching at Module level
- B3  -  parallel naga emit per arm
- B4  -  pipeline reflection pre-warm during canonicalize
- B5  -  wgpu disk cache key fix (per-arm hash)

### B.2 CUDA / PTX emit

- B6  -  tensor-core (wmma/mma) fragment promotion
- B7  -  ldmatrix / cp.async for async tile loads
- B8  -  predicated execution for short divergent branches
- B9  -  PTX-level instruction scheduling

### B.3 Cross-substrate

- B10  -  constant-buffer promotion
- B11  -  texture-memory promotion
- B12  -  **shared-memory promotion (5-50x on tiled ops)**  -  top-3 priority
- B13  -  shared-memory bank-conflict avoidance (depends on B12)
- B14  -  **memory-coalescing analysis (up to 32x on memory-bound ops)**  -  biggest single unrealized speedup
- B15  -  workgroup-uniform branch detection

## Megakernel / runtime  -  section C of PERF_ROADMAP

- C1  -  whole-megakernel egraph (in SEED-4)
- C2  -  scratch-buffer reuse across arms (= A13)
- C3  -  shared prologue extraction
- C4  -  barrier elision for value-flow chains (depends on SEED-4)
- C5  -  three-arm gid-gated middle-arm pattern (in SEED-4; recall-bug class)
- C6  -  pipeline reuse cache hit-rate audit

## Dispatch / driver  -  section D of PERF_ROADMAP

- D1  -  persistent kernel mode (analyses that run thousands of times per scan)
- D2  -  CUDA streams / wgpu queues for independent megakernel arms
- D3  -  async memcpy overlap with compute
- D4  -  CUDA graphs / wgpu command bundles
- D5  -  multi-kernel concurrent launch where occupancy permits
- D6  -  bind-group reuse across launches
- D7  -  push-constant inlining
- D8  -  indirect dispatch
- D9  -  bindless textures / buffers

## Compile-time / cold-start  -  section E

- E1  -  module-level naga cache (= B2)
- E2  -  LRU AST cache (overlaps with Jules ticket but engine side is mine)
- E3  -  incremental re-optimization (depends on SEED-2)
- E4  -  CUDA module persistent across runs
- E5  -  PTX cache shared across processes

## Specialization  -  section F

- F1  -  shape specialization (compiles per-shape variants)
- F2  -  buffer-content folding (constant tables)
- F3  -  type specialization
- F4  -  backend-capability specialization

## Numerical  -  section G

- G1  -  mixed-precision auto-downcast f32→f16 (depends on A16)
- G2  -  reciprocal approximation (1/x → fast_inv)
- G3  -  FMA pattern matching (subsumed by SEED-4)
- G4  -  Horner's rule for polynomial expressions
- G5  -  range-reduced transcendentals
- G6  -  Welford's algorithm for sum-of-squares
- G7  -  block-FMA reduction for accumulation

## Algorithm-level  -  section H

- H1  -  Strassen-like substitution for matmul
- H2  -  FFT for convolution
- H3  -  im2col / direct-conv decision
- H4  -  flash-attention fusion
- H5  -  gemm + bias + activation fusion (TASO/PET pattern)

## PGO / autotune  -  section I

- I1  -  PGO-style hot-path recording
- I2  -  trace-based JIT specialization
- I3  -  autotuning database (persistent)
- I4  -  occupancy-aware autotuning

## Layout / data  -  section J

- J1  -  buffer layout transformation (AoS→SoA at data level)
- J2  -  padding for bank-conflict avoidance (overlaps B13)
- J3  -  buffer alignment hints

## Validator  -  section K

- K1  -  validator skip-cache (= A5)
- K2  -  many sanity checks debug-only
- K3  -  tag-bit assertions guarded (depends on A4)

## Frontend  -  section L

- L1  -  single-pass C lexer
- L2  -  persistent parsed-AST cache (Jules has the data side; engine is mine)
- L3  -  parallel parse across files
- L4  -  lazy scope resolution

## Architecture clarity  -  SEPARATION_AUDIT items

- S0  -  lego-block enforcement: clippy lint **(SEED-1)** + visibility tightening (`pub(crate)` on raw constructors in vyre-ir)
- S1  -  optimizer pass invariants (`requires`/`ensures` predicates, debug-checked)
- S2  -  collapse three OpEntry registries to one with `category: Category` field
- S3  -  KernelDescriptor + lower/emit boundary **(SEED-5)**
- S4  -  vyre-driver shared traits **(SEED-6)**
- S5  -  Region as sidecar, not IR node
- S6  -  fold vyre-driver-megakernel into vyre-runtime
- S7  -  rename `vyre-libs/matching/` → `vyre-libs/scan/`
- S9  -  single u128 tag field (replaces low/high split)
- S10  -  `#[vyre_op]` derive macro + CI gate for unregistered ops
- S11  -  drop build-time driver feature gates (runtime selection only)
- S12  -  examples consume published crates (with [patch.crates-io] for local dev)
- S13  -  workspace member listing discipline

## Folder-structure refactor (mostly mechanical, parallelizable)

- vyre-foundation → split into `vyre-ir` (just IR + Program + validate) + `vyre-opt` (optimizer)
- the dataflow consumer → promoted to top-level published crate (out of `vyre-libs/dataflow/`)
- `vyre-lower/` (new)  -  common lowering, IR → KernelDescriptor
- `vyre-emit-naga/` (new)  -  naga emitter
- `vyre-emit-ptx/` (new)  -  PTX emitter
- `vyre-emit-spirv/` (new)  -  SPIRV emitter
- `vyre-runtime/` absorbs `vyre-driver-megakernel/`
- `vyre-driver/` becomes the shared-driver-code crate (per the 2026-05-01 user clarification)
- `vyre-frontend-c/` → `vyre-frontend-c/`
- `vyre-core/` stays as-is (the GitHub repo is `vyre/` and the local crate dir naming `vyre-core/` for the package named `vyre` is intentional  -  repo structure matches local layout; renaming would diverge them)
- `vyre-reference/` reorganized per CLEANUP_PLAN_2026-05-01.md

## What I am NOT doing

- Anything in `vyre/jules_tickets/` (Jules owns that queue: tests, fixtures, CVE corpus, validator error docs).
- The cleanup-plan in flight (the cleanup-plan agent owns it).

## Order of attack today

**HOLD scope (set 2026-05-01, expanded after audit of last 6h of commits):**
Codex is actively touching:
- `vyre-libs/`, `vyre-primitives/`, `vyre-reference/` (lego-block migration + dual_impls reorg)
- `vyre-driver/`, `vyre-driver-cuda/`, `vyre-driver-wgpu/` (backend/dispatch/emit refactor)
- `vyre-runtime/`, `vyre-foundation/` (likely reads/writes during composition catalog wiring)

**Do NOT touch any of those crates until Codex stops or user signals go.** CC's lane during the hold is:
- New crates that don't exist yet (vyre-lints DONE, vyre-lower DONE, vyre-emit-naga DONE)
- New substrate-aware analysis crates that operate on KernelDescriptor (coalesce, shared-mem, bank-conflict)
- Jules ticket queue bulk-fill (test/fixture/CVE data only)
- Documentation, design docs, migration guides
- Anything outside `Santh/libs/performance/matching/vyre/` (NOT in scope)

1. **SEED-1 done**  -  vyre-lints crate exists, 24/24 tests pass. Allowlist intentionally empty until Codex's migration settles (snapshotting now would be stale in minutes). Not yet enabled in CI for the same reason.
2. **SEED-5 in flight** (KernelDescriptor + wgpu reference conversion). Touches `vyre-lower/`, `vyre-emit-naga/`, `vyre-driver-wgpu/` only  -  zero collision with Codex.
3. **SEED-6 after SEED-5** (vyre-driver shared traits).
4. **SEED-2 + SEED-3** (hash-cons + SoA). Touches `vyre-foundation/` (= vyre-ir + vyre-opt split). Some collision risk if Codex's work stretches into IR layer; check first. Bigger blast radius; do AFTER the simpler seeds land so I'm not fighting two big refactors at once.
5. **SEED-4** (egglog engine). After SEED-2+3 because it depends on the new IR shape.
6. **Then sweep through A.3 (wire the dataflow consumer into optimizer), B.3 GPU craters (B12, B13, B14), folder refactor in parallel.**
7. **Long tail.**

## Anti-loss markers

If context compacts and I forget anything, this file is the spine. Read it on cold-start before any work.

Cross-references for cold start:
- `PERF_ROADMAP_2026-05-01.md` (master roadmap)
- `SEPARATION_AUDIT_2026-05-01.md` (architecture)
- `CLEANUP_PLAN_2026-05-01.md` (cleanup-plan agent's territory)
- `jules_tickets/README.md` (Jules's territory)
- `CHANGELOG.md` Unreleased section (running landing log)
