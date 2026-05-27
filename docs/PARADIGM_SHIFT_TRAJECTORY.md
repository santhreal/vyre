# Vyre release trajectory  -  paradigm-shift source map

> "Math we ship as ops becomes math we use to compile ops. Compounding."
>  -  RECURSION_THESIS.md

This document is the architectural commitment for the next major vyre
release line.
It enumerates the paradigm shifts that distinguish vyre from "another
GPU IR" and trace the decisions every substrate change must validate
against.

This is not a wish list. Every item below has a vyre-self consumer
shipped or scheduled, and the math frontier doc (`MATH_FRONTIER.md`)
identifies the primitive that powers it.

---

## The thesis

**Vyre is the GPU substrate where frontier math is the dispatch
model itself, not just a library on top of it.**

Three classes of substrate where vyre is the only GPU IR that can
express what these classes need:

1. **Heterophilic structures**  -  dispatch graphs whose nodes have
   incompatible feature spaces (compute-bound vs memory-bound vs
   control-flow). Vyre's sheaf-Laplacian dispatch analyzer (#31
   `sheaf_heterophilic_dispatch`) is the first GPU substrate to
   model this correctly. GNN-style homophilic diffusion that every
   other GPU IR uses is mathematically wrong for heterophilic data.

2. **Recursive math substrate**  -  primitives that ship to user
   dialects AND replace ad-hoc heuristics in vyre's compiler. Each
   self-wiring compounds: sinkhorn clusters the dispatch graph,
   tensor_train fuses the chain-shaped Region clusters,
   matroid_intersection picks the discrete fusion subset, AMG
   V-cycle solves the inner linear systems, and vyre's scheduler
   gets faster every time we ship a new primitive. After 60
   self-consumers vyre is self-improving.

3. **Categorical IR**  -  the Region tree IS a string diagram in
   `Cat(GPU buffers, Programs)`. Optimizer rewrites are
   string-diagram rewrites. The 30-pass hand-managed dependency DAG
   becomes a typed functor category where pass ordering, validity,
   and reuse derive from algebra (#52 `functorial_pass_composition`,
   #53 `string_diagram_ir_rewrite`).

---

## Paradigm-shift source batch

Each entry has a Tier-2.5 primitive already on disk + a vyre-self
consumer scaffolded in `vyre-libs::self_substrate`. The required source change is
*lifting these from "wired but unused" to "the substrate's only
implementation."*

### S1. Sheaf-Laplacian dispatch scheduler

**Status**: primitive shipped (`vyre-primitives::graph::sheaf`),
self-consumer shipped (`sheaf_heterophilic_dispatch`).

**Required source change**: replace the existing dispatch-graph clustering in
`vyre-runtime megakernel`'s scheduler with sheaf-diffusion-driven
heterophilic-aware clustering. Today scheduling treats all Regions
as comparable; with sheaf diffusion the scheduler refuses to
fuse type-incompatible Regions in the first place  -  saving the
per-fusion compatibility check entirely.

**Deliverable**: `vyre-runtime megakernel/src/schedule/sheaf.rs`  - 
the production scheduler, calling our self-consumer module. The
old isotropic clustering becomes test-only / deprecated.

### S2. Three-layer recursive solver: scheduler → matroid → AMG

**Status**: all three primitives shipped, all three self-consumers
shipped (`megakernel_schedule`, `matroid_megakernel_scheduler`,
`multigrid_matroid_solver`).

**Required source change**: connect them end-to-end.
`schedule_via_homotopy` produces fractional fusion indicators →
`max_fusion_subset` rounds via matroid intersection → each
augmenting iteration calls `matroid_solve_step` (AMG Jacobi).
Wire as one production call `select_fused_subset(programs) → Vec<u32>`
in `vyre-runtime/src/megakernel/planner.rs`.

**Deliverable**: a single 100-line entry point that replaces the
hand-curated fusion heuristic in `optimizer/passes/fusion.rs`.

### S3. Categorical IR substrate

**Status**: primitives shipped (`functorial`, `string_diagram`,
`monoidal_compose`); self-consumers shipped (`functorial_pass_composition`,
`string_diagram_ir_rewrite`).

**Required source change**: add a `vyre_foundation::transform::category` module
that wraps every existing pass as a categorical functor with
declared source/target IR-views. Pass dependencies derive
automatically from view-compatibility.

**Deliverable**: every pass in `vyre-foundation/src/optimizer/passes/`
declares its `IRView`-source and `IRView`-target. The pass scheduler
runs a topological sort of compatible functors. Adding a new pass
is "declare your view"  -  no DAG editing.

### S4. Provenance + cache invalidation as Datalog query

**Status**: primitive shipped (`scallop_join`); self-consumer shipped
(`scallop_provenance`); causal change-impact shipped
(`do_calculus_change_impact`).

**Required source change**: vyre's `pipeline_disk_cache` invalidation logic
(today: hand-coded "if X changed, invalidate Y, Z") becomes a
single `do_calculus` query against the rule-dependency graph,
running through `scallop_join` for transitive lineage closure.
Cache hit rate measurably improves because invalidations become
provably-minimal (today they're conservative: invalidate-too-much).

**Deliverable**: `vyre-driver-wgpu/src/runtime/cache/invalidation.rs`
calling into the self-consumers. Replaces the hand-managed
invalidation rules.

### S5. Coarse-grained workspace analysis at 1M+ Regions

**Status**: primitives shipped (`mori_zwanzig`, `fmm`, `sinkhorn`);
self-consumers shipped (`mori_zwanzig_region_coarsen`,
`fmm_polyhedral_compress`, `sinkhorn_dispatch_clustering`).

**Required source change**: when scanning 1M+ Region workspaces, route through
the coarsening pipeline:
sinkhorn clusters → MZ projects → FMM compresses all-pairs.
Each layer drops a complexity class. The naive O(N²) polyhedral
fusion becomes O(K²) on K macro-nodes, then O(K log K) via FMM.

**Deliverable**: `vyre-foundation/src/optimizer/scale.rs`  -  a
"scale-aware" mode that automatically engages the coarsening
pipeline when N > threshold.

### S6. Topological loop-fusion / fission

**Status**: primitive shipped (`vietoris_rips`); self-consumer shipped
(`persistent_homology_loop_signature`).

**Required source change**: extend persistent_homology_loop_signature with full
H₁ persistence (cycle counting per ε scale). Wire into the
loop-fusion pass: passes that change the H₁ persistence signature
are flagged as "topology-altering" and require explicit opt-in
from the rule author. Today loop-fusion is purely cost-driven;
topology-driven fusion is the bigger lever.

**Deliverable**: `vyre-foundation/src/optimizer/loop_topology.rs`.

### S7. Pass-precondition compilation

**Status**: primitive shipped (`knowledge_compile`); self-consumer
shipped (`knowledge_compile_pass_precondition`).

**Required source change**: every pass in `vyre-foundation/src/optimizer/passes/`
declares its precondition as a propositional formula via a
`#[vyre_pass(precondition = "...")]` proc macro. Compile-time
d-DNNF compilation; runtime per-Program ddnnf_evaluate. Hand-rolled
match-on-Node validators are deleted.

**Deliverable**: `vyre-macros/src/pass_precondition.rs` proc macro +
the d-DNNF evaluation hook in the pass framework.

### S8. Natural-gradient autotuning

**Status**: primitive shipped (`natural_gradient`); self-consumer
shipped (`natural_gradient_autotuner`); fixed-point release path now
composes `differentiable_autotune` policy gradients with
Fisher-preconditioned natural-gradient updates in
`vyre-self-substrate::math::differentiable_autotune`; driver-tier
`vyre_driver::tuner::NaturalGradientPolicy` converts live latency
measurements into Fisher-preconditioned next-probe workgroup choices;
unset `VYRE_AUTOTUNER` defaults to `Mode::NaturalGradient`,
`VYRE_AUTOTUNER=natural` is accepted by the driver mode resolver, and
`Tuner::best_of_natural_gradient` runs the measured backend-timer
sweep before applying the Fisher policy. CUDA's release launch path now
consumes the same mode through `vyre_driver::launch::LaunchPlan`, and
WGPU derives the same effective config before cache-keying and WGSL
lowering so `@workgroup_size` matches dispatch metadata. When no caller
override or explicit grid shape is present, eligible 1D storage-only
kernels get a natural-gradient cold-start workgroup before grid
inference; the selected shape is cached per program fingerprint,
declared workgroup, element count, and backend launch limit tuple so
policy-vector allocation is not repeated on the hot path. CUDA timed
dispatches now feed real `device_ns` measurements back into the same
bounded launch cache, so later automatic launches can move away from
the cold-start heuristic when hardware timings prove a different
candidate faster. WGPU timed dispatch now promotes timestamp-query
deltas into structured `TimedDispatchResult::device_ns` and records
those measurements through the same resolver instead of leaving them as
trace-only telemetry. Workgroup-local scratch, non-1D kernels, explicit
overrides, and explicit grid dispatches remain fixed to preserve kernel
semantics. Measured launch decisions now persist across process restarts
through the existing bounded tuner TOML store, keyed by program
fingerprint, declared workgroup, element count, backend launch limits,
and backend family.

**Deliverable**: `vyre-driver/src/launch.rs` +
`vyre-driver/src/tuner.rs` + `vyre-driver-wgpu/src/pipeline.rs`.

### S9. IR rewrites in batches via planar scheduling

**Status**: source change shipped. The `planar_rewrite` primitive,
`planar_rewrite_pass_scheduler` self-consumer, foundation
`RewriteBatchCandidates` planner, and scheduler-driven
`ProgramPass::batch_apply()` contract now share one non-overlap model.

**Required source change**: shipped at the optimizer orchestration boundary.
Passes that expose candidate geometry opt into planar batching; when match
count exceeds the threshold, the scheduler calls `batch_apply()` and the pass
receives disjoint rewrite waves instead of relying on silent fusion.

**Deliverable**: `ProgramPass::batch_apply()` and
`ProgramPass::try_batch_apply()` are live scheduler entry points, with tests
that fail if the scheduler falls back to `transform()` for a batched pass.

### S10. Causal back-door pass-ordering

**Status**: source change shipped. The `adjustment_set` primitive,
`adjustment_set_pass_dependency` self-consumer, and foundation
`derived_order` artifact now share the same pass-dependency model.

**Required source change**: shipped through
`vyre-foundation/src/optimizer/derived_order.rs`. The artifact derives
topological order from live `#[vyre_pass]` inventory metadata and materializes
causal invalidation edges for back-door safety checks.

**Deliverable**: `derive_registered_pass_order()` is the load-bearing
release-validation path; `validate_registered_pass_order()` consumes the
derived artifact rather than rebuilding an independent order.

---

## What ships in 1.0  -  the recursion completes

By 1.0 every non-workload-only primitive in `MATH_FRONTIER.md` has
both a user-dialect consumer AND >= 1 vyre-self consumer wired into a
production code path. Workload-only primitives must be explicit in the
recursion allowlist with a local architectural justification; private
helper modules do not count as public primitives. The list of unwired
public primitives shrinks to zero, AND the `xtask::recursion_gate` CI
enforcement is load-bearing through `scripts/check_recursion_gate.sh`
inside release signoff. The gate resolves the real Vyre workspace,
scans primitive and self-substrate modules recursively, parses grouped
Rust imports across newlines, and fails release validation when the
self-consumer evidence disappears.

Three remaining 1.0 paradigm shifts:

### V1. Effects-handler lowering for IR transforms

`#12 effects-handler RFC` reframes lowering passes as algebraic
effect handlers. Each pass is an effect-typed function;
composition is handler nesting. Replaces the imperative pass
loop with an effect-typed pipeline that the type system checks.

### V2. Linear-logic typed BufferAccess

`#18 linear-logic RFC`: BufferAccess gets linear types. The type
system rejects double-write, double-free, and use-after-free at
compile time without runtime checks. Substrate-level memory
safety.

### V3. Liquid types on BufferDecl

`#60 liquid types RFC`: BufferDecl carries SMT-checkable shape
predicates ("buffer is sorted", "indices in [0, n)", "no aliasing
with X"). The optimizer can rely on these without re-deriving them
per pass.

By 1.0, vyre is **self-improving**: every primitive shipped expands
both the user-workload surface AND lets vyre compile itself better.
The substrate IS the identity  -  not a mass of orthogonal features
glued together.

---

## What this means for every commit going forward

1. Every Tier-2.5 primitive ships AS a user op AND AS a
   substrate self-consumer in the same commit. The recursion-gate
   xtask enforces.

2. Every optimizer / scheduler / cache / lowering decision asks
   "what frontier-math primitive expresses this exactly?" If the
   answer is "we'd need to invent a new one"  -  invent it (and ship
   the dual-use entry). If the answer is "primitive #N already
   exists but isn't wired"  -  wire it.

3. Hand-rolled heuristics are technical debt. Each one represents
   a place where vyre is using less than the math it ships. The
   debt is paid down by replacing the heuristic with the canonical
   primitive.

4. Performance is measured against the release bar, not the CPU
   bar. 1000× CPU is table stakes for any GPU substrate. The bar
   is "what becomes expressible at all that no other substrate
   can express." Every paradigm-shift batch must move that bar.

The compounding effect: by 1.0 vyre ships a substrate where each
new primitive simultaneously serves users AND makes vyre itself
better. The market position is not "another GPU IR"  -  it's "the
substrate where frontier math is the dispatch model."

## Per-commit status (P-DOC-3, updated each merge)

Tracks which paradigm-shift batches have landed; update this section
in the commit that implements each gate.

### Substrate-universal lift (shipped)

- `70b2c4b64e` vyre-driver/runtime/aot/wgpu/cuda/megakernel depend
  on vyre-primitives directly.
- `fd97448ea4` `self_substrate` lifted from vyre-libs to vyre-driver
  with re-export shim for backward compat.

### S-batch wires shipped (substrate consumed in production)

| Wire | Commit | Substrate consumer |
|---|---|---|
| P-DRIVER-1 | `717e7bd9dc` | matroid_megakernel_scheduler |
| P-DRIVER-2 | `a5ea175129` | multigrid_matroid_solver |
| P-DRIVER-3 | `36c683d48d` | megakernel_schedule (homotopy) |
| P-DRIVER-4 | `a5ea175129` + `44349b0dc3` | sheaf_heterophilic_dispatch + spectral |
| P-DRIVER-6 / P-CUDA-1 | `277fa6d24e` | submodular_cache_eviction |
| P-RUNTIME-1 | `127641917a` | scallop_provenance lineage |
| P-FOUND-1 | `fbbab5270b` | functorial_pass_composition |
| P-FOUND-2 | `d8e9a1f649` | string_diagram_ir_rewrite |
| P-FOUND-5 | `3ab2d34eae` | adjustment_set_pass_dependency |
| P-FOUND-6 | `4feca243f9` | tensor_network_fusion_order |
| P-FOUND-7 | `c07c0e2573` | dataflow_fixpoint |
| P-FOUND-9/16 | `0f66f2ba5a` | polyhedral_fusion |
| P-FOUND-x | `179c43e0d2` | megakernel_schedule (foundation) |
| P-FOUND-x | `be63f08f63` | matroid + multigrid (foundation) |

### Self-consumers for new primitives

| Primitive | Self-consumer | Commit |
|---|---|---|
| #3 amg_v_cycle | amg_pass_solver | `b5030091f2` |
| #9 sheaf_laplacian_eigenvalue | sheaf_spectral_clustering | `d89a136c3f` |
| #10 matroid_intersection_full | matroid_exact_megakernel | `17e36bbd2e` |
| #12 tensor_train_decompose | tensor_train_compression | `f0fdc41c1f` |

### Observability surface (P-OBS-1/2/3)

- `fa6fa279ac` driver observability + Prometheus exporter
- `0ac11b1bb7` decision-telemetry histograms
- `6fe3137648` substrate-call counter

### Tooling

- `3eb82afc25` `cargo xtask measurement-gate` (P-XTASK-3)

### V1.0 paradigm shifts requiring source work

P-1.0-V1.* (effects-handler lowering) is now load-bearing on the
backend pre-lowering path: `PassScheduler` can enforce effect-handler
postconditions, `pre_lowering::optimize` enables the gate, and pass
metrics expose before/after effect-row bits. P-1.0-V2.* (linear
BufferAccess) is also load-bearing on the same path: rewrites that
introduce new `BufferDecl::linear_type` violations are reverted before
backend lowering, and pass metrics expose before/after violation counts.
P-1.0-V3.* (liquid BufferDecl shapes) is now load-bearing on the same
path: rewrites that introduce new `BufferDecl::shape_predicate`
violations are reverted before backend lowering, repairing rewrites are
allowed, and pass metrics expose before/after shape-violation counts.
Those facts are also consumed by `loop_var_range_fold`: loop-induction
guards against `buf_len(buffer)` are erased when liquid min/max shape
facts prove the branch true or false, turning shape predicates into
less dynamic branch work on the CUDA release path.
