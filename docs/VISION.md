# vyre  -  vision

vyre is **LLVM-for-GPU**: a substrate-neutral compiler whose IR is the
unit of work, the unit of caching, and the unit of optimization. The
moat is composability  -  perfect math primitives compose into perfect
dispatches that compose into perfect programs.

## Three theses, one moat

### 1. Compose perfect primitives, beat monolithic kernels

Every Tier-2.5 primitive in `vyre-primitives/` is a single concern,
hardened, tested across CPU + GPU paths. Domain compositions in
`vyre-libs/` glue primitives together; the optimizer can fuse
across composition boundaries because the underlying primitives are
visible to it. Domain code never reinvents a primitive  -  see
[`docs/lego-block-rule.md`](lego-block-rule.md).

### 2. Recursion thesis  -  math we ship as ops becomes math we use to compile ops

Every primitive shipped to user dialects (security reachability, ML,
parsing) MUST also have a vyre-self consumer that uses the same
Program against vyre's own IR / dispatch graph / cost model. See
[`docs/RECURSION_THESIS.md`](RECURSION_THESIS.md).

The recursion compounds the moat: every new primitive simultaneously
expands the user-workload surface AND lets vyre self-improve.

### 3. Substrate is universal below `vyre-libs`

After commit `fd97448ea4`, `self_substrate` lives in
`vyre-driver/`, so every backend (wgpu, cuda, spirv, megakernel) and
the runtime/aot crates inherit substrate access via a single
dep. `vyre-foundation` keeps a smaller `pass_substrate` for the
optimizer's own use (the foundation/primitives cycle would otherwise
block direct primitive consumption).

## Recursion trajectory  -  current state

**Substrate consumers actively wired in production code paths
(grep-verified):**

- `vyre_driver::self_substrate::matroid_megakernel_scheduler`  - 
  drives `select_fused_subset` in `vyre-runtime megakernel` (P-DRIVER-1).
- `vyre_driver::self_substrate::megakernel_schedule`  -  homotopy
  continuation chains with the matroid scheduler (P-DRIVER-3).
- `vyre_driver::self_substrate::multigrid_matroid_solver`  -  Jacobi
  flow weights inside the matroid augmenting loop (P-DRIVER-2).
- `vyre_driver::self_substrate::sheaf_heterophilic_dispatch`  - 
  divergence flagging in the runtime exchange-adjacency (P-DRIVER-4).
- `vyre_driver::self_substrate::sheaf_spectral_clustering`  - 
  spectral-gap signal in megakernel fusion grouping.
- `vyre_driver::self_substrate::submodular_cache_eviction`  -  pipeline
  cache eviction in vyre-driver-wgpu and vyre-driver-cuda (P-DRIVER-6,
  P-CUDA-1).
- `vyre_driver::self_substrate::do_calculus_change_impact`  - 
  invalidation cascade in pipeline_disk_cache.
- `vyre_driver::self_substrate::scallop_provenance`  -  output lineage
  in `MegakernelReport.region_lineage` (P-RUNTIME-1).
- `vyre_driver::self_substrate::vsa_fingerprint`  -  approximate
  validation cache in vyre-driver-wgpu.
- `vyre_driver::self_substrate::matroid_exact_megakernel`  -  exact
  Edmonds solver behind `select_optimal_fused_subset`.

**Foundation pass_substrate (vyre-foundation has its own smaller
substrate, all 9 modules now consumed by `PassScheduler`):**

- `adjustment_set_pass_dependency` → `PassScheduler::transitive_dependents` (P-FOUND-5)
- `dataflow_fixpoint` → `PassScheduler::reaches`, `invalidation_closure` (P-FOUND-7)
- `functorial_pass_composition` → `PassScheduler::pair_commutes` (P-FOUND-1)
- `string_diagram_ir_rewrite` → `PassScheduler::triple_associates` (P-FOUND-2)
- `tensor_network_fusion_order` → `PassScheduler::fusion_friendly_order`,
  `ordering_cost` (P-FOUND-6)
- `polyhedral_fusion` → `PassScheduler::fusable_pass_pairs` (P-FOUND-9/16)
- `megakernel_schedule` → `PassScheduler::fusion_pressure`
- `matroid_megakernel_scheduler` → `PassScheduler::fusable_subset`
- `multigrid_matroid_solver` → `PassScheduler::smooth_pass_system`

## Headline metric: substrate-call counter

Every substrate consumer increments
`vyre_driver::self_substrate::observability::*_calls` on every
production call. Operators read snapshots via
`DriverObservability::snapshot()` for Prometheus / OpenTelemetry
dashboards. When a substrate path stops getting traffic, the
counter stalls  -  visible in dashboards, fixable in code.

The substrate-decision telemetry
(`vyre_driver::self_substrate::decision_telemetry`) attributes each
decision to the math that made it: which fusion subset size got
picked, how aggressively eviction trimmed, how much provenance
closure found.

## Layering

```
vyre-primitives                  Tier-2.5 math (pure CPU + Program builders)
  └── vyre-libs                  Tier-3 dialects (security, ML, parsing)
       └── vyre-foundation       IR + optimizer (uses pass_substrate)
            └── vyre-driver      Backend trait + self_substrate (lifted from vyre-libs)
                 └── vyre-driver-{wgpu,cuda,spirv,megakernel}
                 └── vyre-runtime, vyre-aot, vyre-frontend-c
```

`vyre-foundation` cannot directly consume `vyre-primitives` (the
optional reverse dep makes a cycle). Foundation maintains its own
`cpu_references.rs` with the small kernels it needs and a parallel
`pass_substrate/` module  -  same Linux-style "arch-local libs" pattern
documented in `cpu_references.rs` (no smell, intentional).

## Release trajectory

The next major release ships when:

1. ≥ 80% of self_substrate modules are actively consumed (grep
   verifiable, not just compiled).
2. Every backend (wgpu, cuda, spirv) exposes the same observability
   surface via `BackendObservabilityProvider`.
3. The `MegakernelReport` includes `region_lineage` populated by
   scallop_provenance (shipped  -  P-RUNTIME-1).
4. Foundation `optimize()` routes through `PassScheduler` whose
   ordering decisions are derivable from substrate, not hand-curated.

Beyond that, V3 paradigm shifts (effects-handler lowering, linear
types, liquid types) move vyre from "good compiler" to "categorical
substrate compiler"  -  see `docs/PARADIGM_SHIFT_TRAJECTORY.md`.
