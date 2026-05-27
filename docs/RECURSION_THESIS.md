# Recursion thesis  -  math we ship as ops becomes math we use to compile ops

This is the architectural thesis of vyre's substrate strategy: every
Tier-2.5 primitive shipped in `vyre-primitives` MUST also have a
**vyre-self consumer**  -  a substrate module in the current substrate
crate boundary that calls the same Program against
vyre's own IR / dispatch graph / cost model.

The recursion compounds the moat. Every new primitive simultaneously
expands the user-workload surface AND lets vyre self-improve. Three
incidents made this concrete:

1. We shipped `semiring_gemm` (#1) for users (security reachability,
   CKY parsing, Viterbi). Same Program now powers vyre's own
   reachability closure on the Region tree (#26 dataflow fixpoint).
2. We shipped `chebyshev_filter` (#5) for users (spectral GNN). Same
   Program now powers vyre's own spectral clustering of dispatch
   graph (#23 spectral schedule).
3. We shipped `differentiable_argmax` (#7) for users (attention,
   structured prediction). Same Program now powers vyre's own
   differentiable autotuner picking workgroup/tile/fusion configs (#27).

## The bar

Three conditions for promotion to Tier-2.5 (`vyre-primitives/src/<domain>/`):

1. **Reusability**  -  ≥ 2 user-dialect (Tier-3) consumers.
2. **Stability**  -  small, named API; no caller asking for breaking changes.
3. **No domain glue**  -  single concern; LAW 7.

PLUS the recursion bar:

4. **≥ 1 vyre-self consumer**  -  a module in the current substrate
   boundary that dispatches the same Program
   against vyre's own IR / dispatch graph / cost model.

If 4 fails, the primitive stays in `vyre-libs` as a Tier-3 dialect
until a self-use materializes.

## Substrate access  -  who can call what

The recursion thesis is realized at three layers:

| Crate | Substrate path | Why |
|---|---|---|
| `vyre-foundation` | `vyre_foundation::pass_substrate::*` | Foundation cannot depend on `vyre-primitives` (the cycle: primitives → foundation IR types). Foundation inlines the math it needs as `cpu_references.rs` and the `pass_substrate` modules consume it. |
| `vyre-driver`, `vyre-driver-{wgpu,cuda,spirv,megakernel}`, `vyre-runtime`, `vyre-aot`, `vyre-frontend-c` | `vyre_primitives::*` directly | These crates depend on `vyre-primitives` (added in this commit for the driver/runtime/aot tier). They call CPU references like `vyre_primitives::math::semiring_gemm_cpu` directly without needing the `Program`-returning wrappers. |
| `vyre-libs::self_substrate` | itself + `vyre_primitives::*` | The orchestration tier  -  wraps primitives in `Program`-returning self-consumers for callers that operate on vyre IR. |
| User dialects | via `vyre-libs::self_substrate` | Tier-3 consumption. |

The crucial point: **substrate is universal below `vyre-libs`**  -  every
backend / runtime / AOT crate has direct access to the math via
`vyre-primitives` without needing the `vyre-libs` orchestration layer.
Above `vyre-libs`, user dialects get the orchestration via the
`self_substrate` wrappers.

`vyre-foundation` is the one exception (cannot depend on primitives
without breaking the foundation/primitives cycle), and it solves the
problem by inlining its own copies of the small math kernels it needs
in `cpu_references.rs`.

## Self-consumer table (current)

**External wiring tally (`vyre_driver::self_substrate`):** 7 consumed / 23 unconsumed among the 30 `pub mod` entries in `vyre-driver/src/self_substrate/mod.rs` (commit `fd97448ea4` referenced 27 modules moved; the tree grew afterward).

**Methodology.** A module is **consumed** iff this command prints at least one line:

```bash
grep -rn 'vyre_driver::self_substrate::<module_name>' libs/performance/matching/vyre/ \
  --include='*.rs' | grep -v '/self_substrate/' | grep -v 'test'
```

Substitute `<module_name>` with the module identifier (same string as `pub mod` in `vyre-driver/src/self_substrate/mod.rs`). Zero lines means **unconsumed** for this tally even if `README` text or `vyre_foundation::pass_substrate::*` calls an analogue.

**Module status**

| Module | Consumed |
|---|---|
| `adjustment_set_pass_dependency` | unconsumed |
| `bellman_tn_order` | unconsumed |
| `cost_model` | unconsumed |
| `dataflow_fixpoint` | unconsumed |
| `differentiable_autotune` | unconsumed |
| `do_calculus_change_impact` | consumed |
| `fmm_polyhedral_compress` | unconsumed |
| `functorial_pass_composition` | unconsumed |
| `kfac_autotune_step` | unconsumed |
| `knowledge_compile_pass_precondition` | unconsumed |
| `matroid_megakernel_scheduler` | consumed |
| `megakernel_schedule` | consumed |
| `mori_zwanzig_region_coarsen` | unconsumed |
| `multigrid_matroid_solver` | unconsumed |
| `natural_gradient_autotuner` | unconsumed |
| `persistent_homology_loop_signature` | unconsumed |
| `planar_rewrite_pass_scheduler` | unconsumed |
| `polyhedral_fusion` | unconsumed |
| `qsvt_matrix_function_fusion` | unconsumed |
| `scallop_provenance` | consumed |
| `scallop_provenance_wide` | unconsumed |
| `sheaf_heterophilic_dispatch` | consumed |
| `sinkhorn_dispatch_clustering` | unconsumed |
| `sinkhorn_full_clustering` | unconsumed |
| `spectral_schedule` | unconsumed |
| `string_diagram_ir_rewrite` | unconsumed |
| `submodular_cache_eviction` | consumed |
| `tensor_network_fusion_order` | unconsumed |
| `tensor_train_chain_fusion` | unconsumed |
| `vsa_fingerprint` | consumed |

**Foundation exception.** `vyre-foundation` imports `crate::pass_substrate::*` (e.g. `scheduler.rs` calling `pass_descendants`, `reachability_closure`, `passes_commute_on`). Those hits do not use `vyre_driver::self_substrate::…`, so they are out of scope for the tally above.

### Primitive to self-consumer mapping (unchanged semantics)

| Primitive | Self consumer module |
|---|---|
| #1 `semiring_gemm` | `self_substrate::dataflow_fixpoint` (#26)  -  region-graph closure under (BoolOr, MinPlus, Lineage) |
| #2 `sinkhorn` | `self_substrate::sinkhorn_dispatch_clustering` (#30)  -  dispatch-graph clustering via entropic transport |
| #5 `chebyshev_filter` | `self_substrate::spectral_schedule` (#23)  -  dispatch-graph low-pass filter |

| #6 `tensor_train_contract` | `self_substrate::tensor_train_chain_fusion` (#6)  -  chain-shaped Region fusion via TT contraction |
| #7 `differentiable_argmax` | `self_substrate::differentiable_autotune` (#27)  -  soft-pick best dispatch config |
| #9 `homotopy_euler_predictor` | `self_substrate::megakernel_schedule` (#22)  -  ILP-relaxation Euler tracker |
| #10 `sum_product_circuit` | `self_substrate::cost_model` (#28)  -  probabilistic dispatch cost + #41 conformal intervals |
| #13 `hypervector_xor_bind` | `self_substrate::vsa_fingerprint` (#29)  -  content-addressable Program cache key |
| #17 `mp_edge_clip` | `self_substrate::spectral_schedule` (#23)  -  spectrum outlier filter |
| #36 `do_calculus` | `self_substrate::do_calculus_change_impact` (#36)  -  rule-graph change-impact analysis |
| #41 `conformal_threshold` | `self_substrate::cost_model` (#28)  -  calibrated cost intervals |
| #11 `planar_rewrite_schedule` | `self_substrate::planar_rewrite_pass_scheduler` (#11)  -  IR rewrite-batch scheduler over disjoint sub-trees |
| #15 `vietoris_rips` | `self_substrate::persistent_homology_loop_signature` (#15)  -  Region-tree loop topology via H₁ persistence |
| #31 `sheaf_diffusion_step` | `self_substrate::sheaf_heterophilic_dispatch` (#31)  -  heterophilic dispatch-graph analysis |
| #34 `qsvt_apply` | `self_substrate::qsvt_matrix_function_fusion` (#34)  -  transport-based fusion analysis (Wasserstein over dispatch graphs) |
| #37 `backdoor_descendants_check` | `self_substrate::adjustment_set_pass_dependency` (#37)  -  optimizer pass-ordering validity via causal back-door |
| #38 `ddnnf_evaluate` | `self_substrate::knowledge_compile_pass_precondition` (#38)  -  pass preconditions as compiled d-DNNF circuits |
| #39 `scallop_join` | `self_substrate::scallop_provenance` (#39)  -  GPU-resident rule provenance closure |
| #45 `argmax_of_marginals` | `self_substrate::submodular_cache_eviction` (#45)  -  pipeline cache eviction via submodular maximization |
| #46 `matroid_exchange_bfs_step` | `self_substrate::matroid_megakernel_scheduler` (#46)  -  discrete fusion-grouping via matroid intersection |
| #50 `jacobi_smooth_step` | `self_substrate::multigrid_matroid_solver` (#50)  -  sparse linear-system solver for matroid CLS-2021 inner loop |
| #51 `fmm_zeroth_*` | `self_substrate::fmm_polyhedral_compress` (#51)  -  hierarchical O(N log N) compression of polyhedral fusion's all-pairs |
| #52 `functor_apply` | `self_substrate::functorial_pass_composition` (#52)  -  IR transform passes as categorical functors |
| #53 `monoidal_compose` | `self_substrate::string_diagram_ir_rewrite` (#53)  -  Region tree as string diagram in Cat(GPU buffers, Programs) |
| #56 `natural_gradient_block_apply` | `self_substrate::natural_gradient_autotuner` (#56)  -  Fisher-information-preconditioned autotuner step |
| #58 `mz_project_step` | `self_substrate::mori_zwanzig_region_coarsen` (#58)  -  Region-tree coarse-graining (O(N²) → O(K²)) |

24 primitives currently have wired self-consumers (primitive-level claim; wiring tally above is path-specific).

## Self-consumer gaps

These primitives still need self-consumers:

| Primitive | Required self consumer |
|---|---|
| #4 `ntt_radix2` | attested-compute proof emission (vyre's own dispatch logs) |
| #8 `clifford_product` | n/a (geometric workloads, no vyre-self use) |
| #14 `sos_certificate` | buffer-safety SOS proofs in `xtask` |
| #16 `newton_schulz` | preconditioned dispatch-cost optimization |
| #35 `tensor_network` | Region-tree contraction order = optimal fusion order |

5 primitives still lack a verified self-consumer. They are not fully
cleared by the recursion bar until source code wires one.

## Enforcement

`xtask::recursion_gate` must walk every registered op id in
`vyre-primitives` and verify at least one current substrate consumer.
Builds must fail if any primitive has zero self-consumers.

## Substrate extraction source work

The dedicated `vyre-substrate` crate is not a doc-only migration.
Extraction requires moving the substrate modules, adding the new
dependency edge for megakernel/foundation callers, and landing
`xtask recursion_gate` in the same source patch.

## Why this matters

Three reasons the recursion thesis is the moat:

1. **Compounding leverage.** Every primitive does double duty. Ship
   N primitives, get 2N use cases.
2. **Self-improvement.** vyre-driver and vyre-foundation get faster
   every time a new primitive is introduced, regardless of which user
   workload it was nominally shipped for.
3. **Trust signal.** Anyone can see that vyre uses its own primitives
   to compile programs. The substrate is the proof that the primitives
   are correct, fast, and composable.

Math we ship as ops becomes math we use to compile ops. The moat
compounds because every new primitive simultaneously helps user
workloads AND lets vyre self-improve.
