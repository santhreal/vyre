# Math primitives  -  placement map

Authoritative map of where every math-based primitive belongs in the
vyre tree. Cross-links `lego-block-rule.md`, `primitives-tier.md`, and
`MATH_FRONTIER.md`. Every new file should consult this doc before
landing.

## The bar

Three conditions for a Tier-2.5 primitive (`vyre-primitives/src/<domain>/`):

1. **Reusability**  -  ≥ 2 user-dialect (Tier-3) consumers.
2. **Stability**  -  small, named API; no caller asking for breaking changes.
3. **No domain glue**  -  single concern; LAW 7.

PLUS the **dual-use bar** added by the recursion thesis (#30):

4. **≥ 1 vyre-self consumer**  -  `vyre-foundation::transform`,
   `vyre-driver-*`, or `xtask` will dispatch the same Program against
   vyre's own IR / dispatch graph / cost model.

If 4 fails, the primitive stays in `vyre-libs` as a Tier-3 dialect
until a self-use materializes. Math-we-ship-as-ops becomes
math-we-use-to-compile-ops; that recursion is the moat.

## Placement table  -  primitives shipped

| Task | Primitive | File | Feature | User dialect (≥ 2) | vyre-self consumer |
|---|---|---|---|---|---|
| #1 | `semiring_gemm` | `vyre-primitives/src/math/semiring_gemm.rs` | `math` | dataflow / parsing CKY / security / Viterbi / GF(2) / Scallop | #19 fusion analysis · #22 megakernel critical path · #23 spectral schedule · #26 dataflow fixpoint · #36 do-calculus · #39 Scallop-join |
| #2 | `sinkhorn_scale` | `math/sinkhorn.rs` | `math` | OT / Wasserstein / typedef-classification | source-change required for dispatch-graph clustering via Sinkhorn-OT distance |
| #4 | `ntt_*` | `hash/ntt.rs` | `hash` | FHE (BFV/CKKS) · zk-SNARKs (PLONK) · lattice (Kyber) · stable polynomial multiply | no shipped vyre-self consumer |
| #5 | `chebyshev_filter` | `graph/chebyshev_filter.rs` | `graph` | spectral GNN · security spectral anomaly · taint propagation | #23 spectral analysis of dispatch graph |
| #7 | `softmax_step` + `differentiable_argmax` | `math/differentiable.rs` | `math` | attention · structured prediction · typedef-classification gradient flow | #27 differentiable autotuner over workgroup/tile/fusion grid |
| #13 | `hypervector_xor_bind` + `majority_bundle` | `hash/hypervector.rs` | `hash` | retrieval · symbolic reasoning · in-memory database | #29 VSA op-cache key (content-addressable Program fingerprint) |
| #39 | (Scallop-join via `Semiring::Lineage`) | `math/semiring_gemm.rs` | `math` | neuro-symbolic Datalog | rule provenance tracking in a rule compiler |
| #40 | `gaussian_rdp_step` | `math/dp_accountant.rs` | `math` | DP-SGD trainers · privacy-preserving observability | profiler telemetry hardening |
| #41 | `conformal_threshold` | `math/conformal.rs` | `math` | calibrated NN intervals · regression detection | #28 dispatch cost-model intervals |
| #43 | `rk4_step` | `math/ode_step.rs` | `math` | neural ODE · physics flow | #9 homotopy_continuation path-tracking |
| #45 | `argmax_of_marginals` | `math/submodular_greedy.rs` | `math` | active learning · coreset · sensor placement | compile-cache eviction as submodular coverage |
| #47 | `count_sketch_update` | `hash/sketch.rs` | `hash` | streaming statistics · approximate quantiles | profiler latency-distribution sketch |
| #55 | `grunwald_letnikov_kernel` | `math/fractional.rs` | `math` | anomalous diffusion · viscoelastic · FractalNet variants | (none  -  kernel data, no GPU dispatch) |

## Placement table  -  source-change candidates (round 1)

| Task | Primitive | File | Feature | Notes |
|---|---|---|---|---|
| #3 | `randomized_svd` | `math/randomized_svd.rs` | `math` | Composes Householder QR (also new). Unblocks #34 QSVT. |
| #6 | `tensor_train_contract` | `math/tensor_train.rs` | `math` | Composes `semiring_gemm`. |
| #8 | `clifford_product` | `geom/clifford.rs` | NEW: `geom` | Multivector geometric product. Unblocks #33 TFN. |
| #9 | `path_tracker_step` | `opt/path_tracker.rs` | NEW: `opt` | Composes `rk4_step`. Splits homotopy: step is primitive, scheduler is substrate. |
| #10 | `sum_product_circuit` | `graph/sum_product_circuit.rs` | `graph` | Composes `level_wave_program` + `reduce::sum`. Unblocks #28 cost model. |
| #11 | `planar_rewrite_step` | `parsing/planar.rs` | `parsing` | 2D grammar productions. Research first. |
| #14 | `sos_admm_step` | `opt/sos.rs` | `opt` | Composes `semiring_gemm` + `householder_qr`. |
| #15 | `vietoris_rips_filtration` + `chunk_reduce` | NEW dir + feature: `topology` | NEW: `topology` | Persistent homology at scale. |
| #16 | `kfac_block` + `shampoo_root` | `math/preconditioner.rs` | `math` | Composes `semiring_gemm` + `householder_qr` + `inv_root_psd`. |
| #17 | `spectral_shape` | `math/spectral_shape.rs` | `math` | Composes #5 chebyshev. Self-applied for #23 dispatch clustering. |

## Placement table  -  substrate (NOT primitives)

The recursion thesis insight: some entries are *substrate*, not ops.
These improve every workload, not one.

| Task | Where | What |
|---|---|---|
| #19 | `vyre-foundation/src/transform/polyhedral.rs` | Polyhedral fusion across Region boundaries. Uses #1 + #5 + #17 on the fusion adjacency. |
| #20 | `vyre-foundation/src/transform/lower_handlers.rs` | Algebraic-effect handler-shape recognition + lowering. Requires a source patch that introduces the IR construct and its migration gate together. |
| #21 | `vyre-foundation/src/ir/buffer_decl.rs` enrichment + `xtask/src/linearity_check.rs` | Linear-logic typed BufferAccess (`Owned` / `Shared` / `Unique` / `Aliased`). Default `Aliased` preserves backward-compat. |
| #22 | `vyre-runtime/src/megakernel/planner.rs` | ILP-relaxed scheduler. Uses #9 homotopy + #45 submodular + #46 matroid intersection. |
| #23 | `vyre-foundation/src/transform/spectral_schedule.rs` | Spectral clustering of dispatch graph. Uses #5 chebyshev + #17 spectral_shape. |
| #26 | `vyre-foundation/src/transform/dataflow_fixpoint.rs` | Region-graph dataflow fixpoint via #1. Replaces hand-rolled IR analyses with semiring-matmul iterations. |
| #27 | `vyre-driver/src/autotuner/differentiable.rs` | Differentiable autotuner via #7. Smoothed argmax over the tuning grid; gradient via cost-model autodiff. |
| #28 | `vyre-driver/src/cost_model/probabilistic.rs` | Probabilistic-circuit dispatch cost model via #10. Conformal intervals from #41 feed #22. |
| #29 | `vyre-foundation/src/ir/fingerprint/vsa.rs` | VSA-based op cache key via #13. Approximate-match cache for compiled Programs. |

## New feature gates required

`vyre-primitives/Cargo.toml` needs three additions for round-1 expansion:

```toml
geom     = ["vyre-foundation"]                 # Clifford / TFN equivariant ops
opt      = ["vyre-foundation", "math"]          # SOS, homotopy, matroid, submodular
topology = ["vyre-foundation", "math", "graph"] # Persistent homology, simplicial complexes
```

The math/parsing/hash/graph/nn/text/matching/decode/nfa/bitset/reduce
gates already exist.

## Decision tree for placing a new primitive

1. **Is it a single op with ≥ 2 user-dialect consumers and ≥ 1
   self-consumer?** → Tier-2.5 primitive in `vyre-primitives/src/<domain>/`.
2. **Does it improve every Program (fusion, scheduling, typing, cache
   keys)?** → Substrate. Goes in `vyre-foundation/src/transform/`,
   `vyre-driver-*/`, or `xtask/`.
3. **Is it a single-caller convenience over existing primitives?** →
   Tier-3 in `vyre-libs/src/<dialect>/`. Lift to Tier-2.5 only when
   the second caller materializes.
4. **Is it a kernel data table (rules, weights, signatures)?** → Tier B
   in the host-side `rules/` dir per `AGENTS.md`. NOT a Rust file.

## Why this doc exists

A placement mistake is a future deletion or rewrite. Every primitive
that lives in the wrong tier breaks the lego rule and forces churn.
Consult this doc before opening a new file in `vyre-primitives` or
`vyre-libs`. When you ship a primitive, add a row above. When you
discover the placement was wrong, move + update the row, and append
a note in the trailing changelog.

## Trailing changelog

- 2026-04-27 (this session): Initial map. 13 primitives shipped under
  the dual-use bar (`semiring_gemm`, `chebyshev_filter`, `hypervector`,
  `ode_step`, `dp_accountant`, `fractional`, `submodular_greedy`,
  `sketch`, `conformal`, `ntt`, `sinkhorn`, `differentiable`).
  Frontier candidates #31-#60 catalogued in `MATH_FRONTIER.md`.
