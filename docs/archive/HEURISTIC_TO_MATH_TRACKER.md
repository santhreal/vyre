# Heuristic to math substrate tracker

Rows are derived from `git log --oneline | head -30` at sweep time, filtered to changes that replace hand-tuned behavior with calls into `vyre-foundation::pass_substrate::*` or `vyre_driver::self_substrate::*` / `vyre_primitives::*`. **Status `shipped`** means a non-test production callsite in `libs/performance/matching/vyre/` invokes the replacement (verified with `grep` below each row). Other rows are `documented only` if the commit message describes a heuristic removal but no production callsite matched the sweep grep.

| Heuristic | Replaced-by | File:Line | Commit | Status |
|---|---|---|---|---|
| Cost-ordered greedy megakernel fusion subset (ignored exchange-graph structure beyond sort order); earlier megakernel wire without full substrate | `schedule_via_homotopy` then `max_fusion_subset` (homotopy seed + matroid intersection); runtime dispatches `select_fused_subset` | `vyre-runtime/src/megakernel/planner.rs` (`select_fused_subset`; doc comment lines 38–40); `vyre-driver-wgpu/src/megakernel.rs` calls `select_fused_subset` | `054fe43959`, `717e7bd9dc`, `36c683d48d` | shipped (`grep -n 'schedule_via_homotopy\\|max_fusion_subset' …/vyre-runtime/src/megakernel/planner.rs`; `grep -n select_fused_subset …/vyre-driver-wgpu/src/megakernel.rs`) |
| `exchange_adj` from same-op pairs only | Adds sheaf diffusion stalk drift: `diffuse_to_equilibrium`, `flag_fusion_incompatible` feeding `exchange_adj` before `select_fused_subset` | `vyre-driver-wgpu/src/megakernel.rs` lines 106–115 (calls into `vyre_driver::self_substrate::sheaf_heterophilic_dispatch::*`) | `a5ea175129` | shipped (`grep -n 'sheaf_heterophilic_dispatch' libs/performance/matching/vyre/vyre-driver-wgpu/src/megakernel.rs`) |
| Discrete-only matroid augmenting steps | Jacobi-smoothed flow weights inside augmenting loop: `matroid_solve_step` | `vyre-driver/src/self_substrate/matroid_megakernel_scheduler.rs` line 129 (`matroid_solve_step`); primitive entry is `vyre_primitives::math::multigrid::jacobi_smooth_step_cpu` inside `multigrid_matroid_solver` | `a5ea175129`, `36c683d48d` | shipped (`grep -n 'matroid_solve_step' libs/performance/matching/vyre/vyre-driver/src/self_substrate/matroid_megakernel_scheduler.rs` excludes tests by reading context; production path is the main `max_fusion_subset` loop) |
| Hand-rolled transitive dependent enumeration for pass invalidation | `pass_descendants` | `vyre-foundation/src/optimizer/scheduler.rs` line 238 `crate::pass_substrate::adjustment_set_pass_dependency::pass_descendants` | `3ab2d34eae` | shipped (`grep -n pass_descendants libs/performance/matching/vyre/vyre-foundation/src/optimizer/scheduler.rs`) |
| Ad-hoc reachability for invalidation | `reachability_closure` | `vyre-foundation/src/optimizer/scheduler.rs` lines 272, 290 | `c07c0e2573` | shipped (`grep -n reachability_closure libs/performance/matching/vyre/vyre-foundation/src/optimizer/scheduler.rs`) |
| Informal “passes commute” reasoning | `passes_commute_on` | `vyre-foundation/src/optimizer/scheduler.rs` line 354 | `fbbab5270b` | shipped (`grep -n passes_commute_on libs/performance/matching/vyre/vyre-foundation/src/optimizer/scheduler.rs`) |
| LRU-style pipeline cache retention | Submodular eviction: `select_retention_set` | `vyre-driver-wgpu/src/runtime/cache/pipeline.rs` line 72 | `7ef81a85d7` | shipped (`grep -n submodular_cache_eviction libs/performance/matching/vyre/vyre-driver-wgpu/src/runtime/cache/pipeline.rs`) |
| Universal substrate access without `vyre-primitives` on backends | Direct `vyre_primitives` deps on driver/runtime/AOT | `Cargo.toml` edits across crates | `70b2c4b64e` | documented only (manifest change; not a heuristic row) |
| Substrate modules lived above driver | Lift `self_substrate` into `vyre-driver`; shim in `vyre-libs` | `vyre-libs/src/lib.rs` re-export | `fd97448ea4` | documented only (packaging; not a heuristic row) |
| Automated agent pass | N/A | N/A | `8c57750b7a` | documented only (no heuristic mapping) |

## Verification commands used for `shipped`

```bash
grep -n 'schedule_via_homotopy\|max_fusion_subset' libs/performance/matching/vyre/vyre-runtime/src/megakernel/planner.rs
grep -n 'select_fused_subset' libs/performance/matching/vyre/vyre-driver-wgpu/src/megakernel.rs
grep -n 'sheaf_heterophilic_dispatch' libs/performance/matching/vyre/vyre-driver-wgpu/src/megakernel.rs
grep -n 'pass_descendants\|reachability_closure\|passes_commute_on' libs/performance/matching/vyre/vyre-foundation/src/optimizer/scheduler.rs
grep -n 'submodular_cache_eviction' libs/performance/matching/vyre/vyre-driver-wgpu/src/runtime/cache/pipeline.rs
grep -n 'matroid_solve_step' libs/performance/matching/vyre/vyre-driver/src/self_substrate/matroid_megakernel_scheduler.rs
```
