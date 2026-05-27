# Migration tracker: substrate packaging changes (downstream consumer view)

This note tracks breaking or behavioral packaging changes that affect
crates downstream of the vyre workspace. Claims below are grounded in
`git log 70b2c4b64e..HEAD -- libs/performance/matching/vyre/` as of the
sweep that produced this file.

## Substrate location moved: `vyre_libs::self_substrate` to `vyre_driver::self_substrate` (backward-compat shim)

Commit `fd97448ea4` moves the self-substrate implementation from `vyre-libs` into `vyre-driver` (`git mv` of the module tree) so every backend, runtime, and AOT crate that depends on `vyre-driver` can import substrate without forcing a `vyre-libs` edge.

Downstream code that used `use vyre_libs::self_substrate::*` continues to work: `vyre-libs` re-exports `vyre_driver::self_substrate` under the old path (see `vyre-libs/src/lib.rs` doc comment near the `self_substrate` shim).

New code should prefer `vyre_driver::self_substrate::*` so imports match
the canonical definition site and remain compatible with any dedicated
substrate-crate extraction.

## New dependency on `vyre-primitives` in driver / runtime / AOT crates

Commit `70b2c4b64e` adds `vyre-primitives` as a dependency to `vyre-driver`, `vyre-runtime`, `vyre-aot`, `vyre-driver-wgpu`, `vyre-driver-cuda`, and `vyre-runtime megakernel` so math below `vyre-libs` can call CPU references and Programs directly (`vyre_primitives::*`) without routing through `vyre-libs`.

`vyre-foundation` intentionally does **not** take a `vyre-primitives` dependency (documented cycle with IR types). Foundation keeps inlined CPU references in `cpu_references.rs` and uses `vyre_foundation::pass_substrate::*` for the optimizer-side substrate.

## Foundation `pass_substrate` (parallel to driver `self_substrate`)

Optimizer scheduling in `vyre-foundation` consumes math through `crate::pass_substrate::*` modules (`vyre-foundation/src/pass_substrate/mod.rs`), not through `vyre_driver::self_substrate` (foundation sits below `vyre-driver` in the dependency graph). This is the foundation-local analogue of the old `vyre-libs::self_substrate` idea: same recursion thesis, different crate boundary.

An external work-id `kimi-21182220` is cited in the migration brief for the rename/split narrative; there is no in-repo string match for that id in `git log` or sources at sweep time, so treat this document as the canonical pointer.

## Real wires landed (megakernel, runtime, PassScheduler)

The following integration points moved from “substrate exists” to “called on live paths” in commits between `70b2c4b64e` and `HEAD`:

- **Matroid intersection in megakernel `select_fused_subset`:** `vyre-runtime megakernel` routes fusion subset selection through `vyre_driver::self_substrate::matroid_megakernel_scheduler::max_fusion_subset` after homotopy seeding (`717e7bd9dc`, `36c683d48d`; builds on earlier `054fe43959` wiring).
- **Homotopy + multigrid Jacobi flow weights:** `schedule_via_homotopy` seeds continuous indicators; `matroid_megakernel_scheduler` applies Jacobi-smoothed augmenting flow via `multigrid_matroid_solver::matroid_solve_step` (`36c683d48d`, `a5ea175129`).
- **Sheaf divergence in `exchange_adj`:** `vyre-runtime` megakernel dispatch builds exchange-graph incompatibility bits using `sheaf_heterophilic_dispatch::diffuse_to_equilibrium` and `flag_fusion_incompatible` before calling `select_fused_subset` (`a5ea175129`).
- **`PassScheduler` queries via substrate:** `transitive_dependents`, `reaches` / `invalidation_closure`, and `pair_commutes` delegate to `pass_substrate::adjustment_set_pass_dependency`, `pass_substrate::dataflow_fixpoint`, and `pass_substrate::functorial_pass_composition` (`3ab2d34eae`, `c07c0e2573`, `fbbab5270b`).

## Change list from `git log 70b2c4b64e..HEAD -- libs/performance/matching/vyre/`

Full one-line history captured during the sweep:

```
fbbab5270b vyre-foundation: PassScheduler::pair_commutes via functorial_pass_composition substrate (P-FOUND-1)
c07c0e2573 vyre-foundation: PassScheduler::reaches + invalidation_closure via dataflow_fixpoint substrate (P-FOUND-7)
3ab2d34eae vyre-foundation: PassScheduler::transitive_dependents via adjustment-set substrate (P-FOUND-5)
a5ea175129 vyre-driver+runtime: real wires for multigrid Jacobi and sheaf diffusion (P-DRIVER-2, P-DRIVER-4)
36c683d48d vyre-runtime megakernel: chain homotopy + matroid for fusion grouping (P-DRIVER-3)
717e7bd9dc vyre-runtime megakernel: route select_fused_subset through matroid intersection (P-DRIVER-1)
fd97448ea4 vyre-driver: lift self_substrate down from vyre-libs (P-UNIFY-1)
8c57750b7a agent/gemini-cli-cc3d0603: automated changes
```

## Action required by downstream dialects

- **`use vyre_libs::self_substrate::*`:** Still valid via the re-export shim from commit `fd97448ea4`; no immediate code change required for compatibility.
- **`use vyre_driver::self_substrate::*`:** Preferred for new code and for any crate that already depends on `vyre-driver`.

When pinning versions, expect manifest edges to reflect the new `vyre-primitives` dependency closure under `vyre-driver` / runtime / AOT as of `70b2c4b64e`.
