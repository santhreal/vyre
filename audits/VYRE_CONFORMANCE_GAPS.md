# VYRE_CONFORMANCE_GAPS  -  Audit Report

**Scope:** `vyre-foundation/src/harness`, `vyre-libs/src/harness`, `vyre-primitives/src/harness`, `vyre-intrinsics/src/harness`, `conform/vyre-conform-runner`, `conform/vyre-test-harness`, `vyre-driver/src/registry`, CI workflows.

**Date:** 2026-04-24

**Auditor:** Kimi Code CLI (security researcher mode)

---

## Executive Summary

The vyre conformance harness is **structurally sound but operationally incomplete**. The inventory-based registration system (`inventory::submit!`) works, the parity matrix exists, and the ULP audit is well-designed. The graph/dataflow/security primitive rows owned by the 2026-04-29 closure pass are now fixed; remaining harness-wide rows such as per-op ULP policy and CI matrix structure stay outside this scoped pass.

**Bottom line:** A green CI run does NOT guarantee that compositions work on GPU. The harness has too many skip paths, too many exemptions, and too little adversarial input.

---

## Findings

### CRITICAL

#### C1 | fixed | Dataflow primitive builders now emit concrete Programs
**File:line:** `vyre-libs/src/dataflow/ssa.rs:12`, `points_to.rs:17`, `callgraph.rs:17`, `slice.rs:17`, `range.rs:17`, `escape.rs:17`, `summary.rs:17`, `loop_sum.rs:17`

**Description:** The named files now expose concrete Program builders or CPU helper surfaces with tests and soundness markers. The scoped dataflow modules no longer contain `Program::empty()` bodies.

**Status 2026-04-29:** fixed. The named dataflow builders now emit concrete Programs or CPU helper surfaces with tests and soundness markers; no `Program::empty()` body remains in the scoped dataflow modules.

---

#### C2 | Three security primitives are byte-for-byte identical but claim distinct semantics
**File:line:** `vyre-libs/src/security/flows_to.rs:23`, `sanitized_by.rs:36`, `taint_flow.rs:11`

**Description:** All three bodies are `csr_forward_traverse(shape, frontier_in, frontier_out, 0xFFFF_FFFF)`. Same mask. Same arg shape. Different OP_IDs (`vyre-libs::security::{flows_to,sanitized_by,taint_flow}`). The conformance tests for these ops use self-loop-only graphs (frontier `{0}` → expected `{0}`) where a no-op implementation would pass. The `sanitized_by` docstring claims sanitizer subtraction happens in the stdlib layer, but the vyre op does NOT accept a sanitizer buffer  -  the soundness contract spans two untested layers.

**Suggested fix:** (a) Make `sanitized_by` accept a `sanitizers_in: &str` buffer and perform the subtraction inside the vyre Program, or (b) collapse the three shims into one canonical `forward_reach_step` op and delete the duplicates. Add non-vacuous fixtures (linear chain + kind-mask diversity) that would fail on a no-op.

---

### HIGH

#### H1 | fixed | Dataflow primitive coverage is registered
**File:line:** `vyre-libs/src/dataflow/mod.rs:34`

**Description:** The scoped dataflow primitives now expose harness coverage or local tests for helper-only surfaces. `composition_discipline` should remain free of security/dataflow exemption rows.

---

#### H2 | `matmul_bias` has a stable OP_ID but no OpEntry registration
**File:line:** `vyre-libs/src/math/linalg/matmul.rs:15` (OP_ID_BIAS), `matmul.rs:506` (only `matmul` registered)

**Description:** `matmul_bias` is a real, tested Cat-A composition (unit tests prove parity against `matmul + bias_add`). It has a stable OP_ID (`vyre-libs::math::matmul_bias`) but is NOT registered in `vyre-libs::harness`. It is invisible to the parity matrix, the ULP audit, and the complexity budget gate.

**Suggested fix:** Add `inventory::submit! { crate::harness::OpEntry::new(OP_ID_BIAS, || matmul_bias(...), Some(...), Some(...)) }` alongside the existing `matmul` registration.

---

#### H3 | `label_by_family` registers with zero fixtures
**File:line:** `vyre-libs/src/security/label_by_family.rs:20-27`

**Description:** This is the **only** registered op in the entire workspace with `test_inputs: None, expected_output: None`. It carries a `UniversalDiffExemption` claiming conformance "lives in the primitive", but `vyre_primitives::label::resolve_family` has its own fixtures and is tested. The shim itself is untested. If the shim ever diverges from the primitive (e.g. buffer rename, extra padding), no automated test catches it.

**Suggested fix:** Add a witness fixture that mirrors the primitive's fixture but runs through the shim. Remove the `UniversalDiffExemption`  -  the shim should prove it forwards correctly.

---

#### H4 | `OpEntry::tolerance()` is dead code
**File:line:** `vyre-libs/src/harness.rs:69-78`

**Description:** The `OpEntry` struct declares a `tolerance(&self) -> u32` method with per-op budgets (`softmax=1`, `attention=4`, `silu=1`, `rms_norm_linear=2`). A global `grep -r \"\.tolerance()\"` across `conform/`, `vyre-libs/`, `vyre-primitives/`, `vyre-intrinsics/` returns **zero call sites**. The parity matrix (`parity_matrix.rs:753`) uses `f32_ulp_tolerance(program)` (program-level transcendental detection). The ULP audit (`ulp_audit.rs:210`) also uses `f32_ulp_tolerance(program)`. The `cpu_vs_backend` lens does raw byte comparison and never consults tolerance at all.

**Suggested fix:** Either (a) wire `entry.tolerance()` into `compare_output_buffers` and the lens so per-op budgets override program-level heuristics, or (b) delete the method and document that tolerance is purely program-level. Leaving dead code that looks authoritative is a hazard.

---

#### H5 | `cpu_vs_backend` lens does raw byte-identity with no F32 tolerance
**File:line:** `conform/vyre-test-harness/src/lens.rs:206`, `conform/vyre-conform-runner/src/lens.rs:179`

**Description:** Both `cpu_vs_backend` implementations compare outputs with `if cpu != gpu { return Fail; }`. This is a pure byte-identity check. For F32-producing ops (especially those with transcendentals), GPU backends legitimately diverge by 1–64 ULP. The lens will falsely report divergence on every F32 op. The `parity_matrix` test avoids this bug by using `compare_output_buffers(program, ...)` which applies `f32_ulp_tolerance(program)`, but external consumers using the `cpu_vs_backend` lens directly will hit false positives.

**Suggested fix:** Replace the raw `cpu != gpu` comparison in both lens files with `compare_output_buffers(program, &cpu, &gpu)` from `vyre_conform_runner::fp_parity`. This unifies the definition of "parity" across the workspace.

---

#### H6 | No `ConvergenceContract` lens  -  8 ops never tested in fixpoint loops
**File:line:** `vyre-libs/src/harness.rs:147` (only `fixpoint_contract` lens exists)

**Description:** The harness has `FixpointContract` (for `bitset_fixpoint`) and `ConvergenceContract` (for dataflow/security fixpoint ops). The `fixpoint` lens only checks `fixpoint_contract`. Eight ops register `ConvergenceContract` (`reaching`, `live`, `ifds`, `flows_to`, `taint_flow`, `sanitized_by`, `dominator_tree`, `bounded_by_comparison`) but are NEVER dispatched in a convergence loop by any lens or test. They are tested as single-dispatch ops, which verifies one step, not the fixpoint closure. The `lens_parity.rs` test explicitly skips them because it only checks `fixpoint_contract`.

**Suggested fix:** Extend the `fixpoint` lens (or add a `convergence` lens) to read `convergence_contract(op_id)` and run the same driver-loop logic. Add a test in `lens_parity.rs` that exercises it.

---

#### H7 | Seven security primitives exempted from fixture gate without tracking
**File:line:** `conform/vyre-conform-enforce/tests/composition_discipline.rs:506-524`

**Description:** The `wip_exemptions` list contains seven security ops. While they DO have fixtures (except `label_by_family`), they are explicitly skipped by `every_op_has_test_fixtures_or_is_explicitly_exempt`. There is **no tracking issue** in any comment  -  just a vague reference to "Task #14 (docs/primitives-tier.md)". Without a ticket URL or issue number in a real tracker, these exemptions are invisible debt.

**Suggested fix:** Replace the vague reference with a concrete GitHub issue URL for each exempt op. Add a CI gate that fails if `wip_exemptions.len()` grows (prevent regression).

---

### MEDIUM

#### M1 | Zero property-based / fuzz coverage for primitives and compositions
**File:line:** `vyre-primitives/src/`, `vyre-libs/src/`, `vyre-intrinsics/src/`

**Description:** Proptest and fuzzing exist only in `vyre-foundation` (wire roundtrip, program stats, graph canonicalization, VAST validation). There are **no** proptest strategies for `vyre-primitives` (e.g. random bitset sizes, random graph shapes), **no** proptest for `vyre-libs` nn/math ops (random tensor shapes, adversarial float values), and **no** fuzz targets for the reference interpreter (`vyre-reference`). The `ulp_audit.rs` adversarial companion is the only adversarial float testing in the entire workspace, and it only covers 8 hard-coded values.

**Suggested fix:** Add `vyre-primitives/fuzz/` with targets for every primitive domain (bitset, graph, reduce, nn). Add proptest in `vyre-libs` for numerical ops: random matrix dimensions for matmul, random sequences for prefix_sum, random floats for softmax. The reference interpreter should be fuzzed with arbitrary `Program`s to catch panics.

---

#### M2 | `rms_norm` uses transcendental `inverseSqrt` but per-op tolerance returns 0
**File:line:** `vyre-libs/src/harness.rs:69-78` (returns 0 for `rms_norm`), `vyre-libs/src/nn/norm/rms_norm.rs:55` (uses `UnOp::InverseSqrt`)

**Description:** `rms_norm` computes `1.0 / sqrt(mean_sq + eps)`  -  this lowers to `UnOp::InverseSqrt`, which `fp_parity.rs` classifies as transcendental (64 ULP tolerance). However, `OpEntry::tolerance()` returns 0 for `rms_norm` because it is not in the match arm. Since `tolerance()` is dead code (Finding H4), this discrepancy is latent, but if anyone ever wires it up, `rms_norm` will falsely fail on GPU.

**Suggested fix:** Add `"vyre-libs::nn::rms_norm" => 64,` to `OpEntry::tolerance()` (or delete the method per H4). Ensure the program-level heuristic and any per-op override agree.

---

#### M3 | `layer_norm` uses F32 division/sqrt but is missing from tolerance list
**File:line:** `vyre-libs/src/harness.rs:69-78`, `vyre-libs/src/nn/norm/layer_norm.rs:76`

**Description:** `layer_norm` performs mean, variance, and `sqrt(var + eps)` over F32 buffers. Like `rms_norm`, it contains transcendental operations but is absent from `OpEntry::tolerance()`. The program-level `f32_ulp_tolerance(program)` correctly detects the transcendental and assigns 64 ULP, but the per-op config is wrong.

**Suggested fix:** Add `"vyre-libs::nn::layer_norm" => 64,` to `OpEntry::tolerance()` (or delete the method per H4).

---

#### M4 | `parity_matrix` hard-skips c11 parser ops and subgroup intrinsics
**File:line:** `conform/vyre-conform-runner/tests/parity_matrix.rs:162-196`

**Status 2026-04-29:** c11 parser witness skips were removed from the scoped conformance runner files. The three subgroup intrinsics still use their dedicated warp-collective coverage because the CPU reference cannot simulate subgroup lanes.

**Suggested fix:** For c11 parser ops, either (a) implement a CPU reference that can reproduce the fixtures, or (b) remove the `OpEntry` registrations until the parser stabilizes  -  do not register untestable ops. For subgroup intrinsics, add a `SubgroupBackendLens` that runs against a mock warp simulator so they can participate in the matrix.

---

#### M5 | CI `no-gpu` matrix row verifies zero backend parity
**File:line:** `.github/workflows/ci.yml:44-45`

**Description:** The CI matrix has a `no-gpu` feature flag that compiles with `vyre-driver-wgpu/no-gpu`. In this configuration, `parity_matrix.rs` skips all GPU backends (only the reference backend remains). The `ulp_audit.rs` test prints `[SKIP] gpu feature not enabled; ULP audit requires WgpuBackend.` and returns immediately. The `gap_cert_artifact.rs` test verifies that `prove` REFUSES to emit a certificate. **None of the load-bearing conformance tests actually run.** A PR could break every F32 op and the `no-gpu` row would stay green.

**Suggested fix:** In the `no-gpu` CI row, run `cargo test --workspace --features gpu --test parity_matrix` on a GPU runner, OR restructure CI so that `no-gpu` only checks compilation/clippy and a separate `gpu` job runs all conformance tests. Do not allow a green CI row that exercises zero backends.

---

#### M6 | `live.rs` and `reaching.rs` produce identical IR
**File:line:** `vyre-libs/src/dataflow/live.rs:37`, `reaching.rs:65`

**Description:** Both bodies call `csr_forward_traverse(shape, fin, fout, mask)`. The docstring for `live.rs` says "backward analysis on a forward primitive  -  caller flips edges", but the edge-flipping happens in the caller, not in the op. Two distinct OP_IDs produce byte-identical Programs. The `structural_fingerprint` gate in `composition_discipline.rs` would catch cross-namespace collisions, but these are in the SAME namespace (`vyre-libs::dataflow::*`) so the same-family exemption allows it.

**Suggested fix:** Make `live.rs` call `csr_backward_traverse` (which exists in `vyre-primitives::graph`) so the op's IR actually encodes backward semantics. Alternatively, introduce a `reverse_program_graph` adapter and compose it.

---

#### M7 | `bounded_by_comparison` test uses hard-coded magic number `16`
**File:line:** `vyre-libs/src/security/bounded_by_comparison.rs:38`

**Description:** The test fixture writes `to_bytes(&[16, 16, 16, 16])` for the edge-kind mask. `16` is presumably `edge_kind::DOMINANCE`, but the code uses a raw literal. If `edge_kind::DOMINANCE` is ever renumbered, the test will silently test the wrong edge kind. The graph is also self-loop-only (same vacuity as Finding C2).

**Suggested fix:** Import `edge_kind::DOMINANCE` and use the named constant. Replace the self-loop graph with a real dominance tree where a no-op would fail.

---

#### M8 | Dataflow test fixtures lack edge-kind diversity
**File:line:** `vyre-libs/src/dataflow/reaching.rs:70-72`, `live.rs:42`, `ifds.rs:64`

**Description:** All three registered dataflow ops use fixtures where `pg_edge_kind_mask` is uniformly `1` (or `ASSIGNMENT`). None test the kind-mask gating that is the primary reason these ops accept a mask parameter. A broken implementation that ignored the mask would pass all three tests.

**Suggested fix:** Add at least one fixture per op where the kind-mask filters out some edges. For `ifds.rs`, include "call" and "return" edge kinds and verify that unmatched pairs are not followed.

---

#### M9 | `lens_parity.rs` never exercises `cpu_vs_backend` or `fixpoint` lenses
**File:line:** `conform/vyre-conform-runner/tests/lens_parity.rs:35-51`

**Description:** This test file is described as "the consolidation target that replaces the scattered per-file parity tests". However, it only runs the `witness` lens (CPU-only) and a structural `fixpoint_contract_reachable` check. It does NOT call `cpu_vs_backend` or `fixpoint` with a real backend. The `cpu_vs_backend` lens bug (Finding H5) is therefore latent and undetected.

**Suggested fix:** Add `#[cfg(feature = "gpu")]` tests that run `lens::cpu_vs_backend` and `lens::fixpoint` against the wgpu backend for a representative subset of ops (one per domain: math, nn, graph, bitset).

---

#### M10 | `vyre-driver` DialectRegistry only registers atomic ops + core + io
**File:line:** `vyre-driver/src/registry/registry.rs:128-135`

**Description:** The `DialectRegistry` (used for dispatch-time op resolution) only ingests `OpDefRegistration` entries. In-tree, these are: `core.indirect_dispatch`, four `io.*` ops, and nine `vyre-libs::math::atomic::*` ops. **None of the vyre-primitives, vyre-libs compositions, or vyre-intrinsics are registered in the DialectRegistry**  -  they use the separate `OpEntry` harness registry instead. This means any code path that resolves ops through `DialectRegistry::lookup` (e.g. `Expr::Call` lowering) cannot find `matmul`, `softmax`, `attention`, etc. The registry and the harness are split universes.

**Suggested fix:** Either (a) dual-register every harness op as an `OpDefRegistration` so the DialectRegistry is complete, or (b) document the architectural split explicitly: "Harness ops are for test/parity only; DialectRegistry ops are for dispatch-time resolution." If (b), ensure no dispatch path accidentally looks up a harness op in the DialectRegistry.

---

## Coverage Quantification

| Tier | Total Ops | With `test_inputs` | With `expected_output` | In `parity_matrix` | In `ulp_audit` |
|---|---|---|---|---|---|
| `vyre-libs` | ~85 | ~78 | ~78 | ~65 (skips c11 + exemptions) | ~12 (F32-only) |
| `vyre-primitives` | ~35 | 35 | 35 | 35 | ~8 (F32-only) |
| `vyre-intrinsics` | 9 | 9 | 9 | 6 (3 subgroup skipped) | ~3 (F32-only) |
| **Dataflow (10 claimed)** | 10 | **3** | **3** | **3** | 0 |
| **Security (7 exempt)** | 7 | 6 | 6 | 6 | 0 |

**Key gaps:**
- **7 dataflow primitives:** 0% harness coverage.
- **`matmul_bias`:** 0% harness coverage despite being a real op.
- **`label_by_family`:** 0% fixture coverage.
- **F32 ULP audit:** Only runs when `gpu` feature is enabled; `no-gpu` CI row skips entirely.
- **ConvergenceContract ops:** 0% fixpoint-loop coverage.

---

## Recommendations (Priority Order)

1. **Delete or panic the 7 empty-body dataflow primitives** (C1)  -  silent false negatives are the highest-severity defect.
2. **Fix the `cpu_vs_backend` lens** (H5)  -  unify on `compare_output_buffers` so F32 ops don't falsely fail.
3. **Add a `ConvergenceContract` lens** (H6)  -  8 fixpoint ops are flying blind.
4. **Register `matmul_bias`** (H2) and **add fixtures to `label_by_family`** (H3).
5. **Kill `OpEntry::tolerance()` or wire it up** (H4)  -  dead code that looks authoritative is worse than no code.
6. **Restructure CI** (M5) so the `no-gpu` row does not pretend to test conformance.
7. **Add proptest/fuzz for primitives** (M1)  -  the current deterministic fixtures are too small to catch edge cases.

---

*End of report.*
