# VYRE_TESTING_GAPS  -  Audit Report

**Scope:** `libs/performance/matching/vyre/*/tests`, `vyre-*/src/**/tests`, conformance harness, Surge run/scan/compile integration tests.  
**Date:** 2026-04-24  
**Auditor:** Kimi Code CLI (security-researcher mode)  
**Method:** Static analysis of test file inventory, line counts, inline `#[cfg(test)]` modules, gap-file presence, and targeted source-read of every crate under `vyre/`.

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total `*.rs` test files (integration + unit) | ~112 |
| Total test lines of code | ~19,098 |
| Crates with **zero** dedicated `tests/` directory | 3 (`vyre-frontend-c`, `vyre-runtime megakernel`, `vyre-macros`*) |
| Crates with `tests/` containing **only SKILL.md** | 2 (`vyre-macros`, `vyre-spec`) |
| `should_panic` / `catch_unwind` occurrences in tests | 36 |
| `proptest`/`fuzz`/`quickcheck` test files | 20 / 112 (17.9 %) |
| `unsafe` blocks in `vyre-runtime/src` | 47 |
| `unsafe` blocks exercised by dedicated safety tests | 0* |

*`*`  -  `grep` for `unsafe` inside `vyre-runtime/tests` returned zero hits; all runtime unsafe is exercised only indirectly.

---

## Answers to Scoped Questions

### 1. What percentage of primitives have a CPU parity test?
**100 % (37/37).**  
All registered primitives in `vyre-primitives` carry `test_inputs` + `expected_output` fixtures and are walked by `conform/vyre-conform-runner/tests/parity_matrix.rs:138–357`, which dispatches each primitive through `vyre_reference::reference_eval`.  
*However:* the only primitive-level test file, `vyre-primitives/tests/integration.rs`, is a 25-line smoke test that does **not** exercise `reference_eval`. Primitive parity is guaranteed only by the cross-crate conformance runner, not by the primitive crate itself.

### 2. Any Program-emit tests that check buffer shape (binding, count) end-to-end?
**No.**  
- `vyre-driver-wgpu/tests/megakernel_emit.rs` emits full Programs but asserts only on `@compute`, `atomicCompareExchangeWeak`, and `atomicAdd` in the WGSL text  -  never on `@binding(N)` or array sizes.  
- `vyre-foundation/tests/buffer_decl_with_count.rs` tests constructor invariants only.  
- `vyre-foundation/tests/execution_plan.rs` tests planning-layer `LayoutStrategy`, not emitted shader layout.  
- `vyre-driver-wgpu/src/pipeline_bindings.rs:52–67` tests a WGSL-string parser on hand-written snippets, not emitted Programs.

### 3. Any stress test that runs 10 000+ files through the pipeline?
**No.**  
The `PersistentEngine` design doc cites "10 000 × 1 KiB scan jobs" as a target (`vyre-driver/src/persistent.rs:12`), but no end-to-end test exercises this volume. The closest approximations are random-input stress (10 000 random buffers per op in `vyre-libs/tests/every_op_random_inputs.rs:173`) and disk-cache races (`vyre-runtime/tests/adversarial_disk.rs:81`), neither of which is a compile → run → scan file pipeline.

### 4. Any race-condition test for PersistentEngine concurrent enqueue/claim?
**Yes  -  but only unit-level, inside the implementation file.**  
`vyre-driver/src/persistent.rs:224–416` contains three tests: 4-producer/1-consumer enqueue race, 4-consumer claim race, and wrap-around correctness.  
**No integration tests** in `vyre-driver/tests/`, `vyre-runtime/tests/`, or `vyre-driver-wgpu/tests/` reference `PersistentEngine` at all. The struct has zero call sites outside its own module.

### 5. Any fuzz test for IFDS convergence?
**No.**  
`vyre-libs/src/dataflow/ifds_gpu.rs:204–408` contains deterministic hand-crafted unit tests (disconnected graph, linear CFG, kill, gen, interprocedural edges, cycles). There is no `proptest`, no random supergraph generation, and no assertion that convergence bounds hold under adversarial graph shapes.

### 6. Any property test for Andersen points-to soundness?
**No  -  and zero tests of any kind.**  
`vyre-libs/src/dataflow/points_to.rs` defines `andersen_points_to` and `andersen_points_to_with_shape` but contains no `#[cfg(test)]` module, no `OpEntry` registration, no `ConvergenceContract`, and no proptest verifying subset-closure or monotonicity.

### 7. Any golden-file test for SARIF output?
**No.**  
SARIF emission lives in `libs/tools/surgec/src/output/sarif.rs` (outside `vyre/` proper). Neither `vyre/` nor `surgec/tests/` contains a golden snapshot, schema validation, or `surgec scan --format sarif` end-to-end assertion. The `surgec/tests/golden/` directory exists but contains only `README.md`; no `.vyre` blobs or diff harness.

---

## Detailed Findings (≥15)

Format: `SEVERITY | file:line | description | suggested fix`

### F1  -  CRITICAL | `vyre-driver-wgpu/tests/` | No end-to-end Program-emit test verifies `@binding` indices or buffer counts in generated WGSL. | Add `emit_buffer_binding_assertions` test that builds a Program with `BufferDecl::storage("x",0,…)` and `BufferDecl::output("y",1,…)`, emits WGSL, and asserts `@binding(0)` / `@binding(1)` presence plus `array<…, N>` size.

### F2  -  CRITICAL | `vyre-driver-wgpu/tests/` | No end-to-end test verifies buffer layout alignment constraints in emitted shaders. | Add alignment propagation test: construct `BufferDecl`s with known `size_bytes()`, emit, parse emitted WGSL with `naga`, and assert `naga::TypeInner::Array.stride == size_bytes()` for each element type.

### F3  -  CRITICAL | `vyre-libs/src/dataflow/points_to.rs:1–62` | Andersen points-to analysis is completely untested  -  no unit tests, no property tests, no registration, no CPU oracle. | Implement CPU-reference subset-closure solver; register `OpEntry` + `ConvergenceContract`; add proptest generating random constraint graphs and asserting `p = &q ⟹ q ∈ pts(p)` and transitive closure monotonicity.

### F4  -  HIGH | `vyre-libs/src/dataflow/ifds_gpu.rs:204–408` | IFDS convergence tested only with 7 hand-crafted unit tests; no fuzz or proptest for adversarial graph shapes, max-iteration exhaustion, or non-terminating cycles. | Add `proptest` generating random `(procs, blocks, facts, edges, seeds)`; assert `solve_cpu` reaches fixpoint within `facts * blocks * procs` iterations and that re-running BFS does not grow the reached set.

### F5  -  HIGH | `vyre-driver/src/persistent.rs:224–416` | PersistentEngine race tests are confined to a private `#[cfg(test)]` module; no integration-scale stress test, no multi-threaded enqueue/claim under memory pressure, no test with 10 000+ items. | Extract a `persistent_engine_stress` integration test in `vyre-driver/tests/` that spawns 16 producer + 16 consumer threads, enqueues 100 000 items through a 1 024-slot ring, and asserts zero loss / zero duplication.

### F6  -  HIGH | `vyre/` (workspace root) | No stress test runs 10 000+ real files through compile → run → scan pipeline. | Add a `pipeline_stress_10k` test in `vyre/tests/` or `vyre-runtime/tests/` that generates 10 000 synthetic `.srg` rules, compiles each to `Program`, dispatches through `WgpuBackend` (or reference), and asserts no internal errors or memory leaks.

### F7  -  HIGH | `vyre-frontend-c/` | `vyre-frontend-c` has **no `tests/` directory** and only 3 source files containing `#[test]` attributes. A full compiler pipeline (C → Vyre IR → object file → ELF) has zero dedicated integration tests. | Create `vyre-frontend-c/tests/` with: (a) end-to-end `pipeline.rs` test compiling a minimal C file to `Program`, (b) `elf_linux.rs` test verifying emitted ELF sections, (c) `object_format.rs` round-trip test.

### F8  -  HIGH | `vyre-runtime megakernel/` | `vyre-runtime megakernel` has **no `tests/` directory** and zero test files. It is a dispatch backend with no conformance coverage. | Add `tests/megakernel_dispatch.rs` that exercises the megakernel path against the same Cat-A fixture set used by `vyre-driver-wgpu/tests/cat_a_conform.rs`.

### F9  -  HIGH | `vyre-macros/tests/` | `vyre-macros/tests/` contains **only `SKILL.md`**  -  zero compile-time macro tests. Procedural macros generating `OpEntry` boilerplate are unverified. | Add `trybuild` or `compiletest_rs` tests for each macro: `inventory::submit!` expansion, derive-op validation, and error-message snapshots for malformed inputs.

### F10  -  HIGH | `vyre-spec/tests/` | `vyre-spec/tests/` contains **only `SKILL.md`**  -  zero tests for the specification schema, TOML validation, or golden-sample round-trips. | Add `tests/schema_roundtrip.rs` and `tests/golden_sample_validate.rs` exercising `vyre-spec/src/golden_sample.rs` against committed fixture TOML files.

### F11  -  HIGH | `vyre-primitives/tests/integration.rs:1–25` | The only dedicated primitive-crate test is a 25-line smoke test verifying `all_entries()` is non-empty and `fnv1a32_program` validates. It does **not** run any primitive through `reference_eval`. | Extend to iterate `all_entries()`, call `reference_eval` on every fixture, and assert byte-identity against `expected_output`  -  duplicating the conformance-runner logic locally so the primitive crate is self-testing.

### F12  -  HIGH | `surgec/tests/golden/` | Golden blob diff directory is empty (only `README.md`). The `regenerate-goldens` binary exists, but no committed `.vyre` blobs or `golden_diff.rs` test validates lowered rule outputs. | Populate `surgec/tests/golden/` with canonical `.vyre` blobs for every shipped Tier-1 rule; add `golden_diff.rs` that fails CI when compilation output drifts.

### F13  -  MEDIUM | `vyre-runtime/tests/socket_ingest.rs:1–84` | Socket ingest test covers only a single TCP payload write; no test for backpressure, partial reads, connection reset, or `AF_XDP` / RDMA paths mentioned in the architecture doc. | Add adversarial socket tests: slow reader (backpressure), RST mid-stream, 1 GiB payload, and concurrent multi-connection ingest.

### F14  -  MEDIUM | `vyre-runtime/tests/uring_smoke.rs:1–144` | `io_uring` smoke test reads only from `/dev/zero`; no test for `O_DIRECT`, file-backed buffers, SQ ring overflow, or kernel-5.4 compatibility edge cases. | Expand to `O_DIRECT` tempfile ingest, SQ ring saturation with 10 000 concurrent requests, and graceful degradation when `IORING_SETUP_SQPOLL` is unavailable.

### F15  -  MEDIUM | `vyre-driver/tests/` | The driver crate has only 3 integration test files (`backend_contract.rs`, `gap_duplicate_op_id.rs`, `gap_error_code_catalog.rs`). Two are gap files < 100 lines. No test exercises the full driver lifecycle: parse → plan → emit → dispatch → readback. | Add `driver_lifecycle_e2e.rs` that builds a minimal multi-op Program, runs it through the driver backend registry, and asserts output correctness.

### F16  -  MEDIUM | `vyre-driver-spirv/tests/` | SPIR-V backend has exactly **one** test file (`spirv_parity.rs`). No device-lost recovery, no validation cross-backend, no determinism contract test  -  all of which exist for the wgpu backend. | Port `gap_device_lost_recovery.rs`, `gap_determinism_contract.rs`, and `gap_validation_cross_backend.rs` patterns from `vyre-driver-wgpu/tests/` to a SPIR-V-specific harness.

### F17  -  MEDIUM | `vyre-foundation/src/optimizer/passes/` | Multiple optimizer passes carry open-work comments and lack dedicated pass-level tests beyond the aggregate `optimizer/tests.rs`. | Add per-pass unit tests in `vyre-foundation/src/optimizer/passes/<pass>/tests.rs` with adversarial Programs designed to trigger edge cases for that pass only.

### F18  -  fixed | `vyre-libs/tests/` | Security ops are no longer exempted from fixture requirements. `sanitized_by`, `path_reconstruct`, `dominator_tree`, `bounded_by_comparison`, `taint_flow`, `flows_to`, and `label_by_family` carry concrete fixtures or forward to primitive fixtures through the universal harness. | Keep `composition_discipline.rs` free of security exemptions and run `CARGO_BUILD_JOBS=1 cargo test -p vyre-conform-enforce --test composition_discipline`.

### F19  -  MEDIUM | `conform/vyre-conform-runner/tests/` | C11 parser ops (`vyre-libs::parsing::c11_*`) are skipped from witness/parity checks in `lens_parity.rs:31`, `parity_matrix.rs:162`, and `universal_harness.rs:83` because the CPU reference cannot reproduce them. | Build a CPU-reference C11 lexer/parser oracle or accept the exemption only after a documented architectural decision with a linked issue.

### F20  -  MEDIUM | `vyre/` (workspace) | `unsafe` code in `vyre-runtime/src` (47 occurrences) and `vyre-driver-wgpu/src` (6 occurrences) has **no dedicated safety tests**  -  no Miri runs, no `loom` concurrency tests, no `assert_unchecked` validation tests. | Add a `safety/` test directory with Miri-compatible tests for every `unsafe` contract, and `loom` models for concurrent structures (`PersistentEngine`, `AsyncUringStream`).

### F21  -  LOW | `vyre-core/tests/ops.rs:1–16` | `vyre-core` ops test is 16 lines and only asserts that `OpCode::MAX` fits in `u8`. No semantic coverage for core op variants. | Expand to round-trip every `OpCode` through wire encode/decode and assert discriminant stability.

### F22  -  LOW | `tests/adversarial/` (workspace root) | The 10 adversarial edge-case tests in `tests/adversarial/` average 23 lines each and cover only primitive bitwise / hash edge cases. No adversarial tests for floating-point, graph traversal, or dataflow convergence boundaries. | Add adversarial tests for: `f32` NaN propagation in `vyre-reference`, CSR graph traversal with `u32::MAX` edges, and `bitset_fixpoint` with all-ones seed.

---

## Coverage Heat-Map by Crate

| Crate | `tests/` files | Inline `#[cfg(test)]` | Total test LOC | Gap assessment |
|-------|---------------|----------------------|----------------|----------------|
| `vyre` (root) | 10 adversarial | 0 | 269 | Very shallow; only primitive edge cases |
| `vyre-frontend-c` | **0** | 3 | ~? | **NO TEST DIRECTORY** |
| `vyre-core` | 2 | 0 | 147 | `ops.rs` is 16 lines; minimal |
| `vyre-driver` | 3 | 1 (persistent.rs) | ~500 | Only unit-level persistent engine races; no e2e lifecycle |
| `vyre-runtime megakernel` | **0** | 0 | 0 | **NO TESTS** |
| `vyre-driver-spirv` | 1 | 0 | ~200 | Single parity file; no recovery/validation tests |
| `vyre-driver-wgpu` | 14 | 3 (cache) | ~2,800 | Best GPU backend coverage, but emit gaps remain |
| `vyre-foundation` | 21 | 6 | ~5,500 | Strong wire/proptest coverage; optimizer pass gaps |
| `vyre-intrinsics` | 1 | 0 | 52 | Only hardware_conform; no parity or fuzz |
| `vyre-libs` | 28 | 2 | ~7,200 | Good differential/proptest; IFDS/Andersen gaps |
| `vyre-macros` | **0*** | 0 | 0 | *Only `SKILL.md` |
| `vyre-primitives` | 1 | 1 (fixpoint, graph, etc.) | 79 | Almost entirely reliant on cross-crate conformance runner |
| `vyre-reference` | 6 | 1 | ~1,500 | Good expr proptest; deterministic float reference covered by conform ULP audit |
| `vyre-runtime` | 7 | 0 | ~1,000 | Socket/uring smoke only; no stress, no unsafe safety tests |
| `vyre-spec` | **0*** | 1 | 0 | *Only `SKILL.md` |
| `conform-runner` | 5 | 0 | ~1,800 | Strong parity/cert coverage; C11 parser skipped |
| `conform-enforce` | 2 | 0 | ~1,100 | Good discipline gates; security ops exempted |

---

## Recommended Priority Order

1. **F3 + F4**  -  Add tests for Andersen and IFDS. These are security-analysis primitives with zero / near-zero coverage.
2. **F7 + F8 + F9 + F10**  -  Create test directories for `vyre-frontend-c`, `vyre-runtime megakernel`, `vyre-macros`, `vyre-spec`.
3. **F1 + F2**  -  End-to-end Program-emit buffer binding / alignment tests. Prevents silent shader layout bugs.
4. **F5 + F6**  -  PersistentEngine integration stress + 10k-file pipeline stress. Catches concurrency and memory-pressure bugs at scale.
5. **F12**  -  Populate golden blobs for `surgec`. Prevents compilation drift.
6. **F18 + F19**  -  Remove exemptions for security ops and C11 parser. Placeholder exemptions are debt.
7. **F20**  -  Miri / `loom` safety tests for `unsafe` contracts.

---

*End of report.*
