# vyre — Master Plan to Release

> Historical import. This file is retained as evidence of the
> 2026-04-18 planning state, but it is not the current plan of record.
> Use `../../../audits/RELEASE_GATE.md` for the active release gate and
> `../../../docs/DOCUMENTATION_GOVERNANCE.md` for precedence. Historical
> claims below that call this file the source of truth are superseded.

**Compiled:** 2026-04-18
**Source documents merged inline:**
- `docs/audits/ROADMAP_PERFORMANCE.md` — every roadmap item P-1..P-34 (text reproduced + per-item gates added)
- `.audits/findings/vyre-deep-analysis.md` — REF / GPU / VAL / CONF / SYNC / EDGE / PERF / MOD / ORPH / ARCH numbered findings (full content)
- `.audits/vyre-conform-p2-closure.md` — closed P0/P1/P2 + outstanding O1-O4
- **NEW** — Pillar 7 deep scan run during this compilation (Section §9)

The plan is organised by **pillar**, not by phase. Each item has explicit gates: the exact `cargo` commands, test names, bench names, and pass thresholds that prove it landed.

---

## §1. Definition of Release

vyre is "release" when **all** of these hold, demonstrated by automated gates not assertion:

1. **Cross-vendor bit-identical determinism** — same IR yields byte-identical output across any conformant backend. Verified by 30-day daily cert across NVIDIA + Apple Silicon (P-19).
2. **Zero CPU↔GPU divergence on any IR program the validator accepts** — validator is tight enough that anything it accepts is well-defined on every backend.
3. **Pipeline dispatch ≤10% per-call overhead** — cert run ~90s in v0.4.0 → <5s with pipeline mode + GPU-resident graph (P-6 + P-8).
4. **Optimizer is provably semantics-preserving** — every `#[vyre_pass]` carries an SMT proof of `∀p. eval(p) == eval(pass(p))` to bounded depth (P-23).
5. **Kernel fusion is byte-identical to unfused** — first provably correct GPU kernel fusion in any Rust GPU compute lib; conform proves it for every fused/unfused pair (P-16).
6. **Subgroup + cooperative-matrix primitives** at parity with hand-tuned CUDA on raw matmul / reduce / scan benchmarks (P-11, P-12).
7. **Surface area ≤500 LOC per file**, every file does one thing, every crate publishes cleanly, every public type documented.
8. **Conform suite 100% green; whole workspace builds with `-D warnings`** including the test backends in `float_semantics.rs`, `admission.rs`, `category_c.rs`.
9. **No mandatory wgpu coupling** — `vyre-conform` certifies a CUDA backend without compiling wgpu; `core` has zero GPU deps.
10. **Replay + bisection** — every cert run deterministically replayable; regression `git bisect`-able to exact commit + dispatch (P-34).

---

## §2. Status Snapshot — 2026-04-18

### Closed today (verified compiling + tests passing)

| ID | What | Where |
|---|---|---|
| P-1 | Registry lookup O(1) (`spec_by_id`) | `vyre-conform/src/vyre-spec/op_registry.rs`, callers updated |
| P-6 | Pipeline-mode dispatch (`compile_native`, passthrough, WgpuPipeline) | `vyre-core/src/pipeline.rs`, `vyre-core/src/backend.rs`, `vyre-wgpu/src/pipeline.rs` |
| P-15 | Optimizer pass framework + `const_fold` + `strength_reduce` | `vyre-core/src/optimizer/{mod, scheduler, rewrite, passes/*}.rs`, `vyre-macros/` |
| P-27 (in-mem half) | In-memory pipeline cache (blake3 keyed) | extends `vyre-wgpu/src/pipeline.rs` |
| O1 | WGSL token-aware mutation | `vyre-conform/src/verify/harnesses/wgsl_mutation/mod.rs` |
| O3 | WeaklyKilledAdversary detection | `vyre-conform/src/meta/{harness, gauntlet}/mod.rs` |

### In flight (Codex agents)
- **codex-1f1f988a:** O2 (CommandEncoder API split) + O4 (rustc_lexer mutation)
- **codex-ce3b7be6:** P-11 (subgroup ops) + P-12 (coop matrix)
- **codex-eb037579:** P-16 (kernel fusion) + P-17 (spec-driven optimizer) + P-18 (proof-carrying dispatches)

When all in-flight agents land cleanly, **9 more items** clear from the open list (eight from the three writers + the auto-resolution of GPU-001 by O2). The plan below counts those as IN FLIGHT, not closed.

---

## §3. PILLAR 1 — Correctness

### 1.1 Reference vs. GPU divergence (🔴 critical)

**REF-001** — `-0.0` truthiness mismatch
- **File:** `vyre-reference/src/value/mod.rs:46`
- **Issue:** `Self::Float(value) => value.to_bits() != 0` returns `true` for `-0.0`; WGSL evaluates `bool(-0.0)` as `false`. Programs whose `Select::cond` evaluates to `-0.0` diverge between CPU reference and GPU.
- **Fix:** `Self::Float(value) => *value != 0.0`
- **Time:** 10 min · **Agent:** me · **Dependency:** none
- **Done if (gates):**
  - `cargo check -p vyre-reference` clean
  - `cargo test -p vyre-reference value::tests::neg_zero_truthiness_is_false` PASS (new regression test asserting `Value::Float(-0.0).is_truthy() == false`)
  - `cargo test -p vyre-reference value::tests::pos_zero_truthiness_is_false` PASS
  - `cargo test -p vyre-reference value::tests::neg_zero_select_branches_to_false` PASS (proptest 100 inputs)
  - Conform: when float_semantics enforcer compiles (Pillar 6), `vyre-conform certify --enforcer float_semantics` includes a `select(-0.0)` adversarial that would have failed before this fix — assert it now passes.

**REF-002** — `BinOp::And` / `Or` typed `U32` but eval returns `Bool`
- **Files:** `vyre-core/src/ir/validate/typecheck.rs` (~208), `vyre-reference/src/typed_ops/mod.rs` (`$and` / `$or` macros)
- **Issue:** `expr_type()` returns `DataType::U32` for `And`/`Or`; reference interpreter produces `Value::Bool`. Programs like `(a && b) + 1` pass validation, fail at interpreter time.
- **Fix:** Make `expr_type` return `DataType::Bool` for `And`/`Or`. Update validator's binop-operand check accordingly.
- **Time:** 30 min · **Agent:** me · **Dependency:** none
- **Done if (gates):**
  - `cargo check -p vyre` clean (downstream type users still happy)
  - `cargo test -p vyre ir::validate::typecheck::tests::and_or_type_is_bool` PASS
  - `cargo test -p vyre ir::validate::expr_rules::tests::and_then_arithmetic_rejected` PASS (regression: program `(a && b) + 1` now fails validation, used to pass)
  - `cargo test -p vyre-reference typed_ops::tests::and_returns_bool_value` PASS (existing test still green)
  - Property: `cargo test -p vyre prop_validate_implies_eval_succeeds` PASS over 200 random programs (any program the validator accepts must execute on the reference interpreter without type errors)

### 1.2 GPU backend bugs (🔴 critical)

**GPU-001** — External `CommandEncoder` mutated before error return (4 modules)
- **In flight via O2** (codex-1f1f988a). Closes when that lands.
- **Done if:** files at `vyre-wgpu/src/engine/{dfa,dataflow,decode/dispatch/gpu,decompress/dispatch_kernel}/mod.rs` show split into `*_immediate` + `*_record` functions; `cargo test -p vyre-wgpu --test encoder_split_*_end_to_end` PASS for each engine.

**GPU-002** — `is_cached_device` compares wrapper addresses
- **File:** `vyre-wgpu/src/runtime/device/device.rs:58-61`
- **Issue:** Two clones of the same `wgpu::Device` live at different addresses; `is_cached_device` says they're different even though `PartialEq` says equal.
- **Fix:** Use `device.global_id()` comparison (stable per logical device).
- **Time:** 20 min · **Agent:** me · **Dependency:** none
- **Done if (gates):**
  - `cargo check -p vyre-wgpu` clean
  - `cargo test -p vyre-wgpu runtime::device::tests::cloned_device_is_recognized_as_cached` PASS (regression)
  - `cargo test -p vyre-wgpu engine::dfa::tests::scan_shared_works_with_cloned_device` PASS

**GPU-003** — `PooledBuffer::buffer()` panics
- **File:** `vyre-wgpu/src/runtime/cache/buffer_pool.rs:133-137`
- **Issue:** `expect()` in production method on `Option<Buffer>`.
- **Fix:** `pub fn buffer(&self) -> Option<&wgpu::Buffer>` or `Result<&wgpu::Buffer, BufferPoolError>`. Update callers to `?`-propagate. Either is acceptable; pick `Result` so error context survives.
- **Time:** 15 min · **Agent:** me or kimi · **Dependency:** none
- **Done if (gates):**
  - `cargo check -p vyre-wgpu` clean (~3-4 callers updated)
  - `cargo test -p vyre-wgpu runtime::cache::buffer_pool::tests::buffer_returns_err_after_drop` PASS (regression — proves no panic)
  - `grep -rn 'PooledBuffer::buffer' vyre-wgpu/src` shows no callsites doing `.unwrap()` on the result

**GPU-004** — `element_size_bytes` panics on `Array` / `Tensor` output buffers
- **Files:** `vyre-wgpu/src/lib.rs:350-358`, `vyre-wgpu/src/pipeline.rs:300-313`, `vyre-core/src/ir/validate/validate.rs`
- **Issue:** Validator accepts `Array` / `Tensor` as output buffer element type; `element_size_bytes` then panics during dispatch.
- **Fix:** Add `validate_output_buffer_element_type` that rejects `Array` / `Tensor` for output buffers.
- **Time:** 30 min · **Agent:** me · **Dependency:** none
- **Done if (gates):**
  - `cargo check -p vyre` clean
  - `cargo test -p vyre ir::validate::tests::array_output_buffer_rejected` PASS (regression)
  - `cargo test -p vyre ir::validate::tests::tensor_output_buffer_rejected` PASS
  - `cargo test -p vyre-wgpu --test backend_dispatch_does_not_panic_on_array_output` PASS (proves panic vector closed)

### 1.3 IR validation gaps (🟡)

**VAL-001** — atomic index type unchecked
- **File:** `vyre-core/src/ir/validate/atomic_rules.rs:63`
- **Time:** 15 min · **Agent:** kimi
- **Done if:** `_index` parameter unprefixed; `cargo test -p vyre ir::validate::atomic_rules::tests::atomic_index_must_be_u32` PASS (adversarial: f32 index now rejected); `cargo test -p vyre ir::validate::atomic_rules::tests::atomic_with_u32_index_accepted` PASS.

**VAL-002** — `Fma` / `Select` operand types unchecked
- **File:** `vyre-core/src/ir/validate/expr_rules.rs:91-103`
- **Time:** 30 min · **Agent:** kimi
- **Done if:** `cargo test -p vyre ir::validate::expr_rules::tests::fma_requires_f32_operands` PASS, `select_requires_matching_branch_types` PASS, plus proptest `prop_validated_fma_evaluates_without_typeerror` over 100 inputs.

**VAL-003** — arithmetic `BinOp` operand mismatch unchecked
- **File:** `vyre-core/src/ir/validate/typecheck.rs:21-27`
- **Time:** 25 min · **Agent:** kimi
- **Done if:** `cargo test -p vyre ir::validate::typecheck::tests::add_with_mixed_types_rejected` PASS (regression); `prop_validated_arith_evaluates_without_typeerror` over 200 inputs.

**VAL-004** — bitwise unary restricts `U64` incorrectly
- **File:** `vyre-core/src/ir/validate/typecheck.rs`
- **Time:** 15 min · **Agent:** kimi
- **Done if:** `cargo test -p vyre ir::validate::typecheck::tests::bitnot_u64_accepted` PASS, plus the same for popcount/clz/ctz/reversebits.

### 1.4 Conform / test infra (🟡)

**CONF-001** — silent `u32::MAX` clamp in atomics probe
- **File:** `vyre-conform/src/enforce/enforcers/atomics.rs:752`
- **Time:** 15 min · **Agent:** kimi · **Dependency:** Pillar 6 must land first to test
- **Done if:** `cargo test -p vyre-conform enforce::enforcers::atomics::tests::probe_glue_returns_err_on_overflow` PASS.

**CONF-002** — `build_scan` false positive on `pub const REGISTERED` prefix
- **File:** `vyre-build-scan/src/flat.rs:149-153`
- **Time:** 20 min · **Agent:** kimi
- **Done if:** `cargo test -p vyre-build-scan flat::tests::registered_foo_const_does_not_match_registered` PASS (regression).

### 1.5 Concurrency observations (🟢)

**SYNC-001** — `OomAllocator` panic poisons locks
- **File:** `vyre-conform/src/meta/oom/alloc.rs:74-89`
- **Time:** 5 min · **Agent:** me
- **Done if:** module-level doc comment documents the limitation explicitly; no code change.

**SYNC-002** — `wgsl_already_validated` redundant validation on poison
- **File:** `vyre-conform/src/pipeline/backend/wgpu/dispatch.rs:33-43`
- **Time:** 20 min · **Agent:** kimi
- **Done if:** uses `parking_lot::RwLock` (no poisoning) OR `into_inner()` on poison; `cargo test -p vyre-conform pipeline::backend::wgpu::dispatch::tests::wgsl_already_validated_survives_poisoned_lock` PASS.

### 1.6 Edge cases (🟢)

**EDGE-001** — `MAX_DECODE_DEPTH` clarity
- **File:** `vyre-core/src/ir/serial/wire/mod.rs`
- **Time:** 5 min · **Agent:** me
- **Done if:** comment updated.

**EDGE-002** — negative `LitI32` powers of two skipped in strength reduction
- **File:** `vyre-core/src/optimizer/passes/strength_reduce.rs`
- **Time:** 30 min · **Agent:** kimi
- **Done if:** `cargo test -p vyre optimizer::passes::strength_reduce::tests::negative_power_of_two_reduced` PASS; `cargo bench -p vyre --bench optimizer_throughput` shows additional reductions.

**EDGE-003** — `pipeline_cache_shard` uses `DefaultHasher`
- **File:** `vyre-wgpu/src/runtime/shader/compile_compute_pipeline.rs`
- **Time:** 15 min · **Agent:** kimi
- **Done if:** uses `FxHash`; `cargo test -p vyre-wgpu` clean.

**Pillar 1 subtotal: 18 items, ~5 hours wall-clock with 3-4 agents in parallel.**

---

## §4. PILLAR 2 — Performance (FULL ROADMAP MERGED HERE)

Roadmap items follow the original ordering (P-1..P-34) with full text + gates. PERF-* findings interleave alphabetically as separate sub-section.

### 4.1 Roadmap Cat 1 — Free money (days, not weeks)

**P-1** — Registry lookup O(n) → O(1)
- **DONE.** `vyre-conform/src/vyre-spec/op_registry.rs::spec_by_id`. Verified with `cargo check -p vyre-conform --lib` (clean) + `gemini-test-account` confirming hot-path used.

**P-2** — Arena-backed `Value`
- **Spec:** `Value::Bytes(Arc<[u8]>)` + `Vec<Value>` slotted locals instead of `HashMap<String, Value>` in reference interpreter. Roadmap: "5-10× on the reference interpreter."
- **Time:** 4 h · **Agent:** codex · **Dependency:** none
- **Done if (gates):**
  - `cargo check -p vyre-reference` clean
  - `cargo test -p vyre-reference` ALL existing tests PASS (no regressions)
  - `cargo bench -p vyre-reference --bench eval_throughput` shows ≥3× speedup vs baseline (record baseline before merging; commit baseline.json into bench fixtures)
  - Property test `cargo test -p vyre-reference prop_arena_value_matches_hashmap_baseline` PASS over 200 random programs (arena-backed evaluator MUST produce same outputs as old hashmap evaluator — keep old as `#[cfg(test)]` reference)

**P-3** — Constant folding + strength reduction
- **DONE via P-15** (Codex's framework includes both as registered passes).
- **Verify:** `cargo test -p vyre optimizer::passes::const_fold::tests` PASS (4 tests), `optimizer::passes::strength_reduce::tests` PASS (2 tests).

**P-4** — `Node` uses `Ident` not `String`
- **Spec:** halve Program clone cost. "Mechanical sed."
- **Time:** 2 h · **Agent:** kimi · **Dependency:** none
- **Done if (gates):**
  - `cargo check --workspace` clean (touches every Node user)
  - `cargo test --workspace` ALL existing tests PASS
  - `cargo bench -p vyre --bench wire_roundtrip` shows ≥30% reduction in clone time
  - Memory bench `cargo bench -p vyre --bench program_clone_memory` shows ≥40% reduction in allocation count for a 100-node program

**P-5** — Zero-copy output-slice readback
- **Spec:** Program declares "output bytes X..Y"; backend reads back only those bytes.
- **Time:** 4 h · **Agent:** codex · **Dependency:** P-6 (done)
- **Done if (gates):**
  - `cargo check -p vyre -p vyre-wgpu` clean
  - `cargo test -p vyre-wgpu --test output_slice_readback` PASS (proves only declared bytes are read back)
  - `cargo bench -p vyre-wgpu --bench small_output_readback` shows ≥10× reduction in readback bytes for a 16MB output where consumer needs 4 bytes

### 4.2 Roadmap Cat 2 — Dispatch model overhaul

**P-6** — Pipeline dispatch mode
- **DONE.** Verified: `cargo test -p vyre pipeline:: --lib` (4 PASS), `cargo test -p vyre --test pipeline_mode_end_to_end` (3 PASS).

**P-7** — Streaming/chunked dispatch
- **Spec:** `push_chunk(bytes) → GPU processes chunk N while CPU stages N+1`. Sustains 100% GPU util on >GPU-RAM inputs.
- **Time:** 6 h · **Agent:** codex · **Dependency:** P-6 (done)
- **Done if (gates):**
  - `cargo check -p vyre -p vyre-wgpu` clean
  - `cargo test -p vyre-wgpu --test streaming_dispatch_end_to_end` PASS — including a 10GB synthetic input split into 1MB chunks
  - `cargo test -p vyre-wgpu --test streaming_chunk_n_overlaps_chunk_n_minus_one` PASS (verifies overlap actually happens via timing assertions, not just sequential)
  - `cargo bench -p vyre-wgpu --bench streaming_throughput` shows sustained ≥80% GPU util on a 4GB input (must include nvidia-smi or wgpu profiler hooks to measure util — not just wall clock)

**P-8** — GPU-resident dispatch graph (FIRST-IN-CLASS)
- **Spec:** "Emit indirect dispatch graph the GPU executes itself. One CPU→GPU launch, hundreds of sequential kernels on-device. Ports CUDA Graphs to every backend via the vyre IR. Saves 10-100µs per op."
- **Time:** 1 day · **Agent:** codex · **Dependency:** P-6 (done)
- **Done if (gates):**
  - `cargo check -p vyre -p vyre-wgpu` clean
  - `cargo test -p vyre-wgpu --test gpu_resident_graph_end_to_end` PASS — assembles a 100-op program into a single graph + dispatches with one CPU launch
  - `cargo bench -p vyre-wgpu --bench graph_dispatch` shows ≥50× reduction in CPU-side launch overhead vs sequential dispatch (100-op program)
  - Conform: graph-dispatched and sequentially-dispatched outputs MUST be byte-identical for the same Program — assertion in the integration test, not relegated to "trust me"

**P-9** — Temporal coalescing
- **Spec:** If a Program is dispatched 1000×/sec with different inputs, runtime batches N dispatches into ONE compound dispatch. 10× faster on rule scanner.
- **Time:** 6 h · **Agent:** codex · **Dependency:** P-6, P-8
- **Done if (gates):**
  - `cargo test -p vyre-wgpu --test temporal_coalescing_correctness` PASS — coalesced and uncoalesced outputs byte-identical
  - `cargo bench -p vyre-wgpu --bench rule_scanner_packet_stream` shows ≥5× throughput improvement vs uncoalesced baseline on a 10K-packet stream

**P-10** — Async copy + multi-stream execution
- **Spec:** `Node::AsyncLoad { tag } / Node::AsyncWait { tag }`. Compute + memcpy overlap.
- **Time:** 6 h · **Agent:** codex · **Dependency:** P-6, PERF-030
- **Done if (gates):**
  - `cargo test -p vyre ir::nodes::tests::async_load_async_wait_round_trip` PASS
  - `cargo test -p vyre-wgpu --test async_copy_overlaps_compute` PASS (timing-based assertion: total wallclock < sum of individual stages)
  - `cargo bench -p vyre-wgpu --bench async_copy_overlap` shows ≥30% reduction in wallclock for a copy-then-compute pattern vs serial baseline

### 4.3 Roadmap Cat 3 — GPU primitive unlock

**P-11** — Subgroup / SIMD-lane ops (DEFINING FEATURE)
- **In flight via codex-ce3b7be6.**
- **Done if:** `vyre-core/src/ops/primitive/subgroup_*.rs` exists for each op family; `cargo test -p vyre ops::primitive::subgroup` ALL PASS; `cargo bench -p vyre --bench subgroup_reduce` shows ≥4× speedup over barrier-based reduce on 1M u32.

**P-12** — Cooperative matrix (tensor cores)
- **In flight via codex-ce3b7be6.**
- **Done if:** `cargo test -p vyre --test coop_matrix_end_to_end` PASS; `cargo bench -p vyre --bench coop_matrix_gemm` shows ≥10× speedup over naive workgroup-shared matmul on 1024×1024.

**P-13** — Indirect/conditional dispatch
- **Spec:** dispatch count determined by prior kernel's output. Foundational for sparse workloads.
- **Time:** 4 h · **Agent:** kimi · **Dependency:** none
- **Done if (gates):**
  - `cargo test -p vyre ir::nodes::tests::indirect_dispatch_round_trip` PASS
  - `cargo test -p vyre-wgpu --test indirect_dispatch_end_to_end` PASS — dispatch count from prior kernel correctly drives next kernel's launch
  - `cargo bench -p vyre-wgpu --bench sparse_workload_throughput` shows ≥3× improvement on a 90%-sparse input vs dense dispatch

**P-14** — Workgroup-shared auto-sizing
- **Spec:** IR-level dataflow proves a bound on SRAM requirement. Emits tightest-possible shader.
- **Time:** 6 h · **Agent:** codex · **Dependency:** none
- **Done if (gates):**
  - `cargo test -p vyre lower::wgsl::tests::workgroup_size_inferred_from_dataflow` PASS — declared SRAM matches dataflow-computed bound
  - `cargo test -p vyre lower::wgsl::tests::workgroup_overrun_rejected` PASS — programs that need more SRAM than declared rejected before lowering
  - `cargo bench -p vyre-wgpu --bench workgroup_packing` shows shader memory footprint reduced by ≥30% for representative ops

### 4.4 Roadmap Cat 4 — Optimizer 2 → 9 passes

**P-15** — Standard optimizer passes framework
- **DONE.** Codex landed framework + 2 reference passes (const_fold, strength_reduce). Verified.

**P-16** — Kernel fusion (DEFIES PRIOR ART)
- **In flight via codex-eb037579** (`vyre-core/src/optimizer/passes/fusion.rs` 416 LOC already on disk).
- **Done if:** `cargo test -p vyre --test fusion_correctness` PASS — `eval(unfused) == eval(fused)` for ≥50 random inputs per pattern; `cargo test -p vyre --test fusion_idempotency` PASS — `pass(pass(p)) == pass(p)`; `cargo bench -p vyre --bench fused_pipeline` shows ≥2× speedup; proptest `prop_fusion_preserves_eval` PASS over 200 random valid Programs.

**P-17** — Spec-driven optimizer (UNIQUE APPROACH)
- **In flight via codex-eb037579** (`vyre-core/src/optimizer/passes/spec_driven.rs` 294 LOC already on disk).
- **Done if:** `cargo test -p vyre --test spec_driven_rewrites` PASS — at minimum: commutative canonicalisation, idempotent dedup, identity-element fold, involutive pair-cancel each have a positive + negative test; eval-preservation proptest over 100 inputs.

**P-18** — Proof-carrying dispatches (FIRST-IN-CLASS)
- **In flight via codex-eb037579.**
- **Done if:** `OptimizationProof` struct exists in `vyre-core/src/optimizer/proof.rs`; `cargo test -p vyre --test proof_cert` PASS — verifies every applied pass writes its `(op_id, law)` entry; `proof.blake3_input` matches the pre-pass program hash; `proof.blake3_output` matches post.

### 4.5 Roadmap Cat 5 — Correctness moats

**P-19** — Cross-vendor bit-identical determinism (THE MOAT)
- **Spec:** Same IR on two vendors today → different bytes. Strict-mode cert gate forces bit-identity across NVIDIA/AMD/Apple/Intel. First in the world. Daily CI.
- **Time:** 1 week engineering + **30 days calendar** burn-in · **Agent:** codex · **Dependency:** Apple Silicon CI hardware; CR-Math IEEE754 reference
- **Done if (gates):**
  - `vyre-core/src/ieee754.rs` ships with CR-Math-grade transcendentals (sin, cos, tan, exp, log, pow at correctly-rounded precision)
  - `cargo test -p vyre-conform --test cross_vendor_bit_identity_local` PASS (local probe — runs same IR on two backend instances simulating different vendors via float-mode flags)
  - **Calendar gate:** 30 consecutive days of green daily cert across at least 2 real vendors (NVIDIA local + Apple Silicon CI). Captured as a signed monthly attestation file in `vyre-sigstore/`. Without 30 days of evidence, claim is not made.

**P-20** — ULP-budget approximate compilation
- **Spec:** Call site declares `max_error: 2ulp`. vyre emits fastest shader that provably meets that bound.
- **Time:** 1 day · **Agent:** codex · **Dependency:** none
- **Done if (gates):**
  - `DispatchConfig.ulp_budget: Option<u8>` field added (additive, non-breaking)
  - `cargo test -p vyre lower::wgsl::tests::ulp_budget_zero_uses_strict_path` PASS
  - `cargo test -p vyre lower::wgsl::tests::ulp_budget_two_uses_fast_approx` PASS — emitted shader provably stays within 2 ulp on a fixed witness set
  - `cargo bench -p vyre-wgpu --bench ulp_budget_speedup` shows ≥2× speedup at 4-ulp budget vs strict mode for transcendentals

**P-21** — Multi-GPU work stealing
- **Spec:** Run ONE `vyre::Program` across N GPUs.
- **Time:** 2 days engineering, needs ≥2 GPUs in one box · **Agent:** codex · **Dependency:** physical hardware
- **Done if (gates):**
  - `cargo check -p vyre-wgpu` clean
  - `cargo test -p vyre-wgpu --test multi_gpu_partition_unit` PASS (mocked partitioner, no real second GPU needed)
  - `cargo test -p vyre-wgpu --test multi_gpu_real_dispatch` PASS — only when a 2-GPU box is available; gated on `VYRE_TWO_GPUS=1` env var
  - `cargo bench -p vyre-wgpu --bench multi_gpu_scan` shows ≥1.6× speedup on 2 GPUs vs 1 GPU for a 1M element prefix sum

**P-22** — GPU-native fuzzing
- **Spec:** `vyre::fuzz::gen_program` random valid IR + differential test reference vs backend.
- **Time:** 4 h · **Agent:** kimi · **Dependency:** none
- **Done if (gates):**
  - `vyre::fuzz` module exists with `gen_program(seed: u64) -> Program` returning syntactically valid Programs
  - `cargo test -p vyre fuzz::tests::generated_programs_pass_validation` PASS over 1000 seeds
  - `cargo run --bin vyre-fuzz-runner -- --seeds 1000 --backend wgpu` discovers ≥1 known-failing pattern from the regression corpus (proves fuzzer is actually stressing the backend)

**P-23** — SMT proof per optimizer pass
- **Spec:** For each `#[vyre_pass]`, emit bounded SMT problem proving `∀p (depth ≤ 8). dispatch(p) == dispatch(pass(p))`. Z3 proves it. Ship Z3 proof in release.
- **Time:** 1 week (design-bound, mostly serial thinking) · **Agent:** codex · **Dependency:** P-15 (done), P-17
- **Done if (gates):**
  - `vyre-core/src/optimizer/smt.rs` module with `prove_pass_preserves_semantics(pass: &dyn Pass, depth: u8) -> Result<Z3Proof, SmtError>`
  - `cargo test -p vyre optimizer::smt::tests::const_fold_proven_to_depth_8` PASS — actually invokes Z3 binary, returns UNSAT-of-negation
  - Per-pass proof artifacts under `proofs/<pass-name>.smt2` committed and validated in CI
  - `cargo run --bin vyre-prove-all` produces a proof report listing every registered pass + its SMT verdict; CI gates on every pass having a green proof

### 4.6 Roadmap Cat 6 — Deep-pass additions

**P-24** — Persistent kernels (STREAMING WORKLOAD KILLER)
- **Spec:** Kernel stays resident on GPU, pulls work from GPU-side queue. Eliminates per-batch launch overhead.
- **Time:** 1 day · **Agent:** codex · **Dependency:** P-6, P-8
- **Done if (gates):**
  - `cargo test -p vyre-wgpu --test persistent_kernel_round_trip` PASS — kernel stays alive across 1000 work items pulled from queue
  - `cargo bench -p vyre-wgpu --bench persistent_kernel_streaming` shows ≥10× throughput improvement over batch-launch baseline for small batches (≤1024 elements each)

**P-25** — AOT kernel specialization + on-disk cache
- **Spec:** Compile specialized versions for observed argument shapes. Cache on disk keyed by spec hash + backend fingerprint.
- **Time:** 6 h · **Agent:** codex · **Dependency:** P-6 (done)
- **Done if (gates):**
  - `cargo test -p vyre-wgpu --test aot_specialization_cache_hits` PASS — second process invocation completes in <100ms (no WGSL compile) for a program that took ≥500ms first time
  - `cargo test -p vyre-wgpu --test aot_specialization_invalidates_on_backend_fingerprint_change` PASS — driver-version change invalidates cache, recompile happens
  - On-disk format: `~/.cache/vyre/aot/<spec-hash>-<backend-fp>.{wgsl,meta}` documented in code comments

**P-26** — Profile-guided backend routing (PGO for GPU)
- **Spec:** Cert gate measures each op's latency on every backend; runtime routes by measured fastest.
- **Time:** 6 h · **Agent:** kimi · **Dependency:** cert-gate latency histograms
- **Done if (gates):**
  - `vyre::routing::PgoRouter` reads `~/.config/vyre/pgo.toml` (or `OUT_DIR` baked-in defaults)
  - `cargo test -p vyre routing::tests::pgo_picks_fastest_backend_per_op` PASS
  - PGO file generated as a side-effect of `vyre-conform certify` runs; format committed under `docs/pgo-format.md`

**P-27** — Program hash → pipeline cache
- **In-memory variant DONE.** Disk variant outstanding.
- **Spec for disk variant:** `blake3(wire_format(program))` → on-disk cache of lowered WGSL + metadata.
- **Time:** 4 h · **Agent:** me · **Dependency:** P-6 (done)
- **Done if (gates):**
  - `cargo test -p vyre-wgpu --test pipeline_cache_disk_persistence` PASS — second process run loads from disk, skips WGSL lowering
  - `cargo bench -p vyre-wgpu --bench cold_start_with_cache` shows ≥80% reduction in first-dispatch latency vs no-cache baseline

**P-28** — Constant-buffer folding + shader monomorphization
- **Spec:** Buffer with compile-time constant contents (e.g. 256-entry LUT) inlined into shader as immediate array.
- **Time:** 6 h · **Agent:** kimi · **Dependency:** none
- **Done if (gates):**
  - `cargo test -p vyre lower::wgsl::tests::const_buffer_inlined_when_compile_time_known` PASS — emitted WGSL contains literal array, no buffer binding for that input
  - `cargo bench -p vyre --bench aes_sbox_lookup` shows ≥30% improvement vs buffer-bound LUT

**P-29** — Dead buffer elimination (as optimizer pass)
- **Spec:** IR dataflow proves a declared buffer feeds no output. Skip allocation + upload + bind.
- **Time:** 4 h · **Agent:** kimi · **Dependency:** P-15 (done)
- **Done if (gates):**
  - `vyre-core/src/optimizer/passes/dead_buffer_elim.rs` registered with framework
  - `cargo test -p vyre optimizer::passes::dead_buffer_elim::tests::unread_buffer_removed` PASS
  - `cargo test -p vyre optimizer::passes::dead_buffer_elim::tests::output_buffer_preserved` PASS (negative — never remove an output)
  - `prop_dead_buffer_elim_preserves_eval` PASS over 100 random programs

**P-30** — Shared-nothing parallelism detection
- **Spec:** IR dataflow proves two ops share no writable state → emit as concurrent dispatches on separate command queues.
- **Time:** 6 h · **Agent:** codex · **Dependency:** none
- **Done if (gates):**
  - `vyre-core/src/ir/transform/parallelism.rs::analyze_concurrency(program)` returns set of `{op_id → stream_id}` assignments
  - `cargo test -p vyre ir::transform::parallelism::tests::write_after_write_serialised` PASS
  - `cargo test -p vyre ir::transform::parallelism::tests::independent_writes_parallelised` PASS
  - `cargo bench -p vyre-wgpu --bench parallel_streams_speedup` shows ≥1.5× speedup for a program with 4 independent ops vs serial dispatch

**P-31** — Distribution-aware algorithm selection (SELF-OPTIMIZING)
- **Spec:** Record observed data distribution per call site. Next call picks algorithm fitting the distribution.
- **Time:** 1 day · **Agent:** codex · **Dependency:** P-26
- **Done if (gates):**
  - `cargo test -p vyre routing::tests::skewed_input_picks_radix_sort` PASS
  - `cargo test -p vyre routing::tests::small_input_picks_insertion_sort` PASS
  - All algorithm variants byte-identical via existing conform suite (no new gate needed; the existing per-op conform proves equivalence)

**P-32** — Backend capability fingerprinting
- **Spec:** Hash actual backend behaviour (subgroup size, rounding, transcendental ULP) into the cert. Driver-update drift detection.
- **Time:** 4 h · **Agent:** kimi · **Dependency:** none
- **Done if (gates):**
  - `vyre::cert::BackendFingerprint::probe(backend) -> [u8; 32]` returns a deterministic blake3 over a fixed witness set's outputs
  - `cargo test -p vyre cert::tests::fingerprint_stable_across_runs` PASS (1000 invocations same value)
  - `cargo test -p vyre cert::tests::fingerprint_diverges_on_simulated_driver_change` PASS — mock backend with different rounding produces different fingerprint

**P-33** — Numerical-stress determinism verification
- **Spec:** Every op cert includes runs against NaN / infinity / overflow-at-boundary / subnormal witnesses.
- **Time:** 6 h · **Agent:** kimi · **Dependency:** none
- **Done if (gates):**
  - `vyre-conform/src/witnesses/numerical_stress.rs` defines NaN, ±Inf, ±0, MIN_NORMAL, subnormal-entry, denormal-exit witnesses
  - Every float-typed op spec automatically includes these witnesses in its KAT corpus
  - `cargo test -p vyre-conform witnesses::numerical_stress::tests::all_float_ops_have_stress_kats` PASS

**P-34** — Replay-based bisection
- **Spec:** Every cert run emits replay log. `cargo vyre replay cert-2026-04-17.log` reruns exactly that sequence on any machine.
- **Time:** 6 h · **Agent:** kimi · **Dependency:** none
- **Done if (gates):**
  - `cargo run --bin vyre-conform-cli -- replay <log-file>` reruns and produces byte-identical output
  - `cargo test -p vyre-conform replay::tests::replay_log_round_trip` PASS — record, replay, assert outputs match
  - Replay log format documented in `docs/replay-format.md`

### 4.7 PERF-* findings (deep analysis additions to roadmap)

These are NOT in the original roadmap but were surfaced by the deep audit. Each has the same gate format.

**PERF-001** — CSE clones HashMap per branch
- **File:** `vyre-core/src/ir/transform/optimize/cse/impl_csectx.rs:8`
- **Time:** 30 min · **Agent:** codex
- **Done if:** `cargo bench -p vyre --bench cse_isolation` shows ≥40% reduction in allocations on a 100-block program; `cargo test -p vyre ir::transform::optimize::cse::tests` ALL PASS (no semantics regression).

**PERF-002** — CSE redundant clone on miss path
- **File:** same file:42
- **Time:** 5 min · **Agent:** codex
- **Done if:** `grep -n 'value.clone()' vyre-core/src/ir/transform/optimize/cse/impl_csectx.rs` returns 0 hits at line 42; existing CSE tests PASS.

**PERF-003** — CSE Call args allocate Vec unconditionally
- **File:** same file:124
- **Time:** 15 min · **Agent:** codex
- **Done if:** uses `Cow<[Expr]>`; `cargo test` clean; `cargo bench` shows reduction in alloc count for Call-heavy programs.

**PERF-004** — `rewrite_program` clones buffer list per pass
- **File:** `vyre-core/src/optimizer/rewrite.rs:3-9`
- **Time:** 20 min · **Agent:** codex
- **Done if:** operates on `&[BufferDecl]`; bench `optimizer_throughput` shows reduction.

**PERF-005..007** — `fold_expr` / `rewrite_expr` clone on no-match; DCE clones surviving nodes
- **Files:** `vyre-core/src/optimizer/rewrite.rs:33-117`, `vyre-core/src/ir/transform/optimize/dce/eliminate_dead_lets.rs:17-77`
- **Time:** combined 2.25 h · **Agent:** codex (one focused session)
- **Done if:** `cargo bench -p vyre --bench optimizer_throughput` shows ≥30% reduction in allocations across optimizer passes; `cargo test -p vyre optimizer::` ALL PASS; proptest `prop_optimizer_idempotent_after_alloc_reduction` PASS over 100 random programs.

**PERF-008/009/022** — `BarrierState` allocation patterns
- **Files:** `vyre-core/src/lower/wgsl/node.rs`
- **Time:** combined 1 h · **Agent:** kimi
- **Done if:** `BarrierState` uses `SmallVec<[String; 4]>` (covers 99% case); CoW-style fork instead of clone-per-branch; `cargo bench -p vyre --bench lower_wgsl` shows reduction.

**PERF-010/011** — Fresh buffer + `pad_to_words` allocation per dispatch
- **Files:** `vyre-wgpu/src/lib.rs`, `vyre-wgpu/src/pipeline.rs`
- **Time:** 2 h combined · **Agent:** me · **Dependency:** P-6 (done) gives the pipeline-level cache; this finishes the per-dispatch buffer reuse.
- **Done if:** `cargo bench -p vyre-wgpu --bench repeated_dispatch_same_shape` shows ≥50% reduction in per-dispatch allocation count for 1000 dispatches of same Program.

**PERF-012** — String matching haystack always allocates
- **File:** `vyre-wgpu/src/engine/string_matching.rs:487-493`
- **Time:** 30 min · **Agent:** kimi
- **Done if:** `cargo bench -p vyre-wgpu --bench string_matching_repeat_haystack` shows constant alloc-per-scan after first call.

**PERF-013/014/028/029/043** — Decomposition enforcer allocation sweep
- **File:** `vyre-conform/src/enforce/enforcers/decomposition.rs`
- **Time:** combined 2 h · **Agent:** kimi
- **Done if:** `cargo bench -p vyre-conform --bench decomposition_scaling` shows ≥40% reduction in alloc count at registry size 100; existing tests ALL PASS.

**PERF-015** — Composition-proof ConformSpec clones
- **File:** `vyre-conform/src/enforce/enforcers/layer7_composition_proof.rs:148-157`
- **Time:** 30 min · **Agent:** kimi · **Dependency:** P-1 (done) — uses `spec_by_id` returning `&'static ConformSpec`
- **Done if:** function signatures take `&ConformSpec` not `ConformSpec`; `cargo bench -p vyre-conform --bench composition_proof_throughput` shows reduction.

**PERF-016** — Optimizer scheduler lacks dirty tracking
- **File:** `vyre-core/src/optimizer/scheduler.rs:47-64`
- **Time:** 1.5 h · **Agent:** codex
- **Done if:** scheduler tracks per-pass dirty bit; `cargo test -p vyre optimizer::scheduler::tests::dirty_tracking_skips_clean_passes` PASS; `cargo bench --bench optimizer_throughput` shows ≥40% wallclock reduction on a stable Program (where most passes don't fire after the first).

**PERF-017** — Linear scan for pass metadata
- **File:** same:94-99
- **Time:** 15 min · **Agent:** codex
- **Done if:** uses `HashMap<&'static str, usize>`; `cargo test` clean.

**PERF-018** — CSE key allocates Vec per Call
- **File:** `vyre-core/src/ir/transform/optimize/cse/impl_exprkey.rs:32-34`
- **Time:** 30 min · **Agent:** codex
- **Done if:** uses `SmallVec<[ExprKey; 4]>`; CSE tests PASS; bench shows alloc reduction for Call-heavy programs.

**PERF-019** — Lowering string-length check is O(output size)
- **File:** `vyre-core/src/lower/wgsl/emit_wgsl.rs:27-34`
- **Time:** 25 min · **Agent:** kimi
- **Done if:** check hoisted; `cargo bench --bench lower_wgsl` shows constant-time-per-buffer.

**PERF-020** — DFA epsilon-closure clones+sorts
- **File:** `vyre-std/src/pattern/nfa_to_dfa.rs:253-255`
- **Time:** 1 h · **Agent:** kimi
- **Done if:** memoizes closure by NFA-state-set hash; `cargo bench -p vyre-std --bench dfa_assemble_bench` shows ≥30% reduction.

**PERF-021** — Failure-link construction clones outputs
- **File:** `vyre-wgpu/src/engine/string_matching.rs:359-381`
- **Time:** 45 min · **Agent:** kimi
- **Done if:** incremental build instead of clone-extend; `cargo bench --bench string_matching_assemble` shows reduction.

**PERF-023/024** — Hasher swaps
- **Time:** combined 25 min · **Agent:** kimi
- **Done if:** `FxHashMap` / `FxHashSet` in CSE + decomposition; existing tests PASS.

**PERF-025** — `SmallVec` audit
- **Time:** 30 min · **Agent:** kimi
- **Done if:** at least the 3 listed sites use `SmallVec`; bench shows allocation reduction.

**PERF-026/027** — Iterator chain pre-sizing
- **Time:** combined 40 min · **Agent:** kimi
- **Done if:** every `.collect()` in inliner + `compact.rs` uses `with_capacity`; bench shows alloc reduction.

**PERF-030** — Runtime sync-over-async
- **File:** `vyre-wgpu/src/runtime/device/device.rs:81` + 3 callers
- **Time:** 2 h · **Agent:** codex
- **Done if:** `WgpuBackend::dispatch_async() -> impl Future<Output = Result<...>>` exists; `cargo test -p vyre-wgpu --test async_dispatch_does_not_block_thread` PASS; `cargo bench --bench async_dispatch_concurrency` shows ≥4× throughput improvement at 8-way concurrency.

**PERF-031** — Batch execution sequential
- **File:** `vyre-conform/src/pipeline/streaming/batch_execution/mod.rs:300-365`
- **Time:** 1.5 h · **Agent:** codex
- **Done if:** uses rayon for case parallelism; `cargo bench -p vyre-conform --bench batch_execution_concurrency` shows ≥3× speedup at 4 threads.

**PERF-032..035** — Buffer pool overhaul
- **File:** `vyre-wgpu/src/runtime/cache/buffer_pool.rs`
- **Time:** combined 2 h · **Agent:** codex (PERF-033 sharding) + kimi (rest)
- **Done if:** sharded by size class; `MAX_BUFFERS_PER_CLASS` raised to 64; `cargo bench --bench buffer_pool_contention` shows ≥5× throughput improvement at 16-thread concurrency vs current single-mutex baseline.

**PERF-036/037/038/039** — Multi-buffer readback consolidation
- **Files:** `vyre-wgpu/src/engine/{decode,dfa,string_matching}/`
- **Time:** combined 3 h · **Agent:** kimi
- **Done if:** each engine uses single packed readback buffer + single poll; `cargo bench --bench {decode,dfa,string_matching}_readback_latency` shows ≥30% reduction in readback wallclock.

**PERF-040** — `encode_parts` byte-by-byte extend
- **File:** `vyre-wgpu/src/runtime/serializer/encode_parts.rs:70-78`
- **Time:** 30 min · **Agent:** kimi
- **Done if:** single bulk copy after upfront size sum; `cargo bench --bench wire_roundtrip` shows reduction for multi-part programs.

**PERF-041..045** — String-formatting hot-path sweep
- **Time:** combined 2 h · **Agent:** kimi
- **Done if:** `cargo bench --bench {lower_wgsl, decomposition, layer1_executable_spec}` show reductions; existing tests PASS.

**PERF-046..050** — Lock contention sweep
- **Files:** various
- **Time:** combined 3.5 h · **Agent:** mostly kimi, codex on 047 (resource queue replacement)
- **Done if:** PERF-046 sharded `DRIVER_CACHES`, PERF-047 `crossbeam::ArrayQueue` for ScanResources, PERF-048 `OnceCell`, PERF-049 `dashmap::DashMap`, PERF-050 `dashmap::DashSet`. All have `cargo bench --bench *_concurrency` showing ≥2× scaling at 8 threads.

**PERF-051..056** — Bench coverage gaps
- **Time:** combined 4.5 h · **Agent:** kimi (one focused session adding all 6 bench files)
- **Done if:** `cargo bench --bench {real_dispatch_latency, cse_isolation, gpu_dfa_scan, buffer_pool_contention, decomposition_scaling, string_matching_scan}` ALL run cleanly and produce stable numbers; baselines committed under `benches/baselines/<bench>.json`.

**Pillar 2 subtotal: 90 items (34 roadmap + 56 PERF), ~120 engineering hours, ~28-32 hours wall-clock with full fleet utilisation.**

---

## §5. PILLAR 3 — Modularity & Extensibility

**MOD-001..003** — Unused deps in `core`
- **File:** `vyre-core/Cargo.toml`
- **Time:** 30 min combined · **Agent:** me
- **Done if:** `cargo machete -p vyre` reports zero unused dependencies; `cargo build -p vyre` clean; downstream consumers (`vyre-conform`, `vyre-wgpu`) still build.

**MOD-004** — `vyre-wgpu` mandatory in `conform`
- **File:** `vyre-conform/Cargo.toml:42`
- **Time:** 1 h · **Agent:** codex
- **Done if:** `cargo build -p vyre-conform --no-default-features` clean (no wgpu deps pulled); `cargo build -p vyre-conform --features gpu` clean; doc explains the feature gate.

**MOD-005/006** — Dual backend abstractions + duplicate wgpu wrapper
- **Files:** `vyre-conform/src/pipeline/backend/`
- **Time:** combined 7 h · **Agent:** codex
- **Done if:** single backend abstraction project-wide; `WgslBackend` is thin extension over `VyreBackend`; `cargo check --workspace` clean; `cargo test -p vyre-conform` (after Pillar 6) ALL PASS.

**MOD-007** — Only 2 optimizer passes
- Closes naturally as P-16/17/18/29 land.

**MOD-008** — `fingerprint_program` slow brittle hash
- **File:** `vyre-core/src/optimizer/mod.rs:139-143`
- **Time:** 30 min · **Agent:** me
- **Done if:** uses `blake3` over Program wire-format bytes; `cargo bench -p vyre --bench fingerprint_throughput` shows ≥10× speedup over `format!("{program:?}")` baseline.

**MOD-009** — Mass `unreachable_pub` warnings
- **Time:** 2 h · **Agent:** kimi
- **Done if:** `cargo check --workspace -- -W unreachable-pub` reports zero warnings on the 9 hotspots from the audit; running it on the full workspace reports ≤20 (down from "hundreds").

**MOD-010** — `conform` API contradicts documented intent
- **File:** `vyre-conform/src/lib.rs`, `pipeline/mod.rs`
- **Time:** 2 h · **Agent:** codex
- **Done if:** public surface trimmed to documented 10 items; `cargo doc --no-deps -p vyre-conform 2>&1 | grep -c '^pub'` matches documented count.

**MOD-011** — `lower_anonymous` exposed as production API
- **File:** `vyre-core/src/lower/wgsl/mod.rs:62-68`
- **Time:** 15 min · **Agent:** me
- **Done if:** guarded with `#[cfg(any(test, feature = "test-helpers"))]`; downstream consumers (none in production today) updated.

**MOD-012** — `pad_to_words` etc duplicated
- **File:** `vyre-wgpu/src/{lib, pipeline}.rs`
- **Time:** 30 min · **Agent:** me
- **Done if:** extracted to `vyre-wgpu/src/util.rs`; both files import; `cargo test -p vyre-wgpu` clean.

**MOD-013** — Conform wgpu dispatch duplicates `vyre-wgpu`
- Closes as MOD-006 lands.

**Pillar 3 subtotal: 13 items, ~14 hours wall-clock.**

---

## §6. PILLAR 4 — Organization & Architecture

**ARCH-001..005** — Monolithic modules (cross-referenced with deep scan §9 finding additional ones)

**ARCH-001** — `float_semantics.rs` 2,686 lines
- **Time:** 4 h · **Agent:** codex
- **Done if:** every file under `vyre-conform/src/enforce/enforcers/float_semantics/` ≤500 LOC; inline WGSL extracted to `*.wgsl` files included via `include_str!`; `cargo test -p vyre-conform enforce::enforcers::float_semantics` (after Pillar 6) ALL PASS.

**ARCH-002** — `vyre-conform/build.rs` 1,076 lines
- **Time:** 6 h · **Agent:** codex
- **Done if:** `vyre-conform/build.rs` ≤200 LOC orchestrating only; heavy parsing/codegen lives in `vyre-build-scan`; `cargo build -p vyre-conform` produces identical `OUT_DIR/*.rs` outputs (byte-for-byte) before vs after.

**ARCH-003** — `vyre-core/build.rs` 339 lines
- Closes as ARCH-002 lands (extracts to `vyre-build-scan`).

**ARCH-004** — `primitive/mod.rs` 751 lines
- **Time:** 2 h · **Agent:** kimi
- **Done if:** split per-op-family files; aggregator ≤200 LOC.

**ARCH-005** — `phase6_calibrate.rs` 473 lines
- **Time:** 2 h · **Agent:** kimi
- **Done if:** binary ≤30 LOC delegating to `pipeline/calibrate/` library module.

**ARCH-006/007** — Naming
- **Time:** combined 4 h · **Agent:** codex / me
- **Done if:** ARCH-006 keep `vyre` crate name, drop workspace name redundancy; ARCH-007 drop `lib.name` rename in `vyre-sigstore`. `cargo check --workspace` clean.

**ARCH-008** — Excluded demos rotting
- **Time:** 4 h (rewire, not delete) · **Agent:** codex
- **Done if:** demos compile against current core API (rewired per memory's "don't delete" rule); removed from workspace exclude; `cargo check -p vyre-demo-rust-lexer-gpu -p vyre-demo-rust-parser-gpu` clean.

**ARCH-009** — `vyre-conform/src/vyre-spec/` vs `vyre-spec` collision
- **Time:** 3 h · **Agent:** codex
- **Done if:** `vyre-conform/src/vyre-spec/` renamed (proposed: `vyre-conform/src/specs/` with explicit doc on the distinction); imports updated; `cargo check -p vyre-conform` clean.

**ARCH-010/011** — Conform false zero-dep claim + type duplication
- **Time:** combined 4 h · **Agent:** codex
- **Done if:** doc-comment fixed; redundant type wrappers replaced with `vyre_spec::*` re-exports; `cargo check -p vyre-conform` clean.

**ARCH-012** — Mixed `mod.rs` and flat module styles
- **Time:** 8 h · **Agent:** codex
- **Done if:** every `module/mod.rs` migrated to `module.rs` + `module/` peer (Rust 2018+ style); `cargo check --workspace` clean; commit message lists every renamed pair.

**ARCH-013** — Inconsistent pluralization
- **Time:** 2 h · **Agent:** kimi
- **Done if:** convention documented in `docs/code-style.md`; offending names renamed to match; `cargo check --workspace` clean.

**ARCH-014** — Duplicate `engine` module names
- **Time:** 1 h · **Agent:** me
- **Done if:** one renamed (proposed: `vyre-conform/src/specs/engine_specs/`); imports updated.

**ARCH-015** — Backend trait split (covered by MOD-005/006)

**ARCH-016** — `pipeline` name collision
- **Time:** 1 h · **Agent:** me
- **Done if:** `vyre-conform/src/pipeline/` renamed to `vyre-conform/src/runner/`; imports updated.

**ARCH-017** — `automod` hides module structure
- **Time:** 3 h · **Agent:** codex
- **Done if:** `vyre-build-scan` emits explicit `mod x; mod y;` lists into a single included file; `grep -c explicit module list` workspace-wide returns 0.

**ARCH-018** — Deep indirection for generated registries
- **Time:** 2 h · **Agent:** codex
- **Done if:** `vyre-build-scan/README.md` documents every `OUT_DIR/*.rs` mapping.

**ARCH-019** — Tests scattered across 3 locations
- **Time:** 4 h · **Agent:** codex
- **Done if:** convention documented; existing files moved to convention; `cargo test --workspace` ALL existing tests still PASS.

**ARCH-020** — Exceptional `[[test]]` entry
- **Time:** 15 min · **Agent:** me
- **Done if:** entry justified in comment OR removed (and test moved to standard location).

**ARCH-021** — Massive lint-allow block
- **Time:** 4 h · **Agent:** codex
- **Done if:** every `#![allow(...)]` in `vyre-core/src/lib.rs` either replaced with localized `#[allow]` at the offending site OR justified with a per-allow comment that survives a future-clippy update; `cargo clippy --workspace -- -D warnings` clean.

**Pillar 4 subtotal: 19 items, ~50 engineering hours, ~25-30 hours wall-clock with parallelism.**

---

## §7. PILLAR 5 — Orphaned & Dead Code

**ORPH-001** — `vyre-core/src/bytemuck_safe.rs` dead
- **Time:** 30 min · **Agent:** me
- **Done if:** `grep -rn 'bytemuck_safe' vyre-core/src vyre-conform/src vyre-wgpu/src vyre-reference/src vyre-std/src` returns 0 hits at use-sites; module + declaration removed; `cargo check -p vyre` clean.

**ORPH-002** — `ir/transform/compiler/*` orphaned
- **Time:** 4 h (rewire don't delete per memory rule) · **Agent:** codex
- **Done if:** files moved into demo crates OR rewired to current core API + un-excluded from workspace; either way `cargo check --workspace` clean.

**ORPH-003** — `backend_is_wgsl()` never used
- **Time:** 15 min · **Agent:** me
- **Done if:** confirmed unused via cross-crate grep; deleted; `cargo check -p vyre` clean.

**ORPH-004** — Unused import in wire tags
- **Time:** 5 min · **Agent:** me
- **Done if:** removed.

**ORPH-005** — `INPUT_HASH_VERSION` constant unused
- **Time:** 5 min · **Agent:** me
- **Done if:** removed.

**ORPH-006** — `GeneratedOutput.sha256` field unread
- **Time:** 30 min · **Agent:** me
- **Done if:** field consumed for incremental rebuild detection (per master plan's intent) OR removed.

**Pillar 5 subtotal: 6 items, ~6 hours wall-clock.**

---

## §8. PILLAR 6 — Conform Test-Compile Errors (BLOCKER)

These are pre-existing test-compile errors that block running ANY test in `vyre-conform`. Listed in the audit doc as "pre-existing test compilation errors in float_semantics.rs, admission.rs, category_c.rs". Without fixing these, none of the conform-side gates above can be exercised.

**CONF-INFRA-001** — Test backends missing `VyreBackend` impl
- **Files:** `vyre-conform/src/enforce/enforcers/float_semantics.rs:929/1171/1180/1187`, `admission.rs`, `category_c.rs`
- **Time:** 4 h · **Agent:** codex (one focused session)
- **Done if (gates):**
  - `cargo test -p vyre-conform --no-run` clean (compiles)
  - Each test backend (`ReductionBackend`, `b3_subnormal::tests::Backend`, `OracleBackend`) has full `VyreBackend` impl including `compile_native` default-impl
  - `BuildError` implements `std::fmt::Display` (E0277 fix)
  - `cargo test -p vyre-conform --lib` runs at least 1000 tests with no compile errors (some may fail for OTHER reasons — that's separate finding territory)

**Pillar 6 subtotal: 1 item, 4 hours. CRITICAL — blocks all conform-side test gates above.**

---

## §9. PILLAR 7 — Deep Scan (NEW findings, 2026-04-18)

These were not in the existing audits. Found by direct codebase scan during master-plan compilation.

### 9.1 Additional LAW 7 violations (files >500 LOC)

The original audit called out 4-5 oversized files. The actual count is **25 files >500 LOC**. Beyond ARCH-001/002/004/005, these need splits too:

| File | Lines | Effort | Agent |
|---|---|---|---|
| `vyre-conform/src/enforce/enforcers/category_b.rs` | 1100 | 4 h | codex |
| `vyre-conform/src/enforce/enforcers/reference_trust.rs` | 922 | 3 h | codex |
| `vyre-conform/src/enforce/enforcers/atomics.rs` | 843 | 3 h | codex |
| `vyre-conform/src/enforce/enforcers/signature_match.rs` | 759 | 3 h | kimi |
| `vyre-conform/src/verify/properties/tests/declared_laws.rs` | 756 | 2 h | kimi |
| `vyre-conform/src/enforce/enforcers/zero_stubs.rs` | 736 | 3 h | kimi |
| `vyre-conform/src/enforce/enforcers/structural_rules.rs` | 736 | 3 h | kimi |
| `vyre-conform/src/vyre-spec/builder.rs` | 710 | 2 h | kimi |
| `vyre-conform/src/verify/harnesses/wgsl_mutation/mod.rs` | 705 | 2 h | me |
| `vyre-conform/src/vyre-spec/string/tokenize/mod.rs` | 704 | 2 h | kimi |
| `vyre-conform/src/proof/algebra/gpu_checker/mod.rs` | 695 | 3 h | codex |
| `vyre-conform/src/proof/algebra/audit/mod.rs` | 673 | 2 h | kimi |
| `vyre-wgpu/src/engine/decode/dispatch/gpu.rs` | 658 | 3 h | codex |
| `vyre-conform/src/proof/algebra/mandatory_inference/cross/mod.rs` | 656 | 2 h | kimi |
| `vyre-wgpu/src/engine/string_matching.rs` | 625 | 3 h | codex |
| `vyre-conform/src/enforce/enforcers/composition_closure.rs` | 622 | 2 h | kimi |
| `vyre-conform/src/enforce/enforcers/decomposition.rs` | 615 | 2 h | kimi |
| `vyre-conform/src/enforce/enforcers/engine_composition.rs` | 609 | 2 h | kimi |
| `vyre-conform/src/enforce/enforcers/category_a.rs` | 594 | 2 h | kimi |
| `vyre-conform/src/enforce/enforcers/no_silent_wrong.rs` | 588 | 2 h | kimi |
| `vyre-conform/src/enforce/enforcers/category_c.rs` | 582 | 2 h | kimi |
| `vyre-conform/src/enforce/enforcers/barrier.rs` | 579 | 2 h | kimi |
| `vyre-conform/src/pipeline/suite/implementation.rs` | 548 | 2 h | kimi |
| `vyre-core/src/ir/transform/inline/expand/impl_calleeexpander.rs` | 511 | 2 h | kimi |

**Done if (gates):** every file in the list ≤500 LOC; `find . -name '*.rs' -path '*/src/*' -not -path '*/target/*' -exec wc -l {} \; | awk '$1 > 500' | wc -l` returns 0; `cargo test --workspace` (after Pillar 6) ALL existing tests PASS.

### 9.2 Net-new correctness findings

**NEW-COR-001** — `unwrap()` audit: 76 calls in non-test code
- **Scope:** workspace-wide
- **Issue:** Each `.unwrap()` in non-test code is a panic vector. 76 of them across `vyre-core/src`, `vyre-wgpu/src`, `vyre-reference/src`, `vyre-std/src`.
- **Time:** 4 h (one careful sweep) · **Agent:** kimi
- **Done if:** every non-test `.unwrap()` either replaced with `?`-propagation OR documented inline as "infallible because <invariant>"; `grep -rn '\.unwrap()' vyre-core/src vyre-wgpu/src vyre-reference/src vyre-std/src | grep -v 'tests' | wc -l` returns ≤20 with documented justifications for the survivors.

**NEW-COR-002** — `expect()` without "Fix:" prefix
- **Files:** `vyre-core/src/optimizer/passes/{const_fold, strength_reduce}.rs:81/85/104/107/114`, `vyre-core/src/ops/security_detection/catalog/detect_*.rs:107/139/162/163`, `vyre-core/src/ops/graph/dfs.rs:388`
- **Issue:** Engineering standard requires every `expect()` to start with `"Fix: ..."`. These don't.
- **Time:** 30 min · **Agent:** kimi
- **Done if:** every `.expect("...")` in non-test code starts with `Fix:`; CI lint enforces with `grep -rn '\.expect("' vyre-core/src | grep -v 'Fix:' | grep -v tests` returning 0.

**NEW-COR-003** — `panic!()` in non-test code (vyre-wgpu)
- **Files:** `vyre-wgpu/src/{lib, pipeline}.rs` — 6 panics in `element_size_bytes`
- **Issue:** Already partly covered by GPU-004, but ALL three arms (Array / Tensor / catch-all) panic. If GPU-004's validator fix lands, these panics become unreachable but still violate the no-panic rule.
- **Time:** 30 min · **Agent:** me · **Dependency:** GPU-004 lands first
- **Done if:** all three panics replaced with `Result::Err(BackendError::new("..."))` propagation; `grep 'panic!' vyre-wgpu/src/{lib, pipeline}.rs` returns 0.

### 9.3 Net-new performance findings

**NEW-PERF-001** — `vyre-core/src/ir/transform/inline/expand/impl_calleeexpander.rs:511 LOC`
- **Issue:** Expanded callee inlining is a hot-path. 511 LOC suggests opportunity for splitting + per-arity specialisation.
- **Time:** 3 h · **Agent:** codex (paired with the LAW 7 split)
- **Done if:** file split into ≤500 LOC per piece; `cargo bench -p vyre --bench inline_throughput` (new) shows ≥20% improvement.

**NEW-PERF-002** — Per-test bench coverage absent
- **Issue:** Several large enforcers (`category_a.rs`, `category_b.rs`, `reference_trust.rs`) have no Criterion benches. We can't measure regressions against them.
- **Time:** 4 h · **Agent:** kimi
- **Done if:** each enforcer >500 LOC has a corresponding `vyre-conform/benches/<enforcer>_throughput.rs`; baselines committed.

### 9.4 Net-new modularity findings

**NEW-MOD-001** — `vyre-macros` is a workspace member but its API surface isn't documented
- **Issue:** `vyre-macros/src/lib.rs` defines the `#[vyre_pass]` proc-macro but no README, no examples, no `#[doc]` on public types.
- **Time:** 1 h · **Agent:** me
- **Done if:** `vyre-macros/README.md` exists; every public macro has a doc-comment with example; `cargo doc --no-deps -p vyre-macros` produces non-empty output.

**NEW-MOD-002** — No `cargo machete` / `cargo udeps` integration
- **Issue:** Unused dependencies (MOD-001/002/003) found by manual audit. CI doesn't enforce.
- **Time:** 30 min · **Agent:** me
- **Done if:** `.github/workflows/dependency-audit.yml` (or equivalent) runs `cargo machete` on every PR; failing build on any unused dep.

### 9.5 Net-new architecture findings

**NEW-ARCH-001** — Workspace exclude comment claims "rewire in 0.4.1" — the rewire never happened
- **File:** `Cargo.toml:30-35`
- **Issue:** Comment says "Rewire in 0.4.1 (tracked)" but it's been kicking around since 0.4.0. Either rewire (per ORPH-002) or remove the misleading comment.
- **Time:** absorbed into ORPH-002 · **Agent:** codex
- **Done if:** demos compile or the exclude is documented as permanent.

**NEW-ARCH-002** — `vyre-conform/fuzz` is excluded but not deleted
- **File:** `Cargo.toml:34-35`
- **Issue:** Same pattern: "rewire in 0.4.1 scope". Either fix or delete.
- **Time:** 2 h · **Agent:** codex
- **Done if:** rewired to 10-item public conform API OR moved to `vyre-conform/src/tests/` (per the comment's suggestion).

**NEW-ARCH-003** — vyre has 12 workspace members but no documentation of crate purposes
- **Issue:** A new contributor cloning the repo can't quickly understand what `core` vs `conform` vs `spec` vs `vyre-reference` vs `std` does.
- **Time:** 1 h · **Agent:** me
- **Done if:** `README.md` workspace section lists each member with a one-line purpose statement.

### 9.6 Net-new test/coverage findings

**NEW-TEST-001** — `vyre-core/tests/ops/primitive/math/test_sub_sat.rs` is the only top-level math test
- **Issue:** ARCH-020 calls out the exceptional `[[test]]` entry. The deeper question: why is sub_sat the only top-level integration test for primitive math? Other primitives (add_sat, mul_sat, etc.) need parity.
- **Time:** 2 h · **Agent:** kimi
- **Done if:** `vyre-core/tests/ops/primitive/math/` has integration tests for every saturating-arith op; `[[test]]` entries either consolidated into one or removed entirely.

**NEW-TEST-002** — Bench baselines not committed
- **Issue:** Existing benches don't have `benches/baselines/<bench>.json` snapshots, so "regression" detection is impossible — there's no record of yesterday's numbers.
- **Time:** 1 h initial + ongoing · **Agent:** me
- **Done if:** every `cargo bench` run writes a baseline file; CI compares new runs against committed baseline; bench regressions fail the build at >5% slower.

### 9.7 Net-new conform findings

**NEW-CONF-001** — `zero_stubs.rs` enforcer can be evaded by case variation
- **File:** `vyre-conform/src/enforce/enforcers/zero_stubs.rs:357-366`
- **Issue:** The enforcer checks `upper.starts_with("TODO")` etc. A comment like `// To do: ...` (with a space) bypasses it. So does `// T0DO` (digit zero).
- **Time:** 30 min · **Agent:** kimi
- **Done if:** detector handles common evasions (whitespace, l33t-speak); regression test for each evasion.

**NEW-CONF-002** — `gemini` 76 unwrap surface is partially in conform code too
- **Note:** The 76 `unwrap()` count from NEW-COR-001 was ONLY in non-conform crates. A separate sweep of `vyre-conform/src` found additional ones — bundled into the same fix.
- **Time:** absorbed into NEW-COR-001

### 9.8 Pillar 7 subtotal

24 net-new items uncovered by deep scan:
- 21 LAW 7 split candidates (§9.1)
- 3 correctness items
- 2 performance items
- 2 modularity items
- 3 architecture items
- 2 test items
- 2 conform items

(Some items in 9.1 also count as architecture, but I list them once for accountability.)

**Total deep scan effort: ~50 hours engineering, ~15 hours wall-clock with parallelism.**

---

## §10. Critical Path

Items that gate other items:

```
Pillar 6 (CONF-INFRA-001)  ──────────→ ALL conform-side gates verifiable

P-15 (DONE) ──┬→ P-16 (in flight) ──→ P-23 (SMT proofs)
              ├→ P-17 (in flight) ──┘
              ├→ P-18 (in flight)
              └→ P-29 (dead buffer elim)

P-6 (DONE) ──┬→ P-7 (streaming)
             ├→ P-8 (graph) ──┬→ P-9 (temporal)
             │                └→ P-24 (persistent kernels)
             ├→ P-10 (async copy)
             ├→ P-25 (AOT)
             └→ P-27 (cache, in-mem done; disk variant remaining)

PERF-030 (async dispatch) ──→ P-10 (async copy)

MOD-006 (drop conform wgpu wrapper) ──→ MOD-013 (auto-closes)
ARCH-002 (build.rs split) ──→ ARCH-003 (auto-closes)

P-26 (PGO) ──→ P-31 (distribution-aware algo selection)

GPU-004 ──→ NEW-COR-003 (panic removal once validator catches)
```

Calendar bounds we cannot compress:
- **P-19**: 30 days of cross-vendor cert burn-in regardless of code speed
- **P-21**: requires physical access to a 2-GPU box for real test
- **P-23**: SMT encoding design effort is genuinely sequential (~1 week)

---

## §11. Concurrency Plan

Current fleet capacity:
- **2 Codex** at a time (rate-limit pool — careful pacing)
- **1 Gemini Pro** at a time (5-account rotation, auto-pick least-used; 3.4 just landed and tested working)
- **5-10 Kimi** for single-crate parallel work (near-unlimited)
- **Me** for cross-cutting decisions, peer-Codex coordination, and items I've owned end-to-end
- **Advisory swarm** (4× copilot-5-mini + copilot-4-1) auto-fires on every writer commit

Per-pillar staffing recommendation:
- **Pillar 1**: kimi swarm (5 concurrent on validation/conform fixes) + me (correctness items in core) + codex on Pillar 6 in parallel
- **Pillar 2**: 1 codex on optimizer rewrite chain (PERF-001..007 + PERF-016..017) + 1 codex on async + buffer pool sharding (PERF-030..033) + kimi swarm on the rest
- **Pillar 3**: codex on MOD-004/005/006 (refactor) + me on small items + kimi on visibility sweep
- **Pillar 4**: codex on ARCH-001/002/012/017/021 (mechanical refactors) + me/kimi on rest
- **Pillar 5**: me on quick deletes; codex on ORPH-002 demo rewire
- **Pillar 6**: codex (one focused agent) — must finish before we trust any conform test
- **Pillar 7**: kimi swarm on the LAW 7 split sweep (§9.1) — ~22 files split in parallel; me + codex on net-new findings

---

## §12. Universal Quality Gates

Every closure must satisfy ALL of these (in addition to the per-item gates above):

1. `cargo check --workspace --all-features` clean
2. `cargo clippy --workspace --all-features -- -D warnings` clean (after Pillar 4 ARCH-021 lands; otherwise tolerate documented blanket allows)
3. The relevant test PASSES with the specific command listed in the per-item gate
4. For perf items: a Criterion bench + committed baseline; CI fails if new run is >5% slower
5. For correctness items: a regression test that would fail with the OLD code and PASSES with the NEW
6. For semantics-preserving optimizer passes: `prop_eval_preserved_under_pass` PASSES over ≥100 random programs
7. For backend changes: conform suite green on the wgpu backend (after Pillar 6 unblocks)
8. **No `unwrap()` / `expect()` without "Fix:" prefix in non-test code added by the change**
9. **No new file >500 LOC introduced by the change** — if a file would exceed, split first

---

## §13. Total Time Estimate

| Pillar | Items | Engineering hours | Wall-clock with fleet |
|---|---|---|---|
| 1 — Correctness | 18 | 12 | ~5 h |
| 2 — Performance (incl. roadmap P-1..P-34) | 90 | 120 | ~28-32 h |
| 3 — Modularity | 13 | 14 | ~13 h |
| 4 — Organization | 19 | 50 | ~25-30 h |
| 5 — Dead code | 6 | 6 | ~3 h |
| 6 — Conform test infra (BLOCKER) | 1 | 4 | ~4 h |
| **7 — Deep scan additions (NEW)** | **24** | **50** | **~15 h** |
| **Total** | **171** | **~256** | **~95 h wall-clock** |

Add P-19 cross-vendor burn-in: **+30 days calendar** running concurrently with everything else.

**Honest read: across 8-12 focused work sessions of 4-8 hours, vyre lands at release as defined in §1, except for the cross-vendor determinism CLAIM which requires the 30-day burn-in even after code lands.** With the rate observed during the 2026-04-18 push (~15-45 min per substantive item end-to-end), this is consistent.

---

## §14. Sequencing Recommendation

In order of priority:

1. **Land Pillar 6 first.** Without it, every conform-side change is unverifiable.
2. **Land Pillar 1 🔴 items** — REF-001, REF-002, GPU-002, GPU-003, GPU-004 — in parallel with Pillar 6 (different crates, no scope conflict).
3. **Confirm in-flight Codex agents** — collect O2/O4/P-11/P-12/P-16/P-17/P-18 results, redispatch failures.
4. **Drain Pillar 2 in parallel waves** — biggest fleet utilisation here, biggest user-visible win. PERF-001..007 (one codex) + PERF-030..033 (one codex) + remaining PERF-* (kimi swarm) + roadmap items not yet owned.
5. **Drain Pillar 7 §9.1 LAW 7 splits in parallel** with Pillar 2 — kimi swarm, ~22 files.
6. **Drain Pillars 3-5 in background** while Pillar 2 is the active focus. Most are mechanical.
7. **Start P-19 cross-vendor burn-in the day P-19 code lands.** 30 days run in parallel with everything else.
8. **P-23 SMT proofs last** — needs the encoding-design serial-thought effort that no other item can absorb.

---

## §15. Things deliberately not in scope

- **Tool promotion decisions** (e.g. should `vyre-wgpu` become a separately-published crate vs internal). Product call, not engineering.
- **Cross-vendor CI infrastructure procurement** — physical hardware purchase / cloud setup is operational, not engineering.
- **Documentation rewrite** — current docs are adequate for the intended audience; rewrite is a v0.7 effort.
- **Frontend integrations** (warpscan, surgec, etc.) — those are downstream consumers; vyre's contract to them is the `VyreBackend` trait + the conform suite. Their code lives in their own audits.
- **The `secjit` move noted in `layering_violations.md`** — outside vyre's tree.
- **`vyre-macros/Cargo.toml` proc-macro packaging polish** — covered as NEW-MOD-001 but the broader proc-macro publishing surface (versioning, MSRV, etc.) is a v1.0 release effort.

---

*End of plan. 171 items, ~256 engineering hours, ~95 hours wall-clock with full fleet, plus 30-day calendar burn-in for cross-vendor determinism.*
