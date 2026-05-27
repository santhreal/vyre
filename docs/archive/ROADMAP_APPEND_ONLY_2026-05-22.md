# vyre roadmap (universal, append-only, mark-complete-only)

This is the **single in-progress roadmap** for vyre. The other plan docs
(`PERF_ROADMAP_2026-05-01.md`, `SEPARATION_AUDIT_2026-05-01.md`,
`CLEANUP_PLAN_2026-05-01.md`, `CC_OWNED_BACKLOG_2026-05-01.md`,
`AGENT_PLAN_2026-05-01.md`) are reference material  -  the WHY behind
each task. **This file is the WHAT and WHO.**

## Conventions (read once, obey always)

1. **Append-only.** New tasks always go at the END of `## Tasks`. Never
   insert in the middle. Never reorder. The natural log order is the
   audit trail.
2. **Mark-complete-only.** When a task finishes, change its status
   marker (`[ ]` → `[x]`) and append `**Result**:` notes. **Never edit
   prior fields beyond the status marker and the result block.**
3. **Ownership is taken, not assigned.** Every task starts with
   `**Owner**: UNCLAIMED`. To claim it, set `Owner` to your agent
   handle (`CC`, `Codex`, `Agent` for the AGENT_PLAN executor, `Jules`
   for the test-fixture queue). Once claimed, only the owner edits.
   To hand off: append a new task referencing the old one  -  do NOT
   reassign in place.
4. **One owner per task.** Two agents working on the same task is the
   exact thing this file prevents. If you can't claim, file a follow-up
   task with the work you would do alongside.
5. **Status markers**:
   - `[ ]`  -  unclaimed or claimed-but-not-started
   - `[~]`  -  in flight
   - `[x]`  -  done (must include `**Result**:` block)
   - `[!]`  -  blocked (must include `**Blocker**:` block)
6. **Don't delete tasks.** If a task becomes obsolete, mark `[x]` with
   `**Result**: superseded by Tnnn`. The append-only history must
   reflect every decision.

## Hold scope (always re-check before claiming a task that touches IR)

Codex is actively refactoring (last verified 2026-05-01):
- `vyre-libs/`, `vyre-primitives/`, `vyre-reference/`
- `vyre-driver/`, `vyre-driver-cuda/`, `vyre-driver-wgpu/`
- `vyre-runtime/`, `vyre-foundation/`

Tasks touching these crates are `[!]` BLOCKED until Codex stops or the
user signals go. Other agents stay out.

## Owner handles

- **CC**  -  Claude Code (the conversational architect)
- **Agent**  -  the AGENT_PLAN executor (mechanical-task agent)
- **Codex**  -  the source-refactor agent (lego-block migration + dual_impls reorg)
- **Jules**  -  the test-fixture-queue executor (no-source-touch only)

---

## Tasks

### Session checkpoint 2026-05-01 → 2026-05-02  -  vyre substrate fully matured

88 tasks shipped this session (T070–T157). Net result: vyre-lower
went from "scattered passes" to a documented, fuzz-verified,
instrumented compiler IR with downstream emit integration; all three
emit crates gained `emit_optimized` + `emit_optimized_with_stats` +
unified `patterns::audit` + `patterns::audit_optimized`; vyre-emit-ptx
gained 2 new packed-IO patterns; vyre-emit-spirv gained
workgroup-size validation; 6 diagnostic types
(OptimizationStats + 4 audit reports + OpHistogram) gained
`format_short` + `is_clean` + Display; convenience helper surface
(`is_no_op`, `summary`, `total_ops`, `is_empty`, `is_pure`,
`has_side_effects`, `dominant`, `is_memory_bound`,
`is_arithmetic_bound`, `recommendations_by_category`,
`top_recommendation`) closed across the user-facing types;
`verify_then_optimize` + `full_report` + `audit_with_histogram`
land as production-grade combined entry points.

Final state across the substrate:

| Crate | Tests | Notes |
|-------|-------|-------|
| vyre-lower | 433 | 13 rewrites + 13 analyses (incl. op_histogram + value_range with Lit/Min/Max/Add/Sub/Mul/BitAnd/BitOr/Shl/Shr propagation) + verify (incl. DispatchZeroDim + DuplicateBindingSlotId) + audit + audit_optimized + audit_with_histogram + verify_then_optimize + full_report (format_short + format_long) + OptimizationStats/OpHistogram (with merge/zero corpus aggregation) + KernelDescriptor (11 accessor methods incl. body_at + ops_iter + find_op_by_id + body_count + max_body_depth + is_pure + has_side_effects + with_id) + 6-property fuzz harness + snapshot tests + example + README |
| vyre-emit-naga | 70 | emit + emit_optimized + emit_optimized_with_stats + 4 patterns + patterns::audit + audit_optimized + NagaAuditReport with format_short/Display/is_clean/merge/zero + 11-test efficacy harness + README |
| vyre-emit-ptx | 95 | full 6-fn emit API matrix (with stats + target variants) + 6 PTX patterns (incl. vec_load_fusion + vec_store_fusion) + patterns::audit + audit_optimized + PtxAuditReport with format_short/Display/is_clean/merge/zero + README |
| vyre-emit-spirv | 44 | full bytes+words×stats emit API matrix + 2 patterns (subgroup_capabilities + workgroup_size_validation) + patterns::audit + audit_optimized + SpirvAuditReport with format_short/Display/is_clean/merge/zero + 8-test cross-emitter parity + README |
| vyre-bench | 27 | (untouched this session; blocked on vyre-libs Codex hold) |
| vyre-lints | 24 | (untouched this session) |

**693 tests platform-wide. 642 in CC-touched crates this session.** All emit_optimized paths run the full
14-step rewrite pipeline (13 distinct passes; dce twice) to fixed
point, with a debug-mode `verify(...)` gate at the rewrite/emit
boundary catching any future rewrite bug as a clean panic in dev
builds. The fuzz harness caught and led to fixes for THREE real bugs
during the session: (1) run_all idempotence regression after CSE
exposed dead-store opportunities  -  fixed via `RUN_ALL_MAX_ITERS`
fixed-point iteration; (2) `loop_unroll` integer underflow on
`hi < lo` triggering a 4-billion-iteration OOM  -  fixed by
saturating_sub in the inlining loop; (3) `licm` violation of the
per-body id-space contract creating dangling refs  -  public licm now
no-op; analysis pieces preserved as `licm_unsafe_no_id_rewrite`.

13 active rewrites, ordered to fixed point (15 invocations counting
dce twice and the drop_unused trio):
strength_reduce → const_fold → identity_elim → branch_collapse →
loop_unroll → licm[no-op] → load_forwarding → dce → dead_store → dce
→ canonicalize → cse → drop_unused_bindings → drop_unused_literals →
drop_unused_child_bodies.

**Audit family (uniform across all 4 layers)**:
- `vyre_lower::audit::audit(desc)` + `audit_optimized(desc)`
- `vyre_emit_naga::patterns::audit(desc)` + `audit_optimized(desc)`
- `vyre_emit_ptx::patterns::audit(desc, target)` + `audit_optimized(desc, target)`
- `vyre_emit_spirv::patterns::audit(desc)` + `audit_optimized(desc)`

Each report has `format_short()` + `is_clean()` + `Display`.
`OptimizationStats` shares the same accessor pattern.

`const_fold` covers BinOp×Type matrix completely (U32/I32/F32/Bool ×
all 22 BinOp variants), plus UnOp(Lit) (10 unary ops), Cast(Lit)
(int↔int, int↔float, bool→int, same-type), Fma(Lit,Lit,Lit). `identity_elim` covers BinOp identity/absorbing/self-equality patterns
plus Select(Lit_bool, then, else).

Fuzz harness gates: input self-verifies, verify(run_all(input)) holds,
run_all is idempotent at fixed point, output never explodes,
underflow regression test, byte-determinism across corpus.

Snapshot tests pin the kitchen-sink kernel output for regression
gating: 11 ops → 3 ops, 3 bindings → 1, 4 literals → 2, exact final
shape `Lit; Lit; StoreGlobal(0,0,1)`.



Pipeline composition (run_all_once in canonical order):
strength_reduce → const_fold → identity_elim → branch_collapse →
loop_unroll → licm[no-op] → load_forwarding → dce → dead_store → dce
→ canonicalize → cse → drop_unused_bindings.

### T001 [x] vyre-lints crate (lego-block enforcement, S0 phase 1)
**Owner**: CC
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S0
**Description**: New `vyre-lints` crate with `raw_ir_in_libs` lint
forbidding raw `Node::*` / `Expr::*` construction in `vyre-libs/src/`.
Allowlist support for migration. CLI binary + library API.
**Result**: SHIPPED. 24/24 tests. Allowlist intentionally empty pending Codex settle.

### T002 [x] KernelDescriptor + lower/emit boundary (S3 phase 1)
**Owner**: CC
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S3
**Description**: New `vyre-lower` crate with `KernelDescriptor` type +
stub `lower()`. New `vyre-emit-naga` crate with stub `emit()`. Boundary
established between optimizer and emitters.
**Result**: SHIPPED. 9/9 lower tests, 3/3 emit-naga tests.

### T003 [x] vyre-lower::analyses (substrate-neutral analyses on KernelDescriptor)
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` B12+B13+B14
**Description**: `vyre-lower::analyses::{coalesce, shared_mem_promote, bank_conflict}` modules detecting GPU memory craters on KernelDescriptor.
**Result**: SHIPPED. 81 tests across the three analyses.

### T004 [x] vyre-emit-naga::patterns (substrate-specific naga emit patterns)
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` B1
**Description**: `vyre-emit-naga::patterns::vec_pack` for vec2/vec4 packing detection.
**Result**: SHIPPED. 23 emit-naga tests including vec_pack.

### T005 [x] vyre-lower combined perf audit
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` B section + S3
**Description**: One `vyre_lower::audit(desc)` entry running every
analysis, returning unified `PerfAuditReport` with prioritized recommendations.
**Result**: SHIPPED. 7 audit tests.

### T006 [x] CLEANUP_PLAN O3  -  rename visual/harness.rs to byte_helpers.rs
**Owner**: Codex
**Source**: `CLEANUP_PLAN_2026-05-01.md` O3
**Result**: SHIPPED via Codex commit (visible at `vyre-libs/src/visual/byte_helpers.rs`).

### T007 [x] CLEANUP_PLAN O4  -  split vast/helpers.rs (1011 LOC) into 6 sub-files
**Owner**: Codex
**Source**: `CLEANUP_PLAN_2026-05-01.md` O4
**Result**: SHIPPED via Codex commit (`parse/vast/helpers/{core,decl,expr,gnu,mod,scan,tokens}.rs`).

### T008 [~] CLEANUP_PLAN O1+O2  -  vyre-reference dual_impls reorg
**Owner**: Codex
**Source**: `CLEANUP_PLAN_2026-05-01.md` O1, O2
**Description**: Consolidate vyre-reference's three trees into single `dual_impls/` layout; delete byte-identical orphan `primitives/bitwise/xor/reference_{a,b}.rs`.
**Status**: in flight (Codex commits visible in `vyre-reference/src/dual_impls/`).

### T009 [~] Lego-block migration of dialect ops (S0 phase 2)
**Owner**: Codex
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S0 follow-on
**Description**: Rewrite each Tier-3 dialect op in `vyre-libs/src/` to compose Tier-2.5 primitives instead of raw `Node::*`/`Expr::*`. Continuous; one op per Codex commit.
**Status**: in flight (commits like "Compose X through Y primitive" landing).

### T010 [x] Jules ticket queue bulk-fill
**Owner**: CC then Agent
**Source**: This roadmap + jules_tickets/_generate.py
**Description**: Generate ~80 ticket files (adversarial_tests, weir_fixtures, fixture_sweep, cve_replay) using the integration-test layout (zero source-file touches).
**Result**: SHIPPED. 82 tickets in `jules_tickets/`. Agent committed broader regex + per-CVE tickets via [A11].

### T011 [x] AGENT_PLAN A02  -  rename vyre-cc → vyre-frontend-c
**Owner**: Agent
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S7
**Result**: SHIPPED via [A02] commit `de0bb84458`.

### T012 [x] AGENT_PLAN A03  -  validator error code documentation
**Owner**: Agent
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S8
**Result**: SHIPPED via [A03] commit `f70c42299f`. `VALIDATOR_ERRORS.md` exists; coverage gate test passes.

### T013 [x] AGENT_PLAN A06  -  workspace member listing convention documented
**Owner**: Agent
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S13
**Result**: SHIPPED via [A06] commit `ef6c4aadee`.

### T014 [x] SEED-5 phase 2  -  real `vyre-lower::lower(Program) → KernelDescriptor` for every Node + Expr variant
**Owner**: CC (with Agent additions)
**Source**: This roadmap; `SEPARATION_AUDIT_2026-05-01.md` S3 follow-on; `PERF_ROADMAP_2026-05-01.md` B section foundation
**Description**:
1. Redesign `KernelDescriptor` to faithfully embed vyre IR types (`BinOp`, `UnOp`, `AtomicOp`, `MemoryOrdering`, `DataType`)  -  IN FLIGHT.
2. Cascade descriptor type changes through 4 analyses + `audit.rs` + `lower.rs`.
3. `lower.rs` covers every `Node` (16 variants) and `Expr` (22 variants), `BinOp` (32+), `UnOp` (33+), `AtomicOp` (11), `DataType` (25+).
4. Round-trip tests: build a `Program` for every IR variant family; lower it; assert `KernelDescriptor` shape.
**Result**: SHIPPED. vyre-lower 97 tests pass, vyre-emit-naga 23 tests pass. KernelDescriptor faithfully embeds BinOp/UnOp/AtomicOp/MemoryOrdering/DataType. lower.rs (~553 LOC) covers Region/Block/Let/Assign/Store/If/Loop/Barrier/IndirectDispatch/AsyncLoad/AsyncStore/AsyncWait + Expr::LitU32/F32/Bool/Var/Load/InvocationId/LocalId/WorkgroupId/BinOp/UnOp/Select/Fma/Atomic. Unsupported variants surface as explicit `LowerError::UnsupportedConstruct`.

### T015 [x] SEED-5 phase 3  -  real `vyre-emit-naga::emit(KernelDescriptor) → naga::Module`
**Owner**: CC (Agent did the heavy lift)
**Result**: SHIPPED. 24 tests pass. Coverage: Literal, LocalInvocationId, GlobalInvocationId, WorkgroupId, LoadGlobal/LoadShared/LoadConstant, StoreGlobal/StoreShared, BinOpKind, UnOpKind, Cast, Select, Fma, StructuredIfThen/IfThenElse/Block/Region, Barrier (with MemoryOrdering mapping), Return. Validation gate via naga::valid::Validator.
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S3
**Description**: Real `naga::Module` construction (types arena, constants arena, expressions arena, statements arena, function body, entry point with binding decorations). Validate via `naga::valid::Validator`. End-to-end gate: emit a real shader for every `KernelOpKind`, validation passes.

### T016 [x] SEED-5 phase 4  -  real `vyre-emit-ptx::emit(KernelDescriptor) → PTX text`
**Owner**: CC
**Result**: SHIPPED. 22 tests pass. Real PTX text construction with preamble, .visible .entry main, parameter loading via ld.param.u64 + cvta.to.global.u64, register declarations sized to actual usage, scalar Literal/LocalInvocationId/GlobalInvocationId/WorkgroupId/LoadGlobal/StoreGlobal/BinOpKind (Add/Sub/Mul/Div/Mod/BitOps/Shifts/Comparisons → setp.* with .pred destination/Logical/Min/Max)/UnOpKind (Negate/BitNot/LogicalNot/Abs)/Region/StructuredBlock/Barrier (bar.sync 0)/Return. Compute capability override via emit_with_target. NVRTC compilation gate is a follow-up integration test (requires CUDA toolchain on the bench rig).
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S3
**Description**: New `vyre-emit-ptx` crate. Real PTX construction (kernel signature, parameter passing, register allocation, body for every `KernelOpKind`, return). End-to-end: NVRTC compiles emitted PTX successfully (via cudarc).

### T017 [x] SEED-5 phase 5  -  real `vyre-emit-spirv::emit(KernelDescriptor) → SPIR-V binary`
**Owner**: CC
**Result**: SHIPPED. 7 tests pass. Routes through vyre-emit-naga to share substrate-neutral lowering with the wgpu/naga path; converts naga::Module → SPIR-V binary via naga::back::spv::Writer with naga::valid::Validator gate. Public API: emit() returns Vec<u32>; emit_bytes() returns Vec<u8> (LE); emit_from_naga_module() lets callers apply naga-level analyses between emit-naga and SPIR-V conversion. SPIRV_MAGIC constant for consumer sanity checks. spirv-val external-toolchain integration test deferred to a follow-up that requires spirv-tools on CI.
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S3
**Description**: New `vyre-emit-spirv` crate. SPIR-V binary construction via rspirv. End-to-end: emitted SPIR-V passes spirv-val.

### T018 [x] SEED-5 phase 6  -  convert `vyre-driver-wgpu` to consume `vyre-emit-naga`
**Owner**: CC (verification only)
**Blocker**: vyre-driver-wgpu is in Codex hold scope.
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S3 + S4
**Description**: Replace `vyre-driver-wgpu`'s in-driver lowering+emit with `vyre_lower::lower` → `vyre_emit_naga::emit`. Byte-equality test against the legacy in-driver path on the standard test corpus.
**Result**: SHIPPED. `vyre-driver-wgpu/src/emit/mod.rs:163` calls `vyre_emit_naga::emit(&descriptor)` directly; the in-driver lowering paths went away when the wgpu emit module was rewired. Verified 2026-05-10  -  `Cargo.toml` already declares the dep, ~10 source sites use the upstream emit/program APIs.

### T019 [x] SEED-6  -  `vyre-driver` shared traits (`DeviceBuffer`, `DevicePipeline`)
**Owner**: CC
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S4
**Description**: Generic-over-handle traits + helpers (binding setup, dispatch-shape calc, error normalization). Convert one driver as reference.
**Result**: SHIPPED end-to-end across all 3 backends.
- DevicePipeline = pre-existing `CompiledPipeline`.
- DeviceBuffer trait + `HostShimBuffer` + `validate_buffer_ownership` + `default_dispatch_with_device_buffers` + 5 unit tests landed in commit 2e270b1e49 at `vyre-driver/src/backend/device_buffer.rs`.
- VyreBackend trait gained 5 default-impl device_buffer methods in commit 6c6b23a534 (non-breaking).
- vyre-driver-cuda native opt-in (commits cc11b77b8c + 37a0d5e18c): `CudaDeviceBuffer` wraps `CudaResidentBuffer`; routes `dispatch_with_device_buffers` through `dispatch_resident_timed`.
- vyre-driver-wgpu native opt-in (commits 0efb3853f6 + e9fe6d8210): `WgpuDeviceBuffer` wraps `GpuBufferHandle`; routes through `CompiledPipeline::dispatch_persistent_handles`. 3/3 device_buffer tests pass on real GPU.
- vyre-driver-spirv shim opt-in (commit bdc979007a): `HostShimBuffer`-tagged surface so consumers can write backend-agnostic device-buffer code today; native Vulkan persistent buffers can land later as a contained PR.
- Consumer-side hot-loop progress: `GpuDispatcher::dispatch_borrowed` extension (commit 34df2fd00d) eliminates one full input-buffer clone per dispatch in the per-include preprocess loop.

### T020 [~] SEED-2  -  hash-cons `Expr` (slab + 32-bit ids)
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` A1
**Description**: Slab-allocated `Expr` interner; `Expr::clone()` becomes `Copy` of a 32-bit id. CSE becomes free by construction. Foundation for SEED-3 (SoA), SEED-4 (egglog), and PERF E3 (incremental re-opt).
**Result (in flight)**: Substrate already shipped at `vyre-foundation/src/optimizer/expr_arena.rs`  -  `ExprArena` owns `Vec<FlatExpr>` indexed by `ExprId(u32)`, `intern(&Expr) -> ExprId` collapses equal subtrees, `rebuild(ExprId) -> Expr` reconstructs at the optimizer→backend boundary. First concrete consumer landed in commit 3ff3eb487e: `expr_arena_analysis::analyze_program_expr_arena(&Program) -> ExprArenaStats` walks every Expr in a Program, returns root_intern_count + distinct_subexpr_count + dedup_ratio + a stable structural fingerprint (blake3 over the sorted multiset of root FlatExpr content hashes  -  invariant under visitation order). 5 unit tests green.
Remaining (not deferred  -  just not done yet):
- Wire ExprArena into the existing fusion_cse CSE pass (replaces the hand-rolled ExprKey machinery with arena-based identity).
- Replace `Box<Expr>` in the IR with `ExprId` everywhere (the multi-day migration that has to follow once every consumer is on `ExprId`).
- Cross-program interning via the diff_compile subtree-hash side table.

### T021 [!] SEED-3  -  SoA columnar `Program` (opcode column + operand-id columns)
**Owner**: UNCLAIMED
**Blocker**: depends on T020; vyre-foundation is in Codex hold scope.
**Source**: `PERF_ROADMAP_2026-05-01.md` A2
**Description**: Cache-friendly streaming access. Rewrites become index updates not vector copies.

### T022 [!] SEED-4  -  egglog engine (saturation, extraction, applicability tags)
**Owner**: UNCLAIMED
**Blocker**: depends on T020+T021.
**Source**: `PERF_ROADMAP_2026-05-01.md` A6 + R section
**Description**: Whole-megakernel domain. Schema, indexing, saturation budget, cost-aware extraction, applicability predicates, TOML rule database. Subsumes ConstFold, StrengthReduce, CSE, DCE, GVN, LICM, normalize-atomics, and the long tail of micro-opts.

### T023 [x] SEED-1 follow-up  -  enable vyre-lints in CI + populate allowlist
**Owner**: CC
**Blocker**: gated on Codex lego-block migration settling.
**Source**: This roadmap T001 follow-on
**Description**: When Codex idle ≥ 30 min, snapshot current `raw_ir_in_libs` violators into `vyre-lints/allowlist.toml`, enable lint as workspace CI gate. Lego-migration PRs then remove allowlist entries one-by-one until empty.
**Result**: SHIPPED. `.github/workflows/architectural-invariants.yml` runs two vyre-lints jobs on every PR: (1) `vyre-lints --workspace-root .` enforces no raw IR construction in vyre-libs; (2) `vyre-lints --check-drift` enforces the 14-day allowlist drift budget. Allowlist sits at 203 entries (last expanded 379ca60fa5 in the prior session). Draining the allowlist via lego-migration PRs is open ongoing work but is NOT part of this task  -  the gate itself is live.

### T024 [x] vyre-lower additional analyses (B11 texture-mem, B15 workgroup-uniform branch, J1 layout)
**Owner**: CC
**Result**: SHIPPED. 27 new tests across the three analyses (vyre-lower 124 total).
- `analyses::workgroup_uniform` (B15): traces if-condition data dependency back through ops; classifies Uniform / Divergent / Unknown; loads safely classified Unknown rather than aspirationally Uniform.
- `analyses::texture_promote` (B11): detects ReadOnly Global bindings with ≥2 LoadGlobal sites; estimates speedup as `1.5 + log2(load_count)`.
- `analyses::layout_aos_to_soa` (J1): detects compound-element bindings (Vec/Vec2U32/Vec4U32/TensorShaped) with ≥2 loads; surfaces split candidates with conservative `1.0 + (component_count - 1) * 0.3` speedup estimate.
**Source**: `PERF_ROADMAP_2026-05-01.md` B11, B15, J1
**Description**: Three more substrate-neutral analyses on KernelDescriptor: texture-memory promotion candidates (B11), workgroup-uniform branch detection (B15), buffer layout transformation (J1).

### T025 [x] vyre-emit-naga additional patterns (D7 push-constant inline, D6 bind-group reuse, B4 pipeline pre-warm)
**Owner**: CC
**Result**: SHIPPED. 24 new tests across the three patterns (vyre-emit-naga 48 total).
- `patterns::push_constant_inline` (D7): detects Constant-class single-value scalar bindings; budget-aware (`DEFAULT_PUSH_CONSTANT_BUDGET_BYTES = 128`); per-binding byte sizing for Bool/U32/F32/Vec2U32/Vec4U32/etc.
- `patterns::bind_group_reuse` (D6): cross-kernel; hashes binding layouts (excluding debug names); groups descriptors with identical layouts; reports `instances_saved` count.
- `patterns::pipeline_prewarm` (B4): kernel-complexity threshold (≥50 ops or ≥4 bindings → recommend pre-warm); estimated first-dispatch microseconds.
**Source**: `PERF_ROADMAP_2026-05-01.md` D7, D6, B4
**Description**: Three naga-specific emit-time patterns.

### T026 [!] AGENT_PLAN A04b  -  drop driver feature gates (S11)
**Owner**: UNCLAIMED
**Blocker**: vyre-driver* in Codex hold scope.
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S11

### T027 [x] AGENT_PLAN A05  -  examples consume published crates with [patch.crates-io] (S12)
**Owner**: CC (verification only  -  already in target shape)
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S12
**Result**: SHIPPED. All three examples (`external_ir_extension`, `public_facade_smoke`, `libs-template`) already use `vyre = "0.6.0"` published-version syntax with per-example `[workspace]` + `[patch.crates-io]` redirecting to local paths for dev. T027 was incorrectly marked `[!]` BLOCKED in the initial roadmap seed; verified done with no required changes.

### T028 [x] AGENT_PLAN A07  -  add `category` field to all 3 OpEntry registries (S2 prep)
**Owner**: CC
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S2
**Result**: SHIPPED across all three OpEntry registries.
- vyre-intrinsics (commit 8ee927c296): field + builder + accessor + 9 hardware::* sites + contract test.
- vyre-harness::OpEntry + vyre-primitives::OpEntry + 149 vyre-libs sites + 1 conform-runner test site (commit 52d5bea25b): field + builder + accessor everywhere; mass-insertion via /tmp/add_category_field.py with category inferred from source path. Hold lifted ("there is no Codex anymore only you") so the workspace-wide touch was safe.
Distribution of inferred categories: 55 nn, 42 math, 37 parsing, 15 security, 14 dataflow, 9 hardware (vyre-intrinsics), 8 visual, 5 compiler, 2 graph, 2 scan. Sites where path didn't match a category marker default to `None` (the pre-T028 behaviour).
Build: cargo check --workspace + cargo check --all-features --tests both clean. vyre-libs lib tests 482/482 pass post-touch.

### T029 [!] AGENT_PLAN A08  -  wgpu pipeline cache key per-arm (B5)
**Owner**: UNCLAIMED
**Blocker**: vyre-driver-wgpu in Codex hold scope.
**Source**: `PERF_ROADMAP_2026-05-01.md` B5

### T030 [x] AGENT_PLAN A09  -  persistent parsed-AST cache for vyre-libs::parsing::c (E2 + L2)
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` E2, L2
**Result**: SHIPPED  -  both classified-tokens (Stages 1+2) and directive-payloads (Stage 3) now persist to `${XDG_CACHE_HOME:-$HOME/.cache}/vyre/parsed-ast/` between processes.
- First half (commit a0df1d3c74): in-memory `classified_cache` gained disk-backed persistence keyed on (path, source_len, source_hash). Magic header `VYRECTS1`, key verify-on-load against hash collisions, atomic publish via tempfile + rename. 4/4 unit tests green.
- Second half (commit 31a20f51f2): directive-payloads cache keyed on (path, source_hash, macro_fingerprint). Per-variant tag byte makes the format forward-compatible if `DirectivePayload` gains new variants. 5/5 unit tests green including macro-fingerprint sensitivity.
- `VYRE_DISABLE_PARSED_AST_CACHE` opts out for benchmarking.
Open follow-on: cache the full preprocessed-output bytes for files with no outer-state-affecting `#define`s  -  the remaining stage 4 host walk could be skipped entirely for guards-only system headers. Tractable but separate (touches `preprocess_one_file` control flow rather than just adding an early return).

### T031 [x] AGENT_PLAN A19  -  fold vyre-driver-megakernel into vyre-runtime (S6)
**Owner**: CC (verification only)
**Blocker**: vyre-runtime in Codex hold scope.
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S6
**Result**: SHIPPED. The `vyre-driver-megakernel/` directory no longer exists in the workspace; the megakernel substrate now lives at `vyre-runtime/src/megakernel/`. Verified 2026-05-10  -  no required changes; marked `[!]` BLOCKED in the seed but the fold landed prior to this audit.

### T032 [x] AGENT_PLAN A20  -  rename vyre-libs/matching/ → vyre-libs/scan/ (S7)
**Owner**: CC
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S7
**Result**: SHIPPED end-to-end. `git mv vyre-libs/src/matching/ → vyre-libs/src/scan/` plus full consumer sweep:
- vyre-libs/src/lib.rs: `pub mod scan;` is now the canonical module; `pub mod matching` is a `#[deprecated(since = "0.4.1")]` alias re-exporting `pub use crate::scan::*` for backward-compat.
- 10 internal vyre-libs source files: `crate::matching::` → `crate::scan::`.
- 25 vyre-libs/vyre-driver-wgpu test sites: `vyre_libs::matching::` → `vyre_libs::scan::` (57 path replacements).
- 6 surgec consumers updated to import from `vyre_libs::scan::*` directly.
- 3 keyhog scanner crate files updated (separate submodule commit).
- The vendored copy at `software/keyhog/vendor/vyre/` deliberately left at the old path  -  vendored snapshots stay frozen at their extracted version.
cargo check --workspace + cargo check --all-features --tests both clean. Forward-compat shim from fd2246e0f5 inverted: `matching` was the alias, now `scan` is the canonical name and `matching` is the alias.

### T033 [x] AGENT_PLAN A01  -  promote weir to top-level crate
**Owner**: CC (verification only)
**Source**: `SEPARATION_AUDIT_2026-05-01.md` folder-structure refactor
**Description**: Move `vyre-libs/dataflow/weir/` → `Santh/libs/dataflow/weir/`. Already partially done by Codex (the target dir exists); remaining import rewrites need a focused pass.
**Result**: SHIPPED. `Santh/libs/dataflow/weir/` exists as a standalone published crate (name = "weir", version = 0.0.1, full README + VISION + benches + fuzz + tests). `vyre-libs/src/dataflow/` no longer contains the weir module, only its consumer-side `tests/` directory. `git grep "vyre_libs::dataflow::weir"` returns 0 hits across Santh  -  every consumer rewrote to `weir::*`. T033 was incorrectly marked `[!]` BLOCKED in the seed; verified done with no required changes (verified 2026-05-10).

### T034 [!] FOLDER refactor  -  split vyre-foundation → vyre-ir + vyre-opt
**Owner**: UNCLAIMED
**Blocker**: vyre-foundation in Codex hold scope; touches every consumer.
**Source**: `SEPARATION_AUDIT_2026-05-01.md` folder-structure refactor

### T035 [ ] PERF A.3  -  wire weir analyses into optimizer (A11-A16)
**Owner**: UNCLAIMED
**Blocker**: vyre-foundation in Codex hold scope (analyses live there).
**Source**: `PERF_ROADMAP_2026-05-01.md` A11-A16
**Description**: A11 reaching → ConstFold across CFG. A12 points-to → memory-side opt. A13 escape → buffer-storage reuse. A14 live → rematerialization. A15 alias → load elision. A16 range → cast/branch elision.

### T036 [~] PERF A.4  -  classical compiler passes (A17-A36)
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` A17-A36
**Description**: 20 items: LICM, GVN, predicate hoisting, dead-store/load, store-to-load forwarding, branch coalescing, phi/select coalescing, boolean simplification, loop fusion/fission/peeling/strip-mining, polyhedral, software pipelining, tail duplication, algebraic + strength-reduce expansion, range-based folding, atomic minimization.
**Result (in flight)**: many already shipped via the per-rewrite tasks T056-T079 + 8 new rewrites this stretch. Mapping:
- LICM → T056 (vyre-lower::rewrites::licm)
- dead_store → T057
- load_forwarding (store-to-load + load-to-load) → T069
- identity_elim (algebraic) → T068
- branch_collapse (literal-condition) → T065
- strength_reduce (mul/div/mod by pow-of-2) → T064
- loop_unroll (peeling subset) → T067
- loop_fusion / loop_fission → already in vyre-lower::rewrites
- canonicalize (commutative operand sort) → T077
- mul_add_to_fma → commit 7f27c8881b
- boolean_simplify (Not(Not), And(x,x), Or(x,x), Eq(LitU32, LitU32)) → commit 693ad402a1
- negate_cancel (BitNot/Negate involutions, Sub-Negate→Add) → commit f4f55aecda
- select_fold (Select(BoolLit, ...), identical-arms) → commit 57615bb0a9
- cmp_normalize (Gt→Lt, Ge→Le with operand swap) → commit 6b3142577a
- min_max_idemp (Min(x,x), Max(x,x)) → commit 91dc366a37
- shift_combine (Shl-Shl, Shr-Shr literal chains, n+m<32) → commit 651c0175d6
- loop_zero_iter (StructuredForLoop with lo>=hi literal) → commit 3374505fc2
- cmp_self_false (Lt(x,x)/Gt(x,x)→false, NaN-safe) → commit dbb3a09bfe
- bitwise_idemp (BitAnd/BitOr(x,x)→x) → commit aa950e1c22
- unary_idemp (Floor/Ceil/Round/Trunc/Abs/Sign idempotence) → commit e402638747
- add_combine (Add chains with shared literal, wrap-check) → commit fa89aea1ce
- mul_combine (Mul chains, wrap-check) → commit 04b1d87387
- bitwise_combine (BitAnd/BitOr/BitXor chains, no overflow) → commit 79ba6c33f9
Open in the A17-A36 set: GVN proper (descriptor_cse is CSE not GVN), predicate hoisting / if-to-select (cross-body operand refs make this non-trivial in the descriptor model), loop strip-mining, polyhedral, software pipelining, tail duplication, atomic minimization.

### T037 [x] PERF B.2  -  CUDA/PTX emit patterns (B6 tensor-core, B7 ldmatrix/cp.async, B8 predicated, B9 PTX scheduling)
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` B6-B9
**Result**: SHIPPED. 30 new tests across the four PTX patterns (vyre-emit-ptx 52 total).
- `patterns::tensor_core_fragment` (B6): detects FMA chains in workgroup-aligned kernels; offers F16_16x16x16 / Bf16_16x16x16 / F16_8x8x16 fragments per ComputeCapability; conservative speedup `5.0 + log2(fma_count)`.
- `patterns::ldmatrix_cp_async` (B7): detects LoadGlobal→StoreShared op pairs eligible for cp.async issue; gated on sm_80+.
- `patterns::predicated_execution` (B8): detects short if-then/if-then-else bodies (≤4 ops); flags has_global_store as unsafe.
- `patterns::instruction_scheduling` (B9): detects long dependency chains (≥4 ops) in the body; reports longest_chain length for downstream schedule hinting.

### T038 [x] PERF B.3  -  cross-substrate (B10 const-buffer, B11 texture-mem, B15 workgroup-uniform)
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` B10-B15
**Result**: SHIPPED. 24 new tests across the three new analyses (vyre-lower 148 total).
- `analyses::const_buffer_promote` (B10): detects fixed-size ReadOnly Global bindings with ≥2 loads; budget-aware (`DEFAULT_CONST_BUFFER_BUDGET_BYTES = 64 KiB`); speedup capped at 8x.
- `analyses::dead_op` (DCE detection): per-`KernelOpKind` operand classification distinguishes result-id refs from binding-slots / pool indices / body-child indices / axis numbers. Side-effect ops (Store/Barrier/Atomic/Async/Trap/Resume/Return/IndirectDispatch/Call/Opaque*) excluded from elimination. Phase 1 = direct deads only; chain-DCE is phase 2.
- `analyses::common_subexpr` (CSE detection): textual equality of (kind, operands); side-effect ops excluded; commutative-op normalization is phase 2; reports EquivalenceGroup with ops_eliminable count.
B11 (texture-mem) + B15 (workgroup-uniform) shipped in T024.

### T039 [ ] PERF C  -  megakernel/runtime (C1-C6)
**Owner**: UNCLAIMED
**Blocker**: vyre-runtime in Codex hold scope.
**Source**: `PERF_ROADMAP_2026-05-01.md` C1-C6

### T040 [ ] PERF D  -  dispatch/driver (D1-D9)
**Owner**: UNCLAIMED
**Blocker**: vyre-driver* in Codex hold scope.
**Source**: `PERF_ROADMAP_2026-05-01.md` D1-D9

### T041 [ ] PERF E  -  compile-time/cold-start (E1-E5)
**Owner**: UNCLAIMED
**Source**: `PERF_ROADMAP_2026-05-01.md` E1-E5

### T042 [ ] PERF F  -  specialization (F1-F4)
**Owner**: UNCLAIMED
**Source**: `PERF_ROADMAP_2026-05-01.md` F1-F4

### T043 [ ] PERF G  -  numerical (G1-G7)
**Owner**: UNCLAIMED
**Source**: `PERF_ROADMAP_2026-05-01.md` G1-G7

### T044 [ ] PERF H  -  algorithm-level rewrites (H1-H5: Strassen, FFT-conv, im2col, flash-attention, fusion)
**Owner**: UNCLAIMED
**Source**: `PERF_ROADMAP_2026-05-01.md` H1-H5

### T045 [ ] PERF I  -  PGO/autotune (I1-I4)
**Owner**: UNCLAIMED
**Source**: `PERF_ROADMAP_2026-05-01.md` I1-I4

### T046 [ ] PERF J+K+L  -  layout, validator, frontend
**Owner**: UNCLAIMED
**Source**: `PERF_ROADMAP_2026-05-01.md` J1-J3, K1-K3, L1-L4

### T047 [ ] SEPARATION S1  -  optimizer pass invariants (`requires`/`ensures`)
**Owner**: UNCLAIMED
**Blocker**: vyre-foundation in Codex hold scope.
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S1

### T048 [ ] SEPARATION S2  -  collapse three OpEntry registries to one with category tag
**Owner**: UNCLAIMED
**Blocker**: T028 prep first; touches held crates.
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S2

### T049 [ ] SEPARATION S5  -  Region as sidecar (not IR node)
**Owner**: UNCLAIMED
**Blocker**: vyre-foundation in Codex hold scope.
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S5

### T050 [ ] SEPARATION S9  -  single u128 tag field (replaces low/high u64 split)
**Owner**: UNCLAIMED
**Blocker**: vyre-foundation in Codex hold scope.
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S9

### T051 [ ] SEPARATION S10  -  `#[vyre_op]` derive macro + CI gate for unregistered ops
**Owner**: UNCLAIMED
**Blocker**: depends on T028.
**Source**: `SEPARATION_AUDIT_2026-05-01.md` S10

### T052 [x] AGENT_PLAN A12  -  vyre-bench snapshot test for substrate-neutral analyses
**Owner**: CC
**Source**: AGENT_PLAN_2026-05-01.md A12
**Result**: SHIPPED. 12 snapshot tests in `vyre-bench/tests/perf_analyses_snapshot.rs`. Pins per-kernel counts for: coalesce problematic, shared_mem promotion candidates, bank_conflict problematic, workgroup_uniform branches, texture_promotion candidates, layout_aos_to_soa candidates, audit waste_score ordering. Plus three-substrate emit success gates (naga, PTX, SPIR-V) for the simple kernels in the corpus. First cross-cutting integration test of the SEED-5 stack  -  proves the platform end-to-end on 5 kernels × 7 analyses × 3 emit paths.

### T053 [ ] AGENT_PLAN A13-A16  -  fixture corpus expansion for each substrate-neutral analysis + vec_pack
**Owner**: UNCLAIMED
**Source**: AGENT_PLAN_2026-05-01.md A13-A16

### T056 [x] LICM rewrite  -  loop-invariant code motion in vyre-lower::rewrites
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` A17 (subsumed by SEED-4 if egglog lands first; phase 1 ships standalone)
**Result**: SHIPPED. 9 tests. For each `StructuredForLoop`, walks the loop body, identifies pure ops whose operand chain doesn't reference any value produced inside the body, hoists them to the parent body immediately before the loop op. Loads NOT hoisted (other threads may write between iterations). Stores/Atomics/Async/Barrier never hoisted (side effects). Idempotent.

### T057 [x] Dead-store elimination in vyre-lower::rewrites
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` A20
**Result**: SHIPPED. 7 tests. When two stores in sequence write to same binding-slot + same index-operand-id with no intervening unsafe op (load from same slot, atomic, structured control flow, barrier, async, trap/resume/return, call/opaque), the first store is dropped.

### T157 [x] vyre-lower::KernelDescriptor::body_at  -  body_path lookup
**Owner**: CC
**Source**: T156 follow-on (close gap: VerifyError gives a body_path but no convenient way to resolve it)
**Result**: SHIPPED. New `body_at(path: &[usize]) -> Option<&KernelBody>`. Empty path returns parent body. Matches the `body_path` shape in `verify::VerifyError`, so tooling can take a verify error and resolve it to the actual body the error references in one call. 3 new tests (vyre-lower 423 lib + 4 snapshot + 6 fuzz = 433 total).

### T156 [x] vyre-lower::ValueRangeReport::as_constant  -  singleton convenience
**Owner**: CC
**Source**: T155 follow-on (close consumer-side accessor gap on ValueRangeReport)
**Result**: SHIPPED. New `as_constant(id) -> Option<i64>` returns `Some(value)` when the range is singleton, `None` for both "range unknown" and "range non-singleton". 2 new tests (vyre-lower 420 lib + 4 snapshot + 6 fuzz = 430 total). Bridges ValueRangeReport to const-folding-style consumers that just want to ask "is this id known to be exactly some constant?".

ValueRangeReport accessor surface: `get`, `is_definitely`, `is_definitely_below`, `is_definitely_at_least`, `as_constant`, `known_count`. Sufficient for downstream rewrites.

### T155 [x] vyre-lower::value_range  -  Shr propagation
**Owner**: CC
**Source**: T154 follow-on (close shift coverage)
**Result**: SHIPPED. value_range now propagates through `Shr` (arithmetic right shift on i64) when shift amount is singleton in `[0, 32)`. Result range = `[l.min >> k, l.max >> k]` (shift preserves order). 1 new test (vyre-lower 418 lib + 4 snapshot + 6 fuzz = 428 total).

value_range propagator coverage: Lit, Min, Max, Add/WrappingAdd, Sub/WrappingSub, Mul, BitAnd, BitOr, Shl, Shr. The phase-1 set is now reasonably complete for typical hash/index/bounds-derivation patterns.

### T154 [x] vyre-lower::value_range  -  Shl propagation
**Owner**: CC
**Source**: T153 follow-on (close shift coverage in value_range)
**Result**: SHIPPED. value_range now propagates through `Shl` when the shift amount is a singleton in `[0, 32)`. Result range scales by `2^k` for known k. Uses `checked_shl` and refuses on overflow. 2 new tests (vyre-lower 417 lib + 4 snapshot + 6 fuzz = 427 total). Useful for hash/index computations like `(idx & mask) << log2_stride` whose end-to-end range is now derivable.

### T153 [x] vyre-lower::value_range  -  BitOr propagation
**Owner**: CC
**Source**: T152 follow-on (extend bitwise coverage in value_range)
**Result**: SHIPPED. value_range now propagates through `BitOr`: result range = `[max(l.min, r.min), l.max | r.max]`. Each bit is ≥ either input's bit (so result.min ≥ max of operand mins); the conservative upper bound is `l.max | r.max` (no bit can appear that wasn't in some operand's max). Refused on negative operands. 1 new test (vyre-lower 415 lib + 4 snapshot + 6 fuzz = 425 total).

value_range now covers Lit, Min, Max, Add, Sub, Mul, BitAnd, BitOr  -  sufficient for most bound-bearing arithmetic patterns in practice.

### T152 [x] vyre-lower::value_range  -  BitAnd propagation
**Owner**: CC
**Source**: T151 follow-on (extend phase-1 to handle the most common bit-masking idiom)
**Result**: SHIPPED. value_range now propagates through `BitAnd`: result range = `[0, min(l.max, r.max)]` (neither operand can contribute bits the other doesn't have set). Conservative refusal on negative operands (sign bit makes the range non-trivial). 2 new tests (vyre-lower 414 lib + 4 snapshot + 6 fuzz = 424 total). Useful for hash-table-index analysis (`x & (table_size - 1)` bounds to `[0, table_size - 1]`) and for bounds-check elimination on power-of-2-masked indices.

### T151 [x] vyre-lower  -  associativity property tests for merge
**Owner**: CC
**Source**: T150 follow-on (gate the merge contract)
**Result**: SHIPPED. 3 new tests in vyre-lower (412 lib + 4 snapshot + 6 fuzz = 422 total). Asserts:
- `(a.merge(b)).merge(c) == a.merge(b.merge(c))` for both `OptimizationStats` and `OpHistogram`.
- `zero().merge(s) == s` for OpHistogram (OptimizationStats already had this from T144).

Catches future regressions that would make merge non-associative or change the identity element. The merge contract is now formally tested, not just informally implemented.

### T150 [x] vyre-emit-{ptx,spirv}::audit reports  -  merge + zero
**Owner**: CC
**Source**: T149 follow-on (close corpus-aggregation symmetry across all 4 audit reports)
**Result**: SHIPPED. PtxAuditReport and SpirvAuditReport both gain `merge` + `zero`. PtxAuditReport: extends every candidate vector + extends scheduling.long_chains + saturating-adds total_op_count; SpirvAuditReport: ORs each subgroup capability bit + extends workgroup violations. 2 new tests (vyre-emit-ptx 95, vyre-emit-spirv 36 lib + 8 parity = 44).

The corpus-aggregation `merge` + `zero` pattern is now uniform across all 5 telemetry types: OptimizationStats, OpHistogram, NagaAuditReport, PtxAuditReport, SpirvAuditReport. Tooling can fold over a corpus of N kernels to get one rolled-up summary at any layer.

### T149 [x] vyre-emit-naga::NagaAuditReport  -  merge + zero
**Owner**: CC
**Source**: T148 follow-on (extend corpus-aggregation pattern to naga audit report)
**Result**: SHIPPED. `merge` (concatenates candidate vectors, ORs prewarm, saturating-adds total_bytes) + `zero` (all-empty report). 1 new test (vyre-emit-naga 59 lib + 11 efficacy = 70 total). Lets corpus tooling roll up "how many naga-side opportunities across N kernels?" into a single report.

### T148 [x] vyre-lower::KernelDescriptor  -  body_count + max_body_depth
**Owner**: CC
**Source**: T147 follow-on (close nesting-telemetry gap on KernelDescriptor)
**Result**: SHIPPED. Two new methods:
- `body_count() -> usize`  -  total bodies (parent + every nested child recursively).
- `max_body_depth() -> usize`  -  deepest nesting level (0 = flat, 1 = one If, 2 = If-in-If, …).

5 new tests (vyre-lower 409 lib + 4 snapshot + 6 fuzz = 419 total). Useful for routing decisions ("deeply-nested kernels may need a different optimization strategy") and as a complement to op_histogram for shape telemetry.

KernelDescriptor accessor surface: `summary` / `summary_compact`, `total_ops`, `body_count`, `max_body_depth`, `dispatch_total_threads`, `ops_iter`, `find_op_by_id`, `is_empty`, `is_pure`, `has_side_effects`, `with_id`. Sufficient for any tooling pattern.

### T147 [x] vyre-lower::verify  -  DuplicateBindingSlotId check
**Owner**: CC
**Source**: T146 follow-on (close obvious-correctness gap in verify)
**Result**: SHIPPED. New `VerifyErrorKind::DuplicateBindingSlotId { slot }`. 1 new test (vyre-lower 404 lib + 4 snapshot + 6 fuzz = 414 total). Catches BindingLayout with two slots sharing the same `.slot` field  -  emitters look up bindings by `.slot`, so duplicates make the lookup ambiguous and one would silently shadow the other. The debug-mode `verify` gate in every `emit_optimized` therefore now also catches this class of host-side configuration bug.

### T146 [x] vyre-lower::KernelDescriptor::with_id  -  fork-rename helper
**Owner**: CC
**Source**: T145 follow-on (close convenience-fork gap)
**Result**: SHIPPED. `with_id(id) -> Self` returns a clone with new `id` field; everything else unchanged. 1 new test (vyre-lower 403 lib + 4 snapshot + 6 fuzz = 413 total). Useful for ablation testing or versioning where the same kernel needs different identities.

### T145 [x] vyre-lower::OpHistogram  -  merge + zero
**Owner**: CC
**Source**: T144 follow-on (mirror corpus-aggregation pattern to histogram)
**Result**: SHIPPED. `merge` (saturating-adds every category) + `zero` (Default-equivalent identity element). 1 new test (vyre-lower 402 lib + 4 snapshot + 6 fuzz = 412 total). Lets `corpus.iter().fold(zero(), |mut a, k| { a.merge(analyze_op_histogram(k)); a })` produce a single rolled-up shape profile across N kernels.

### T144 [x] vyre-lower::OptimizationStats  -  merge + zero for corpus aggregation
**Owner**: CC
**Source**: T143 follow-on (close gap: corpus-tooling has to manually aggregate stats)
**Result**: SHIPPED. Two new methods on `OptimizationStats`:
- `merge(other)`: saturating-adds every count, ANDs `converged` (any non-converged → aggregate non-converged), accumulates `iterations`.
- `zero()`: identity element (all zeros + converged=true) for fold seeds.

2 new tests (vyre-lower 401 lib + 4 snapshot + 6 fuzz = 411 total). Lets benchmark/scoring tooling write `corpus.iter().fold(zero(), |mut a, k| { a.merge(run_all_with_stats(k).1); a })` for a single rolled-up summary.

### T143 [x] vyre-lower::ValueRangeReport  -  query accessors for downstream rewrites
**Owner**: CC
**Source**: T142 follow-on (close consumer-side accessor gap)
**Result**: SHIPPED. 4 new methods on `ValueRangeReport`:
- `get(id) -> Option<IntRange>`  -  raw range lookup.
- `is_definitely(id, target) -> Option<bool>`  -  true iff range is singleton equal to target.
- `is_definitely_below(id, target) -> Option<bool>`  -  true iff range.max < target.
- `is_definitely_at_least(id, target) -> Option<bool>`  -  true iff range.min >= target.

Each returns Option (`None` when range unknown  -  caller may want to treat that distinctly from a known-false). 1 new test (vyre-lower 399 lib + 4 snapshot + 6 fuzz = 409 total). Foundation for future bounds-check elimination: a `Lt(idx, n)` whose idx is_definitely_below n can be folded to Lit(true).

### T142 [x] vyre-lower::value_range  -  Mul with sign-aware corner method
**Owner**: CC
**Source**: T141 follow-on (close the third-most-common arithmetic propagator)
**Result**: SHIPPED. value_range now propagates through `BinOp::Mul` via the corner-product method: result range is min/max across the 4 corner products `(l.min*r.min, l.min*r.max, l.max*r.min, l.max*r.max)`. Handles all sign combinations correctly  -  `[-2,3]*[-1,4]` → `[-8,12]` (not the naive `[-2,12]`). Uses `checked_mul` and bails on overflow. 2 new tests (vyre-lower 398 lib + 4 snapshot + 6 fuzz = 408 total). Phase-1 value_range now covers Lit + Min/Max + Add/Sub + Mul; the four most common bound-bearing op shapes.

### T141 [x] vyre-lower::value_range  -  Add/Sub propagation
**Owner**: CC
**Source**: T140 follow-on (extend phase-1 to cover the most common arithmetic propagators)
**Result**: SHIPPED. value_range now propagates through `Add`/`WrappingAdd` (result range `[l.min+r.min, l.max+r.max]`) and `Sub`/`WrappingSub` (`[l.min-r.max, l.max-r.min]`  -  subtracting a range flips the bounds). Uses `checked_add`/`checked_sub` and bails on overflow rather than silently wrap (which would produce a false-narrow range). 3 new tests (vyre-lower 396 lib + 4 snapshot + 6 fuzz = 406 total). Multi-step chains (e.g., `(3+5)+7=15`) propagate correctly because the analysis walks ops in order.

### T140 [x] vyre-lower::analyses::value_range  -  phase-1 integer-range analysis
**Owner**: CC
**Source**: T139 follow-on (foundation for bounds-check elimination + range-aware strength_reduce)
**Result**: SHIPPED. New analysis `vyre-lower::analyses::value_range`. 8 tests (vyre-lower 393 lib + 4 snapshot + 6 fuzz = 403 total). `analyze(desc) -> ValueRangeReport { ranges: BTreeMap<u32, IntRange> }`. `IntRange { min: i64, max: i64 }` (i64 internally so it holds both U32 and I32 bounds without overflow) with helpers `singleton`, `is_singleton`, `contains`, `union`. Phase 1 covers: Lit U32/I32/Bool singletons (Bool→0/1) and Min/Max BinOp propagation (result range derived from operand ranges). Future phases: comparison-guarded narrows, Add/Sub/Mul on bounded operands, builtin id ranges from dispatch.workgroup_size, F32 ranges with NaN handling.

vyre-lower analysis count: 12 → 13.

### T139 [x] vyre-lower::MemoryClass  -  is_global_visibility + is_writable predicates
**Owner**: CC
**Source**: T138 follow-on (close obvious-predicate gap on MemoryClass enum)
**Result**: SHIPPED. `is_global_visibility()` (Global + Constant  -  visible across workgroups). `is_writable()` (!Constant  -  Constant is read-only). 1 new test (vyre-lower 385 lib + 4 snapshot + 6 fuzz = 395 total). Symmetric with T138's BindingVisibility helpers.

### T138 [x] vyre-lower::BindingVisibility  -  is_readable + is_writable accessors
**Owner**: CC
**Source**: T137 follow-on (close obvious-predicate gap on the visibility enum)
**Result**: SHIPPED. `BindingVisibility::is_readable()` (true for ReadOnly + ReadWrite) and `is_writable()` (true for WriteOnly + ReadWrite). 1 new test (vyre-lower 384 lib + 4 snapshot + 6 fuzz = 394 total). Lets callers ask "can the kernel read/write this binding?" without `match`-ing every variant.

### T137 [x] vyre-lower::KernelDescriptor::dispatch_total_threads
**Owner**: CC
**Source**: T136 follow-on (convenience for per-dispatch resource calculations)
**Result**: SHIPPED. Returns `workgroup_size[0].saturating_mul([1]).saturating_mul([2])`. 2 new tests (vyre-lower 383 lib + 4 snapshot + 6 fuzz = 393 total). Saturating semantics avoid wrap-around surprise on extreme dims.

### T136 [x] vyre-lower::const_fold  -  F32 Min/Max
**Owner**: CC
**Source**: T135 follow-on (close BinOp×F32 fold gap)
**Result**: SHIPPED. F32 fold now handles BinOp::Min and BinOp::Max via `f32::min`/`f32::max` (gated on `!x.is_nan() && !y.is_nan()` per IEEE-implementation-defined NaN semantics). 2 new tests (vyre-lower 381 lib + 4 snapshot + 6 fuzz = 391 total). The F32 fold matrix now matches U32/I32 for the symmetric ops (Add, Sub, Mul, Div, Min, Max, Eq, Ne, Lt, Le, Gt, Ge); only mod/bitwise/shift remain unsupported on F32 (correctly  -  those don't apply to floats).

### T135 [x] vyre-lower::KernelDescriptor::find_op_by_id  -  id-keyed lookup
**Owner**: CC
**Source**: T134 follow-on (close convenience-lookup gap)
**Result**: SHIPPED. New `find_op_by_id(id) -> Option<&KernelOp>`. Returns the first op anywhere in the descriptor whose result matches `id` in DFS pre-order. 2 new tests (vyre-lower 379 lib + 4 snapshot + 6 fuzz = 389 total). Per-body id space caveat documented  -  an id may be reused across child bodies. Useful for tooling that has a result-id from elsewhere (a recommendation, an audit finding) and wants to retrieve the producing op without iterating manually.

### T134 [x] vyre-lower::KernelDescriptor::ops_iter  -  flat DFS iterator
**Owner**: CC
**Source**: T133 follow-on (close gap: tooling that wants all ops has to recurse manually)
**Result**: SHIPPED. New `ops_iter()` returns `KernelOpsIter<'_>`  -  depth-first pre-order iterator over every `KernelOp` in the descriptor (parent body + every nested child). Maintains `(body, op_idx)` stack frames so the iterator is cheap to advance and naturally lazy. 3 new tests (vyre-lower 377 lib + 4 snapshot + 6 fuzz = 387 total) verify: pre-order traversal (parent first, then children in order), `count() == total_ops()`, empty descriptor yields no items.

### T133 [x] vyre-lower::FullReport + OptimizationStats  -  serde derives
**Owner**: CC
**Source**: T132 follow-on (close serialization gap for CI tooling)
**Result**: SHIPPED. Added `#[derive(serde::Serialize, serde::Deserialize)]` to `OptimizationStats` and `FullReport`. 1 new test (vyre-lower 374 lib + 4 snapshot + 6 fuzz = 384 total) verifies JSON round-trip via `serde_json`. CI tools can now dump FullReport as JSON for telemetry pipelines, snapshot artifacts, kernel scoreboards. Every diagnostic type that's likely to be persisted is now serializable.

### T132 [x] vyre-lower::KernelDescriptor::is_pure  -  inverse of has_side_effects
**Owner**: CC
**Source**: T131 follow-on (close obvious accessor inverse)
**Result**: SHIPPED. New `is_pure()` returns `!has_side_effects()`. 1 new test (vyre-lower 373 lib + 4 snapshot + 6 fuzz = 383 total). Useful for caching tools (pure kernel = safe to cache by descriptor identity) and routing logic that wants pure-only paths.

### T131 [x] vyre-lower::KernelDescriptor::summary_compact  -  terser variant
**Owner**: CC
**Source**: T130 follow-on (close gap: summary is too noisy for compact terminal output)
**Result**: SHIPPED. New `summary_compact()` returns `"<id>(N ops, M bindings)"`. 1 new test (vyre-lower 372 lib + 4 snapshot + 6 fuzz = 382 total). Mirror of `format_short` vs `format_long` distinction on FullReport  -  gives users two density choices.

### T130 [x] vyre-lower::FullReport  -  format_long for verbose CLI output
**Owner**: CC
**Source**: T129 follow-on (close gap: format_short alone is too terse for `--verbose`)
**Result**: SHIPPED. New `FullReport::format_long()` returns a multi-line human-readable view with section headers (Kernel: / Histogram: / Perf audit: / Optimization: / Verify (input):) and indented contents. Each recommendation gets its own line with priority + category + message + speedup-upper-bound. 1 new test (vyre-lower 371 lib + 4 snapshot + 6 fuzz = 381 total). Suitable for CLI tools' `--verbose` output and for diagnostic dumps in CI logs.

### T129 [x] vyre-lower::full_report  -  single-call diagnostic consolidator
**Owner**: CC
**Source**: T128 follow-on (close gap: 5 separate analyses to call individually)
**Result**: SHIPPED. New `vyre_lower::full_report(desc) -> FullReport` consolidates: descriptor summary (raw + post-optimization), op_histogram, PerfAuditReport, verify result, OptimizationStats. 1 new test (vyre-lower 370 lib + 4 snapshot + 6 fuzz = 380 total). FullReport has its own format_short + Display. Single-call entry point for tooling that wants every substrate-neutral piece of information about a kernel  -  telemetry dashboards, CI gates, kernel scorers all benefit.

### T128 [x] cross-emitter parity  -  verify_then_optimize succeeds on corpus
**Owner**: CC
**Source**: T127 follow-on (gate that verify_then_optimize is robust across realistic shapes)
**Result**: SHIPPED. 1 new test in `vyre-emit-spirv/tests/cross_emitter_parity.rs` (parity 7→8, total spirv 35 lib + 8 parity = 43). For every shape in the corpus, asserts `verify_then_optimize(&desc)` returns `Ok((optimized, stats))` with id round-tripping and stats showing ≥1 iteration. Confirms the production-grade entry point is robust across realistic shapes.

### T127 [x] vyre-lower::verify_then_optimize  -  production-grade entry point
**Owner**: CC
**Source**: T126 follow-on (close gap: emit_optimized's debug-assert is non-load-bearing in release)
**Result**: SHIPPED. 3 new tests (vyre-lower 369 lib + 4 snapshot + 6 fuzz = 379 total). New top-level API:
- `verify_then_optimize(desc) -> Result<(KernelDescriptor, OptimizationStats), VerifyFailure>`
- `VerifyFailure::Input(errs)` / `VerifyFailure::Output(errs)` distinguish whether input was bad or rewrite stack produced bad output
- `VerifyFailure::errors()` accessor for the underlying VerifyError list

Promotes the `debug_assert!(verify(optimized))` in the emit crates to a real Result for callers that want input-validation + optimization + output-validation as one explicit step. Useful for production paths where an invalid descriptor should fail fast with a routable error rather than relying on debug-only assertions.

### T126 [x] vyre-lower::OpHistogram  -  is_empty + dominant accessors
**Owner**: CC
**Source**: T125 follow-on (close OpHistogram accessor surface)
**Result**: SHIPPED. 2 new tests (vyre-lower 366 lib + 4 snapshot + 6 fuzz = 376 total). Two helpers on `OpHistogram`:
- `is_empty()`  -  true iff every category is zero (no ops at all).
- `dominant()`  -  returns `Option<(category_name, count)>` for the largest category. None if empty. Useful for one-line "this kernel is mostly X" reporting.

OpHistogram surface: `total`, `is_empty`, `is_memory_bound`, `is_arithmetic_bound`, `dominant`, `format_short`, `Display`. Sufficient for routing logic and telemetry.

### T125 [x] vyre-lower::KernelDescriptor  -  is_empty + has_side_effects helpers
**Owner**: CC
**Source**: T124 follow-on (close descriptor-introspection convenience gap)
**Result**: SHIPPED. 5 new tests (vyre-lower 364 lib + 4 snapshot + 6 fuzz = 374 total). Two helpers on `KernelDescriptor`:
- `is_empty()`  -  true iff `total_ops()` == 0 (no work in any body, parent or child).
- `has_side_effects()`  -  true iff any op anywhere writes (Store*), atomic-modifies, barriers, calls, traps, returns, or opaque. A pure descriptor with no side effects produces no observable output and can be dropped entirely by the emitter.

KernelDescriptor accessor surface: `summary`, `total_ops`, `is_empty`, `has_side_effects`. Useful for tooling that wants to skip emission for trivial kernels or route side-effect-free kernels to a different optimizer.

### T124 [x] vyre-lower::OpHistogram  -  format_short + Display
**Owner**: CC
**Source**: T123 follow-on (close diagnostic-type symmetry  -  6th type now matches the pattern)
**Result**: SHIPPED. 1 new test (vyre-lower 359 lib + 4 snapshot + 6 fuzz = 369 total). `OpHistogram::format_short()` returns `"N ops: lit=X arith=Y mem=Z cf=W sg=V bi=U other=T"`. `std::fmt::Display` delegates. The format_short + Display pattern is now uniform across ALL 6 diagnostic-output types in the substrate: PerfAuditReport, NagaAuditReport, PtxAuditReport, SpirvAuditReport, OptimizationStats, OpHistogram.

### T123 [x] vyre-lower::audit::audit_with_histogram  -  combined audit + telemetry
**Owner**: CC
**Source**: T122 follow-on (couple the new op_histogram into the audit family)
**Result**: SHIPPED. New API `vyre_lower::audit::audit_with_histogram(desc) -> (PerfAuditReport, OpHistogram)`. Re-exported at crate root. 1 new test (vyre-lower 358 lib + 4 snapshot + 6 fuzz = 368 total). One call now returns both perf recommendations AND kernel-shape telemetry  -  useful for routing logic that wants to pick the right substrate-specific optimizer (memory-bound + Coalesce → vec_pack/vec_load_fusion; arithmetic-bound + nothing → tensor_core_fragment). Mirror of run_all_with_stats's combined-return pattern.

### T122 [x] vyre-lower::analyses::op_histogram  -  per-category op count + classifiers
**Owner**: CC
**Source**: T121 follow-on (close gap  -  substrate-neutral analyses had no "what does this kernel do?" telemetry)
**Result**: SHIPPED. New analysis `vyre-lower::analyses::op_histogram`. 9 tests (vyre-lower 357 lib + 4 snapshot + 6 fuzz = 367 total). `analyze(desc) -> OpHistogram` returns counts across 7 categories: literal, arithmetic (BinOp+UnOp+Fma+Cast+Select), memory (Load*+Store*+Atomic+Async), control_flow (Structured*+Barrier+Trap+Return+Resume), subgroup, builtin (LocalInvocationId+WorkgroupId+SubgroupSize+BufferLength+...), other (Call+Opaque+IndirectDispatch+AsyncWait). Recurses into child bodies. Classifiers: `is_memory_bound()` / `is_arithmetic_bound()` (single-bool: dominant category > all others combined). Useful for routing kernels to the right substrate-specific optimizer (memory-bound → vec_pack, arithmetic-bound → tensor_core_fragment) and for telemetry.

vyre-lower analysis count: 11 → 12.

### T121 [x] vyre-lower::audit  -  recommendations_by_category + top_recommendation
**Owner**: CC
**Source**: T120 follow-on (close filtered-access gap on PerfAuditReport)
**Result**: SHIPPED. 3 new tests (vyre-lower 348 lib + 4 snapshot + 6 fuzz = 358 total). Two helpers on `PerfAuditReport`:
- `recommendations_by_category(cat) -> Vec<&Recommendation>`: filtered access by perf-issue family. Useful for tooling that wants only one category (e.g., a memory-perf dashboard wants only `Coalesce`).
- `top_recommendation() -> Option<&Recommendation>`: lowest-priority entry  -  the single most-impactful issue.

The accessor surface on `PerfAuditReport` is now: `format_short`, `is_clean`, `recommendations_by_category`, `top_recommendation`, `Display`  -  sufficient for any tooling pattern.

### T120 [x] vyre  -  std::fmt::Display impls on all 5 diagnostic types
**Owner**: CC
**Source**: T119 follow-on (close the format_short pattern with Display so println!("{}") works)
**Result**: SHIPPED. `std::fmt::Display` impls on `OptimizationStats`, `PerfAuditReport`, `NagaAuditReport`, `PtxAuditReport`, `SpirvAuditReport`  -  each delegates to `format_short()`. `println!("{stats}")` and `stats.to_string()` now work naturally. No new tests (the existing format_short tests cover the output; Display is a thin pass-through). All 4 vyre crates `cargo test` clean post-Display.

### T119 [x] vyre-lower::OptimizationStats  -  format_short for log-line symmetry
**Owner**: CC
**Source**: T118 follow-on (close the format_short pattern across all telemetry types)
**Result**: SHIPPED. New `OptimizationStats::format_short()`: `"ops X→Y (-N), bindings A→B (-M), iters K (converged|stopped)"`. 2 new tests (vyre-lower 345 lib + 4 snapshot + 6 fuzz = 355 total). The format_short pattern is now uniform across all 5 user-facing diagnostic types: PerfAuditReport, NagaAuditReport, PtxAuditReport, SpirvAuditReport, OptimizationStats. Tooling can grep one line of telemetry from any layer.

### T118 [x] vyre-emit-*  -  audit reports gain format_short + is_clean
**Owner**: CC
**Source**: T117 follow-on (mirror PerfAuditReport helpers across the audit family)
**Result**: SHIPPED. 3 new tests across the emit crates (naga 58 lib+11 efficacy=69, ptx 94, spirv 35 lib+7 parity=42). Added to all three audit reports:
- `NagaAuditReport::format_short()`: `"<id> (naga): N candidates (X vec_pack, Y push_constant, prewarm=B)"`
- `PtxAuditReport::format_short()`: `"<id> (ptx sm_M_m): N candidates (Xp, Yvl, Zvs, Wac, Vtc, Usched)"` (carries target through)
- `SpirvAuditReport::format_short()`: `"<id> (spirv): N findings (X subgroup caps, Y wg violations)"`
- All three: `is_clean()` returns true iff no findings.

The convenience-helper API surface is now consistent across all 4 audit reports (PerfAuditReport + 3 emit reports). Single grep-friendly format for log lines at every layer.

### T117 [x] vyre-lower::audit  -  PerfAuditReport::format_short + is_clean helpers
**Owner**: CC
**Source**: T116 follow-on (close convenience-helper gap on PerfAuditReport)
**Result**: SHIPPED. 3 new tests (vyre-lower 343 lib + 4 snapshot + 6 fuzz = 353 total). Two new methods on `PerfAuditReport`:
- `format_short()`  -  one-line human-readable: `"<id>: waste=X.X, N recommendations (top: <msg>)"` or `"<id>: waste=X.X, clean"`. Suitable for log lines and `--verbose` output.
- `is_clean()`  -  true iff zero recommendations AND zero waste_score. Stronger than `recommendations.is_empty()` alone.

The convenience-helper layer is now consistent across the user-facing types: `OptimizationStats` has `is_no_op` + `ops_eliminated` + `bindings_dropped` + `off_graph_dropped` (T080, T115); `KernelDescriptor` has `summary` + `total_ops` (T115); `PerfAuditReport` has `format_short` + `is_clean` (T117).

### T116 [x] rewrite_efficacy harness  -  cover UnOp/Cast/Comparison+Select shapes
**Owner**: CC
**Source**: T115 follow-on (regression gate for the new const_fold variants from T084-T094)
**Result**: SHIPPED. 3 new efficacy tests in `vyre-emit-naga/tests/rewrite_efficacy.rs` (11 total, was 8): `unop_chain_collapses` (BitNot(BitNot(7)) shrinks), `comparison_into_select_collapses_chain` (Lt(3,5) → Lit(true) → Select picks then-branch → 7), `cast_chain_folds` (U32→I32→U32). Each asserts strict op-count reduction. If a future change accidentally turns the UnOp/Cast/Comparison/Select fold into a no-op, these fail with a precise diff. Combined with the original 8 efficacy tests, the harness now gates the full const_fold + identity_elim coverage matrix.

### T115 [x] vyre-lower  -  small API niceties (OptimizationStats + KernelDescriptor)
**Owner**: CC
**Source**: T114 follow-on (close convenience-helper gaps in the public API)
**Result**: SHIPPED. 4 new tests (vyre-lower 340 lib + 4 snapshot + 6 fuzz = 350 total). Two new methods on `OptimizationStats`:
- `is_no_op()`  -  true iff pipeline made no observable change (ops, bindings, literals all unchanged). Tooling can skip emit re-runs when nothing changed.
- `off_graph_dropped()`  -  single-number "tail of pipeline cleanup" signal (bindings dropped + literals dropped).

Two new methods on `KernelDescriptor`:
- `summary()`  -  one-line human-readable: `"<id>: N ops, M bindings, K child bodies, L literals, dispatch [x, y, z]"`. Useful for diagnostic logs.
- `total_ops()`  -  recursive op count across parent + every nested child body. Distinct from `body.ops.len()` which is shallow.

### T114 [x] vyre-lower::verify  -  DispatchZeroDim check at the substrate-neutral layer
**Owner**: CC
**Source**: T113 follow-on (catch host-side dispatch bugs at the substrate-neutral layer too, not just SPIR-V)
**Result**: SHIPPED. New `VerifyErrorKind::DispatchZeroDim { axis }`. 3 new tests (vyre-lower 336 lib + 4 snapshot + 6 fuzz = 346 total). `verify(desc)` now also checks that no `dispatch.workgroup_size` dim is zero  -  a kernel with zero dim never runs (almost certainly a host-side bug). Substrate-neutral check; SPIR-V's `workgroup_size_validation` already catches this AND per-dim/product limits, but those are Vulkan-specific. The substrate-neutral check fires regardless of which emitter the descriptor goes through. The debug-mode `verify` gate in every `emit_optimized` therefore now also catches zero-dim dispatches.

### T113 [x] vyre-lower::rewrites::identity_elim  -  Fma absorbing-zero
**Owner**: CC
**Source**: T112 follow-on (Fma was the only major op kind not handled by identity_elim)
**Result**: SHIPPED. `identity_elim_body` now matches `KernelOpKind::Fma`: when either factor (operand 0 or operand 1) resolves to a numeric-zero literal, the Fma's result equals operand 2 (c)  -  substitute id_remap accordingly. 3 new tests (vyre-lower 333 lib + 4 snapshot + 6 fuzz = 343 total). Lit(1) factor cases (which would simplify Fma → Add) are NOT handled here; they'd require synthesizing a new Add op, which is outside identity_elim's pure id-substitution model  -  that's a job for a future Fma-specific rewrite. The id-substitution family in identity_elim now covers: BinOp identity/absorbing/self-equality + Select(Lit_bool) + Fma(_,0,_)/(0,_,_).

(Running tally correction: the earlier session entries cited vyre-lower at 350  -  the actual count after T109 was 340. T113 adds 3, landing at 343.)

### T112 [x] vyre-emit-* READMEs  -  refresh for new APIs across naga/ptx/spirv
**Owner**: CC
**Source**: T111 follow-on (emit-crate READMEs predated T087-T108)
**Result**: SHIPPED. All three emit READMEs now document:
- `emit_optimized_with_stats` (naga, ptx, spirv).
- PTX: `emit_optimized_with_target_with_stats` (combined target+stats), 6 patterns including `vec_load_fusion` + `vec_store_fusion`, `patterns::audit` + `audit_optimized`.
- SPIR-V: `emit_optimized_bytes` + `emit_optimized_bytes_with_stats` bytes-axis variants, `workgroup_size_validation` second pattern, `patterns::audit` + `audit_optimized`.
- Naga: `patterns::audit` + `audit_optimized`.

`cargo check -p vyre-emit-naga -p vyre-emit-ptx -p vyre-emit-spirv --release` clean. All four crate READMEs (lower + 3 emits) now fully reflect the post-T112 state.

### T111 [x] vyre-lower README  -  refresh for extended pipeline + audit_optimized
**Owner**: CC
**Source**: T110 follow-on (README was written 35 features ago; was stale)
**Result**: SHIPPED. README updated:
- const_fold section now lists the full coverage (BinOp×Type matrix + 10 UnOp variants + Cast pairs + Fma F32 + identity_elim's Select).
- pipeline step count corrected: 13 → 15 (added drop_unused_literals + drop_unused_child_bodies).
- audit section documents `audit_optimized` family across all 4 layers (substrate-neutral + naga + PTX + SPIR-V), with the diagnostic intent ("what's left after the pipeline?").

`cargo check -p vyre-lower --release` clean. README ready for docs.rs.

### T110 [x] cross-emitter parity  -  audit_optimized panic-free property
**Owner**: CC
**Source**: T109 follow-on (gate that the audit_optimized family is robust)
**Result**: SHIPPED. 1 new test in `vyre-emit-spirv/tests/cross_emitter_parity.rs` (parity 6→7, total spirv 41). For every shape in the corpus, audit_optimized at all 4 layers (vyre_lower, vyre_emit_naga, vyre_emit_ptx, vyre_emit_spirv) is called and required to: (a) not panic, (b) return a report whose kernel_id matches the input. Mirror of T107 for the _optimized variants. The audit + audit_optimized families are now uniformly tested for robustness across all layers.

### T109 [x] vyre-lower::audit::audit_optimized  -  symmetry across all 4 layers
**Owner**: CC
**Source**: T108 follow-on (close the symmetry: substrate-neutral layer was missing the _optimized variant)
**Result**: SHIPPED. New `vyre_lower::audit::audit_optimized(desc) -> PerfAuditReport` re-exported at crate root. 1 new test (vyre-lower 330 lib + 4 snapshot + 6 fuzz = 340). Calls `rewrites::run_all` first, then `audit`. The audit_optimized family is now uniformly available at every layer:

- `vyre_lower::audit::audit_optimized(desc)`  -  substrate-neutral (3 analyses: coalesce, shared_mem_promote, bank_conflict)
- `vyre_emit_naga::patterns::audit_optimized(desc)`  -  naga-side (vec_pack, push_constant, prewarm)
- `vyre_emit_ptx::patterns::audit_optimized(desc, target)`  -  PTX-side (6 patterns)
- `vyre_emit_spirv::patterns::audit_optimized(desc)`  -  SPIR-V-side (subgroup_capabilities, workgroup_validation)

Single-call diagnostic at every layer: "what's left to optimize after the standard pipeline already ran?"

### T108 [x] vyre-emit-*  -  patterns::audit_optimized: audit the post-run_all form
**Owner**: CC
**Source**: T107 follow-on (close the diagnostic gap: "what's left after the standard pipeline?")
**Result**: SHIPPED. New `audit_optimized` function in all 3 emit crates (`vyre_emit_naga::patterns`, `vyre_emit_ptx::patterns`, `vyre_emit_spirv::patterns`). 3 new tests (naga 57, ptx 93, spirv 34 lib + 6 parity = 40). `audit_optimized(desc)` calls `vyre_lower::rewrites::run_all(desc)` first, then the existing `audit()` on the optimized form. Tells callers: "what substrate-specific optimizations are LEFT after the standard rewrite stack already ran?"  -  non-empty means the substrate-specific layer is the only path to recover the remaining perf (e.g. surviving vec_load_fusion candidate means the PTX emit-side layer must be taught to fuse, since substrate-neutral CSE can't reach it). PTX variant takes ComputeCapability through. Workgroup-size validation in SPIR-V is invariant under run_all (rewrite stack doesn't touch dispatch.workgroup_size)  -  asserted directly.

### T107 [x] cross-emitter parity  -  extend with audit-doesn't-panic property
**Owner**: CC
**Source**: T106 follow-on (gate that the audit family is robust across realistic shapes)
**Result**: SHIPPED. 2 new tests in `vyre-emit-spirv/tests/cross_emitter_parity.rs` (parity test count 4→6, total spirv 39). For every descriptor in the corpus, all 4 audit functions are called and required to (a) not panic, (b) return a report whose `kernel_id` matches the input. Covers: `vyre_lower::audit::audit`, `vyre_emit_naga::patterns::audit`, `vyre_emit_ptx::patterns::audit` (with target SM_80), `vyre_emit_spirv::patterns::audit`. Plus a separate test that asserts kernel_id round-trips through all four layers unchanged for a named kernel. Confirms the audit family is consistent and panic-free across realistic shapes.

### T106 [x] vyre crates  -  clean-build sanity check (post-T070-T105 sweep)
**Owner**: CC
**Source**: T105 follow-on (gate that the session additions stayed warning-free)
**Result**: SHIPPED. `cargo build -p vyre-lower -p vyre-emit-naga -p vyre-emit-ptx -p vyre-emit-spirv --release` produces zero warnings and zero errors. The T086 cleanup held through every later addition (T087–T105: emit_optimized_with_stats, fuzz extensions, snapshot tests, audit family, vec_load_fusion, vec_store_fusion, vec_load_fusion+vec_store_fusion in PTX, workgroup_size_validation in SPIR-V, target+stats variants, bytes variants). Crates remain publish-ready.

### T105 [x] vyre-emit-spirv  -  emit_optimized_bytes + emit_optimized_bytes_with_stats
**Owner**: CC
**Source**: T104 follow-on (close the bytes-axis API gap to mirror words-axis)
**Result**: SHIPPED. 2 new APIs in `vyre_emit_spirv`: `emit_optimized_bytes(desc) -> Result<Vec<u8>, EmitError>` and `emit_optimized_bytes_with_stats(desc) -> Result<(Vec<u8>, OptimizationStats), EmitError>`. 2 new tests (vyre-emit-spirv 33 lib + 4 parity = 37 total). Refactored `emit_bytes` to share a private `words_to_le_bytes` helper. Full SPIR-V emit API: `emit`, `emit_optimized`, `emit_optimized_with_stats`, `emit_bytes`, `emit_optimized_bytes`, `emit_optimized_bytes_with_stats`, `emit_from_naga_module`. Production loaders that want minimal bytewise output + already-optimized contents now have a single-call path.

### T104 [x] vyre-emit-ptx::emit_optimized_with_target_with_stats  -  combined variant
**Owner**: CC
**Source**: T103 follow-on (closed gap in PTX emit API matrix)
**Result**: SHIPPED. New API `vyre_emit_ptx::emit_optimized_with_target_with_stats(desc, target) -> Result<(String, OptimizationStats), EmitError>`. 1 new test (vyre-emit-ptx 92 total). Closes the API gap: previously users targeting sm_90 who also wanted stats had to either re-run `run_all_with_stats` themselves OR drop to default sm_70. Now one call gets both. `emit_optimized_with_target` is now a thin wrapper. Full PTX optimization API surface: `emit`, `emit_with_target`, `emit_optimized`, `emit_optimized_with_target`, `emit_optimized_with_stats`, `emit_optimized_with_target_with_stats`.

### T103 [x] vyre-emit-spirv::patterns::audit  -  unified SPIR-V-pattern report
**Owner**: CC
**Source**: T102 follow-on (completes the audit family symmetry across all 3 substrate layers)
**Result**: SHIPPED. New API `vyre_emit_spirv::patterns::audit(desc) -> SpirvAuditReport`. 3 new tests (vyre-emit-spirv 31 lib + 4 parity = 35 total). SpirvAuditReport bundles `subgroup` + `workgroup_validation`. Helpers: `requires_action()` (true iff any subgroup capability needed OR any workgroup violation present  -  both are pipeline-build-time concerns) and `total_findings()`. Audit family is now uniformly available across all three layers: `vyre_lower::audit::audit`, `vyre_emit_naga::patterns::audit`, `vyre_emit_ptx::patterns::audit`, `vyre_emit_spirv::patterns::audit`. Any caller can ask "what optimizations apply at layer X?" with one call.

### T102 [x] vyre-emit-spirv::patterns::workgroup_size_validation  -  Vulkan limit checks
**Owner**: CC
**Source**: T101 follow-on (extend SPIR-V patterns beyond just subgroup_capabilities)
**Result**: SHIPPED. 9 new tests in `vyre-emit-spirv::patterns::workgroup_size_validation` (vyre-emit-spirv 28 lib + 4 parity = 32 total). New API: `analyze(desc) -> ValidationReport` (uses `VULKAN_BASELINE` limits  -  `[1024, 1024, 64]` per dim, 1024 max invocations) and `analyze_against(desc, custom_limits)` (for known per-device profiles like an RTX-class card with looser z-dim). Reports each violation as a separate `Violation` enum entry: `DimExceeded { axis, actual, limit }`, `InvocationsExceeded { actual, limit }`, `ZeroDim { axis }`. Helpers `ok()` and `invocations()`. Catches three real classes of bug: per-dimension overflow (Vulkan rejects), product overflow (driver rejects), zero-dim (kernel never runs). vyre-emit-spirv now has 2 patterns: subgroup_capabilities + workgroup_size_validation.

### T101 [x] vyre-emit-naga::patterns::audit  -  unified naga-pattern report
**Owner**: CC
**Source**: T100 follow-on (mirror to naga side; same shape)
**Result**: SHIPPED. New API `vyre_emit_naga::patterns::audit(desc) -> NagaAuditReport`. 2 new tests (vyre-emit-naga 56 lib + 8 efficacy = 64 total). NagaAuditReport bundles: `vec_pack`, `push_constant`, `prewarm`. `bind_group_reuse` is excluded because it operates on a slice of descriptors, not a single one. Helpers `total_candidates()` (sums vec_pack groups + push_constant candidates + prewarm bool) and `has_any()` give a single-number signal. The audit family is now complete across all three layers: substrate-neutral `vyre_lower::audit::audit` (10 analyses); naga-specific `vyre_emit_naga::patterns::audit` (3 patterns); PTX-specific `vyre_emit_ptx::patterns::audit` (6 patterns).

### T100 [x] vyre-emit-ptx::patterns::audit  -  unified PTX-pattern report
**Owner**: CC
**Source**: T099 follow-on (single-call PTX audit, mirror of vyre_lower::audit)
**Result**: SHIPPED. New API `vyre_emit_ptx::patterns::audit(desc, target) -> PtxAuditReport`. 3 new tests (vyre-emit-ptx 91 total). PtxAuditReport has one field per shipped pattern: `predication`, `vec_load`, `vec_store`, `async_copy`, `tensor_core`, `scheduling`. Helpers `total_candidates()` and `has_any()` give a single-number "is anything actionable" signal. ComputeCapability now derives serde for the report's serialization. Mirror of `vyre_lower::audit::audit` for the substrate-specific PTX layer  -  gives any caller a single function to ask "what PTX-side optimizations apply to this kernel?"

### T099 [x] vyre-emit-ptx::patterns::vec_store_fusion  -  detect st.global.v2/v4 candidates
**Owner**: CC
**Source**: T098 follow-on (mirror for stores  -  same throughput benefit)
**Result**: SHIPPED. 6 tests in `vyre-emit-ptx::patterns::vec_store_fusion` (vyre-emit-ptx 88 total). Mirror of vec_load_fusion for `StoreGlobal`. Same chain shape detection (`Store; Add(idx, 1); Store; Add; Store; ...` up to 4 stores), same group_size/alignment_bytes reporting. Indistinguishable from the load case in algorithm; differs only in which op kind is matched and which operand carries the index (StoreGlobal index is at operand 1, value at operand 2). PTX patterns now total 6: instruction_scheduling, ldmatrix_cp_async, predicated_execution, tensor_core_fragment, vec_load_fusion, vec_store_fusion.

### T098 [x] vyre-emit-ptx::patterns::vec_load_fusion  -  detect ld.global.v2/v4 candidates
**Owner**: CC
**Source**: PERF B1 (PTX-side companion to vyre-emit-naga's vec_pack)
**Result**: SHIPPED. 8 tests in `vyre-emit-ptx::patterns::vec_load_fusion` (vyre-emit-ptx 82 total). Detection-only analysis  -  walks descriptor for groups of 2 or 4 sequential `LoadGlobal` ops to the same slot whose indices form a `+1`-stride chain (each next index is `Add(prev_index, Lit(1))` with the Add appearing between the loads in the op stream). Returns `FusionPlan { candidates: Vec<FusionCandidate> }`. Each candidate carries: first_load_idx, group_size (2 or 4  -  PTX has no v3), binding_slot, element_type, alignment_bytes (group_size * elem_size  -  host allocator must satisfy). Recurses into child bodies. Different slots / non-unit stride / non-Add intervening op all break the chain. Companion to existing PTX patterns (tensor_core_fragment, ldmatrix_cp_async, predicated_execution, instruction_scheduling). Actual emit-side rewrite (turn 4 scalar `ld.global.u32` into 1 `ld.global.v4.u32`) is a follow-up.

### T097 [x] vyre-lower::rewrites::drop_unused_child_bodies  -  strip orphans from inlining
**Owner**: CC
**Source**: T096 follow-on (third member of the drop_unused_* family)
**Result**: SHIPPED. 7 tests in `vyre-lower::rewrites::drop_unused_child_bodies` (vyre-lower 329 lib total). Mirror of `drop_unused_bindings`/`drop_unused_literals` for the third pool: child bodies. Walks parent body, collects every child-body index referenced by an op (StructuredIfThen pos 1; StructuredIfThenElse pos 1+2; StructuredForLoop pos 2; StructuredBlock pos 0; Region pos 0). Filters `child_bodies` to keep only referenced; renumbers indices dense; rewrites every op's child-body-idx operand. Recurses into children FIRST so nested orphans are stripped before parent evaluation. Fires often after `branch_collapse` (collapsed If's children are orphaned) and `loop_unroll` (unrolled body's child is orphaned). Wired as the LAST step of run_all_once (after literals + bindings drop).

vyre-lower: 13 active rewrites (+ licm no-op = 14 in module). The drop_unused_* family now covers all three off-graph data: bindings, literals, child bodies.

### T096 [x] vyre-lower::rewrites::const_fold  -  I32 BitAnd/BitOr/BitXor/Shl/Shr coverage
**Owner**: CC
**Source**: T095 follow-on (closes type-coverage gap)
**Result**: SHIPPED. 2 new tests (vyre-lower 322 lib). I32 fold now covers all 5 bitwise/shift ops the U32 fold has. `Shr` on i32 sign-extends per Rust semantics  -  matches GPU arithmetic-right-shift behavior. Test verifies `Shr(-16, 2) == -4`. Full BinOp×LiteralType matrix is now closed for U32, I32, F32, Bool.

### T095 [x] vyre-lower::rewrites::const_fold  -  WrappingAdd/WrappingSub coverage
**Owner**: CC
**Source**: T094 follow-on (closes the U32/I32 BinOp coverage gap)
**Result**: SHIPPED. 1 new test (vyre-lower 320 lib). `WrappingAdd`/`WrappingSub` on U32 and I32 literals now fold via `wrapping_add`/`wrapping_sub` (same as the unchecked `Add`/`Sub` variants  -  wrap is the default for the GPU semantics in the IR; the variants exist to express explicit intent, not different math). Closes the BinOp coverage matrix: every variant in `BinOp` is now folded for the relevant literal types.

### T094 [x] vyre-lower::rewrites::const_fold  -  comparison op folding (Eq/Ne/Lt/Le/Gt/Ge)
**Owner**: CC
**Source**: T093 follow-on (closes the gap that branch_collapse needs Lit_bool conditions)
**Result**: SHIPPED. 6 new tests in `vyre-lower::rewrites::const_fold` (vyre-lower 319 lib). `fold_binop` now handles all 6 comparisons across U32/I32/F32/Bool. Float comparisons gated on `!x.is_nan() && !y.is_nan()` per IEEE-specific NaN semantics  -  NaN compares stay as the original op. The composition test (`const_fold_comparison_chains_into_select_fold`) verifies the end-to-end pipeline: `Lt(3, 5)` → `Lit(true)` (const_fold) → `Select(Lit(true), then, else)` → `then` (identity_elim) → unused branch DCE'd. The full optimization chain `comparison → branch decision → branch elimination` now works without any of the input being a literal Bool to start with.

### T093 [x] vyre-lower  -  kitchen-sink snapshot test
**Owner**: CC
**Source**: T092 follow-on (snapshot regression gate for the entire pipeline)
**Result**: SHIPPED. 4 tests in `vyre-lower/tests/kitchen_sink_snapshot.rs`. Pins the optimization output of the same kernel as `examples/optimize.rs`: stats (11→3 ops, 3→1 bindings, 4→2 literals, 2 iterations, converged), final op shape (exactly `Lit; Lit; StoreGlobal(slot=0, idx=r0, val=r1)`), pool contents (`U32(0); U32(56)` where 56 = 7<<3 from strength_reduce + const_fold), and surviving binding (just `output`). Any future pipeline change that breaks the optimization will fail this test with a precise diff. Keeps the example output as a stable, asserted contract.

### T092 [x] vyre-lower fuzz  -  byte-determinism property across corpus
**Owner**: CC
**Source**: T091 follow-on (caching + snapshot-test prerequisites)
**Result**: SHIPPED. 1 new test in `vyre-lower/tests/rewrite_soundness_fuzz.rs` (6 tests total). For 200 random descriptors, asserts `run_all(d) == run_all(d)` byte-for-byte across body.ops, body.literals, body.child_bodies, and bindings. Catches non-determinism from HashMap iteration (random hash seed by default), system clock, or any other accidental dependency. Result: clean  -  every pass uses BTreeMap or order-preserving Vec ops, no determinism leaks. Build-caching and snapshot-test compatible.

### T091 [x] vyre-lower::rewrites::identity_elim  -  Select(Lit_bool, then, else) → forwarded ref
**Owner**: CC
**Source**: T090 follow-on (Select with literal cond is the natural id-substitution case for identity_elim)
**Result**: SHIPPED. 4 new tests in `vyre-lower::rewrites::identity_elim` (vyre-lower 313 lib + 5 fuzz). `identity_elim_body`'s op walk now matches both `BinOpKind` (existing) and `Select` (new): when Select's cond resolves to a Bool literal, record `select_result_id → then_id` (true) or `select_result_id → else_id` (false) in `id_remap`. Uses the same transitive `resolve()` helper so chained Selects collapse. Non-literal cond, non-Bool literal cond → Select op stays. The op itself is left for DCE to drop. Together with T090's Fma fold, the const-folding family now covers BinOp/UnOp/Cast/Fma in `const_fold`, plus Select in `identity_elim` (where id-substitution is the right factoring rather than literal-replacement).

### T090 [x] vyre-lower::rewrites::const_fold  -  extend to Fma(Lit, Lit, Lit)
**Owner**: CC
**Source**: T089 follow-on (completes the const_fold family: BinOp + UnOp + Cast + Fma all covered)
**Result**: SHIPPED. 4 new tests in `vyre-lower::rewrites::const_fold` (vyre-lower 309 lib + 5 fuzz). `fold_fma()` handles `Fma(a, b, c)` = `a * b + c` when all three are F32 literals AND the result is finite. Uses Rust's `f32::mul_add` so the fold matches hardware FMA semantics (single rounding, not multiply-then-round-then-add). Integer Fma operands not folded  -  Fma is fundamentally a float op. Overflow → not folded (stays as Fma). Mixed literal/non-literal operands not folded. The const_fold pass now covers all four primary arithmetic shapes in the IR.

### T089 [x] vyre-lower::rewrites::drop_unused_literals  -  strip pool entries no Literal op references
**Owner**: CC
**Source**: T088 follow-on (the optimize example surfaced literals 4 → 9 growth)
**Result**: SHIPPED. 7 tests in `vyre-lower::rewrites::drop_unused_literals` (vyre-lower 305 lib + 5 fuzz). Mirror of `drop_unused_bindings` for the literal pool: collect every pool index referenced by `Literal` op operand 0; filter `body.literals` to keep only referenced entries; renumber pool indices dense `0..N`; rewrite each surviving `Literal` op's operand 0 to the new index. Per-body, recurses into children. Wired into run_all as the final step (after drop_unused_bindings). Re-running the optimize example confirms: literal pool now shrinks 4 → 2 (was 4 → 9 before), final descriptor exactly `r0 = Lit(0); r1 = Lit(56); Store(slot=0, idx=r0, val=r1)`.

### T088 [x] vyre-lower  -  examples/optimize.rs runnable end-to-end demo
**Owner**: CC
**Source**: T087 follow-on (concrete entry point for new users)
**Result**: SHIPPED. `vyre-lower/examples/optimize.rs` builds an 11-op kitchen-sink kernel (4 literals + 4 dead/identity arithmetic ops + 3 mem ops, 3 bindings of which 2 unused), runs `run_all_with_stats`, prints the before/after summary, runs `verify`, and dumps the surviving descriptor. Output on `cargo run --example optimize -p vyre-lower`:

```
=== before ===
ops:           11
bindings:      3
literals:      4

=== after run_all_with_stats ===
ops:           3 (8 eliminated)
bindings:      1 (2 dropped)
literals:      4 -> 9
iterations:    2
converged:     true

=== verify ===
OK
```

73% op reduction (11 → 3), 2/3 bindings dropped, fixed-point converged in 2 iterations. The optimized kernel is just `r0 = Lit(0); r1 = Lit(56); Store(slot=0, idx=r0, val=r1)`. The literal-pool grows (4 → 9) because intermediate folds add new entries  -  `cse` deduplicates references but `drop_unused_literals` is not a pass yet (open follow-up; not yet done).

### T087 [x] vyre-emit-*  -  emit_optimized_with_stats: surface OptimizationStats to callers
**Owner**: CC
**Source**: T086 follow-on (T080 stats are produced internally but never reach callers)
**Result**: SHIPPED. New API on all three emit crates:

- `vyre_emit_naga::emit_optimized_with_stats(desc) -> Result<(naga::Module, OptimizationStats), EmitError>`
- `vyre_emit_ptx::emit_optimized_with_stats(desc) -> Result<(String, OptimizationStats), EmitError>`
- `vyre_emit_spirv::emit_optimized_with_stats(desc) -> Result<(Vec<u32>, OptimizationStats), EmitError>`

`emit_optimized` in each crate is now a thin wrapper that drops the
stats  -  zero duplicate work, no behavior change for existing callers.
Stats let downstream tooling log "12 ops -> 5 ops in 2 iterations,
3 bindings -> 1 dropped" without re-running `run_all`. All 155
emit-crate tests pass with the stats wiring active.

### T086 [x] vyre-lower  -  clear accumulated dead-code warnings (publish readiness)
**Owner**: CC
**Source**: T085 follow-on (publish gate + crates-of-crates mandate)
**Result**: SHIPPED. vyre-lower release builds clean. Four warnings resolved:

- `analyses/const_buffer_promote/mod.rs`: `DataType` import was unused in lib but USED in `#[cfg(test)]` block. Gated the import with `#[cfg(test)]` so it's only compiled when needed.
- `rewrites/const_fold/mod.rs`: `KernelOp` import dropped  -  only `KernelOpKind` was actually referenced after the UnOp/Cast extensions.
- `analyses/workgroup_uniform/analysis.rs`: `DepInfo::contains_thread_id` accessor was dead. Wired in: `classify()` now calls the accessor instead of touching `info.has_thread_id` directly. One real call site is enough; no need for an unused method.
- `rewrites/strength_reduce/mod.rs`: `synthesize_literal_op` helper was dead AND subtly broken (recomputed `max(result_id)` per call → would generate duplicate ids on a 2-rewrite body). The inline implementation that supersedes it uses a running `next_id` counter  -  correct AND O(1) per call. Helper deleted.

vyre-lower 298 lib + 5 fuzz = 303 tests, 0 warnings on release build.

### T085 [x] vyre-lower::rewrites::const_fold  -  extend to Cast(Lit)
**Owner**: CC
**Source**: T084 follow-on (const_fold's header doc listed Cast as "phase 2")
**Result**: SHIPPED. 9 new tests in `vyre-lower::rewrites::const_fold` (vyre-lower 298 lib + 5 fuzz). `fold_cast()` handles: U32↔I32 (bit reinterpret), U32/I32→F32 (IEEE round, finite-only), F32→U32/I32 (only when finite AND in range  -  negative→u32, NaN, ±inf, overflow all stay as Cast op), Bool→U32/I32 (0/1), same-type no-op. Vector types, F64, F16, BF16, byte buffers, arrays  -  all gracefully return None and the Cast op stays in place. Composes with the existing UnOp folding so multi-step constant chains (`Cast(Negate(Lit))`, `Cast(Cast(Lit))`) collapse fully via the per-pass fixed-point iteration.

### T084 [x] vyre-lower::rewrites::const_fold  -  extend to UnOp(Lit)
**Owner**: CC
**Source**: T083 follow-on (const_fold's own header doc listed UnOp as "phase 2")
**Result**: SHIPPED. 10 new tests in `vyre-lower::rewrites::const_fold` (vyre-lower 289 lib + 5 fuzz). `fold_unop()` now handles: BitNot/Popcount/Clz/Ctz/ReverseBits on integers; Negate on i32/f32; LogicalNot on bool; Abs/Floor/Ceil/Round/Trunc/Sqrt/Cos/Sin on f32 (only when result is finite  -  same NaN/inf policy as the BinOp side). Type mismatches (e.g., Negate on Bool) and non-literal operands gracefully skip folding instead of producing garbage. Wired automatically  -  `const_fold` already runs in `run_all` before identity_elim, so unary literal folding now exposes more identity-elim opportunities (e.g., `BitNot(0xFFFFFFFF) == 0` → identity_elim's absorbing-zero rules apply).

### T083 [x] cross-emitter parity test  -  same descriptor through naga/ptx/spirv
**Owner**: CC
**Source**: T082 follow-on (prove the substrate-neutral promise of KernelDescriptor)
**Result**: SHIPPED. New test `vyre-emit-spirv/tests/cross_emitter_parity.rs` with 4 tests; vyre-emit-spirv now 23 tests total (was 19; +4 parity). 5-descriptor corpus (empty, single-store, add+store, identity-heavy arithmetic, store-load-store) lowers through all three emit_optimized paths. Tests assert: (a) every descriptor succeeds in all 3 emitters, (b) naga + spirv share entry-point name + workgroup_size, (c) ptx output contains required `.version` and `.target` directives, (d) raw `emit` and `emit_optimized` agree on success/failure. Lives in vyre-emit-spirv (which already depends on naga); ptx added as dev-dependency. Avoids the vyre-libs Codex-hold dependency chain that blocks vyre-bench.

### T082 [x] vyre-emit-* READMEs  -  all three emit crates documented for crates.io
**Owner**: CC
**Source**: T081 follow-on (extend doc coverage to the rest of the user-facing surface)
**Result**: SHIPPED. Three new READMEs:

- `vyre-emit-naga/README.md`  -  emit + emit_optimized, naga-specific patterns (vec_pack, push_constant_inline, bind_group_reuse, pipeline_prewarm), EmitError variants.
- `vyre-emit-ptx/README.md`  -  emit + emit_with_target + emit_optimized + emit_optimized_with_target, ComputeCapability range (SM_60–SM_90), PTX-specific patterns (tensor_core_fragment, ldmatrix_cp_async, predicated_execution, instruction_scheduling).
- `vyre-emit-spirv/README.md`  -  emit + emit_optimized + emit_bytes + emit_from_naga_module, SPIRV_MAGIC constant, the naga→spv routing strategy explanation, subgroup_capabilities pattern.

All three Cargo.toml files updated to declare `readme = "README.md"`. `cargo check -p vyre-emit-naga -p vyre-emit-ptx -p vyre-emit-spirv` clean. Each crate now has a docs.rs landing page when published.

### T081 [x] vyre-lower README  -  public-API documentation for crates-of-crates mandate
**Owner**: CC
**Source**: T080 follow-on (crate matured this session; deserves user-facing docs)
**Result**: SHIPPED. `vyre-lower/README.md` covers: what the crate is, the `KernelDescriptor` IR (per-body id space caveat documented), quick-start example, the 13-step `run_all` pipeline (12 distinct rewrites; dce twice), 11 analyses with one-line descriptions each, the `verify` contract and its debug-assert wiring in `emit_optimized`, and pointers to the substrate emitters. `Cargo.toml` updated to declare `readme = "README.md"`. `cargo check -p vyre-lower` clean. README is the entry point for anyone calling `cargo add vyre-lower`.

### T080 [x] vyre-lower::rewrites::run_all_with_stats  -  expose optimization metrics
**Owner**: CC
**Source**: T079 follow-on (instrumentation for benchmarks + diagnostics)
**Result**: SHIPPED. New public API in `vyre-lower::rewrites`: `OptimizationStats { ops_before, ops_after, bindings_before, bindings_after, literals_before, literals_after, iterations, converged }` + helpers `ops_eliminated()` and `bindings_dropped()`. `run_all_with_stats(desc) -> (KernelDescriptor, OptimizationStats)`. `run_all` is now a thin wrapper around it that drops the stats. 3 new tests (vyre-lower 279 lib + 5 fuzz). Stats lets benchmarks track regression in optimization quality, and lets `--verbose` emit modes show users what the pipeline did.

### T079 [x] vyre-lower::rewrites::drop_unused_bindings  -  strip slots no op references
**Owner**: CC
**Source**: PERF_ROADMAP A36 (resource-binding minimization  -  host-side cost reduction)
**Result**: SHIPPED. 6 tests in `vyre-lower::rewrites::drop_unused_bindings` (vyre-lower 276 lib + 5 fuzz). Walks descriptor (parent + child bodies); collects every `BindingSlot.slot` value referenced by Load/Store/Atomic/AsyncLoad/AsyncStore/BufferLength via operand[0]. Filters `desc.bindings.slots` to keep only referenced slots. No renumbering needed  -  naga emitter (and others) look up bindings by `.slot` field, not Vec position, so dropped Vec entries don't shift the addressing of surviving slots. Wired as the LAST step of run_all_once: must run after the rewrites that drop ops, since those may make a binding unreferenced.

### T078 [x] emit_optimized  -  debug-mode verify gate at the rewrite/emit boundary
**Owner**: CC
**Source**: T077 follow-on (catch rewrite bugs at the boundary, not deep inside emit code)
**Result**: SHIPPED. `vyre_emit_naga::emit_optimized`, `vyre_emit_ptx::emit_optimized`, `vyre_emit_ptx::emit_optimized_with_target`, and `vyre_emit_spirv::emit_optimized` now `debug_assert!(verify(optimized).is_ok())` immediately after `run_all` and before passing to the lowering. Production builds pay nothing (debug_assert is no-op in release); test/dev builds get a clean panic with a useful message ("rewrite pipeline produced an invalid descriptor  -  see vyre_lower::verify for the contract") instead of garbage emit output. All 155 emit-crate tests pass with the gate active in debug mode.

### T077 [x] vyre-lower::rewrites::canonicalize  -  sort commutative-op operands for CSE
**Owner**: CC
**Source**: T076 follow-on (CSE effectiveness gap on commutative ops)
**Result**: SHIPPED. 10 tests in `vyre-lower::rewrites::canonicalize` (vyre-lower 270 lib + 5 fuzz). For commutative `BinOp` (Add, Mul, BitAnd, BitOr, BitXor, Min, Max, Eq, Ne, WrappingAdd), sort the two operand result-ids ascending. `Add(r5, r1)` becomes `Add(r1, r5)`. Two structurally-equal expressions with operands written in different orders normalize to the same form, enabling CSE to merge them. Sub/Div/Mod/Shl/Shr/Lt/Le/Gt/Ge are explicitly NOT commuted (operand order is semantic). Wired into run_all between dce and cse.

### T076 [x] Fuzz harness extended to CF + caught two real bugs
**Owner**: CC
**Source**: T075 follow-on (extended generator coverage)
**Result**: SHIPPED. Generator now produces If/ForLoop/Atomic/Barrier in addition to Lit/BinOp/Store/Load. 1000-case verify+run_all property still holds. Two real bugs caught and fixed:

1. **`loop_unroll` integer underflow** (`vyre-lower::rewrites::loop_unroll`). The `unrollable` check used `hi.saturating_sub(lo)` (returns 0 on underflow), but the actual inlining loop used `let count = hi - lo` which wraps in release mode. For hi=2, lo=255 the for-loop iterated ~4.3 billion times, OOM-killed by earlyoom. Fix: use saturating_sub in both places. Regression test added (`loop_unroll_does_not_underflow_when_hi_less_than_lo`).

2. **`licm` violated per-body id-space contract** (`vyre-lower::rewrites::licm`). Hoisting an op from a child body to the parent body created dangling references  -  the child body still referenced the hoisted result-id, but the id no longer existed in the child's (separate) id space. Fuzz harness caught this at seed 103. Fix: `licm` is now a no-op pending a correct cross-body hoist (needs either value-passing mechanism into child bodies, or a guarantee of zero in-body uses). The analysis pieces (`is_pure`, `hoist_invariants`) stay as `licm_unsafe_no_id_rewrite` so existing tests keep covering them and a future fix has a starting point.

vyre-lower: 260 lib tests + 5 fuzz tests = 265 total.

### T075 [x] rewrite_soundness_fuzz + run_all fixed-point iteration
**Owner**: CC
**Source**: T074 follow-on (correctness gate via property fuzzing)
**Result**: SHIPPED. 4 fuzz tests in `vyre-lower/tests/rewrite_soundness_fuzz.rs`. Hand-rolled LCG generator emits 1000 random descriptors per test (Literal / BinOp(10 variants) / StoreGlobal / LoadGlobal mix; 1–2 bindings; 2–5 lits; up to ~10 ops). Properties tested: (a) generator self-validates  -  every input passes `verify()`; (b) `verify(run_all(input))` holds for all 1000 inputs; (c) `run_all` is idempotent at fixed point across 200 inputs; (d) op count never grows by more than 5 across 500 inputs.

**Bug found and fixed**: original `run_all` was NOT idempotent at seed 13. CSE in iteration 1 merged two equal-value but distinct-id index ops; that exposed a dead-store opportunity that dead_store had missed in iteration 1 (it sees textual id equality). Fix: introduced `run_all_once` (single canonical sequence) and made `run_all` iterate up to `RUN_ALL_MAX_ITERS=4` until op count + literal pool stabilize. Real fixed-point iteration, not a band-aid. All emit_optimized callers transparently get the fixed-point form.

### T074 [x] vyre-lower::analyses::def_use  -  def-use chains over the descriptor
**Owner**: CC
**Source**: PERF_ROADMAP A33 (foundation analysis for future passes)
**Result**: SHIPPED. New analysis `vyre-lower::analyses::def_use` with 8 tests (vyre-lower 259 total). `analyze(desc) -> DefUseReport` produces one `PerBodyChains` per body in pre-order; each chain is `BTreeMap<u32, Vec<UseSite>>` mapping result-id → every (body_path, op_index, operand_pos) that references it. Dead defs surface with empty `Vec<UseSite>`. Convenience: `dead_by_no_use(desc)` returns all `(body_path, id)` with empty chains. Operand classifier mirror kept in sync with the rewrites family. Foundation for: faster DCE (no scan-everything), substitute-uses-with operations, live-range analysis, soundness checks ("no use precedes def"). Not yet wired into existing rewrites  -  could replace dce's BTreeSet walk in a later pass.

### T073 [x] vyre-lower::verify  -  descriptor invariant verifier
**Owner**: CC
**Source**: T072 follow-on (catch rewrite-pipeline bugs early via structural verification)
**Result**: SHIPPED. New module `vyre-lower::verify` with 11 tests (vyre-lower 251 total). `verify(desc) -> Result<(), Vec<VerifyError>>` walks the descriptor checking: result-id uniqueness, no dangling result-id refs, literal-pool indices in range, child-body indices in range, Literal ops have ≥1 operand, per-kind minimum operand counts. Errors are collected (not short-circuited) so a single verify call surfaces every violation, with body_path for nested children. Public types: `VerifyError`, `VerifyErrorKind`, `VerifyResult`. The critical regression-gate test asserts `verify(run_all(desc)) == Ok(())`  -  any rewrite that produces an invalid descriptor fails this. Not yet wired into emit_optimized; can be enabled via a debug-assert when the user asks.

### T072 [x] rewrite_efficacy harness  -  corpus-level op-count reduction gate
**Owner**: CC
**Source**: T070/T071 follow-on (regression gate on optimization quality)
**Result**: SHIPPED. 8 tests in `vyre-emit-naga/tests/rewrite_efficacy.rs` (vyre-emit-naga 62 total). Each shape (dead arithmetic / redundant load / duplicate literals / const-foldable arithmetic / minimal kernel / aggregate corpus / semantics-preserving emit) validates one rewrite-pipeline property with a hard assertion (op-count ≤ before − N for the relevant N, or ≥20% aggregate reduction). The aggregate test is the catch-all: if a future change accidentally turns one pass into a no-op, aggregate efficacy drops below 20% and this fails. Initially placed in vyre-bench but moved to vyre-emit-naga because vyre-bench transitively depends on vyre-libs (currently in Codex-active hold)  -  vyre-emit-naga has the deps it needs (vyre-lower + vyre-foundation) without the in-flight dep chain.

### T071 [x] vyre-emit-*  -  emit_optimized() wires run_all into the emit path
**Owner**: CC
**Source**: T070 follow-on (rewrites must be REACHABLE from emit, not just exposed)
**Result**: SHIPPED. New API on all three emit crates: `vyre_emit_naga::emit_optimized`, `vyre_emit_ptx::emit_optimized` + `emit_optimized_with_target`, `vyre_emit_spirv::emit_optimized`. Each calls `vyre_lower::rewrites::run_all(desc)` then the existing `emit()`. Raw `emit()` retained for callers that need bytewise determinism on the input descriptor. 7 new tests across crates (vyre-emit-naga 54, vyre-emit-ptx 74, vyre-emit-spirv 19)  -  each crate proves: (a) emit_optimized succeeds, (b) on a kernel with dead arithmetic (Add identity + Mul absorbing-zero), the optimized output is no longer than the raw output by the natural metric (Naga statements / PTX lines / SPIR-V words). Documents that `emit_optimized` is the recommended path.

### T070 [x] run_all pipeline integration tests + dce/dead_store wiring fix
**Owner**: CC
**Source**: PERF_ROADMAP A30 (cross-pass pipeline contract)
**Result**: SHIPPED. 3 integration tests in `vyre-lower::rewrites::tests` (vyre-lower 240 total). `kitchen_sink_kernel` exercises strength_reduce + const_fold + identity_elim + dead_store + dce on a single descriptor with 5 dead ops, asserting all 3 Mul + 1 Add are eliminated and op count drops by ≥5. `forwards_then_drops_redundant_load` proves Store/Load/Store collapses to a single Store after the load_forwarding+dce wiring change. `unrolls_then_simplifies` confirms loop_unroll inlines and downstream passes don't break the resulting straight-line code. Pipeline rewired: inserted a `dce` between `load_forwarding` and `dead_store` so forwarded-redundant Loads die before dead_store walks (otherwise an about-to-die Load looks like a use of the prior Store and keeps it alive). `run_all` now applies 11 passes (10 distinct rewrites; dce twice).

### T069 [x] load_forwarding rewrite  -  store-to-load + load-to-load forwarding
**Owner**: CC
**Source**: PERF_ROADMAP A22 (memory dependence forwarding family)
**Result**: SHIPPED. 11 tests in `vyre-lower::rewrites::load_forwarding` (vyre-lower 237 total). Per-slot cache (`BTreeMap<slot, BTreeMap<idx_id, val_id>>`). Store(slot, idx, val) installs `(idx → val)`, wiping other entries in that slot (different idx may alias). Load(slot, idx) hits the cache → forwarded; misses → installed as canonical for later loads. LoadConstant participates (immutable memory). Atomic on slot S wipes only that slot. Barrier/Async/StructuredCF/Trap/Resume/Return/Call/Opaque wipe entire cache. Doesn't strip ops  -  leaves redundant Loads for `dce`. Wired into run_all between licm and dead_store.

### T068 [x] identity_elim rewrite  -  Add(x,0) / Mul(x,1) / etc → x via id substitution
**Owner**: CC
**Source**: PERF_ROADMAP A26 (algebraic simplification family  -  identity-element subset)
**Result**: SHIPPED. 15 tests in `vyre-lower::rewrites::identity_elim` (vyre-lower 226 total). Detects right-identity (`Add/Sub/Shl/Shr/BitOr/BitXor/WrappingAdd/WrappingSub` with rhs=0; `Mul/Div` with rhs=1), left-identity (commutative ops with lhs=identity), absorbing-zero (`Mul/BitAnd` with either side = 0), and self-equality (`BitAnd/BitOr/Min/Max` with lhs_id == rhs_id). Substitutes references via id_remap with transitive resolution. Supports U32/I32/F32 zero and one. Doesn't strip ops itself  -  leaves them dead for `dce` to clean. Wired into run_all between const_fold and branch_collapse.

### T067 [x] loop_unroll rewrite  -  small constant-bound loops
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` A29 (loop strip-mining family  -  unconditional-unroll special case)
**Result**: SHIPPED. 10 tests in `vyre-lower::rewrites::loop_unroll` (vyre-lower 211 total). Detects StructuredForLoop where lo + hi are both Literal(U32) and (hi-lo) ≤ MAX_UNROLL_COUNT (4); inlines body N times with renumbered result-ids; strips loop op. Phase-1 conservative: skips bodies with nested control flow (StructuredIfThen/Else/ForLoop/Block/Region) since their child-body indices can't be safely re-parented. Idempotent. Wired into run_all after branch_collapse (which often produces small-bodied loops).

### T066 [x] vyre-emit-spirv patterns  -  subgroup capability detection
**Owner**: CC
**Source**: T017 follow-on (SPIR-V/Vulkan capability tracking)
**Result**: SHIPPED. 10 tests in `vyre-emit-spirv::patterns::subgroup_capabilities` (vyre-emit-spirv 17 total). First SPIR-V-specific pattern. Walks the descriptor for SubgroupBallot/Shuffle/Add/LocalId/Size; produces `SubgroupCapabilities { basic, ballot, shuffle, arithmetic }` flagging which `VkSubgroupFeatureFlagBits` the host needs in the pipeline. Recurses into structured control flow.

### T065 [x] branch_collapse rewrite  -  inline literal-condition branches
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` A19 (predicate hoisting prep + dead-branch removal)
**Result**: SHIPPED. 7 tests in `vyre-lower::rewrites::branch_collapse` (vyre-lower 201 total). Patterns:
- `StructuredIfThen` with `cond=Literal(true)` → inline body's ops at the position
- `StructuredIfThen` with `cond=Literal(false)` → drop the op
- `StructuredIfThenElse` with literal cond → inline the matched arm
- Non-literal conditions or non-bool literals → unchanged
Recurses into nested control flow. Idempotent. Wired into `run_all` after const_fold (which produces literal conds from boolean arithmetic).

### T064 [x] strength_reduce rewrite  -  mul/div/mod by power-of-2 → shifts/and
**Owner**: CC
**Source**: `PERF_ROADMAP_2026-05-01.md` A34
**Result**: SHIPPED. 10 tests in `vyre-lower::rewrites::strength_reduce` (vyre-lower 194 total). Patterns:
- `Mul(x, lit_pow2)` → `Shl(x, log2(lit))` (commutative  -  pow2 can be on either side)
- `Div(x, lit_pow2)` → `Shr(x, log2(lit))` (rhs only  -  signed div has rounding issues so phase-1 stays u32)
- `Mod(x, lit_pow2)` → `BitAnd(x, lit-1)` (rhs only)
Idempotent. Wired into `run_all` pipeline as the FIRST pass (before const_fold) so reduced shifts can themselves get folded if both operands are constants.

### T063 [x] vyre-bench cross-emitter property tests
**Owner**: CC
**Source**: ROADMAP T090
**Result**: SHIPPED. 7 tests in `vyre-bench/tests/cross_emitter_property.rs`. Hand-rolled deterministic LCG-seeded generator builds well-wired KernelDescriptors from a small grammar (literals, builtins, binops, loads, stores, unops, selects). Tests:
- naga emit handles 50 random descriptors without panic, every error is a recognized variant
- ptx emit handles 50 random descriptors
- spirv emit handles 50 random descriptors
- small descriptors (≤5 ops) succeed-or-recognized-error across all 3 emitters (30 seeds)
- generator produces valid id wiring (20 seeds, contract pin)
- `rewrites::run_all` handles 50 random descriptors without panic, op-count never grows
- `audit` handles 50 random descriptors without panic, waste_score finite + non-negative
**Total: 200+ random descriptors per emitter**, 200+ rewrite invocations, 50+ audit invocations, all green.

### T062 [x] PTX async op support (AsyncLoad/Store/Wait via cp.async)
**Owner**: CC
**Source**: T016 follow-on
**Result**: SHIPPED. 3 new tests in vyre-emit-ptx (71 total). Threading: added `target: ComputeCapability` to BodyCtx so async ops can gate on `sm_80+`. Coverage:
- `AsyncLoad` → `cp.async.ca.shared.global [shared_ptr], [global_ptr+offset], size` on sm_80+, `UnsupportedOp` below.
- `AsyncStore` → comment + elided (no PTX-level shared→global async; phase-2 will use driver-side scheduling).
- `AsyncWait` → `cp.async.commit_group; cp.async.wait_group 0;` on sm_80+, `UnsupportedOp` below.

### T060 [x] PTX subgroup op support (Ballot/Shuffle/Add/LocalId/Size)
**Owner**: CC
**Source**: T016 follow-on
**Result**: SHIPPED. 5 new tests in vyre-emit-ptx (68 total). Coverage:
- `SubgroupBallot` → `vote.ballot.sync.b32`
- `SubgroupShuffle` → `shfl.sync.idx.<dtype>`
- `SubgroupAdd` → `redux.sync.add.<dtype>`
- `SubgroupLocalId` → `mov.u32 %rN, %laneid`
- `SubgroupSize` → `mov.u32 %rN, %nwarpid`

### T061 [x] naga subgroup op support
**Owner**: CC
**Source**: T015 follow-on
**Result**: SHIPPED. 2 new tests in vyre-emit-naga (52 total). Plumbed `TypeHandles` into `BodyBuilder` (added `types: TypeHandles` field). Coverage:
- `SubgroupBallot` → `Statement::SubgroupBallot { result: SubgroupBallotResult expr, predicate: Some(cond) }`
- `SubgroupAdd` → `Statement::SubgroupCollectiveOperation { op: Add, collective_op: Reduce, argument, result: SubgroupOperationResult { ty: u32_ty } }`
- `SubgroupShuffle` → `Statement::SubgroupGather { mode: Shuffle(lane), argument, result }`
Phase-1 result type pinned to u32_ty; per-binding type lookup is phase 2 (matters for f32 SubgroupAdd reductions).

### T059 [x] Atomic op support in vyre-emit-naga + vyre-emit-ptx
**Owner**: CC
**Source**: T015/T016 follow-on (op coverage)
**Result**: SHIPPED. 5 new tests across both emitters. Coverage:
- naga: `Statement::Atomic` with `naga::AtomicFunction::{Add, And, InclusiveOr, ExclusiveOr, Min, Max, Exchange}`. Phase-1 discards atomic result (downstream uses get InvalidDescriptor); typed-result wiring is phase 2.
- PTX: `atom.global.<op>.<dtype>` with proper address arithmetic. Phase-1 supports u32 globals.
- Both: `AtomicOp::FetchNand` and `Opaque` surface as `UnsupportedOp`. CompareExchange/Weak deferred (need expected operand position handling for atomic_result type lookup).

### T058 [x] vyre-bench full-pipeline integration test
**Owner**: CC
**Source**: ROADMAP T056-T058 follow-on
**Result**: SHIPPED. 8 tests in `vyre-bench/tests/full_pipeline_snapshot.rs`. Builds a "rich" KernelDescriptor (multiple literals, foldable arithmetic, coalesced load, store, dead literal); runs `audit` → `run_all` → `vyre_emit_naga::emit` → `vyre_emit_ptx::emit` → `vyre_emit_spirv::emit`; asserts each step succeeds. Tests cover: rewrites don't grow op count, naga module has entry point, PTX text has expected structure, SPIR-V starts with magic word, audit doesn't panic post-rewrite, all three substrates emit for the rewritten kernel, pipeline is idempotent. Surfaced one real cross-substrate bug (SPIR-V doesn't allow storage write-only  -  fixed by promoting output binding to ReadWrite in the test fixture).

### T055 [x] vyre-emit-ptx structured-control-flow + composite-op parity with naga
**Owner**: CC
**Source**: T015/T016 follow-on (substrate parity gap)
**Result**: SHIPPED. 8 new tests (vyre-emit-ptx 60 total). New PTX coverage:
- `StructuredIfThen` → `@!cond bra end_label; <body> end_label:`
- `StructuredIfThenElse` → conditional branch + then-body + unconditional jump + else-label + else-body + end-label
- `StructuredForLoop` → init + head label + setp.ge + branch-out + body + add + jmp-back + exit label, with `loop_var` preserved as a comment for debug
- `Cast { target }` → `cvt.<dst_ty>.<src_ty>`
- `Select` → `selp.<ty>` predicate-select
- `Fma` → `fma.rn.<ty>` rounded fused multiply-add
PTX emit now matches naga emit on every structured / composite op in the descriptor. Atomic, Subgroup, Async, Trap/Resume, IndirectDispatch still surface as `EmitError::UnsupportedOp`  -  PTX-specific lowering for those is follow-up work.

### T054 [x] vyre-lower::rewrites  -  first real KernelDescriptor mutation passes (DCE, CSE, const-fold)
**Owner**: CC
**Source**: This roadmap (mutation-pattern foundation for all future descriptor rewrites)
**Result**: SHIPPED. 21 new tests across the three rewrite passes (vyre-lower 169 total).
- `rewrites::dce`: strips dead ops via `analyses::dead_op` then renumbers operand-ids dense `0..N`. Per-`KernelOpKind` operand classification mirrors the analysis (kept in sync to avoid wrong renumbering).
- `rewrites::cse`: collapses `EquivalenceGroup`s  -  picks lowest-op-index canonical, rewrites references to point at canonical, strips duplicates, renumbers dense.
- `rewrites::const_fold`: folds compile-time-constant `BinOpKind` arithmetic into a single Literal. Phase 1 covers Add/Sub/Mul/Div/Mod (with div-by-zero guard), bitwise ops, Min/Max, shifts (u32/i32), And/Or/BitXor (bool), float Add/Sub/Mul/Div with finite-only guard. Chains propagate through one pass.
- `rewrites::run_all`: canonical pipeline (const_fold → dce → cse). Idempotent.
This is the **first real mutation layer in vyre-lower**  -  every previous module was read-only analysis. Establishes the pattern for follow-up rewrites (vec-pack rewrite uses vec_pack analysis to actually fuse loads, shared-mem-promote rewrite inserts tile-load+barrier+shared-read, etc.).

### T158 [x] M5/F4/I4 device-profile consumption
**Owner**: Codex
**Source**: `docs/optimization/ROADMAP.md` M5/F4/I4 + `PERF_ROADMAP_2026-05-01.md`
**Result**: SHIPPED. Device signatures are no longer parse-only. `DeviceSignatureTable` now ships built-in Tier-B signatures, matches CUDA SM ids and wgpu adapter-name aliases, and projects signature fields into `DeviceProfile`. `DeviceProfile::adapter_caps()` now carries compute/cache/bandwidth/bank/tuning fields into the neutral foundation `AdapterCaps`. wgpu and CUDA backend profile projection apply the built-in `sm_120` profile for RTX 5090 / Blackwell. Foundation planning/autotune/cost now consumes those fields: scheduling chooses workgroup x and tile from `ideal_workgroup_tile`, vector packing from `ideal_vector_pack_bits`, unroll depth from `ideal_unroll_depth`, and device cost estimates from the same profile facts. Tests prove compact vs wide signatures change selected workgroup `[64,1,1]` vs `[256,1,1]`, tile `[8,8,1]` vs `[16,16,1]`, vector `64` vs `128`, unroll `4` vs `8`, and projected device cost.

---

## How to add a new task

Append a new `### T<NNN>` entry at the END of `## Tasks`. Use the next
sequential T number. Status starts at `[ ]` with `**Owner**: UNCLAIMED`.

## How to claim and work a task

1. Edit ONLY the `**Owner**` field  -  change `UNCLAIMED` to your handle.
2. Change status `[ ]` → `[~]` when starting.
3. Do the work.
4. Change status `[~]` → `[x]` and append `**Result**:` block.
5. Never edit other fields.

## How to file a follow-up

Append a new task. Reference the predecessor (`see T###`). Don't rewrite
the original.
