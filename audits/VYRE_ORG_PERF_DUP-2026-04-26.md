# VYRE_ORG_PERF_DUP  -  Organization / Performance / Duplication Audit

**Scope:**
- `vyre-runtime/src/megakernel/**`
- `vyre-runtime megakernel/**`
- `vyre-driver-wgpu/**`
- `vyre-libs/src/parsing/**`
- `vyre-libs/src/visual/**`
- `docs/audits/plans` under `libs/performance/matching/vyre/`

**Date:** 2026-04-26
**Auditor:** Kimi Code CLI
**Standard:** LAWS 0–8, STANDARDS, RESEARCH PROTOCOL

---

## 1. Duplicated Policy

### 1.1 `vyre-runtime/src/megakernel/builder.rs:239-284` ↔ `builder.rs:407-452`
**Status:** Fixed in `vyre-runtime/src/megakernel/builder.rs` by extracting `persistent_lane_prologue(...)`.
**Issue:** `persistent_body_with_io` and `persistent_body_jit` copy-paste the exact same shutdown flag, `lane_id`, `slot_base`, tenant-base load, tenant-mask load, and outer `if_then` wrapper.
**Fix:** Extract a `persistent_prologue(workgroup_size_x: u32) -> Vec<Node>` helper that emits the common bindings; call it from both body builders.

### 1.2 `vyre-runtime/src/megakernel/builder.rs:22-24` ↔ `handlers.rs:13-15` ↔ `scheduler.rs:26-28` ↔ `dispatcher.rs:20-22`
**Status:** Fixed in `vyre-runtime/src/megakernel/ir_util.rs`.
**Issue:** `atomic_load_relaxed` is defined identically in four separate files.
**Fix:** Extract to a shared `crate::megakernel::ir_util::atomic_load_relaxed` helper and delete the four local copies.

### 1.3 `vyre-driver-wgpu/src/pipeline_persistent.rs:167-196` ↔ `vyre-driver-wgpu/src/runtime/prerecorded.rs:250-274`
**Status:** Fixed in `vyre-driver-wgpu/src/pipeline_binding.rs` via `clear_outputs_for_bound(...)`.
**Issue:** `clear_outputs` iterates bound handles, skips `preserve_input_contents`, and computes `word_count * 4` with identical logic in both files.
**Fix:** Move to a shared `pipeline::clear_outputs_for_bound()` helper consumed by both modules.

### 1.4 `vyre-driver-wgpu/src/pipeline_persistent.rs:347-377` ↔ `vyre-driver-wgpu/src/runtime/prerecorded.rs:276-306`
**Status:** Fixed in `vyre-driver-wgpu/src/pipeline_binding.rs` via `validate_handle(...)` and `usage_for_binding(...)`.
**Issue:** `validate_handle` and `usage_for_binding` are exact duplicates across persistent and prerecorded pipeline modules.
**Fix:** Collapse into one `validate_handle()` and one `usage_for_binding()` in a shared `pipeline/binding.rs` module.

### 1.5 `vyre-driver-wgpu/src/async_dispatch.rs:80-115` ↔ `vyre-driver-wgpu/src/engine/streaming.rs:54-97`
**Status:** Fixed by adding `vyre-driver-wgpu/src/thread_pool.rs` with one bounded crossbeam worker-pool implementation shared by async dispatch and host-ingress streaming.
**Issue:** `AsyncDispatchPool::new()` and `StreamingPool::new()` duplicate the same crossbeam-channel worker-pool setup (bounded queue, `available_parallelism().clamp(1,32)`, thread spawning, `catch_unwind`, error channel send).
**Fix:** Create a single `crate::thread_pool::BoundedWorkerPool<T>` generic and reuse it in both places.

### 1.6 `vyre-libs/src/visual/*/mod.rs`  -  `to_bytes` closure
**Status:** Fixed by adding `visual::harness::u32_words_to_le_bytes(...)` and using it across visual inventory entries.
**Issue:** The closure `|w: &[u32]| w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()` is copy-pasted identically in every visual harness block (blur, composite, downsample, filter_chain, glass, gradient, shadow  -  12 occurrences total).
**Fix:** Define once in `visual::harness` or `vyre_libs::test_util` and import it.

### 1.7 `vyre-libs/src/parsing/c/lex/lexer.rs:5-119` ↔ `go/lex.rs:37-64` ↔ `python/lex.rs:65-99`
**Issue:** `byte_load`, `byte_eq`, `byte_between`, `is_alpha`, `is_ident_start`, `is_ident_continue` are reimplemented with minor naming differences in every frontend lexer.
**Fix:** Extract to `parsing::core::lex_ir_helpers` (or `vyre_primitives::gpu_lex_utils`) and share across frontends.

---

## 2. Wrong Workgroup / Grid Semantics

### 2.1 `vyre-driver-wgpu/src/pipeline.rs:316-322` / `async_dispatch.rs:244-254` / `pipeline_compound.rs:65-71`
**Status:** Fixed by preserving `workgroup_shape` in `WgpuPipeline` and rejecting derived default dispatch for non-1D workgroups unless `DispatchConfig.grid_override` is supplied.
**Issue:** `workgroup_size` is flattened to a scalar product (`effective_wg[0] * effective_wg[1] * effective_wg[2]`) and the dispatch grid is always `[count, 1, 1]`. A shader with `@workgroup_size(8, 8, 1)` that uses `global_invocation_id.y` for spatial indexing will miscompute addresses or write out of bounds.
**Fix:** Store the original `[u32; 3]` workgroup size and, when no `grid_override` is set, compute per-axis dispatch counts that preserve the program's intended grid shape.

### 2.2 `vyre-driver-wgpu/src/runtime/workgroup_size.rs:1-2`
**Status:** Fixed by renaming the exported constant to `DEFAULT_1D_WORKGROUP_SIZE`.
**Issue:** Hardcodes `WORKGROUP_SIZE = [64, 1, 1]`. Multiple modules `pub use` this constant, but it is silently overridden by the tuner and by per-program `workgroup_size`. The constant creates a false assumption that all kernels are 1D×64.
**Fix:** Rename to `DEFAULT_1D_WORKGROUP_SIZE` and audit all call sites to ensure they don't assume 1D when the program is 2D/3D.

### 2.3 `vyre-runtime/src/megakernel/c_frontend.rs:598`
**Issue:** Control-plane bootstrap uses `Expr::eq(Expr::gid_x(), Expr::u32(0))` to single-thread logic, but the rest of the megakernel manually computes global index as `workgroup_x * workgroup_size_x + local_x`. If `gid_x()` maps to `workgroup_id.x`, multiple threads (one per workgroup) will race the control plane.
**Fix:** Replace with the explicit `eq(add(mul(workgroup_x(), u32(workgroup_size_x)), local_x()), u32(0))` pattern used everywhere else, or document that `gid_x()` MUST resolve to `global_invocation_id.x`.

### 2.4 `vyre-runtime/src/megakernel/builder.rs:539`
**Status:** Fixed in `persistent_body_priority(...)` by binding `lane_id` before `priority_scan_body(...)`.
**Issue:** `persistent_body_priority` extends `scheduler::priority_scan_body(lane_count)` but never binds the `lane_id` variable that `priority_scan_body` requires in scope.
**Fix:** Insert the standard `lane_id` binding (`workgroup_x * workgroup_size_x + local_x`) into `persistent_body_priority` before the `body.extend(priority_scan_body(...))` call.

### 2.5 `vyre-libs/src/visual/blur/mod.rs:250-261`
**Status:** Fixed by making `gaussian_blur_2pass(...)` return explicit horizontal and vertical `Program` stages instead of one barrier-separated dispatch.
**Issue:** Two dependent global passes (horizontal blur → `scratch`, vertical blur → `output`) are packed into one `Program` separated by `Node::Barrier`. Standard GPU compute cannot synchronize across workgroups inside a single dispatch; the second pass may read `scratch` values that have not been written yet by other workgroups.
**Fix:** Split into two separate `Program`s (or two dispatches): `blur_h_program(input, scratch)` and `blur_v_program(scratch, output)`.

### 2.6 `vyre-libs/src/parsing/c/lex/lexer.rs:780` / `c/preprocess/expansion.rs:690` / `c/parse/structure.rs:148` / `python/lex.rs:178` / `go/lex.rs:92`
**Issue:** Launch config is `[256,1,1]` but `haystack_len`/`num_tokens` can be 4096 or larger. `InvocationId` only indexes 0..255; if the runtime does not automatically scale the grid, the remaining bytes are never processed.
**Fix:** Either switch to a loop-stride pattern (`for pos in (t..haystack_len).step_by(256)`) or document that callers **must** dispatch `ceil_div(haystack_len, 256)` workgroups.

---

## 3. Avoidable Allocations / Copies

### 3.1 `vyre-runtime/src/megakernel/descriptor.rs:148-152`
**Status:** Fixed by making `Megakernel::publish_packed_slot(...)` generic over borrowed arg containers and publishing `PackedOpDescriptor` args as slices.
**Issue:** `publish_into` clones every `op.args` Vec when building the packed tuple vec: `ops.iter().map(|op| (op.opcode, op.args.clone()))`.
**Fix:** Change `Megakernel::publish_packed_slot` to accept `&[(u8, &[u32])]` so the descriptor can lend its args without cloning.

### 3.2 `vyre-runtime/src/megakernel/c_frontend.rs:645`
**Issue:** `c_frontend_phase_dispatch_nodes` clones the entire `handler.body` `Vec<Node>` for every handler on every manifest compilation.
**Fix:** Store `body: Arc<[Node]>` (or `Arc<Vec<Node>>`) in `CFrontendPhaseHandler` so the clone becomes a cheap Arc bump.

### 3.3 `vyre-driver-wgpu/src/lib.rs:349-356`
**Status:** Fixed by storing the persistent pool in `ArcSwap<BufferPool>` and loading it without an `RwLock`.
**Issue:** `current_persistent_pool` takes an `RwLock` read lock **only** to clone an `Arc<PoolInner>` inside `BufferPool`. The lock is pure overhead because `BufferPool` is already `Arc`-backed.
**Fix:** Change `WgpuBackend.persistent_pool` to `Arc<BufferPool>` (or `Arc<ArcSwap<BufferPool>>` if recovery needs to swap it), eliminating the lock entirely.

### 3.4 `vyre-driver-wgpu/src/engine/graph.rs:86-88`
**Status:** Fixed by adding an internal borrowed `CompoundResource<'_>` path so graph execution, coalesced dispatch, and legacy compound shims can lend host inputs without cloning into owned `Resource` values.
**Issue:** `GpuResource::Borrowed(bytes) => Resource::Borrowed(bytes.clone())` clones every input buffer even though `dispatch_compound_v2` only borrows the `Resource` values.
**Fix:** Change `Resource::Borrowed` to `Resource::Borrowed(&[u8])` (or `Cow<[u8]>`) so the graph can lend its existing `Vec<u8>` without cloning.

### 3.5 `vyre-driver-wgpu/src/buffer/handle.rs:428-450`
**Status:** Fixed by replacing the process-wide `Mutex<FxHashMap<...>>` resident registry with `DashMap`.
**Issue:** Every buffer acquisition/destruction locks a **global** `Mutex<FxHashMap<u64, Weak<GpuBufferInner>>>` (`resident_buffers()`). Under high concurrency this serializes all pool operations.
**Fix:** Replace the global mutex with `dashmap::DashMap` or shard the registry by id bits.

### 3.6 `vyre-libs/src/parsing/c/preprocess/mod.rs:44`
**Issue:** `c_translation_phase_line_splice` always allocates two new `Vec`s even when the source contains zero backslash-newline pairs (the common case).
**Fix:** Return a `Cow<'_, [u8]>`-like enum, or return borrowed slices when no splices exist.

### 3.7 `vyre-libs/src/visual/blur/mod.rs:49`
**Status:** Fixed by adding `GaussianKernel` and `gaussian_blur_2pass_with_kernel`, so hot paths can precompute one fixed-point kernel and reuse it across stage builds.
**Issue:** `gaussian_weights(clamped, sigma)` returns a freshly allocated `Vec<u32>` every time `gaussian_blur_2pass` is called. If the effect is rebuilt per frame (e.g. animated sigma), this allocates on the CPU every frame.
**Fix:** Accept `weights: &[u32]` as a parameter, or add a cached builder that reuses the weight vector.

---

## 4. Fake Capability Wording

### 4.1 `vyre-runtime/src/megakernel/io.rs:3-7`
**Status:** Fixed by aligning the module docs with the landed protocol plus Linux `uring` ingest drivers: the megakernel owns the queue protocol, while runtime drivers own concrete registered-mapped or GPUDirect NVMe reads.
**Claim:** "services requests via io_uring (Linux) or standard file I/O (portable)" and "eliminates CPU bounce buffers  -  NVMe bytes land directly in GPU-visible memory."
**Reality:** The file only defines in-memory buffer protocols and a `MegakernelIoQueue` struct; there is no io_uring code, no pump thread, and no NVMe DMA implementation.
**Fix:** Rewrite doc to describe the protocol only: "Host-side protocol definitions for a GPU↔Host DMA request queue. Actual io_uring/NVMe integration must be provided by the runtime scheduler."

### 4.2 `vyre-runtime/src/megakernel/advanced/zero_copy_io.rs:1-4`
**Status:** Fixed by rewriting the docs around the actual `AsyncLoad` IR fragment and the runtime scheduler/driver responsibility boundary.
**Claim:** "Zero-Copy NVMe Integration via AsyncLoad" and "Bridges the wyre megakernel to io_uring completely bypassing host memory round-tripping."
**Reality:** The module only emits a single `AsyncLoad` IR node with string bindings; there is no NVMe bridge, no io_uring, and no runtime DMA verification.
**Fix:** Rewrite doc to match reality: "IR fragment that emits a GPU-side `AsyncLoad` node. Backend schedulers must map the capability-table strings to actual DMA sources/destinations."

### 4.3 `vyre-driver-wgpu/src/lib.rs:740-745`
**Status:** Fixed in `WgpuBackend::supports_async_compute()`; it now returns `false` while preserving real host-side async dispatch through `dispatch_async`.
**Claim:** `supports_async_compute` returns `true` claiming async compute is supported because `dispatch_async` "overlaps host-side staging/readback with GPU execution."
**Reality:** wgpu uses a **single universal queue**; there is no GPU async compute queue. Host-side threading is not GPU async compute.
**Fix:** Return `false` and document that host-side threading is not the same as GPU async compute.

### 4.4 `vyre-driver-wgpu/src/engine/multi_gpu.rs:1`
**Claim:** Module is marketed as "multi-GPU work partitioning."
**Reality:** The doc itself admits it is **"mockable host-side scheduling only"** and does not probe adapters or submit work. `WgpuBackend::is_distributed` returns `false`.
**Fix:** Rename module to `work_partitioner` or `mock_partitioner` and remove "multi-GPU" from user-facing docs.

### 4.5 `vyre-driver-wgpu/README.md:28`
**Status:** Fixed; the README now describes a validation cache without the false "three-level" cache claim, while code keeps distinct structural validation, backend capability validation, and the backend-local cache.
**Claim:** "Tiered validation cache  -  three-level cache for repeated shader validation."
**Reality:** `WgpuBackend` only has `validation_cache: Arc<dashmap::DashSet<blake3::Hash>>`  -  a **single** hash set, not three levels.
**Fix:** Delete the "three-level" claim or implement the missing tiers.

### 4.6 `vyre-libs/src/parsing/c/lex/mod.rs:1` / `c/lex/lexer.rs:8`
**Claim:** "Maximally-munching DFA-driven lexer kernel" and "GPU DFA lexer pipeline: classifier table, lexer kernel, token constants, keyword recogniser."
**Reality:** The implementation is a 900-line hand-written sequential `if_then` chain, not a DFA table. No classifier table exists.
**Fix:** Rewrite doc to "Sequential SIMT byte classifier" or generate an actual DFA transition table.

### 4.7 `vyre-libs/src/parsing/core/delimiter.rs:55-58`
**Claim:** "Implements a subgroup parallel scan algorithm to trace depth transitions natively without warp divergence."
**Reality:** The code performs an O(n) serial prefix walk per lane (lines 78–100). There is no subgroup scan.
**Fix:** Rewrite doc to "Per-lane inclusive prefix depth scan" or implement the actual subgroup scan.

### 4.8 `libs/performance/matching/vyre/audits/RELEASE_1000X_PLAN.md:39`
**Claim:** "G5 decode-scan fused  -  no HBM round-trip for decoded bytes" and "inflate/lz4 use a cooperative-thread-block literal-copy kernel."
**Reality:** The same plan notes at line 145: "CRITICAL inflate traps on BTYPE=1/2/3." Only stored-block inflate works; decode→scan fusion is not actually landed.
**Fix:** Remove G5 from the "landed" list until BTYPE=1/2/3 is implemented and the fusion pass is wired in production.

---

## 5. Stale Docs

### 5.1 `vyre-driver-wgpu/README.md:29`
**Status:** Fixed; the README now names `lowering::lower_with_features` as the internal lowering entry.
**Issue:** References `vyre::lower::wgsl::emit` as the lowering path. No such module or function exists; the actual entry is `crate::lowering::lower_with_features`.
**Fix:** Update to the correct lowering path.

### 5.2 `vyre-driver-wgpu/PUBLIC_API.md`
**Issue:** Auto-generated file (5,267 lines) that lists every blanket trait implementation from dependencies (`crossbeam_epoch::Pointable`, `khronos_egl::Downcast`, `wgpu_types::send_sync::WasmNotSend`, etc.). These are not part of the crate's public API contract and mislead consumers.
**Fix:** Regenerate from a filtered `cargo rustdoc` pass, or replace with a hand-curated API summary.

### 5.3 `vyre-driver-wgpu/src/engine.rs:6-11`
**Status:** Fixed by replacing the stale release-roadmap paragraph with the current backend-execution ownership boundary.
**Issue:** Claims "Dialect-specific engines (dataflow, decode, decompress, dfa, string matching) were removed alongside the WGSL-string dialects they rode on; they return in 0.7 against the naga-AST emitter." This roadmap statement is outdated and references removed modules that no longer exist.
**Fix:** Remove the stale 0.6/0.7 roadmap language.

### 5.4 `vyre-runtime/src/megakernel/handlers.rs:23-26`
**Status:** Fixed by extracting `claimed_slot_bindings()` and using it in both interpreted and JIT claimed-slot paths, so `opcode` and `arg0..arg2` are actually in scope before payload handlers run.
**Issue:** Doc for `OpcodeHandler` claims `arg0..arg2` are always in scope. In JIT mode (`persistent_body_jit` / `claimed_slot_body_jit` in `builder.rs`), these arguments are never bound before the payload processor is spliced, so custom handlers referencing `arg0` will fail in JIT builds.
**Fix:** Bind `opcode` and `arg0..arg2` in the JIT claimed-slot path before splicing payload processor nodes.

### 5.5 `vyre-runtime/src/megakernel/mod.rs:215-221`
**Status:** Obsolete; `vyre-runtime/src/uring/driver.rs` now contains the Linux ingest driver referenced by the doc comment.
**Issue:** `dispatch_with_io_queue` doc mentions "the Linux `uring` ingest driver". No such driver exists in this codebase.
**Fix:** Remove the non-existent `uring` reference from the doc comment.

### 5.6 `libs/performance/matching/vyre/audits/V7_api.toml:22` / `V7_api.toml:70`
**Status:** Fixed by updating V7-API-002 and V7-API-008 to `vyre-intrinsics` paths and verified current line anchors.
**Issue:** References `vyre-ops/src/lib.rs` and `vyre-ops/Cargo.toml`. The crate `vyre-ops` was renamed to `vyre-intrinsics`; these paths no longer exist.
**Fix:** Update all `vyre-ops` paths to `vyre-intrinsics` and verify line numbers against the new crate.

---

## 6. Wildcard API Surfaces

### 6.1 `vyre-runtime/src/megakernel/mod.rs:49-101`
**Issue:** The module re-exports ~80 items from submodules via massive `pub use` blocks, flattening internal ABI constants (`C_FRONTEND_*_WORDS`, `SLOT_WORDS`, `ARG0_WORD`, etc.) into the public API surface. Callers cannot distinguish stable API from internal wire-format details.
**Fix:** Split re-exports into `pub use` (stable public API) and `pub(crate) use` (internal ABI). Move constants like `C_FRONTEND_*_WORDS` to `pub(crate)` since callers should not depend on raw word offsets.

### 6.2 `vyre-runtime megakernel/src/lib.rs:9-22`
**Status:** Fixed by moving task slot/flag constants under `vyre_runtime::megakernel::raw_abi` while keeping typed task and policy types at the crate root.
**Issue:** Re-exports every public item from `core`, `policy`, and `task` with `pub use`, leaking internal constants (`TASK_FLAG_PAUSED`, `TASK_SLOT_WORDS`, etc.) into the crate root.
**Fix:** Narrow the re-exports: hide flag constants and slot-size internals behind `task` module privacy, or expose them via a `raw_abi` submodule.

### 6.3 `vyre-driver-wgpu/src/lib.rs:39-71`
**Issue:** `pub mod lowering`, `pub mod engine`, `pub mod runtime`, `pub mod buffer` expose deep internals. For example, `lowering::naga_emit` lets consumers depend on Naga IR emission details that should be private.
**Fix:** Narrow visibility: `pub(crate) mod lowering`, and re-export only the stable types (`lower_with_features` or `WgpuIR`). Keep `engine` and `runtime` as `pub` only if they contain user-facing types, otherwise `pub(crate)`.

### 6.4 `vyre-libs/src/parsing/c/lex/keyword.rs:1` / `c/lex/lexer.rs:1` / `c/parse/declarations.rs:1` / `c/sema/lookup.rs:1` / `c/sema/registry.rs:1` / `c/sema/walk.rs:1` / `core/ast/binding.rs:1` / `core/ast/blocks.rs:1` / `core/ast/shunting.rs:1` / `core/ast/shunting/operator.rs:1`
**Issue:** `use crate::parsing::c::lex::tokens::*;`  -  wildcard import of 200+ token constants across 10 files.
**Fix:** Replace with explicit `use crate::parsing::c::lex::tokens::{TOK_IDENTIFIER, TOK_AUTO, …}` or a narrow `use super::tokens`.

### 6.5 `vyre-driver-wgpu/src/runtime/mod.rs:47-57`
**Issue:** Re-exports `cache::lru::AccessTracker`, `cache::tier::{AccessStats, CacheError, LruPolicy}`, `device::cached_device`, `device::init_device`, and `shader::compile_compute_pipeline`. These are implementation details of the backend runtime, not a public API.
**Fix:** Make these `pub(crate)` or move them to a `runtime/private.rs` module that is not re-exported at the crate root.

---

## 7. Module-Size / Lego-Block Violations

### 7.1 `vyre-runtime/src/megakernel/c_frontend.rs`  -  966 lines
**Issue:** Mixes manifest layout arithmetic, phase state machine, GPU IR generation (bootstrap, dispatch, guard, fault), region validation, and error types in one file.
**Fix:** Split into `c_frontend/manifest.rs` (region layout + constants), `c_frontend/phase_machine.rs` (`CFrontendPhase` + transitions), `c_frontend/ir.rs` (all `Node` emitters), and `c_frontend/error.rs`.

### 7.2 `vyre-runtime/src/megakernel/builder.rs`  -  632 lines
**Issue:** Contains 10 public builder functions, 4 private program wrappers, buffer declarations, 5 distinct body generators (interpreted, JIT, priority, C-frontend, IO-polling), opcode dispatch, and packed-slot decoding.
**Fix:** Split into `builder/program.rs` (wrappers + buffer decls), `builder/body.rs` (interpreted + priority bodies), `builder/jit.rs` (JIT-specific builders), and `builder/opcode_dispatch.rs` (`dispatch_opcode_body` + `packed_slot_body`).

### 7.3 `vyre-runtime/src/megakernel/telemetry.rs`  -  813 lines
**Issue:** Combines telemetry structs, a full `CountMinSketch` implementation, ring buffer decoding, window grouping, control snapshotting, sketch generation, launch policy recommendations, and tests.
**Fix:** Extract `telemetry/sketch.rs` (`CountMinSketch`), `telemetry/decode.rs` (`RingTelemetry` + `ControlSnapshot` decoding), and `telemetry/policy.rs` (`recommend_launch` + `priority_accounting`).

### 7.4 `vyre-driver-wgpu/src/pipeline.rs`  -  1,028 lines
**Issue:** Contains `CachedPipelineArtifact`, `BufferBindingInfo`, `OutputBindingLayout`, `WgpuPipeline` struct + 3 trait impls, `IndirectDispatch`, `find_indirect_dispatch`, full `NodeVisitor` impl (`IndirectDispatchCollector` with 20 methods), output layout math, trap decoding, and budget enforcement.
**Fix:** Split into `pipeline/artifact.rs`, `pipeline/layout.rs`, `pipeline/indirect.rs`, and `pipeline/dispatch.rs`.

### 7.5 `vyre-driver-wgpu/src/lib.rs`  -  915 lines
**Issue:** Contains the entire `WgpuBackend` definition, `DispatchArena`, `BackendValidationCapabilities` impl, `WgpuBackendStats`, `WgpuIR`, `Executable` impl, inventory submissions, capability queries, lifecycle hooks, and tests.
**Fix:** Split into `backend.rs` (WgpuBackend struct + impls), `backend/capabilities.rs`, `backend/lifecycle.rs`. Keep `lib.rs` as a thin facade with `mod` declarations and re-exports only.

### 7.6 `vyre-driver-wgpu/src/buffer/handle.rs`  -  733 lines
**Issue:** Contains `GpuBufferHandle`, `GpuBufferInner`, `StagingBufferPool`, `BindGroupCache`, resident-buffer registry, readback logic with deadline polling, and tests.
**Fix:** Split into `buffer/handle.rs` (handle only), `buffer/staging.rs` (`StagingBufferPool`), `buffer/bind_group_cache.rs` (`BindGroupCache`), and `buffer/registry.rs` (resident buffer registry).

### 7.7 `vyre-libs/src/parsing/c/parse/vast.rs`  -  8,250 lines
**Issue:** Monolithic module containing `c11_build_vast_nodes`, `c11_classify_vast_node_kinds`, `c11_annotate_typedef_names`, `c11_build_expression_shape_nodes`, and dozens of private helper families.
**Fix:** Split into `vast/build.rs`, `vast/classify.rs`, `vast/typedef.rs`, `vast/expr_shape.rs`. Keep `vast.rs` as a thin `mod.rs` with re-exports.

### 7.8 `vyre-libs/src/parsing/c/lower/ast_to_pg_nodes.rs`  -  1,572 lines
**Issue:** Contains both the GPU lowering pass (`c_lower_ast_to_pg_nodes`, `c_lower_ast_to_pg_semantic_graph`) and the CPU reference oracle (`reference_ast_to_pg_nodes`, `reference_ast_to_pg_semantic_graph`, `PgReferenceDecodeError`).
**Fix:** Split into `lower/gpu.rs` and `lower/reference.rs`; keep `ast_to_pg_nodes.rs` as the public facade.

### 7.9 `vyre-libs/src/visual/blur/mod.rs`  -  288 lines
**Issue:** Inlines an entire separable Gaussian blur instead of composing the existing Tier-2.5 `conv1d_node` twice (horizontal `stride=1`, vertical `stride=width`). A Tier-3 composition should call Tier-2.5, not reimplement it.
**Fix:** Delete the manual kernel loop; build the blur by calling `conv1d_node` for each pass with appropriate `stride`/`count` params and a lightweight unpack/repack wrapper. Target ~80 lines.

---

## Summary Counts

| Category | Count |
|----------|-------|
| Duplicated policy | 7 |
| Wrong workgroup/grid semantics | 6 |
| Avoidable allocations/copies | 7 |
| Fake capability wording | 8 |
| Stale docs | 6 |
| Wildcard API surfaces | 5 |
| Module-size / lego-block violations | 9 |
| **Total** | **48** |
