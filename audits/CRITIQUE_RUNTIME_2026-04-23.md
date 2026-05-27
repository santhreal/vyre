# CRITIQUE  -  vyre-runtime/src/ dispatch + pipeline-cache + megakernel orchestration

**Date:** 2026-04-23  
**Scope:** `vyre-runtime/src/` (read-only audit)  
**Auditor:** Kimi Code CLI  
**Standard:** Laws 0–8 + Unix/SQLite standard + maximal elegance

---

## Executive Summary

| Severity | Count | Categories |
|----------|-------|------------|
| **CRITICAL / HIGH** | 3 | fingerprint stability, storage-buffer portability, CQE reordering |
| **MEDIUM** | 5 | missing adapter limits, aliasing sub-regions, partial batch state, mutex bottleneck, Arc hot-path |
| **INFO** | 2 | no unsafe spawn in production, FINDING-54 error surfaced |

---

## 1. Fingerprint not stable under buffer declaration reordering

**Severity:** HIGH  
**File:** `vyre-runtime/src/pipeline_cache.rs:264`  
**Description:** `PipelineFingerprint::of` calls `canonicalize::run(program.clone())` then `to_wire()`. `structural_eq` (the ground-truth for Program equality) explicitly ignores buffer declaration order (`buffers_equal_ignoring_declaration_order`), yet `to_wire()` serializes buffers in declaration order. Two semantically identical programs with buffers reordered will produce **different fingerprints**, silently fragmenting the content-addressed cache.

**Fix:** Before hashing, sort the buffer table into a canonical order (e.g., by binding index, then by name) so that `to_wire()` output is invariant under declaration reordering. Alternatively, make `to_wire()` itself emit buffers in canonical order.

**Test hint:** Construct two programs where the only difference is `vec![buf_a, buf_b]` vs `vec![buf_b, buf_a]`; assert `PipelineFingerprint::of(&p1) == PipelineFingerprint::of(&p2)`.

---

## 2. Batch program declares 9 storage buffers  -  exceeds WebGPU default limit

**Severity:** HIGH  
**File:** `vyre-runtime/src/megakernel/dispatcher.rs:285–309`  
**Description:** `build_batch_program` emits 8 `BufferDecl::storage` bindings + 1 `BufferDecl::output` binding. In the wgpu backend, `output` buffers are lowered to `wgpu::BufferUsages::STORAGE`. The WebGPU default `max_storage_buffers_per_shader_stage` is **8**. `vyre-driver-wgpu` requests the adapter limit, so it works on high-end hardware (often 16+), but on Intel integrated GPUs and some mobile adapters the limit is exactly 8. Pipeline compilation will fail at runtime with a generic backend error rather than a structured upfront rejection.

**Fix:** Add upfront validation in `BatchDispatcher::new` that counts the total storage-class bindings (including outputs) and rejects with `PipelineError::Backend` naming the limit if it exceeds `backend.device_limits().max_storage_buffers_per_shader_stage`. Consider fusing the rule tables (meta + transitions + accept) into one buffer to stay under the portable limit.

**Test hint:** Bootstrap a `BatchDispatcher` against a mocked backend whose `device_limits()` reports `max_storage_buffers_per_shader_stage = 8`; assert it returns a structured error before compile.

---

## 3. `UringMegakernelPump` assumes FIFO CQE order  -  silent data corruption

**Severity:** HIGH  
**File:** `vyre-runtime/src/uring/pump.rs:198–228`  
**Description:** `drain_into_ring` pops `PendingPublish` from a `VecDeque` in submission order (`pop_front`) while ignoring `cqe.user_data`. io_uring **does not guarantee FIFO completion** for independent reads across different files (cache hits vs disk I/O can reorder). If CQE 2 arrives before CQE 1, the pump publishes the wrong `(tenant_id, opcode, args)` into the megakernel ring for the data that actually landed in VRAM. This is a textbook silent data-corruption bug.

**Fix:** Index `pending` by `chunk_idx` (or `user_data`) in a `HashMap<u64, PendingPublish>`; on each CQE, look up the matching metadata via `cqe.user_data` instead of `pop_front`.

**Test hint:** Submit two reads with distinct `(chunk_idx, slot_idx, opcode)` pairs; simulate out-of-order CQEs (swap res/user_data in a test harness); assert each slot is published with the metadata that matches its `chunk_idx`.

---

## 4. `BatchDispatcher` never validates `workgroup_size_x` against adapter limits

**Severity:** MEDIUM  
**File:** `vyre-runtime/src/megakernel/dispatcher.rs:136–158`  
**Description:** `BatchDispatcher::new` checks `workgroup_size_x != 0` and `worker_groups != 0`, but never intersects `workgroup_size_x` with `backend.device_limits().max_compute_workgroup_size_x` or `max_compute_invocations_per_workgroup`. A caller passing `workgroup_size_x = 1024` on an adapter limited to 256 will get a late, generic backend compile failure instead of a structured `QueueFull` or `Backend` error at construction time.

**Fix:** Query `backend.device_limits()` and return `PipelineError::Backend` with a `Fix:` message if `workgroup_size_x` exceeds the adapter’s x-dimension limit or if `workgroup_size_x * 1 * 1` exceeds `max_compute_invocations_per_workgroup`.

**Test hint:** Mock backend with `max_compute_workgroup_size_x = 64`; pass `workgroup_size_x = 128`; assert structured rejection.

---

## 5. `InMemoryPipelineCache` serializes all lookups with a `Mutex`

**Severity:** MEDIUM  
**File:** `vyre-runtime/src/pipeline_cache.rs:122–154`  
**Description:** `get_arc` takes `self.inner.lock().unwrap()` on every cache hit. For a hot cache accessed by many dispatch threads, this creates unnecessary contention  -  every lookup serializes behind one mutex. A `std::sync::RwLock` (or `parking_lot::RwLock`) would allow concurrent reads.

**Fix:** Replace `Mutex<HashMap<…>>` with `RwLock<HashMap<…>>` (or `dashmap` if cross-crate deps are acceptable). `get_arc` only needs read access; `put` needs write access.

**Test hint:** Spawn N threads doing `get_arc` on a warm cache; measure that throughput scales linearly with thread count (it will not today).

---

## 6. `GpuMappedBuffer::sub_region` takes `&self`, permitting aliasing mutable handles

**Severity:** MEDIUM  
**File:** `vyre-runtime/src/uring/stream.rs:120–133`  
**Description:** `sub_region(&self, …)` returns another `GpuMappedBuffer<'a>` without requiring `&mut self`. Because the type is `unsafe impl Sync`, multiple threads can create overlapping sub-regions and  -  if any code path calls `unsafe { as_mut_slice() }` on them  -  violate Rust’s aliasing rules. The lifetime `'a` is syntactically preserved, but the exclusivity contract of the underlying `&'a mut [u8]` is not.

**Fix:** Change `sub_region` to take `&mut self` so overlapping sub-regions cannot be created without mutable access. Alternatively, make `sub_region` return a borrowed sub-handle whose lifetime is tied to `&self` (e.g., `GpuMappedBuffer<'_>`), but the current `PhantomData<&'a mut [u8]>` design already over-approximates the lifetime; the real gap is the aliasing invariant.

**Test hint:** Miri test: create one `GpuMappedBuffer`, call `sub_region` twice from two threads, then call `as_mut_slice` on both  -  Miri should flag the aliasing violation.

---

## 7. `batch_publish` leaves partial ring state on mid-batch failure

**Severity:** MEDIUM  
**File:** `vyre-runtime/src/megakernel/mod.rs:454–475`  
**Description:** `batch_publish` loops over items and calls `publish_slot` for each, propagating the first error via `?`. Slots published before the failure remain `PUBLISHED` in the ring; there is no rollback. The caller may re-try the batch, causing duplicate slots or colliding with the GPU which may already be processing the first half.

**Fix:** Document the partial-state hazard, or return a `BatchPublishReport { published: Vec<slot_idx>, error: Option<…> }` so the caller knows exactly which slots were dirtied. Even better: validate every slot upfront (bounds + in-flight check) before writing any status word.

**Test hint:** Publish a 3-item batch where slot 2 is still in-flight; assert slot 0 and 1 were written and slot 2 was not.

---

## 8. Five `Arc::clone` bumps per `BatchDispatcher::dispatch` in hot path

**Severity:** MEDIUM  
**File:** `vyre-runtime/src/megakernel/dispatcher.rs:183–201`  
**Description:** Every call to `dispatch` clones five `GpuBufferHandle`s (`offsets`, `metadata`, `work_queue`, `haystack`, plus the three rule buffers). `GpuBufferHandle` is an `Arc<GpuBufferInner>`, so each `.clone()` is an atomic increment. At 1M-file scan volumes this is 5M atomics per dispatch  -  measurable cache-coherency traffic. The inputs slice could borrow `&GpuBufferHandle` instead.

**Fix:** Change `dispatch_persistent` (in the driver layer) to accept `&[GpuBufferHandle]` (or `&[&GpuBufferHandle]`) so the caller borrows instead of cloning. If the driver API requires owned handles, document the cost and provide a zero-copy batch-dispatch variant.

**Test hint:** Benchmark `BatchDispatcher::dispatch` in a loop with `criterion`; compare Arc-clone path vs borrowed-handle path.

---

## 9. No `spawn` / `thread::scope` with `&static` capture in production code

**Severity:** INFO  
**File:** `vyre-runtime/src/tenant.rs:502–503` (test only)  
**Description:** The only `thread::spawn` in the crate lives inside a `#[cfg(test)]` block and captures an `Arc::clone(&reg)`. No production code spawns threads. No `thread::scope` or `crossbeam::scope` usage was found. Lifetime annotations on `GpuMappedBuffer<'a>`, `AsyncUringStream<'a>`, and `NvmeGpuIngestDriver<'a>` correctly tie the handles to the backing allocation.

**Fix:** None required. Continue keeping production code single-threaded at the runtime layer.

---

## 10. FINDING-54 (64-slot IO queue overflow)  -  error IS surfaced, not swallowed

**Severity:** INFO  
**File:** `vyre-runtime/src/megakernel/io.rs:121–126`, `vyre-runtime/src/uring/driver.rs:185`  
**Description:** `MegakernelIoQueue::new` rejects `slot_count > IO_SLOT_COUNT` (64) with `PipelineError::QueueFull` carrying an actionable `Fix:` string. `publish_slot` rejects out-of-bounds slots with the same variant. The `NvmeGpuIngestDriver::poll_completions` caller propagates via `?` to its own `Result`. There is no downstream wrapper that swallows or remaps the error to a generic variant.

**Fix:** None required. The rejection path is loud and structured.

**Test hint:** Already covered by existing tests in `io.rs:406–441`.

---

## Honorable Mentions (not counted in severity tally)

| Issue | Location | Note |
|-------|----------|------|
| `FxHashMap` determinism | `vyre-foundation/src/transform/optimize/canonicalize.rs:254–261` | `hash_str` uses a fixed seed, so `expr_sort_key` is deterministic across runs. **Not a finding.** |
| `program.clone()` in fingerprint path | `vyre-runtime/src/pipeline_cache.rs:264` | Clones the `Arc` fields of `Program`; cheap and correct. **Not a finding.** |
| `tenant.rs` `Arc::clone` in test loop | `vyre-runtime/src/tenant.rs:502` | Test-only, 32 iterations. **Not a production finding.** |

---

## Competitor Comparison

- **wgpu / naga:** wgpu’s pipeline cache uses a stable hash of the SPIR-V blob, not the source IR. They do not canonicalize source-level operand order, so equivalent shaders with different AST shapes cache-miss. Vyre’s canonicalize pass is ahead of the competition here, but the buffer-order instability (Finding 1) is a gap wgpu does not have because SPIR-V binding indices are explicit.
- **Vulkan validation layers:** Would catch the 9-storage-buffer limit (Finding 2) at `vkCreatePipelineLayout` with `VK_ERROR_OUT_OF_DEVICE_MEMORY` or `VK_ERROR_INITIALIZATION_FAILED`. Vyre currently surfaces this as a generic backend string  -  less actionable.
- **tokio-uring / glommio:** Both use `user_data` correlation for out-of-order completions. Vyre’s pump (Finding 3) violates this baseline expectation.

---

## Remediation Priority

1. **Fix Finding 3 (CQE reordering)** immediately  -  silent data corruption is unacceptable.
2. **Fix Finding 2 (9 storage buffers)**  -  fuse tables or add upfront validation; portability to Intel iGPU is a hard requirement for a scanner runtime.
3. **Fix Finding 1 (fingerprint stability)**  -  sort buffers before wire encoding.
4. Fix Findings 4–8 in priority order; they are correctness/performance debt, not corruption.
