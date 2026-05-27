# AUDIT: VYRE_DRIVER_DISPATCH  -  vyre-driver dispatch hot path

**Scope:** `libs/performance/matching/vyre/vyre-driver/src/backend` (VyreBackend trait, dispatch plumbing), `pipeline.rs`, `routing.rs`, `shadow.rs`, `persistent.rs`, `speculate.rs`, and the live wgpu backend (`vyre-driver-wgpu/src/lib.rs`, `pipeline.rs`, `pipeline_persistent.rs`, `buffer/handle.rs`).

**Auditor:** Kimi Code CLI  
**Date:** 2026-04-24  
**Standard:** SEVERITY | file:line | defect | suggested fix

---

## CRITICAL

**C1 | `vyre-driver-wgpu/src/pipeline.rs:210-246` | `compile_with_device_queue` recomputes `execution_plan::plan`, `output_layouts_from_program`, `find_indirect_dispatch`, `load_or_compile_disk_wgsl`, and `buffer_bindings` derivation **before** checking the in-memory pipeline cache. On every cache hit all of this work is thrown away.**
- Fix: Move the `pipeline_cache.get(&artifact_key.hash)` check to the top of the function. The cache key must cover all metadata inputs (or metadata must be stored alongside the artifact). Cache the `buffer_bindings`, `output_bindings`, and `execution_plan` inside `CachedPipelineArtifact` so a hit returns immediately without allocations or analysis.

**C2 | `vyre-driver-wgpu/src/lib.rs:503-528` | `WgpuBackend` never overrides `VyreBackend::dispatch_async`. The default trait implementation calls `self.dispatch()` synchronously and wraps the result in `ReadyPending`.**
- Every async dispatch is fully blocking on the host; no host-device overlap exists despite the API implying concurrency. Fix: Override `dispatch_async` with a true async path using `wgpu::Buffer::map_async` + `PendingDispatch` handle, or delete the async surface if wgpu cannot support it.

**C3 | `vyre-driver-wgpu/src/buffer/handle.rs:126-138` | `readback_until` allocates a new `wgpu::Buffer` (MAP_READ staging buffer) and a new `std::sync::mpsc::channel` for **every output buffer on every dispatch**.**
- GPU allocator churn + OS thread synchronization per readback. At high throughput this dominates dispatch latency. Fix: Reuse staging buffers from the persistent pool or a thread-local ring buffer. Replace `mpsc::channel` with a thread-local waker or reuse a single channel per backend.

## HIGH

**H1 | `vyre-driver-wgpu/src/buffer/handle.rs:135` | GPU→staging copy is always full-buffer (`0..byte_len`); `OutputLayout::copy_offset` / `copy_size` are computed but never passed to `copy_buffer_to_buffer`.**
- Byte-range narrowing (`trim_start`, `read_size`) happens only on the CPU after the entire buffer is mapped. A 1 GiB buffer with a 16 B `output_byte_range` still copies 1 GiB across PCIe. Fix: Use `encoder.copy_buffer_to_buffer(src, copy_offset, dst, 0, copy_size)` where `copy_size` comes from `OutputLayout`.

**H2 | `vyre-driver-wgpu/src/lib.rs:526` | `WgpuBackend::dispatch` allocates a `Vec<&[u8]>` on every call to borrow inputs, even though `dispatch_borrowed` takes `&[&[u8]]` and the inputs are already `&[Vec<u8>]`.**
- The corpse file `backend_impl.rs` used `SmallVec<[&[u8]; 8]>` (stack-allocated for ≤8 inputs), but the live code allocates from the heap. Fix: Use `SmallVec<[&[u8]; 8]>` in the live `lib.rs` implementation.

**H3 | `vyre-driver-wgpu/src/pipeline_persistent.rs:210-262` | `legacy_handles_from_inputs` allocates `input_bindings: Vec<_>` via `.collect()`, plus dynamically-growing `input_handles` and `output_handles` `Vec`s on every dispatch.**
- Even with a warm pipeline cache, the hot path pays three heap allocations. Fix: Pre-size vectors with `with_capacity(self.buffer_bindings.len())` or use `SmallVec` for bindings. Prefer `dispatch_persistent` (caller-owned handles) for repeated dispatches.

**H4 | `vyre-driver/src/pipeline.rs:381-721` | Generic `on_disk` pipeline cache module (`on_disk::compute_cache_key`, `load`, `store`) is well-tested dead code. Zero production call sites.**
- The wgpu backend reinvented its own disk cache in `pipeline_disk_cache.rs`. The generic module is architectural debt. Fix: Delete the orphaned `on_disk` module and migrate all backends to a unified cache surface, or consolidate wgpu's disk cache into the generic module.

**H5 | `vyre-driver/src/speculate.rs` | `AdaptiveSpeculator` (375 lines) has zero production call sites outside its own tests. No backend calls `record()`.**
- The speculative dispatch path described in the module docs does not exist in any live backend. Fix: Wire `AdaptiveSpeculator` into the wgpu fused-kernel dispatch loop, or delete the module per LAW 1 (no stubs).

## MEDIUM

**M1 | `vyre-driver/src/persistent.rs` | `PersistentEngine` ring buffer is host-only (`std::sync::RwLock` + `AtomicU32`). No device-side kernel maps the atomics.**
- The docs claim a persistent GPU kernel lives behind a `persistent` cargo feature, but that feature does not exist in any `Cargo.toml`. No backend (including wgpu megakernel) instantiates `PersistentEngine`. Fix: Implement the Vulkan async-compute persistent kernel that maps the same atomics, or delete the module.

**M2 | `vyre-driver/src/pipeline.rs:357-361` | `PassthroughPipeline::dispatch` does `if *config == DispatchConfig::default()` on every call.**
- `DispatchConfig` contains `Option<String>` fields (`profile`, `label`), so this compares strings character-by-character for backends that do not override `compile_native` (passthrough path). Fix: Cache a `bool is_compile_config_default` at `PassthroughPipeline` construction time and branch on that.

**M3 | `vyre-driver-wgpu/src/lib.rs:97` | Pipeline cache hit still constructs a fresh `Arc<WgpuPipeline>` via `Arc::new(Self { ... })` on every dispatch.**
- The cache stores `CachedPipelineArtifact`, but metadata (`buffer_bindings`, `output_bindings`, `execution_plan`) is recomputed and then wrapped in a new `Arc`. This allocates on every dispatch. Fix: Cache the complete `WgpuPipeline` (or at least its metadata `Arc`s) so a hit returns an existing `Arc` without allocation.

**M4 | `vyre-driver-wgpu/src/pipeline.rs:201` | `compile_with_device_queue` takes `_dispatch_arena: DispatchArena` but never uses it.**
- The arena is constructed in `WgpuBackend::acquire`, cloned on every dispatch, and ignored. Fix: Remove the dead parameter or wire the arena into buffer acquisition so the pool and arena share size classes.

**M5 | `vyre-driver/src/routing.rs:127-138` | `RoutingTable::observe_sort_u32` overwrites the previous observation on every call. `select_sort_backend` uses only the current call's distribution.**
- There is no actual profile-guided optimization across dispatches; the "routing table" is just a single-value store. Fix: Maintain an EMA or histogram per call site, or rename the API to avoid misleading PGO semantics.

**M6 | `vyre-driver/src/backend/registry.rs:129-137` | `registered_backends_by_precedence` allocates a new `Vec` and sorts on every call. No `OnceLock` cache.**
- Unlike `registered_backends()` which freezes into a `OnceLock`, this function does `registered_backends().to_vec()` followed by `sort_by`. If called on a dispatch hot path, it is O(n log n) with allocation. Fix: Freeze the sorted slice in a `OnceLock` on first call.

**M7 | `vyre-driver-wgpu/src/pipeline_persistent.rs:343-395` | `cached_bind_groups` creates a new `Vec<wgpu::BindGroupLayoutEntry>` and `Vec<Arc<wgpu::BindGroup>>` on every bind-group cache miss.**
- The entries are deterministic per pipeline but recomputed per miss. Fix: Pre-compute the `Vec<wgpu::BindGroupLayoutEntry>` per bind-group layout at pipeline compile time and reuse them.

**M8 | `vyre-driver-wgpu/src/lib.rs:711-716` | `WgpuBackend::try_recover` panics on lock poisoning via `.expect("Fix: WgpuBackend persistent_pool lock poisoned during recovery")`.**
- The sibling helper `current_persistent_pool()` properly maps poisoning to `BackendError`, but recovery uses `expect`. Fix: Replace `.expect` with a structured error return: `self.persistent_pool.write().map_err(vyre::BackendError::poisoned_lock)?`.

**M9 | `vyre-driver-wgpu/src/pipeline.rs:653-658` | `workgroup_count` computation silently clamps to `u32::MAX` on overflow via `.try_into().unwrap_or(u32::MAX)`.**
- A malformed or very large program could dispatch `u32::MAX` workgroups instead of returning an error. Fix: Replace `unwrap_or(u32::MAX)` with an explicit `try_into().map_err(|_| BackendError::new("workgroup count exceeds u32::MAX"))`.

## LOW

**L1 | `vyre-driver-wgpu/src/pipeline.rs:374-382` | Pipeline cache eviction uses `iter().next()` which is not LRU; evicts an arbitrary entry.**
- The comment admits this. At scale, a hot pipeline can be evicted by a cold one. Fix: Replace the `DashMap` with `moka::sync::Cache` (already used for bind groups) or a linked-hash-map LRU.

**L2 | `vyre-driver/src/backend/vyre_backend.rs:157-158` | Default `dispatch_borrowed` allocates `Vec<Vec<u8>>` from `&[&[u8]]` and delegates to `dispatch`.**
- Every backend that does not override `dispatch_borrowed` pays a per-input `to_vec()` allocation. Fix: Make `dispatch_borrowed` the fundamental primitive and have `dispatch` collect borrows and delegate upward, or remove the default so backends must opt in consciously.

---

## Summary

| Area | Finding Count | Key Theme |
|------|---------------|-----------|
| Per-dispatch allocations | 6 | Vec staging buffers, mpsc channels, Vec<&[u8]>, metadata Arcs |
| Pipeline caching | 4 | Cache lookup too late, metadata not cached, generic on_disk dead, eviction not LRU |
| Orphaned modules | 3 | AdaptiveSpeculator, PersistentEngine, on_disk cache  -  tested but unwired |
| Async / overlap | 1 | dispatch_async is synchronous in wgpu |
| Readback correctness | 1 | Full-buffer GPU copy despite byte-range metadata |
| Routing / registry | 2 | No real PGO history, precedence sort uncached |
| Error handling / safety | 2 | Panic on poison, silent u32::MAX clamp |

**Recommended priority order:** C1 (cache lookup placement), C2 (async fake-out), C3 (staging buffer churn), H1 (byte-range readback), H3 (handle Vec allocs), H5 (delete or wire speculate), M1 (delete or wire persistent), M3 (Arc churn on cache hit).
