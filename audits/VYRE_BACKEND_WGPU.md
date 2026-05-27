# VYRE_BACKEND_WGPU  -  wgpu + megakernel hot-path audit

**Scope:** `libs/performance/matching/vyre/vyre-driver-wgpu/src` (every file), `vyre-runtime megakernel/src`  
**Date:** 2026-04-24  
**Auditor:** Kimi Code CLI  
**Standard:** LAWS 0–8, STANDARDS, RESEARCH PROTOCOL  

---

## Executive Summary

The wgpu backend has solid caching at the **pipeline** layer (shader modules, compute pipelines, bind-group layouts) and a well-designed **persistent buffer pool** (`buffer::BufferPool`), but the **legacy synchronous dispatch path** (`record_and_readback`)  -  which is the default for `WgpuBackend::dispatch`  -  rebuilds bind groups, command encoders, and readback buffers from scratch on every call.  

Push constants are enabled at device creation but never emitted, small dispatches are never auto-merged, and the megakernel trait has **zero backend implementation** in the wgpu crate.  

**Total findings:** 18 (4 CRITICAL, 7 HIGH, 6 MEDIUM, 1 LOW).  

## Closure status  -  2026-04-29 scoped WGPU/megakernel pass

Status in this section is authoritative for the 2026-04-29 deferred-work
closure. Rows not listed here remain historical audit findings, not closure
evidence.

| Finding | Status | Source / proof |
|---|---|---|
| CRITICAL bind groups rebuilt per dispatch | fixed | `vyre-driver-wgpu/src/engine/record_and_readback.rs` accepts `bind_group_cache`; `vyre-driver-wgpu/src/pipeline/tests.rs::direct_record_and_readback_reuses_bind_groups` passed. |
| CRITICAL readback staging allocation in `GpuBufferHandle::readback_until` | fixed | `vyre-driver-wgpu/src/buffer/handle.rs` uses `StagingBufferPool` on hot readback paths. |
| CRITICAL compound readback allocation | fixed | `vyre-driver-wgpu/src/pipeline_compound.rs` acquires readback buffers from the pipeline staging pool and releases them after mapping. |
| CRITICAL megakernel trait unreachable from wgpu path | fixed | `vyre-driver-wgpu/src/megakernel.rs` implements `MegakernelDispatch` for `WgpuMegakernelDispatcher`; `vyre-driver-wgpu/tests/dispatch_megakernel.rs` calls the trait and passed on the RTX 5090. |
| HIGH deadline readback spin loop | fixed | `vyre-driver-wgpu/src/buffer/handle.rs` uses `recv_timeout` with bounded polling instead of `yield_now` spin. |
| MEDIUM arbitrary pipeline cache eviction | fixed | `vyre-driver-wgpu/src/runtime/cache/pipeline.rs` owns bounded LRU cache; `WgpuBackend::stats` reports hit/miss/eviction counters. |
| PHASE1 megakernel atomic element / CAS projection gates | fixed | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs` emits atomic element types; `expr.rs` projects compare-exchange old values; `vyre-driver-wgpu/tests/megakernel_emit.rs` validates WGSL and dispatches shutdown lifecycle. |
| MEDIUM raw `dispatch_wgsl` recompiles identical shaders | fixed | `WgpuBackend` now owns a device-generation-cleared raw WGSL pipeline cache; `ext::tests::dispatch_wgsl_reuses_backend_pipeline_cache` passed on the RTX 5090. |

---

## Findings

### CRITICAL

**CRITICAL | `vyre-driver-wgpu/src/engine/record_and_readback.rs:228` | Bind groups rebuilt per dispatch in the default hot path**
> `device.create_bind_group(&wgpu::BindGroupDescriptor { ... })` is invoked inside `record_and_readback` for every dispatch. The persistent path (`pipeline_persistent.rs`) has a `BindGroupCache` (moka LRU) keyed by handle signature, but the legacy path used by `WgpuPipeline::dispatch_borrowed` → `WgpuBackend::dispatch` bypasses it entirely.  
> **Impact:** Descriptor-heap churn and CPU overhead on every dispatch; nullifies the benefit of pipeline caching for repeated calls.  
> **Fix:** Route `record_and_readback` through the same `BindingSignature` → `BindGroupCache` lookup used by `dispatch_persistent`. If the cache is not warmed, fall back to creation.

**CRITICAL | `vyre-driver-wgpu/src/buffer/handle.rs:126` | `GpuBufferHandle::readback_until` allocates a fresh staging buffer per readback**
> `device.create_buffer(&wgpu::BufferDescriptor { usage: COPY_DST | MAP_READ, ... })` is called on every `readback_until` invocation with no pooling.  
> **Impact:** For high-frequency persistent dispatch loops (e.g. inference servers), this thrashes the GPU allocator and PCIe BAR.  
> **Fix:** Acquire readback staging buffers from `runtime::cache::BufferPool` (the existing global tiered pool) or add a dedicated readback ring to `BufferPool`.

**CRITICAL | `vyre-driver-wgpu/src/pipeline_compound.rs:85` | Compound dispatch also allocates fresh readback buffers**
> `dispatch_compound` creates an un-pooled `wgpu::Buffer` for every request in the batch: `device.create_buffer(... COPY_DST | MAP_READ ...)`.  
> **Impact:** Batching multiple dispatches into one encoder is undermined by per-request allocator churn on the readback side.  
> **Fix:** Reuse the existing `PooledBuffer` / `BufferPool` infrastructure for compound readback staging buffers.

**CRITICAL | `vyre-driver-wgpu/src/megakernel.rs:1` / `vyre-runtime megakernel/src/lib.rs:114` | `MegakernelDispatch` trait is dead code  -  no wgpu implementation exists**
> The `MegakernelDispatch` trait (with `dispatch_megakernel`) is defined in `vyre-runtime megakernel` and re-exported by `vyre-driver-wgpu/src/megakernel.rs`, but `WgpuBackend` never implements it.  
> **Impact:** The entire megakernel mode is unreachable from the wgpu backend. Callers that attempt megakernel dispatch get a compile-time or runtime failure depending on how they wire the trait.  
> **Fix:** Either implement `MegakernelDispatch for WgpuBackend` (resident kernel + ring-buffer queue) or delete the trait and all megakernel re-exports until an implementation is ready (LAW 1  -  no stubs).

---

### HIGH

**HIGH | `vyre-driver-wgpu/src/engine/record_and_readback.rs:354` | One `queue.submit` per dispatch  -  no automatic merging of small dispatches**
> `queue.submit(std::iter::once(encoder.finish()))` is called once per `record_and_readback` invocation. The encoder contains a single compute pass.  
> **Impact:** CPU→GPU submission overhead dominates for small programs (the exact case the P-6 pipeline cache was meant to address).  
> **Fix:** Add an internal dispatch coalescing buffer (time- or count-bounded) that accumulates small dispatches targeting the same pipeline into one command encoder + one `queue.submit`. Expose an opt-out for latency-sensitive callers.

**HIGH | `vyre-driver-wgpu/src/buffer/handle.rs:142-163` | Busy-wait spin loop when readback has a deadline**
> When `deadline: Option<Instant>` is `Some`, `readback_until` spins: `loop { device.poll(Maintain::Poll); match receiver.try_recv() { Empty => { if now >= deadline { err }; thread::yield_now(); } ... } }`.  
> **Impact:** Burns an entire CPU core while waiting for the GPU. On servers handling thousands of concurrent dispatches, this is a scheduler catastrophe.  
> **Fix:** Replace the spin loop with `device.poll(Maintain::wait_for(submission))` capped by the deadline (use a condvar or tokio timeout if async context is available). If wgpu does not support timed waits, document the limitation and reject deadline-based persistent readback.

**HIGH | `vyre-driver-wgpu/src/pipeline.rs:322-328` | Push constants enabled at device creation but never used**
> `runtime/device/device.rs:177-180` enables `wgpu::Features::PUSH_CONSTANTS` and sets `max_push_constant_size`, yet `pipeline.rs:322-328` hardcodes `push_constant_ranges: &[]` for every compute pipeline.  
> **Impact:** Small per-dispatch uniforms (4–16 bytes typical for workgroup config, iteration counters, etc.) still require buffer allocation, `queue.write_buffer`, bind-group creation, and descriptor binding  -  ~10× more CPU overhead than a single `pass.set_push_constant` call.  
> **Fix:** Extend the lowering path to emit push constants for `MemoryKind::Push` / small `MemoryKind::Uniform` bindings, and pass the corresponding `PushConstantRange` at pipeline layout creation.

**HIGH | `vyre-driver-wgpu/src/lib.rs:1` | `lib.rs` is 819 lines  -  violates LAW 2 (every file <500 lines)**
> The file contains the `WgpuBackend` struct definition, multiple `impl` blocks, `inventory::submit!` registrations, `WgpuIR`, `Executable` impl, and tests.  
> **Impact:** God file. New developers cannot understand the backend contract in 5 minutes. Swapping the dispatch strategy requires editing the same file that holds device-lost recovery and capability queries.  
> **Fix:** Split into `backend/dispatch.rs`, `backend/lifecycle.rs`, `backend/capability.rs`, and `backend/registration.rs`. Keep `lib.rs` to module declarations and re-exports only.

**HIGH | `vyre-driver-wgpu/src/pipeline.rs:1` | `pipeline.rs` is 1002 lines  -  violates LAW 2**
> Contains `WgpuPipeline`, `BufferBindingInfo`, `OutputBindingLayout`, `OutputLayout`, compilation, cache management, visitor-based indirect dispatch detection, `CompiledPipeline` impl, and tests.  
> **Impact:** Compilation logic, output layout math, and cache eviction all share one file.  
> **Fix:** Extract `pipeline_compile.rs`, `pipeline_layout.rs`, `output_layout.rs`, and `pipeline_cache.rs`.

**HIGH | `vyre-driver-wgpu/src/lib.rs:164-319` + `backend_impl.rs:1-615` | Duplicate `WgpuBackend` implementation across two files**
> `lib.rs` and `backend_impl.rs` both contain nearly identical `impl WgpuBackend` blocks (adapter_info, device_limits, stats, acquire, current_device_queue, compile_streaming, validate_with_cache, probe_op, VyreBackend trait, inventory submits). Differences are subtle (e.g., `current_persistent_pool` returns `Result` in one, `Option`-like unwrap in the other).  
> **Impact:** Maintenance nightmare. A bug fix or capability update must be applied in two places; drift is guaranteed. Which impl is active depends on module inclusion, creating silent mis-compilation risk.  
> **Fix:** Delete one copy. If `backend_impl.rs` is not referenced by `lib.rs` (it appears to be dead code), delete it immediately. If it is referenced, consolidate into a single file.

---

### MEDIUM

**MEDIUM | `vyre-driver-wgpu/src/pipeline.rs:374-382` | Pipeline cache eviction is not LRU  -  uses arbitrary shard order**
> `if pipeline_cache.len() > MAX_PIPELINE_CACHE_ENTRIES { if let Some(key) = pipeline_cache.iter().next().map(|r| *r.key()) { pipeline_cache.remove(&key); } }`  
> **Impact:** Under adversarial or diverse workloads, hot pipelines can be evicted while cold ones survive, causing unnecessary recompilation.  
> **Fix:** Migrate to the true LRU already present in `vyre-driver/src/pipeline.rs` (referenced in the inline comment), or attach an `Instant` timestamp to each entry and evict the oldest.

**MEDIUM | `vyre-driver-wgpu/src/runtime/shader/compile_compute_pipeline.rs:48` + `ext.rs:30` | `dispatch_wgsl` compiles uncached pipelines every call**
> `crate::runtime::compile_compute_pipeline(device, ..., wgsl, "main")` creates a fresh `ShaderModule` and `ComputePipeline` on every invocation. The comment excuses this because it is "not a dispatch hot path", but `probe_op` (parity testing) and external consumers may call it repeatedly.  
> **Impact:** WGSL compilation is the single most expensive CPU operation in the backend; repeating it for identical shaders is unacceptable at scale.  
> **Fix:** Add a small `DashMap<(String, entry_point), Arc<ComputePipeline>>` cache scoped to the `WgpuBackend` instance, or reuse the existing `pipeline_cache` by normalizing the WGSL source into a cache key.

**MEDIUM | `vyre-driver-wgpu/src/runtime/cache/buffer_pool.rs:126` | `BufferKey` clones `wgpu::Device` as a HashMap key**
> `BufferKey { device: device.clone(), size_class, usage_bits }`. While `wgpu::Device` is internally reference-counted, using it as a hash/equality key relies on pointer identity. If wgpu ever changes `Device` to a handle type without stable identity, the pool will silently create duplicate shards for the same physical device.  
> **Fix:** Hash by a stable device fingerprint (e.g., adapter name + backend + device ID string) rather than the `wgpu::Device` handle itself.

**MEDIUM | `vyre-driver-wgpu/src/engine/streaming.rs:124-161` | `StreamingDispatch` submits each chunk on a separate worker thread with no GPU command batching**
> Each `push_chunk` spawns work onto the global `StreamingPool` (capped at 4 threads). Each worker calls `pipeline.dispatch(&[bytes], &config)`, which internally creates its own command encoder and calls `queue.submit`.  
> **Impact:** GPU commands are not batched across chunks; the GPU sees N independent submissions instead of one coalesced encoder. The 4-thread cap serializes chunk dispatch on the CPU.  
> **Fix:** Keep the worker pool for CPU staging, but funnel GPU submission through a single dispatch queue that batches chunks into one encoder per batch (or per millisecond window).

**MEDIUM | `vyre-driver-wgpu/src/pipeline_persistent.rs:124-138` | Batched persistent dispatch is opt-in, not the default**
> `dispatch_persistent` calls `dispatch_persistent_batched(&[item])` with a single-item slice. Callers must explicitly know about and use `dispatch_persistent_batched` to get command-encoder merging.  
> **Impact:** The default API is the slow path. Most callers will dispatch one item at a time.  
> **Fix:** Make `dispatch_persistent` accept `impl IntoIterator<Item = DispatchItem>` and batch internally. Provide `dispatch_persistent_single` for the rare case that needs isolation.

**MEDIUM | `vyre-driver-wgpu/src/pipeline.rs:246-262` | Race window allows duplicate pipeline compilation**
> `if let Some(existing) = pipeline_cache.get(&artifact_key.hash) { return Ok(...); }` is followed by compilation and then `pipeline_cache.entry(...).or_insert_with(...)`. Between the `get` and `or_insert`, another thread may compile the same pipeline.  
> **Impact:** Duplicate compilation waste under high-concurrency cold starts.  
> **Fix:** Use `pipeline_cache.entry(key).or_try_insert_with(|| compile(...))` (DashMap supports this pattern) so the compilation happens inside the shard lock.

---

### LOW

**LOW | `vyre-driver-wgpu/src/buffer/handle.rs:318-334` + `engine/record_and_readback.rs:451-458` | Zero-padding fallback uses CPU `queue.write_buffer` instead of GPU `clear_buffer`**
> `write_padded` uploads zeros via `queue.write_buffer(buffer, offset, &SCRATCH_ZEROS[..chunk])` when the input is shorter than the allocation. While a static scratch array avoids per-call Vec allocation, it still consumes PCIe bandwidth.  
> **Impact:** For large buffers with small inputs, unnecessary host→device bus traffic.  
> **Fix:** Use `encoder.clear_buffer(...)` for the zero region when the dispatch path already holds a command encoder (the persistent and compound paths). Reserve `write_padded` only for the initial upload path where no encoder exists yet.

---

## Cross-cutting themes

| Theme | Files | Severity |
|---|---|---|
| **Legacy path leaks resources** | `record_and_readback.rs`, `buffer/handle.rs`, `pipeline_compound.rs` | CRITICAL |
| **No automatic dispatch coalescing** | `record_and_readback.rs`, `streaming.rs`, `pipeline_persistent.rs` | HIGH |
| **Push constants declared but dead** | `pipeline.rs`, `runtime/device/device.rs` | HIGH |
| **God files / duplication** | `lib.rs`, `pipeline.rs`, `backend_impl.rs` | HIGH |
| **Megakernel mode is a stub** | `megakernel.rs`, `vyre-runtime megakernel/src/lib.rs` | CRITICAL |

---

## Competitor comparison

* **WebGPU / Dawn** (Google): Batches multiple dispatches into one `CommandEncoder` by default in their high-level compute API; push constants are the default for small uniforms.  
* **wgpu-rs compute examples** (Rust GPU ecosystem): Most examples use persistent bind groups and pre-recorded command buffers for repeated dispatch. vyre-wgpu’s persistent path matches this, but the **default** `dispatch_borrowed` path does not.  
* **Burn / Candle** (Rust ML backends): Use buffer pools for readback staging and push constants for kernel metadata; vyre is behind on both.  

---

## Remediation priority

1. **Delete or implement megakernel stub** (LAW 1).  
2. **Deduplicate `lib.rs` / `backend_impl.rs`** (LAW 2).  
3. **Add bind-group caching to `record_and_readback`** (highest perf impact).  
4. **Pool readback buffers in `handle.rs` and `pipeline_compound.rs`**.  
5. **Implement push-constant lowering for small uniforms**.  
6. **Add automatic small-dispatch coalescing** (time-batched encoder).  
7. **Split god files** (`lib.rs`, `pipeline.rs`).  
8. **Replace readback spin loop with true wait**.  
9. **Cache `dispatch_wgsl` pipelines**.  
10. **Fix pipeline cache compile race**.

---

*End of audit.*
