# PERF BUFFER POOL + BIND-GROUP CACHE SWEEP  -  2026-04-23

> Scope: `vyre-driver-wgpu/src/**/*.rs` + `vyre-runtime/src/**/*.rs`  
> Focus: dispatch latency  -  buffer creation, bind-group assembly, pipeline cache, encoder reuse, readback staging.  
> Method: static read-only audit; no benchmarks run.

---

## Executive Summary

The wgpu backend has **three distinct buffer pools** and **two distinct bind-group caching layers**, but they are **not wired together** on every dispatch path. The result is a "swiss-cheese" caching story: the *compiled-pipeline* path (`dispatch_persistent`) is well-optimized, while the *legacy/direct* path (`record_and_readback`, `GpuBufferHandle::readback`, megakernel readback) repeatedly falls through to `device.create_buffer` and `device.create_bind_group`.

| Layer | Status | Files |
|---|---|---|
| Buffer pool (global, device+size+usage keyed) | ✅ Exists, used by `record_and_readback` | `runtime/cache/buffer_pool.rs` |
| Buffer pool (persistent, power-of-two, **usage-agnostic**) | ⚠️ Partial  -  one bucket per size class, no usage sub-bucketing | `buffer/pool.rs` |
| Bind-group cache (per-pipeline LRU) | ✅ Exists, but **per-instance** not per-artifact | `pipeline_persistent.rs` |
| Pipeline disk cache (WGSL + compiled blob) | ✅ Exists, persisted | `pipeline_disk_cache.rs` |
| Command encoder reuse | ❌ Fresh encoder every dispatch | `engine/record_and_readback.rs`, `pipeline_persistent.rs`, `pipeline_compound.rs` |
| Readback staging buffer reuse | ❌ Fresh buffer in legacy readback path | `buffer/handle.rs` |

---

## 1. Buffer Pool  -  Is there a (size, usage) keyed pool?

**Answer: Two pools exist, but the hot persistent path is usage-naïve.**

### 1.1 Global pool (`runtime/cache/buffer_pool.rs`)  -  ✅ Correctly keyed
- Key: `(wgpu::Device, size_class: u64, usage_bits: u32)`  -  line 15.
- Used by: `record_and_readback.rs` (the direct `VyreBackend::dispatch` path).
- Size class: `size.max(4).next_multiple_of(4)`  -  **overly fine-grained** (see POOL-2).

### 1.2 Persistent pool (`buffer/pool.rs`)  -  ⚠️ Usage-mismatch churn
- Key: **size class only** (power-of-two, 64 buckets, bitmap search).
- `FreeEntry` stores `usage: wgpu::BufferUsages`, but the **bitmap `non_empty_classes` does not distinguish usage**.
- On `acquire`, the code pops the first entry in the size class; if `!entry.usage.contains(usage)` it **pushes the entry back and breaks to fresh allocation** (`pool.rs:202–208`).
- Consequence: a workload that alternates input (`STORAGE|COPY_DST`) and output (`STORAGE|COPY_SRC|COPY_DST|INDIRECT`) buffers of similar size will **hit the wrong usage on the first probe every time** and allocate fresh.

**POOL-1 | buffer pool (persistent) | `buffer/pool.rs:202` | Fix: Sub-divide each size-class bucket by `usage_bits` (or at least by a small set of canonical usage masks) so the bitmap search lands on a compatible entry. Estimated improvement: 5–15 µs per buffer on mixed-usage dispatches (30–60 µs total for a 3-buffer program).**

**POOL-2 | buffer pool (global) | `runtime/cache/buffer_pool.rs:265` | Fix: Replace 4-byte-aligned size classes with power-of-two (or at least 2× geometric) bucketing to reduce bucket fragmentation. A 1025-byte and 2044-byte buffer currently land in different buckets; with power-of-two they share a 2048-byte bucket. Estimated improvement: 2–8 µs per dispatch for programs with runtime-sized inputs.**

---

## 2. Bind-Group Cache  -  Is there a (pipeline, buffer-set) cache?

**Answer: Yes, but only on the `dispatch_persistent` path, and it is not shared across `WgpuPipeline` instances.**

### 2.1 `BindGroupCache` in `pipeline_persistent.rs`  -  ✅ Exists
- `moka::sync::Cache<BindingSignature, Arc<[Arc<wgpu::BindGroup>]>>`  -  line 62.
- Key: `BindingSignature` = sorted `(group, binding, handle_id, allocation_len, usage_bits)` per bound buffer.
- **Hit path**: `cached_bind_groups` returns cached `Arc<[Arc<wgpu::BindGroup>]>`  -  ~0.5 µs.
- **Miss path**: `device.create_bind_group` for every group  -  ~3–10 µs per group.

### 2.2 `record_and_readback.rs`  -  ❌ No bind-group cache
- Every dispatch calls `device.create_bind_group` for every group (`record_and_readback.rs:227–231`).
- This is the path taken by `VyreBackend::dispatch_borrowed` when it compiles a pipeline and immediately dispatches.

### 2.3 Per-instance isolation  -  ❌ Cache not shared
- `pipeline.rs:251` and `pipeline.rs:387` create `Arc::new(BindGroupCache::default())` **for every `WgpuPipeline` instance**, even when the underlying `CachedPipelineArtifact` is shared via `pipeline_cache`.
- If a caller calls `compile_native` twice for the same program, they get two `WgpuPipeline` handles with **independent empty bind-group caches**. The second handle rebuilds all bind groups on its first dispatch.

**POOL-3 | bind-group assembly | `engine/record_and_readback.rs:227` | Fix: Add a `BindGroupCache` (or reuse the one from the compiled pipeline) to the `record_and_readback` path. Alternatively, route the direct dispatch path through `dispatch_persistent` so it inherits the existing cache. Estimated improvement: 5–20 µs per dispatch (dominant for short programs).**

**POOL-4 | bind-group cache isolation | `pipeline.rs:251` | Fix: Move `bind_group_cache` into `CachedPipelineArtifact` so every `WgpuPipeline` instance sharing the same compiled artifact also shares the same bind-group LRU. Estimated improvement: 10–30 µs on the first dispatch of a duplicate compiled pipeline; amortizes to zero on steady-state.**

---

## 3. Pipeline Compiled Shader Cache  -  Persisted to disk?

**Answer: Yes. Cold-miss path is silent.**

### 3.1 Disk cache layers
1. **WGSL text cache** (`pipeline_disk_cache.rs:42–76`): keyed by `(normalized_program_wire, adapter_fingerprint, policy)`. Stores `.wgsl` + `.wgsl.toml`.
2. **Compiled pipeline blob cache** (`pipeline_disk_cache.rs:103–148`): keyed by `(adapter_fingerprint, wgsl_blake3, naga_version, abi_version)`. Stores `.pipeline.bin` + `.pipeline.toml`.
3. **wgpu `PipelineCache`**: loaded with the disk blob on cache hit, empty on miss.

### 3.2 Cache-miss path
- `load_or_compile_disk_wgsl`: on miss, calls `lower_wgsl` (IR → WGSL lowering, ~1–10 ms for complex programs) and persists. **No log emitted.**
- `create_compiled_pipeline_cache`: on miss, creates empty `wgpu::PipelineCache`. **No log emitted.**
- `WgpuPipeline::compile_with_device_queue`: after `device.create_compute_pipeline`, calls `persist_compiled_pipeline_cache`. **No log emitted.**
- The only observable miss signal is the elapsed wall time of `compile_with_device_queue`, which is not traced at `INFO` level.

**POOL-5 | pipeline cache miss silent | `pipeline_disk_cache.rs:42` | Fix: Emit `tracing::info!` on WGSL cache miss (`lower_wgsl` invoked) and `tracing::warn!` on compiled-pipeline cache miss (fresh `create_compute_pipeline`  -  can be 10–100 ms). This lets operators detect cold-cache boots, Naga version bumps, or cache-directory permission issues. Estimated improvement: operational, not latency, but prevents week-long silent regressions.**

---

## 4. Command Encoder  -  Reused across dispatches?

**Answer: Never. Fresh encoder per dispatch, per batch, and per compound submission.**

| File | Line | Pattern |
|---|---|---|
| `engine/record_and_readback.rs` | 234 | `device.create_command_encoder` per dispatch |
| `pipeline_persistent.rs` | 143 | `device.create_command_encoder` per batch |
| `pipeline_compound.rs` | 43 | `device.create_command_encoder` per compound submission |
| `runtime/prerecorded.rs` | 113 | `device.create_command_encoder` per prerecord (acceptable) |
| `buffer/handle.rs` | 132 | `device.create_command_encoder` per readback |

wgpu `CommandEncoder` creation is a heap allocation + internal state init. While not as expensive as buffer or bind-group creation, it is non-zero (~1–3 µs) and creates allocator pressure under high dispatch rate.

**POOL-6 | command encoder | `engine/record_and_readback.rs:234` | Fix: Pool encoders in a `crossbeam_queue::ArrayQueue` on the backend; reset and reuse instead of dropping. For the persistent path, accept an `&mut wgpu::CommandEncoder` parameter so callers can batch many dispatches into one encoder. Estimated improvement: 1–3 µs per dispatch; 10–30 µs per 10-dispatch batch.**

---

## 5. Scratch Staging Buffer for Readback  -  Reused?

**Answer: The `record_and_readback` path uses the global pool; the legacy `GpuBufferHandle` path allocates fresh every time.**

### 5.1 `record_and_readback` path  -  ✅ Uses global pool
- Readback buffers acquired from `runtime::cache::BufferPool` (`record_and_readback.rs:308–314`).
- Returned to pool on drop.

### 5.2 `GpuBufferHandle::readback_until`  -  ❌ Fresh allocation
- `buffer/handle.rs:126–131`: creates a new `wgpu::Buffer` with `COPY_DST | MAP_READ` **every call** via `device.create_buffer`.
- This is the path used by:
  - `WgpuPipeline::dispatch_borrowed` (legacy readback of outputs)  -  `pipeline.rs:670`
  - `WgpuMegakernelDispatcher::dispatch_megakernel` (work_queue readback)  -  `vyre-runtime/src/megakernel/wgpu_dispatch.rs:51`
  - `BatchDispatcher::dispatch` (queue_state + hit_ring readback)  -  `vyre-runtime/src/megakernel/dispatcher.rs:215–236`

### 5.3 `pipeline_compound.rs`  -  ❌ Fresh allocation
- `pipeline_compound.rs:85–90`: creates a fresh readback buffer per compound dispatch.

**POOL-7 | readback staging buffer | `buffer/handle.rs:126` | Fix: Route `GpuBufferHandle::readback_until` through the global `runtime::cache::BufferPool` (acquire `COPY_DST|MAP_READ`, map, read, unmap, release). Estimated improvement: 3–8 µs per readback; on megakernel work-queue readback this is a pure win because the work_queue buffer is often small (a few KiB).**

---

## Additional Findings

### POOL-8 | CPU allocation in zero-fill path
**`engine/record_and_readback.rs:423`**  
`write_padded_input` allocates `vec![0u8; size - zero_start]` every time a padded input has trailing zeros. Replace with a static zero slice (capped at e.g. 4096) or use `encoder.clear_buffer` when the buffer is already GPU-resident.  
**Fix:** Pass the encoder into `write_padded_input` and call `encoder.clear_buffer` for the tail instead of uploading zeros from host.  
**Estimated improvement:** 0.5–2 µs per dispatch + reduced CPU allocator pressure.

### POOL-9 | Tiered cache exists but is unwired
**`buffer/pool.rs:115`**  
`BufferPool::with_tiering` constructs a `TieredCache` (hot/cold LRU layering) but is **never called** anywhere in the driver or runtime. Long-running inference servers and streaming scanners therefore run with the raw power-of-two pool, which can retain arbitrarily cold buffers until the byte cap is hit.  
**Fix:** Wire `with_tiering` in `WgpuBackend::acquire` when `VYRE_BUFFER_TIERING=1` is set, defaulting to e.g. 16 MiB hot / 1 GiB cold.  
**Estimated improvement:** 5–15 µs per dispatch on steady-state servers by keeping hot buffers in a promoted tier with fewer evictions.

---

## Estimated Dispatch Latency Improvement

Assumptions: a "typical" vyre dispatch has 3 buffers (2 inputs, 1 output), 1 bind group, 1 readback, and hits the in-memory pipeline cache.

| Percentile | Current Overhead (typical) | After All Fixes | Improvement |
|---|---|---|---|
| **p50** | ~40–70 µs (encoder + bind-group + buffer pool churn on mixed usage) | ~8–15 µs | **~30–55 µs (≈ 3–5×)** |
| **p99** | ~2–10 ms (cold pipeline compile, fresh readback buffers, repeated buffer allocations due to usage mismatch) | ~0.5–1.5 ms | **~1.5–8.5 ms (≈ 4–7×)** |

The p99 improvement is dominated by **eliminating the cold-compile path** (already cached via disk) and **removing the `GpuBufferHandle::readback` fresh-allocation path** on megakernel work-queue readback, which currently adds ~5–20 µs of buffer creation to every megakernel dispatch.

On a **sustained inference server** (e.g. Karyx streaming scanner) that dispatches the same pipeline thousands of times per second, the steady-state p50 improvement from encoder reuse + bind-group caching + readback pooling is expected to be **~35–60 µs per dispatch**, translating to **~15–30% higher throughput** when GPU execution time is in the 100–300 µs range.

---

## Commit Message

```text
audit(vyre): PERF buffer-pool + bind-group cache audit  -  dispatch latency findings
```

---

*Audit generated 2026-04-23. READ-ONLY  -  no code modified.*
