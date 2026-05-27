# Thread-Safety Audit — vyre-wgpu runtime

**Date:** 2026-04-18  
**Scope:** `vyre-wgpu/src/runtime/`, `vyre-wgpu/src/pipeline*.rs`, `vyre-wgpu/src/engine/`  
**Auditor:** Automated static analysis + manual review  
**Rules:** CORE LAWS apply — no stubs, no band-aids, no "later".

---

## Executive Summary

The vyre-wgpu crate contains **one CRITICAL** and **four HIGH** thread-safety findings. The most severe is a worker-pool serialization bug that causes all streaming dispatch workers to contend on a single `Mutex<mpsc::Receiver>`, completely defeating parallelism. Multiple global caches perform non-atomic file I/O without locks, permitting on-disk corruption under concurrent compilation. Several hot-path mutexes are held during expensive GPU driver calls, creating process-wide serialization points.

No `static mut`, `unsafe impl Send/Sync`, or `RefCell`/`Cell` in `Arc` were found — the codebase is clean of the most egregious Rust concurrency anti-patterns. The issues that remain are architectural: lock granularity, poison recovery, and global mutable state.

---

## Findings

### TSAFE-01 — `engine/streaming.rs:40` — CRITICAL

**Current:** `StreamingPool` wraps a single `mpsc::Receiver` in `Arc<Mutex<...>>` and shares it across all worker threads. Every worker must acquire the mutex before calling `recv()`, which blocks until a job arrives. Because `recv()` is blocking, the mutex is held for an unbounded duration, which means **only one worker can ever wait for a job at a time**. The other workers are serialized on the mutex. A four-worker pool effectively runs as a single-threaded pool.

**Fix:** Replace `std::sync::mpsc` with `crossbeam::channel` or `flume`, both of which support multiple consumers without an external mutex. Alternatively, spawn one dedicated channel (or mpsc channel pair) per worker and round-robin dispatch jobs.

---

### TSAFE-02 — `runtime/shader/compile_compute_pipeline.rs:56-65` — HIGH

**Current:** `PipelineCache::get` takes `&mut self` because it touches the LRU list (`self.lru.touch`). As a result, every cache lookup—even a hit—must acquire a `write()` lock on the shard. This eliminates all read concurrency for shader compilation and turns the 32-shard `RwLock` array into a series of exclusive bottlenecks.

**Fix:** Split the cache into a read-only lookup table (`RwLock::read()`) and a separate LRU promotion/eviction path that takes `write()`. If LRU tracking on every hit is mandatory, replace the `RwLock` shards with a lock-free concurrent hash table (e.g., `dashmap`) or a `Mutex` with documented serialization semantics.

---

### TSAFE-03 — `runtime/aot.rs:53-134` — HIGH

**Current:** `load_or_compile_with_config` performs non-atomic read-check-write against `~/.cache/vyre/aot/`. Two threads compiling the same program can race:

1. Both miss the cache.
2. Both lower IR to WGSL.
3. Both call `fs::write(&wgsl_path, &wgsl)` concurrently.

`fs::write` truncates-then-writes, so interleaved writes produce corrupted cache files. A subsequent read may return malformed WGSL or fail TOML parsing, causing a persistent crash loop for that program hash.

**Fix:** Guard the read-check-write sequence with a `std::sync::Mutex` keyed by the cache directory, or use atomic writes (write to a temp file + `fs::rename`).

---

### TSAFE-04 — `pipeline_disk_cache.rs:10-32` — HIGH

**Current:** Same TOCTOU race as TSAFE-03, but for the disk WGSL cache (`~/.cache/vyre/pipeline/`). `fs::read_to_string` is followed by `fs::write` with no locking or atomicity. Concurrent compilation of the same `Program` can corrupt the on-disk artifact.

**Fix:** Atomic temp-file writes (`{path}.tmp` → `fs::rename`) or a process-wide `Mutex` guarding the cache directory.

---

### TSAFE-05 — `pipeline_compound.rs:43-44` — HIGH

**Current:** `dispatch_compound` calls `runtime::cached_device()` and uses that global `Device`/`Queue` pair for **all** pipelines in the request list. Each `WgpuPipeline` carries its own `device_queue: Arc<(wgpu::Device, wgpu::Queue)>`. If the caller mixes pipelines compiled on different devices (or on fresh devices from `WgpuPipeline::compile`), wgpu will panic or reject the submission because buffers and pipeline layouts are bound to a different device than the command encoder.

**Fix:** Validate that all pipelines in `requests` share the same device before coalescing, or group requests by device and dispatch per-device.

---

### TSAFE-06 — `runtime/shader/compile_compute_pipeline.rs:84-90` — MEDIUM

**Current:** After compiling a pipeline outside the lock (correct), the code re-acquires the write lock but **never double-checks** whether another thread already inserted the same pipeline while it was compiling. This permits redundant shader compilations under load, wasting GPU driver time.

**Fix:** After acquiring the second write lock, check `pipelines.get(&cache_key)` again before inserting.

---

### TSAFE-07 — `runtime/shader/compile_compute_pipeline.rs:103-138` — MEDIUM

**Current:** `driver_pipeline_cache` holds the global `Mutex<HashMap<wgpu::Device, wgpu::PipelineCache>>` while calling `unsafe { device.create_pipeline_cache(...) }`. Driver cache creation can be expensive (driver may allocate internal bookkeeping structures) and blocks all other threads that need a driver cache.

**Fix:** Create the `wgpu::PipelineCache` outside the lock, then acquire the mutex only for the brief `HashMap::insert`.

---

### TSAFE-08 — `runtime/cache/buffer_pool.rs:94-129` — MEDIUM

**Current:** `BufferPool::acquire` holds the global `Mutex<HashMap<BufferKey, Vec<wgpu::Buffer>>>` while calling `device.create_buffer()`. GPU buffer allocation is a potentially expensive driver call. The global pool serializes all buffer allocations across the entire process.

**Fix:** Pre-allocate buffers outside the lock (e.g., `let buf = device.create_buffer(...)` before locking), or use a lock-free bucketed free-list (e.g., one `Mutex<Vec<Buffer>>` per size class, or `crossbeam::queue::SegQueue`).

---

### TSAFE-09 — `runtime/cache/buffer_pool.rs:196-211` — MEDIUM

**Current:** `PooledBuffer::drop` does:

```rust
let Ok(mut buffers) = pool.buffers.lock() else { return; };
```

If the pool mutex is poisoned, the buffer is silently dropped and lost to the pool forever. Repeated panics while holding the pool lock (e.g., from a buggy `device.create_buffer` callback) will exhaust GPU memory because buffers are never returned.

**Fix:** Recover from poison with `lock().unwrap_or_else(|e| e.into_inner())` so buffers are always returned to the pool.

---

### TSAFE-10 — `runtime/device/device.rs:19-28` — MEDIUM

**Current:** `cached_device()` creates a fresh `(wgpu::Device, wgpu::Queue)` on every call and leaks it via `Box::leak`. The device clone is registered in a global `Mutex<Vec<wgpu::Device>>`, but the queue is leaked without tracking. Concurrent callers multiply the leak without bound.

**Fix:** Return an `Arc`-owned pair instead of `&'static`, or make `cached_device()` a true singleton that reuses the same leaked pair.

---

### TSAFE-11 — `pipeline.rs:245-257` — MEDIUM

**Current:** `PIPELINE_CACHE` eviction uses `cache.keys().next()` under a write lock, but `FxHashMap` does not preserve insertion order. The evicted entry is effectively random, not LRU. Under concurrent load, hot pipelines may be evicted while cold ones remain, causing thundering-recompile across threads.

**Fix:** Integrate the crate's existing `IntrusiveLru` into each shard, or document that eviction is arbitrary and increase `MAX_PIPELINE_CACHE_ENTRIES_PER_SHARD` to compensate.

---

### TSAFE-12 — `runtime/shader/compile_compute_pipeline.rs:126-135` — MEDIUM

**Current:** An `unsafe` block bypasses the crate-level `#![deny(unsafe_code)]` lint to call `device.create_pipeline_cache(...)`. The safety justification (`data: None`, `fallback: true`) is sound today but fragile; a future refactor that changes these parameters would silently break safety invariants.

**Fix:** Extract into a small, dedicated safe wrapper whose signature enforces the contract (e.g., `fn create_empty_pipeline_cache(device: &wgpu::Device) -> wgpu::PipelineCache`), keeping the `unsafe` block to a single, reviewed line.

---

### TSAFE-13 — `engine/dfa.rs:144-158` — LOW

**Current:** `GpuDfa` is implicitly `Send + Sync` because all its fields are. There is no explicit static assertion. If a maintainer adds a `std::rc::Rc`, `RefCell`, or other `!Send`/`!Sync` field in the future, the type will silently become non-thread-safe and break callers that share it across threads.

**Fix:** Add `static_assertions::assert_impl_all!(GpuDfa: Send, Sync);` in the test module.

---

### TSAFE-14 — `engine/string_matching.rs:20-29` — LOW

**Current:** `GpuLiteralSet` has the same implicit `Send + Sync` risk as `GpuDfa`. It stores raw `wgpu::Buffer` handles without an explicit `Send + Sync` assertion.

**Fix:** Add `assert_impl_all!(GpuLiteralSet: Send, Sync);` in tests.

---

### TSAFE-15 — `engine/streaming.rs:51-58` — HIGH

**Current:** If one streaming worker panics while holding the `Mutex<mpsc::Receiver>`, the mutex is poisoned. All remaining workers detect the poison on their next iteration, log an error, and return. A single panicked job silently kills the entire global worker pool. No recovery or respawn mechanism exists.

**Fix:** Use `lock().unwrap_or_else(|e| e.into_inner())` to recover from poison and keep workers alive. Additionally, catch panics inside the job closure (the outer code already uses `catch_unwind` for the runner, but the `recv()` mutex acquisition itself is unprotected).

---

## Category Matrix

| Category | Count | Findings |
|----------|-------|----------|
| `static mut` | 0 | — |
| `OnceLock<T>` / `OnceCell<T>` with non-Sync T | 0 | All `OnceLock`/`LazyLock` instances wrap `Sync` types |
| `Arc<Mutex<T>>` held across `await` | 0 | No async mutex usage found |
| `Arc<RwLock<T>>::write()` in read-heavy loop | 1 | TSAFE-02 (write-lock on every cache hit) |
| Iterating `FxHashMap` while another thread inserts | 0 | All hash maps are protected by locks or `&mut self` |
| `RefCell` / `Cell` in `Arc` | 0 | None found |
| Drop order hazards | 0 | None identified |
| Missing `Send + Sync` on `dyn Trait` | 0 | All trait objects carry `+ Send + Sync` |
| `wgpu::Device` / `Queue` sharing without `Arc` clone | 1 | TSAFE-10 (leaked reference) |
| Pipeline cache without interior sync | 0 | Caches are synchronized, but with wrong granularity |
| Additional: Mutex serialization / poison / file races | 10 | TSAFE-01, 03, 04, 05, 06, 07, 08, 09, 11, 15 |
| Additional: `unsafe` fragility / static assertions | 2 | TSAFE-12, 13, 14 |

---

## Remediation Priority

1. **Immediate (CRITICAL):** TSAFE-01 — streaming worker pool is serialized.
2. **This sprint (HIGH):** TSAFE-02, TSAFE-03, TSAFE-04, TSAFE-05 — cache serialization and disk corruption.
3. **Next sprint (MEDIUM):** TSAFE-06, TSAFE-07, TSAFE-08, TSAFE-09, TSAFE-10, TSAFE-11, TSAFE-12 — redundant work, poison recovery, leaks, eviction fairness.
4. **Backlog (LOW):** TSAFE-13, TSAFE-14 — defensive static assertions.
5. **Re-audit (HIGH):** TSAFE-15 — worker pool cascading failure; consider merging with TSAFE-01 fix.

---

*End of audit.*
