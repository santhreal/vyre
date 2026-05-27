# Resource Leak Audit — 2026-04-18

**Scope:** Full workspace (`vyre-*`, `core/`, `tests/`, `demos/`, `examples/`, `benches/`)  
**Method:** ripgrep + manual trace for `Box::leak`, `mem::forget`, `Arc` cycles, `thread::spawn`, `OnceLock`/`LazyLock`, `wgpu::Buffer` creation, file-handle RAII, tracing spans.  
**Findings:** 35  
**Code modified:** None (audit only).

---

## Legend

| Severity | Meaning |
|----------|---------|
| **CRITICAL** | Unbounded growth; will OOM or exhaust GPU handles at scale. |
| **HIGH** | Fixed-size but large footprint; no eviction path. |
| **MEDIUM** | Justified for current design, but lacks documented bound or recovery. |
| **LOW** | Test-only, CLI-one-shot, or bounded by small cardinality; acceptable but should be noted. |

---

## Category A — `Box::leak` (deliberate; every instance must be justified)

### LEAK-01 — `vyre-wgpu/src/runtime/device/device.rs:20` — CRITICAL
**Current:**
```rust
pub fn cached_device() -> Result<&'static (wgpu::Device, wgpu::Queue)> {
    let cached = Box::leak(Box::new(fresh_gpu()?));
    cached_device_registry()
        .lock()?
        .push(cached.pair.0.clone());
    Ok(&cached.pair)
}
```
Every call leaks a new `CachedGpu` (device + queue + adapter info) onto the heap and clones the `Device` handle into a global `Vec`. The test at line 155 explicitly asserts this is *not* a singleton, confirming the growth is intentional.  
**Fix:** Remove `cached_device()` entirely or gate it behind a `#[deprecated]` singleton that returns a cached `&'static` from a `OnceLock`, deduplicating by device fingerprint.

### LEAK-02 — `vyre-wgpu/src/runtime/device/device.rs:33` — CRITICAL
**Current:**
```rust
pub fn cached_adapter_info() -> Result<&'static wgpu::AdapterInfo> {
    let cached = Box::leak(Box::new(fresh_gpu()?));
    Ok(&cached.adapter_info)
}
```
Same pattern as LEAK-01 but leaks the full `CachedGpu` just to borrow `adapter_info`.  
**Fix:** Store adapter info in a `OnceLock<AdapterInfo>` and return a reference to that.

### LEAK-03 — `vyre-wgpu/src/runtime/device/device.rs:144-147` — CRITICAL
**Current:**
```rust
fn cached_device_registry() -> &'static Mutex<Vec<wgpu::Device>> {
    static REGISTRY: OnceLock<Mutex<Vec<wgpu::Device>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Vec::new()))
}
```
Cloned `wgpu::Device` handles accumulate forever. Every `cached_device()` call pushes a new entry. No truncation, no eviction.  
**Fix:** Replace with a `LazyLock<Arc<(Device, Queue)>>` singleton, or at minimum a `Weak` reference so handles can drop when the last caller releases them.

### LEAK-04 — `vyre-conform/src/runner/calibrate.rs:134` — MEDIUM
**Current:**
```rust
let op: &'static OpSpec = Box::leak(Box::new(op));
```
Calibration runner leaks the selected `OpSpec` so it can be referenced by `GenerationPlan` without lifetime parameters. Calibration is a long-lived batch process, so the leak is bounded by the number of ops calibrated in one run (usually one).  
**Fix:** Acceptable for a one-shot CLI, but document the bound in the function doc comment.

### LEAK-05 — `vyre-conform/src/runner/calibrate.rs:455-457` — LOW
**Current:**
```rust
fn leak_string(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}
```
Helper used to leak report strings into `&'static str` for downstream consumers. Bounded by report cardinality.  
**Fix:** Document bound; or switch to `Arc<str>` if callers can accept an owned type.

### LEAK-06 — `vyre-conform/src/runner/bin/gen_tests.rs:129` — LOW
**Current:**
```rust
.map(|spec| Box::leak(Box::new(spec)) as &'static OpSpec)
```
CLI tool leaks filtered specs before returning them. Bounded by the number of specs matching the filter (≤ total op count).  
**Fix:** Document; acceptable for one-shot CLI.

### LEAK-07 — `vyre-conform/src/runner/bin/contribute.rs:24` — LOW
**Current:**
```rust
ContributeFilter::Op(Box::leak(op.into_boxed_str()))
```
Leaks the `--op` CLI argument string. One string per invocation.  
**Fix:** Acceptable for one-shot CLI; document.

### LEAK-08 — `vyre-conform/src/generate/archetypes/arithmetic.rs:16` — LOW
**Current:**
```rust
let label = Box::leak(format!("{}_{}", op.id, self.id()).into_boxed_str());
```
Leaks a formatted label for every archetype materialization. Bounded by (ops × archetypes).  
**Fix:** Acceptable for test-generation; document bound.

### LEAK-09 — `vyre-conform/src/spec/primitive/common.rs:231` — LOW
**Current:**
```rust
Some(Box::leak(rows.into_boxed_slice()))
```
KAT spec rows are leaked into `&'static [SpecRow]`. Rows are constructed once per primitive spec and live for the process.  
**Fix:** Acceptable for static spec tables; document.

### LEAK-10 — `vyre-conform/src/spec/primitive/common.rs:397-407` — LOW
**Current:**
```rust
fn leak_bytes(bytes: Vec<u8>) -> &'static [u8] { Box::leak(bytes.into_boxed_slice()) }
fn leak_input_slices(inputs: Vec<&'static [u8]>) -> &'static [&'static [u8]] { Box::leak(inputs.into_boxed_slice()) }
fn leak_str(text: impl Into<String>) -> &'static str { Box::leak(text.into().into_boxed_str()) }
```
Three helper functions centralizing deliberate leaks for spec construction.  
**Fix:** Document that these are spec-table leaks with cardinality bounded by the static rule set.

### LEAK-11 — `vyre-conform/src/spec/engine_specs/dfa.rs:160-163` — LOW
**Current:**
```rust
fn vec_to_leaked_slice<T>(v: Vec<T>) -> &'static [T] {
    let boxed = v.into_boxed_slice();
    Box::leak(boxed)
}
```
Used during DFA spec deserialization. Bounded by number of DFA engine specs.  
**Fix:** Document.

### LEAK-12 — `vyre-conform/src/spec/engine_specs/eval/serde.rs:216-218` — LOW
**Current:**
```rust
fn vec_to_leaked_slice<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}
```
Same helper pattern as LEAK-11 for eval deserialization.  
**Fix:** Document.

### LEAK-13 — `vyre-conform-generate/src/verify/harnesses/backend.rs:71-94` — MEDIUM
**Current:**
```rust
static REGISTRY: Mutex<Vec<&'static dyn HarnessBackend>> = Mutex::new(Vec::new());
/// Backends must be leaked to 'static lifetime (e.g. via [Box::leak]) before registration...
pub fn register_backend(backend: &'static dyn HarnessBackend) { ... }
```
The registry *requires* callers to leak backends. This is a design-level leak contract. Every registered backend lives forever.  
**Fix:** Document the contract and the expected upper bound (usually <10 backends). Consider `Arc<dyn HarnessBackend>` if lifetime erasure is not strictly required.

### LEAK-14 — `vyre-conform/src/verify/harnesses/backend.rs:71-94` — MEDIUM
**Current:** Identical duplicate of LEAK-13 in the sibling crate `vyre-conform`.  
**Fix:** Same as LEAK-13.

### LEAK-15 — `vyre-conform/tests/loom_certificate_writer.rs:13-14` — LOW
**Current:**
```rust
let reporters: &'static [loom::sync::Mutex<Box<dyn Reporter>>] =
    Box::leak(vec![reporter].into_boxed_slice());
```
Test-only leak inside a `loom::model` closure. Bounded by test execution.  
**Fix:** Acceptable for test code.

---

## Category B — `OnceLock<T>` / `LazyLock<T>` holding heap resources (never cleared)

### LEAK-16 — `vyre-conform/src/runner/backend/wgpu/context.rs:17` — HIGH
**Current:**
```rust
static GPU: OnceLock<Mutex<Option<Arc<GpuContext>>>> = OnceLock::new();
```
`GpuContext` contains a `wgpu::Device` and `wgpu::Queue`. Once set, it is never dropped. Even `clear_gpu_cache` only sets the `Option` to `None` inside the `Mutex`, but the `Arc<GpuContext>` may still have outstanding clones from `get_gpu()` callers. No forced eviction path.  
**Fix:** Track active clones with `Arc::strong_count`; refuse to clear while >1, or switch to a weak-cache pattern.

### LEAK-17 — `vyre-wgpu/src/pipeline.rs:41-43` — HIGH
**Current:**
```rust
static PIPELINE_CACHE: LazyLock<
    [RwLock<FxHashMap<PipelineCacheKey, Arc<CachedPipeline>>>; PIPELINE_CACHE_SHARDS],
> = LazyLock::new(|| ...);
```
Compiled `wgpu::ComputePipeline`, `BindGroupLayout`, and metadata are cached per `(shader_hash, device)` pair. Shards are bounded to 8 entries each, but entries are `Arc<CachedPipeline>` — the underlying GPU pipeline objects are only dropped when the `Arc` count reaches zero. Because `PipelineCacheKey` stores a full `wgpu::Device` (not an `Arc<Device>`), unique devices create unique keys even if they represent the same physical adapter, causing duplicate entries.  
**Fix:** Key by device fingerprint (e.g., `adapter_info.name` + `adapter_info.device_id`) instead of the opaque `wgpu::Device` handle.

### LEAK-18 — `vyre-core/src/ops/registry/registry.rs:47` — MEDIUM
**Current:**
```rust
static RUNTIME_REGISTRY: OnceLock<RwLock<Vec<&'static OpSpec>>> = OnceLock::new();
```
Runtime-registered op specs accumulate forever. Bounded by the number of external crates calling `register_op_spec`, but no eviction or duplicate detection.  
**Fix:** Add a `HashSet<OpId>` deduplication guard, or document the expected upper bound.

### LEAK-19 — `vyre-conform/src/runner/replay.rs:166` — MEDIUM
**Current:**
```rust
static REPLAY_SENDER: OnceLock<Option<SyncSender<AppendJob>>> = OnceLock::new();
```
The background writer thread is spawned at line 174, but its `JoinHandle` is discarded (`Ok(_) => Some(tx)`). The `SyncSender` lives forever in the `OnceLock`. The thread is never joined, so on process exit any buffered `BufWriter` data may not be flushed.  
**Fix:** Store the `JoinHandle` in a second `OnceLock` and expose a `flush_and_join_replay()` shutdown hook.

### LEAK-20 — `vyre-conform/src/enforce/enforcers/layer8_feedback_loop.rs:11-12` — MEDIUM
**Current:**
```rust
static MUTATION_CACHE: LazyLock<Mutex<HashMap<MutationProbeKey, bool>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
```
Caches mutation-probe results keyed by `(source_hash, test_hash, mutation)`. Unbounded growth if the enforcement suite processes many distinct source files.  
**Fix:** Add an LRU cap (e.g., 10 000 entries) or a `clear_mutation_cache()` API called between batches.

### LEAK-21 — `vyre-conform/src/runner/backend/wgpu/dispatch.rs:28` — LOW
**Current:**
```rust
static SEEN: LazyLock<RwLock<HashSet<u64>>> = LazyLock::new(|| RwLock::new(HashSet::new()));
```
WGSL validation cache. Grows with the number of distinct shaders. In practice bounded to hundreds, but theoretically unbounded.  
**Fix:** Acceptable for current scale; document expected cardinality.

### LEAK-22 — `vyre-conform-generate/src/proof/algebra/mandatory_inference.rs:15` — LOW
**Current:**
```rust
static CACHE: Mutex<Option<HashMap<CacheKey, Vec<AlgebraicLaw>>>> = Mutex::new(None);
```
Caches inferred algebraic laws per `(op_id, is_binary, fn_hash)`. Bounded by the number of operations with algebra proofs (low cardinality).  
**Fix:** Document bound.

### LEAK-23 — `vyre-conform/src/proof/algebra/mandatory_inference.rs:15` — LOW
**Current:** Identical duplicate of LEAK-22 in sibling crate `vyre-conform`.  
**Fix:** Document bound.

### LEAK-24 — `vyre-wgpu/src/runtime/shader/compile_compute_pipeline.rs:103-104` — MEDIUM
**Current:**
```rust
static DRIVER_CACHES: LazyLock<Mutex<HashMap<wgpu::Device, wgpu::PipelineCache>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
```
One `wgpu::PipelineCache` per unique `wgpu::Device`. If `cached_device()` (LEAK-01) creates many devices, this map grows in lockstep. No eviction.  
**Fix:** Use `Weak` device references as keys, or clear entries when device refcount drops.

### LEAK-25 — `vyre-wgpu/src/engine/streaming.rs:34` — MEDIUM
**Current:**
```rust
static POOL: LazyLock<StreamingPool> = LazyLock::new(StreamingPool::new);
```
`StreamingPool::new` spawns 1–4 worker threads (line 46). The threads run an infinite `loop { ... }` and are never joined. The `LazyLock` keeps the pool alive for the process lifetime.  
**Fix:** Acceptable for a global executor, but add a `Drop` impl on `StreamingPool` that sends a termination signal and joins workers.

### LEAK-26 — `vyre-core/tests/support/workgroup_gpu.rs:6` — LOW
**Current:**
```rust
static GPU: LazyLock<(wgpu::Device, wgpu::Queue)> = LazyLock::new(init_required_gpu);
```
Test-only GPU resource held for the entire test process.  
**Fix:** Acceptable for test support code.

---

## Category C — Detached threads (`std::thread::spawn` without join)

### LEAK-27 — `vyre-conform/src/meta/harness.rs:257` — HIGH
**Current:**
```rust
std::thread::spawn(move || {
    let output = child.wait_with_output();
    let _ = tx.send(output);
});
```
The join handle is discarded. The thread only exits when `child.wait_with_output()` returns (on process exit or timeout-kill at line 289). If the child process panics or hangs, the thread lingers silently.  
**Fix:** Store the `JoinHandle` and join it before returning `TestOutcome::Hung` or `TestOutcome::Pass`.

### LEAK-28 — `vyre-conform/src/runner/replay.rs:172-174` — MEDIUM
**Current:**
```rust
match thread::Builder::new()
    .name("vyre-replay-writer".into())
    .spawn(move || writer_loop(rx))
{
    Ok(_) => Some(tx),
    ...
}
```
The `JoinHandle` from `thread::Builder::spawn` is dropped immediately (`Ok(_)`). The writer thread may hold buffered `BufWriter`s that are not flushed on abnormal process exit.  
**Fix:** Store the handle in a static `OnceLock<JoinHandle<()>>` and join it in a `replay_shutdown()` function.

---

## Category D — `wgpu::Buffer` created without pool (bypassing `BufferPool`)

The workspace has a `BufferPool` (`vyre-wgpu/src/runtime/cache/buffer_pool.rs`) that correctly returns buffers on `Drop`. The following call sites create buffers directly via `device.create_buffer` / `device.create_buffer_init`, bypassing the pool. Each dispatch allocates fresh GPU memory; under high throughput this causes allocator churn and implicit driver-level fragmentation even though Rust `Drop` eventually frees the `wgpu::Buffer`.

### LEAK-29 — `vyre-wgpu/src/engine/decompress/dispatch_kernel/buffers.rs:48-67` — MEDIUM
**Current:**
```rust
device.create_buffer_init(&wgpu::util::BufferInitDescriptor { ... })
device.create_buffer(&wgpu::BufferDescriptor { ... })
```
`create_storage_buffer` and `raw_buffer` are direct allocations.  
**Fix:** Route through `BufferPool::global().acquire(...)` when size classes match common shapes.

### LEAK-30 — `vyre-wgpu/src/engine/string_matching/offsets.rs:10-30` — MEDIUM
**Current:**
```rust
device.create_buffer_init(&wgpu::util::BufferInitDescriptor { ... })
```
`storage_init`, `byte_storage_init`, and `zeroed_buffer` all allocate directly.  
**Fix:** Use pooled buffers for fixed-size parameter blocks.

### LEAK-31 — `vyre-wgpu/src/engine/dfa/buffers.rs:19-46` — MEDIUM
**Current:**
```rust
device.create_buffer(&wgpu::BufferDescriptor { ... })
device.create_buffer_init(&wgpu::util::BufferInitDescriptor { ... })
```
`buffer` and `storage_buffer` are direct allocations used by every DFA dispatch.  
**Fix:** Pool the 4 standard DFA binding buffers (input, matches, match_count, params).

### LEAK-32 — `vyre-wgpu/src/pipeline_compound.rs:95-162` — MEDIUM
**Current:**
```rust
device.create_buffer_init(&wgpu::util::BufferInitDescriptor { ... })
```
Output and intermediate buffers are created per dispatch. `live_buffers` keeps them alive until readback, but they are never reused.  
**Fix:** Introduce a compound-dispatch buffer cache keyed by `(program_id, output_bytes)`.

### LEAK-33 — `vyre-wgpu/src/engine/dataflow.rs:397-408` — MEDIUM
**Current:**
```rust
let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor { ... });
let readback_count_buffer = device.create_buffer(&wgpu::BufferDescriptor { ... });
```
Readback pair allocated per `dispatch_and_copy_record` call.  
**Fix:** Pool readback buffers of standard sizes (4 B, 16 B, 64 B, 1 KiB).

### LEAK-34 — `vyre-wgpu/src/engine/dataflow/bfs/bfs_reachability.rs:243-254` — MEDIUM
**Current:**
```rust
let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor { ... });
let readback_count_buffer = device.create_buffer(&wgpu::BufferDescriptor { ... });
```
Empty-path readback buffers allocated even when `sources.is_empty()`.  
**Fix:** Return a static zero-filled readback placeholder instead of allocating GPU memory for the no-op case.

### LEAK-35 — `vyre-wgpu/src/engine/dfa/scan_record.rs:103-112` — MEDIUM
**Current:**
```rust
let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { ... });
let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { ... });
```
Per-record input and params buffers allocated directly.  
**Fix:** Pool small fixed-size parameter buffers; reuse input buffers when scanning the same corpus repeatedly.

---

## Category E — File handles

No true file-handle leaks were found. All `File::open`, `File::create`, and `OpenOptions::open` calls are either:
- Moved into `BufReader`/`BufWriter` and dropped at end of scope, or
- Part of the bounded replay-writer cache (`vyre-conform/src/runner/replay.rs:191`) which flushes and drains on channel close.

One suboptimal pattern was noted at `vyre-conform/src/runner/streaming/regression_sinking.rs:105` (`File::open` on a directory in a loop for `sync_all`), but the handle is dropped immediately and does not constitute a leak.

---

## Category F — Tracing spans

No span leaks were found. All `tracing::info_span!` calls are followed by `.enter()` or `drop(_entered)` before returning. There are no `.await` points inside entered spans.

---

## Category G — `Arc<T>` cycles / `Arc<Weak<T>>` issues

No `Arc` reference cycles were found. The only `Weak` usage is in `vyre-wgpu/src/runtime/cache/buffer_pool.rs:33` (`PooledBuffer` → `Weak<BufferPoolInner>`), which is correctly paired with `Arc::downgrade` at line 127 and `Weak::upgrade` inside `Drop` at line 201. No struct holds a reciprocal `Arc` back to its parent.

---

## Category H — `std::mem::forget` / `ManuallyDrop` / `into_raw`

Zero occurrences of `std::mem::forget`, `ManuallyDrop::new`, or `into_raw` (without matching `from_raw`) were found in production code. `ManuallyDrop` appears only as a string literal in an enforcer pattern denylist.

---

## Summary Table

| ID | File | Line | Severity | Category |
|----|------|------|----------|----------|
| LEAK-01 | `vyre-wgpu/src/runtime/device/device.rs` | 20 | CRITICAL | `Box::leak` |
| LEAK-02 | `vyre-wgpu/src/runtime/device/device.rs` | 33 | CRITICAL | `Box::leak` |
| LEAK-03 | `vyre-wgpu/src/runtime/device/device.rs` | 144-147 | CRITICAL | `OnceLock` + accumulation |
| LEAK-04 | `vyre-conform/src/runner/calibrate.rs` | 134 | MEDIUM | `Box::leak` |
| LEAK-05 | `vyre-conform/src/runner/calibrate.rs` | 455-457 | LOW | `Box::leak` |
| LEAK-06 | `vyre-conform/src/runner/bin/gen_tests.rs` | 129 | LOW | `Box::leak` |
| LEAK-07 | `vyre-conform/src/runner/bin/contribute.rs` | 24 | LOW | `Box::leak` |
| LEAK-08 | `vyre-conform/src/generate/archetypes/arithmetic.rs` | 16 | LOW | `Box::leak` |
| LEAK-09 | `vyre-conform/src/spec/primitive/common.rs` | 231 | LOW | `Box::leak` |
| LEAK-10 | `vyre-conform/src/spec/primitive/common.rs` | 397-407 | LOW | `Box::leak` |
| LEAK-11 | `vyre-conform/src/spec/engine_specs/dfa.rs` | 160-163 | LOW | `Box::leak` |
| LEAK-12 | `vyre-conform/src/spec/engine_specs/eval/serde.rs` | 216-218 | LOW | `Box::leak` |
| LEAK-13 | `vyre-conform-generate/src/verify/harnesses/backend.rs` | 71-94 | MEDIUM | `Box::leak` (design contract) |
| LEAK-14 | `vyre-conform/src/verify/harnesses/backend.rs` | 71-94 | MEDIUM | `Box::leak` (design contract) |
| LEAK-15 | `vyre-conform/tests/loom_certificate_writer.rs` | 13-14 | LOW | `Box::leak` (test) |
| LEAK-16 | `vyre-conform/src/runner/backend/wgpu/context.rs` | 17 | HIGH | `OnceLock` |
| LEAK-17 | `vyre-wgpu/src/pipeline.rs` | 41-43 | HIGH | `LazyLock` |
| LEAK-18 | `vyre-core/src/ops/registry/registry.rs` | 47 | MEDIUM | `OnceLock` |
| LEAK-19 | `vyre-conform/src/runner/replay.rs` | 166 | MEDIUM | `OnceLock` + detached thread |
| LEAK-20 | `vyre-conform/src/enforce/enforcers/layer8_feedback_loop.rs` | 11-12 | MEDIUM | `LazyLock` |
| LEAK-21 | `vyre-conform/src/runner/backend/wgpu/dispatch.rs` | 28 | LOW | `LazyLock` |
| LEAK-22 | `vyre-conform-generate/src/proof/algebra/mandatory_inference.rs` | 15 | LOW | `static` |
| LEAK-23 | `vyre-conform/src/proof/algebra/mandatory_inference.rs` | 15 | LOW | `static` |
| LEAK-24 | `vyre-wgpu/src/runtime/shader/compile_compute_pipeline.rs` | 103-104 | MEDIUM | `LazyLock` |
| LEAK-25 | `vyre-wgpu/src/engine/streaming.rs` | 34 | MEDIUM | `LazyLock` + detached threads |
| LEAK-26 | `vyre-core/tests/support/workgroup_gpu.rs` | 6 | LOW | `LazyLock` (test) |
| LEAK-27 | `vyre-conform/src/meta/harness.rs` | 257 | HIGH | detached thread |
| LEAK-28 | `vyre-conform/src/runner/replay.rs` | 172-174 | MEDIUM | detached thread |
| LEAK-29 | `vyre-wgpu/src/engine/decompress/dispatch_kernel/buffers.rs` | 48-67 | MEDIUM | GPU buffer bypass |
| LEAK-30 | `vyre-wgpu/src/engine/string_matching/offsets.rs` | 10-30 | MEDIUM | GPU buffer bypass |
| LEAK-31 | `vyre-wgpu/src/engine/dfa/buffers.rs` | 19-46 | MEDIUM | GPU buffer bypass |
| LEAK-32 | `vyre-wgpu/src/pipeline_compound.rs` | 95-162 | MEDIUM | GPU buffer bypass |
| LEAK-33 | `vyre-wgpu/src/engine/dataflow.rs` | 397-408 | MEDIUM | GPU buffer bypass |
| LEAK-34 | `vyre-wgpu/src/engine/dataflow/bfs/bfs_reachability.rs` | 243-254 | MEDIUM | GPU buffer bypass |
| LEAK-35 | `vyre-wgpu/src/engine/dfa/scan_record.rs` | 103-112 | MEDIUM | GPU buffer bypass |

---

## Recommendations (in priority order)

1. **Fix LEAK-01/02/03 immediately.** The `cached_device()` family is the only unbounded growth path in production runtime code. Replace with a single `LazyLock` singleton or remove the API entirely.
2. **Fix LEAK-27.** The meta-harness detached thread can silently swallow panics from `cargo test` child processes. Always join.
3. **Fix LEAK-19/28.** The replay writer thread needs a shutdown hook; data loss is possible on SIGTERM.
4. **Add eviction to LEAK-17.** Pipeline cache keys should use device fingerprints, not opaque handles.
5. **Document bounds for all LOW findings.** Every `Box::leak` and `LazyLock` should carry a doc comment stating the cardinality bound and why it is safe.
