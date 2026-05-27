# PERF HOT-PATH ALLOCATION AUDIT  -  vyre GPU dispatch

**Date:** 2026-04-23  
**Scope:** Every file on the GPU dispatch hot path (vyre-core → vyre-driver → vyre-driver-wgpu → vyre-runtime → vyre-foundation validate)  
**Method:** Static code review of every `Vec`, `String`, `format!`, `to_string`, `to_vec`, `HashMap`, `Box`, `Arc::new`, and channel construction reachable from `VyreBackend::dispatch` / `CompiledPipeline::dispatch_borrowed`.  
**Assumptions for sizing:** Typical small dispatch = 4 buffer bindings, 1 output, 4 KiB average buffer, 16 nodes, pipeline-cache **hit** (the common case). Sizes are heap allocations only; stack spills from `SmallVec` are noted but not counted.

---

## Findings (numbered by severity × frequency)

### **PERF-HOT-01** | PER-DISPATCH | `vyre-driver-wgpu/src/pipeline.rs:247` | Root cause: `BindGroupCache` (moka LRU + atomic counters) instantiated fresh on every pipeline-cache hit  
`compile_with_device_queue` returns `Arc::new(Self { bind_group_cache: Arc::new(BindGroupCache::default()), ... })` even when the compiled artifact is a cache hit. `BindGroupCache::default()` calls `moka::sync::Cache::builder().max_capacity(1024).build()`, which allocates internal segment tables (~8–16 KiB), eviction channels, and atomic bookkeeping. Because the cache is never shared across dispatches, every single dispatch pays this cost and **every bind group is a cache miss**.

**Fix:** Move `bind_group_cache` into `CachedPipelineArtifact` so it is shared across all `WgpuPipeline` instances that reference the same compiled shader. Return `Arc::clone(&cached_pipeline.bind_group_cache)` instead of `Arc::new(BindGroupCache::default())`.

---

### **PERF-HOT-02** | PER-BUFFER | `vyre-driver-wgpu/src/engine/record_and_readback.rs:193` | Root cause: `vec![0u8; size]` zero-fill allocation inside the buffer-binding loop  
For every non-output buffer that lacks input data, the code does:
```rust
queue.write_buffer(handle, 0, &vec![0u8; size]);
```
This allocates a zeroed `Vec<u8>` on the CPU heap for **every** binding, every dispatch. A 4 KiB buffer = 4 KiB alloc + memset + PCIe upload of zeros.

**Fix:** Replace with `encoder.clear_buffer(handle, 0, Some(size as u64))` (GPU-side zero, no CPU alloc). Where `CommandEncoder` is not yet available, use a thread-local `static ZERO_PAD: [u8; 4096] = [0; 4096]` and write in chunks.

---

### **PERF-HOT-03** | PER-BUFFER | `vyre-driver-wgpu/src/engine/record_and_readback.rs:383` | Root cause: `mapped[...].to_vec()` copies readback into a new `Vec<u8>` per output  
After GPU mapping succeeds, the readback path does `outputs.push(mapped[trim_start..end].to_vec())`, allocating a new `Vec` and copying the visible slice. For a 4 KiB output this is 4 KiB alloc + memcpy.

**Fix:** Pre-size the outer `outputs` vec and reuse a thread-local `Vec<u8>` scratch buffer, or return `Vec<Vec<u8>>` via `bytes.split_off()` from a single pre-allocated staging vec where possible. At minimum, `outputs.push(Vec::with_capacity(end - trim_start))` then `extend_from_slice` to avoid double alloc.

---

### **PERF-HOT-04** | PER-BUFFER | `vyre-driver-wgpu/src/pipeline_persistent.rs:426` | Root cause: `legacy_contents` allocates `vec![0u8; len]` padded buffer per binding  
`legacy_contents` returns a freshly allocated `Vec<u8>` padded to alignment for every buffer binding, every dispatch:
```rust
let mut padded = vec![0u8; len];
```
With 4 buffers × 4 KiB = 16 KiB of CPU heap churn per dispatch.

**Fix:** Remove `legacy_contents` entirely; write input bytes directly via `queue.write_buffer` and use `encoder.clear_buffer` for padding. If padding is unavoidable, use a thread-local reusable `Vec<u8>` resized with `resize(len, 0)`.

---

### **PERF-HOT-05** | PER-DISPATCH | `vyre-driver-wgpu/src/engine/record_and_readback.rs:79` | Root cause: `FxHashMap<u32, PooledBuffer>` created per dispatch for GPU buffer lookup  
```rust
let mut gpu_buffers: rustc_hash::FxHashMap<u32, PooledBuffer> =
    FxHashMap::with_capacity_and_hasher(request.buffer_bindings.len(), Default::default());
```
A `FxHashMap` allocates its bucket array (~256 bytes for 4 entries) and then grows. This is pure overhead: the keys are sequential binding slots and the values are consumed in declaration order.

**Fix:** Replace with a `SmallVec<[(u32, PooledBuffer); 8]>` or a fixed-size array indexed by binding slot (bindings are dense 0..N). Only fall back to `HashMap` when `buffer_bindings.len() > 16`.

---

### **PERF-HOT-06** | PER-DISPATCH | `vyre-driver-wgpu/src/engine/record_and_readback.rs:72` | Root cause: `FxHashMap<u32, usize>` reverse-lookup map for input indices  
```rust
let input_idx_by_binding: rustc_hash::FxHashMap<u32, usize> = input_bindings
    .iter()
    .enumerate()
    .map(|(idx, info)| (info.binding, idx))
    .collect();
```
Allocates a second `FxHashMap` (~128 bytes) to avoid an O(N²) scan. The scan is already gone, but the map itself is still allocated and dropped per dispatch.

**Fix:** Because buffer declarations are dense and small, pre-compute the mapping into a `[(u32, usize); 8]` on the stack, or simply do a linear scan over `input_bindings` (already a `Vec` of at most ~8 elements)  -  the constant factor beats the allocator.

---

### **PERF-HOT-07** | PER-DISPATCH | `vyre-driver-wgpu/src/engine/record_and_readback.rs:305-346` | Root cause: `std::sync::mpsc::channel()` allocated per readback buffer  
For every output buffer the dispatch path allocates a new `mpsc::channel` to wait for `map_async`:
```rust
let (sender, receiver) = std::sync::mpsc::channel();
slice.map_async(..., move |result| { let _ = sender.send(result); });
pending.push((output, readback_buffer, receiver));
```
A channel is ~200–400 bytes of heap (buffer + mutex + condvar). With 1 output = 1 channel; with N outputs = N channels.

**Fix:** Use a single `mpsc::channel` and tag each message with an output index, or batch all `map_async` callbacks into a single `std::sync::Barrier` / atomic counter. Better: replace channels with `wgpu::Buffer::map_async` + a single `poll(Maintain::Wait)` and then iterate slices without callbacks (wgpu guarantees mapping is complete after the wait).

---

### **PERF-HOT-08** | PER-DISPATCH | `vyre-driver-wgpu/src/pipeline.rs:639` | Root cause: `Vec<&[u8]>` collected from owned inputs in `WgpuPipeline::dispatch`  
```rust
let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
self.dispatch_borrowed(&borrowed, config)
```
The `CompiledPipeline::dispatch` default path already does this, and `WgpuPipeline::dispatch` duplicates it. Every call allocates a `Vec` of references even when the caller already had borrowed slices.

**Fix:** Remove `WgpuPipeline::dispatch` entirely and rely on the `CompiledPipeline` default, or change `dispatch` to be an inline no-op that forwards directly to `dispatch_borrowed` without collecting.

---

### **PERF-HOT-09** | PER-DISPATCH | `vyre-driver-wgpu/src/lib.rs:520` / `backend_impl.rs:354` | Root cause: `Vec<&[u8]>` collected in `WgpuBackend::dispatch`  
```rust
let borrowed: Vec<&[u8]> = inputs.iter().map(Vec::as_slice).collect();
self.dispatch_borrowed(program, &borrowed, config)
```
Same pattern as PERF-HOT-08 but at the backend entry point. This is redundant because `dispatch_borrowed` is the preferred path and `dispatch` only exists for trait conformance.

**Fix:** Replace body with a direct call to `dispatch_borrowed` using a stack array for up to 8 inputs:
```rust
let borrowed: smallvec::SmallVec<[&[u8]; 8]> = inputs.iter().map(Vec::as_slice).collect();
self.dispatch_borrowed(program, &borrowed, config)
```

---

### **PERF-HOT-10** | PER-DISPATCH | `vyre-driver-wgpu/src/engine/record_and_readback.rs:61` | Root cause: `Vec<&BufferBindingInfo>` filter-collect per dispatch  
```rust
let input_bindings: Vec<&BufferBindingInfo> = request
    .buffer_bindings
    .iter()
    .filter(|b| b.kind != MemoryKind::Shared && !b.is_output)
    .collect();
```
Allocates a `Vec` of references (~64 bytes for 4 elements) that is immediately consumed to build the hash map.

**Fix:** Fuse the filter into the hash-map construction loop without the intermediate `Vec`.

---

### **PERF-HOT-11** | PER-DISPATCH | `vyre-driver-wgpu/src/pipeline_persistent.rs:206` | Root cause: `Vec<&BufferBindingInfo>` filter-collect in `legacy_handles_from_inputs`  
```rust
let input_bindings: Vec<_> = self
    .buffer_bindings
    .iter()
    .filter(|b| b.kind != MemoryKind::Shared)
    .collect();
```
Same anti-pattern as PERF-HOT-10 in the persistent dispatch path.

**Fix:** Iterate `buffer_bindings` directly and skip shared kinds inline; remove the intermediate `Vec`.

---

### **PERF-HOT-12** | PER-DISPATCH | `vyre-driver-wgpu/src/engine/record_and_readback.rs:202-232` | Root cause: `Vec<wgpu::BindGroup>` + inner `Vec<wgpu::BindGroupEntry>` rebuilt every dispatch  
Bind groups are recreated from scratch every dispatch even when the buffer handles haven't changed:
```rust
let mut bind_groups = Vec::with_capacity(...);
let mut entries = Vec::new();
```

**Fix:** Cache bind groups by buffer-handle id signature in the existing `BindGroupCache` (see PERF-HOT-01). Once the cache is warm, bind-group creation drops to zero.

---

### **PERF-HOT-13** | PER-DISPATCH | `vyre-driver-wgpu/src/engine/record_and_readback.rs:305` | Root cause: `Vec<(OutputBindingLayout, PooledBuffer)>` for readback staging  
```rust
let mut readback_buffers = Vec::with_capacity(request.output_bindings.len());
```
Allocates a vec of tuples for every output. Small but unnecessary.

**Fix:** Use a `SmallVec<[(OutputBindingLayout, PooledBuffer); 4]>`.

---

### **PERF-HOT-14** | PER-DISPATCH | `vyre-driver-wgpu/src/engine/record_and_readback.rs:338` | Root cause: `Vec<(OutputBindingLayout, PooledBuffer, Receiver)>` for pending map_async  
```rust
let mut pending = Vec::with_capacity(readback_buffers.len());
```
Same as PERF-HOT-13; holds the receiver from the per-buffer channel (PERF-HOT-07).

**Fix:** Collapse with PERF-HOT-07 (single channel) and use `SmallVec`.

---

### **PERF-HOT-15** | PER-DISPATCH | `vyre-driver-wgpu/src/engine/streaming.rs:84` | Root cause: `crossbeam_channel::unbounded()` + `Box<dyn FnOnce()>` per streaming chunk  
```rust
let (sender, receiver) = unbounded();
self.sender.send(Box::new(move || { ... }))?;
```
Every `push_chunk` allocates a new MPMC channel and boxes the closure.

**Fix:** Pre-create a bounded job channel with reusable `ChunkJob` structs (enum of pending/finished). Use a thread-local or pool-allocated job object instead of `Box::new`.

---

### **PERF-HOT-16** | PER-DISPATCH | `vyre-driver/src/backend/vyre_backend.rs:157` | Root cause: `Vec<Vec<u8>>` + `to_vec()` in default `dispatch_borrowed`  
```rust
let owned: Vec<Vec<u8>> = inputs.iter().map(|input| (*input).to_vec()).collect();
self.dispatch(program, &owned, config)
```
When a backend does not override `dispatch_borrowed`, every input slice is copied into a new `Vec<u8>`. The wgpu backend overrides this, but passthrough / reference backends do not.

**Fix:** Make `dispatch_borrowed` a required trait method (remove default) so every backend must implement a zero-copy path.

---

### **PERF-HOT-17** | PER-DISPATCH | `vyre-driver/src/backend/vyre_backend.rs:217` | Root cause: `Box<dyn PendingDispatch>` in default `dispatch_async`  
```rust
Ok(Box::new(crate::backend::pending_dispatch::ReadyPending { outputs }))
```
Every async dispatch boxes the pending handle. For backends that don't support true async, this is pure overhead.

**Fix:** Provide an enum-based `PendingDispatch` (`enum PendingDispatch { Ready(...), Async(...) }`) stored inline instead of `Box<dyn>`. This removes the heap allocation and vtable indirection.

---

### **PERF-HOT-18** | PER-DISPATCH | `vyre-driver/src/backend/compiled_pipeline.rs:52` | Root cause: `Vec<Vec<u8>>` + `to_vec()` in default `CompiledPipeline::dispatch_borrowed`  
Same as PERF-HOT-16 at the `CompiledPipeline` trait level.

**Fix:** Remove the default impl and force backends to provide a zero-copy `dispatch_borrowed`.

---

### **PERF-HOT-19** | PER-DISPATCH | `vyre-driver/src/pipeline.rs:284` | Root cause: `Arc::new(program.clone())` + `format!(...)` + `DispatchConfig::clone()` in passthrough compile  
```rust
Ok(Arc::new(PassthroughPipeline {
    id: format!("{}:passthrough", backend.id()),
    backend,
    program,
    compile_config: config.clone(),
}))
```
When a backend does not implement `compile_native`, the framework clones the entire `Program` (Arc<Node> tree + buffers), clones `DispatchConfig` (clones `Option<String>` fields), and formats a String.

**Fix:** Store `program: Arc<Program>` without cloning the inner data (already `Arc` in many places), and store `&'static str` id fragments without `format!`.

---

### **PERF-HOT-20** | PER-DISPATCH | `vyre-driver-wgpu/src/lib.rs:545` / `backend_impl.rs:386` | Root cause: `DispatchConfig::clone()` on timeout path  
```rust
dispatch_config = config.clone();
dispatch_config.timeout = Some(remaining);
```
`DispatchConfig` contains three `Option<String>` fields; `clone()` allocates when any are `Some`. Even on the happy path (no timeout) the branch is checked; on the timeout path the clone is unconditional.

**Fix:** Replace `Option<String>` in `DispatchConfig` with `Option<&'static str>` or `Option<Arc<str>>` so clone is refcount-only. Alternatively, make `DispatchConfig` `Copy` by using fixed-size string buffers or `SmolStr`.

---

### **PERF-HOT-21** | PER-DISPATCH | `vyre-driver-wgpu/src/pipeline.rs:383-397` | Root cause: New `Arc<WgpuPipeline>` allocated on every cache hit  
Even when the compiled artifact is cached, a brand-new `Arc<WgpuPipeline>` is boxed and returned. This is ~200 bytes of heap allocation plus the `Arc` refcount increment.

**Fix:** Store `WgpuPipeline` inside `CachedPipelineArtifact` and return `Arc::clone(&artifact.pipeline_wrapper)` directly, avoiding the fresh allocation.

---

### **PERF-HOT-22** | PER-COMPILE (cold path, but triggered on first dispatch) | `vyre-driver-wgpu/src/pipeline.rs:275-329` | Root cause: `Vec<Arc<wgpu::BindGroupLayout>>` + `Vec<wgpu::BindGroupLayoutEntry>` + `format!` on pipeline compilation  
On cache miss, bind-group layout entries are collected into a new `Vec` per group, and layout labels are formatted:
```rust
let mut bind_group_layouts_vec: Vec<Arc<wgpu::BindGroupLayout>> = Vec::with_capacity(...);
let entries: Vec<wgpu::BindGroupLayoutEntry> = ...
    .collect();
label: Some(&format!("vyre P-6 bind group layout {group_index}")),
```

**Fix:** This is cold-path acceptable, but the `format!` should be a `const` label or a stack buffer (`arrayvec::ArrayString`) to avoid heap alloc.

---

### **PERF-HOT-23** | PER-DISPATCH (error path) | `vyre-driver/src/backend/error.rs:158-166` | Root cause: `BackendError::new` allocates `String` for every error  
```rust
pub fn new(message: impl Into<String>) -> Self {
    let message = message.into();
    ...
    Self::Raw(format!("{message}. Fix: include backend-specific recovery guidance."))
}
```
Any validation failure, timeout, or GPU error triggers at least one `String` allocation, often two (the incoming message + the wrapped format).

**Fix:** Replace `BackendError::Raw(String)` with `BackendError::Raw(Cow<'static, str>)` or a dedicated `Arc<str>` so errors can carry static strings without allocation. Only allocate when dynamic data (e.g., buffer sizes) must be interpolated.

---

### **PERF-HOT-24** | PER-DISPATCH (error path) | `vyre-driver-wgpu/src/backend_impl.rs:376` / `lib.rs:548` | Root cause: Timeout error string interpolates `elapsed` + `deadline`  
```rust
return Err(vyre::BackendError::new(format!(
    "dispatch cancelled after DispatchConfig.timeout before GPU submission: took {elapsed:?}, budget {deadline:?}. ..."
)));
```
On timeout expiry, a large format string is allocated.

**Fix:** Use `Cow<'static, str>` and pre-baked error templates; only format the dynamic numbers into a small scratch buffer.

---

### **PERF-HOT-25** | PER-DISPATCH (error path) | `vyre-driver-wgpu/src/lib.rs:376` / `backend_impl.rs:375` | Root cause: `validate_with_cache` error path allocates `error.to_string()`  
```rust
vyre_driver::backend::validation::validate_program(program, self).map_err(|error| {
    vyre::BackendError::InvalidProgram { fix: error.to_string() }
})?;
```
On validation failure, the `ValidationError` message is copied into a new `String`.

**Fix:** Store `fix: Arc<str>` or `Cow<'static, str>` in `BackendError::InvalidProgram`.

---

### **PERF-HOT-26** | PER-VALIDATION | `vyre-foundation/src/validate/validate.rs:56` | Root cause: `FxHashSet` + `FxHashMap` allocated per `validate_with_options` call  
```rust
let mut seen_names = FxHashSet::default();
let mut seen_bindings = FxHashSet::default();
let mut buffer_map: FxHashMap<&str, &BufferDecl> = FxHashMap::default();
let mut scope = FxHashMap::default();
```
Even though `validate_with_cache` avoids redundant validation, the first dispatch of every program still pays for four hash-map/set allocations.

**Fix:** Pre-size the sets with `with_capacity_and_hasher` (already partially done) and consider a thread-local reusable `Scope` object that is `clear()`ed between calls.

---

### **PERF-HOT-27** | PER-COMPILE (first dispatch) | `vyre-foundation/src/execution_plan.rs:209` | Root cause: `ExecutionPlan` allocates `Vec<TrackDecision>` + `Vec<BufferPlan>` with `String` names  
`plan()` constructs a large tree of owned data:
- `Vec<TrackDecision>` (7 elements)
- `MemoryPlan { buffers: Vec<BufferPlan> }` where each `BufferPlan` has `name: String`
- `FusionPlan { entry_op_id: Option<String> }`

**Fix:** Store names as `Arc<str>` inside `BufferPlan` and `TrackDecision::reason` as `&'static str` (already is). `entry_op_id` should be `Arc<str>`.

---

### **PERF-HOT-28** | PER-COMPILE (first dispatch) | `vyre-runtime/src/pipeline_cache.rs:258-289` | Root cause: `canonical_wire` clones `Program` into `Vec<BufferDecl>` + `Vec<Node>` + `Vec<u8>`  
```rust
let canonical = vyre_foundation::transform::optimize::canonicalize::run(program.clone());
let mut sorted_buffers = canonical.buffers().to_vec();
let entry = canonical.entry().to_vec();
let normalised = Program::wrapped(sorted_buffers, workgroup, entry);
normalised.to_wire().expect(...)
```
Fingerprinting clones the entire program tree, sorts buffers into a new vec, clones the entry node vec, and serializes to a fresh `Vec<u8>`.

**Fix:** Compute the fingerprint from an in-place sort of a borrowed slice (using a scratch stack array for small programs) and stream-hash via `blake3::Hasher::update` instead of `to_wire()` into an intermediate `Vec<u8>`.

---

### **PERF-HOT-29** | PER-LOWERING (first dispatch / cache miss) | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:93` | Root cause: `HashSet<String>` for atomic target scan  
```rust
let mut atomic_targets = HashSet::<String>::new();
```
Allocated fresh for every WGSL lowering. Only relevant on cache miss, but a cache miss is expensive already.

**Fix:** Use `FxHashSet<Arc<str>>` (buffer names are already `Arc<str>` in the IR) to avoid cloning strings into the set.

---

### **PERF-HOT-30** | PER-LOWERING (first dispatch / cache miss) | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:831-836` | Root cause: `format!` per temporary variable in Naga emission  
```rust
pub(crate) fn next_temp_name(&mut self, prefix: &str) -> String {
    let name = format!("__vyre_{prefix}_{}", self.temp_counter);
    self.temp_counter += 1;
    name
}
```
Every SSA temporary gets a freshly formatted `String`. A 100-node program can generate hundreds of temporaries.

**Fix:** Use a thread-local `arrayvec::ArrayString<32>` or pre-allocate a `String` scratch buffer and `write!` into it, returning a `SmolStr` / `Arc<str>`.

---

### **PERF-HOT-31** | PER-DISPATCH | `vyre-driver-wgpu/src/pipeline_bindings.rs:16` | Root cause: `Vec<(u32, u32)>` allocated for WGSL binding reflection  
```rust
pub(crate) fn declared_bindings(wgsl: &str) -> Vec<(u32, u32)> {
    let mut bindings: Vec<(u32, u32)> = Vec::with_capacity(4);
```
This is called during pipeline compilation; on cache hit it is skipped, but if any caller invokes it directly it allocates.

**Fix:** Return `SmallVec<[(u32, u32); 8]>`.

---

### **PERF-HOT-32** | PER-DISPATCH | `vyre-driver-wgpu/src/buffer/handle.rs:126-131` | Root cause: GPU readback buffer allocation in `readback_until`  
```rust
let readback = device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("vyre persistent handle readback"),
    size: read_len,
    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    mapped_at_creation: false,
});
```
Allocates a GPU-side buffer for every `readback_until` call. This is GPU heap, not CPU heap, but still a per-dispatch allocation.

**Fix:** Use the buffer pool (`BufferPool::acquire`) for readback staging buffers, or reuse a single persistent MAP_READ buffer sized to the max output.

---

### **PERF-HOT-33** | PER-DISPATCH | `vyre-driver-wgpu/src/buffer/handle.rs:138` | Root cause: `mpsc::channel` per `readback_until` call  
```rust
let (sender, receiver) = std::sync::mpsc::channel();
```
Same pattern as PERF-HOT-07 but in the persistent-buffer readback path.

**Fix:** Same fix as PERF-HOT-07  -  use a single channel or polling without channels.

---

### **PERF-HOT-34** | PER-DISPATCH | `vyre-driver-wgpu/src/runtime/cache/buffer_pool.rs:146-152` | Root cause: `PooledBuffer` allocates `String` label on every acquire  
```rust
Ok(PooledBuffer {
    key,
    id,
    label: label.to_string(),
    buffer: Some(buffer),
    pool: Arc::downgrade(&self.inner),
})
```
`label.to_string()` copies the label into a new `String` for every buffer acquisition, even on pool hits.

**Fix:** Store `label: Arc<str>` or use a `&'static str` for known labels (all labels in the codebase are static strings).

---

### **PERF-HOT-35** | PER-DISPATCH | `vyre-driver-wgpu/src/runtime/cache/buffer_pool.rs:244-261` | Root cause: `FxHashMap` bucket array allocated per size-class on first miss  
`queue_for` does:
```rust
Arc::clone(queues.entry(key.clone()).or_insert_with(|| Arc::new(SegQueue::new())))
```
The first time a given `(device, size_class, usage)` triple is seen, a new `SegQueue` is boxed and a new hash-map entry is allocated. In a long-running process this amortizes, but burst workloads see allocator pressure.

**Fix:** Pre-populate the pool with a fixed array of `SegQueue`s for common size classes (4, 16, 64, 256, 1024, 4096, 16384 bytes) instead of a hash map.

---

## Top 10 by Cost × Frequency (per 1M dispatches)

| Rank | Finding | Cost Model | Est. Bytes / Dispatch | × 1M |
|------|---------|-----------|----------------------|------|
| 1 | **PERF-HOT-01** BindGroupCache recreated | PER-DISPATCH | ~12 KiB (moka internals) | **12.0 GiB** |
| 2 | **PERF-HOT-04** `legacy_contents` padded vec | PER-BUFFER (×4) | ~4 KiB × 4 = 16 KiB | **16.0 GiB** |
| 3 | **PERF-HOT-02** `vec![0u8; size]` zero-fill | PER-BUFFER (×4) | ~4 KiB × 4 = 16 KiB | **16.0 GiB** |
| 4 | **PERF-HOT-03** readback `to_vec()` | PER-OUTPUT (×1) | ~4 KiB | **4.0 GiB** |
| 5 | **PERF-HOT-05** `gpu_buffers` FxHashMap | PER-DISPATCH | ~256 bytes | **256 MiB** |
| 6 | **PERF-HOT-06** `input_idx_by_binding` FxHashMap | PER-DISPATCH | ~128 bytes | **128 MiB** |
| 7 | **PERF-HOT-07** `mpsc::channel` per readback | PER-BUFFER (×1) | ~400 bytes | **400 MiB** |
| 8 | **PERF-HOT-09** `Vec<&[u8]>` in backend dispatch | PER-DISPATCH | ~64 bytes | **64 MiB** |
| 9 | **PERF-HOT-12** bind-group vec rebuild | PER-DISPATCH | ~256 bytes | **256 MiB** |
| 10 | **PERF-HOT-15** streaming channel + boxed closure | PER-CHUNK | ~600 bytes | **600 MiB** |

*Note:* PERF-HOT-04 and PERF-HOT-02 are in different code paths (`pipeline_persistent` vs `record_and_readback`). A typical dispatch goes through **one** of them, not both. The combined worst-case is ~20–24 KiB of CPU heap alloc per dispatch from the top 4 findings alone.

## Total Estimated Allocation per 1M Dispatches

Assuming:
- Pipeline cache **hit** (common case)
- 4 buffer bindings, 1 output, 4 KiB average buffer
- `record_and_readback` path (non-persistent)
- No timeout / no error

| Category | Est. Total Bytes / 1M Dispatches |
|----------|----------------------------------|
| **CPU heap (large)**  -  zero vecs, readback copies, BindGroupCache, padded contents | ~48–52 GiB |
| **CPU heap (small)**  -  hash maps, vec metadata, channels, boxed closures, string labels | ~2–3 GiB |
| **GPU heap**  -  readback staging buffers, command encoders, bind groups | ~8–12 GiB |
| **Grand total** | **~60 GiB** |

## Critical Action Items

1. **BindGroupCache sharing (PERF-HOT-01)**  -  single biggest win. Move cache into `CachedPipelineArtifact`.
2. **Remove `vec![0u8; size]` (PERF-HOT-02)**  -  use `clear_buffer` or static zero slice.
3. **Remove `legacy_contents` (PERF-HOT-04)**  -  write directly without padded CPU vec.
4. **Collapse hash maps to stack arrays / SmallVec (PERF-HOT-05, PERF-HOT-06)**.
5. **Replace per-readback channels with batch poll (PERF-HOT-07, PERF-HOT-33)**.
6. **Make `DispatchConfig` cheap to clone (PERF-HOT-20)**  -  `Arc<str>` for labels.

---
*Commit message template:*
```
audit(vyre): PERF hot-path allocation audit  -  every per-dispatch allocation identified and costed
```
