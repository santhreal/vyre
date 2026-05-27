# VYRE Runtime / Driver / Primitives  -  Read-Only Security Audit

**Scope:** `vyre-runtime`, `vyre-driver`, `vyre-driver-wgpu`, `vyre-driver-spirv`, `vyre-runtime megakernel`, `vyre-primitives`, `vyre-foundation`, `vyre-intrinsics`, `vyre-harness`  
**Ignored:** parser, SurgeC, test-only code (except where test infrastructure panics in production paths), `vyre-frontend-c`, `vyre-libs`, `vyre-macros`, `vyre-spec`.  
**Date:** 2026-04-26

---

## 1. Panics in Public APIs on Malformed Input

### 1.1 `vyre-foundation/src/validate/validate.rs:301`  -  `pop().unwrap()` on empty scope stack
**Status:** fixed in `vyre-foundation/src/validate/validate.rs`.

```rust
Frame::PopScope => {
    let frame = self.scope_stack.pop().unwrap();
    ...
}
```
**Issue:** A malformed `Program` with a `PopScope` frame that has no matching `PushScope` causes an immediate panic in the public `validate()` entry point.  
**Fix:**
```rust
let frame = self.scope_stack.pop().ok_or_else(||
    err("malformed program: PopScope without matching PushScope. Fix: regenerate program from a trusted compiler.".to_string())
)?;
```

### 1.2 `vyre-foundation/src/validate/validate.rs:319`  -  `pop().unwrap()` on empty alias stack
**Status:** fixed in `vyre-foundation/src/validate/validate.rs`.

```rust
Frame::PopAlias => {
    let (reads, atomics) = self.alias_stack.pop().unwrap();
    ...
}
```
**Issue:** Same pattern for alias frames.  
**Fix:**
```rust
let (reads, atomics) = self.alias_stack.pop().ok_or_else(||
    err("malformed program: PopAlias without matching PushAlias. Fix: regenerate program from a trusted compiler.".to_string())
)?;
```

### 1.3 `vyre-foundation/src/validate/validate.rs:326`  -  `last_mut().unwrap()` on empty scope stack
**Status:** fixed in `vyre-foundation/src/validate/validate.rs`.

```rust
Frame::InsertLoopVar(var) => {
    let frame = self.scope_stack.last_mut().unwrap();
    ...
}
```
**Issue:** A loop-var insertion outside any scope panics.  
**Fix:**
```rust
let frame = self.scope_stack.last_mut().ok_or_else(||
    err("malformed program: loop variable outside any scope. Fix: regenerate program from a trusted compiler.".to_string())
)?;
```

### 1.4 `vyre-foundation/src/validate/validate.rs:453`  -  `last_mut().unwrap()` on empty scope stack
**Status:** fixed in `vyre-foundation/src/validate/validate.rs`.

```rust
fn visit_let(... ) {
    let frame = self.scope_stack.last_mut().unwrap();
    ...
}
```
**Issue:** Same as 1.3, reachable via `visit_let`.  
**Fix:** Same pattern as 1.3.

### 1.5 `vyre-runtime/src/replay.rs:190`  -  Division by zero on malicious log header
**Status:** fixed in `vyre-runtime/src/replay.rs`; existing-log headers pass through `validate_capacity(existing_cap)` before modulo.

```rust
let existing_cap = u64::from_le_bytes(cap_bytes);
...
return Ok(Self {
    ...
    next_slot: cursor % existing_cap,
});
```
**Issue:** When opening an existing replay log, `existing_cap` is read from the file header without validating `> 0`. A crafted header with `capacity = 0` causes a division-by-zero panic.  
**Fix:**
```rust
if existing_cap == 0 {
    return Err(ReplayLogError::HeaderMismatch { path: path.to_string_lossy().into_owned() });
}
```

### 1.6 `vyre-runtime/src/megakernel/protocol.rs:371`  -  `encode_control` panics on overflow
```rust
pub fn encode_control(... ) -> Vec<u8> {
    try_encode_control(...)
        .expect("megakernel control buffer length must fit usize; reduce observable_slots")
}
```
**Issue:** Public API panics when `observable_slots` is large enough to overflow `usize`.  
**Fix:** Remove the panicking wrapper or make it return `Result<Vec<u8>, ProtocolError>`.

### 1.7 `vyre-runtime/src/megakernel/protocol.rs:424`  -  `encode_empty_ring` panics on overflow
```rust
pub fn encode_empty_ring(slot_count: u32) -> Vec<u8> {
    try_encode_empty_ring(slot_count).expect("megakernel ring buffer length must fit usize")
}
```
**Issue:** Same pattern for ring buffer size.  
**Fix:** Return `Result` instead of panicking.

### 1.8 `vyre-runtime/src/megakernel/protocol.rs:445`  -  `encode_empty_debug_log` panics on overflow
```rust
pub fn encode_empty_debug_log(record_capacity: u32) -> Vec<u8> {
    try_encode_empty_debug_log(record_capacity)
        .expect("megakernel debug-log buffer length must fit usize")
}
```
**Issue:** Same pattern for debug log size.  
**Fix:** Return `Result` instead of panicking.

### 1.9 `vyre-runtime/src/megakernel/io.rs:398-401`  -  `encode_empty_io_queue` panics on overflow
```rust
pub fn encode_empty_io_queue(slot_count: u32) -> Vec<u8> {
    try_encode_empty_io_queue(slot_count)
        .expect("megakernel IO queue length must fit the compiled poll window")
}
```
**Issue:** Public API panics on overflow.  
**Fix:** Return `Result<Vec<u8>, PipelineError>`.

---

## 2. Poisoned Locks (DoS by Panic)

### 2.1 `vyre-driver/src/persistent.rs:139`  -  `RwLock::write().expect()` in `enqueue`
**Status:** fixed in `vyre-driver/src/persistent.rs`.

```rust
*self.slots.get(slot_idx as usize)
    .expect("slot index masked by ring_size-1")
    .write()
    .expect("slot rwlock poisoned") = item;
```
**Issue:** If any worker thread panics while holding a slot lock, subsequent `enqueue` calls panic instead of returning a structured error.  
**Fix:**
```rust
let guard = self.slots[slot_idx as usize].write().map_err(|e| QueueFull {
    fix: "persistent engine slot lock poisoned; restart required",
})?;
```

### 2.2 `vyre-driver/src/persistent.rs:175`  -  `RwLock::read().expect()` in `claim`
**Status:** fixed in `vyre-driver/src/persistent.rs`.

```rust
*self.slots.get(slot_idx as usize)
    .expect("slot index masked by ring_size-1")
    .read()
    .expect("slot rwlock poisoned")
```
**Issue:** Same as 2.1 on the consumer side.  
**Fix:** Use `read().map_err(...)?` and return `None` or a structured error.

### 2.3 `vyre-driver-wgpu/src/buffer/pool.rs:251`  -  `Mutex::lock().expect()` in `acquire`
**Status:** fixed in `vyre-driver-wgpu/src/buffer/pool.rs`.

```rust
let mut cache = tiering.lock().expect("tiering lock poisoned");
```
**Issue:** A poisoned tiering lock crashes the buffer pool on any allocation.  
**Fix:**
```rust
let mut cache = tiering.lock().map_err(|_| BackendError::new("buffer pool tiering lock poisoned"))?;
```

### 2.4 `vyre-driver-wgpu/src/buffer/pool.rs:360`  -  `Mutex::lock().expect()` in `release`
**Status:** fixed in `vyre-driver-wgpu/src/buffer/pool.rs`.

```rust
let mut cache = tiering.lock().expect("tiering lock poisoned");
```
**Issue:** Same as 2.3 on buffer release.  
**Fix:** Same pattern.

### 2.5 `vyre-driver-wgpu/src/buffer/handle.rs:465`  -  `Mutex::lock().expect()` in `Drop`
**Status:** fixed in `vyre-driver-wgpu/src/buffer/handle.rs`.

```rust
resident_buffers()
    .lock()
    .expect("Fix: resident buffer registry lock poisoned during removal.")
    .remove(&self.id);
```
**Issue:** Dropping a `GpuBufferInner` after a lock-poisoning panic aborts the thread, potentially during stack unwinding.  
**Fix:**
```rust
if let Ok(mut guard) = resident_buffers().lock() {
    guard.remove(&self.id);
} else {
    tracing::error!("resident buffer registry poisoned; leaking buffer id {}", self.id);
}
```

---

## 3. GPU Fallback Dishonesty & Silent Error Dropping

### 3.1 `vyre-driver-wgpu/src/async_dispatch.rs:100`  -  dropped send result
**Status:** fixed in `vyre-driver-wgpu/src/async_dispatch.rs`; worker result send failures are logged with the lost result payload instead of discarded.

```rust
let _ = job.response.send(result);
```
**Issue:** If the caller drops the `DeferredDispatch` (timeout, panic), the GPU result - success or validation failure - is silently discarded.  
**Fix:**
```rust
if let Err(e) = job.response.send(result) {
    tracing::error!("async dispatch result lost: receiver dropped: {e}");
}
```

### 3.2 `vyre-driver-wgpu/src/engine/streaming.rs:80`  -  dropped send result
**Status:** fixed in `vyre-driver-wgpu/src/engine/streaming.rs`; streaming worker result send failures are logged instead of discarded.

```rust
let _ = job.response.send(result);
```
**Issue:** Same as 3.1 for streaming chunks.  
**Fix:** Same pattern.

### 3.3 `vyre-driver-wgpu/src/buffer/handle.rs:300`  -  `map_async` error swallowed
**Status:** fixed in `vyre-driver-wgpu/src/buffer/handle.rs`; readback `map_async` send failures are logged with the lost mapping result.

```rust
slice.map_async(wgpu::MapMode::Read, move |result| {
    let _ = sender.send(result);
});
```
**Issue:** If the readback deadline fires and the receiver is dropped, `map_async` errors (device loss) vanish.  
**Fix:**
```rust
if let Err(e) = sender.send(result) {
    tracing::error!("readback map_async result lost: receiver dropped: {e:?}");
}
```

### 3.4 `vyre-driver-wgpu/src/pipeline_compound.rs:207`  -  `map_async` error swallowed
**Status:** fixed in `vyre-driver-wgpu/src/pipeline_compound.rs`; compound readback `map_async` send failures are logged with the lost mapping result.

```rust
slice.map_async(wgpu::MapMode::Read, move |res| {
    let _ = sender.send(res);
});
```
**Issue:** Same as 3.3 in compound dispatch.  
**Fix:** Same pattern.

### 3.5 `vyre-driver-wgpu/src/runtime/readback_ring.rs:142-148`  -  `map_async` failure hidden as "free"
**Status:** fixed in `vyre-driver-wgpu/src/runtime/readback_ring.rs`.

```rust
slot.buffer.slice(..byte_len).map_async(wgpu::MapMode::Read, move |result| {
    if result.is_ok() {
        state_clone.store(SLOT_READY, Ordering::Release);
    } else {
        state_clone.store(SLOT_FREE, Ordering::Release);
    }
});
```
**Issue:** On `map_async` error, the slot becomes `SLOT_FREE`. `collect_slot` returns `None`, so the caller believes the slot is merely unready rather than permanently failed.  
**Fix:** Add a `SLOT_ERROR` state and propagate it:
```rust
match result {
    Ok(()) => state_clone.store(SLOT_READY, Ordering::Release),
    Err(e) => {
        tracing::error!("readback_ring map_async failed: {e:?}");
        state_clone.store(SLOT_ERROR, Ordering::Release);
    }
}
// In collect_slot, return Err when state == SLOT_ERROR.
```

### 3.6 `vyre-driver-wgpu/src/runtime/ring.rs:47-51`  -  `map_async` error causes infinite stall
**Status:** fixed in `vyre-driver-wgpu/src/runtime/ring.rs`.

```rust
output_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
    if result.is_ok() {
        ready_clone.store(true, std::sync::atomic::Ordering::Release);
    }
});
```
**Issue:** On error, `ready` stays `false` forever. `poll_oldest` returns `None` forever, causing the caller to spin or stall.  
**Fix:** Track errors explicitly:
```rust
match result {
    Ok(()) => ready_clone.store(true, Ordering::Release),
    Err(e) => {
        tracing::error!("ring map_async failed: {e:?}");
        // store an error flag or at least log
    }
}
```

### 3.7 `vyre-driver-wgpu/src/buffer/pool.rs:261`  -  `ArrayQueue::push` overflow silently drops GPU buffer
**Status:** fixed in `vyre-driver-wgpu/src/buffer/pool.rs`.

```rust
let _ = self.inner.free[correct_class][correct_kind as usize].push(entry);
```
**Issue:** `ArrayQueue::push` returns `Err(item)` when full. The `let _ =` silently drops the `FreeEntry`, leaking the underlying `Arc<wgpu::Buffer>`.  
**Fix:**
```rust
if let Err(overflow) = self.inner.free[correct_class][correct_kind as usize].push(entry) {
    tracing::warn!("buffer pool class {correct_class} full; dropping leaked entry");
    drop(overflow);
}
```

### 3.8 `vyre-driver-wgpu/src/pipeline_disk_cache.rs:381`  -  file unlock failure ignored
**Status:** fixed in `vyre-driver-wgpu/src/pipeline_disk_cache.rs`; metadata unlock failure now logs and rejects the cache metadata read.

```rust
let res = file.read_to_string(&mut text);
let _ = file.unlock();
if res.is_err() { ... }
```
**Issue:** `file.unlock()` can fail on NFS or when the lock was never acquired. Ignoring it leaves the cache metadata file locked for other processes, causing silent hangs.  
**Fix:**
```rust
if let Err(e) = file.unlock() {
    tracing::warn!("pipeline cache metadata unlock failed: {e}");
}
```

### 3.9 `vyre-driver-wgpu/src/engine/record_and_readback.rs:67`  -  `device.poll` result ignored
**Status:** fixed in `vyre-driver-wgpu/src/engine/record_and_readback.rs`; readiness polling matches all current `MaintainResult` variants before inspecting map slots.

```rust
pub(crate) fn is_ready(&self) -> bool {
    let (device, _) = &*self.device_queue;
    let _ = device.poll(wgpu::Maintain::Poll);
    self.pending.iter().all(|(...)| map_slot_is_complete(...))
}
```
**Issue:** `wgpu::Device::poll` returns `MaintainResult` which can signal device loss. Ignoring it means `is_ready()` returns `false` instead of surfacing the fatal error.  
**Fix:**
```rust
match device.poll(wgpu::Maintain::Poll) {
    wgpu::MaintainResult::Ok | wgpu::MaintainResult::SubmissionQueueEmpty => {}
    other => tracing::error!("poll returned unexpected result: {other:?}"),
}
```

### 3.10 `vyre-driver-wgpu/src/engine/record_and_readback.rs:87`  -  `device.poll` result ignored in retry loop
**Status:** fixed in `vyre-driver-wgpu/src/engine/record_and_readback.rs`; readback now waits until map callbacks complete and returns a bounded diagnostic timeout instead of exhausting a fixed 8-iteration loop.

```rust
for _ in 0..8 {
    if all_mapped { break; }
    let _ = device.poll(wgpu::Maintain::Wait);
}
```
**Issue:** Device-lost conditions are silently swallowed. The loop exhausts its 8 iterations and returns a misleading *"GPU readback callback was not invoked"* error.  
**Fix:**
```rust
match device.poll(wgpu::Maintain::Wait) {
    wgpu::MaintainResult::Ok | wgpu::MaintainResult::SubmissionQueueEmpty => {}
    other => return Err(BackendError::new(format!("device poll failed during readback: {other:?}"))),
}
```

### 3.11 `vyre-driver-wgpu/src/runtime/device/device.rs:338`  -  `device.poll` ignored in capability probe
**Status:** fixed by API exhaustiveness in `vyre-driver-wgpu/src/runtime/device/device.rs`; the current `wgpu::MaintainResult` surface has only `Ok` and `SubmissionQueueEmpty`, both matched explicitly. Future variants fail compilation until the probe handles them.

```rust
let _ = device.poll(wgpu::Maintain::Wait);
pollster::block_on(device.pop_error_scope()).is_none()
```
**Issue:** If the device is lost during the subgroup capability probe, the poll error is discarded. The function then checks `pop_error_scope()`, which may return a stale or misleading result.  
**Fix:**
```rust
match device.poll(wgpu::Maintain::Wait) {
    wgpu::MaintainResult::Ok | wgpu::MaintainResult::SubmissionQueueEmpty => {}
    other => return false,
}
```

### 3.12 `vyre-driver-wgpu/src/engine/streaming/async_copy.rs:153`  -  thread join panic hidden
**Status:** fixed in `vyre-driver-wgpu/src/engine/streaming/async_copy.rs`; drop logs the panic payload instead of reducing it to a boolean.

```rust
InFlight::Thread(join) => {
    let _ = join.join();
}
```
**Issue:** If the async-copy host thread panicked, the panic payload is lost.  
**Fix:**
```rust
if let Err(e) = join.join() {
    tracing::error!("async copy thread panicked during drop: {e:?}");
}
```

### 3.13 Missing `wgpu::ErrorScope` around pipeline compilation
**Status:** fixed in `vyre-driver-wgpu/src/pipeline.rs` and `vyre-driver-wgpu/src/runtime/shader/compile_compute_pipeline.rs`; shader module and compute pipeline creation now run inside a validation error scope and return structured compile/GPU errors.

**Files:**
- `vyre-driver-wgpu/src/pipeline.rs:433-444`
- `vyre-driver-wgpu/src/runtime/shader/compile_compute_pipeline.rs:51-69`

**Issue:** `create_compute_pipeline` does not return `Result`; it panics or emits an async validation error. Without an error scope, a compilation failure cannot be caught and propagated as a structured `BackendError`.  
**Fix:**
```rust
device.push_error_scope(wgpu::ErrorFilter::Validation);
let module = device.create_shader_module(...);
let pipeline = device.create_compute_pipeline(...);
let _ = device.poll(wgpu::Maintain::Wait);
if let Some(err) = pollster::block_on(device.pop_error_scope()) {
    return Err(BackendError::KernelCompileFailed { message: err.to_string() });
}
```

### 3.14 Missing `wgpu::ErrorScope` around dispatch submission
**Status:** fixed in `vyre-driver-wgpu/src/engine/record_and_readback.rs`; command buffer finish and queue submission now run inside a validation error scope and return a structured `DispatchFailed` diagnostic.

**File:** `vyre-driver-wgpu/src/engine/record_and_readback.rs:358-535`

**Issue:** The entire command encoder is submitted without an error scope. If wgpu detects a validation error during submission (oversized workgroup, bad bind group), the error is async and may only surface as a generic mapping failure or hang.  
**Fix:**
```rust
device.push_error_scope(wgpu::ErrorFilter::Validation);
let mut encoder = device.create_command_encoder(...);
// ... record ...
let submission = queue.submit(std::iter::once(encoder.finish()));
let _ = device.poll(wgpu::Maintain::Wait);
if let Some(err) = pollster::block_on(device.pop_error_scope()) {
    return Err(BackendError::DispatchFailed { message: err.to_string(), code: None });
}
```

---

## 4. Secret / Log Leakage

### 4.1 `vyre-driver-wgpu/src/pipeline.rs:431`  -  WGSL source dumped to predictable world-readable path
**Status:** fixed in `vyre-driver-wgpu/src/runtime/shader.rs`; both pipeline compile paths use a content-addressed dump helper with private default directory, `create_new`, and `0600` files on Unix.

```rust
if std::env::var_os("VYRE_DUMP_WGSL").is_some() {
    let _ = std::fs::write("/tmp/wgsl_dump.wgsl", &wgsl);
}
```
**Issue:** Shader source (potentially proprietary IP) is written to `/tmp/wgsl_dump.wgsl`, a predictable path that may be world-readable on multi-user systems. Race conditions allow symlink attacks.  
**Fix:** Write to a secure temporary file:
```rust
if std::env::var_os("VYRE_DUMP_WGSL").is_some() {
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("vyre_wgsl_{}.wgsl", std::process::id()));
    let _ = std::fs::write(&tmp, &wgsl);
}
```

### 4.2 `vyre-driver-wgpu/src/runtime/shader/compile_compute_pipeline.rs:49`  -  same predictable dump
**Status:** fixed in `vyre-driver-wgpu/src/runtime/shader.rs`; the shared dump helper no longer writes to a predictable global file.

```rust
if std::env::var_os("VYRE_DUMP_WGSL").is_some() {
    let _ = std::fs::write("/tmp/wgsl_dump.wgsl", wgsl_source);
}
```
**Issue:** Identical to 4.1 in a different compilation path.  
**Fix:** Same as 4.1.

### 4.3 `vyre-driver-wgpu/src/pipeline_disk_cache.rs:300`  -  path and errno logged without sanitization
```rust
tracing::error!(path = %path.display(), errno = ?error.raw_os_error(), "{context}");
```
**Issue:** `path.display()` may contain sensitive directory structures or user PII. `raw_os_error()` leaks low-level system state.  
**Fix:** Redact or hash the path; avoid logging raw errno in production:
```rust
tracing::error!(path_hash = %hash_path(path), "pipeline cache {context} failed: {error}");
```

---

## 5. DoS by Unbounded Allocation on Malformed Input

### 5.1 `vyre-runtime/src/replay.rs:189-191`  -  untrusted capacity read from file header
```rust
let existing_cap = u64::from_le_bytes(cap_bytes);
...
capacity: existing_cap,
```
**Issue:** `existing_cap` is not capped. `replay_all` does:
```rust
let mut out = Vec::with_capacity(self.capacity as usize);
for step in 0..self.capacity { ... }
```
A crafted log file with `capacity = u64::MAX` causes an immediate OOM abort (or hours of spinning).  
**Fix:** Reject unreasonable capacities when opening an existing file:
```rust
const MAX_REPLAY_CAPACITY: u64 = 1_000_000;
if existing_cap > MAX_REPLAY_CAPACITY {
    return Err(ReplayLogError::CapacityOverflow { count: existing_cap });
}
```

### 5.2 `vyre-runtime/src/megakernel/protocol.rs:369-371`  -  `encode_control` unbounded allocation
```rust
pub fn encode_control(... ) -> Vec<u8> {
    try_encode_control(...).expect(...)
}
// inside try_encode_control:
let mut bytes = vec![0u8; total_bytes];
```
**Issue:** A caller passing `observable_slots = u32::MAX / 4` yields `total_bytes ≈ 4 GiB`. The panicking wrapper gives no chance to catch the error.  
**Fix:** Cap `total_bytes` to a reasonable max (e.g., 64 MiB) inside `try_encode_control`, and remove the panicking public wrappers.

### 5.3 `vyre-runtime/src/megakernel/protocol.rs:423-425`  -  `encode_empty_ring` unbounded allocation
**Issue:** Same pattern; `slot_count = u32::MAX / SLOT_WORDS` yields a multi-gigabyte `vec![0u8; total_bytes]`.  
**Fix:** Same as 5.2.

### 5.4 `vyre-runtime/src/megakernel/protocol.rs:443-445`  -  `encode_empty_debug_log` unbounded allocation
**Issue:** Same pattern for debug log.  
**Fix:** Same as 5.2.

---

## 6. Malformed Input Handling  -  Bounds / Integer Overflow

### 6.1 `vyre-foundation/src/vast.rs:279,287,313,329,363-364,387`  -  duplicated unchecked multiplications
```rust
let node_bytes_len = (hdr.node_count as usize) * NODE_STRIDE_U32 * 4;
let file_bytes_len = (hdr.file_count as usize) * 12;
```
**Issue:** These expressions are reached after `total_byte_len()` succeeds with `checked_mul`, but future refactoring could reorder them and expose an exploitable overflow.  
**Fix:** Reuse the already-validated `expected` value, or replace with `checked_mul`:
```rust
let node_bytes_len = (hdr.node_count as usize)
    .checked_mul(NODE_STRIDE_U32 * 4)
    .unwrap_or(usize::MAX);
```

### 6.2 `vyre-reference/src/hashmap_interp.rs:79`  -  unchecked multiplication in buffer sizing
```rust
let min_bytes = decl.count() as usize * stride;
```
**Issue:** On 32-bit targets, a malicious `Program` with `count = 0x4000_0000` and `stride = 8` wraps to `0`, causing the size check to pass even though the true size is 8 GiB. Downstream OOB checks in `oob.rs` prevent memory corruption, but the overflow is latent.  
**Fix:**
```rust
let min_bytes = (decl.count() as usize)
    .checked_mul(stride)
    .ok_or_else(|| Error::interp(format!("buffer size overflow")))?;
```

### 6.3 `vyre-driver-wgpu/src/pipeline_disk_cache.rs:163`  -  `unsafe` pipeline cache from unauthenticated disk blob
```rust
unsafe {
    device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
        label: Some("vyre persistent compiled pipeline cache"),
        data: data.as_deref(), // loaded from ~/.cache/vyre/pipeline
        fallback: true,
    })
}
```
**Issue:** The blob is only protected by CRC32. If an attacker tampers with the cache, the `unsafe` block passes attacker-controlled bytes to wgpu and the GPU driver.  
**Fix:** Authenticate the blob cryptographically (e.g., ed25519 or HMAC keyed by the pipeline artifact key) before the `unsafe` call, or skip disk caching when the signature is invalid.

---

## 7. Untrusted Ring Args  -  Assessment

The megakernel ring protocol (`vyre-runtime/src/megakernel/protocol_api.rs`, `protocol.rs`, `io.rs`) **does** validate `slot_idx` against buffer capacity and `args.len()` against `ARGS_PER_SLOT` before writing. Public entry points (`publish_slot`, `publish_packed_slot`, `publish_into`) return `Result<..., PipelineError>` on out-of-bounds inputs rather than panicking.  

**One brittle internal helper:** `vyre-runtime/src/megakernel/protocol.rs:669-672`
```rust
fn write_word(bytes: &mut [u8], word_idx: usize, value: u32) {
    let off = word_idx * 4;
    bytes[off..off + 4].copy_from_slice(&value.to_le_bytes());
}
```
This function has no bounds check, but all callers currently validate indices first. **Defense-in-depth fix:**
```rust
fn write_word(bytes: &mut [u8], word_idx: usize, value: u32) {
    let off = word_idx * 4;
    if off + 4 > bytes.len() {
        panic!("write_word out of bounds: word_idx={word_idx} buf_len={}", bytes.len());
    }
    bytes[off..off + 4].copy_from_slice(&value.to_le_bytes());
}
```
(or return `Result` and propagate it to callers).

Similarly, `vyre-runtime/src/megakernel/io.rs:248-259` (`read_word` / `write_word`) index into `self.words[idx]` without bounds checking, but the public methods (`publish_slot`, `completion`, `is_recycled`) validate `slot_idx` first.

---

## Summary Table

| Category | Count | Most Critical Files |
|----------|-------|---------------------|
| Panics in public APIs | 9 | `vyre-foundation/src/validate/validate.rs`, `vyre-runtime/src/replay.rs`, `vyre-runtime/src/megakernel/protocol.rs` |
| Poisoned locks | 5 | `vyre-driver/src/persistent.rs`, `vyre-driver-wgpu/src/buffer/pool.rs`, `vyre-driver-wgpu/src/buffer/handle.rs` |
| GPU fallback / silent errors | 14 | `vyre-driver-wgpu/src/engine/record_and_readback.rs`, `vyre-driver-wgpu/src/runtime/readback_ring.rs`, `vyre-driver-wgpu/src/runtime/ring.rs`, `vyre-driver-wgpu/src/pipeline.rs` |
| Secret / log leakage | 3 | `vyre-driver-wgpu/src/pipeline.rs`, `vyre-driver-wgpu/src/runtime/shader/compile_compute_pipeline.rs`, `vyre-driver-wgpu/src/pipeline_disk_cache.rs` |
| DoS by unbounded allocation | 4 | `vyre-runtime/src/replay.rs`, `vyre-runtime/src/megakernel/protocol.rs` |
| Malformed input / bounds | 3 | `vyre-foundation/src/vast.rs`, `vyre-reference/src/hashmap_interp.rs`, `vyre-driver-wgpu/src/pipeline_disk_cache.rs` |
| Untrusted ring args | 0 exploitable | Protocol validation is correct; internal helpers are brittle but guarded. |
