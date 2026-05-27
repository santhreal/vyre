# Safety Audit — vyre-wgpu

**Date:** 2026-04-18  
**Scope:** `vyre-wgpu/src/**/*.rs` (production code only; `#[cfg(test)]` modules excluded)  
**Auditor:** Kimi Code CLI  
**Methodology:** Manual line-by-line review of all `.rs` files under `vyre-wgpu/src/`, cross-referenced with `grep` for `unsafe`, `.unwrap()`, `.expect(`, `panic!`, `todo!`, `unimplemented!`, `transmute`, `Mutex::lock().unwrap()`, array indexing, integer casts, and division.  

**Summary:** 20 findings. 0 `unsafe` blocks with missing or invalid SAFETY comments (the single `unsafe` block is well-documented and sound). The dominant risk class is **unprovable panic vectors** via `.expect()`, direct `panic!()`, and unguarded slice indexing on GPU readback paths. At internet scale any of these can turn a malformed shader response or an unusual OS condition into a host-wide crash.

---

### SAFE-01 — WgpuBackend::default() panics without GPU
**File:** `vyre-wgpu/src/lib.rs:77`  
**Severity:** HIGH

```rust
impl Default for WgpuBackend {
    fn default() -> Self {
        Self::acquire().expect(
            "WgpuBackend::default() requires a GPU adapter. \
             Fix: run on a host with a compatible GPU or use WgpuBackend::acquire() to handle the error.",
        )
    }
}
```

**Risk:** `Default::default()` is a common trait call used by frameworks, deserializers, and generic glue. If the host has no compatible GPU adapter, this panics instead of returning a `Result` or `Option`. A deployment running on a VM without GPU passthrough crashes at startup.

**Fix:** Remove the `Default` impl or make it return a sentinel/backend that degrades gracefully. If `Default` must exist, change `WgpuBackend` to an `Option<WgpuBackend>` wrapper, or remove the impl and force callers to use `WgpuBackend::acquire()`.

---

### SAFE-02 — Streaming worker thread spawn panics on resource exhaustion
**File:** `vyre-wgpu/src/engine/streaming.rs:65`  
**Severity:** HIGH

```rust
std::thread::Builder::new()
    .name(format!("vyre-wgpu-streaming-{index}"))
    .spawn(move || { ... })
    .expect("Fix: failed to spawn vyre-wgpu streaming worker thread");
```

**Risk:** `std::thread::spawn` can fail when the process hits `nproc` / `RLIMIT_NPROC` or the kernel runs out of PID space. On a heavily loaded host this panic kills the entire process.

**Fix:** Return `Result` from `StreamingPool::new` (or `StreamingPool::global`) and propagate the `std::io::Error` as a `BackendError`.

```rust
let handle = std::thread::Builder::new()
    .name(...)
    .spawn(move || { ... })
    .map_err(|e| BackendError::new(format!("thread spawn failed: {e}")))?;
```

---

### SAFE-03 — Partitioner expect on non-empty device list
**File:** `vyre-wgpu/src/engine/multi_gpu.rs:70`  
**Severity:** MEDIUM

```rust
let target = partitions
    .iter_mut()
    .min_by_key(|partition| (partition.total_cost, partition.device_index))
    .expect("Fix: validated non-empty device list before partitioning.");
```

**Risk:** Although `validate_inputs` is called immediately before, the invariant is not compiler-enforced. A future refactor could inline or reorder the code, bypassing validation and causing a panic at runtime.

**Fix:** Use `if let Some(target) = ...` or `ok_or_else` to return the existing `Result<Vec<Partition>, String>` instead of unwrapping.

---

### SAFE-04 — Encoder presence expect in graph dispatch
**File:** `vyre-wgpu/src/engine/dataflow.rs:352`  
**Severity:** MEDIUM

```rust
let encoder = if let Some(encoder) = encoder {
    encoder
} else {
    owned_encoder
        .as_mut()
        .expect("Fix: owned encoder must be present when encoder is omitted")
};
```

**Risk:** The `expect` is logically unreachable today because `owned_encoder` is populated with `encoder.is_none().then(...)`, but this is a maintenance hazard. If the `is_none()` logic drifts, the panic fires.

**Fix:** Replace the `expect` with an `else` block that returns `Err(Error::Dataflow { ... })` or use `let else`:

```rust
let Some(encoder) = owned_encoder.as_mut() else {
    return Err(Error::Dataflow { message: "...".into() });
};
```

---

### SAFE-05 — PooledBuffer Deref panics on released buffer
**File:** `vyre-wgpu/src/runtime/cache/buffer_pool.rs:181`  
**Severity:** HIGH

```rust
impl Deref for PooledBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().unwrap_or_else(|| {
            panic!(
                "pooled buffer `{}` no longer owns its inner wgpu::Buffer. Fix: do not dereference a PooledBuffer after release.",
                self.label
            )
        })
    }
}
```

**Risk:** `Deref` is invoked implicitly by the compiler on `.`, `&`, and method calls. A double-use or accidental move of a `PooledBuffer` after `Drop` causes a panic in seemingly innocent code.

**Fix:** Remove the `Deref` impl. Force callers to use `buffer()` which already returns `Result<&wgpu::Buffer, BufferPoolError>`. Panicking inside `Deref` violates Rust idioms and makes debugging harder.

---

### SAFE-06 — Unvalidated readback slice in record_and_readback
**File:** `vyre-wgpu/src/engine/record_and_readback.rs:268`  
**Severity:** HIGH

```rust
let result = mapped
    [request.output.trim_start..request.output.trim_start + request.output.read_size]
    .to_vec();
```

**Risk:** `OutputLayout` fields are derived from program metadata. A buggy shader that writes out-of-bounds metadata, a corrupted `Program`, or a future layout-calculation bug can make `trim_start + read_size` exceed `mapped.len()`, causing a panic.

**Fix:** Bound-check before slicing:

```rust
let end = request.output.trim_start.saturating_add(request.output.read_size);
if end > mapped.len() {
    return Err(BackendError::new("readback slice out of bounds".into()));
}
let result = mapped[request.output.trim_start..end].to_vec();
```

---

### SAFE-07 — Unvalidated readback slice in compound dispatch
**File:** `vyre-wgpu/src/pipeline_compound.rs:286`  
**Severity:** HIGH

```rust
let result =
    mapped[self.output.trim_start..self.output.trim_start + self.output.read_size].to_vec();
```

**Risk:** Same as SAFE-06: the compound dispatch path trusts `OutputLayout` without verifying the slice fits the mapped buffer. A malformed program or GPU-side corruption triggers a panic.

**Fix:** Same bound-check pattern as SAFE-06, returning `BackendError` on mismatch.

---

### SAFE-08 — DFA scan assumes readback buffer is at least 4 bytes
**File:** `vyre-wgpu/src/engine/dfa.rs:298`  
**Severity:** HIGH

```rust
let bytes = read_buffer(
    &self.device,
    &resources.match_readback,
    readback_len,
    "matches",
    submission,
)?;
let reported = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
```

**Risk:** `read_buffer` maps `0..readback_len` and returns `mapped.to_vec()`. In theory the length should match, but driver bugs, device loss, or wgpu internal truncation could yield a shorter `Vec`. Direct indexing `bytes[0..3]` then panics.

**Fix:** Guard the length:

```rust
if bytes.len() < 4 {
    return Err(Error::Dfa { message: "match count readback too short".into() });
}
let reported = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
```

---

### SAFE-09 — DFA match slice trusts shader-reported count without length check
**File:** `vyre-wgpu/src/engine/dfa.rs:309`  
**Severity:** CRITICAL

```rust
for fields in bytes[4..4 + captured_usize * 12].chunks_exact(12) {
    matches.push(vyre::Match::new(
        u32::from_le_bytes([fields[0], ...]),
        ...
    ));
}
```

**Risk:** `captured` is `reported.min(self.max_matches)`. A malicious or buggy shader can report `max_matches` even though the actual buffer contains fewer bytes. The slice `bytes[4..4 + captured_usize * 12]` will panic when `bytes.len()` is smaller than the end index. At internet scale a single adversarial input crashes the service.

**Fix:** Validate the slice bounds before iterating:

```rust
let needed = 4usize.saturating_add(captured_usize.saturating_mul(12));
if bytes.len() < needed {
    return Err(Error::Dfa { message: "match readback shorter than shader-reported count".into() });
}
```

---

### SAFE-10 — Finding count readback assumes 4-byte mapped range
**File:** `vyre-wgpu/src/engine/dataflow.rs:125`  
**Severity:** HIGH

```rust
let mapped = slice.get_mapped_range();
let count = u32::from_ne_bytes(mapped[..4].try_into().map_err(|source| { ... })?)
    .min(max_findings);
```

**Risk:** The expression `mapped[..4]` is evaluated *before* `try_into()`. If the mapped range is shorter than 4 bytes (driver anomaly), the slice indexing panics immediately.

**Fix:** Check length first:

```rust
if mapped.len() < 4 {
    return Err(Error::Dataflow { message: "finding count readback too short".into() });
}
let count = u32::from_ne_bytes(mapped[..4].try_into().unwrap());
```

---

### SAFE-11 — dispatch_wgsl output_word_count div_ceil overflow
**File:** `vyre-wgpu/src/ext.rs:49`  
**Severity:** HIGH

```rust
let output_word_count = output_size.div_ceil(4).max(1);
```

**Risk:** `usize::div_ceil(4)` computes `(self + 3) / 4`. When `output_size` is `usize::MAX`, `self + 3` overflows. In debug mode this panics; in release mode it wraps, producing a tiny incorrect word count that later causes buffer under-allocation and silent corruption.

**Fix:** Use `checked_div_ceil` (Rust 1.84+) or `checked_add`:

```rust
let output_word_count = output_size
    .checked_add(3)
    .and_then(|n| n.checked_div(4))
    .max(Some(1))
    .ok_or_else(|| "output_size overflow".to_string())?;
```

---

### SAFE-12 — Pipeline compile output_word_count div_ceil overflow
**File:** `vyre-wgpu/src/pipeline.rs:173`  
**Severity:** HIGH

```rust
let output_word_count = output.full_size.div_ceil(4).max(1);
```

**Risk:** Identical to SAFE-11. `output.full_size` comes from `count.checked_mul(element_size)`, but the preceding check only catches `usize` overflow of the product, not the subsequent `div_ceil(4)`. A pathological program with `full_size = usize::MAX` overflows here.

**Fix:** Same `checked_add`/`checked_div` guard as SAFE-11.

---

### SAFE-13 — Compound dispatch output_bytes multiplication overflow
**File:** `vyre-wgpu/src/pipeline_compound.rs:65`  
**Severity:** HIGH

```rust
let output_bytes = self.output_word_count * 4;
```

**Risk:** `output_word_count` is a `usize`. If it is close to `usize::MAX / 4`, this multiplication overflows. In debug mode it panics; in release it wraps, leading to a tiny buffer allocation and GPU out-of-bounds writes.

**Fix:** Use `checked_mul`:

```rust
let output_bytes = self
    .output_word_count
    .checked_mul(4)
    .ok_or_else(|| BackendError::new("output_bytes overflow".into()))?;
```

---

### SAFE-14 — Record-and-readback output_bytes multiplication overflow
**File:** `vyre-wgpu/src/engine/record_and_readback.rs:80`  
**Severity:** HIGH

```rust
let output_bytes = request.output_word_count * 4;
```

**Risk:** Same multiplication-overflow pattern as SAFE-13. `output_word_count` is derived from program metadata and can be attacker-controlled.

**Fix:** Same `checked_mul` guard, returning `BackendError` on overflow.

---

### SAFE-15 — words_for_output_bytes div_ceil overflow
**File:** `vyre-wgpu/src/engine/decompress.rs:172`  
**Severity:** MEDIUM

```rust
pub(crate) fn words_for_output_bytes(format: &str, bytes: usize) -> Result<usize> {
    ...
    Ok(bytes.div_ceil(4).max(1))
}
```

**Risk:** `bytes.div_ceil(4)` overflows when `bytes` is `usize::MAX`. Although callers validate against `MAX_DECOMPRESS_OUTPUT_BYTES`, the helper is `pub(crate)` and could be reused by future code that skips the caller-side check.

**Fix:** Defensive arithmetic:

```rust
let words = bytes
    .checked_add(3)
    .and_then(|n| n.checked_div(4))
    .unwrap_or(bytes) // fallback, though should error
    .max(1);
```

---

### SAFE-16 — IntrusiveLru with_capacity panics on zero
**File:** `vyre-wgpu/src/runtime/cache/lru.rs:44`  
**Severity:** MEDIUM

```rust
pub fn with_capacity(capacity: usize) -> Self {
    assert!(
        capacity > 0,
        "IntrusiveLru capacity must be non-zero. Fix: configure at least one cache slot."
    );
    ...
}
```

**Risk:** A caller that computes capacity from external configuration (e.g., TOML) could pass `0`, causing a panic. The function already returns `Self`, so returning a `Result` is a compatible API change.

**Fix:** Replace `assert!` with an early `Err` or return `Option` / `Result`:

```rust
if capacity == 0 {
    return Err(CacheError::InvalidCapacity);
}
```

---

### SAFE-17 — Literal-set automaton build unchecked node indexing
**File:** `vyre-wgpu/src/engine/string_matching/lexer.rs:64`  
**Severity:** MEDIUM

```rust
let slot = usize::from(byte);
let next = nodes[state].next[slot];
```

**Risk:** `state` is a `usize` that is only mutated inside this function, so the invariant is maintained locally. However, there is no explicit bounds check before indexing `nodes[state]`. If `state` is ever corrupted by an arithmetic bug, this panics.

**Fix:** Defensive bound check or use `nodes.get(state).ok_or(...)?`.

---

### SAFE-18 — Literal-set BFS loop unchecked node indexing
**File:** `vyre-wgpu/src/engine/string_matching/lexer.rs:113`  
**Severity:** MEDIUM

```rust
while let Some(state) = queue.pop_front() {
    ...
    for byte in 0..BYTE_CLASSES {
        let next = nodes[state].next[byte];
        ...
    }
}
```

**Risk:** Same as SAFE-17: `state` comes from the BFS queue. The queue only receives indices that were previously pushed, but there is no runtime guard.

**Fix:** `nodes.get(state).ok_or(...)?` inside the loop.

---

### SAFE-19 — Decompress readback assumes mapped range length matches size
**File:** `vyre-wgpu/src/engine/decompress/dispatch_kernel/readback.rs:40`  
**Severity:** MEDIUM

```rust
let mut bytes = vec![0_u8; output_len];
bytes.copy_from_slice(&mapped[..output_len]);
```

**Risk:** `mapped[..output_len]` panics if `mapped.len() < output_len`. While wgpu guarantees the mapped range matches the slice size, driver bugs or future wgpu changes could violate this. The code relies on an external invariant without a local guard.

**Fix:**

```rust
if mapped.len() < output_len {
    return Err(Error::Gpu { message: "decompress readback truncated".into() });
}
bytes.copy_from_slice(&mapped[..output_len]);
```

---

### SAFE-20 — Output clear buffer size multiplication overflow
**File:** `vyre-wgpu/src/engine/record_and_readback.rs:184`  
**Severity:** MEDIUM

```rust
encoder.clear_buffer(buf, 0, Some((request.output_word_count * 4) as u64));
```

**Risk:** `request.output_word_count * 4` is a `usize` multiplication that can overflow before the `as u64` cast. On 64-bit systems `usize` is `u64`, so overflow is possible with maliciously large `output_word_count`.

**Fix:** Compute in `u64` space with `checked_mul`:

```rust
let clear_size = (request.output_word_count as u64)
    .checked_mul(4)
    .ok_or_else(|| BackendError::new("clear_buffer size overflow".into()))?;
encoder.clear_buffer(buf, 0, Some(clear_size));
```

---

## Notable Non-Findings

- **`runtime/shader/compile_compute_pipeline.rs:129`** — The single `unsafe` block in the crate calls `device.create_pipeline_cache` with `data: None` and `fallback: true`. The `// SAFETY` comment is present and the justification is sound (no untrusted bytes supplied, driver falls back to empty cache). **No action required.**
- **Mutex poisoning** — Every `Mutex::lock()` in production code is matched with `map_err` and converted to a structured `Error` or `BackendError` rather than unwrapped. This is exemplary.
- **No `todo!()` or `unimplemented!()`** — Grep confirmed zero occurrences in production code.
- **No `transmute`** — Grep confirmed zero occurrences in production code.

---

*End of audit.*
