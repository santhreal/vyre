# DETERMINISM Audit — 2026-04-18

**Scope:** Full workspace (`vyre-core`, `vyre-wgpu`, `vyre-reference`, `vyre-conform`, `vyre-conform-spec`, `vyre-build-scan`, `vyre-std`, `vyre-primitives`, `vyre-macros`, `vyre-sigstore`, `vyre-spec`, benches, tests, examples).  
**Goal:** Identify every source of non-determinism that can leak into program output, persisted state, or ground-truth references.  
**Method:** `ripgrep` + manual code review of all `HashMap`/`HashSet` iteration, `rand::*`, `SystemTime`/`Instant`, `thread::spawn`/`rayon`, `f32`/`f64` accumulation, `env::var`, global mutable state, and host `libm` intrinsics.

---

## Executive Summary

| Severity | Count |
|----------|-------|
| HIGH     | 6     |
| MEDIUM   | 17    |
| LOW      | 4     |
| **Total**| **27**|

Every HIGH finding either changes serialized bytes on every run (certificates, AOT metadata, fingerprints) or uses a platform-dependent host `libm` as the ground-truth oracle for GPU parity tests.

---

## Findings

### 1. `SystemTime::now` / `Instant::now` in serialized or persisted output

**DET-01** — `vyre-conform/src/runner/certify/implementation.rs:506` — **HIGH**  
`certificate_timestamp()` calls `SystemTime::now().duration_since(UNIX_EPOCH)` and embeds the resulting wall-clock seconds into every `Certificate.timestamp`. The `Certificate` struct is serialized to JSON and hashed with blake3. Re-running an identical certification at a different second produces a different JSON blob and hash.  
**Fix:** Accept an optional `timestamp: u64` parameter from the caller (or use a deterministic counter + build-time seed). Do not read the system clock.

**DET-02** — `vyre-conform/src/runner/certify/implementation.rs:515` — **HIGH**  
`next_certificate_sequence()` calls `CERTIFICATE_SEQUENCE.fetch_add(1, Ordering::SeqCst)`. The monotonic counter is process-local and resets to 0 on every process start. Identical certification runs produce different `monotonic_sequence` values, changing the serialized `Certificate` JSON.  
**Fix:** Derive the sequence from a content-addressable hash of the certification inputs (backend id + spec set + witness count), or accept it as an explicit caller parameter.

**DET-03** — `vyre-wgpu/src/runtime/aot.rs:108,173` — **MEDIUM**  
`AotMetadata.created_unix_ms` is populated by `now_unix_ms()`, which calls `SystemTime::now()`. The timestamp is written into the on-disk `{key}.toml` metadata on every cache miss. `metadata_matches` ignores the field, but the file bytes differ across identical compilations, breaking bit-exact reproducibility of the cache directory.  
**Fix:** Remove `created_unix_ms` from `AotMetadata`, or set it to `0` when reproducibility is required.

**DET-04** — `vyre-core/src/routing/pgo.rs:173` — **MEDIUM**  
`measure_backend()` pushes `start.elapsed()` (from `Instant::now()`) into a `samples` vector, then stores the median as `latency_ns` inside `BackendLatency` observations. `PgoTable::save()` writes the observations to `~/.config/vyre/pgo.toml`. System noise means the raw nanosecond timings vary from run to run, so the serialized TOML file is non-deterministic even when the selected backend is stable.  
**Fix:** Round latencies to the nearest millisecond or percentile bucket before persistence, or use a deterministic synthetic benchmark instead of wall-clock time.

**DET-05** — `vyre-core/tests/new_op_generator.rs:101` — **LOW**  
`unique_suffix()` reads `SystemTime::now()` to generate a nanosecond suffix for test operation IDs and filesystem directories. Tests create directories with non-deterministic names, leaving non-reproducible filesystem side effects.  
**Fix:** Use a fixed test seed (e.g. `42`) or a content hash of the test inputs.

---

### 2. `HashMap` / `HashSet` iteration order in output-producing code

**DET-06** — `vyre-core/src/ops/registry/registry.rs:47,84` — **MEDIUM**  
`RUNTIME_REGISTRY` is a `OnceLock<RwLock<Vec<&'static OpSpec>>>`. External crates register ops at startup via `register_op_spec()`. The append order depends on `ctor`/`inventory` initialization timing, which is linker-dependent and non-deterministic across builds. `registry()` chains this runtime snapshot after the static registry. Any downstream code that iterates the full registry (serialization, dump tools, hash computations) could emit ops in a different order per run.  
**Fix:** Sort the runtime snapshot by a stable key (`op.id()`) before chaining it into the registry iterator, or use a `BTreeMap` keyed by `op.id()`.

**DET-07** — `vyre-wgpu/src/runtime/cache/tiered_cache.rs:107` — **LOW**  
`eviction_candidate()` falls back to `entries.keys().next().copied()` when the access tracker yields no cold entries. `entries` is an `FxHashMap<u64, CacheEntry>`. `keys().next()` returns an arbitrary key dependent on the hasher’s random state and insertion history, making eviction order non-deterministic. While this only affects performance (which cache entry is evicted), it can alter downstream timing-dependent behavior in tight loops.  
**Fix:** Fall back to the smallest key (or LRU counter) instead of `HashMap` iteration order.

---

### 3. `std::env::var` reads that affect output, caches, or paths

**DET-08** — `vyre-wgpu/src/runtime/aot.rs:40-42,147,150` — **MEDIUM**  
`backend_fingerprint()` reads `VYRE_BACKEND_FINGERPRINT_OVERRIDE`, `WGPU_BACKEND`, and `VYRE_WGPU_ADAPTER`. `cache_dir()` reads `VYRE_AOT_CACHE_DIR` and `HOME`. These variables directly alter the cache key, the WGSL specialization path, and the on-disk cache location. A CI runner with a different `HOME` or `VYRE_AOT_CACHE_DIR` will produce cache misses and write different files.  
**Fix:** Document all environment variables as "build-host local only"; for reproducible builds, require an explicit `--cache-dir` CLI argument and reject environment overrides when a `VYRE_REPRODUCIBLE=1` flag is set.

**DET-09** — `vyre-core/src/routing/pgo.rs:150-153` — **MEDIUM**  
`default_pgo_path()` reads `XDG_CONFIG_HOME` and falls back to `HOME/.config`. Different values on different machines change where the PGO routing table is read/written, causing divergent backend routing decisions across hosts.  
**Fix:** Default to a path relative to the project root or to a caller-provided `DispatchConfig` field; only use XDG/HOME when explicitly opted in.

**DET-10** — `vyre-wgpu/tests/pipeline_cache_disk_persistence.rs:9` / `vyre-wgpu/tests/aot_specialization_cache_hits.rs:10,33` / `vyre-wgpu/benches/cold_start_with_cache.rs:9` — **MEDIUM**  
Tests and benches call `std::env::set_var("VYRE_AOT_CACHE_DIR", &dir)` to override the cache directory. In concurrent test runners (`cargo test --jobs N`), multiple tests mutate the same global process environment, causing race conditions and non-deterministic cache collisions.  
**Fix:** Pass the cache directory through an explicit parameter or a thread-local override instead of mutating the global process environment.

---

### 4. Global mutable state read during output generation

**DET-11** — `vyre-conform/src/runner/certify/implementation.rs:23,515` — **HIGH**  
`static CERTIFICATE_SEQUENCE: AtomicU64 = AtomicU64::new(0);` is incremented on every certificate issuance. The value is read during `Certificate` construction and serialized to JSON. Two identical certification runs in the same process (or separate processes, because the counter resets) will produce different sequence numbers.  
**Fix:** Replace the atomic counter with a deterministic content hash or an explicit caller-supplied sequence.

**DET-12** — `vyre-conform/src/runner/streaming/regression_sinking.rs:64,87` — **LOW**  
`static REGRESSION_COUNTER: AtomicU64 = AtomicU64::new(0);` is used to name regression files (`stream-{:016x}.bin`). File names depend on dispatch interleaving and process lifetime.  
**Fix:** Name files from a content hash of the failing input (e.g. `blake3(input).to_hex()[:16]`), or use an explicit deterministic counter.

---

### 5. Floating-point intrinsics that vary by CPU microarchitecture / libm

**DET-13** — `vyre-reference/src/typed_ops/float_ops.rs:34-36` — **HIGH**  
The reference interpreter implements `UnOp::Sqrt`, `UnOp::Sin`, and `UnOp::Cos` by delegating to the host platform’s `libm` via `f32::sqrt()`, `f32::sin()`, and `f32::cos()`. This interpreter is the ground-truth oracle for GPU parity tests. Results differ across operating systems (glibc, musl, macOS libm, Windows ucrt) and across CPU microarchitectures when the compiler selects different SIMD polynomial approximations (e.g. AVX-512 vs SSE2).  
**Fix:** Link the reference interpreter against a correctly-rounded, cross-platform math library (e.g. the `core_math` crate already used in `vyre-conform/src/enforce/enforcers/float_semantics/transcendentals.rs`) and use that for all reference computation.

**DET-14** — `vyre-core/tests/adversarial/float/sin.rs:19` (and ~25 other lines) — **HIGH**  
`exact_unary_test!` compares GPU output **bit-for-bit** (`got.to_bits() == exp.to_bits()`) against the host `f32::sin` result. Because `sin` implementations differ across platforms by a few ULPs, these tests spuriously fail on some hosts.  
**Fix:** Replace `exact_unary_test!` with `approx_unary_test!` (or a tighter ULP tolerance) for all transcendental ops, and use a portable correctly-rounded reference (e.g. `core_math::cr_sinf`) as the expected value.

**DET-15** — `vyre-core/tests/adversarial/float/cos.rs:19` (and ~25 other lines) — **HIGH**  
Identical to DET-14: `exact_unary_test!` uses host `f32::cos` as the bit-exact expected value.  
**Fix:** Same as DET-14.

**DET-16** — `vyre-core/tests/adversarial/float/sqrt.rs:22` (and ~15 other lines) — **MEDIUM**  
`exact_unary_test!` uses host `f32::sqrt` as the bit-exact expected value. `sqrt` is generally more consistent than transcendentals, but edge cases (subnormals, very large/small values) can still differ by 1 ULP across `libm` implementations.  
**Fix:** Use a deterministic correctly-rounded `sqrt` reference (e.g. `core_math::cr_sqrtf`) or validate via tolerance for edge-case inputs.

**DET-17** — `vyre-core/tests/lower/wgsl.rs:386` / `vyre-wgpu/src/lowering/legacy_wgsl.rs:125` — **MEDIUM**  
Taylor-series validation for WGSL `sin` uses `x.sin()` (host `libm`) as the "exact" reference to measure ULP error. If the host `libm` changes (e.g. OS upgrade, different CI runner), the measured ULP error of the WGSL approximation changes, which could cause the `max_ulp <= 2.0` assertion to fail on some platforms.  
**Fix:** Use a correctly-rounded portable reference (`core_math::cr_sinf`) as the exact baseline for ULP error measurement.

**DET-18** — `vyre-conform/src/enforce/enforcers/float_semantics/div_sqrt.rs:202` — **MEDIUM**  
`reference_bytes()` uses `black_box(x.sqrt())` as the canonical CPU reference for GPU parity enforcement. If the host and GPU have different `sqrt` rounding for edge cases, the enforcement finding may be platform-dependent.  
**Fix:** Use the same correctly-rounded portable reference (`core_math::cr_sqrtf`) that the transcendentals enforcer already uses.

---

### 6. `f32` / `f64` arithmetic order sensitive to compiler reordering

**DET-19** — `vyre-core/src/ops/security_detection/detector_support/entropy.rs:17-24` — **MEDIUM**  
```rust
counts
    .iter()
    .filter(|&&count| count != 0)
    .map(|&count| { let p = count as f32 / len; -p * p.log2() })
    .sum()
```
Sequential `f32` sum. LLVM may auto-vectorize and change addition order, causing tiny cross-platform / cross-build differences in the final entropy value.  
**Fix:** Use a compensated summation algorithm (Kahan) or reduce with an explicitly sequential loop and `#[repr(align)]` hints to discourage auto-vectorization, or round the final result to a tolerance that masks the reordering noise.

**DET-20** — `vyre-core/src/ops/stats/sliding_entropy.rs:28-36` — **MEDIUM**  
Same pattern as DET-19 but with `f64`:  
```rust
counts.iter().filter(|&&count| count != 0).map(|&count| { let p = f64::from(count) / len; -p * p.log2() }).sum::<f64>() as f32
```
`f64` accumulation order is not guaranteed stable across compiler auto-vectorization.  
**Fix:** Same as DET-19.

**DET-21** — `vyre-core/src/ops/stats/std_dev.rs:14-26` — **MEDIUM**  
Two-phase `f64` sum: first `mean`, then `squared` deviations. Vectorization of either sum can shift the final `f32` bit pattern.  
**Fix:** Use a single-pass Welford algorithm (inherently sequential) or compensated summation.

**DET-22** — `vyre-core/src/ops/stats/variance.rs:14-26` — **MEDIUM**  
Identical two-phase `f64` sum pattern as DET-21, without the final `sqrt`.  
**Fix:** Same as DET-21.

**DET-23** — `vyre-core/src/ops/stats/chi_square.rs:24-30` — **MEDIUM**  
```rust
counts.iter().map(|&count| { let delta = f64::from(count) - expected; delta * delta / expected }).sum::<f64>() as f32
```
`f64` sum could be reordered by the compiler.  
**Fix:** Use compensated summation or an explicit sequential reduction loop.

**DET-24** — `vyre-wgpu/src/engine/decode.rs:49-66` — **MEDIUM**  
Loop-based `f64` entropy accumulation:  
```rust
let mut entropy = 0.0_f64;
for &count in &counts { if count == 0 { continue; } let p = count as f64 / total; entropy -= p * p.log2(); }
```
Same vectorization/reordering concern as the iterator sums above.  
**Fix:** Same as DET-19.

---

### 7. `Hash` trait implementations using `Debug` string stability

**DET-25** — `vyre-core/src/ir/model/expr/expr_kinds.rs:195,201,220,231,238` — **MEDIUM**  
Multiple `hash_expr` implementations call `hasher.write(format!("{:?}", self.op).as_bytes())` for `BinOp`, `UnOp`, `DataType`, `AtomicOp`, and `MemoryOrdering`. While `Debug` output is stable for simple enums today, any future variant that adds a field with non-deterministic `Debug` (e.g. a `HashMap`, a raw `fn` pointer, or an internal address) will silently break IR hash stability.  
**Fix:** Replace all `format!("{:?}", x).as_bytes()` with `x.hash(hasher)` (or a stable integer discriminant), since all these types already implement `Hash`.

**DET-26** — `vyre-core/src/ir/model/node_kind.rs:380` — **MEDIUM**  
The `Barrier` node kind hashes `MemoryOrdering` via `hasher.write(format!("{:?}", self.ordering).as_bytes())`, the same fragile pattern as DET-25.  
**Fix:** Same as DET-25.

**DET-27** — `vyre-conform/src/spec/published.rs:72-86` / `vyre-conform-spec/src/spec/published.rs:72-86` — **HIGH**  
`fingerprint_spec()` hashes `spec.category`, `spec.comparator`, `spec.convention`, and `spec.oracle_override` by calling `format!("{:?}", ...)` and feeding the string into SHA-256. `Category::Intrinsic` holds a `BackendAvailabilityPredicate` (a `fn` pointer wrapper). Today it has a custom `Debug` impl that prints `"BackendAvailabilityPredicate(..)"`, so the fingerprint is deterministic. If that custom `Debug` is ever removed, the raw function pointer address would leak into the SHA-256 digest, making fingerprints non-deterministic across compilations.  
**Fix:** Hash `Category` fields manually (e.g. hash the `hardware` string and a fixed `"C"` tag) instead of relying on `Debug`. Remove the `format!("{:?}", ...)` pattern from the fingerprint pipeline entirely.

---

### 8. `sort_unstable` on collections where output order could diverge

**DET-28** — `vyre-conform/src/enforce/enforcers/divergence.rs:131` — **LOW**  
`findings.sort_unstable_by_key(divergence_key);` sorts divergence findings by `(String, u8)`. While each spec-scenario pair currently yields at most one finding, if duplicates ever arise, their stringified output order would vary across runs. The sorted list is immediately converted to strings and returned.  
**Fix:** Use `sort_by_key` (stable) instead of `sort_unstable_by_key` for any collection that is converted to human-readable or serialized output.

---

## Cross-Reference Matrix

| Category | Findings |
|----------|----------|
| `HashMap` / `HashSet` iteration order | DET-06, DET-07 |
| `rand::*` without explicit seed | *(none found — all RNG usage is seeded)* |
| `SystemTime::now` / `Instant::now` in hash or persisted | DET-01, DET-02, DET-03, DET-04, DET-05, DET-11, DET-12 |
| `std::thread::spawn` / `rayon::par_iter` order-sensitive accumulation | *(none found — all `par_iter` usages are `map→collect` on indexed iterators)* |
| `f32` / `f64` arithmetic order sensitive to reordering | DET-19, DET-20, DET-21, DET-22, DET-23, DET-24 |
| `iter::collect::<HashSet<_>>()` serialized or compared | *(none found — serialized structs use `BTreeMap` correctly)* |
| Environment variable reads affecting output | DET-08, DET-09, DET-10 |
| `std::env::var("HOME")` etc. | DET-08, DET-09 |
| Global mutable state read during output | DET-11, DET-12 |
| Floating-point intrinsics varying by CPU | DET-13, DET-14, DET-15, DET-16, DET-17, DET-18 |
| `Hash` impls relying on `Debug` | DET-25, DET-26, DET-27 |
| `sort_unstable` on output-producing collections | DET-28 |

---

## Remediation Priority

1. **Immediate (HIGH):** DET-01, DET-02, DET-13, DET-14, DET-15 — Certificates and reference interpreter ground-truth are the most visible sources of non-determinism.
2. **This Sprint (MEDIUM):** DET-03, DET-04, DET-08, DET-10, DET-11, DET-16, DET-17, DET-18, DET-19–DET-24, DET-25–DET-27 — Cache metadata, env vars, float accumulation, and hash fragility.
3. **Next Sprint (LOW):** DET-05, DET-07, DET-09, DET-12, DET-28 — Test-only naming, eviction fallbacks, and stable sorting.

---

*Audit performed by Kimi Code CLI on 2026-04-18. No code was modified during this audit.*
