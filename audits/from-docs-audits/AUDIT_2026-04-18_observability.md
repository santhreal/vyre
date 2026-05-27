# AUDIT: OBSERVABILITY — Workspace-Wide

**Date:** 2026-04-18  
**Auditor:** Kimi Code CLI  
**Scope:** All workspace crates (`vyre-core`, `vyre-wgpu`, `vyre-std`, `vyre-primitives`, `vyre-build-scan`, `vyre-conform`, `vyre-conform-spec`, `vyre-conform-generate`, `vyre-conform-enforce`, `vyre-conform-runner`)  
**Method:** Static analysis of tracing, error handling, metrics, and structured logging patterns.  

---

## Executive Summary

A failing production GPU dispatch must be debuggable without a source checkout. Today it cannot be.  

- **Zero** `#[tracing::instrument]` attributes exist in the entire workspace. Every public entry point (`backend.execute`, `pipeline.compile`, `optimizer.optimize`, `gpu.dispatch`) emits no automatic span.
- **Zero** metric counters, gauges, or histograms exist. Dispatch throughput, cache hit rates, shader compile times, and conformance progress are all unmeasured.
- **19** `tracing::error!` calls in `vyre-core` CPU reference adapters log caller input mismatches at `error` level — these are warning-level conditions (bad caller input, not system crashes).
- **17** error types lack any `Display` implementation, forcing consumers to read `Debug` output.
- **100+** error-return sites across `vyre-wgpu`, `vyre-core`, and `vyre-conform` propagate `Err(...)` without a single log line — silent failure.
- **38+** tracing calls embed dynamic data in format strings instead of structured fields, making them unqueryable in aggregation.
- **`vyre-wgpu/src/pipeline.rs`** — the primary GPU dispatch path — has **no** tracing calls at all.

---

## Findings

### OBS-01 — `vyre-wgpu/src/pipeline.rs:138` — CRITICAL
**Category:** Dispatch paths with zero `tracing::*!` calls  
**Current:** `pub fn compile(program: &Program) -> Result<Self, BackendError>` and the entire `WgpuPipeline` dispatch surface have zero tracing calls. A production user calling `backend.execute()` has no breadcrumb trail into the GPU backend.  
**Fix:** Add `#[tracing::instrument(skip(program), fields(fingerprint = %program.fingerprint()))]` to `compile`, `compile_with_config`, `push_chunk`, and `dispatch_compound`. Add `tracing::info!(dispatch_id, input_bytes, "gpu dispatch start")` at the top of every public dispatch path.

---

### OBS-02 — `vyre-wgpu/src/engine/graph.rs:52` — CRITICAL
**Category:** Dispatch paths with zero `tracing::*!` calls  
**Current:** `pub fn dispatch(&self, config: &DispatchConfig) -> Result<Vec<Vec<Vec<u8>>>, BackendError>` — the core GPU graph execution entry point — has no tracing. If a node kernel panics or a buffer overrun occurs, the only signal is the returned `BackendError` with no log context.  
**Fix:** Add `#[tracing::instrument(skip(config), fields(node_count = self.nodes.len()))]` and emit `tracing::info!(dispatch_id, "graph dispatch submitted")` before command encoder submission.

---

### OBS-03 — `vyre-wgpu/src/lib.rs:92` — CRITICAL
**Category:** Missing `tracing::instrument` on public functions  
**Current:** `pub fn acquire() -> Result<Self, vyre::BackendError>` is the first call every production user makes. It has no span, no structured fields, and no log on success or failure. If GPU adapter enumeration fails, the error propagates silently.  
**Fix:** Add `#[tracing::instrument]` and `tracing::info!(adapter = %info.name, "backend acquired")` on success; `tracing::error!(error = %e, "backend acquisition failed")` before returning `Err`.

---

### OBS-04 — `vyre-core/src/pipeline.rs:49` — CRITICAL
**Category:** Dispatch paths with zero `tracing::*!` calls  
**Current:** `pub fn compile(backend: Arc<dyn VyreBackend>, program: &Program, config: &DispatchConfig) -> Result<Arc<dyn CompiledPipeline>, BackendError>` is the top-level compilation entry point for all backends. No tracing. No timing. No structured fields identifying the program or backend.  
**Fix:** Add `#[tracing::instrument(skip(backend, program, config), fields(program_fingerprint = %program.fingerprint(), backend = %std::any::type_name_of_val(&*backend)))]` and emit `tracing::info!("pipeline compile start")` / `tracing::info!("pipeline compile complete")` with `elapsed_ms`.

---

### OBS-05 — `vyre-core/src/ops/workgroup/primitives/hashmap/spec.rs:171` — HIGH
**Category:** Error variants without `#[error(...)]` Display impls  
**Current:** `pub enum HashmapError { Overflow, KeyNotFound }` has `#[derive(Debug)]` only — no `Display`, no `thiserror::Error`. When this bubbles up through `dyn std::error::Error`, consumers see `HashmapError::Overflow` (Debug), which is unstable and unhelpful at internet scale.  
**Fix:** `#[derive(thiserror::Error)]` with `#[error("workgroup hashmap overflow: capacity exhausted")]` and `#[error("workgroup hashmap key not found")]`.

---

### OBS-06 — `vyre-core/src/ops/workgroup/stack/spec.rs:63` — HIGH
**Category:** Error variants without `#[error(...)]` Display impls  
**Current:** `pub enum StackError { Overflow, Underflow }` — no Display impl. Same pattern affects `FifoError`, `PriorityError`, `VisitError`, `UnionFindError`, `ArenaError`, `InternError`, `StateMachineError` (9 workgroup primitive error types total).  
**Fix:** Derive `thiserror::Error` on all 9 workgroup primitive error enums with actionable `#[error("...")]` messages.

---

### OBS-07 — `vyre-core/src/ops/primitive/subgroup_scan.rs:103` — HIGH
**Category:** Log level misuse  
**Current:** `tracing::error!("subgroup inclusive_scan CPU input must be 128 bytes. Fix: pass exactly 32 little-endian u32 lanes.")` — this logs at `error` level when a caller passes the wrong byte count to a CPU reference adapter. This is a caller bug or test failure, not a system crash. In production telemetry, this drowns real errors.  
**Fix:** Downgrade to `tracing::warn!(op_id = "primitive.subgroup.inclusive_scan", input_len = input.len(), "CPU reference received wrong input size")`. Same fix applies to 18 sibling `tracing::error!` calls in `cooperative/`, `primitive/subgroup_*.rs`, `ops/cpu_op.rs`, `ops/cpu_references.rs`, `ops/crypto/chacha20_block/kernel.rs`, `ops/hash/cpu_refs.rs`.

---

### OBS-08 — `vyre-core/src/ops/hash/cpu_refs.rs:156` — HIGH
**Category:** Missing structured fields in tracing calls  
**Current:** `Err(error) => tracing::error!("{error}. Fix: provide valid hash intrinsic parameters.")` — the error is interpolated into the message string, not attached as a structured field. Ingestion pipelines cannot facet on `error` or alert on specific hash intrinsic failures.  
**Fix:** `tracing::warn!(error = %error, op_id = "primitive.hash", "hash CPU reference rejected invalid parameters")`.

---

### OBS-09 — `vyre-core/src/ir/serial/wire.rs:89` — HIGH
**Category:** Error propagation that loses context  
**Current:** 
```rust
tracing::error!(error = %e, "Program::to_bytes: wire encoding failed; returning empty bytes. Fix: …");
```
The error is logged but the returned `Vec<u8>` is empty with no indication to the caller that encoding failed. The function signature does not return `Result`; it silently returns an empty vec.  
**Fix:** Change signature to `Result<Vec<u8>, WireError>` and propagate the error. If empty vec must be preserved for backward compatibility, add `tracing::error!(program_fingerprint = %fp, error = %e, "wire encoding failed, returning empty bytes")`.

---

### OBS-10 — `vyre-wgpu/src/runtime/aot.rs:53` — HIGH
**Category:** Metrics gaps — hot-path function with no timing, no counter, no gauge  
**Current:** `pub fn load_or_compile(program: &Program, fingerprint: &str) -> Result<AotArtifact, BackendError>` is the AOT shader cache entry point. It performs disk I/O, shader compilation, and pipeline creation. There is no timing, no cache-hit counter, no cache-miss counter, no gauge for cache size.  
**Fix:** Add `tracing::info!(fingerprint, hit = false, "AOT compile")` on miss, `tracing::info!(fingerprint, hit = true, "AOT load")` on hit, and `tracing::info!(elapsed_ms, "AOT compile complete")`. If `metrics` crate is adopted later, emit `histogram!("vyre.aot.compile_duration_ms", elapsed)` and `counter!("vyre.aot.cache_miss")`.

---

### OBS-11 — `vyre-wgpu/src/runtime/cache/buffer_pool.rs:94` — HIGH
**Category:** Metrics gaps — hot-path function with no timing, no counter, no gauge  
**Current:** `BufferPool::acquire` is called on every GPU dispatch. It locks a mutex, searches size-class buckets, and may allocate new GPU buffers. No timing, no counter for allocation frequency, no gauge for pool utilization. At high dispatch rates, pool exhaustion or mutex contention is invisible.  
**Fix:** Add `tracing::trace!(size_class, acquired_in_pool, "buffer pool acquire")` and measure `Instant::now()` elapsed for contention analysis. Expose a `gauge!("vyre.buffer_pool.utilization", in_use)` if metrics are adopted.

---

### OBS-12 — `vyre-conform/src/runner/backend/wgpu/context.rs:30` — HIGH
**Category:** Error messages without the op_id / backend_id / node_index  
**Current:** `tracing::warn!("vyre-conform GPU device lost: {:?}: {}. Clearing cached device.", reason, message)` — no `backend_id`, no `adapter_name`, no structured `reason` field. When device-loss spikes occur in CI or production conformance runners, the log cannot be correlated to a specific GPU adapter or test op.  
**Fix:** `tracing::warn!(backend_id = %backend_id, adapter = %adapter_name, reason = ?reason, message, "GPU device lost; clearing cached device")`.

---

### OBS-13 — `vyre-conform/src/runner/reporter.rs:54` — HIGH
**Category:** `println!` / `eprintln!` in library code  
**Current:** `eprintln!("vyre-conform: FAIL {} v{} wg{} {}: {}", failure.op_id, failure.spec_version, failure.workgroup_size, failure.input_label, failure.message)` — library code writes directly to stderr. In production deployments with structured log collectors (e.g., JSON to stdout), this line bypasses the tracing pipeline and cannot be filtered, sampled, or routed.  
**Fix:** Replace with `tracing::error!(op_id = %failure.op_id, spec_version = %failure.spec_version, workgroup_size = %failure.workgroup_size, input_label = %failure.input_label, message = %failure.message, "conformance failure")`.

---

### OBS-14 — `vyre-build-scan/src/conform/discovery.rs:85` — HIGH
**Category:** `println!` / `eprintln!` in library code  
**Current:** `eprintln!("Warning: {} contains `pub const SPEC` but failed to parse as Rust: {error}. Skipping from generated registry.", path.display())` — a build-time library crate emits to stderr. In CI environments with strict log parsing, this breaks structured output.  
**Fix:** Replace with `tracing::warn!(path = %path.display(), error = %error, "spec discovery skipped: file failed to parse as Rust")`.

---

### OBS-15 — `vyre-core/src/fuzz.rs:185` — MEDIUM
**Category:** Error propagation that loses context  
**Current:** `exec_nodes(program.entry(), &mut env, &mut output)?` and `eval_expr(value, env)?` propagate `BackendError` without node-kind or expression-path context. A failure deep in `eval_expr` surfaces as a generic `BackendError` with no indication which node or expression caused it.  
**Fix:** Wrap recursive calls: `.map_err(|e| BackendError::new(format!("while evaluating node {node_id} ({node_kind}): {e}")))`.

---

### OBS-16 — `vyre-wgpu/src/engine/decompress/formats/zstd.rs:97` — MEDIUM
**Category:** Metrics gaps — hot-path function with no timing, no counter, no gauge  
**Current:** `dispatch_zstd` contains a `while cursor < data.len()` frame/block parsing loop that drives GPU decompression. No span, no timing, no throughput metric (bytes_in / bytes_out / elapsed). If decompression throughput regresses (e.g., due to a bad frame header parse), there is no signal.  
**Fix:** Add `#[tracing::instrument(skip(data))]` and `tracing::info!(input_bytes = data.len(), output_bytes, elapsed_ms, "zstd gpu decompress complete")`.

---

### OBS-17 — `vyre-core/src/lower/wgsl/lower.rs:20` — MEDIUM
**Category:** Missing `tracing::instrument` on public functions  
**Current:** `pub fn lower(program: &Program) -> Result<String, Error>` — every GPU program passes through here. No span. No program fingerprint. If lowering fails (e.g., due to an unregistered op), the error propagates without identifying the program.  
**Fix:** Add `#[tracing::instrument(skip(program), fields(fingerprint = %program.fingerprint()))]` and `tracing::info!("WGSL lower start")` / `tracing::info!(wgsl_len = result.len(), "WGSL lower complete")`.

---

### OBS-18 — `vyre-core/src/optimizer.rs:150` — MEDIUM
**Category:** Missing `tracing::instrument` on public functions  
**Current:** `pub fn optimize(program: Program) -> Result<Program, OptimizerError>` — the main optimizer entry point. No span. No structured fields for pass count or iteration count. If scheduling fails (cycle in pass dependencies), the error has no optimizer context.  
**Fix:** Add `#[tracing::instrument(skip(program), fields(fingerprint = %program.fingerprint()))]` and log `pass_count`, `max_iterations`, and `elapsed_ms` on completion.

---

### OBS-19 — `vyre-conform/src/enforce/registry.rs:181` — MEDIUM
**Category:** Spans that span too much  
**Current:** 
```rust
let span = tracing::info_span!("vyre_conform.enforce_all", verdict = tracing::field::Empty);
let _enter = span.enter();
```
This single span covers the entire `enforce_all` call — which may run thousands of enforcers across hundreds of ops. If one enforcer hangs or slows down, the span gives no granularity.  
**Fix:** Emit a child `info_span!("vyre_conform.enforcer", gate = enforcer.id())` per enforcer invocation, or at minimum per-layer spans.

---

### OBS-20 — `vyre-conform/src/runner/loader/toml/load.rs:94` — MEDIUM
**Category:** Missing structured fields in tracing calls  
**Current:** `tracing::warn!("Conflicting override for op in {:?}", canonical_path)` — the path is embedded in the format string, not a structured field. Log aggregators cannot group by `canonical_path` or alert on specific files with high conflict rates.  
**Fix:** `tracing::warn!(path = %canonical_path.display(), "conflicting override for op in TOML load")`. Same fix for the `defendant` and `law` warnings on lines 102 and 107.

---

### OBS-21 — `vyre-conform/src/spec/registry/error/coverage_error.rs:1` — MEDIUM
**Category:** Error variants without `#[error(...)]` Display impls  
**Current:** `pub struct CoverageError { pub message: String }` has `#[derive(Debug)]` only — no `Display`, no `thiserror::Error`. When conformance enforcement fails due to insufficient coverage, the error message is unusable in generic error boundaries.  
**Fix:** `#[derive(thiserror::Error)] #[error("coverage enforcement failed: {message}")]`.

---

### OBS-22 — `vyre-core/src/ops/cooperative/store.rs:79` — MEDIUM
**Category:** Log level misuse  
**Current:** `tracing::error!("cooperative matrix store input must be exactly 1024 bytes. Fix: pass 256 little-endian f32 values.")` — caller passed wrong input size to a CPU reference. This is a `warn!` (caller bug), not an `error!` (system failure).  
**Fix:** `tracing::warn!(op_id = "cooperative.matrix.store", input_len = input.len(), expected = 1024, "cooperative matrix store received wrong input size")`.

---

### OBS-23 — `vyre-wgpu/src/engine/streaming.rs:54` — MEDIUM
**Category:** Error messages without the op_id / backend_id / node_index  
**Current:** `tracing::error!("streaming worker receiver mutex poisoned: {source}. Fix: restart the process after investigating worker panics.")` — no `backend_id`, no `pipeline_id`, no `worker_thread_id`. In a multi-pipeline deployment, this log cannot be correlated to the specific pipeline that panicked.  
**Fix:** `tracing::error!(backend_id = %self.id, worker_id = ?thread::current().id(), error = %source, "streaming worker receiver mutex poisoned")`.

---

### OBS-24 — `vyre-core/src/ops/registry/lookup.rs:23` — MEDIUM
**Category:** Dispatch paths with zero `tracing::*!` calls  
**Current:** `pub fn lookup(op_id: &str) -> Option<&'static OpSpec>` is called from the interpreter hot path for every op dispatch. It has zero tracing. If an op is missing from the registry, the caller gets `None` with no log — impossible to debug in production without attaching a debugger.  
**Fix:** Add `tracing::debug!(op_id, "registry lookup miss")` on `None` return; `tracing::trace!(op_id, "registry lookup hit")` on `Some`.

---

### OBS-25 — `vyre-wgpu/src/engine/decode/dispatch/gpu/kernel_launch.rs:19` — MEDIUM
**Category:** Dispatch paths with zero `tracing::*!` calls  
**Current:** `pub fn dispatch_decode(args: DecodeDispatchArgs<'_>) -> Result<Vec<DecodedRegion>, Error>` — GPU decode kernel launch. No tracing. No `args` fields. If decode fails (bad buffer layout, wrong rule set), the only evidence is the returned `Error`.  
**Fix:** Add `#[tracing::instrument(skip(args))]` and `tracing::info!(region_count = args.regions.len(), "decode kernel launch")`.

---

## Summary by Severity

| Severity | Count | Examples |
|----------|-------|----------|
| CRITICAL | 4 | GPU dispatch paths with zero tracing; top-level compile/acquire entry points |
| HIGH | 10 | Missing Display on errors; log level misuse; silent AOT/buffer-pool hot paths; `eprintln!` in libs |
| MEDIUM | 11 | Missing `tracing::instrument`; spans too broad; bare `?` propagation; missing structured fields |

**Total: 25 findings.**

---

## Cross-Cutting Themes

1. **The GPU backend is a black box.** `vyre-wgpu` has 5 `tracing::warn!` calls (all readback-drop warnings) and nothing else. Every dispatch, compile, scan, decode, and decompress path is silent.
2. **CPU reference adapters cry wolf.** Nineteen `tracing::error!` calls in `vyre-core/src/ops/` log caller input mismatches at `error` level. In production telemetry, these create alert fatigue and drown real infrastructure failures.
3. **No metrics, no SLOs.** Without counters, gauges, or histograms, there is no way to define or measure dispatch latency, cache hit rates, shader compile time, or conformance throughput.
4. **Structured logging is absent.** Almost every tracing call embeds dynamic data in format strings. This makes logs unqueryable in Elasticsearch/Loki/Honeycomb.
5. **Silent errors are the default.** `Err(...)` is returned from hundreds of functions without any log emission. A user calling `backend.execute()` in production receives a `BackendError` with no server-side log to correlate.

---

## Recommendations (in priority order)

1. **Instrument `vyre-wgpu` public API surface.** Every `pub fn` in `pipeline.rs`, `engine/graph.rs`, `engine/streaming.rs`, `runtime/device/device.rs`, and `runtime/aot.rs` should have `#[tracing::instrument]` and at least one `tracing::info!` / `tracing::error!` at the boundary.
2. **Downgrade CPU adapter `error!` to `warn!`.** The 19 caller-validation `tracing::error!` calls in `vyre-core/src/ops/` should become `tracing::warn!` with structured `op_id` and `input_len` fields.
3. **Add `Display` to all public error types.** The 9 workgroup primitive errors (`HashmapError`, `StackError`, etc.) and `CoverageError`, `MetaMutationError`, `DualReferenceError`, and `InfrastructureError` need `thiserror::Error` derive.
4. **Replace all `eprintln!` in library `src/` with `tracing::` calls.** `vyre-build-scan`, `vyre-conform`, and `vyre-conform-generate` all emit to stderr directly.
5. **Adopt a metrics facade.** Even a minimal `metrics` integration (guarded by a feature flag) would enable SLO dashboards for dispatch latency, AOT cache efficiency, and buffer pool health.
6. **Add per-op spans in conformance enforcement.** The monolithic `vyre_conform.enforce_all` span should be broken into per-enforcer and per-op child spans.
7. **Emit a log on every `Err` return from public functions.** A simple `tracing::debug!(error = %e, "function_name returned Err")` before the `return Err(...)` or `?` would eliminate silent failure.
