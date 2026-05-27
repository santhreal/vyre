# AUDIT: ERROR HANDLING — vyre-conform

**Date:** 2026-04-18  
**Scope:** `vyre-conform/src/`, `vyre-conform-spec/src/`, `vyre-conform-enforce/src/`, `vyre-conform-runner/src/`, `vyre-conform-generate/src/`  
**Standard:** CLAUDE.md engineering rules (`Fix:` prefix, `#[non_exhaustive]`, no swallowed errors, no silent defaults, structured error types)  
**Findings:** 30  

---

## ERR-01 — vyre-conform/src/spec/primitive/common.rs:252 — **critical**
**Current:** `std::fs::read_to_string(&path).ok()?` swallows the underlying `std::io::Error`, returning `None` instead of actionable diagnostics.  
**Fix:** Propagate `std::io::Error` via `Result` or at minimum log the error with `tracing::warn!` before returning `None`.

## ERR-02 — vyre-conform/src/spec/primitive/common.rs:253 — **critical**
**Current:** `toml::from_str(&content).ok()?` discards TOML deserialization errors.  
**Fix:** Map `toml::de::Error` into a structured error or log it before returning `None`.

## ERR-03 — vyre-conform/src/runner/backend/wgpu/context.rs:67 — **critical**
**Current:** `adapter.request_device(...).ok()?` silently discards `wgpu::RequestDeviceError`.  
**Fix:** Return `Result<(Device, Queue), RequestDeviceError>` so callers can distinguish "no GPU" from "GPU lost".

## ERR-04 — vyre-conform/src/enforce/enforcers/layer8_feedback_loop.rs:121 — **high**
**Current:** `MUTATION_CACHE.lock().ok()?` discards mutex-poisoning information.  
**Fix:** Use `lock().expect("Fix: MUTATION_CACHE is never poisoned")` or return `Result`; silently dropping poison loses crash signals.

## ERR-05 — vyre-conform/src/verify/harnesses/mutation/categorize.rs:152 — **high**
**Current:** `let _ = cache_mutation_probe(key, killed);` discards `Result<(), String>` without logging.  
**Fix:** Propagate the error or log with `tracing::warn!` so feedback-loop cache failures are observable.

## ERR-06 — vyre-conform/src/meta/harness.rs:304 — **high**
**Current:** `let _ = Command::new("kill")...` discards process-kill failure, silently leaking orphan processes.  
**Fix:** Match on `Result` and return an error or log with `tracing::error!` when the child cannot be terminated.

## ERR-07 — vyre-conform/src/proof/algebra/minimizer.rs:15 — **critical**
**Current:** `binary_pair_violates(...).unwrap_or(false)` silently aborts minimization on checker error.  
**Fix:** Return `Result<(u32, u32), MinimizerError>` so the caller knows the minimizer failed rather than believing no violation exists.

## ERR-08 — vyre-conform/src/enforce/enforcers/divergence.rs:218 — **high**
**Current:** `inputs.first().cloned().unwrap_or_default()` silently substitutes empty input when the caller provided none.  
**Fix:** Return `BackendError::missing_input()` instead of manufacturing fake data.

## ERR-09 — vyre-conform/src/runner/suite/implementation/execute.rs:417 — **high**
**Current:** `inputs.first().cloned().unwrap_or_default()` silently substitutes empty input in CPU-mirror backend dispatch.  
**Fix:** Return `BackendError` when `inputs` is empty; do not default to a zero-length payload.

## ERR-10 — vyre-conform/src/meta/harness.rs:370 — **high**
**Current:** `fs::read_to_string(&path).unwrap_or_default()` silently returns empty string on read failure, corrupting meta-findings append logic.  
**Fix:** Propagate the `io::Error` or at least log it before falling back.

## ERR-11 — vyre-conform/src/enforce/enforcers/signature_match/parse.rs:19 — **high**
**Current:** `resolve_wgsl` returns `Result<String, String>`, forcing string-based error handling.  
**Fix:** Define `SignatureMatchError` enum with variants `WgslMissing`, `WgslDecodeFailed`, etc.

## ERR-12 — vyre-conform/src/enforce/enforcers/signature_match/parse.rs:38 — **high**
**Current:** `parse_wgsl` returns `Result<naga::Module, String>`, erasing naga's structured diagnostics.  
**Fix:** Wrap `naga::front::wgsl::ParseError` in a dedicated `WgslParseError` variant.

## ERR-13 — vyre-conform/src/meta/observe/render.rs:26 — **medium**
**Current:** `render_chrome` returns `Result<String, String>`, erasing `serde_json::Error`.  
**Fix:** Use `RenderError` enum with `Json(serde_json::Error)` variant.

## ERR-14 — vyre-conform/src/spec/registry/error.rs:165 — **medium**
**Current:** `InvalidOracle` display omits `Fix:` prefix, violating CLAUDE.md engineering standards.  
**Fix:** Prefix the message with `Fix: declare a valid oracle override or remove the field.`

## ERR-15 — vyre-conform/src/spec/registry/error.rs:167 — **medium**
**Current:** `InvalidSpecRow` display omits `Fix:` prefix.  
**Fix:** Append `Fix: correct the malformed spec-table row.` to the format string.

## ERR-16 — vyre-conform/src/spec/value/error.rs:3 — **high**
**Current:** `ValueError` enum is not marked `#[non_exhaustive]`.  
**Fix:** Add `#[non_exhaustive]` to prevent downstream crates from breaking when new variants are added.

## ERR-17 — vyre-conform/src/spec/registry/error.rs:5 — **high**
**Current:** `CoverageError` enum is not marked `#[non_exhaustive]`.  
**Fix:** Add `#[non_exhaustive]` to the enum definition.

## ERR-18 — vyre-conform/src/adversarial/mutations/catalog.rs:76 — **high**
**Current:** `MutationError` enum is not marked `#[non_exhaustive]`.  
**Fix:** Add `#[non_exhaustive]` and replace `ApplyError(String)` with a structured variant.

## ERR-19 — vyre-conform/src/runner/reporter.rs:45 — **medium**
**Current:** `eprintln!` used for progress reporting instead of `tracing::info!`.  
**Fix:** Replace `eprintln!` with `tracing::info!(op_id, input_count, "running inputs");` to respect log-level filtering.

## ERR-20 — vyre-conform/src/spec/primitive/common.rs:332 — **medium**
**Current:** `eprintln!` emits a runtime warning that bypasses structured logging.  
**Fix:** Use `tracing::warn!(op_id, actual, expected, "cpu_by_id received undersized input")`.

## ERR-21 — vyre-conform/src/spec/primitive/common.rs:156 — **critical**
**Current:** `signature_from_core` panics with `panic!` instead of returning an error.  
**Fix:** Change return type to `Result<Option<OpSignature>, PrimitiveAdapterError>` and return `Err(PrimitiveAdapterError::MultipleOutputs { ... })`.

## ERR-22 — vyre-conform/src/spec/primitive/common.rs:262 — **critical**
**Current:** `kat_path` panics on non-primitive id instead of returning `Result<PathBuf, KatPathError>`.  
**Fix:** Return `Err(KatPathError::NotPrimitive(id))` and let the caller decide whether to abort.

## ERR-23 — vyre-conform/src/runner/backend/gpu_parity.rs:41 — **critical**
**Current:** `panic!` on GPU dispatch failure inside a test helper used in production certification paths.  
**Fix:** Return `Result<Vec<u8>, BackendError>` and propagate the error up to the certify runner.

## ERR-24 — vyre-conform/src/enforce/enforcers/common.rs:12 — **critical**
**Current:** `program_with_output_size` panics on Tensor element size instead of returning `Result`.  
**Fix:** Return `Err(EnforcerError::UnsizedTensor { buffer: buffer.name() })`.

## ERR-25 — vyre-conform/src/adversarial/mutations/catalog.rs:80 — **medium**
**Current:** `MutationError::ApplyError(String)` wraps a raw string instead of structured fields.  
**Fix:** Introduce `ApplyError { source: String, offset: usize, expected: &'static str }` to give callers actionable data.

## ERR-26 — vyre-conform/src/spec/types/conform/proof_token.rs:20 — **medium**
**Current:** `ProofTokenError::VerificationFailed(String)` wraps a raw string.  
**Fix:** Replace with `VerificationFailed { detail: String, violated_law: Option<String> }` or similar structured variant.

## ERR-27 — vyre-conform/src/generate/emit/gen_error.rs:11 — **medium**
**Current:** `GenError::InvalidPlan { reason: String }` wraps a free-form string.  
**Fix:** Decompose into `InvalidPlan { cause: PlanErrorKind, affected_test: String }` to enable programmatic handling.

## ERR-28 — vyre-conform/src/spec/engine_specs/eval/vm/dispatch.rs:228 — **high**
**Current:** `first_position` uses `unwrap_or(ABORT_SENTINEL)` to mask missing signal data.  
**Fix:** Return `Option<u32>` and let the caller handle the abort logic explicitly.

## ERR-29 — vyre-conform/src/runner/execution.rs:136 — **high**
**Current:** `.expect("chain arena just received the GPU output")` will crash the certify runner if the invariant is violated.  
**Fix:** Return `CertifyError::ArenaIndexMissing` instead of panicking.

## ERR-30 — vyre-conform/src/enforce/enforcers/wire_format_eq.rs:191 — **medium**
**Current:** `original_output.as_ref().ok()` silently discards the wire-roundtrip error when building the violation report.  
**Fix:** Capture the `Err` variant in `WireFormatEquivViolation` so the report shows why round-tripping failed.

---

## Summary by Category

| Category | Count |
|---|---|
| Swallowed errors (`.ok()`, `let _ =`) | 6 |
| `.unwrap_or(default)` silencing real failures | 4 |
| `Result<T, String>` instead of structured errors | 3 |
| Missing `Fix:` prefix | 2 |
| Missing `#[non_exhaustive]` on error enums | 3 |
| `eprintln!` / `println!` for diagnostics | 2 |
| Panics that should be returned errors | 5 |
| `unwrap()` / `expect()` in production | 2 |
| Error variants wrapping strings | 3 |
| **Total** | **30** |
