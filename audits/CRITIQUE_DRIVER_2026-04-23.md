# vyre-driver/src/ Security & Architecture Critique
**Date:** 2026-04-23  
**Scope:** `vyre-driver/src/` (backend abstraction layer)  -  read-only audit.  
**Auditor:** Kimi Code CLI  
**Laws applied:** 0–8, Unix/SQLITE standard, maximal elegance, test everything.

---

## Methodology
1. Grepped every `unwrap()`, `expect()`, `todo!()`, `unimplemented!()` in non-test code.  
2. Inspected every `#[derive(Debug)]` on a public type for missing `Display`/`Error` trails.  
3. Verified `shadow.rs` against F-IR-41 (exhaustive matrix replacement).  
4. Analyzed `registry/migration.rs` for semver boundary enforcement, cycle safety, and duplicate hijacking.  
5. Grepped every public `fn` for missing `#[must_use]` where return-value suppression is a soundness hazard.  
6. Grepped every `&mut self` receiver to confirm genuine mutation (no false write claims).

---

## Findings

### 1. Unwrap / Expect in Non-Test Code  -  Backend Wrapper Must Never Panic

#### Finding 1.1  -  CRITICAL | `diagnostics.rs:319`
```rust
serde_json::to_string(self).unwrap()
```
`Diagnostic::to_json()` panics if any field ever becomes non-serializable (future-proofing regression). The `#[allow(clippy::unwrap_used)]` annotation is a band-aid, not a fix. A backend wrapper must propagate `BackendError` with a `Fix:` hint.

**Fix:** Change return type from `String` to `Result<String, BackendError>`. Map `serde_json::Error` into `BackendError::new(format!("diagnostic JSON serialization failed: {e}. Fix: inspect the Diagnostic for non-serializable fields."))`.

**Test hint:** Add an adversarial test that constructs a `Diagnostic` with a malformed `Cow<'static, str>` (simulate via `DiagnosticCode::new` with invalid UTF-8 surrogate) and assert it returns `Err` rather than panicking.

---

#### Finding 1.2  -  CRITICAL | `shadow.rs:238`
```rust
fn program_fingerprint(program: &Program) -> [u8; 32] {
    let wire = program.to_wire().expect("Fix: conformance Program must always serialize");
    ...
}
```
`program_fingerprint` is called from the non-test `assert_exhaustive_byte_identity` path. `Program::to_wire()` returns `Result`; an IR bug or future extension can trigger a driver panic during conformance.

**Fix:** Make `program_fingerprint` return `Result<[u8; 32], ConformanceError>` (or `BackendError`). In `assert_exhaustive_byte_identity`, map the error into `ConformanceError::ReferenceRejected` or a new `ConformanceError::FingerprintFailed`.

**Test hint:** Inject a stub `Program` whose `to_wire()` returns `Err` and assert `assert_exhaustive_byte_identity` surfaces the error instead of panicking.

---

#### Finding 1.3  -  CRITICAL | `registry/registry.rs:133`
```rust
Self::validate_no_duplicates(defs.iter()).unwrap_or_else(|err| panic!("{err}"));
```
`DialectRegistry::from_inventory()` panics when two linked dialect crates claim the same op id. This is a process startup failure with no recovery path for backend authors.

**Fix:** Refactor `from_inventory()` → `try_from_inventory() -> Result<Self, DuplicateOpIdError>`. Propagate the error through `registry_swap()` and `global()`. Since `global()` currently returns `Guard<Arc<Self>>`, change it to `Result<Guard<Arc<Self>>, BackendError>` (or introduce a fallible `global()` and an infallible `global_unchecked()` documented as panicking).

**Test hint:** Register two `inventory::submit!` blocks with colliding op ids in a test binary and assert `try_from_inventory()` returns `Err(DuplicateOpIdError { .. })` instead of aborting the process.

---

#### Finding 1.4  -  CRITICAL | `registry/registry.rs:145`
```rust
vyre_foundation::extern_registry::verify().unwrap_or_else(|errors| {
    let message = errors.into_iter().map(|e| e.to_string()).collect::<Vec<_>>().join("; ");
    panic!("extern dialect inventory is invalid. Fix: {message}");
});
```
Same pattern as 1.3: invalid extern inventory triggers a startup panic rather than a structured error.

**Fix:** Return the verification error as `BackendError` or `DuplicateOpIdError` and propagate through `try_from_inventory()`.

**Test hint:** Submit an `ExternOp` whose dialect has no matching `ExternDialect` and assert the registry initialization returns a structured error.

---

### 2. Debug-Only Public Surfaces  -  Missing Display / Error Trails

Backend authors and runtime observability pipelines need `#[derive(thiserror::Error)]`-style `Display` output, not `{:?}` noise, for every public type that appears in logs, diagnostics, or error chains.

#### Finding 2.1  -  HIGH | `backend/error.rs:10`
```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCode { ... }
```
`ErrorCode` is the machine-readable classification of every backend failure. It has no `Display` impl. When logged via `tracing::error!("{}", err.code())`, the output is a `Debug` dump instead of a stable human name (e.g., `KernelCompileFailed`).

**Fix:** `impl std::fmt::Display for ErrorCode` mapping each variant to its kebab-case name (`device-out-of-memory`, `shader-compile-failed`, etc.). Alternatively derive `strum::Display` if the dependency is acceptable.

**Test hint:** Assert that `format!("{}", ErrorCode::KernelCompileFailed) == "shader-compile-failed"` and that every variant round-trips without panicking.

---

#### Finding 2.2  -  HIGH | `diagnostics.rs:59`
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Severity { Error, Warning, Note }
```
`Severity` lacks `Display`. It appears in every rendered diagnostic and LSP JSON payload. Debug formatting leaks internal enum structure into user-facing logs.

**Fix:** `impl Display for Severity` forwarding to `self.label()` (`"error"`, `"warning"`, `"note"`).

**Test hint:** Assert `format!("{}", Severity::Warning) == "warning"`.

---

#### Finding 2.3  -  HIGH | `diagnostics.rs:124`
```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpLocation { ... }
```
`OpLocation` is the primary anchor for every diagnostic. Without `Display`, backend authors and IDE integrators must hand-roll formatting.

**Fix:** `impl Display for OpLocation` producing `op `foo` operand[2] attr `mode``  -  identical to the human rendering in `Diagnostic::render_human()` so there is a single source of truth.

**Test hint:** Construct `OpLocation::op("math.add").with_operand(2).with_attr("mode")` and assert `Display` output matches the rustc-style location string.

---

#### Finding 2.4  -  MEDIUM | `registry/enforce.rs:21`
```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnforceVerdict { Allow, Deny { policy: &'static str, detail: String } }
```
Conformance gates surface `EnforceVerdict` to callers. Debug formatting is inappropriate for CI logs and editor integrations.

**Fix:** `impl Display for EnforceVerdict` rendering `Allow` as `"allow"` and `Deny` as `"deny[{policy}]: {detail}"`.

**Test hint:** Assert `format!("{}", EnforceVerdict::Deny { policy: "registry_gate", detail: "Fix: ...".into() })` contains the policy and detail verbatim.

---

### 3. `shadow.rs` Legacy Sampling Residue

#### Finding 3.1  -  MEDIUM | `lib.rs:26`
```rust
/// Sampled CPU-reference shadow execution of live dispatches.
pub mod shadow;
```
The module-level doc comment in `lib.rs` is stale. F-IR-41 replaced sampling with an exhaustive conformance matrix. The comment misleads new backend authors into believing the driver still relies on statistical sampling.

**Fix:** Update the doc comment to:
```rust
/// Exhaustive CPU-vs-backend conformance matrix for compiled pipelines.
/// Replaces the legacy sampled shadow path (F-IR-41); every witness tuple
/// is executed and compared byte-for-byte.
```

**Test hint:** N/A (documentation fix). Verify with `cargo doc --no-deps` that the rendered module description no longer mentions sampling.

---

#### Finding 3.2  -  CLEAN
`shadow.rs` itself contains **no legacy sampling code**. `assert_exhaustive_byte_identity` iterates the full `ConformanceMatrix`. The test `exhaustive_matrix_catches_divergence_hidden_by_sampling` correctly references sampling only as a historical negative example.

---

### 4. `registry/migration.rs`  -  Semver Boundaries & Chain Safety

#### Finding 4.1  -  CRITICAL | `registry/migration.rs:332-349`
```rust
pub fn apply_chain(...) -> Result<(&'static str, Semver), MigrationError> {
    let mut current_op = op_id;
    let mut current_ver = from;
    loop {
        let Some(m) = self.lookup(current_op, current_ver) else {
            return Ok((current_op, current_ver));
        };
        (m.rewrite)(attrs)?;
        current_op = m.to.0;
        current_ver = m.to.1;
    }
}
```
**No cycle detection.** A malicious or buggy inventory with reciprocal migrations (v1→v2 and v2→v1) causes an infinite loop. This is a denial-of-service against the wire decoder.

**Fix:** Add a `visited: HashSet<(String, Semver)>` or a step counter (`steps > 1024`). On cycle detection, return `MigrationError::CycleDetected { op_id, version }`.

**Test hint:** Register `test.op_cycle` v1→v2 and v2→v1, call `apply_chain("test.op_cycle", v1, ...)` and assert it returns `Err(MigrationError::CycleDetected { .. })` within bounded time.

---

#### Finding 4.2  -  HIGH | `registry/migration.rs:296-311` (construction) & `332-349` (traversal)
**No monotonicity enforcement.** `MigrationRegistry::global()` accepts any `from`/`to` pair. A registration can declare v2→v1, enabling downgrade. Combined with forward migrations, this creates cycles (see 4.1) and violates semver semantics.

**Fix:** During registry construction, reject migrations where `m.to.1 <= m.from.1` (using `Semver`’s `Ord` impl). Return a structured error: `BackendError::new("migration registry contains non-monotonic step ... Fix: ...")`.

**Test hint:** Attempt to register v2→v1 and assert construction fails with a clear error.

---

#### Finding 4.3  -  CRITICAL | `registry/migration.rs:299-301`
```rust
for m in inventory::iter::<Migration> {
    forward.insert((m.from.0.to_owned(), m.from.1), m);
}
```
**Duplicate registrations are silently overwritten.** If two crates submit migrations for the same `(op_id, from_version)`, the last-linked crate wins. A malicious dependency can hijack a standard op’s migration chain, rewriting attributes arbitrarily.

**Fix:** Detect collisions during `global()` construction. If a key already exists, return `Err(BackendError::new(...))` or panic with a `Fix:` hint in debug builds, but prefer propagating a `BackendError` to comply with the no-panic mandate.

**Test hint:** Submit two conflicting migrations for `test.op_dup` v1→v2 and v1→v3. Assert that `MigrationRegistry::global()` returns an error (or the registry exposes a fallible constructor).

---

#### Finding 4.4  -  Question Answered
> Can a 2.0 cert authenticate as 1.0 via a chain that skips an intermediate migration?

**No  -  by default.** `apply_chain` only walks *forward* via `lookup(current_op, current_ver)`. There is no backward lookup, so a 2.0 payload cannot downgrade to 1.0.

**However,** because monotonicity is *not enforced* (Finding 4.2), a buggy or malicious registration could insert a backward step (2.0→1.0). In that case the answer becomes **yes**. The boundary is therefore **not reliably enforced**.

---

### 5. Missing `#[must_use]`  -  Soundness Bugs from Ignored Returns

#### Finding 5.1  -  HIGH | `registry/migration.rs:115`
```rust
pub fn insert(&mut self, key: impl Into<String>, value: AttrValue) -> Option<AttrValue> {
```
Ignoring the `Option<AttrValue>` return silently drops the previous attribute value. In a migration chain, this causes data loss: an existing attribute is overwritten without the migration author knowing.

**Fix:**
```rust
#[must_use = "ignoring the previous value may silently drop attributes during a migration chain"]
```

**Test hint:** Write a migration that calls `attrs.insert("mode", ...)` on a map that already contains `"mode"`. Assert that the compiler warns (or that the test fails if `#[must_use]` is ignored via `let _ = ...`).

---

#### Finding 5.2  -  HIGH | `registry/migration.rs:132`
```rust
pub fn rename(&mut self, from: &str, to: impl Into<String>) -> bool {
```
Ignoring the `bool` return means a migration cannot tell whether the source key existed. A `false` return indicates the key was absent; proceeding as if the rename succeeded leaves the attribute map in an invalid state for downstream schema validation.

**Fix:**
```rust
#[must_use = "ignoring the result may cause the migration to proceed with a missing attribute"]
```

**Test hint:** Call `rename("missing", "new")` and drop the result; assert the compiler emits a `must_use` warning.

---

#### Finding 5.3  -  MEDIUM | `registry/migration.rs:120`
```rust
pub fn remove(&mut self, key: &str) -> Option<AttrValue> {
```
Ignoring the return means a migration does not know whether the attribute was actually removed. Downstream steps that expect the attribute to be gone may still see it, or vice versa.

**Fix:**
```rust
#[must_use = "ignoring the result may break downstream migration steps that depend on the removed attribute"]
```

**Test hint:** Same pattern as 5.1  -  assert `must_use` warning fires.

---

### 6. False Write Claims  -  `&mut self` with No Mutation

**No findings.** Every `&mut self` receiver in `vyre-driver/src/` performs genuine mutation:
- `AttrMap::{insert,remove,rename}` mutate the inner `HashMap`.
- `TomlDialectStore::{scan_dir,load_file}` mutate `manifests` and `diagnostics`.
- `FixedUniqueU32::observe` mutates `values`, `len`, and `overflowed`.
- `ConformanceMatrix::push` mutates `cases`.
- `PgoTable::certify_op` mutates `routes`.
- `NagaGenCtx::register_expression` is a trait contract whose name implies mutation.

The codebase is clean on this axis.

---

## Summary  -  Top 3 to Escalate

| Rank | Finding | Severity | Why Escalate |
|------|---------|----------|--------------|
| 1 | **4.1**  -  Migration chain has no cycle detection (`registry/migration.rs:332-349`) | CRITICAL | Infinite loop = thread hang / DoS. A single malicious `inventory::submit!` can freeze the wire decoder. Fix is a bounded visited-set. |
| 2 | **4.3**  -  Duplicate migrations silently overwritten (`registry/migration.rs:299-301`) | CRITICAL | Supply-chain attack surface: a downstream crate can hijack a standard op’s migration chain, rewriting attributes arbitrarily without detection. |
| 3 | **1.1**  -  `Diagnostic::to_json()` unwraps (`diagnostics.rs:319`) | CRITICAL | Panic in the diagnostic serialization path breaks LSP integrations, CI annotators, and log pipelines. Any future non-serializable field triggers a production crash. |

### Honorable Mentions (fix in same PR)
- **1.3 / 1.4**  -  Registry startup panics on duplicate/invalid inventory. Refactor to `Result` propagation.
- **4.2**  -  Non-monotonic migrations accepted. Enforce `to >= from` at construction time.
- **5.1 / 5.2**  -  Missing `#[must_use]` on `AttrMap::insert` and `rename`. Silent data loss in migration chains.
- **2.1**  -  `ErrorCode` lacks `Display`. Every backend error log currently dumps `Debug` formatting.

---

*Audit generated by automated read-only analysis. All line numbers refer to commit `HEAD` of `vyre-driver/src/` as of 2026-04-23.*
