# Audit: DOC TRUTH — Documentation vs. Code Mismatch

**Date:** 2026-04-18
**Scope:** All crates in the vyre workspace — doc comments, READMEs, ARCHITECTURE.md, Cargo.toml references, and inline code examples.
**Method:** Manual source review + `cargo doc --no-deps` + signature verification against actual definitions.
**Rule:** A doc that lies is worse than no doc. Every finding below is a mismatch between what the prose promises and what the code delivers.

---

## Findings

### DOC-01 — `vyre-core/README.md:25` — **high**
- **Current:** Doc example imports `cached_device`, `compile_compute_pipeline`, `bg_entry` from `vyre::runtime`.
- **Reality:** These symbols live in `vyre-wgpu`, not `vyre`. The example will not compile against the `vyre` crate as documented.
- **Fix:** Change imports to `vyre_wgpu::runtime::{cached_device, compile_compute_pipeline, bg_entry}` or rewrite the example to use the `vyre-wgpu` public surface.

### DOC-02 — `vyre-core/README.md:77` — **high**
- **Current:** "Buffer readback and a complete runnable version are in `examples/02_xor_gpu_dispatch.rs`."
- **Reality:** `vyre-core/examples/02_xor_gpu_dispatch.rs` exists, but its entire body is gated behind `#[cfg(feature = "gpu")]` (line 13–19). The `vyre-core` crate (published as `vyre`) defines no `gpu` feature in its `Cargo.toml`, so the example is always a no-op `fn main() {}` on the non-gpu path. A user following the README will not find a runnable example.
- **Fix:** Remove the reference or point to an existing, working example such as `examples/hello_vyre`.

### DOC-03 — `vyre-conform/README.md:28` — **high**
- **Current:** `use vyre_conform::{certify, registry, CertificateStrength, WgslBackend};`
- **Reality:** `vyre_conform` does not export `WgslBackend`. The type exists in `vyre-wgpu` (as `WgpuBackend`). The README example will not compile.
- **Fix:** Remove `WgslBackend` from the `vyre_conform` import and import the correct backend type from `vyre_wgpu`.

### DOC-04 — `ARCHITECTURE.md:248` — **high**
- **Current:** `pub fn certify(backend: &dyn VyreBackend) -> Result<Certificate, Violation>;`
- **Reality:** The actual signature in `vyre-conform/src/runner/certify/implementation.rs:361` is:
  ```rust
  pub fn certify(
      backend: &dyn vyre::VyreBackend,
      specs: &[OpSpec],
      strength: CertificateStrength,
  ) -> Result<Certificate, String>
  ```
  The error type is `String`, not `Violation`, and two required arguments (`specs`, `strength`) are omitted from the architecture manifest.
- **Fix:** Update the manifest signature to match the code exactly.

### DOC-05 — `README.md:105` — **high**
- **Current:** `let cert: Certificate = certify(&backend).expect("Fix: certificate failed");`
- **Reality:** `certify` requires three arguments (`backend`, `specs`, `strength`) and returns `Result<Certificate, String>`. The README example is missing two arguments and misstates the error type.
- **Fix:** Update the example to pass `&specs` and `CertificateStrength::Standard`, and handle `Result<Certificate, String>`.

### DOC-06 — `docs/targets.md:25-77` — **high**
- **Current:** Documents a `vyre-ir` crate with `inventory` and `explicit_registration` features, a feature-flag matrix, and build-time `compile_error!` guards. Describes Tier-2 registration via `vyre::register!` macros.
- **Reality:** No `vyre-ir/Cargo.toml` exists. The crate is `vyre-core` (published as `vyre`). Neither `inventory` nor `explicit_registration` features exist in `vyre-core/Cargo.toml`. There is no `vyre-ir/build.rs` emitting `compile_error!`.
- **Fix:** Rewrite `docs/targets.md` to document the actual `vyre` crate and its real feature set (`default`, `wgpu_subgroups`, `test-helpers`).

### DOC-07 — `STILL_UNFIXED.md:70` — **medium**
- **Current:** "`wgpu_subgroups` and `test-helpers` features removed from `core/Cargo.toml`"
- **Reality:** Both features are still present in `vyre-core/Cargo.toml:84-85`:
  ```toml
  wgpu_subgroups = []
  test-helpers = []
  ```
- **Fix:** Correct the claim to state the features still exist, or remove them from `Cargo.toml` if they are truly dead.

### DOC-08 — `vyre-core/tests/consumption_mode.rs:41,50,57` — **medium**
- **Current:** Tests are gated behind `#[cfg(feature = "hash-only")]`, `#[cfg(feature = "decode-only")]`, and `#[cfg(feature = "primitive-only")]`.
- **Reality:** These features were removed from `vyre-core/Cargo.toml` per `docs/release/v0.4.0.md:53` ("`parity-oracle`, `decode-only`, `hash-only`, `primitive-only` feature flags. One feature set, one product."). The tests will never compile or run.
- **Fix:** Delete the dead `#[cfg]` gates from `tests/consumption_mode.rs`.

### DOC-09 — `vyre-core/fuzz/gpu/README.md:4` — **medium**
- **Current:** "it depends on `vyre` with the `gpu` feature"
- **Reality:** `vyre-core/Cargo.toml` defines no `gpu` feature. The fuzz crate’s `Cargo.toml` depends on plain `vyre = { path = "../.." }` with no feature flags.
- **Fix:** Remove the false feature claim.

### DOC-10 — `vyre-core/INTERNAL_SPEC.md:35` — **medium**
- **Current:** "warpscan uses vyre directly for GPU condition evaluation (`gpu-conditions` feature)"
- **Reality:** No `gpu-conditions` feature exists in any `Cargo.toml` in the workspace.
- **Fix:** Delete the feature reference or rename it to the actual mechanism used.

### DOC-11 — `vyre-core/docs/testing/running/local-workflow.md:97` — **medium**
- **Current:** `cargo test -p vyre --features oom-injection tests/adversarial/oom`
- **Reality:** The `oom-injection` feature exists only in `vyre-conform/Cargo.toml:73` and `vyre-conform-enforce/Cargo.toml:48`, not in `vyre-core` (published as `vyre`). Running this command produces a Cargo error.
- **Fix:** Change the command to `cargo test -p vyre-conform --features oom-injection tests/adversarial/oom`.

### DOC-12 — `vyre-core/docs/testing/categories/adversarial.md:147` — **medium**
- **Current:** "The allocator is behind `#[cfg(feature = "oom-injection")]`"
- **Reality:** The `oom-injection` feature is not defined in `vyre-core/Cargo.toml`. The doc is in the `vyre-core` book but implies the feature is available on `vyre`.
- **Fix:** Clarify that the feature lives in `vyre-conform`, not `vyre-core`.

### DOC-13 — `vyre-core/docs/testing/categories/support.md:129` — **medium**
- **Current:** "typically wgpu if the `gpu` feature is enabled or the reference interpreter otherwise"
- **Reality:** `vyre-core` has no `gpu` feature. The `gpu` feature is defined in `vyre-conform` and `vyre-conform-runner`.
- **Fix:** Remove the `gpu` feature reference from the `vyre-core` test book.

### DOC-14 — `vyre-core/docs/testing/worked-example/02-first-test.md:162` — **medium**
- **Current:** "typically wgpu when the `gpu` feature is enabled, or the reference interpreter otherwise"
- **Reality:** Same as DOC-13 — `vyre-core` has no `gpu` feature.
- **Fix:** Remove the `gpu` feature reference.

### DOC-15 — `ARCHITECTURE.md:324` — **medium**
- **Current:** "Copy `vyre-core/src/ops/template_op.rs` to `vyre-core/src/ops/{category}/{name}.rs`"
- **Reality:** `vyre-core/src/ops/template_op.rs` does not exist.
- **Fix:** Replace with the actual scaffolding path or generator command (e.g., `cargo run -p vyre --bin vyre_new_op`).

### DOC-16 — `vyre-core/docs/parallel-contribution.md:48-64` — **medium**
- **Current:** Shows a `lib.rs` snippet containing `pub mod bytecode; pub mod conform; pub mod engine; pub mod runtime;` and other modules.
- **Reality:** The actual `vyre-core/src/lib.rs` has a completely different structure (lint preamble only, no hand-edited `pub mod` lines; modules are auto-discovered via `automod`). The snippet describes a fictional layout.
- **Fix:** Replace the snippet with the actual current `lib.rs` content.

### DOC-17 — `vyre-core/docs/contributing.md:56` — **medium**
- **Current:** `cp core/src/ops/template_op.rs core/src/ops/primitive/bitwise/my_op.rs`
- **Reality:** `core/src/ops/template_op.rs` does not exist.
- **Fix:** Update to the correct template path or generator command.

### DOC-18 — `docs/release/v0.4.0.md:15` — **medium**
- **Current:** "`vyre-std` | 0.1.0 | DFA, Aho-Corasick, regex → NFA → DFA composites"
- **Reality:** `vyre-std/Cargo.toml:11` states "Migration marker crate: vyre-std dissolved into canonical vyre-primitives", and `vyre-std/README.md` confirms it no longer exports anything.
- **Fix:** Describe `vyre-std` accurately as a dissolved migration marker or redirect users to `vyre-primitives`.

### DOC-19 — `vyre-core/src/ops/registry/registry.rs:73` — **medium**
- **Current:** `registry()` has no `# Panics` section.
- **Reality:** The function calls `.expect("runtime OpSpec registry lock poisoned")` at line 85. It will panic if the static `RwLock` is poisoned. `register_op_spec` (line 65) documents the identical panic condition, but `registry()` does not.
- **Fix:** Add a `# Panics` section documenting lock-poison behavior.

### DOC-20 — `vyre-core/src/optimizer/scheduler.rs:80` — **medium**
- **Current:** `schedule_passes()` documents only `# Errors` (`PassSchedulingError` variants).
- **Reality:** The function contains two internal-invariant `.expect()` calls at lines 125 and 132 that can panic. While these represent algorithmic invariants, the public API still has a panic path that is not disclosed.
- **Fix:** Add a `# Panics` section disclosing the internal-invariant panic paths.

### DOC-21 — `vyre-wgpu/src/engine/multi_gpu.rs:44` — **low**
- **Current:** `partition_work_stealing()` documents only `# Errors`.
- **Reality:** After `validate_inputs(devices, items)?`, line 70 calls `.expect("Fix: validated non-empty device list before partitioning.")` on `partitions.iter_mut().min_by_key(...)`. The public function body contains a panic path that is not documented.
- **Fix:** Add a `# Panics` section documenting the panic path.

### DOC-22 — `xtask/README.md:3,23` — **low**
- **Current:** "`xtask/` lives **inside** `vyre-conform/` because `vyre-conform` is already a Cargo workspace root."
- **Reality:** `xtask/` is at the workspace root (`./xtask/`), not inside `vyre-conform/`.
- **Fix:** Correct description to "workspace root".

### DOC-23 — `vyre-wgpu/README.md:29` — **low**
- **Current:** "the crate lowers `Program` to WGSL via `vyre::lower::wgsl::emit`"
- **Reality:** The public lowering entry point is `vyre::lower::wgsl::lower`, not `emit`. There is no public `emit` function in that module.
- **Fix:** Change `vyre::lower::wgsl::emit` to `vyre::lower::wgsl::lower`.

### DOC-24 — `vyre-spec/src/category.rs:59` — **low**
- **Current:** `#[non_exhaustive] pub enum Category` doc comment describes semantics but never warns callers that new variants may be added in the future.
- **Reality:** Because the enum is `#[non_exhaustive]`, downstream `match` arms must include a wildcard pattern to remain forward-compatible. The doc comment does not mention this obligation.
- **Fix:** Add a doc line: "This enum is `#[non_exhaustive]`; callers must include a wildcard pattern to remain forward-compatible."

### DOC-25 — `vyre-spec/src/data_type.rs:15` — **low**
- **Current:** `#[non_exhaustive] pub enum DataType` doc comment explains integer-first design but omits the forward-compatibility requirement.
- **Reality:** Same as DOC-24 — callers must handle future variants with a wildcard arm, but the docs do not say so.
- **Fix:** Add a doc line noting `#[non_exhaustive]` and the wildcard-pattern requirement.

---

## Summary

| Category | Count |
|----------|-------|
| Doc example / import lies (won't compile) | 5 |
| Feature flag that doesn't exist | 6 |
| Function signature / return type mismatch | 3 |
| Missing `# Panics` disclosure | 3 |
| Missing `#[non_exhaustive]` forward-compat warning | 2 |
| Outdated crate / file references | 4 |
| False claim in audit doc | 1 |
| **Total findings** | **25** |

**Severity breakdown:** High 6 | Medium 13 | Low 6

---

*End of audit. No code was modified.*
