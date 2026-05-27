# AUDIT D — DUPLICATION / CONFLICT OF INTEREST

**Date:** 2026-04-18  
**Scope:** Read-only. Zero tolerance for parallel implementations that drift.  
**Crates audited:** vyre-core, vyre-conform, vyre-build-scan, vyre-spec, vyre-std, vyre-macros, vyre-reference, vyre-wgpu, vyre-sigstore, xtask, demos/rust_lexer_gpu, demos/rust_parser_gpu, examples/hello_vyre  
**Total findings:** 33 — Critical: 5 | High: 16 | Medium: 9 | Low: 3

---

## SUMMARY TABLE

| ID     | Domain                       | Severity | Status   |
|--------|------------------------------|----------|----------|
| DUP-01 | SHA-256 implementation ×3    | CRITICAL | OPEN     |
| DUP-02 | `sha256_hex` helper ×6       | CRITICAL | OPEN     |
| DUP-03 | `verify` (signing) ×2 crates | HIGH     | OPEN     |
| DUP-04 | `verify_laws` ×2 in checker  | HIGH     | OPEN     |
| DUP-05 | `verify_laws_witnessed` ×3   | HIGH     | OPEN     |
| DUP-06 | `compile` fn ×4 sites        | HIGH     | OPEN     |
| DUP-07 | Disk cache dir logic ×2      | HIGH     | OPEN     |
| DUP-08 | Disk cache format ×2         | MEDIUM   | OPEN     |
| DUP-09 | `Token` enum ×2 in vyre-core | MEDIUM   | OPEN     |
| DUP-10 | `validate` free-fn scope     | MEDIUM   | OPEN     |
| DUP-11 | `Buffer` struct ×2           | MEDIUM   | OPEN     |
| DUP-12 | `toml` dep not workspace ×2  | MEDIUM   | OPEN     |
| DUP-13 | `walkdir` dep not workspace  | MEDIUM   | OPEN     |
| DUP-14 | `vyre-build-scan` not ws ×4  | MEDIUM   | OPEN     |
| DUP-15 | `hello_vyre` not workspace   | LOW      | OPEN     |
| DUP-16 | `xtask` not workspace        | LOW      | OPEN     |
| DUP-17 | Dev-dep cycle (core↔wgpu)    | HIGH     | OPEN     |
| DUP-18 | Dev-dep cycle (core↔ref)     | HIGH     | OPEN     |
| DUP-19 | Dev-dep cycle (std↔conform)  | HIGH     | OPEN     |
| DUP-20 | `pub use vyre::VyreBackend`  | MEDIUM   | OPEN     |
| DUP-21 | `pub use crate::generate::*` | MEDIUM   | OPEN     |
| DUP-22 | Registry proliferation ×6    | HIGH     | OPEN     |
| DUP-23 | Parallel sign/verify impls   | HIGH     | OPEN     |
| DUP-24 | PipelineCache vs TieredCache | MEDIUM   | OPEN     |
| DUP-25 | `OpKind` enum ×2             | LOW      | OPEN     |
| DUP-26 | `clap` / `fs2` not workspace | LOW      | OPEN     |
| DUP-27 | Two `op_registry` concepts   | HIGH     | OPEN     |
| DUP-28 | `OpSignature` ×2 crates      | CRITICAL | OPEN     |
| DUP-29 | `ConformSpec` name collision ×2   | HIGH     | OPEN     |
| DUP-30 | `Value` enum ×2 crates       | HIGH     | OPEN     |
| DUP-31 | `DefendantCatalog` ×2        | CRITICAL | OPEN     |
| DUP-32 | `EnforceGate` bypassed ×4    | HIGH     | OPEN     |
| DUP-33 | Defender codegen ×2          | HIGH     | OPEN     |

---

## FINDINGS

---

### DUP-01 — SHA-256 hand-rolled implementation ×3

**CHECK:** 1 (same function), 7  
**SEVERITY:** CRITICAL  
**LOCATIONS:**
- `vyre-conform/src/verify/golden/util/sha256.rs:5` — `pub(crate) fn sha256(data: &[u8]) -> [u8; 32]`
- `vyre-conform/src/verify/regression/hex/sha256.rs:3` — `pub(super) fn sha256(input: &[u8]) -> [u8; 32]`
- `xtask/src/hash/sha256.rs:11` — `pub(crate) fn sha256(bytes: &[u8]) -> [u8; 32]`

**DESCRIPTION:** Three independent, hand-rolled SHA-256 implementations exist in the workspace. The `golden` and `regression` copies differ in constant layout style (golden uses `0x428a_2f98` formatting, regression uses `0x428a2f98`), padding computation strategy (golden: `while (msg.len() % 64) != 56`, regression: `while (msg.len() + 8) % 64 != 0`), and variable naming. A fourth site (`vyre-conform/src/enforce/enforcers/reference_trust/external.rs:169`) declares `fn sha256(input: &[u8]) -> Vec<u8>` but delegates to `blake3::hash`, creating a naming lie: it is not SHA-256.

The `sha2` crate is already in `workspace.dependencies`. Using it universally would eliminate all three hand-rolled bodies and the risk of independent bugs in each.

**IMPACT:** Divergent SHA-256 outputs would corrupt golden fixtures and regression corpus keys. Finding from internet-scale lens: golden hash mismatches cause silent test invalidation — a failed backend would appear conformant if the golden key is computed by a different implementation than the checker.

**UNIFY PLAN:**
1. Delete `vyre-conform/src/verify/golden/util/sha256.rs` and `regression/hex/sha256.rs` and `xtask/src/hash/sha256.rs`.
2. In every callsite, replace with `sha2::Sha256::digest(data).into()`.
3. Add `sha2.workspace = true` to each crate that needs it (already workspace-pinned).
4. Rename `external.rs::sha256` to `blake3_hash` to eliminate the false name.

---

### DUP-02 — `sha256_hex` helper ×6 sites

**CHECK:** 1 (same function)  
**SEVERITY:** CRITICAL  
**LOCATIONS:**
- `vyre-conform/src/verify/golden/util.rs:33` — delegates to hand-rolled sha256 + `vyre_reference::hash::hex::bytes_to_hex`
- `vyre-conform/src/verify/regression/hex/sha256_hex.rs:3` — delegates to a different hand-rolled sha256
- `vyre-conform/src/verify/regression/hex.rs:5` — re-delegates
- `vyre-conform/src/enforce/enforcers/gate_7_coverage.rs:404` — delegates to `sha2::Sha256::digest` + `vyre_reference::hash::hex::bytes_to_hex`
- `vyre-build-scan/src/conform.rs:66` — inline implementation
- `xtask/src/hash/sha256_hex.rs:12` — delegates to xtask's own sha256
- `vyre-conform/tests/adversarial/gate7_omitted_rows.rs:7` — local fn in test

**DESCRIPTION:** Six independent implementations of SHA-256-to-hex conversion. Three of them disagree on which SHA-256 to call (hand-rolled golden, hand-rolled regression, sha2 crate). The gate_7 and golden paths produce identical hex if their SHA-256 produces the same bytes, but this is accidental.

**IMPACT:** If any hand-rolled SHA-256 has a bug on a specific input class, golden hex keys diverge from gate_7 hex keys, causing silent test-cache corruption.

**UNIFY PLAN:** Centralise in one place — `vyre-build-scan/src/hash.rs` is a reasonable home since build-scan is a zero-dependency utility crate. Expose `pub fn sha256_hex(bytes: &[u8]) -> String` backed by `sha2::Sha256`. Add `sha2.workspace = true` to `vyre-build-scan`. All six sites become one import.

---

### DUP-03 — Parallel `verify` implementations for ed25519 signatures

**CHECK:** 1 (same function), 6  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-sigstore/src/lib.rs:53` — `pub fn verify(cert_bytes: &[u8], sig: &Signature, pubkey: &VerifyingKey) -> Result<(), VerifyError>`
- `vyre-conform/src/runner/certify/signing.rs:82` — `pub fn verify(cert: &Certificate, signature: &Signature, verifying_key: &VerifyingKey) -> Result<(), SignatureError>`

**DESCRIPTION:** `vyre-sigstore` was explicitly created to be the canonical ed25519 sign/verify primitive so that "downstream auditors can verify certificates without pulling the maintainer harness." Despite this, `vyre-conform/certify/signing.rs` imports `ed25519_dalek::{Signer, Verifier}` directly, re-implements `sign` and `verify` from scratch, and defines its own `SignatureError` type — all without referencing `vyre-sigstore` at all. This defeats the purpose of the crate.

Specifically: `signing.rs` serialises the `Certificate` to canonical bytes and then calls `verifying_key.verify(&bytes, signature)` — exactly what `vyre-sigstore::verify` does, except vyre-sigstore operates on pre-serialised bytes while signing.rs wraps serialisation. The two differ only in that signing.rs inlines `canonical_bytes` rather than calling through `vyre-sigstore`.

**IMPACT:** Two independent implementations of signature verification will eventually diverge. A bug fixed in vyre-sigstore silently stays unfixed in conform/certify/signing. An auditor who reads vyre-sigstore to understand the verification protocol is not reading the code path that actually runs in certification.

**UNIFY PLAN:**
1. Add `vyre-sigstore` to `vyre-conform`'s `[dependencies]`.
2. Rewrite `signing.rs::verify` to: serialise cert → call `vyre_sigstore::verify(bytes, sig, pubkey)`.
3. Rewrite `signing.rs::sign` to: serialise cert → call `vyre_sigstore::sign(bytes, key)`.
4. Delete `SignatureError` in signing.rs; re-export `vyre_sigstore::VerifyError`.
5. Remove the direct `ed25519_dalek` dep from vyre-conform (it currently pulls it for signing.rs).

---

### DUP-04 — `verify_laws` duplicated in `checker.rs` and `checker/verify_laws.rs`

**CHECK:** 1 (same function)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-conform/src/proof/algebra/checker.rs:103` — `pub fn verify_laws(...) -> Vec<LawResult>`
- `vyre-conform/src/proof/algebra/checker/verify_laws.rs:19` — `pub fn verify_laws(...) -> Vec<LawResult>`

**DESCRIPTION:** Two `verify_laws` functions with identical signatures in the same module hierarchy. `checker.rs` is the parent module; `checker/verify_laws.rs` is a child. Cargo/Rust allow both to exist and both are `pub`. External callers using `use crate::proof::algebra::checker::verify_laws` may resolve either depending on import path. The submodule version appears to be the canonical intended implementation (it references `verify_one_law_witnessed` from the parent via `super::`), while the checker.rs version is a leftover from before the refactor into the submodule directory.

**IMPACT:** Callers inside vyre-conform may call the wrong function. The two bodies may have diverged silently — the checker.rs version (lines 103-110) uses `laws.iter().map(...)` while the checker/verify_laws.rs version uses `laws.par_iter().map(...)` with rayon. Different parallelism = different wall-clock timing, but same results if correct. However, under `loom`, the checker/verify_laws.rs version is `#[cfg(not(loom))]`, leaving the loom path to the checker.rs fallback — a subtle correctness trap.

**UNIFY PLAN:** Delete `pub fn verify_laws` from `checker.rs`. Re-export from the submodule: `pub use checker::verify_laws::verify_laws;` or simply remove the shadowing definition. Confirm all callers use the submodule path.

---

### DUP-05 — `verify_laws_witnessed` duplicated ×3

**CHECK:** 1 (same function)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-conform/src/proof/algebra/checker.rs:180` — `#[cfg(not(loom))] pub fn verify_laws_witnessed(...)`
- `vyre-conform/src/proof/algebra/checker/verify_laws_witnessed.rs:12` — `#[cfg(loom)] pub fn verify_laws_witnessed(...)`
- `vyre-conform/src/proof/algebra/checker/verify_laws_witnessed_1.rs:13` — `#[cfg(not(loom))] pub fn verify_laws_witnessed(...)`

**DESCRIPTION:** Three versions of the same `verify_laws_witnessed` function exist simultaneously. `verify_laws_witnessed_1.rs` is `#[cfg(not(loom))]` and uses `par_iter()`. `checker.rs:180` is also `#[cfg(not(loom))]` and uses `par_iter()` with an identical body. The loom variant is correctly isolated in `verify_laws_witnessed.rs`. This likely arose from incremental extraction of checker.rs into a submodule directory without deleting the original.

**IMPACT:** Any fix to the non-loom path must be applied twice to take effect. A future maintainer debugging a witnessed test failure will read `checker.rs` and apply the fix, not knowing the duplicate in `verify_laws_witnessed_1.rs` supersedes it at the module level.

**UNIFY PLAN:** Delete `pub fn verify_laws_witnessed` from `checker.rs:180` entirely. Determine which of `verify_laws_witnessed.rs` vs `verify_laws_witnessed_1.rs` is the authoritative split — given their `cfg` attributes are complementary (loom vs not-loom), they together form one complete implementation and should be the canonical location. Remove `verify_laws_witnessed_1.rs` and inline its `#[cfg(not(loom))]` body into `verify_laws_witnessed.rs` via `#[cfg(not(loom))]` / `#[cfg(loom)]` blocks.

---

### DUP-06 — `compile` function ×4 sites with diverging semantics

**CHECK:** 1 (same function)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-core/src/pipeline.rs:49` — `pub fn compile(backend, program, config) -> Arc<dyn CompiledPipeline>` — dispatches through `VyreBackend` trait
- `vyre-wgpu/src/pipeline.rs:125` — `WgpuPipeline::compile(program) -> Result<Self>` — wgpu-specific; calls `compile_with_config`
- `vyre-wgpu/src/engine/dfa.rs:93` — `GpuDfa::compile(device, transitions, ...) -> Result<Self>` — DFA-specific GPU pipeline
- `vyre-core/tests/support/workgroup_gpu.rs:12` — `pub fn compile(device, label, source, entry_point) -> wgpu::ComputePipeline` — raw wgpu module compile used in integration tests

**DESCRIPTION:** Four functions named `compile` that each mean something different: trait-dispatched IR compilation, wgpu IR-to-WGSL pipeline construction, DFA-to-GPU-pipeline, and raw shader module compilation. The fourth (in `tests/support`) operates at the `wgpu` primitive level and duplicates `vyre-wgpu`'s internal `compile_compute_pipeline` from `vyre-wgpu/src/runtime/shader/compile_compute_pipeline.rs` — same logic, different name, in a test support file. 

**IMPACT:** New contributors searching for "how does compile work" find four definitions pointing in different directions. The test support `compile` bypasses the pipeline cache entirely, meaning integration tests under `vyre-core/tests/` pay full shader recompile cost on every run.

**UNIFY PLAN:** Rename the `tests/support/workgroup_gpu.rs::compile` to `compile_raw_wgsl` and document its difference. Consider exposing a `vyre-wgpu` test helper crate feature (`test-helpers`) so core's integration tests import the cached version. Document the semantic difference between the four in a single "compilation model" doc comment at `vyre-core/src/pipeline.rs`.

---

### DUP-07 — Disk cache directory resolution logic ×2

**CHECK:** 7 (parallel Cache implementations)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-std/src/pattern/cache.rs:109` — `fn cache_dir() -> PathBuf` — resolves `XDG_CACHE_HOME/vyre/dfa/` with `HOME/.cache` fallback and then `.vyre-cache` ultimate fallback
- `vyre-wgpu/src/pipeline_disk_cache.rs:83` — `fn disk_pipeline_cache_dir() -> PathBuf` — resolves `XDG_CACHE_HOME/vyre/pipeline/` with `HOME/.cache` fallback, but **no third fallback**

**DESCRIPTION:** Both functions implement XDG Base Directory Specification resolution for their respective cache subdirectory. The logic is semantically identical for the first two branches. They differ only in: (a) the subdirectory name (`dfa` vs `pipeline`), (b) `vyre-std` has a `.vyre-cache` sandboxed fallback that `vyre-wgpu` omits. This means in environments without `HOME` and without `XDG_CACHE_HOME` (e.g., some CI containers), the DFA cache degrades gracefully while the pipeline cache panics with `unwrap_or_else(|| ".".into())` → writing `.wgsl` files into the current directory.

**IMPACT:** Divergent fallback policies. The `vyre-wgpu` fallback writes cache files into the current working directory on headless CI, polluting the repo. The `vyre-std` fallback is correct but is not shared.

**UNIFY PLAN:** Extract to a workspace-internal utility: `vyre-build-scan` or a new micro-crate `vyre-cache-dir` with `pub fn cache_dir(subdir: &str) -> PathBuf`. Both caches call this single function. The function applies the three-level fallback (`XDG_CACHE_HOME` → `HOME/.cache` → `.vyre-cache`) universally.

---

### DUP-08 — Disk cache file format ×2 (incompatible versions)

**CHECK:** 7 (parallel Cache implementations)  
**SEVERITY:** MEDIUM  
**LOCATIONS:**
- `vyre-std/src/pattern/cache.rs:20` — `const CACHE_VERSION: &str = "vyre-std.dfa.v2"` — uses FNV-1a key, custom 17-byte binary frame
- `vyre-wgpu/src/pipeline_disk_cache.rs:8` — `const DISK_PIPELINE_CACHE_VERSION: u32 = 1` — uses blake3 key, TOML metadata sidecar + bare `.wgsl` file

**DESCRIPTION:** Two disk caches use entirely different file format strategies: the DFA cache uses a custom binary frame (format tag, start, state_count, payload_len, payload) while the pipeline cache uses a separate `.toml` sidecar + a `.wgsl` text file. The version numbering is also incompatible (`&str` tag vs `u32`). Neither format is documented as stable. Neither has a migration path. The DFA cache uses a hand-rolled FNV-1a hasher (a struct `Fnv1a` defined inline) while the pipeline cache uses `blake3`.

**IMPACT:** Cache format evolution requires two independent migration strategies. The hand-rolled `Fnv1a` in vyre-std is a private struct that duplicates what `rustc_hash` or any real FNV crate provides. The blake3 crate is already a workspace dependency.

**UNIFY PLAN:** Define a shared cache abstraction in `vyre-std` or `vyre-build-scan`: `struct DiskCache<K, V>` with pluggable key serialisation (blake3 for both — drop FNV-1a) and a common versioned binary frame format. Both crates adopt this abstraction, reducing the format to one design. Delete the inline `Fnv1a` struct; use `blake3::hash` (already a workspace dep in vyre-std? — add it).

---

### DUP-09 — `Token` enum defined twice in `vyre-core` string_matching

**CHECK:** 2 (same type)  
**SEVERITY:** MEDIUM  
**LOCATIONS:**
- `vyre-core/src/ops/string_matching/nfa_scan/kernel/token.rs:2` — `pub enum Token { Literal(u8), Any, StarLiteral(u8), StarAny }`
- `vyre-core/src/ops/string_matching/wildcard_match/kernel/token.rs:2` — `pub enum Token { Literal(u8), Any, Star }`

**DESCRIPTION:** Two `Token` enums for two string-matching operations. They share 2 of 3/4 variants (`Literal` and `Any`) but differ in how they model star-matching. `nfa_scan` distinguishes `StarLiteral(u8)` (star followed by a specific byte) from `StarAny` (star followed by any), while `wildcard_match` collapses both into `Star`. This may be intentional design (different automaton classes) but results in duplicate parse paths, duplicate tests, and duplicate token names in the same crate.

**IMPACT:** A third string-matching op will likely introduce a third `Token` enum unless a shared abstraction is established. Changes to what "Literal" means in one op may silently not propagate.

**UNIFY PLAN:** Define a base `WildcardToken` in a shared module `vyre-core/src/ops/string_matching/token.rs` with the common variants. Each op extends or wraps it. Alternatively, document explicitly in both files why they intentionally diverge and add a `// DIVERGES FROM: ../nfa_scan/kernel/token.rs — reason: ` comment so a third op author sees the precedent.

---

### DUP-10 — `validate` as a free function vs method across crates

**CHECK:** 1 (same function)  
**SEVERITY:** MEDIUM  
**LOCATIONS:**
- `vyre-core/src/ir/validate/validate.rs:35` — `pub fn validate(program: &Program) -> Vec<ValidationError>` — IR-level free function
- `vyre-conform/src/enforce/enforcers/decomposition/config.rs:57` — `pub fn self.validate() -> Result<(), String>` — method on `DecompositionConfig`
- `vyre-conform/src/spec/types/conform/source.rs:48` — `pub fn validate(self) -> Result<(), &'static str>` — consuming method on `Source`
- `vyre-wgpu/src/engine/decode.rs:363` — `pub fn self.validate() -> Result<(), DecodeError>` — method on `DecodeConfig`
- `vyre-core/src/ops/graph/csr.rs:33` — `pub fn self.validate() -> Result<()>` — method on `CsrGraph`

**DESCRIPTION:** The free function `validate` in `vyre-core::ir::validate` is a crate-level entry point for IR correctness, while all other `validate` calls are struct methods on domain-specific config types. The concern is the free-function form in vyre-core: it is re-exported at the crate root as `pub use ir::{validate, Program}` (lib.rs:176), making `vyre::validate` a public API alongside `vyre::Program`. This conflates "validate a Program" with the concept of validation as a method, making it harder for callers to discover that programs have an in-built validity contract.

**IMPACT:** External crates calling `vyre::validate(program)` instead of a hypothetical `program.validate()` breaks if the IR type ever adds an inherent `validate` method. The free-function form is also harder to chain: `pipeline::compile(backend, &program, &config)` silently accepts an invalid program; callers who forget to call `vyre::validate` before compile get no compile-time guidance.

**UNIFY PLAN:** Add `impl Program { pub fn self.validate() -> Vec<ValidationError> }` delegating to the free function. Deprecate the free-function re-export at the crate root. Update `compile` to optionally call `validate` internally (gated on a `DispatchConfig` field to preserve backwards-compatible performance behaviour).

---

### DUP-11 — `Buffer` struct in two crates

**CHECK:** 2 (same type)  
**SEVERITY:** MEDIUM  
**LOCATIONS:**
- `vyre-reference/src/oob.rs:20` — `pub struct Buffer { bytes: Vec<u8>, element: IrDataType }` — reference interpreter buffer with OOB semantics
- `vyre-core/src/ir/model/program.rs:83` — `pub struct BufferDecl { ... }` — IR declaration of a buffer (binding, name, access, element, count)

**DESCRIPTION:** `Buffer` in vyre-reference represents runtime bytes for one declared buffer. `BufferDecl` in vyre-core is the IR declaration. These are correctly distinct types. However, `vyre-reference::Buffer` is not re-exported from `vyre-reference/src/lib.rs` — it is only used internally by `interp`. This is correct: it is implementation detail. The finding is that the name `Buffer` without qualification is a future collision risk since `vyre-core` also has `BufferPool` (in vyre-wgpu) and the IR model has `BufferDecl`. A future public `Buffer` type in any crate will conflict with existing imports.

**IMPACT:** Low immediate risk but predictable naming collision as the API surface grows (e.g., typed Buffer handles for callers).

**UNIFY PLAN:** Rename `vyre-reference/src/oob.rs::Buffer` to `ReferenceBuffer` to disambiguate. Document in a naming convention: IR declarations are `*Decl`; runtime instances are `*Buffer` or `*Instance`; pool handles are `Pooled*`.

---

### DUP-12 — `toml` dependency not using workspace in `vyre-core`

**CHECK:** 3 (Cargo.toml deduplication)  
**SEVERITY:** MEDIUM  
**LOCATIONS:**
- `vyre-core/Cargo.toml` `[dependencies]` line: `toml = "=0.8.23"` — raw version string, not `toml.workspace = true`
- `vyre-core/Cargo.toml` `[build-dependencies]` line: `toml = "=0.8.23"` — same

**DESCRIPTION:** `toml` is pinned in `workspace.dependencies` at `"=0.8.23"`. Every other crate that uses `toml` (vyre-conform, vyre-wgpu, etc.) correctly writes `toml.workspace = true`. `vyre-core` hardcodes the version twice — once in `[dependencies]` and once in `[build-dependencies]`. If the workspace pin is ever bumped, `vyre-core` will silently use a different version from all other crates, potentially causing binary-incompatible TOML reading.

**IMPACT:** Silent version skew. Two different `toml` versions in one binary = two copies of the same crate = link-time bloat and potential panics if one crate's `Value` type is passed to the other's parser.

**UNIFY PLAN:** Replace both occurrences with `toml.workspace = true`.

---

### DUP-13 — `walkdir` not workspace in `vyre-core` build-dependencies

**CHECK:** 3 (Cargo.toml deduplication)  
**SEVERITY:** MEDIUM  
**LOCATIONS:**
- `vyre-core/Cargo.toml` `[build-dependencies]`: `walkdir = "=2.5.0"` — raw pin, not workspace

**DESCRIPTION:** `walkdir` is in `workspace.dependencies` at `"=2.5.0"`. `vyre-build-scan` and `vyre-conform` both use `walkdir.workspace = true`. `vyre-core/build.rs` pulls `walkdir` via a hardcoded pin. Same version skew risk as DUP-12.

**UNIFY PLAN:** Change to `walkdir.workspace = true`.

---

### DUP-14 — `vyre-build-scan` referenced by path without workspace in 4 places

**CHECK:** 3 (Cargo.toml deduplication)  
**SEVERITY:** MEDIUM  
**LOCATIONS:**
- `vyre-core/Cargo.toml` `[dev-dependencies]`: `vyre-build-scan = { path = "../vyre-build-scan", version = "0.1.0" }`
- `vyre-core/Cargo.toml` `[build-dependencies]`: `vyre-build-scan = { path = "../vyre-build-scan", version = "0.1.0" }`
- `vyre-conform/Cargo.toml` `[build-dependencies]`: `vyre-build-scan = { path = "../vyre-build-scan", version = "0.1.0" }`

**DESCRIPTION:** `vyre-build-scan` is declared in `workspace.dependencies`. Three references to it use raw `path + version` instead of `.workspace = true`, meaning the version number must be kept in sync manually in three places.

**IMPACT:** A bump to `vyre-build-scan`'s version will require editing three separate `Cargo.toml` files instead of one. Easy to miss one, breaking `cargo publish`.

**UNIFY PLAN:** Change all three to `vyre-build-scan.workspace = true` (or `{ workspace = true }` for the path-only form). Verify that `workspace.dependencies` entry already includes the path, which it does.

---

### DUP-15 — `examples/hello_vyre` Cargo.toml not using workspace

**CHECK:** 3 (Cargo.toml deduplication)  
**SEVERITY:** LOW  
**LOCATIONS:**
- `examples/hello_vyre/Cargo.toml` — `edition = "2021"` hardcoded; `vyre = { path = "../../vyre-core" }` and `vyre-spec = { path = "../../vyre-spec" }` not workspace

**DESCRIPTION:** The hello_vyre example hardcodes `edition = "2021"` instead of `edition.workspace = true`, and references both internal crates by raw paths instead of `workspace = true`. If the workspace edition changes, the example silently stays on 2021.

**UNIFY PLAN:** Replace `edition = "2021"` with `edition.workspace = true`. Replace both dependency entries with `vyre.workspace = true` and `vyre-spec.workspace = true`.

---

### DUP-16 — `xtask` Cargo.toml duplicates workspace metadata

**CHECK:** 3 (Cargo.toml deduplication)  
**SEVERITY:** LOW  
**LOCATIONS:**
- `xtask/Cargo.toml` — `edition = "2021"`, `authors = ["Santh Project <contact@santh.dev>"]`, `license = "MIT OR Apache-2.0"` all hardcoded

**DESCRIPTION:** `xtask` duplicates workspace-level package metadata that is already declared in `workspace.package`. This is the same data that all other member crates obtain via `*.workspace = true`.

**UNIFY PLAN:** Add `edition.workspace = true`, `authors.workspace = true`, `license.workspace = true` to `xtask/Cargo.toml`. Remove the inline values.

---

### DUP-17 — Dev-dependency cycle: vyre-core ↔ vyre-wgpu

**CHECK:** 4 (circular dev-dependencies)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-core/Cargo.toml` `[dev-dependencies]`: `vyre-wgpu = { path = "../vyre-wgpu" }`
- `vyre-wgpu/Cargo.toml` `[dependencies]`: `vyre = { workspace = true }` (which is vyre-core)

**DESCRIPTION:** `vyre-core`'s integration tests and benchmarks depend on `vyre-wgpu` (for GPU parity tests), while `vyre-wgpu` has `vyre-core` as a normal compile-time dependency. Cargo permits this pattern for dev-dependencies, but it means:
1. `vyre-core` cannot be tested in isolation without building `vyre-wgpu`.
2. Any API change in `vyre-core` that breaks `vyre-wgpu` will cause `vyre-core`'s own test suite to fail during the transition, preventing incremental development.
3. The link produces a two-way build dependency that slows incremental compilation: changing `vyre-core` triggers a full rebuild of `vyre-wgpu` before tests can run.

**IMPACT:** Violates the layered architecture invariant (core must not know about backends). Even as a dev-dep, it tightly couples development workflow.

**UNIFY PLAN:** Extract GPU parity tests from `vyre-core/tests/` into a separate workspace member (e.g., `tests/gpu-parity/`). This member can freely depend on both `vyre-core` and `vyre-wgpu`. The `vyre-core` dev-dep on `vyre-wgpu` is removed entirely.

---

### DUP-18 — Dev-dependency cycle: vyre-core ↔ vyre-reference

**CHECK:** 4 (circular dev-dependencies)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-core/Cargo.toml` `[dev-dependencies]`: `vyre-reference = { path = "../vyre-reference" }`
- `vyre-reference/Cargo.toml` `[dependencies]`: `vyre = { workspace = true }` (vyre-core)

**DESCRIPTION:** Same pattern as DUP-17. `vyre-core`'s test harness uses `vyre-reference` for oracle checks, while `vyre-reference` depends on `vyre-core` for IR types. `vyre-reference` is also in `workspace.dependencies`, so this is a recognised workspace member — but the dev-dep direction makes `vyre-core` tests unbuildable without `vyre-reference`.

**UNIFY PLAN:** Move tests that use `vyre-reference` into the same separate `tests/gpu-parity/` member crate suggested in DUP-17. Alternatively, for pure-Rust reference tests, create a `tests/reference-parity/` crate. Either way, `vyre-core/Cargo.toml` loses both GPU and reference dev-dependencies.

---

### DUP-19 — Dev-dependency cycle: vyre-std ↔ vyre-conform

**CHECK:** 4 (circular dev-dependencies)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-std/Cargo.toml` `[dev-dependencies]`: `vyre-conform = { path = "../vyre-conform" }`
- `vyre-conform/Cargo.toml` `[dependencies]`: `vyre = { workspace = true }` (vyre-core), `vyre-reference.workspace = true`
- `vyre-conform/Cargo.toml` does NOT list `vyre-std` as a dependency

**DESCRIPTION:** `vyre-std` uses `vyre-conform` to run conformance tests against its DFA assembly pipeline. `vyre-conform` depends on `vyre-core` and `vyre-reference`, not on `vyre-std` directly. The cycle is indirect: `vyre-std → (dev) → vyre-conform → vyre-core → (builds all of vyre-core)`, and `vyre-std` itself depends on `vyre-core`. So building `vyre-std` for tests requires building `vyre-conform`, which requires building `vyre-core`, which `vyre-std` already depends on. This chain is linear, not circular, but it means `cargo test -p vyre-std` triggers a full `vyre-conform` build.

**IMPACT:** Developer cycle time: changing one line in `vyre-std/src/arith.rs` triggers recompilation of the entire conform harness to run `vyre-std`'s tests. This is a LAW 4 violation: the engine is not swappable in isolation.

**UNIFY PLAN:** Replace the `vyre-conform` dev-dep in `vyre-std` with a targeted property-test harness (`proptest`) that directly exercises `vyre-std`'s DFA pipeline correctness without pulling in the full conformance infrastructure. If parity against the backend is needed, use the same `tests/gpu-parity/` member from DUP-17.

---

### DUP-20 — API leakage: `pub use vyre::VyreBackend` at vyre-conform crate root

**CHECK:** 5 (re-export hygiene)  
**SEVERITY:** MEDIUM  
**LOCATIONS:**
- `vyre-conform/src/lib.rs:34` — `pub use vyre::VyreBackend;`

**DESCRIPTION:** `VyreBackend` is a type defined in `vyre-core` (the `vyre` crate). Re-exporting it from `vyre-conform`'s root creates a second public path: `vyre_conform::VyreBackend`. Any external crate that writes `use vyre_conform::VyreBackend` imports a type that semantically belongs to `vyre`, not `vyre-conform`. This is a direct API leakage: if `VyreBackend` is ever moved or renamed in `vyre-core`, both the `vyre::VyreBackend` and `vyre_conform::VyreBackend` paths break, doubling the surface that needs updating.

The module-gated re-export `pub mod backend { pub use vyre::{BackendError, DispatchConfig, VyreBackend}; }` (lib.rs:39) is acceptable — it documents intent. The root-level re-export is the problem.

**IMPACT:** Published docs for `vyre-conform` will show `VyreBackend` as a top-level item, misleading users into thinking `vyre-conform` is where to get the backend trait.

**UNIFY PLAN:** Remove the root-level `pub use vyre::VyreBackend;` from `lib.rs:34`. The `pub mod backend` at line 37-40 already serves this purpose for callers who need to import backend types alongside conform types.

---

### DUP-21 — Glob re-export `pub use crate::generate::*` and `pub use crate::proof::algebra::*`

**CHECK:** 5 (re-export hygiene)  
**SEVERITY:** MEDIUM  
**LOCATIONS:**
- `vyre-conform/src/lib.rs:109` — `pub use crate::generate::*` (inside `pub mod generator`)
- `vyre-conform/src/lib.rs:104` — `pub use crate::proof::algebra::*` (inside `pub mod algebra`)

**DESCRIPTION:** Glob re-exports expose every `pub` item from `generate` and `proof::algebra` at the `vyre_conform::generator` and `vyre_conform::algebra` namespaces. Any new `pub` item added to either module is silently promoted to the crate's public API without the author choosing to expose it. Since `generate` and `proof::algebra` are marked `pub(crate)` in the module declarations (lib.rs:151: `pub(crate) mod generate`, `pub(crate) mod proof`), the glob re-exports act as a bypass of the access control, effectively making every internal type publicly accessible through the re-export path.

**IMPACT:** API instability: internal refactors in `generate` or `proof::algebra` that rename or remove types will be breaking changes for external consumers of `vyre_conform::generator::*`.

**UNIFY PLAN:** Replace each glob with an explicit named re-export list. Review what is actually required by consumers (the `generator` compat module comment says "Test generation compatibility exports") and expose only those types. Every type that should be hidden stays `pub(crate)`.

---

### DUP-22 — Registry proliferation: six distinct registry concepts

**CHECK:** 8 (parallel registries)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-core/src/backend/registry.rs` — `BackendRegistration` via `inventory::collect!` — link-time backend discovery
- `vyre-core/src/optimizer.rs:72` — `PassRegistration` via `inventory::collect!` — link-time optimizer pass discovery
- `vyre-core/src/ops/registry/registry.rs` — `GENERATED_REGISTRY` (build-time) + `RUNTIME_REGISTRY` (`OnceLock<RwLock<Vec<...>>>`) — ConformSpec lookup
- `vyre-conform/src/spec/op_registry.rs` — `ALL_SPECS_CACHE` (`LazyLock<Vec<ConformSpec>>`) + `SPECS_BY_ID` (`LazyLock<FxHashMap<...>>`) — conform-side ConformSpec registry
- `vyre-conform/src/runner/loader/toml/registry.rs` — `TomlRegistry` — in-memory TOML rules (ops, witnesses, defendants, laws, independence)
- `vyre-build-scan/src/config.rs` — `Registry<'a>` + `RustSpecRegistry<'a>` — build-time scanner configuration structs

**DESCRIPTION:** Six distinct "registry" abstractions exist across four crates. Three of them specifically register `ConformSpec` or related op data:
- `vyre-core::ops::registry` — compile-time generated `&[&ConformSpec]` with runtime extension
- `vyre-conform::spec::op_registry` — conform-side `Vec<ConformSpec>` cloned from core + categorised
- `vyre-conform::runner::loader::toml::registry::TomlRegistry` — rules loaded from TOML

The conform-side registry (`ALL_SPECS_CACHE`) is populated by calling per-category `specs()` functions, which themselves query `vyre-core::ops::registry`. This is a two-level registry for the same data: core holds `&ConformSpec` references; conform clones them into owned `Vec<ConformSpec>`. Any time a new op is added to core, the conform registry automatically picks it up only if its category's `specs()` function queries core. If someone adds an op to a new category without wiring `specs()` into `vyre-conform/src/spec/op_registry.rs`, it silently disappears from conform.

**IMPACT:** Silent coverage gap: a new operation present in `vyre-core`'s generated registry is NOT automatically in conform's registry unless the category's `specs()` is listed in `ALL_SPECS_CACHE`'s LazyLock constructor.

**UNIFY PLAN:**
1. Consolidate `vyre-core` and `vyre-conform` op registries: conform's `op_registry` should be a thin adapter over core's `registry()`, not a separate `LazyLock`.
2. Replace the manual `specs.extend(crate::spec::X::specs())` list with a macro or `automod`-generated list that scans all category modules.
3. Document the three remaining registries (Backend, Pass, TomlRules) as intentionally separate with a clear ownership model.

---

### DUP-23 — Parallel sign/verify signing modules: signing.rs vs vyre-sigstore

**CHECK:** 6 (parallel Validator/Verifier/Checker implementations)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-conform/src/runner/certify/signing.rs` — defines `SignatureError`, `canonical_bytes`, `sign`, `verify` all from scratch using `ed25519_dalek` directly
- `vyre-sigstore/src/lib.rs` — defines `VerifyError`, `sign`, `verify`, `canonical_digest` for the same purpose

(See also DUP-03 for the detailed analysis; this finding focuses on the architectural duplication, DUP-03 on the code-level duplication.)

**DESCRIPTION:** The existence of `vyre-sigstore` was motivated precisely to allow downstream auditors to verify conformance certificates without depending on `vyre-conform`. This goal is undermined because `vyre-conform` does not use `vyre-sigstore` for its own signing. An auditor who reads vyre-sigstore to understand the certificate verification protocol is studying dead code paths from `vyre-conform`'s perspective.

**UNIFY PLAN:** (Same as DUP-03.) `vyre-conform/certify/signing.rs` must delegate to `vyre-sigstore`. The `SignatureError` type in signing.rs wraps or re-exports `vyre-sigstore::VerifyError`. No raw `ed25519_dalek` calls outside `vyre-sigstore`.

---

### DUP-24 — Two LRU-backed caches in vyre-wgpu with duplicated eviction logic

**CHECK:** 7 (parallel Cache implementations)  
**SEVERITY:** MEDIUM  
**LOCATIONS:**
- `vyre-wgpu/src/runtime/cache/tiered_cache.rs` — `TieredCache<P>` with `LruPolicy` using `AccessTracker` from `lru.rs`
- `vyre-wgpu/src/runtime/shader/pipeline_cache.rs` — `PipelineCache` with its own `IntrusiveLru<u64, ()>` from the same `lru.rs`

**DESCRIPTION:** Both caches use `IntrusiveLru` from `vyre-wgpu/src/runtime/cache/lru.rs`, so they share the LRU primitive. However, `PipelineCache` implements its own eviction logic (token-based two-way map: `tokens_by_key: FxHashMap<String, u64>` and `keys_by_token: FxHashMap<u64, String>`) that duplicates the tracking `TieredCache`'s `AccessTracker` provides. The `TieredCache` has a `TierPolicy` abstraction for promotion/eviction; `PipelineCache` bypasses this and implements eviction inline.

**IMPACT:** A bug fix in eviction logic for one cache (e.g., off-by-one in capacity check) will not propagate to the other. The two-way token map in `PipelineCache` is 40+ lines of non-trivial bookkeeping that `TieredCache` could provide via its `index: FxHashMap<u64, usize>`.

**UNIFY PLAN:** `PipelineCache` should be reimplemented as a `TieredCache<LruPolicy>` with a single tier sized to `MAX_PIPELINE_CACHE_ENTRIES`. The value type becomes `wgpu::ComputePipeline` (wrapped via a newtype to satisfy `TieredCache`'s size-based API). Delete the token-map bookkeeping from `PipelineCache`.

---

### DUP-25 — `OpKind` enum defined in two places

**CHECK:** 2 (same type)  
**SEVERITY:** LOW  
**LOCATIONS:**
- `vyre-conform/src/enforce/enforcers/decomposition/op_kind.rs:7` — `pub enum OpKind` (production code, 7 variants)
- `vyre-conform/tests/diagnostics/ui/match_dispatch.rs:1` — `pub enum OpKind` (test UI diagnostic, 1 variant)

**DESCRIPTION:** The test file `match_dispatch.rs` under `tests/diagnostics/ui/` defines its own `pub enum OpKind` as a simplified stand-in. Because it lives in a UI test file, Rust treats it as a separate compilation unit and both can coexist. The risk is that a developer searching for `OpKind` finds the test definition and mistakes it for canonical production code.

**IMPACT:** Low operational risk (test isolation). Documentation risk: `rustdoc` or IDE navigation may surface both.

**UNIFY PLAN:** Rename the test's stand-in to `pub enum FakeOpKind` or `TestOpKind` to make the test-only nature explicit.

---

### DUP-26 — `clap` and `fs2` not in workspace.dependencies

**CHECK:** 3 (Cargo.toml deduplication)  
**SEVERITY:** LOW  
**LOCATIONS:**
- `vyre-conform/Cargo.toml` `[dependencies]`: `clap = { version = "=4.5.21", features = ["derive"] }` — not in workspace
- `vyre-conform/Cargo.toml` `[dependencies]`: `fs2 = "0.4"` — not in workspace

**DESCRIPTION:** `clap` and `fs2` appear in only one crate each. They are not currently workspace-level candidates unless a second crate needs them. However, `clap` is a major dependency (CLI framework) and should be pinned in `workspace.dependencies` to prevent accidental version bumps via `cargo update` if a second CLI crate is added. `fs2` is unpinned (`"0.4"` instead of `"=0.4.x"`) unlike every other dep in the workspace, which uses `"=x.y.z"` exact pins.

**IMPACT:** `fs2 = "0.4"` will float to the latest 0.4.x patch on `cargo update`, diverging from the workspace's exact-pin discipline.

**UNIFY PLAN:** Pin `fs2` to an exact version (`"=0.4.3"` or latest 0.4.x). Move both to `workspace.dependencies` for consistency and future-proofing.

---

### DUP-27 — Two `op_registry` concepts serving the same consumers

**CHECK:** 8 (parallel registries), 2 (same type)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-core/src/ops/registry/registry.rs` — `pub fn registry() -> impl Iterator<Item = &'static ConformSpec>` — ground truth, build-generated + runtime-registered
- `vyre-conform/src/spec/op_registry.rs` — `pub fn all_specs() -> Vec<ConformSpec>` — cached clone of core registry, partitioned by category

**DESCRIPTION:** The conform-side registry (`vyre-conform::spec::op_registry`) is a manual aggregation of per-category `specs()` calls that ultimately read from the core registry. The categorisation in conform (`primitive`, `decode`, `graph`, `hash`, `match_ops`, `string`) mirrors the directory structure in `vyre-core/src/ops/`. This means every new operation category added to `vyre-core` requires:
1. A new directory under `vyre-core/src/ops/<category>/`
2. A new `vyre-conform/src/spec/<category>/` module
3. A new `specs.extend(crate::spec::<category>::specs())` line in `op_registry.rs`

Step 3 is manual and error-prone. If omitted, the new category's operations are invisible to all conform checking, golden generation, and certification.

Furthermore, the conform registry clones every `ConformSpec` into a `Vec` on first access (copying all `&'static str` fields into `String`s at `LazyLock` time), while the core registry provides zero-copy `&'static ConformSpec` references. This 1× allocation per spec exists only because conform needs owned `Vec<ConformSpec>` — which itself is a design smell.

**IMPACT:** Silent coverage gap on new op categories. Memory waste from cloning static data.

**UNIFY PLAN:** Replace the manual `specs.extend(...)` list with an `automod`-generated module scan (the workspace already uses `automod` in multiple places). Change `all_specs()` return type to `&'static [&'static ConformSpec]` and make it a thin wrapper over `vyre::ops::registry()` collected into a `LazyLock<Vec<&'static ConformSpec>>`. The per-category partitioning for conform should be derived from `ConformSpec::category`, not a parallel module hierarchy.

---

---

### DUP-28 — `OpSignature` defined in vyre-spec AND vyre-conform

**CHECK:** 2 (same type)  
**SEVERITY:** CRITICAL  
**LOCATIONS:**
- `vyre-spec/src/op_signature.rs` — `pub struct OpSignature { inputs: Vec<DataType>, output: DataType }` with `min_input_bytes()`
- `vyre-conform/src/spec/types/conform/data.rs` (or equivalent) — `pub struct OpSignature { ... }` with identical fields and `min_input_bytes()` method

**DESCRIPTION:** `vyre-spec` was explicitly designed as the "frozen data contracts" crate. Despite this, `vyre-conform` re-defines `OpSignature` independently rather than importing it from `vyre-spec`. The conform version imports `vyre_spec::DataType` but still wraps it in a locally-defined struct. If either definition adds or changes a field, they silently diverge, producing inconsistent byte counts for all operations.

**IMPACT:** Any bug in `min_input_bytes()` must be fixed in two places. If the two structs disagree on field ordering and vyre-conform serialises its own `OpSignature` while vyre-spec's is used for registry checks, byte-offset calculations for multi-input ops will be wrong.

**UNIFY PLAN:** Delete `vyre-conform`'s `OpSignature`. Import and use `vyre_spec::OpSignature` everywhere in vyre-conform. `vyre-spec` already is a dependency of vyre-conform. Add any conform-only fields as a separate `ConformSignatureExt` newtype if needed.

---

### DUP-29 — `ConformSpec` name collision: two types with identical names, divergent semantics

**CHECK:** 2 (same type), 3  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-core/src/ops/registry/registry.rs` — `ConformSpec` as a compile-time static metadata record (the core IR registration)
- `vyre-conform/src/spec/types` — `ConformSpec` as a fat runtime spec with CPU reference functions, WGSL generators, test tables, and law declarations

**DESCRIPTION:** Both types are called `ConformSpec` but contain entirely different data and serve different roles. The core `ConformSpec` is a lightweight static struct; the conform `ConformSpec` is a heavyweight runtime struct embedding function pointers and owned test data. `vyre-conform` imports `vyre::ops::ConformSpec` in some places and `crate::spec::types::ConformSpec` in others, creating silent ambiguity. A developer reading either file must verify which `ConformSpec` is in scope — a violation of LAW 4 (understandability in 5 minutes).

**IMPACT:** Future contributors will write code that accidentally uses one where the other is expected. IDE autocompletion will suggest both. Code review cannot catch the confusion without full type context.

**UNIFY PLAN:** Rename vyre-conform's type to `ConformSpec`. Update all ~300 usages within vyre-conform. The rename is mechanical but necessary. Document in `ConformSpec`'s doc comment that it extends the corresponding `vyre::ops::ConformSpec` with runtime test harness data.

---

### DUP-30 — `Value` enum defined twice: vyre-reference and vyre-conform

**CHECK:** 2 (same type)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-reference/src/value.rs` (or equivalent) — `pub enum Value` with variants `U32, Bool, Bytes, F32, F64, I32, I64` using `Arc<[u8]>` for Bytes
- `vyre-conform/src/spec/value.rs` — `pub enum Value` with identical variants plus `Tensor`, using `Vec<u8>` for Bytes

**DESCRIPTION:** `vyre-conform` explicitly provides `to_reference_values()` and `from_reference_values()` round-trip converters that copy data between the two `Value` types, confirming they model the same domain. The only differences are: Bytes storage (`Arc<[u8]>` vs `Vec<u8>`) and the extra `Tensor` variant in conform. The conversion functions are lossy for Tensor (which has no reference equivalent), indicating a design that was never resolved.

**IMPACT:** Every call to the reference interpreter through conform allocates a full clone of input data via the conversion. At internet scale (billions of test invocations in a conformance run), this is measurable overhead. More critically, if a new variant is added to either `Value`, the converter silently fails to handle it.

**UNIFY PLAN:** Unify into one `Value` enum in `vyre-reference` (or in `vyre-spec`). Add the `Tensor` variant there with `Vec<u8>` backing. Remove the converters. Change `vyre-reference`'s Bytes storage from `Arc<[u8]>` to `Vec<u8>` (or vice versa, choosing the more efficient form). `vyre-conform` imports `vyre_reference::Value` directly.

---

### DUP-31 — `DefendantCatalog` struct defined twice within vyre-conform

**CHECK:** 2 (same type), 7  
**SEVERITY:** CRITICAL  
**LOCATIONS:**
- `vyre-conform/src/meta/observe/defender.rs` — inline `DefendantCatalog` struct definition
- `vyre-conform/src/meta/observe/defender/defendant_catalog.rs` — standalone `DefendantCatalog` struct definition (byte-for-byte identical fields and derives)

**DESCRIPTION:** The same struct exists in two files within the same crate. The parent module `defender.rs` likely includes the child via `mod defendant_catalog` creating two types with the same name in adjacent scopes. Rust's module system allows both to coexist, meaning callers may use either depending on import path. The two definitions will silently diverge the moment one is modified.

**IMPACT:** Any field added to one definition must be manually added to the other. A partial update produces a compilation error only if the types are used interchangeably — not always the case if both are used in their respective scopes.

**UNIFY PLAN:** Delete the inline definition in `defender.rs`. Add `pub use defender::defendant_catalog::DefendantCatalog;` to re-export from the canonical location. Run `cargo check` to confirm all usages resolve to one type.

---

### DUP-32 — Parallel `pub fn run(specs: &[ConformSpec])` free functions shadow `EnforceGate::run` trait

**CHECK:** 6 (parallel Validator/Verifier/Checker implementations)  
**SEVERITY:** HIGH  
**LOCATIONS:**
- Multiple enforcer modules in `vyre-conform/src/enforce/enforcers/`: `no_silent_wrong`, `overflow_contract`, `cost_certificate`, `divergence` — each exports a standalone `pub fn run(specs: &[ConformSpec]) -> Vec<Finding>` free function
- `vyre-conform/src/enforce/EnforceGate` trait — `fn run(&self, ctx: &EnforceCtx) -> Vec<Finding>` — the canonical dispatching interface

**DESCRIPTION:** The `EnforceGate` trait provides a uniform interface for all enforcement checks. Multiple enforcer modules additionally export free-function `run` variants that bypass the trait. These free functions have different signatures (`specs: &[ConformSpec]` vs `ctx: &EnforceCtx`) and are called directly from pipeline code instead of through the trait dispatch. This creates two parallel invocation paths for enforcement — the `EnforceGate` registry path and the direct-call path — that can fall out of sync.

**IMPACT:** An enforcer added to the direct-call pipeline but not registered as an `EnforceGate` would silently execute only in some contexts. Similarly, a refactor that moves an enforcer to the trait path might not remove the free function, leaving dead code that is never called.

**UNIFY PLAN:** Remove all standalone `pub fn run(specs: &[ConformSpec])` free functions from enforcer modules. Move their logic entirely into the corresponding `EnforceGate::run` impl. Provide a top-level `enforce_all(ctx: &EnforceCtx) -> Vec<Finding>` that iterates the trait object registry uniformly. Gate the entire pipeline on this single dispatch path.

---

### DUP-33 — Defender corpus codegen duplicated between vyre-build-scan and vyre-conform

**CHECK:** 1 (same function), 8  
**SEVERITY:** HIGH  
**LOCATIONS:**
- `vyre-build-scan/src/conform/defenders.rs` — parses TOML manifests from `defenders/` directory, emits `corpus_catalogs() -> Vec<DefendantCatalog>` Rust source
- `vyre-conform/build/defender_corpus.rs` (build script) — identical TOML-parsing + codegen logic for the same `defenders/` directory

**DESCRIPTION:** The build-time code generation that scans the `defenders/` TOML corpus and emits a Rust source file is implemented identically in two places. `vyre-build-scan` was specifically designed to own this kind of filesystem-scan-to-codegen logic. Yet `vyre-conform`'s `build.rs` reimplements the same scan independently. The two implementations must be kept in sync whenever the TOML schema or output format changes.

**IMPACT:** A schema change to the `defenders/` TOML format must be applied to both implementations. If one is updated and the other is not, `cargo build` for vyre-conform succeeds but with stale generated code from the build-scan path, and the inline build.rs path produces different output — a silent inconsistency that only manifests at runtime.

**UNIFY PLAN:** Delete `vyre-conform/build/defender_corpus.rs`. Have `vyre-conform`'s `build.rs` call `vyre_build_scan::scan_conform(...)` (the function that appears to have been the intended entry point). `vyre-build-scan` is already a build-dependency of vyre-conform. This is a one-function call replacement.

---

## CROSS-CUTTING OBSERVATIONS

### Cache Directory Standard

Three caches currently store data under `~/.cache/vyre/`:
- `~/.cache/vyre/dfa/` — DFA pattern cache (vyre-std)
- `~/.cache/vyre/pipeline/` — WGSL pipeline cache (vyre-wgpu)
- (Implicit) `wgpu::PipelineCacheDescriptor` uses a driver-managed path

There is no shared `vyre-cache` module coordinating these. A user running `VYRE_NO_CACHE=1` only bypasses the DFA cache (vyre-std checks this env var). The pipeline cache has no equivalent bypass. A `vyre-cache-dir` utility (50 lines) would give both caches identical bypass semantics and XDG path resolution.

### Workspace Pinning Discipline

The workspace correctly uses exact `=x.y.z` pins for all external dependencies. The exceptions are:
- `fs2 = "0.4"` in vyre-conform (DUP-26)
- `vyre-conform deps: clap = "=4.5.21"` — actually exact-pinned, acceptable

The discipline is otherwise excellent and should be enforced via `deny.toml` if not already.

### `inventory::collect!` Sites

Three `inventory::collect!` sites exist:
- `BackendRegistration` in vyre-core/src/backend/registry.rs
- `PassRegistration` in vyre-core/src/optimizer.rs
- `Handler` in vyre-conform/tests/catb_round2_inventory_linkme.rs (test only)

The test site is intentional (testing that the enforce gate catches `inventory::collect!` usage). The two production sites are fine as they register distinct types. No unification is needed here — this is the correct use of `inventory`.

---

## SEVERITY LEGEND

| Severity | Meaning |
|----------|---------|
| CRITICAL | Correctness: divergent implementations may produce different results |
| HIGH     | Architecture: layering violation or maintenance-trap that will cause bugs |
| MEDIUM   | Quality: technical debt that compounds with every new contributor |
| LOW      | Hygiene: inconsistency that causes confusion but no immediate risk |
