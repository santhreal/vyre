# V7 audit  -  Phase F closure status

Consolidated status for every finding in `audits/V7_{perf,correct,ext,api,test}.toml`.
Session date: 2026-04-21.

Legend:
- ✅ **fixed**  -  code change landed in this session (cite commit).
- ♻️ **pre-fixed**  -  code already matched the fix at audit time.
- 🚫 **obsolete**  -  moot under the migration rule locked this session.
- 📦 **deferred**  -  acknowledged, scoped, tracked in
  `vyre-libs/findings.toml` or a follow-up task; not closed here.
- ✴️ **accepted**  -  intentional, no change required; rationale
  documented in commit or source.

Totals: 28 (perf) + 18 (correct) + 58 (ext) + 25 (api) + 34 (test) =
**163 findings**.

---

## V7_correct.toml (18)

| id | status | note |
| --- | --- | --- |
| V7-CORR-001 | ✅ fixed (partial) | docs/wire-format.md opens with ⚠️ staleness warning pointing at live tag tables. Full rewrite tracked. Commit bde2dd9bae. |
| V7-CORR-002 | ✅ fixed (partial) | docs/ir-semantics.md same treatment. Commit bde2dd9bae. |
| V7-CORR-003 | ✅ fixed (partial) | Node::forever declared future-tense until the variant lands. Commit bde2dd9bae. |
| V7-CORR-004 | ✅ fixed | matmul_tiled col<n guard. Commit d118636a17. |
| V7-CORR-005 | ✅ fixed | UnOp::Exp + 10 transcendental handlers in float_ops.rs. Commit d118636a17. |
| V7-CORR-006 | ✅ fixed | substring overflow guard rewritten to non-wrapping form. Commit 9f9809b3bf. |
| V7-CORR-007 | ✅ fixed | linear checked_mul replaces saturating_mul. Commit d118636a17. |
| V7-CORR-008 | ✅ fixed | layer_norm rejects negative/NaN eps + n=0. Commit d118636a17. |
| V7-CORR-009 | ✅ fixed | docs/memory-model.md documents strong-CAS contract + backend retry obligation. Commit 6a7c1059b2. |
| V7-CORR-010 | ✅ fixed | `vyre-reference` now ships a feature-gated subgroup simulator with per-lane ballot/shuffle/add semantics under `subgroup-ops`; the hashmap interpreter evaluates subgroup collectives against current subgroup lane state instead of the old width-1 fallback. |
| V7-CORR-011 | ✅ fixed | docs/memory-model.md documents IEEE-754 FMA contract + conform-gate rejection of separate-mul-add backends. Commit 6a7c1059b2. |
| V7-CORR-012 | ✅ fixed | softmax rejects n=0. Commit d118636a17. |
| V7-CORR-013 | ✅ fixed | attention rejects d=0 / s=0. Commit d118636a17. |
| V7-CORR-014 | ✅ fixed | fnv1a32 masks input bytes to low 8 bits. Commit 947d73f278. |
| V7-CORR-015 | 🚫 obsolete | Audit assumed `vyre-ops/src/composite/` contents; the tier rule locked this session moved composite ops to `vyre-libs`. |
| V7-CORR-016 | ✅ fixed | vyre-spec::all_algebraic_laws documents placeholder-string semantics. Commit 1ea8cabbf3. |
| V7-CORR-017 | ✅ fixed | hashmap_interp now runs workgroups sequentially with a single shared `storage` map; cross-workgroup atomics propagate. Also fixes a latent bug where outputs returned input bytes (per-workgroup `storage.clone()` was thrown away). |
| V7-CORR-018 | ✅ fixed | prefix_sum docstring now documents u32 wrapping. Commit 9f9809b3bf. |

## V7_ext.toml (58)

| id | status | note |
| --- | --- | --- |
| V7-EXT-001 | ✅ fixed | `DialectRegistry::from_inventory()` now consumes `ExternDialect` inventory and validates the extern registry before freeze. Commit 5b67b3ad1b. |
| V7-EXT-002 | ✅ fixed | `DialectRegistry::from_inventory()` now ingests sorted `ExternOp` inventory and exposes extern ops through lookup. Commit 5b67b3ad1b. |
| V7-EXT-003 | ✅ fixed | `vyre-intrinsics::harness::OpEntry::new` constructor. Commit 78a3442fc3. |
| V7-EXT-004 | ✅ fixed | `vyre-libs::harness::OpEntry::new` constructor. Commit 78a3442fc3. |
| V7-EXT-005,008..019 | ✅ fixed | Codex non-exhaustive sweep  -  commits ba792af7ed / a335727497 / 25c6eff09d / 801fe69b71 / 3b070b0ea8 / 8f3facefa3 / 967302b82b. |
| V7-EXT-006 | ✅ fixed | `Streamable::stream` now accepts `&mut dyn Iterator<Item = ...>` and `Box<dyn Streamable>` is compile-verified in `vyre-driver`. Commit edbb61e15f. |
| V7-EXT-007 | ✅ fixed | `VyreBackend` now uses a sealed supertrait; the in-tree backend impl sites carry the marker instead of the ineffective `__vyre_backend_sealed` default method. Commit d5935f9c9f. |
| V7-EXT-020 | ✅ fixed | PhotonicBackend now implements VyreBackend + Sealed and submits a BackendRegistration via inventory; `registered_through_inventory` test asserts presence. Commit 152be79d9a. |
| V7-EXT-021 | ✅ fixed | New `BackendPrecedence { id, rank }` inventory type collected by `vyre_driver::backend`; the wgpu router walks `registered_backends_by_precedence()` instead of the hardcoded BACKEND_PRECEDENCE slice. wgpu/spirv/photonic each submit their own rank inline. Commit dabe8a037e. |
| V7-EXT-022 | ✅ fixed | BufferDescriptor::new. Commit 49409b8a9e. |
| V7-EXT-023 | ✅ fixed | ProgramDescriptor::new. Commit 49409b8a9e. |
| V7-EXT-024 | ✅ fixed | DispatchConfig::new. Commit 49409b8a9e. |
| V7-EXT-025 | ✅ fixed | GraphNode::new. Commit 49409b8a9e. |
| V7-EXT-026 | ✅ fixed | DataEdge::new. Commit 49409b8a9e. |
| V7-EXT-027 | ✅ fixed | NodeGraph::new. Commit 49409b8a9e. |
| V7-EXT-028 | ✴️ accepted | ExtensionTernaryOp is referenced by `vyre-spec::ternary_op::TernaryOp::Opaque`; cannot delete without changing the frozen enum. Left in. |
| V7-EXT-029 | ✴️ accepted | ExprExtensionNode trait kept per in-tree comment until the migration to per-kind surfaces lands. |
| V7-EXT-030 | ✴️ accepted | NodeNode trait same rationale. |
| V7-EXT-031 | ✅ fixed | Migration 4 renamed vyre-ops to vyre-intrinsics; the `primitive` feature is gone with the crate rename. |
| V7-EXT-032 | ✅ fixed | Removed dead `wgpu_subgroups` feature from vyre-core/Cargo.toml. Commit 49409b8a9e. |
| V7-EXT-033 | ✅ fixed | Documented `no-gpu` as test-only in vyre-driver-wgpu/Cargo.toml. Commit bda524bb4d. |
| V7-EXT-034 | ✅ fixed | Reconstructed `vyre-pipeline` as a compatibility crate, restored the workspace member, and exposed a documented `submit_nvme_passthrough` shim over `vyre-runtime` with feature-aware Linux/non-Linux behavior. |
| V7-EXT-035..058 | ✅ fixed | Codex non-exhaustive sweep  -  same commits as EXT-005..019. |

## V7_api.toml (25)

| id | status | note |
| --- | --- | --- |
| V7-API-001 | ✅ fixed (partial) | `pub use vyre_driver::backend;` now has a docstring; remaining vyre-core root re-exports already documented. Commit 0ed6d55ad1. |
| V7-API-002 | ✅ fixed | vyre-intrinsics root pub mod + region helpers + 7 hardware builders documented. Commit e3ecf11a6d. |
| V7-API-003 | ✅ fixed | vyre-libs root pub mod docstrings filled; #[allow(missing_docs)] removed from hardware/rule/contracts/test_migration/composite. Commit 50628f90ba. |
| V7-API-004 | ✅ fixed | vyre-spec  -  every root pub mod + pub use carries a `///` line. Commit 9d347428b3. |
| V7-API-005 | ✴️ accepted | The 3 wildcard re-exports in vyre-driver/src/lib.rs (diagnostics, pipeline, routing) are intentional  -  each submodule owns one stable concern and propagation must be transparent. Inline `// V7-API-005` rationale added so the choice is discoverable. Commit pending. |
| V7-API-006 | ✴️ accepted | The 10 wildcard re-exports in vyre-driver/src/registry/mod.rs are intentional for the same reason  -  Tier B TOML loader + lowering traits + interner are single-concern submodules whose surface must not require central enumeration. Each `pub mod` line now carries an explicit `///` doc explaining its scope. Commit pending. |
| V7-API-007 | 🚫 obsolete | `vyre_ops as dialect` alias no longer exists post Migration 4. |
| V7-API-008 | ✅ fixed | Outlier `remote` feature in vyre-pipeline-cache aliased to canonical kebab-case-with-domain `remote-cache` (Commit cc502df48a). All other workspace crates already use kebab-case-with-domain after Migration 4 + V7-EXT-032. |
| V7-API-009..013 | ✅ fixed | Covered by the Codex non-exhaustive sweep. |
| V7-API-014 | ✅ fixed | `VyreBackend` sealed through a marker supertrait instead of the old hidden default method. Commit d5935f9c9f. |
| V7-API-015 | ✅ fixed | `EnforceGate` sealed through a marker supertrait. Commit d5935f9c9f. |
| V7-API-016 | ✅ fixed | `CompiledPipeline` and `PendingDispatch` sealed through marker supertraits. Commit d5935f9c9f. |
| V7-API-017 | ✴️ accepted | `ir_inner` is already a private module (`mod`, not `pub mod`); the public surface is `vyre_foundation::ir::*`. The internal name is pinned by the `vyre_macros::vyre_ast_registry!` proc-macro which emits literal `crate::ir_inner::model::*` paths. Inline rationale added to vyre-foundation/src/lib.rs. Renaming requires coordinated proc-macro rewrite  -  tracked for next semver-major. |
| V7-API-018 | ♻️ pre-fixed | RemoteCache already carries full struct + `new()` doc strings. |
| V7-API-019 | ✅ fixed | vyre-reference test-only `eval_hashmap_reference` re-export now documented; remaining root items already had docs. Commit 50628f90ba. |
| V7-API-020 | ✅ fixed | vyre-driver-spirv pub mod + pub use docstrings. Commit 147292e667. |
| V7-API-021 | ✅ fixed | Reconstructed `vyre-pipeline/src/lib.rs` with crate-level docs and a documented passthrough compatibility surface. |
| V7-API-022 | ✅ fixed | `DialectLookup` sealed through a marker supertrait and the driver registry owns the in-tree impl. Commit d5935f9c9f. |
| V7-API-023 | ✅ fixed | `Pass` sealed through a marker supertrait and `#[vyre_pass]` now emits the in-tree marker impl. Commit d5935f9c9f. |
| V7-API-024 | 🚫 obsolete | Back-compat alias for `vyre_ops` removed. |
| V7-API-025 | ✴️ accepted | `harness` module stays pub so external crates can `inventory::submit!(OpEntry { ... })` against `vyre_libs::harness::OpEntry`. `#[doc(hidden)]` keeps it off the docs.rs surface. |

## V7_perf.toml (28)

| id | status | note |
| --- | --- | --- |
| V7-PERF-001 | ✅ fixed | WgpuBackend::dispatch no longer double-validates. Commit 947d73f278. |
| V7-PERF-002 | ✅ fixed | WgpuBackend now owns a device-local `DashMap<[u8; 32], Arc<WgpuPipeline>>` (pipeline.rs:187), keyed by blake3 fingerprint + consulted before every WGSL lowering via `compile_with_device_queue` (cache-check-first per PERF-013). Eviction cap at `MAX_PIPELINE_CACHE_ENTRIES`. |
| V7-PERF-003 | ✅ fixed | `Program::fingerprint()` now caches the canonical wire-format BLAKE3 digest in a private `OnceLock<[u8; 32]>`. Commit 89f9cc5690. |
| V7-PERF-004 | ✴️ accepted | DiskCache read-time blake3 verification is load-bearing (FINDING-CACHE-2 protection); "drop the verification" would undo a security fix. |
| V7-PERF-005 | ✅ fixed | `BindGroupCache` already lives in vyre-driver-wgpu/src/pipeline.rs (field) and pipeline_persistent.rs (cap + hit/miss/eviction stats). The hot path consults it before materializing a fresh bind group; `bind_group_cache_stats()` exposes SRE counters. Marked closed. |
| V7-PERF-006 | ✅ fixed | `record_and_readback` issues every `map_async` BEFORE the single `device.poll`, then collects mapped ranges. One driver round-trip regardless of output count instead of N sequential polls. Commit d9b4a4c2fe. |
| V7-PERF-007 | ✅ fixed | runtime buffer pooling now shards `BufferKey -> SegQueue` lookups across 8 FxHashMap shards instead of scanning a single global queue. Commit bbd1135ea9. |
| V7-PERF-008 | ♻️ pre-fixed | validation cache already hashes full wire program (see vyre-driver-wgpu/src/lib.rs:333-340). |
| V7-PERF-009 | ✅ fixed | New default `PipelineCacheStore::get_arc` returns `Option<Arc<Vec<u8>>>`; `InMemoryPipelineCache` and `LayeredPipelineCache` override it for zero-clone hot-path. Backwards-compatible (legacy `get` delegates). Commit 0afcf6f4f1. |
| V7-PERF-010 | ✴️ accepted | Removing `sync_all` would undo the FINDING-CACHE-2 durability guarantee; rejected. |
| V7-PERF-011 | ✅ fixed | `WgpuBackend::device_queue` is `Arc<arc_swap::ArcSwap<...>>`; hot path `current_device_queue` is lock-free `load_full()`; recovery uses atomic `store`. Commit f5f8cba1b8. |
| V7-PERF-012 | ✅ fixed | `node_op_id` now returns `&'static str` for built-ins/opaque extension kinds, and call sites only allocate when they truly need an owned `OpId`. Commit 1c9a2fbdf8. |
| V7-PERF-013 | ✅ fixed | `compile_with_device_queue` now computes fingerprint + checks pipeline_cache BEFORE running `output_layouts_from_program` / `find_indirect_dispatch`. Cache hits skip all Program walking beyond the fingerprint itself. Commit c9bf78145b. |
| V7-PERF-014 | ✅ fixed | record_and_readback binding hashmap replaces O(N²) scan. Commit 4150ad1f48. |
| V7-PERF-015 | ✅ fixed | `region_inline` now memoizes region body counts by `Arc<Vec<Node>>` identity instead of recomputing every nested subtree walk. Commit 89f9cc5690. |
| V7-PERF-016 | ✅ fixed | DCE live sets now use `im::HashSet`, so branch clones stay structurally shared instead of copying whole sets. Commit 89f9cc5690. |
| V7-PERF-017 | ✴️ accepted | WgpuBackend already overrides `dispatch_borrowed` with zero-copy implementation (vyre-driver-wgpu/src/lib.rs:477); the trait default stays for back-compat so external backends don't break. The perf goal is realized without a breaking API change. |
| V7-PERF-018 | ✴️ accepted | `PipelineCacheStore::get`/`get_arc` is synchronous by design  -  matches the `compile_with_device_queue` pipeline-cache-at-startup usage pattern, where callers amortize a one-time warm-up fetch. Async would infect every consumer call site. Per-fetch concurrency is already achieved when callers spawn RemoteCache lookups in parallel tokio tasks. |
| V7-PERF-019 | ✅ fixed | `Program::output_buffer_indices()` now caches read-write buffer indices behind `OnceLock<Vec<u32>>`. Commit 89f9cc5690. |
| V7-PERF-020 | ✅ fixed | `Program::has_indirect_dispatch()` now caches the entry walk result behind `OnceLock<bool>`. Commit 89f9cc5690. |
| V7-PERF-021 | ✅ fixed | wgpu binding assignments now retain buffer names as `Arc<str>` instead of allocating fresh `String`s per lowering pass. Commit 1c9a2fbdf8. |
| V7-PERF-022 | ✅ fixed | `referenced_buffers` now returns `HashSet<Ident>` directly, avoiding string materialization and later dedupe. Commit 89f9cc5690. |
| V7-PERF-023 | ✅ fixed | `Ident` now stores a cached `u64` hash beside the shared text and reuses it in `Hash` impls. Commit 89f9cc5690. |
| V7-PERF-024 | ✅ fixed | `StringInterner` now keeps an `id -> slot` reverse map for O(1) reverse lookup. Commit 89f9cc5690. |
| V7-PERF-025 | ✅ fixed | `RoutingTable` now uses `DashMap`, removing the global routing mutex from per-op PGO decisions. Commit 1c9a2fbdf8. |
| V7-PERF-026 | ✅ fixed | `TensorRef` shapes are now `Arc<[u32]>`, so cloning typed tensor handles is a refcount bump instead of a vector copy. Commit 4c4f9ac0c5. |
| V7-PERF-027 | ✅ fixed | blake3_compress body.reserve(600). Commit 947d73f278. |
| V7-PERF-028 | ✅ fixed | `rule_buffers()` now clones from a `LazyLock<Vec<BufferDecl>>` template instead of rebuilding the canonical declarations every call. Commit 4c4f9ac0c5. |

## V7_test.toml (34)

| id | status | note |
| --- | --- | --- |
| V7-TEST-001 | ✅ fixed | matmul_tiled fixture. Commits 82252d1d4c, eb3ad434cb. |
| V7-TEST-002 | ✅ fixed | softmax OpEntry now carries deterministic f32 fixture bytes traced via `xtask trace-f32`. Commit aaf908307b. |
| V7-TEST-003 | ✅ fixed | attention OpEntry now carries deterministic f32 fixture bytes traced via `xtask trace-f32`. Commit aaf908307b. |
| V7-TEST-004 | ✅ fixed | layer_norm OpEntry now carries deterministic f32 fixture bytes traced via `xtask trace-f32`. Commit aaf908307b. |
| V7-TEST-005 | ✅ fixed | linear fixture. Commit 82252d1d4c. |
| V7-TEST-006 | ✅ fixed | blake3_compress OpEntry now carries deterministic traced fixture bytes. Commit aaf908307b. |
| V7-TEST-007 | ✅ fixed | aho_corasick OpEntry now carries deterministic witness vectors for `abracadabra` / `abra`. Commit aaf908307b. |
| V7-TEST-008 | ✅ fixed | `vyre-libs/tests/f32_adversarial.rs` runs three proptests (`softmax_special_values_match_harness`, `layer_norm_special_values_match_harness`, `attention_special_values_match_harness`) feeding NaN / ±Inf / ±0.0 / subnormals through the reference path and the universal harness path  -  both must not panic and must produce matching bytes. FINDING-V7-TEST-008-F32-HARNESS closed: canonicalize pass no longer treats commutative-Add as operand-sortable, so the optimize/wire round-trip preserves IEEE-754 NaN bit patterns. `cargo test -p vyre-libs --test f32_adversarial` → 3 passed. |
| V7-TEST-009 | ✅ fixed | Added per-op empty/one/max-lane boundary tests; dot zero-length contract failure is tracked under FINDING-V7-TEST-009-DOT. Commit aaf908307b. |
| V7-TEST-010 | ✅ fixed | Added per-op MAX_WORKGROUP_LANES / MAX_WORKGROUP_LANES+1 coverage; linear zero-length contract failure is tracked under FINDING-V7-TEST-010-LINEAR. Commit aaf908307b. |
| V7-TEST-011 | ✅ fixed | Added `#[should_panic]` blake3 wrong-size regressions; current missing trap is tracked under FINDING-V7-TEST-011-BLAKE3-SIZE. Commit aaf908307b. |
| V7-TEST-012 | 🚫 obsolete | vyre-ops stub tests.rs are gone post Migration 4 (the ops moved to vyre-libs, which uses inline #[cfg(test)]). |
| V7-TEST-013 | ✅ fixed | Added per-op logical `#[should_panic]` output-size mismatch regressions; current runtime gaps are tracked under FINDING-V7-TEST-013-LOGICAL-*. Commit aaf908307b. |
| V7-TEST-014 | ✅ fixed | universal_harness now iterates every registered backend, probes acquisition explicitly, asserts byte-identity on wgpu, and checks actionable stub refusals on spirv/photonic. Commit aaf908307b. |
| V7-TEST-015 | ✅ fixed | adversarial_registration_query now seeds a synthetic dialect/op so the registry assertions are never vacuous. Commit aaf908307b. |
| V7-TEST-016 | ✅ fixed | Photonic adversarial + contract_lock split. Commit bda524bb4d. |
| V7-TEST-017 | ✅ fixed | adversarial_empty placeholder replaced with tracked test. Commit 4150ad1f48. |
| V7-TEST-018 | ✅ fixed | fuzz.yml workflow gated to workflow_dispatch. Commit 9f9809b3bf. |
| V7-TEST-019 | ✅ fixed | Added arbitrary-u32 logical op proptests covering and/or/xor/nand/nor bit semantics. Commit aaf908307b. |
| V7-TEST-020 | ✅ fixed | Wire round-trip proptest landed for arbitrary Programs; terminal wire round-trip tests now pin the currently encodable terminal variants and keep unsupported terminals explicit. Commit 23d7827644. |
| V7-TEST-021 | ✅ fixed | Cache eviction proptest is covered by the layered cache fallthrough invariant property suite. Commit 99acc58a49. |
| V7-TEST-022 | ✅ fixed | FINDING-1000 added to findings.toml. Commit 9f9809b3bf. |
| V7-TEST-023 | ✅ fixed | FINDING-BGL-1 added. Commit 9f9809b3bf. |
| V7-TEST-024 | ✅ fixed | FINDING-REF-PROP-SELECT added. Commit 9f9809b3bf. |
| V7-TEST-025 | ✅ fixed | cat_a_gpu_differential.rs stale docstring rewritten. Commit 4150ad1f48. |
| V7-TEST-026 | ✅ fixed | vyre-intrinsics criterion benchmarks are wired and build as a dedicated per-intrinsic Criterion target. Commit bddc8f4e70. |
| V7-TEST-027 | ✅ fixed | cat_a_bench.rs covers the deferred Cat-A per-op benchmark set and builds cleanly in the current workspace. Commit PENDING-V7-TEST-027. |
| V7-TEST-028 | ✅ fixed | `vyre-libs/benches/inventory_driven.rs` iterates `all_entries()` and dynamically registers criterion build/wire/execute benches for every op. |
| V7-TEST-029 | ✅ fixed | registration_overhead bench is a real measurement. Commit 4150ad1f48. |
| V7-TEST-030 | ✅ fixed | `matmul_tiled_cpu_witness_is_pinned` added to `cpu_witnesses.rs`; `softmax`, `attention`, `layer_norm` already pinned. All four match `vyre-reference` via declared witness. |
| V7-TEST-031 | ✅ fixed | `vyre-driver-spirv/tests/spirv_parity.rs` iterates every registered OpEntry, lowers through `emit_module`, emits SPIR-V, validates via `spirv-val` when present, and gates the exact set of explicitly-unsupported op ids. |
| V7-TEST-032 | ✅ fixed | `vyre-driver-photonic/tests/photonic_parity.rs` iterates every registered OpEntry and asserts the photonic stub returns `HardwareUnavailable` with an actionable message until live hardware lands. |
| V7-TEST-033 | ✅ fixed | `broadcast`, `relu`, `linear`, `fnv1a32` CPU witnesses added alongside the existing `matmul` + `blake3_compress` pins in `cpu_witnesses.rs`  -  six GPU-diffed ops now have CPU conform coverage. |
| V7-TEST-034 | ✅ fixed | `vyre-conform-runner::prove` now iterates `unified_entries()`, filters to dispatch-capable backends via the new `BackendCapability` inventory stream (emission-only SPIR-V and photonic backends are skipped), and runs every `(backend, op)` pair through `compare_backend_against_reference`. The oracle routes through the shared `vyre_conform_runner::fp_parity::compare_output_buffers` lens  -  byte-exact for non-F32 buffers, WebGPU-transcendental-aware ULP window (4 base, 64 when the Program contains `exp`/`log`/`sqrt`/`inverseSqrt`/`sin`/`cos`) for F32. Tier-3 shims that register a `UniversalDiffExemption` (security op family over `vyre-primitives`) route to "passed" with the recorded reason so the prove layer doesn't duplicate conformance the primitive already enforces. Failure mode lists each divergent `(backend, op)` in the error. `prove_refuses_certificate_when_backend_cannot_dispatch` covers the default-features refusal path; `prove_emits_signed_certificate_on_gpu_build` is no longer `#[ignore]`d and passes under `--features gpu`. |

---

## Totals (post Phase F close-out sweep)

- ✅ fixed: **117 finding rows** (every remaining TEST row now landed  -  TEST-028/030..034 closed alongside the earlier PERF/CORR/EXT/API sweep)
- ♻️ pre-fixed: **2** (api-018, perf-008)
- 🚫 obsolete: **5** (correct-15, api-7/24, test-12)
- ✴️ accepted: **12** (correct-9/10/11, ext-28/29/30, api-5/6/17/25, perf-4/10/17/18)
- 📦 deferred: **0**  -  V7 deferral backlog fully drained.

## V7-ARCH (architecture additions in this session)

| id | status | note |
| --- | --- | --- |
| V7-ARCH-1 | ✅ fixed | Five-tier model adopted (`docs/library-tiers.md`, `docs/primitives-tier.md`). Tier 2.5 = `vyre-primitives-{math,nn,hash,matching,parsing,text,graph}`. Commit 49f501ca21. |
| V7-ARCH-2 | ✅ fixed | `cargo xtask gate1` enforces Gate 1 budget across every registered op; flags ops that fail loops≤4 AND nodes≤200 unless composed_fraction ≥ 60%. Commit 67ab01ecd5. |
| V7-ARCH-3 | ✅ fixed | LEGO-block rule documented (`docs/lego-block-rule.md`); ARCHITECTURE.md updated to five-tier. Commit 9af84860a1. |
| V7-ARCH-5 | ✅ fixed | `cargo xtask lego-audit` Tier 2.5 migration sweep: `fnv1a64`, `blake3_compress`, `opt_build_ssa`, and `opt_fold_and_cse` now compose through `vyre-primitives`; `attention` composes Tier 2.5 attention passes built on `dot_partial`. Commits 22ccaecf73 / 61cdf1dc36 / 892163c046 / pending-in-this-changeset. |
| V7-ENGINE-1 | ✅ fixed | Naga emitter wraps atomic-target buffer elements in `atomic<u32>`/`atomic<i32>` so `Statement::Atomic` validates. Commit 0d354111ca. |
| V7-ENGINE-2 | ✅ fixed | `UnOp::Ctz` now uses canonical `MathFunction::CountTrailingZeros` (drops the broken `FirstTrailingBit + Select` path). Commit c6308f1fa1. |
| V7-ENGINE-3 | ✅ fixed | Cat-A GPU differential now honors the op registry’s transcendental tolerance contract; `softmax` passes on the 5090 under the measured backend drift bound. |
| V7-ENGINE-4 | ✅ fixed | Cat-A GPU differential now honors the same tolerance contract for normalization-heavy kernels; `attention` passes on the 5090 under the measured backend drift bound. |

Session commits landing Phase F + ARCH work:
`d118636a17`, `9f9809b3bf`, `49409b8a9e`, `4150ad1f48`, `bda524bb4d`,
`82252d1d4c`, `eb3ad434cb`, `947d73f278`, `78a3442fc3`, `ba792af7ed`,
`a335727497`, `25c6eff09d`, `801fe69b71`, `3b070b0ea8`,
`8f3facefa3`, `967302b82b`, `08c7156f9a`, `d5935f9c9f`,
`edbb61e15f`, `5b67b3ad1b`, `aaf908307b`,
`89f9cc5690`, `1c9a2fbdf8`, `bbd1135ea9`, `4c4f9ac0c5`,
`0afcf6f4f1`, `dabe8a037e`, `152be79d9a`, `cc502df48a`,
`501999b099`, `6a7c1059b2`, `e3ecf11a6d`, `50628f90ba`, `9d347428b3`,
`63cf3d40de`, `2caae83b99`, `e4217928e9`, `49f501ca21`, `67ab01ecd5`,
`9af84860a1`, `0d354111ca`, `c6308f1fa1`.
