# PHASE8_ORCH  -  Audit Report

**Scope:** `libs/tools/surgec/src/scan/*` (except collector / decode), `vyre-driver/src/speculate.rs`, `vyre-driver/src/pipeline.rs` on_disk submodule, `vyre-driver/src/persistent.rs`, `vyre-foundation/src/optimizer/passes/fuse_cse.rs`.

**Date:** 2026-04-24

---

## Architectural Answers (8 Specific Asks)

| # | Question | Answer |
|---|----------|--------|
| 1 | Does the scan loop fuse rules (G2 fuse_cse) before dispatch or run each rule as separate Program? | **Runs each rule as a separate Program.** `dispatch_rules` iterates `for rule in &doc.rules` and calls `dispatch_single_clause` per clause. `fuse_cse` exists but has **zero production callers** in the scan pipeline. |
| 2 | Is the exemption filter a vyre Program or a Rust hot-loop? | **Rust hot-loop.** `apply_exemptions` is a nested CPU loop over `findings × exemptions` with `glob::Pattern` matching. No GPU involvement. |
| 3 | Is confidence scoring a subgroup prefix-sum kernel or a CPU reduce? | **CPU scalar reduce.** `compute_confidence` is a per-finding `f32` arithmetic function invoked from `dispatch_single_clause`. No device buffer, no parallel reduction. |
| 4 | Is AdaptiveSpeculator actually queried before every dispatch or only at startup? | **Neither.** `AdaptiveSpeculator` is exported but has **zero production callers** outside its own unit tests. It is dead scaffolding. |
| 5 | Does the on_disk cache key include GPU driver version AND device generation? | **Yes.** `compute_cache_key` length-prefixes both `driver_version` and `device_gen` into the blake3 hash. However, the module itself has **zero production callers**; no backend invokes `load` or `store`. |
| 6 | Is the PersistentEngine ring buffer actually consumed by a device-side kernel, or is it host-only? | **Host-only.** The module docs state the persistent GPU kernel lives behind a `persistent` cargo feature, but **that feature does not exist** in any `Cargo.toml`. The `PersistentEngine` is a host-side `std::sync::RwLock` + `AtomicU32` queue with no device-visible mapping. |
| 7 | Rule provenance chain  -  is it a vyre Program graph walk or CPU chain traversal? | **CPU hash chain.** `Provenance` is built from `blake3(rule_source)` + `program.fingerprint()` + `backend_adapter_fingerprint()`. No graph walk; no vyre Program. |
| 8 | Confidence sort  -  quicksort on CPU (O(n log n) serial) or radix sort on GPU? | **CPU serial sort.** `exploit_graph.rs:275` calls `chains.sort_by(...)` (Rust introsort). No GPU radix sort is used for confidence ordering. |

---

## Findings

| SEVERITY | file:line | defect | fix |
|----------|-----------|--------|-----|
| **CRITICAL** | `libs/tools/surgec/src/scan/dispatch.rs:111` | Scan loop dispatches each rule/clause as a separate Program; G2 fuse_cse optimizer exists but is never invoked. | Wire `fuse_cse` into `dispatch_rules` to fuse compatible rules into a single megakernel dispatch before backend submission. |
| **CRITICAL** | `libs/performance/matching/vyre/vyre-foundation/src/optimizer/passes/fuse_cse.rs:52` | `fuse_cse` has zero production callers in the scan pipeline; fusion is dead code. | Integrate `fuse_cse` into the rule-compilation path so `dispatch_rules` receives already-fused Programs. |
| **CRITICAL** | `libs/performance/matching/vyre/vyre-driver/src/speculate.rs:161` | `AdaptiveSpeculator.should_speculate()` has zero production callers; speculative dispatch is never toggled. | Insert `AdaptiveSpeculator` query before every backend dispatch and feed post-dispatch counter tails into `record()`. |
| **CRITICAL** | `libs/performance/matching/vyre/vyre-driver/src/pipeline.rs:405` | `on_disk` cache key derivation and I/O exist but no backend calls `load`/`store`; G8 on-disk cache is phantom code. | Implement on-disk caching in backend `compile_native` implementations (e.g. wgpu, cuda) using `compute_cache_key_for`. |
| **CRITICAL** | `libs/tools/surgec/src/scan/exemptions.rs:89` | `apply_exemptions` has zero production callers; exemption filter is orphaned from the scan output path. | Wire `apply_exemptions` into the collector or dispatch post-processing stage before findings are emitted. |
| **HIGH** | `libs/tools/surgec/src/scan/dispatch.rs:270` | Confidence scoring is scalar CPU `f32` arithmetic per-finding, not a GPU subgroup prefix-sum or parallel reduction. | Batch raw scores into a device buffer and compute confidence via a vyre parallel-prefix-sum kernel. |
| **HIGH** | `libs/tools/surgec/src/scan/exemptions.rs:112` | Exemption matching is a nested CPU hot-loop with `glob::Pattern` evaluation over every `finding × exemption` pair. | Compile exemptions into a GPU-side filter program (bitset or trie) and evaluate on device. |
| **HIGH** | `libs/performance/matching/vyre/vyre-driver/src/persistent.rs:80` | `PersistentEngine` ring buffer is host-only `std::sync::RwLock`/`AtomicU32`; no device kernel consumes it. | Either implement the Vulkan async-compute persistent kernel that maps the same atomics, or delete the module to eliminate architectural debt. |
| **HIGH** | `libs/tools/surgec/src/scan/exploit_graph.rs:275` | Chain confidence sort uses CPU `Vec::sort_by` (introsort, serial O(n log n)), not GPU radix sort. | Dispatch chain sorting to a vyre radix-sort or bitonic-sort primitive for large chain counts. |
| **HIGH** | `libs/tools/surgec/src/scan/provenance.rs:55` | `canonical_rule_source_hash` uses unstable Debug formatting (`{rule:#?}`); provenance changes across compiler versions. | Replace Debug format with a canonical stable wire format or serialized AST bytes. |
| **HIGH** | `libs/tools/surgec/src/scan/dispatch.rs:362` | `decode_result_slots` under-allocates `Vec` capacity (`bytes.len() / 8` instead of `/ 4`). | Change capacity to `bytes.len() / 4`. |
| **HIGH** | `libs/tools/surgec/src/scan/dispatch.rs:221` | `dispatch_single_clause` clones the entire `Program` for every clause dispatch just to override `workgroup_size`. | Add `workgroup_size` override to `DispatchConfig` so the backend can specialize without cloning `Program`. |
| **HIGH** | `libs/tools/surgec/src/scan/exemptions.rs:133` | Expiry check uses `today != expires` exclusion, meaning an exemption that expires today is still valid today. | Remove the equality guard so `date_string_ge(today, expires)` correctly invalidates on the expiry date. |
| **HIGH** | `libs/tools/surgec/src/scan/exploit_graph.rs:418` | `gpu_components` loops over unassigned pivots and dispatches a separate SCC kernel per pivot with host round-trip each. | Batch all pivots into one dispatch or replace with a single GPU-connected-components primitive. |
| **HIGH** | `libs/tools/surgec/src/scan/exploit_graph.rs:323` | `build_directed_edges` uses O(n²) nested CPU loops over all node pairs to build edges. | Pre-filter candidates by primitive transition existence or move edge building to a GPU join kernel. |
| **MEDIUM** | `libs/tools/surgec/src/scan/confidence.rs:21` | `compute_confidence` uses undocumented magic constants (`0.35`, `0.15`, `0.10`, `0.20`) with no configurability. | Expose confidence weights as TOML-configurable parameters or backend profile fields. |
| **MEDIUM** | `libs/tools/surgec/src/scan/confidence.rs:270` | `dispatch.rs` constructs a dummy `Finding` (`confidence=0.0`, `provenance=default`) solely to call `compute_confidence`. | Refactor `compute_confidence` to accept the minimal fields it needs (`rule_name`, `primitive`, `byte_offsets`). |
| **MEDIUM** | `libs/tools/surgec/src/scan/provenance.rs:71` | `backend_adapter_fingerprint` hardcodes the string `"wgpu"` and directly calls `vyre_driver_wgpu::runtime::cached_adapter_info()`. | Add `adapter_fingerprint()` to the `VyreBackend` trait so every backend self-describes without string matching. |
| **MEDIUM** | `libs/performance/matching/vyre/vyre-driver/src/pipeline.rs:500` | `on_disk::store` writes to `.bin.tmp` but leaves temp files if the process crashes before rename. | Use a unique temp filename and a `Drop` guard or periodic reaper to clean up stale `.tmp` files. |
| **MEDIUM** | `libs/performance/matching/vyre/vyre-driver/src/persistent.rs:62` | `PersistentEngine` uses per-slot `std::sync::RwLock` for a device feeder queue; adds host-side synchronization overhead. | Replace `RwLock` with a lock-free ring buffer using raw atomic writes to a mapped buffer. |
| **MEDIUM** | `libs/tools/surgec/src/scan/filter.rs:127` | `rule_enabled` only inspects the first `enabled` attribute and silently ignores duplicates. | Validate duplicate `enabled` attributes at AST parse time and reject or warn. |
| **MEDIUM** | `libs/tools/surgec/src/scan/exploit_graph.rs:526` | `best_path` recomputes `transition_weight` product on CPU after GPU path reconstruction. | Compute path confidence product inside the GPU `path_reconstruct` kernel and return it as an output buffer. |
| **MEDIUM** | `libs/performance/matching/vyre/vyre-driver/src/speculate.rs:171` | `record()` treats zero-attempted-tiles reports as no-ops, masking kernel crashes that write zero counters. | Distinguish empty-input reports from kernel-failure reports via an explicit sentinel or error buffer. |
| **MEDIUM** | `libs/tools/surgec/src/scan/dispatch.rs:323` | `optimal_workgroup_size` hardcodes register-pressure boundaries (`24` Lets → `64` threads) without adapter profiling. | Query adapter-specific register file size and use a spilling cost model instead of a Let-count heuristic. |
| **LOW** | `libs/tools/surgec/src/scan/bundle_runner.rs:16` | `MAX_BUNDLE_BYTES` is a hardcoded `256 MiB` constant with no override mechanism. | Make `MAX_BUNDLE_BYTES` overridable via environment variable or runtime config. |

---

## Summary

PHASE8_ORCH is a **scaffolding graveyard**: `fuse_cse`, `AdaptiveSpeculator`, `on_disk` cache, `apply_exemptions`, and `PersistentEngine` all exist as well-documented, well-tested modules, but **none are wired into the production scan path**. The actual orchestration loop (`dispatch_rules`) falls back to the naivest possible implementation: one GPU dispatch per rule clause, CPU scalar confidence, CPU nested-loop exemption matching, and CPU serial sorting. At internet scale this is **O(rules × dispatches)** host round-trips with no batching, no speculation, and no persistent kernel amortization.

The immediate fix priority is:
1. **Wire or delete** the five orphaned modules (`fuse_cse`, `AdaptiveSpeculator`, `on_disk`, `apply_exemptions`, `PersistentEngine`). Unconnected code is architectural debt that rots.
2. **Fuse before dispatch.** The G2 pass is already written; calling it costs one line.
3. **Move exemption filtering to compile-time or GPU.** Running glob matching over every finding at scan time is quadratic host work.
4. **Replace CPU sorts/reductions with vyre primitives.** Confidence scoring, chain sorting, and graph edge building should all be device-side where data already lives.
