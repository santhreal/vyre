# PHASE10_DIFF  -  Differential Scan + Watch Mode + Warm-Start Audit

**Date:** 2026-04-24  
**Scope:** `libs/tools/surgec/src/scan/diff_scan.rs`, `watcher.rs`, `diff_replay.rs`, `libs/performance/matching/vyre/vyre-primitives/src/fixpoint/bitset_fixpoint.rs`, `vyre-driver/src/pipeline.rs`, `vyre-driver/src/persistent.rs`  
**Auditor:** Kimi Code CLI  
**Standard:** LAWS 0–8, STANDARDS, RESEARCH PROTOCOL  

---

## Executive Summary

The differential scan and watch mode surfaces are **architecturally incomplete** and contain **multiple critical defects** that violate the project’s non-negotiable laws. G10 transitive closure is a CPU placeholder where a GPU primitive was promised. G7 PersistentEngine is implemented but completely unwired from watch mode. G8 on-disk pipeline cache is implemented but dead code, guaranteeing a miss every run. The diff-replay subsystem performs CPU metadata comparison instead of vyre Program semantics. At internet scale, these defects translate to O(corpus) work per file change, full shader recompilation on every process start, and silent under-scanning due to uncanonicalized path keys.

---

## Direct Answers to Specific Questions

1. **Is the transitive closure in diff_scan CPU BFS or vyre csr_forward_traverse?**  
   → **CPU BFS.** `diff_scan.rs:75` `transitive_closure` walks a `HashMap<PathBuf, Vec<PathBuf>>` on the host CPU with `VecDeque`. It does not lower the graph to a `ProgramGraph` or dispatch `csr_forward_traverse`.

2. **Is the include-graph cached across invocations or rebuilt per scan?**  
   → **Neither  -  it is dead code.** `build_include_graph` (`diff_scan.rs:163`) is defined but **never called** from the CLI path (`main.rs`). There is no `.surgec-cache/depgraph.bin` I/O. The graph is not built, not cached, and not used.

3. **Does watcher actually stream changes through G7 PersistentEngine ring buffer?**  
   → **No.** `watcher.rs:48` uses the `notify` crate + `std::sync::mpsc`. `PersistentEngine` (`persistent.rs`) is a standalone proven module with exhaustive tests, but **zero call sites** exist in `surgec/src/scan`. Watch mode pays per-file backend acquisition and full-target rescan on every event.

4. **Is bitset_fixpoint_warm_start actually called from diff_scan or unused?**  
   → **Unused from diff_scan.** It is invoked only in the v3 lowerer (`lower/mod.rs:950`) for `Fixpoint` AST expressions. The diff scan module has no warm-start seed plumbing and no inter-file state sharing.

5. **Does the G8 on_disk pipeline cache hit on warm-start, or is there a miss-every-run bug?**  
   → **Miss-every-run bug.** `pipeline.rs:292` `compile()` / `compile_shared()` never call `on_disk::load` or `on_disk::store`. The `on_disk` submodule (`pipeline.rs:381–721`) is fully implemented, tested, and documented  -  but **unwired** from the compile path.

6. **Any O(files²) work in diff_scan when we should be O(changed × closure)?**  
   → **Yes, implicitly.** Because transitive closure is unimplemented, the CLI falls back to `scan_selected_paths` (`main.rs:852`), which creates a new `Collector` per changed file. Each `Collector` rebuilds dispatch plans for **all rules** and re-dispatches against the backend. The work is O(changed × rules × dispatch_overhead), which on large rule sets approaches O(files²) relative to the minimal closure set. Additionally, `watcher.rs:111` rescans the **entire target_root** on every single file event, which is O(corpus) per event.

7. **Is rule-differential replay (diff_replay.rs) a vyre Program or CPU comparison?**  
   → **CPU comparison.** `diff_replay.rs:21` builds a `HashSet<Provenance>` from baseline rule metadata (source hash + program fingerprint) and filters candidate findings on the CPU. The baseline corpus is **never scanned**. There is no vyre Program dispatch for the diff operation.

---

## Findings

### 2026-04-29 scoped WGPU/megakernel closure status

The WGPU/megakernel rows in this audit have been rechecked against current
source. Surgec watcher/diff-source rows remain outside this WGPU-owned patch.

| Finding | Status | Source / proof |
|---|---|---|
| C5 G8 on-disk pipeline cache unwired | fixed/stale | Current `vyre-driver-wgpu/src/pipeline.rs` calls `load_or_compile_disk_wgsl`, `create_compiled_pipeline_cache`, and `persist_compiled_pipeline_cache`; `pipeline_disk_cache` tests `cache_key_isolates_wire_from_adapter` and `normalized_cache_digest_erases_runtime_storage_lengths` passed. |
| C3 watcher bypasses PersistentEngine | out of scoped source ownership | This row targets `surgec/src/scan/watcher.rs`, not WGPU/megakernel source. |
| C4 watch mode rescans target root | out of scoped source ownership | This row targets `surgec` scan orchestration. |
| C1/C2 diff transitive closure / include graph | out of scoped source ownership | These rows target `surgec` diff scan code and are not WGPU/megakernel implementation rows. |

### CRITICAL

**C1 | diff_scan.rs:75 | transitive_closure is CPU BFS instead of vyre GPU primitive**
The function does a host-side `HashMap<PathBuf, Vec<PathBuf>>` BFS with `VecDeque`. Per the G10 design doc in the module header, this should be a GPU-accelerated closure via `csr_forward_traverse` composed with `bitset_fixpoint`. At repo scale, CPU BFS on deep include chains (e.g., Chromium, LLVM) is a throughput bottleneck.
Fix: Lower the dep graph to a `ProgramGraph`, dispatch `csr_forward_traverse` + `bitset_fixpoint` on the GPU, and read back the reached bitset. Map bit indices back to paths via a stable node-ID table.

**C2 | diff_scan.rs:163 | Include-graph is dead code  -  never built, never cached**
`build_include_graph` is defined and tested, but grep shows **zero non-test callers** in the surgec tree. The module docstring promises persistence at `.surgec-cache/depgraph.bin`; no such I/O exists. Every diff scan that eventually wires this function will rebuild the graph from scratch.
Fix: Call `build_include_graph` from `run_scan` when `--diff` is active. Persist the graph with a content-hash key (blake3 of all source mtimes) and reload on subsequent invocations. Delete the graph if the key mismatches.

**C3 | watcher.rs:48 | Watch mode bypasses G7 PersistentEngine entirely**
`watch_scan` creates a `notify::RecommendedWatcher` and an `mpsc` channel. On every event it calls `emit_current_findings`, which acquires a fresh GPU backend and runs a full scan. The `PersistentEngine` ring buffer  -  proven correct with multi-producer/multi-consumer tests  -  is never instantiated.
Fix: Construct one `PersistentEngine` at `watch_scan` startup. Batch changed files into `WorkItem`s and `enqueue` them. Consume results from the ring buffer instead of per-file dispatch.

**C4 | watcher.rs:111 | Watch mode rescans entire target_root on every file change**
`emit_current_findings` takes `&config.target_root` and passes it to `Collector::new(document.clone(), target_root)`, which walks the full directory tree. A single-line edit in a 1M-file repo triggers a 1M-file rescan.
Fix: Pass the event’s `change_set` filtered file list to `scan_selected_paths`. Do not walk the full tree.

**C5 | pipeline.rs:292 | G8 on_disk cache is fully implemented but dead code  -  guaranteed miss every run**
`compile_shared()` calls `backend.compile_native()` and falls back to `PassthroughPipeline`. It never consults `on_disk::load` before compiling or `on_disk::store` after. The `on_disk` module has 340 lines of tests, key derivation, atomic store, and length-extension resistance  -  all unreachable.
Fix: In `compile_shared`, compute the cache key via `on_disk::compute_cache_key_for`, call `on_disk::load`, and if a backend blob exists, load it natively. After successful compilation, call `on_disk::store`.

**C6 | diff_replay.rs:21 | diff_replay is CPU HashSet comparison, not vyre Program semantics**
`run_diff_replay` builds a `HashSet<Provenance>` from baseline rule metadata and filters candidate findings on the CPU. The baseline corpus is never scanned. This is not a vyre dispatch; it cannot leverage GPU batching or persistent engine streaming.
Fix: If the design intent is true semantic diff-replay, compile a vyre Program that dispatches both rule sets and emits findings only when the candidate fires and the baseline does not. If metadata-only diff is intended, document the limitation in the module header and CLI help.

**C7 | diff_scan.rs:6 | G10 transitive closure is a documented stub that ships dead code**
The module docstring admits the implementation is a "surface placeholder" and that "real body lands in G10." This violates LAW 1 (no stubs). The `scaffold_transitive_closure` function referenced in the docstring does not exist.
Fix: Delete the placeholder comments and unexported helpers, or implement G10 fully. Do not ship documented fiction.

### HIGH

**H1 | diff_scan.rs:42 | DepGraph uses PathBuf keys without canonicalization guarantee**
`DepGraph` stores `HashMap<PathBuf, Vec<PathBuf>>`. `transitive_closure` does exact `HashMap` lookups. On case-insensitive filesystems (Windows, macOS default) or with mixed relative/abs paths, edges are silently missed, causing under-scanning.
Fix: Normalize every path with `std::fs::canonicalize` before insertion. Store stable `u32` node IDs in the graph and maintain a side map `id → PathBuf`.

**H2 | diff_scan.rs:210 | Missing-header fallback creates symbolic edges that never match**
When `canonicalize` fails (missing header), the code falls back to `PathBuf::from(include)`  -  the raw include string. A subsequent lookup for the real file path will never match this symbolic key, so dependents of missing headers are silently excluded from the closure.
Fix: Track unresolved includes in a separate symbolic layer. When a changed file’s path matches a previously unresolved include string, resolve the edge dynamically and emit a warning.

**H3 | watcher.rs:140 | GPU backend is re-acquired on every watch event**
`acquire_gpu_backend` calls `vyre_driver_wgpu::WgpuBackend::acquire()` inside `emit_current_findings`, which runs on every event. Adapter enumeration and device creation are expensive (tens to hundreds of ms) and can exhaust driver session limits under rapid edits.
Fix: Acquire the backend once in `watch_scan` and share an `Arc<dyn VyreBackend>` across the event loop.

**H4 | main.rs:852 | scan_selected_paths creates a new Collector per changed file, paying per-file dispatch overhead**
For each changed file, the code does `Collector::new(compiled.clone(), path)` and `scan_gpu_report`. Each call rebuilds dispatch plans for all rules and re-dispatches individually. No batching, no pipeline reuse.
Fix: Refactor `Collector` to accept a `Vec<PathBuf>` and batch clause dispatches across files, or stream files through the G7 `PersistentEngine`.

**H5 | diff_scan.rs:113 | parse_includes is a naive byte scanner with false positives**
The scanner matches `#include "..."` inside C comments, string literals, and `#if 0` blocks. This creates spurious edges (false positives) and misses macro-generated includes (false negatives).
Fix: Skip `//` and `/* */` comment regions, and ignore `#include` tokens inside string literals. Alternatively, delegate to `tree-sitter` or the compiler’s preprocessor for robust extraction.

**H6 | collector.rs:989 | DFA/NFA cache is in-memory only, no cross-process persistence**
`compiled_dfa_for_literals` uses `static DFA_CACHE: OnceLock<Mutex<HashMap<...>>>`. The cache dies with the process. Large rule sets recompile the same literal DFAs on every invocation.
Fix: Integrate with the G8 on-disk cache (or a separate `~/.cache/vyre/dfa/` store) keyed by blake3 of the sorted literal set.

**H7 | bitset_fixpoint.rs:131 | warm_start lacks GPU conform test with non-zero seed**
While the reference evaluator (`reference_eval_warm_start`) is tested, there is no conform harness that exercises `bitset_fixpoint_warm_start` against a real GPU backend with a non-zero seed. The convergence semantics depend on comparing `c0` (pre-OR) vs `next`, which is subtle and could regress in a backend lowering.
Fix: Add a conform harness test in `vyre-primitives/src/fixpoint/mod.rs` or the conform suite that dispatches `bitset_fixpoint_warm_start` with a non-zero seed and asserts correct convergence.

**H8 | watcher.rs:310 | compile_cached recomputes dependency fingerprints without memoization**
`dependency_fingerprint` does a DFS over the full dependency graph for every invalidated rule on every watch event. For large rule sets with shared dependencies, the same subgraph is traversed repeatedly.
Fix: Cache the transitive dependency fingerprint inside `ParsedRuleFile` and invalidate only when the file’s direct dependencies change.

**H9 | main.rs:258 | changed_files from git diff is not intersected with scan target**
`changed_files` returns every path from `git diff --name-only`. If the user passes a `--diff` range plus a `target_path` that is a subdirectory, files outside `target_path` are still passed to `scan_selected_paths`, potentially scanning files the user did not intend to cover.
Fix: Filter `changed_files` with `path.starts_with(&target_path)` before passing to `scan_selected_paths`.

### MEDIUM

**M1 | pipeline.rs:495 | G8 store leaves orphaned .bin.tmp files on crash**
`on_disk::store` writes to `{key}.bin.tmp` then renames to `{key}.bin`. If the process crashes between write and rename, the temp file is never cleaned up. Over time these accumulate in `~/.cache/vyre/pipelines/`.
Fix: On store (or on load-miss), scan the cache directory and delete `.bin.tmp` files whose mtime is older than the process start.

**M2 | watcher.rs:265 | rebuild_all clones every parsed document on every rule change**
`merged_document` clones every `surge::ast::Document` in the parsed map. For large rule sets this is an O(rules) memory copy on every edit. While not catastrophic, it is unnecessary.
Fix: Store parsed documents in `Arc<Document>` and build the merged document by reference.

**M3 | diff_scan.rs:198 | build_include_graph hardcodes C/C++ extensions despite docstring claims**
The extension filter is `Some("c") | Some("h") | Some("cpp") | Some("hpp")`. The module docstring claims production use wires through Python (`import`), Go (`import`), and Rust (`use` + `mod`) parsers. No such wiring exists.
Fix: Add language-specific import extractors or delete the false claim from the docstring.

---

## Competitor Comparison

- **Bazel / Buck2** incremental builds: Both use persistent, versioned action graphs stored on disk and invalidated by content hash. surgec’s depgraph is in-memory only and unexported.
- **clangd / rust-analyzer** watch mode: Both stream file events into an incremental index and only re-check affected translation units. surgec’s watcher rescans the entire tree.
- **YARA** scanner: Reuses compiled rule objects across file scans in a single batch. surgec creates a new `Collector` (and re-dispatches plans) per file.

---

## Remediation Priority

| Priority | Finding | Owner |
|---|---|---|
| P0 | C3, C4  -  Wire PersistentEngine into watch mode | runtime/scan |
| P0 | C5  -  Wire G8 on_disk into compile_shared | vyre-driver |
| P0 | C1, C2  -  Implement G10 GPU transitive closure | surgec/scan |
| P1 | H3, H4  -  Backend reuse + Collector batching | surgec/scan |
| P1 | C6  -  Decide if diff_replay should be vyre Program | surgec/scan |
| P1 | H6  -  DFA on-disk cache | vyre-libs/matching |
| P2 | H1, H2  -  Path canonicalization + symbolic edges | surgec/scan |
| P2 | H5  -  Robust include parsing | surgec/scan |
| P2 | M1  -  Temp file cleanup | vyre-driver |

---

*End of audit.*
