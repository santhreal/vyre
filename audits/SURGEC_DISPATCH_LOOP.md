# SURGEC_DISPATCH_LOOP  -  Audit Report

**Scope:** `libs/tools/surgec/src/scan/{dispatch.rs, collector.rs, filter.rs, provenance.rs}` and orchestrator glue.  
**Date:** 2026-04-24  
**Auditor:** Kimi Code CLI (security researcher)  
**Focus:** How surgec invokes vyre, loop structure, fusion, readback, pipelining, decoder chains, exemption filtering, and PersistentEngine integration.

---

## Executive Summary

The SURGEC scan dispatch loop is **deeply nested, entirely serial, and dispatch-heavy**.  
For every file it walks every decoded layer, for every layer it walks every clause dispatch plan, for every plan it runs per-signal GPU hit-discovery (possibly multiple dispatches), and finally dispatches the rule clause itself.  
**No file-level pipelining, no rule fusion on the hot path, no PersistentEngine / megakernel enqueue, and no GPU-side exemption pre-filter.**  The fusion infrastructure (`fuse.rs`) exists but is **not wired into the collector**.  At internet scale (millions of files, thousands of rules) this is a wall-time and energy catastrophe.

---

## Findings

### F-01 | CRITICAL | collector.rs:203-269
**Multiple nested loops with no pipelining  -  file→layer→plan→signal→clause.**

`scan_gpu_with_context` iterates `WalkDir` entries one-at-a-time, reads the full file into RAM, then calls `scan_collected_file`.  Inside that, `for layer in layers { for plan in dispatch_plans { … } }`.  Inside `dispatch_rules`, `for rule { for clause { … } }`.  The GPU sits idle while the next file is read and decoded; the CPU sits idle while the GPU executes.  There is no double-buffering, no async file I/O, and no `dispatch_async` usage.

**Suggested fix:** Restructure into a producer/consumer pipeline.  Producer thread walks the filesystem and enqueues `(file_bytes, layers)` into a bounded channel.  Consumer (GPU thread) pulls batches and calls `dispatch_async` (or `StreamingDispatch`) so file N+1 is staged while file N executes.  See `vyre-driver-wgpu/src/engine/streaming.rs` for the existing chunk-stream primitive that is already built but never called from surgec.

---

### F-02 | CRITICAL | collector.rs:609-666 & dispatch.rs:96-149
**Every clause gets its own fake `CompiledDocument` and independent `dispatch_rules` call  -  rules are NOT fused.**

`rule_dispatch_plans` fabricates a `CompiledDocument` containing exactly one rule and one clause.  `scan_collected_file` then calls `dispatch_rules` on that micro-document.  The megakernel fusion path (`compile/fuse.rs`) that can pack N rules into a single GPU kernel with opcode dispatch is **not used** by the collector.  The result: `N_files × N_clauses` kernel launches instead of `N_files × 1` megakernel launches.

**Suggested fix:** Replace the per-plan dispatch path with `FusionPlan::optimize` → `CompiledPipeline` reuse.  Build one fused program per `CompiledDocument` at scan start, cache the `Arc<dyn CompiledPipeline>` via `VyreBackend::compile_native`, and dispatch all applicable rules for a file in a single kernel launch (or a small number of batched persistent-kernel slots).

---

### F-03 | HIGH | collector.rs:367-388 & decode.rs:162-178
**Decoder chains (base64 → scan, hex → scan, inflate → scan) are CPU-decoded in separate layers, each scanned independently.**

`scan_inputs_for_file` pushes the raw bytes as layer 0, then appends every decoded layer produced by `decode_layers`.  Each layer is scanned in isolation.  The fused `base64_decode_then_aho_corasick` program in `vyre-libs::decode` (exposed via `fuse_base64_decode_scan_program`) that would decode **on-GPU** and scan the decoded buffer without host readback is never used.

**Suggested fix:** For supported encodings (base64, hex, deflate), detect the encoding type at compile time and route the slot through the fused GPU decode→scan path.  Keep the CPU recursive decoder as a fallback for archive formats (zip, tar) that cannot be expressed as a single GPU kernel.

---

### F-04 | HIGH | collector.rs:703-808
**Per-signal GPU hit-discovery dispatches a separate kernel for every signal string_id.**

`build_clause_inputs` groups patterns by `string_id`, then calls `gpu_hits_for_signal` for each group.  Inside that, literal families dispatch `aho_corasick` (a full GPU kernel), and single literals dispatch `substring_search` (another kernel).  If a clause has 50 signals, that is 50 GPU dispatches **before** the clause evaluation dispatch.  The DFA cache (`compiled_dfa_for_literals`) is global-static with a `Mutex`, so concurrent scans contend.

**Suggested fix:** Batch all literal patterns for a clause into one multi-pattern DFA dispatch, or use the megakernel path where hit-discovery and clause evaluation are fused into a single persistent kernel.  Replace the `static DFA_CACHE: OnceLock<Mutex<…>>` with a `DashMap` or shard the cache by thread ID to eliminate lock contention.

[PARTIAL 2026-04-24] The DFA-cache half of this finding is closed by F-12  -  `compiled_dfa_for_literals` now uses `OnceLock<DashMap<…>>` (sharded, lock-free reads). Per-signal kernel batching (the "50 dispatches per clause" half) remains open  -  it requires a multi-pattern DFA path or fused megakernel hit-discovery, both bigger refactors than belong in this audit pass.

---

### F-05 | HIGH | dispatch.rs:221-222
**Workgroup-size tuning clones the entire `Program` on every clause dispatch.**

`dispatch_single_clause` does `let mut tuned: Program = program.clone();` then mutates `workgroup_size`.  While the `entry` `Arc<Vec<Node>>` is shared, the `Program` struct itself (buffers, regions, metadata) is heap-allocated and cloned on every single clause of every file.  For a 1000-rule document scanned over 10⁵ files this is 10⁸ needless allocations.

**Suggested fix:** Move workgroup sizing to compile time.  Store the tuned workgroup size in the `ClauseDispatchPlan` (or in the `CompiledPipeline` cache key) so the hot path only passes a `&Program` reference.  If dynamic sizing is required, compute it once per program and store in an `Arc<AtomicU32>`.

---

### F-06 | HIGH | collector.rs:173-271
**File scan is strictly serial: open → read → decode → dispatch, with no overlap.**

The loop body in `scan_gpu_with_context` reads bytes, builds layers, builds clause inputs (which may itself dispatch GPU kernels), then dispatches the clause, all before moving to the next file.  The `StreamingDispatch` primitive in `vyre-driver-wgpu` (which keeps one chunk in-flight while staging the next) is not used.  The `dispatch_async` trait method exists but is never called from surgec.

**Suggested fix:** Introduce a `scan_gpu_streaming` path that uses `StreamingDispatch` or `dispatch_async` + `PendingDispatch::await_result`.  Overlap file I/O (via `tokio::fs::read` or `io_uring`) with GPU execution so the bus is never idle in both directions simultaneously.

---

### F-07 | MEDIUM | exemptions.rs:85-110
**Exemption filter is post-dispatch CPU-only, not pre-dispatch GPU-masked.**

`apply_exemptions` walks every `EvaluatedRule` (a finding) and tests glob/path/hash/severity on the host **after** the GPU has already executed the rule and read back the result buffer.  For a corpus where 90 % of findings are suppressed by `vendor/**` path exemptions, the GPU wasted 90 % of its cycles.

**Suggested fix:** Move the exemption mask to pre-dispatch.  At compile time, build a GPU-visible bitmask of rules that are exempted per path glob or content-hash prefix.  Before enqueueing a rule slot into the megakernel ring, mask it out on the host.  For hash-based exemptions (which require full file content), keep a CPU fast-path that hashes the first 4 KiB and skips dispatch if the hash matches a bloom-filter of exempted hashes.

---

### F-08 | MEDIUM | record_and_readback.rs:354-378  [STALE  -  file moved/restructured 2026-04-24]
**Readback is synchronous blocking (`map_async` + `device.poll`) with no host-side concurrency between files.**

The wgpu backend batches all `map_async` calls for one dispatch before a single `device.poll(Maintain::wait_for(submission))`.  While this is efficient **within** one dispatch, surgec does not overlap the poll wait with work from the next file.  The host thread burns wall time waiting for the GPU fence.

**Suggested fix:** Use the `dispatch_async` path to fire dispatches for file N+1 while awaiting the readback of file N.  The wgpu backend’s `PendingDispatch` default is a trivial ready-handle, but the backend already supports true async via `StreamingDispatch`; surgec simply needs to call it.

[STALE 2026-04-24] File `record_and_readback.rs` no longer exists in `libs/tools/surgec/src/scan/`. The recording + readback path was restructured during the F-SPEED-3 megakernel batching landing. The async-overlap concern is now tracked under the streaming dispatch lane (`scan_gpu_with_context`)  -  re-audit needed against the current code, not the stale path.

---

### F-09 | MEDIUM | collector.rs:1246-1284
**`pack_bytes_as_u32_words` expands file bytes 4× on the CPU before every GPU literal scan.**

`build_clause_inputs` packs the raw file bytes into `Vec<u8>` of LE u32 words (4 bytes per input byte).  A 128 MiB file becomes 512 MiB of packed haystack.  This allocation happens per file, per layer, and is not reused.  The GPU substring kernel then reads one u32 per byte, wasting 75 % of memory bandwidth.

**Suggested fix:** Change the substring / DFA kernels to consume `array<u8>` directly (WGSL supports `array<u8>` in storage buffers).  If the kernel ABI requires u32 alignment, map the byte buffer as `array<u32>` with a length uniform and read individual bytes via bit-shifts inside the shader.  Eliminate the 4× host-side expansion entirely.

---

### F-10 | MEDIUM | collector.rs:751-753
**`counts`/`offsets`/`lengths` buffers are allocated at max signal-slot size regardless of actual signal count.**

`slot_count` is derived from `max_id + 1`.  If a clause uses string_ids `{0, 10000}`, the allocation is `10001 × MAX_CACHED_POSITIONS × 4` bytes (~800 KiB) even though only 2 signals are active.  For rules with sparse high-id signals this wastes GPU memory and host→device copy bandwidth.

**Suggested fix:** Dense-pack the signal ids at compile time (the `signal_registry` already exists for this purpose).  If sparse ids are intentional, switch to an indirection table: a small `signal_index → slot_index` mapping buffer so the GPU buffers are sized to the actual signal count.

---

### F-11 | MEDIUM | provenance.rs:51-56  [STALE  -  verified 2026-04-24]
**Rule source hash uses `format!("{rule:#?}")` which is not stable across Rust compiler versions.**

`canonical_rule_source_hash` hashes the debug representation of the AST `Rule`.  `Debug` output for structs is not guaranteed stable across `rustc` versions, feature flags, or `surge` crate refactors.  A minor compiler upgrade would change every hash, breaking diff replay and exemption pinning.

**Suggested fix:** Hash a canonical wire-format or TOML serialization of the rule instead.  Use the same stable byte sequence that ` CompiledDocument::encode` produces, or define a `CanonicalSerialize` trait with a versioned schema.

[CLOSED 2026-04-24] Already fixed. `canonical_rule_source_hash` (provenance.rs:51-58) hashes `postcard::to_allocvec(rule)` with a versioned domain tag `b"surge-rule-v3-postcard"` and the artifact scope. Postcard is a stable, schema-pinned wire format  -  unaffected by `Debug` formatting drift. Tests `same_rule_same_provenance` / `different_rule_different_provenance` cover stability + sensitivity.

---

### F-12 | MEDIUM | collector.rs:989-1041  [CLOSED 2026-04-24]
**Global DFA cache is a single `Mutex<HashMap>`  -  contention and poison risk at scale.**

`compiled_dfa_for_literals` uses `static DFA_CACHE: OnceLock<Mutex<HashMap<…>>>`.  Every file scan that hits a new literal set contends on one global mutex.  The poison-recovery logic (lines 1014–1019) is commendable but proves the design is fragile: a panic during rehash tears the map, and the fix is to drop the whole cache.

**Suggested fix:** Replace with `dashmap::DashMap` or `moka` for lock-free reads.  Shard the cache by BLAKE3 prefix so concurrent scans on different rule sets do not collide.  Alternatively, hoist DFA compilation to `Collector` construction time so the cache is per-scan and needs no global synchronization.

[CLOSED 2026-04-24] Switched the `OnceLock<Mutex<HashMap<…>>>` to `OnceLock<DashMap<…>>`. DashMap shards by hash, so concurrent scans on different rule sets contend on separate shards. `entry().or_insert_with` handles the compile-once-on-miss race and the poison-recovery `std::mem::replace` dance is gone  -  DashMap does not propagate panic poison across shards. Regression tests `dfa_cache_returns_shared_arc_for_identical_literal_set` + `dfa_cache_separates_different_literal_sets` cover the contract.

---

### F-13 | MEDIUM | filter.rs:139-171
**`filter_document` drains and `shrink_to_fit`s in place, causing O(N²) behavior for large rule sets.**

`filter_document` calls `document.rules.drain(..)` then `kept.shrink_to_fit()` for both top-level and artifact rules.  For a document with millions of rules (internet-scale registries), `shrink_to_fit` may reallocate and copy the surviving vector, and it is done twice (rules + artifact rules).

**Suggested fix:** Build a new `Vec` with `with_capacity(document.rules.len())`, push survivors, and swap it in.  Do not `shrink_to_fit` on the hot path; the memory savings are negligible compared to the compiled `CompiledDocument` size, and the reallocation cost is unpredictable.

---

### F-14 | LOW | dispatch.rs:323-354
**`optimal_workgroup_size` heuristic is naive: only counts `Node::Let`, not actual register pressure.**

The heuristic assumes register pressure is proportional to the count of `Let` bindings (≥24 → 64 wg, else 256).  It does not account for `Expr` complexity (e.g., nested `SubgroupShuffle`, `Atomic`, or `Call` nodes that consume far more registers than a simple `Let`).  A clause with 10 `Let`s each binding a `SubgroupBallot` may spill while a clause with 30 `Let`s of literal constants fits easily.

**Suggested fix:** Extend the heuristic with a weighted register-pressure estimator that traverses the entry nodes and assigns register cost per `Expr` variant.  Better yet, query the backend’s actual pipeline compilation for register usage (wgpu does not expose this directly, but Naga could be extended to report it).

---

### F-15 | LOW | collector.rs:1156-1198
**`select_hits_for_dispatch` heuristic silently drops hits without audit logging.**

When hits exceed `MAX_CACHED_POSITIONS` (4096), the function subsamples front-loaded and tail hits.  There is no warning or metric emitted when truncation occurs.  At internet scale, a densely-patterned file (e.g., minified JS with 10⁴ encoded blobs) could lose the one true-positive hit that sits in the middle of the distribution.

**Suggested fix:** Emit a rate-limited diagnostic (via `scan_skip_note!`) whenever `hits.len() > MAX_CACHED_POSITIONS`, naming the rule, file, and number of dropped hits.  Consider a two-pass dispatch: first pass with capped hits, and if any slot fires, re-run with the full hit list for that slot.

---

### F-16 | LOW | collector.rs:752-753
**`MAX_CACHED_POSITIONS` is a hardcoded constant with no configurability.**

`const MAX_CACHED_POSITIONS: usize` is imported from `crate::index` and used directly.  There is no `DecodeConfig`-style override.  A user scanning firmware blobs with dense repetitive patterns cannot raise the cap without editing source and recompiling.

**Suggested fix:** Add `max_cached_positions` to `DecodeConfig` (or a new `ScanConfig`) and thread it through `Collector::with_scan_config`.  Validate the cap at scan start against host memory limits.

---

### F-17 | LOW | exemptions.rs:211-228  [CLOSED 2026-04-24]
**`date_string_ge` falls back to lexicographic compare for malformed dates, silently changing behavior.**

If an exemption author writes `"2026-1-1"` instead of `"2026-01-01"`, the parser returns `None` and the code falls through to `a >= b`.  This happens to work for some strings and fail for others, but the silent fallback means the exemption may expire at the wrong time without any warning.

**Suggested fix:** Treat malformed `expires` as a compile-time error in `compile_exemptions`.  Do not allow exemptions with ambiguous dates to enter the runtime filter at all.  Log a structured warning with the exemption ID and the invalid date string.

[CLOSED 2026-04-24] `compile_exemptions` now strictly validates `expires` upfront via `parse_iso_date` (exactly `YYYY-MM-DD`, 10 ASCII chars, valid month/day ranges). Rejection error names the offending exemption id and the bad string. Three regression tests cover the happy path + the audit's literal `"2026-1-1"` example + a garbage string. Runtime `date_string_ge` is unchanged since malformed `today` can only happen via API misuse (compile-time-rejected `expires` can never reach it).

---

### F-18 | LOW | dispatch.rs:186-199
**Per-clause error recovery swallows the `Err` variant without structured logging.**

When `dispatch_single_clause` fails, the error is formatted into a string and printed via `scan_skip_note!`.  The caller (`scan_collected_file`) never sees the structured error, so metrics pipelines cannot count "how many clause dispatches failed per rule" or "which backend error codes are trending."

**Suggested fix:** Return a `ScanReport` that includes a `Vec<RuleError>` with rule name, clause index, file path, and structured error code.  Let the CLI decide whether to print, log, or emit SARIF for skipped clauses.

---

## Architecture Diagram (Simplified)

```text
WalkDir (serial, one file at a time)
  └── read_bytes(path) → Vec<u8>  [CPU]
        └── scan_inputs_for_file
              ├── Layer 0: raw bytes
              ├── Layer 1: base64 decoded
              ├── Layer 2: gzip decoded
              └── …
                    └── for each layer:
                          └── for each ClauseDispatchPlan:
                                ├── applicability.matches()?  [CPU]
                                ├── build_clause_inputs
                                │     └── for each signal:
                                │           ├── gpu_hits_for_literal  [GPU dispatch]
                                │           └── gpu_hits_for_literal_family  [GPU dispatch]
                                └── dispatch_rules(backend, &plan.document, &inputs)
                                      └── for each rule (1):
                                            └── for each clause (1):
                                                  ├── optimal_workgroup_size  [CPU]
                                                  ├── program.clone()  [CPU alloc]
                                                  ├── backend.dispatch_borrowed  [GPU dispatch]
                                                  │     ├── compile/validate  [CPU]
                                                  │     ├── record_and_readback  [GPU]
                                                  │     │     ├── map_async (batched)
                                                  │     │     ├── device.poll (blocking)
                                                  │     │     └── get_mapped_range
                                                  │     └── return Vec<u8>
                                                  └── decode_result_slots  [CPU]
```

**Key observation:** There are **5 levels of nested iteration** (file → layer → plan → signal → clause) and **no persistent kernel or batching** across any of them.

---

## Competitor Comparison

| System | Loop Structure | Rule Fusion | Decode→Scan | Exemption Filter | Readback |
|--------|---------------|-------------|-------------|------------------|----------|
| **Surgesc (current)** | Serial, nested 5-deep | Not used on hot path | CPU decode, then separate GPU scan per layer | Post-dispatch CPU | `map_async` + blocking poll per dispatch |
| **YARA (AV industry)** | File-at-a-time, but rules are Aho-Corasick-merged into one DFA | Yes  -  all strings in one automaton | N/A (no decode chain) | Pre-scan module mask | N/A (CPU) |
| **Semgrep** | File-at-a-time, AST-based | No (separate rule engine) | N/A | Pre-filter on path/lang | N/A (CPU) |
| **Hyperscan** | Stream or block mode, all patterns in one compiled database | Yes  -  multi-pattern NFA/DFA | N/A | N/A (library, not scanner) | N/A (CPU) |
| **CodeQL** | Database → query evaluation | Queries are compiled but not fused across queries | N/A | Post-query filter | N/A (CPU) |

Surgesc’s vyre backend is theoretically capable of fusing and persistent dispatch (the `FusionPlan` and `Megakernel` exist), but the collector layer does not use these capabilities.  At scale, this makes surgesc slower than a well-tuned YARA or Hyperscan deployment despite having a GPU.

---

## Recommendations Summary

1. **Fuse rules once, dispatch once per file.** Wire `FusionPlan::optimize` + `compile_native` into `Collector::scan_gpu`.  Cache the `Arc<dyn CompiledPipeline>` for the document lifetime.
2. **Batch files and overlap I/O with GPU.** Use `StreamingDispatch` or `dispatch_async` so file N+1 is read while file N executes.
3. **Move hit-discovery into the fused kernel.** Eliminate the per-signal `aho_corasick` / `substring_search` dispatches by compiling pattern DFAs into the same megakernel as the clause body.
4. **GPU-side decode→scan fusion.** For base64/hex/deflate layers, use `fuse_base64_decode_scan_program` instead of CPU decoding.
5. **Pre-dispatch exemption mask.** Build a per-file bitmask from path globs and bloom-filtered hashes before enqueueing any GPU work.
6. **Stable rule hashing.** Replace `Debug`-based `canonical_rule_source_hash` with a canonical wire-format hash.
7. **Lock-free DFA cache.** Replace the global `Mutex<HashMap>` with `DashMap` or per-`Collector` caches.

---

*End of audit.*
