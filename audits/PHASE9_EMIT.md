# PHASE9_EMIT  -  Audit Report: Emit / Dedupe / SARIF / Confidence

**Date:** 2026-04-24  
**Scope:** `libs/performance/matching/vyre/vyre-libs/src/matching/hit_buffer.rs`, `libs/tools/surgec/src/scan/finding*`, `libs/scanner/secfinding`, all SARIF writers.  
**Auditor:** Kimi Code CLI (security researcher)  
**Standard:** SEVERITY | file:line | defect | fix

---

## Executive Summary

| Question | Answer | Finding IDs |
|---|---|---|
| 1. Is `compact_hits` a subgroup prefix-sum kernel or CPU-side compact-after-readback? | **Scalar GPU clamp** (`[1,1,1]` workgroup), not prefix-sum. Production path clamps on CPU after readback anyway. | EMIT-01, EMIT-02 |
| 2. Is dedupe a GPU hashset (CHD  -  G9) or CPU `BTreeSet`? | **CPU `BTreeSet` only in tests.** Production megakernel does not dedupe. Collector dedupes after readback with `sort`+`dedup`. | EMIT-03, EMIT-04, EMIT-05 |
| 3. Is SARIF encode a byte-level vyre Program or serde serialise on CPU? | **Pure CPU serde.** No GPU program. 18+ hand-rolled writers, no shared schema validation. | EMIT-06, EMIT-07, EMIT-08 |
| 4. Confidence scoring  -  bitset dot-product on GPU or f32 multiply on CPU? | **CPU `f32` only.** Computed after full readback. Hot-path constructs dummy `Finding` with cloned Strings. | EMIT-09, EMIT-10, EMIT-11 |
| 5. Is the emit path zero-copy (GPU→stdout) or readback-to-host first? | **Readback-to-host first.** `hit_ring.readback()` → CPU decode → CPU serde → stdout. | EMIT-12, EMIT-13 |
| 6. Suppression/exemption filter on GPU before readback or CPU after? | **CPU after.** All exemption, suppression, and confidence post-processing runs on materialized host vectors. | EMIT-14, EMIT-15, EMIT-16 |
| 7. Is hit_buffer overflow counter atomic-increment on GPU or post-readback? | **GPU atomic-increment is correct in the primitive, but the production dispatcher never reads it.** Silent drops go unreported. | EMIT-17, EMIT-18 |

---

## Findings

### EMIT-01 | CRITICAL
`libs/performance/matching/vyre/vyre-libs/src/matching/hit_buffer.rs:172`  
**Defect:** `compact_hits_with_layout` launches as a `[1, 1, 1]` scalar workgroup that performs a single `min(cursor, max_capacity, buffer_cap)` clamp. It is billed as a "compaction" helper but does zero parallel work, wasting a full GPU dispatch to write one `u32`. A CPU scalar clamp is ~1000× faster and avoids command-buffer overhead.  
**Fix:** Delete `compact_hits` and `compact_hits_with_layout`. Replace with a CPU-side `let live_len = cursor.min(capacity)` in the caller. If a true GPU-side prefix-sum compaction is required, implement a subgroup-wide parallel compact (copy-if with subgroup ballot + exclusive prefix sum) and benchmark against the CPU path.

### EMIT-02 | HIGH
`libs/performance/matching/vyre/vyre-runtime/src/megakernel/dispatcher.rs:256`  
**Defect:** The production `BatchDispatcher::dispatch` path never uses the `compact_hits` GPU primitive. Instead it clamps `hit_count` CPU-side after readback: `queue_state_words[queue_state_word::HIT_HEAD].min(batch.hit_capacity())`. This means `compact_hits` is dead code in production and the design documentation ("GPU-side hit-buffer append and compaction helpers") is misleading.  
**Fix:** Remove the dead `compact_hits` GPU program from the production build, or wire it into the megakernel so compaction actually happens on-device. Update doc comments to reflect the real architecture.

### EMIT-03 | CRITICAL
`libs/performance/matching/vyre/vyre-runtime/src/megakernel/dispatcher.rs:706-727`  
**Defect:** `decode_hits` blindly decodes every hit record from the readback buffer into a `Vec<HitRecord>` with no deduplication. If two rules match the same byte offset, or if a rule matches overlapping spans, the caller receives duplicate records. At internet scale this inflates downstream SARIF/JSON sizes and can trigger duplicate alerts in CI.  
**Fix:** Insert a GPU-side dedupe pass before readback (e.g. sort hits by `(file_idx, rule_idx, match_offset)` on device via bitonic sort, then unique-count with a subgroup ballot), or at minimum dedupe on the host immediately after `decode_hits` using a `HashSet<HitRecord>` before returning `BatchDispatchReport`.

### EMIT-04 | HIGH
`libs/tools/surgec/src/scan/collector.rs:870-875`  
**Defect:** `gpu_hits_for_signal` sorts and calls `hits.dedup()` on the CPU after the GPU has already returned the full hit list. For a file with millions of matches, this is O(n log n) host work that could have been avoided by deduplicating on-device during the matching kernel.  
**Fix:** Fold deduplication into the GPU hit-discovery kernel. Use a device-side hash table or sorted output + unique kernel so the host only sees deduplicated results.

### EMIT-05 | MEDIUM
`libs/tools/surgec/src/scan/collector.rs:1156-1198`  
**Defect:** `select_hits_for_dispatch` re-sorts and re-dedups hits that were already sorted/deduped at line 870. It then performs an O(n) `selected.contains(&hit)` scan (line 1183) inside a loop over tail candidates  -  quadratic in `cap` (256). While small, this pattern is unnecessary and violates LAW 4 (maximal elegance).  
**Fix:** Remove the redundant second sort/dedup. Replace `Vec::contains` with a `HashSet<Hit>` for the tail merge, or restructure so tail sampling is performed in a single pass without re-checking membership.

### EMIT-06 | CRITICAL
`libs/tools/surgec/src/output/sarif.rs:235-297`  
**Defect:** SARIF encoding is pure CPU `serde_json` serialization over heap-allocated Rust structs. For large scans (10⁶+ findings) this serializes one object at a time on a single thread, bottlenecking the entire pipeline. There is no streaming or GPU-accelerated token-to-bytes path.  
**Fix:** Implement a zero-allocation streaming SARIF serializer (e.g. `std::io::Write` wrapper that emits JSON tokens directly), or shard findings across `rayon` threads and concatenate sorted JSONL slices. Benchmark against `simd-json` or `sonic-rs` before custom-building.

### EMIT-07 | HIGH
`libs/scanner/secreport/src/render/json.rs:37-151`  
**Defect:** `render_sarif_generic` builds the entire SARIF document in memory using `serde_json::json!` macro allocations for every finding. It also emits a hard-coded `$schema` pointing to `master/Schemata/sarif-schema-2.1.0.json` (line 144) which is a moving GitHub target  -  schema drift will break downstream parsers.  
**Fix:** Emit the schema from a compile-time-embedded schema string (pin to a git SHA or vendor the schema). Switch to a streaming serializer so the peak heap size is O(1) in finding count.

### EMIT-08 | HIGH
`libs/performance/analysis/reportkit/src/renderers/sarif.rs:181-209`  
**Defect:** `render_sarif_writer` claims zero-copy design (uses `&'a str` lifetimes) but still delegates to `serde_json::to_writer`, which internally allocates a `serde_json::Serializer` buffer and performs per-field reflection. There are **18+ distinct hand-rolled SARIF writers** across the project (surgec, karyx, venin, soleno, keyhog, vulnir, sear, jsdet, standard/santh-conform, reportkit, etc.) with no shared schema validation or conformance test.  
**Fix:** Extract a single `sarif-core` crate with `#[derive(Serialize)]` structs validated against a vendored OASIS schema. Delete all duplicate struct definitions. Add a differential test that round-trips every writer through `sarif-sdk` validation.

### EMIT-09 | CRITICAL
`libs/tools/surgec/src/scan/confidence.rs:17-26`  
**Defect:** `compute_confidence` runs on the CPU after every GPU dispatch, using hardcoded magic numbers (`0.35`, `0.15`, `0.20`) with no explanation of their statistical origin. The `f32::clamp` at line 25 was acknowledged in an inline audit comment ("AUDIT_2026-04-24 F-CONF-01") as propagating NaN, yet the fix (`if !raw.is_finite() { return 0.0; }`) was already applied  -  but the comment was left behind as permanent debt. More importantly, confidence is computed **per-clause** after readback, meaning the GPU does zero work toward scoring.  
**Fix:** Move confidence scoring into the GPU dispatch kernel. Encode rule metadata (specificity, sanitizer flags) as device buffers and compute a bitset/dot-product score in parallel with the match. If that is architecturally infeasible, document the statistical derivation of every magic constant and add property-based tests that prove monotonicity.

### EMIT-10 | HIGH
`libs/tools/surgec/src/scan/dispatch.rs:270-280`  
**Defect:** `dispatch_single_clause` constructs a full `Finding` struct (cloning `rule_name`, `qualified_rule_name`, `primitive`  -  three `String` allocations) solely to pass it into `compute_confidence`. This happens on the hot path for **every clause that fires**, generating unnecessary heap pressure.  
**Fix:** Change `compute_confidence` to accept a `&CompiledRule` + `&[u32]` (byte offsets) instead of an owned `Finding`. Remove the dummy struct construction entirely.

### EMIT-11 | MEDIUM
`libs/tools/surgec/src/scan/confidence.rs:39-40`  
**Defect:** `apply_chain_confidence` calls `.clamp(0.0, 1.0)` on an `f32` sum without checking finiteness first. If `finding.finding.confidence` is NaN (e.g. from a prior corrupted deserialize), `clamp` returns NaN and the NaN propagates into SARIF output, poisoning severity filters downstream.  
**Fix:** Add an explicit `is_finite()` guard before clamp, identical to the one in `compute_confidence`: `if !sum.is_finite() { finding.finding.confidence = 0.0; } else { finding.finding.confidence = sum.clamp(0.0, 1.0); }`.

### EMIT-12 | CRITICAL
`libs/performance/matching/vyre/vyre-runtime/src/megakernel/dispatcher.rs:259-266`  
**Defect:** The emit path is **not** zero-copy. `batch.hit_ring().readback(device, queue, &mut hit_bytes)` copies the entire sparse hit ring from GPU memory to a host `Vec<u8>`, which is then cast to `&[u32]` and decoded into `Vec<HitRecord>`. For large batches this is a full PCIe round-trip plus two host allocations before any filtering or formatting occurs.  
**Fix:** Implement a device-side filter/dedupe/compaction stage so only live hits are read back. If the backend supports mapped buffers (e.g. `wgpu::BufferUsages::MAP_READ`), map the hit ring directly into host address space without an explicit `readback` copy.

### EMIT-13 | HIGH
`libs/tools/surgec/src/main.rs:530-537`  
**Defect:** After the GPU readback, findings are fully materialized on the host, wrapped in SARIF structs, serialized to JSON, and only then written to stdout. There is no streaming path; the entire output buffer must fit in host RAM before the first byte reaches the consumer.  
**Fix:** Stream SARIF output directly to `std::io::stdout().lock()` using a custom `std::io::Write` implementation that emits JSON tokens incrementally, rather than `serde_json::to_string_pretty` which builds the whole string in memory.

### EMIT-14 | CRITICAL
`libs/tools/surgec/src/scan/exemptions.rs:89-110`  
**Defect:** `apply_exemptions` filters findings on the CPU **after** all GPU work, readback, decoding, and `Finding` construction is complete. Exemptions that suppress 99% of results still pay the full GPU→host→serde tax for the discarded 99%. At internet scale this is wasted energy and latency.  
**Fix:** Push exemption criteria into the GPU kernel where possible. Compile rule-name globs and path globs into a device-side bloom filter or hash table. Skip hit emission for exempted rules before the atomic add on `hit_slot`.

### EMIT-15 | HIGH
`libs/tools/surgec/src/scan/auto_suppress.rs:28-57`  
**Defect:** `propose_suppressions` is a pure CPU post-scan pass that counts result slots per `(rule, file)` pair. It cannot react to noise until the entire scan is finished and all findings are resident in host memory. For a 1M-file corpus this means gigabytes of findings may accumulate before suppression is even evaluated.  
**Fix:** Maintain a running suppression tally during per-file scan loops. If a `(rule, file)` pair crosses the threshold mid-scan, emit the proposal immediately and short-circuit further dispatches for that pair.

### EMIT-16 | HIGH
`libs/scanner/secfinding/src/filter.rs:124-203`  
**Defect:** `FindingFilter::filter` is applied after the full scan pipeline on materialized `Finding` objects. The filter supports `min_confidence`, but confidence is already computed post-readback (EMIT-09). There is no GPU-side pre-filtering for severity, tags, or confidence, so suppressed findings still traverse PCIe and host memory.  
**Fix:** Add a GPU-side predicate kernel that evaluates `min_confidence`, `min_severity`, and tag inclusion as a bitset intersection before hit emission. Only pass through hits that survive the predicate.

### EMIT-17 | HIGH
`libs/performance/matching/vyre/vyre-libs/src/matching/hit_buffer.rs:101`  
**Defect:** The `HIT_BUFFER_OVERFLOW_COUNT` is incremented atomically on the GPU when `slot >= max_capacity`, which is correct. However, the atomic uses `Expr::u32(0)` as the byte offset, assuming the buffer is exactly one `u32`. If the buffer layout ever changes (e.g. multi-word counter for 64-bit saturation), this offset is a latent foot-gun with no static assertion.  
**Fix:** Add a `const_assert` that `HIT_BUFFER_OVERFLOW_COUNT` has `count == 1` and `DataType::U32`, and replace the magic `Expr::u32(0)` with a named constant `OVERFLOW_COUNTER_WORD_OFFSET`.

### EMIT-18 | CRITICAL
`libs/performance/matching/vyre/vyre-runtime/src/megakernel/dispatcher.rs:256`  
**Defect:** While `hit_buffer.rs` correctly counts overflows on the GPU, `BatchDispatcher::dispatch` never reads `HIT_BUFFER_OVERFLOW_COUNT`. The only truncation signal is `hit_count = min(HIT_HEAD, hit_capacity)`, which silently discards excess hits. Operators have no visibility into how many hits were lost, leading to false negatives in security scans.  
**Fix:** After the hit-ring readback, read back the overflow counter (or embed it in `queue_state`) and surface it in `BatchDispatchReport` as `overflow_count: u32`. Propagate this field through `ScanReport` so the CLI can emit a diagnostic warning when hits are dropped.

### EMIT-19 | MEDIUM
`libs/tools/surgec/src/scan/dispatch.rs:362`  
**Defect:** `decode_result_slots` allocates `Vec::with_capacity(bytes.len() / 8)` but then pushes **every non-zero slot**. If all slots are hits, the actual need is `bytes.len() / 4`, so the vector reallocates at 50% capacity  -  a guaranteed wasted allocation on the hot path.  
**Fix:** Use `bytes.len() / 4` as the capacity hint, or use `bytes.len() / 4` with a `retain`-style scan that counts non-zero elements first.

### EMIT-20 | MEDIUM
`libs/tools/surgec/src/output/sarif.rs:398-400`  
**Defect:** `finding_region` hardcodes `byte_length: 1` for every finding, regardless of the actual match length. This produces incorrect SARIF regions for multi-byte matches (e.g. a 16-byte literal match is reported as 1 byte), breaking byte-accurate navigation in GitHub Code Scanning and VS Code SARIF Viewer.  
**Fix:** Use the actual match length from `finding.finding.byte_offsets` (or `result_slots` resolved length) instead of hardcoding `1`. If lengths are unavailable, compute `max_offset - min_offset + 1` from `byte_offsets`.

### EMIT-21 | MEDIUM
`libs/performance/analysis/reportkit/src/renderers/sarif.rs:161`  
**Defect:** `byte_length` is computed as `finding.end.saturating_sub(finding.start)`, which silently emits `0` for zero-length or reversed regions. A zero-length region in SARIF is technically valid but usually indicates a parser bug; downstream viewers may skip rendering it.  
**Fix:** Assert that `end >= start` during `ReportFinding` construction. If `end == start`, emit `byte_length: 1` so the region is still visible, or skip the region field entirely and log a structured warning.

### EMIT-22 | LOW
`libs/tools/surgec/src/scan/confidence.rs:131`  
**Defect:** `looks_like_sanitizer` allocates a new `String` via `to_ascii_lowercase()` on every call. This function is called repeatedly inside `compute_confidence` and `chain_has_sanitizer_bypass` loops. For large result sets this is unnecessary heap churn.  
**Fix:** Use `text.eq_ignore_ascii_case("sanitiz")` or a byte-level `memmem` search on the original slice instead of allocating a lowercase copy.

---

## Cross-Reference Matrix

| Area | File(s) | Findings |
|---|---|---|
| `compact_hits` | `hit_buffer.rs`, `dispatcher.rs` | EMIT-01, EMIT-02 |
| Dedupe | `dispatcher.rs`, `collector.rs`, `exploit_graph.rs` | EMIT-03, EMIT-04, EMIT-05 |
| SARIF encode | `output/sarif.rs`, `secreport/json.rs`, `reportkit/sarif.rs` | EMIT-06, EMIT-07, EMIT-08, EMIT-20, EMIT-21 |
| Confidence | `confidence.rs`, `dispatch.rs` | EMIT-09, EMIT-10, EMIT-11, EMIT-22 |
| Emit path | `dispatcher.rs`, `main.rs` | EMIT-12, EMIT-13 |
| Suppression | `exemptions.rs`, `auto_suppress.rs`, `filter.rs` | EMIT-14, EMIT-15, EMIT-16 |
| Overflow counter | `hit_buffer.rs`, `dispatcher.rs` | EMIT-17, EMIT-18 |
| Misc hot-path | `dispatch.rs` | EMIT-19 |

---

## Recommended Priority Order

1. **EMIT-18**  -  Silent hit drops are false negatives in security scans; fix first.
2. **EMIT-14 / EMIT-16**  -  Move exemption/filter predicates to GPU to eliminate wasted PCIe traffic.
3. **EMIT-06 / EMIT-08**  -  Unify SARIF writers and add streaming serialization.
4. **EMIT-03 / EMIT-04**  -  GPU-side dedupe to remove O(n log n) host bottlenecks.
5. **EMIT-09 / EMIT-10**  -  Compute confidence on-device or at least remove allocations.
6. **EMIT-12 / EMIT-13**  -  Zero-copy / streaming emit path.
7. **EMIT-20**  -  Fix hardcoded `byte_length: 1` for correct SARIF geometry.

---

*End of PHASE9_EMIT audit.*
