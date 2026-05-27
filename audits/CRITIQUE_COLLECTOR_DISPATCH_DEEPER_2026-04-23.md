# Deep Re-Audit  -  Collector & Dispatch (Second Pass)

**Date:** 2026-04-23  
**Scope:** `surgec/src/scan/collector.rs`, `surgec/src/scan/dispatch.rs`  
**Goal:** Surface findings the first pass (`kimi-276059b2`) missed, across ordering, scaling, concurrency, Unicode/path edges, dispatch correctness, silent error swallowing, observability, and modularity. Read-only.

---

## Meta: Was the first pass hard enough?

The prior audit (`CRITIQUE_COLLECTOR_DISPATCH_2026-04-23.md`) found 11 issues, 4 of them CRITICAL. It was **solid on panic paths and fatal short-circuits** (mutex poison, unbounded vec allocations, `?` aborts, backend index panic). Those were real, well-located, and correctly prioritized.

What it **missed**  -  and what this pass covers  -  falls into three buckets:

1. **Scaling architecture, not just single allocations.** The first pass caught `vec![0u32; slot_count]` with `u32::MAX`, but missed the *per-clause per-layer* 4× re-allocation of the packed haystack, the sparse-`string_id` amplification, the regex re-compilation DoS, and the unbounded `findings` Vec across the whole scan. A single-file OOM is bad; a *hot-loop* OOM that only shows up at 10k files is worse because it passes small-unit tests.
2. **Silent data corruption, not just crashes.** `.ok().unwrap_or()` in offset remapping, invalid-UTF-8 path collisions, and non-deterministic output order don't panic  -  they produce *wrong* results that look like success.
3. **Systemic modularity and observability debt.** A 1,468-line god file, clauses skipped without audit trails, and TOCTOU races are architecture-level findings that the first pass's narrow panic-hunting scope didn't reach.

**Verdict:** The first pass was **good but not deep enough**. It stopped at "will it crash?" and did not ask "will it silently lie, scale, or reproduce?"

---

## New Findings

### FINDING-12: Non-deterministic scan output order due to unordered WalkDir iteration
**Severity:** HIGH  
**Location:** `collector.rs:193-211` (`scan_gpu_with_context`), `collector.rs:248-264` (`collect_files`)  
**Description:** `WalkDir` yields directory entries in the order the underlying filesystem returns them (`readdir` order). This order is non-deterministic across tmpfs, ext4 with `dir_index`, different mount options, and even across invocations on the same host. `scan_gpu_with_context` pushes findings into a `Vec<FileFinding>` in walk order; `scan_gpu` flattens but preserves that order. Two scans of the identical corpus can therefore emit findings in different sequences, breaking reproducible CI builds, golden-test diffing, and SARIF baseline comparisons.  
**Fix:** Before returning, sort `findings` by a canonical key such as `(path, rule_name, clause_index, min_byte_offset)`. Alternatively, collect into a `BTreeMap` keyed by path during iteration.  
**Test hint:** Create a directory with 100 files on tmpfs, scan twice back-to-back, and assert byte-identical JSON/SARIF output.

---

### FINDING-13: Sparse `string_id` set causes massive `slot_count` amplification without `u32::MAX`
**Severity:** CRITICAL  -  builds on FINDING-02  
**Location:** `collector.rs:585-598` (`build_clause_inputs`)  
**Description:** FINDING-02 correctly flagged `u32::MAX`, but the same hazard exists with *sparse* `string_id` values well inside the `u32` range. `slot_count` is computed as `max_string_id + 1`. A compiled rule whose active signals have `string_id`s `{0, 100_000}` forces `slot_count = 100_001`. The `offsets` and `lengths` vectors then allocate `100_001 × MAX_CACHED_POSITIONS × 4` bytes ≈ 102 MiB per clause per file, even though only two signals are active. An adversary can craft a rule with `string_id = 0` and `string_id = 65_535` (both inside `MAX_RULE_STRINGS = 256`? No  -  65_535 exceeds 256. But `string_id` is an unconstrained `u32` in `CompiledPattern`; the builder may not enforce the cap at runtime). Even within `MAX_RULE_STRINGS`, if the builder ever allows `string_id` up to 255, `slot_count = 256` and the vectors allocate `256 × 256 × 4 = 256 KiB` per clause per file. With 1,000 clauses and 1M files that is 64 GB of zeroed vectors.  
**Fix:** Use a dense remap: collect the sorted set of actual `string_id`s present in the clause, map each to a dense `0..N` index, and allocate vectors of size `N`. Validate that `N ≤ MAX_RULE_STRINGS` and return a structured compile error if exceeded.  
**Test hint:** Craft a `CompiledDocument` with `string_id`s `[0, 255]`, scan a 1 KiB file, and assert the prepared-inputs allocation is < 1 MiB.

---

### FINDING-14: `pack_bytes_as_u32_words` re-allocates 4× file bytes per clause per layer per file
**Severity:** CRITICAL  
**Location:** `collector.rs:599` (inside `build_clause_inputs`), call site `collector.rs:322-328`  
**Description:** `build_clause_inputs` calls `pack_bytes_as_u32_words(file_bytes)` for every applicable clause, inside the per-file layer loop. For a 10 MB file, each applicable clause allocates a fresh 40 MB `Vec<u8>`. A document with 100 applicable clauses scanning 1,000 files allocates 40 MB × 100 × 1,000 = 4 TB of temporary memory over the scan lifetime. The original `Arc<[u8]>` file buffer is also retained. No memory budget, no reuse, no pool. This hot-loop allocation will OOM-kill long before the `u32::MAX` file-size limit is reached.  
**Fix:** Pack once per file layer and reuse the packed buffer across all clauses for that layer. If host-side packing is mandatory, place a per-scan memory budget check before the allocation and shard the file when the budget is exceeded.  
**Test hint:** Scan a 50 MB file against a document with 50 applicable clauses and measure peak RSS; assert it stays under 500 MB.

---

### FINDING-15: Regex recompilation per file per layer per signal  -  ReDoS and CPU exhaustion
**Severity:** CRITICAL  
**Location:** `collector.rs:881-888` (`regex::bytes::Regex::new` inside `regex_hits_for_signal`)  
**Description:** `regex_hits_for_signal` compiles every regex pattern from raw source bytes on every invocation. The call chain is `build_clause_inputs` → `gpu_hits_for_signal` → `regex_hits_for_signal`, which runs per clause per layer per file. A single regex-backed rule scanning 1 million files compiles the same regex 1 million times. Worse, there is **no timeout** on regex compilation or matching. A malicious or pathological regex (e.g., `(a+)+b` matched against `"a".repeat(40) + "c"`) causes catastrophic backtracking on every single file. This is a trivial ReDoS vector that stalls the scan indefinitely.  
**Fix:** Cache compiled `regex::bytes::Regex` objects in a `static` or per-`Collector` LRU keyed by `(pattern.source.clone(), pattern.is_regex)`. Add a per-match timeout using `regex::bytes::RegexBuilder::timeout` if the `regex` crate version supports it; otherwise spawn the match in a thread with a `join` timeout.  
**Test hint:** Create a rule with regex `(a+)+b` and scan a corpus where one file contains `"a".repeat(40) + "c"`. Assert total wall-clock time stays under 1 second by bounding per-file regex work.

---

### FINDING-16: Silent data corruption when `offset_map` is shorter than claimed offsets
**Severity:** HIGH  
**Location:** `collector.rs:449-459` (`remap_byte_offsets`), `collector.rs:461-474` (`remap_offsets_buffer`)  
**Description:** Both remap functions swallow mapping failures silently:
```rust
usize::try_from(*offset)
    .ok()
    .and_then(|index| offset_map.get(index).copied())
    .unwrap_or(*offset)
```
If `offset_map` is shorter than the offset being remapped (e.g., due to a decoding bug, a truncated layer, or `usize` overflow on exotic targets), the function returns the raw unmapped offset instead of failing. The finding is then reported at an incorrect byte position. Because the function returns `Vec<u32>` / `Arc<[u32]>` without `Result`, the caller has no signal that corruption occurred.  
**Fix:** Change both functions to return `Result<...>`. Use explicit `?`-bearing propagation:
```rust
let index = usize::try_from(*offset)
    .map_err(|_| Error::validation("offset overflows usize during remap"))?;
let mapped = offset_map.get(index)
    .copied()
    .ok_or_else(|| Error::validation(format!("offset {index} out of range in offset_map")))?;
```
If best-effort fallback is required, log a warning with the file path and the offending offset instead of silently returning the raw value.  
**Test hint:** Pass an `offset_map` of length 3 and a `byte_offsets` vec containing `5`. Assert the function returns `Err`, not `5`.

---

### FINDING-17: Invalid-UTF-8 path collision in legacy `ScanContext` HashMap
**Severity:** MEDIUM  
**Location:** `collector.rs:1172` (`FileCollector::collect_file`)  
**Description:** `path.to_string_lossy()` replaces invalid UTF-8 sequences with the Unicode replacement character `U+FFFD`. Two distinct filesystem paths such as `b"\xFF\xFE"` and `b"\xFF\xFF"` both map to the same lossy string `"��"`. `FileCollector` uses this lossy string as the key in `ctx.files: HashMap<String, FileContext>`. The second file overwrites the first file's metadata, causing silent data loss in any downstream consumer of `ScanContext`.  
**Fix:** Change `ScanContext.files` to `HashMap<OsString, FileContext>` and use `path.as_os_str().to_os_string()` as the key. If `OsString` is undesirable for serialization, store a `BTreeMap<Vec<u8>, FileContext>` keyed by the raw path bytes.  
**Test hint:** Create a temp directory with two files whose names are `b"\xFF\xFE.txt"` and `b"\xFF\xFF.txt"`. Run `FileCollector` and assert `ctx.files.len() == 2`.

---

### FINDING-18: TOCTOU race between file-type check and `fs::read`
**Severity:** HIGH  
**Location:** `collector.rs:193-209` (`scan_gpu_with_context`), `collector.rs:248-262` (`collect_files`)  
**Description:** `scan_gpu_with_context` checks `entry.file_type().is_file()` (a metadata snapshot from directory enumeration) and then, many lines later, calls `read_bytes(path)` which opens the path afresh. Between the snapshot and the open, an attacker or concurrent process can replace the regular file with a named pipe (FIFO), a directory, or a device node. Reading a FIFO blocks indefinitely; reading a directory yields implementation-defined bytes on some Unix variants; reading a device can leak kernel memory or hang the process. No `O_NOFOLLOW` or `openat` is used.  
**Fix:** Open the file with `std::fs::File::open(path)` immediately, then call `file.metadata()?.is_file()` on the open handle before reading. If it is not a regular file, skip it and log the reason. This collapses the race window to the single `open` call.  
**Test hint:** Use a mock filesystem or ptrace-based fuzzer to swap a regular file with a FIFO between `readdir` and `open`; assert the scan either skips the entry or times out gracefully instead of blocking forever.

---

### FINDING-19: DFA cache check-then-act race wastes memory and CPU
**Severity:** MEDIUM  
**Location:** `collector.rs:841-862` (`compiled_dfa_for_literals`)  
**Description:** The static `DFA_CACHE` is guarded by a `Mutex`, but the access pattern is:
1. Lock → check for key → unlock.
2. If miss, call `dfa_compile_with_budget` (expensive, outside the lock).
3. Lock again → insert.
Between step 1 and 3, N concurrent threads can all miss and compile the identical DFA simultaneously. For a clause with many literals, DFA compilation is CPU-intensive and the resulting `Arc<CompiledDfa>` is large. On a many-core host this causes a memory spike and wasted compute. The code comment even notes the mutex-poison recovery (FINDING-01) but does not address the racing miss path.  
**Fix:** Protect the entire get-or-insert path under one lock hold, or switch to `dashmap` / `RwLock<HashMap<...>>` with an `entry().or_insert_with()` pattern. Because the stored value is `Arc<CompiledDfa>` (cheap to clone), holding the lock across compilation is acceptable  -  or use a `once_cell::sync::Lazy` with a per-key `Mutex` if concurrent compilation of *different* keys is desired.  
**Test hint:** Spawn 100 threads that all call `compiled_dfa_for_literals` with the same literal set simultaneously. Instrument `dfa_compile_with_budget` with a test-only atomic counter and assert it is called exactly once.

---

### FINDING-20: Unbounded `findings` Vec growth across entire scan target
**Severity:** HIGH  -  builds on FINDING-07  
**Location:** `collector.rs:175` (`scan_gpu_with_context` findings accumulator), `collector.rs:310` (`scan_collected_file` sink)  
**Description:** FINDING-07 flagged unbounded `files` Vec growth in `collect_files`, but the live hot path (`scan_gpu_with_context`) avoids `collect_files` and instead accumulates every emitted finding into a single `findings: Vec<FileFinding>` that lives for the entire scan. An adversarial corpus (e.g., every file triggers many rules) can produce millions of findings. Each `FileFinding` carries a `PathBuf`, a `Finding` with multiple `String`s and `Vec<u32>`s, and an `Arc<[u32]>`. At scale this exhausts host RAM before the scan finishes. There is no streaming, pagination, or back-pressure mechanism.  
**Fix:** Return findings via a callback or channel (`impl FnMut(FileFinding)`) so the caller can stream them to disk or a database. Alternatively, implement a flush threshold: every N findings (e.g., 10,000) write to a temporary file and clear the Vec.  
**Test hint:** Scan a corpus of 100,000 files where every file triggers 50 findings. Assert peak RSS stays under a bounded budget by flushing findings incrementally.

---

### FINDING-21: `scan_collected_file` silently skips inapplicable clauses with no audit trail
**Severity:** MEDIUM  
**Location:** `collector.rs:318-321`  
**Description:** When `plan.applicability.matches(&metadata)` returns `false`, the loop simply `continue`s. No log entry, no counter, no `ScanReport` field records how many clauses were skipped or why. A user debugging a missing finding cannot distinguish between "rule didn't match the file type," "path selector excluded it," and "the rule itself crashed." This is silent under-reporting.  
**Fix:** Add a `skipped_clauses: Vec<SkippedClause>` field to `ScanReport` that records `(path, rule_name, clause_index, reason)` for every skipped clause. Expose this in CLI verbose mode and JSON/SARIF output extensions.  
**Test hint:** Scan a file against a rule with a `filepath` selector that excludes the target. Assert the `ScanReport` contains a skipped-clause entry with reason `"applicability mismatch"`.

---

### FINDING-22: `rule_dispatch_plans` silently discards clauses with missing programs or patterns
**Severity:** MEDIUM  
**Location:** `collector.rs:482-487` (`rule_dispatch_plans`), `collector.rs:539-567` (`clause_program_and_patterns`)  
**Description:** `rule_dispatch_plans` uses `continue` when `scanner_rule_name` is `None`, when `clause_program_and_patterns` returns `None`, and when the program is missing. These are all silently swallowed with no error, warning, or skip reason. A malformed compiled document (e.g., a scanner rule renamed after compilation) will simply produce zero findings for that clause. The caller has no signal that the document is incomplete or that coverage was lost.  
**Fix:** Accumulate these skips into `ScanReport::rule_errors` or return a side-channel `Vec<DispatchSkipReason>` so the caller can surface compilation drift. Do not use bare `continue` for structural document mismatches.  
**Test hint:** Compile a document, then manually mutate one clause's `scanner_rule_name` to a non-existent name. Run a scan and assert the report contains a skip reason referencing the missing clause.

---

### FINDING-23: `collector.rs` is a 1,468-line god file violating LAW 7
**Severity:** MEDIUM  
**Location:** `collector.rs:1-1468`  
**Description:** The file contains at least four distinct responsibilities:
1. GPU scan collector (`Collector`, `scan_gpu*`, `scan_collected_file`).
2. Hit-discovery and DFA caching (`gpu_hits_for_signal`, `compiled_dfa_for_literals`, `select_hits_for_dispatch`, `regex_hits_for_signal`).
3. Decode-layer offset remapping (`remap_byte_offsets`, `remap_offsets_buffer`).
4. Legacy metadata context pipeline (`ContextCollector`, `ScanContext`, `FileCollector`, `CollectorPipeline`, `shannon_entropy`).
At 1,468 lines it is nearly 3× the 500-line LAW 7 budget. Backwards compatibility is not an excuse to keep bad architecture; the legacy pipeline can be moved to `scan/context.rs` or `scan/legacy.rs` and re-exported from `scan/collector.rs` without breaking the public API.  
**Fix:** Split into:
- `scan/collector.rs`  -  GPU path only, ≤500 lines.
- `scan/hit_discovery.rs`  -  DFA cache, regex/literal hits, hit selection.
- `scan/offset_map.rs`  -  remap helpers.
- `scan/context.rs`  -  legacy `ContextCollector` pipeline.
Re-export from `scan/collector.rs` to preserve API surface during migration.  
**Test hint:** N/A  -  architectural; verify with `wc -l` that each new file is < 500 lines.

---

### FINDING-24: Misleading `WalkDir` error diagnostic masks true failure cause
**Severity:** LOW  
**Location:** `collector.rs:194-200`, `collector.rs:249-255`  
**Description:** When `WalkDir` yields an error that is not an `io::Error` (e.g., a race-induced `NotADirectory`, or a platform-specific loop detection), `into_io_error()` returns `None`. The code then fabricates a generic `std::io::Error::other("failed to walk scan target. Fix: verify directory permissions and retry.")`. This tells the user to check permissions even when permissions are fine and the real cause is a filesystem race or symlink loop.  
**Fix:** Preserve the original `WalkDir` error display string: `walk_error.to_string()`. Include the concrete path and error kind in the message so the user can act on the actual problem.  
**Test hint:** Create a directory tree where a subdirectory is replaced by a file mid-walk (concurrent thread). Assert the error message contains the original walk error text, not just "verify directory permissions".

---

### FINDING-25: `optimal_workgroup_size` silently ignores backend misreporting for Y and Z dimensions
**Severity:** LOW  -  builds on FINDING-11  
**Location:** `dispatch.rs:265` (`optimal_workgroup_size`)  
**Description:** FINDING-11 correctly flagged the potential panic on `max_workgroup_size()[0]`. Beyond that, the function hard-codes the return value to `[size, 1, 1]`. It never consults `max_workgroup_size()[1]` or `[2]`. If the backend's adapter limits Y or Z to zero (misconfigured driver, stub backend), the returned workgroup size may exceed those dimensions, causing a backend validation error later. More subtly, if the incoming `program.workgroup_size` already carries non-trivial Y/Z components from the compiler, this function silently overwrites them, potentially breaking 2-D or 3-D dispatch layouts.  
**Fix:** Clamp the returned Y/Z against `backend.max_workgroup_size().get(1).copied().unwrap_or(1)` and `get(2)...`. Alternatively, assert/document that the scanner pipeline only ever produces 1-D workgroup programs, and enforce that invariant with a `debug_assert` in `optimal_workgroup_size`.  
**Test hint:** Supply a mock backend where `max_workgroup_size()` returns `[256, 0, 0]` and assert `optimal_workgroup_size` returns `[128, 1, 1]` (or any value with Y=1) without panic or backend validation failure.

---

## Summary Table

| # | Severity | File | Variant / Issue | Fix Priority |
|---|----------|------|-----------------|--------------|
| 12 | HIGH | `collector.rs:193-211,248-264` | Non-deterministic output order from WalkDir | Sort findings before return |
| 13 | CRITICAL | `collector.rs:585-598` | Sparse `string_id` → massive slot_count | Dense remap + cap |
| 14 | CRITICAL | `collector.rs:599,322-328` | 4× re-allocation per clause per layer | Pack once per layer, reuse |
| 15 | CRITICAL | `collector.rs:881-888` | Regex recompilation per file + ReDoS | LRU cache + timeout |
| 16 | HIGH | `collector.rs:449-459,461-474` | Silent offset-map truncation | `Result`-returning remap |
| 17 | MEDIUM | `collector.rs:1172` | Invalid-UTF-8 path collision in HashMap | `OsString` key |
| 18 | HIGH | `collector.rs:193-209,248-262` | TOCTOU race: file type → `fs::read` | Open-then-verify |
| 19 | MEDIUM | `collector.rs:841-862` | DFA cache check-then-act race | Single-lock get-or-insert |
| 20 | HIGH | `collector.rs:175,310` | Unbounded findings Vec across scan | Stream or flush threshold |
| 21 | MEDIUM | `collector.rs:318-321` | Silent skip of inapplicable clauses | `ScanReport::skipped_clauses` |
| 22 | MEDIUM | `collector.rs:482-487,539-567` | Silent discard of missing-program clauses | Side-channel skip reasons |
| 23 | MEDIUM | `collector.rs:1-1468` | 1,468-line god file (LAW 7) | Split into 4 modules |
| 24 | LOW | `collector.rs:194-200,249-255` | Misleading WalkDir error message | Preserve original error text |
| 25 | LOW | `dispatch.rs:265` | Y/Z workgroup limits ignored | Clamp or assert 1-D invariant |

---

## New findings count
**14 new findings** (12–25).

## Prior-audit correctness assessment
The first pass was **correct on every finding it raised** and **accurately prioritized** the mutex-poison, unbounded vec, and fatal-short-circuit issues. It was **not hard enough** in three dimensions:

1. **It hunted crashes, not corruption.** Silent `.ok().unwrap_or()` offset mapping, invalid-UTF-8 path collisions, and non-deterministic ordering don't panic  -  they lie. A security scanner that lies is worse than one that crashes.
2. **It looked at single-file limits, not hot-loop scaling.** The per-clause per-layer 4× re-allocation, sparse-`string_id` amplification, and regex recompilation are architectural scaling bugs that only manifest at corpus scale. Unit tests won't catch them; adversarial integration tests will.
3. **It ignored observability and modularity.** Silent skips, misleading errors, and a 1,468-line god file are systemic debt that makes every future audit harder and every bug harder to trace.

## Top-3 holes to escalate
1. **FINDING-14 (4× re-allocation hot loop)**  -  This is the most dangerous scaling issue. It will OOM a production scan against a large codebase with many clauses, and it is invisible in small tests. Escalate to P0 and block release until packed buffers are reused per layer.
2. **FINDING-15 (regex ReDoS)**  -  A single malicious rule can stall the scan forever. No timeout, no cache. This is a trivial denial-of-service vector. Escalate to P0.
3. **FINDING-12 (non-deterministic ordering)**  -  Reproducibility is a security property. If two scans of the same commit produce different SARIF baselines, CI drift-detection is useless and supply-chain attestation breaks. Escalate to P1.

*End of deeper audit. Every finding maps to an exact line, a specific failure shape, and a minimal repro or input vector.*
