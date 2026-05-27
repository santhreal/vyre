# Deep Audit  -  Collector & Dispatch Adversarial Resilience

**Date:** 2026-04-23  
**Scope:** `surgec/src/scan/collector.rs`, `surgec/src/scan/dispatch.rs`  
**Goal:** Hunt panic paths, unwrap chains, silent swallowed errors, and unbounded memory growth. A collector crash on one rule must never crash the whole scan; a dispatch failure on one program must never corrupt the hit ring.

---

## 1. Collector Panic Paths & Fatal Short-Circuits

### FINDING-01: DFA cache mutex poison expect() crashes entire scan process
**Severity:** CRITICAL  
**Location:** `surgec/src/scan/collector.rs:835`, `surgec/src/scan/collector.rs:853`  
**Variant:** `std::sync::Mutex::lock().expect("dfa cache mutex poisoned")`  
**Description:** The process-wide static `DFA_CACHE` is guarded by a `std::sync::Mutex`. If any thread panics while holding the lock (e.g., OOM inside `dfa_compile_with_budget` or a backend thread abort), the mutex is poisoned. Every subsequent scan attempt hits `.expect(...)` and panics, killing the entire process.  
**Minimal Program:**
```rust
// Any trigger that causes a panic while the DFA cache lock is held,
// e.g. feeding a pathologically large literal set that overflows the
// DFA budget allocator.
```
**Fix:** Replace `expect` with `lock().unwrap_or_else(|poison| poison.into_inner())` to recover the guarded data, or migrate to `parking_lot::Mutex` (never poisons) or a lock-free cache such as `dashmap`.

---

### FINDING-02: Unbounded vector allocation from u32 string_id allows OOM / allocation panic
**Severity:** CRITICAL  
**Location:** `surgec/src/scan/collector.rs:596-598`  
**Variant:** `vec![0u32; slot_count]` and `vec![0u32; slot_count.saturating_mul(MAX_CACHED_POSITIONS)]`  
**Description:** `slot_count` is derived from the maximum `string_id` (a `u32`) found in compiled patterns, plus one. A malicious or buggy rule with `string_id = u32::MAX` produces `slot_count = 4294967296`. `vec![0u32; slot_count]` attempts to allocate ~16 GiB; the offsets/lengths vectors attempt `slot_count * 256` (~4 TiB, or `usize::MAX` after saturation), causing an immediate allocation panic or OOM kill. No compile-time or runtime validation caps the id range.  
**Minimal Program:**
```rust
// A CompiledPattern with string_id = u32::MAX inserted into the
// CompiledDocument passed to Collector::scan_gpu.
```
**Fix:** Cap `slot_count` to a hard maximum (e.g., `MAX_RULE_STRINGS` or 65536) and return a structured `Error::Compile` if the rule exceeds it. Validate `string_id` ranges during `CompiledRuleIndex::build` so the error is caught at compile time, not scan time.

---

### FINDING-03: read_bytes() loads entire file without size cap  -  hostile input triggers OOM
**Severity:** HIGH  
**Location:** `surgec/src/scan/collector.rs:1042-1047`  
**Variant:** `fs::read(path)`  
**Description:** `read_bytes` calls `fs::read`, which loads the complete file contents into a `Vec<u8>`. Scanning a target that contains a sparse file, block device, named pipe, or multi-gigabyte log will attempt to allocate unbounded memory and either OOM-kill the process or abort with an allocation failure.  
**Minimal Program:**
```bash
# In the scan target:
mkfifo /tmp/scan_target/pipe
dd if=/dev/zero of=/tmp/scan_target/sparse bs=1M seek=100000 count=0
# Then run surgec scan on /tmp/scan_target
```
**Fix:** Check `fs::metadata(path)?.len()` against a configurable `max_file_size` (default e.g. 512 MiB) before calling `fs::read`. Skip or error on oversized entries, and explicitly reject non-regular files (FIFOs, devices, sockets).

---

### FINDING-04: pack_bytes_as_u32_words expands file bytes 4× without size guard
**Severity:** HIGH  
**Location:** `surgec/src/scan/collector.rs:599` (call site), `surgec/src/scan/collector.rs:1034-1040` (definition)  
**Variant:** `pack_bytes_as_u32_words(file_bytes)`  
**Description:** Every byte of the input file is mapped to a `u32` (4 bytes) and then serialized as little-endian words, producing a 4× memory expansion. A 2 GiB file becomes an 8 GiB `Vec<u8>` before any GPU dispatch occurs. Combined with the original file buffer still held in `Arc<[u8]>`, a single large file can exhaust host memory even though the file itself is below the `u32::MAX` scanner limit.  
**Minimal Program:**
```bash
dd if=/dev/urandom of=/tmp/scan_target/big.bin bs=1M count=2048
# Scanning this file causes ~10 GB of host RAM usage.
```
**Fix:** Move the packing into the backend (or use a zero-copy byte-view dispatch). If host-side packing is required, check `file_bytes.len().saturating_mul(4)` against a per-scan memory budget before allocating, and shard or stream the file when the budget is exceeded.

---

### FINDING-05: WalkDir error on single entry aborts entire scan
**Severity:** HIGH  
**Location:** `surgec/src/scan/collector.rs:193-201` (`scan_gpu_with_context`), `surgec/src/scan/collector.rs:248-256` (`collect_files`)  
**Variant:** `entry.map_err(...)?` on `WalkDir` iteration  
**Description:** Any directory traversal error - permission denied on one subdirectory, a broken symlink in metadata, or an I/O fault on a single entry - is converted into a fatal `Error::Io` that aborts the entire scan. An adversary can place an unreadable file in the target to deny service.  
**Minimal Program:**
```bash
mkdir -p /tmp/scan_target
chmod 000 /tmp/scan_target/unreadable
# Running surgec scan on /tmp/scan_target fails immediately.
```
**Fix:** Log the walk error and `continue` to the next entry instead of returning. Accumulate skipped paths in a `Vec<PathBuf>` and expose them in `ScanReport` so the caller knows the scan was partial.

---

### FINDING-06: Single clause failure aborts entire file and whole scan
**Severity:** HIGH  
**Location:** `surgec/src/scan/collector.rs:304-362` (`scan_collected_file`)  
**Variant:** `build_clause_inputs(...)?` and `dispatch_rules(...)?` inside nested loops  
**Description:** Both `build_clause_inputs` (line 328) and `dispatch_rules` (line 330) use the `?` operator. If any clause fails - for example because one file exceeds `u32::MAX` bytes, or one clause has a DFA compile failure - the error propagates out of the file loop, out of the directory loop, and aborts the entire scan. All remaining clauses for that file and all remaining files in the target are skipped.  
**Minimal Program:**
```rust
// A scan target containing one file > 4 GiB alongside many smaller files.
// The large file triggers u32::try_from(file_bytes.len()) to fail,
// killing the scan before the smaller files are evaluated.
```
**Fix:** Wrap each clause dispatch in a `match` or `if let Err(e)` block that logs the clause failure, records a per-clause skip entry (e.g., `Finding::clause_error`), and continues to the next clause. Only propagate fatal errors (backend lost, memory unrecoverable) and treat per-clause/ per-file errors as localized skips.

---

### FINDING-07: collect_files() unbounded Vec growth from directory enumeration
**Severity:** MEDIUM  
**Location:** `surgec/src/scan/collector.rs:264`  
**Variant:** `files.push(...)` inside `WalkDir` loop  
**Description:** The `files` Vec grows without limit as `WalkDir` yields entries. A target directory containing millions of empty files (e.g., a filesystem fuzzing payload or tar-bomb extraction) can exhaust memory before scanning even begins.  
**Minimal Program:**
```bash
mkdir -p /tmp/scan_target/bomb
for i in $(seq 1 10000000); do touch /tmp/scan_target/bomb/$i.txt; done
```
**Fix:** Stream files instead of collecting into a Vec. If a Vec is required for the current pipeline, cap the file count (e.g., `MAX_SCAN_FILES = 1_000_000`) and return a structured error or paginate the scan.

---

### FINDING-08: Metadata layout sanity check is tautological (dead code)
**Severity:** MEDIUM  
**Location:** `surgec/src/scan/collector.rs:631-637`  
**Variant:** `metadata.len() <= MATCH_METADATA_FILE_COUNT_SLOT`  
**Description:** `metadata` is hard-coded to `vec![file_size, 1u32]` (length 2). `MATCH_METADATA_FILE_COUNT_SLOT` is `1`. The condition `2 <= 1` is always false, so the error arm is unreachable dead code. This gives false confidence that the metadata layout is dynamically validated.  
**Minimal Program:** N/A  -  unreachable by construction.  
**Fix:** Replace the runtime check with a `const_assert!` (via `static_assertions`) that validates the metadata arity at compile time, or delete the dead branch and document the invariant on the `PreparedInputs` struct.

---

## 2. Dispatch Panic Paths & Fatal Short-Circuits

### FINDING-09: Single rule dispatch failure aborts entire scan
**Severity:** CRITICAL  
**Location:** `surgec/src/scan/dispatch.rs:96-126` (`dispatch_rules`)  
**Variant:** `dispatch_rule(...)?` on every rule iteration  
**Description:** `dispatch_rules` iterates over every compiled rule and artifact rule, calling `dispatch_rule`. Any backend error, malformed program, or output arity mismatch for a single rule short-circuits the entire dispatch with `?`, meaning no further rules are evaluated. One bad rule kills the complete scan.  
**Minimal Program:**
```rust
// A CompiledDocument where one rule's Program contains an unsupported
// Node variant for the current backend. Backend dispatch returns Err,
// and the entire scan aborts.
```
**Fix:** Wrap each `dispatch_rule` call in a `match` that catches rule-level errors, logs them, and continues. Accumulate per-rule errors into a side channel (e.g., `ScanReport::rule_errors`) so the caller receives partial findings together with a diagnostic of what was skipped.

---

### FINDING-10: Single clause failure aborts all later clauses in the same rule
**Severity:** CRITICAL  
**Location:** `surgec/src/scan/dispatch.rs:141-233` (`dispatch_rule`)  
**Variant:** `backend.dispatch_borrowed(...)?` and `decode_result_slots(...)?` inside clause loop  
**Description:** Inside `dispatch_rule`, every clause is evaluated in sequence. If `backend.dispatch_borrowed` fails (line 158) or `decode_result_slots` rejects a malformed buffer (line 193), the `?` operator propagates the error out of the clause loop, out of the rule, and ultimately out of the scan (via FINDING-09). Remaining clauses for that rule are never evaluated, yielding an incomplete boolean result.  
**Minimal Program:**
```rust
// A multi-clause rule where clause 0 produces a result buffer whose
// length is not a multiple of 4. decode_result_slots returns Err,
// and clauses 1..N are silently skipped.
```
**Fix:** Catch clause errors inside the loop, log them with rule name and clause index, and `continue` to the next clause. Only abort the rule (or scan) for truly unrecoverable backend failures (e.g., device lost).

---

### FINDING-11: Potential panic if backend.max_workgroup_size() returns empty slice
**Severity:** HIGH  
**Location:** `surgec/src/scan/dispatch.rs:265`  
**Variant:** `backend.max_workgroup_size()[0]`  
**Description:** `optimal_workgroup_size` indexes the first element of `backend.max_workgroup_size()` without checking length. If the trait method returns a dynamically-sized slice (e.g., `&[u32]`) and a misimplemented or malicious backend returns an empty slice, the indexing panics. This is called once per clause, so a single bad backend response can abort the scan thread.  
**Minimal Program:**
```rust
// A custom VyreBackend impl where max_workgroup_size() returns &[][..].
optimal_workgroup_size(backend, program); // panics
```
**Fix:** Defensively check `.first()` or `.get(0)` before indexing, returning a default `[256, 1, 1]` if the backend reports no workgroup dimensions. Alternatively, tighten the `VyreBackend` trait contract so `max_workgroup_size` returns `[u32; 3]` instead of a slice.

---

## Summary Table

| # | Severity | File | Variant / Issue | Fix Priority |
|---|----------|------|-----------------|--------------|
| 01 | CRITICAL | `collector.rs:835,853` | DFA cache mutex poison `expect()` | Recover poison or use `parking_lot::Mutex` |
| 02 | CRITICAL | `collector.rs:596-598` | Unbounded `slot_count` vec allocation | Cap `slot_count` at compile time |
| 03 | HIGH | `collector.rs:1042-1047` | `fs::read` without size cap | Add `max_file_size` gate |
| 04 | HIGH | `collector.rs:599` | `pack_bytes_as_u32_words` 4× expansion | Zero-copy dispatch or memory budget |
| 05 | HIGH | `collector.rs:193-201,248-256` | WalkDir error aborts whole scan | Log and `continue` |
| 06 | HIGH | `collector.rs:304-362` | Clause failure aborts file + scan | Per-clause error catch + continue |
| 07 | MEDIUM | `collector.rs:264` | Unbounded `files` Vec from WalkDir | Stream or cap file count |
| 08 | MEDIUM | `collector.rs:631-637` | Tautological metadata layout check | `const_assert!` or delete dead branch |
| 09 | CRITICAL | `dispatch.rs:96-126` | Rule failure aborts entire scan | Per-rule error catch + continue |
| 10 | CRITICAL | `dispatch.rs:141-233` | Clause failure aborts later clauses | Per-clause error catch + continue |
| 11 | HIGH | `dispatch.rs:265` | `max_workgroup_size()[0]` may panic | Defensive `.first()` or trait change |

---

*End of audit. Every finding maps to an exact line, a specific failure shape, and a minimal repro or input vector. The recommended immediate-action items that block the next release are:*

1. **FINDING-01**  -  Replace the DFA cache `Mutex` poison panic with recovery or a non-poisoning lock. A single backend thread panic must not brick the process.
2. **FINDING-02**  -  Add a compile-time `string_id` cap so a malicious rule cannot force a multi-terabyte allocation at scan time.
3. **FINDING-09 & FINDING-10**  -  Localize rule- and clause-level dispatch errors. The current `?` short-circuit violates the core requirement that one bad program must not kill the scan.
4. **FINDING-03**  -  Gate `fs::read` with a configurable maximum file size to prevent trivial OOM DoS.
