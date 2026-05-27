# Third-Pass Meta-Critique  -  Commits Since 2026-04-23 17:00

**Date:** 2026-04-23  
**Scope:** Fixes landing in the last 60 minutes against the F-IR + FIX-REVIEW surface.  
**Commits audited:**
- `afc93b8002` fix(vyre-ir): close FIX-REVIEW Finding #3 + #11
- `5384566c34` test(vyre-ir): close FIX-REVIEW Finding #21
- `0805f6d4fc` fix(surgec-scan): close collector audit FINDING-01/03/04/05/06
- `98955763b6` fix(surgec-scan): close collector audit FINDING-02 + FINDING-08
- `56c71b392d` fix(surgec-scan): close collector audit FINDING-07
- `4b621b69c9` fix(surgec-scan): close dispatch audit FINDING-09 + FINDING-10

**Method:** Read-only line-by-line audit of the delta. Every item below is a finding the fix author *still* missed; earlier audit findings are referenced for context but not replayed.

---

## Requested checks that passed (no finding)

- **`dispatch_single_clause` sink.push vs error returns**  -  AUDITED. `sink.push` is the final statement before `Ok(())`. Every `?` error path (`dispatch_borrowed`, output arity checks, `decode_result_slots`, `resolve_result_slots`) precedes the push. No partial mutation of the sink occurs on error.
- **`canonical_f64_zero` NaN boundary**  -  AUDITED. `value == 0.0` short-circuits before `is_sign_negative()` is evaluated. IEEE-754 requires `NaN != 0.0` on all conforming targets (x86_64, AArch64, PPC, RISC-V, WASM). `is_sign_negative()` reads the raw sign bit and is target-consistent. The zero-canonicalisation contract is sound.

---

## Findings

### 01. CRITICAL | collector.rs:928-932 | Mutex poison recovery comment falsely claims HashMap atomicity, hiding corruption risk
The code comment states: *"HashMap insert is atomic at the Rust level, and the values are `Arc`-shared immutable DFAs."* `std::collections::HashMap` reallocation/rehashing during grow is **not** atomic. If a panic occurs mid-rehash - e.g., the allocator returns `null` on a large grow, or a custom `Drop` on a key/value panics - the bucket array may be in a torn state. `poison.into_inner()` returns this corrupted map to future callers. Subsequent `get()` or `insert()` on the corrupted map can return a `CompiledDfa` for the *wrong* literal set, loop indefinitely, or panic. A security scanner that silently uses the wrong DFA produces false negatives. The comment is a dangerous false-safety claim.

**Fix:** Replace `std::sync::Mutex` with `parking_lot::Mutex` (never poisons) or `dashmap` (lock-free). If the stdlib mutex must stay, delete the false atomicity claim and add a defensive `clear()` + rebuild-on-poison path instead of blind `into_inner()` reuse.

**Test hint:** Instrument `dfa_compile_with_budget` with a test-only hook that panics on the Nth invocation while the cache mutex is held for insert. After recovery, assert that a lookup for a *different* key returns `None` (or migrate to `parking_lot` and run under Miri to verify no undefined behaviour).

---

### 02. HIGH | collector.rs:369-427 | `build_clause_inputs` failure still aborts the entire file - FINDING-06 fix is incomplete
The fix for FINDING-06 wrapped `scan_collected_file` at the call site in `scan_gpu_with_context` (line 232-238), but `scan_collected_file` itself still propagates `build_clause_inputs` and `dispatch_rules` errors via `?` (lines 387-395). A single clause that exceeds `MAX_SIGNAL_SLOTS`, carries a `string_id` that overflows `usize`, or fails DFA compilation aborts **all remaining clauses and all decode layers** for that file. Worse, the shared `findings` sink may already contain partial results from earlier clauses of the same file. The caller logs the error and continues to the next file, leaving the partial findings in the final report with no marker that the file scan was incomplete.

**Fix:** Move the `match` / `continue` pattern **inside** `scan_collected_file` so `build_clause_inputs` and `dispatch_rules` errors are logged per-clause, a `SkippedClause` entry is recorded, and the loop advances to the next plan.

**Test hint:** Build a document with two clauses where clause 0 matches and clause 1 forces `slot_count > MAX_SIGNAL_SLOTS`. Scan a single file and assert that (a) clause 0 findings are present, (b) clause 1 produces a skip log, and (c) the `ScanReport` flags the file as partially scanned.

---

### 03. HIGH | collector.rs:173-241 | `scan_gpu_with_context` streaming path has no `MAX_SCAN_FILES` cap
FINDING-07 added `MAX_SCAN_FILES = 1_000_000` to `collect_files()` (line 282-312), but the live hot path `scan_gpu_with_context` enumerates files directly via `WalkDir` without any cap (lines 193-240). A target with tens of millions of files - e.g., a tar-bomb extraction or nested symlink farm - will be fully enumerated and each file read and decoded, because the cap only protects the legacy `collect_files` path. Combined with unbounded `findings` Vec growth (FINDING-20), the streaming path still exhausts host RAM before the scan finishes.

**Fix:** Refactor both paths to share a single capped enumerator, or insert the same `if files_seen >= MAX_SCAN_FILES { return Err(QuotaExceeded...) }` guard inside the `scan_gpu_with_context` WalkDir loop.

**Test hint:** Create a temp directory with `MAX_SCAN_FILES + 1` empty regular files. Assert that `scan_gpu_with_context` returns a `QuotaExceeded` error when it reaches the last file, without reading it.

---

### 04. HIGH | collector.rs:1143-1151 + 1135-1141 | 512 MiB cap yields ~2.63 GiB peak per file, not "far below typical host memory"
The commit message claims the 512 MiB cap "bounds the packed buffer at 2 GiB instead of unbounded" and "keeps the per-file allocation far below typical host memory". Worst-case peak RSS for one 512 MiB file:
- Original file buffer (`Arc<[u8]>`): 512 MiB
- `pack_bytes_as_u32_words` buffer: 512 MiB × 4 = 2 GiB
- `offsets` vec (`MAX_SIGNAL_SLOTS × MAX_CACHED_POSITIONS` × 4 bytes): 65 536 × 256 × 4 ≈ 64 MiB
- `lengths` vec: another ≈ 64 MiB
- `counts` + `metadata` + overhead: < 1 MiB
**Total peak ≈ 2.63 GiB per file.** On a 4 GiB VM this leaves ~1.3 GiB for the OS, GPU driver, backend scratch space, and the process image. The fix capped file size but did not calculate worst-case peak, add a per-scan memory budget, or backpressure. A single 512 MiB file can still OOM a constrained CI runner or container.

**Fix:** Lower `MAX_FILE_BYTES` to 128 MiB (peak ~640 MiB) or query available system memory at scan start and set the cap to `min(512 MiB, available_ram / 4)`.

**Test hint:** Run the collector inside a 4 GiB cgroup against a 512 MiB random file. Assert peak RSS stays under 3 GiB; if it exceeds the budget the scan must abort with a structured `Error::Io` before allocating the packed buffer.

---

### 05. HIGH | dispatch.rs:119-127,138-143,188-194 + collector.rs:200-205,217-221,233-237 | Unbounded `eprintln!` flood from adversarial input can exhaust disk
The per-rule and per-clause error localisation fixes replaced fatal `?` aborts with `eprintln!` + `continue`. A malicious `CompiledDocument` containing thousands of malformed rules or clauses generates one stderr line per failure. When stderr is redirected to a log file - standard practice in CI/CD and containerised scanners - an adversary can fill the disk, causing a denial-of-service that outlives the scan process itself. No rate limit, no deduplication, and no skip-count aggregation exists across any of the six `eprintln!` sites in the scan surface.

**Fix:** Add a `skip_counter: AtomicUsize` (or `DashMap<String, AtomicUsize>` keyed by error category). Print the first 10 unique diagnostics, then emit a single summary line: `surgec: +1234 similar errors skipped. Fix: ...`. Route diagnostics through a bounded channel or ring buffer instead of unbuffered stderr.

**Test hint:** Feed a document with 10 000 intentionally-broken clauses to `dispatch_rules` with `std::io::stderr()` redirected to a temp file. Assert the temp file size stays under 1 MiB.

---

### 06. MEDIUM | opaque_payload_endian.rs:287-316 | 10MB stress test lacks `#[ignore]` and may flake on debug builds or constrained CI
The 10MB alternating stress test runs unconditionally on every `cargo test`. It allocates 10 MiB and sorts 10 million `char`s. The 2-second wall-clock ceiling is generous for release mode but can be exceeded on debug builds (where `sort_unstable` is unoptimised) or on overloaded shared CI runners where CPU time is throttled. The 1MB random test already provides adequate coverage for the fast path.

**Fix:** Add `#[ignore = "stress test: run with --ignored on performant hosts"]` to the 10MB test. Keep the 1MB test as the default fast-path regression.

**Test hint:** Run `cargo test` in debug mode on a 2-core GitHub Actions runner. Verify the 1MB test completes in <2s while the 10MB test is skipped by default.

---

### 07. MEDIUM | opaque_payload.rs:187-212 | `canonical_f64_zero` lacks adversarial proptest coverage
`canonical_f64_zero` was added with only four unit-tested bit patterns for pass-through (smallest negative subnormal, -1.0, negative qNaN, positive sNaN). There is no proptest generating random `u64` bit patterns and asserting the canonicalisation contract. A future refactor could subtly break the boundary - e.g., replacing `value == 0.0` with `value.is_subnormal()` - and the tiny unit suite would not catch the regression.

**Fix:** Add a proptest in `tests/opaque_payload_endian.rs` that generates random `u64` seeds, converts to `f64` via `from_bits`, calls `canonical_f64_zero`, and asserts: (a) if input bits are `0x0000_0000_0000_0000` or `0x8000_0000_0000_0000`, output bits are `0x0000...0000`; (b) for all other bit patterns, output bits equal input bits.

**Test hint:** Proptest with 10 000 random `u64` seeds covering all exponent/mantissa combinations.

---

### 08. MEDIUM | program.rs:172-175 + 389-404 | `BufferDecl.count` is `pub`, bypassing the zero-count workgroup invariant
The fix added `assert!(count > 0)` to `BufferDecl::workgroup` (line 398-404) and `with_count` (line 345-351), but `count` is declared as a `pub` field (line 175). Direct field mutation (`let mut b = BufferDecl::workgroup("x", 64, DataType::U32); b.count = 0;`) bypasses both constructor guards entirely. The invariant is only enforced at two call sites, not on the type itself. Additionally, the new `workgroup` assert has **zero test coverage** - the test file `tests/buffer_decl_with_count.rs` only exercises `with_count`.

**Fix:** Make `count` private (`pub(crate)` or private with a checked setter) so the invariant cannot be violated by direct mutation. Add a `#[should_panic]` test for `BufferDecl::workgroup("x", 0, DataType::U32)`.

**Test hint:** `BufferDecl::workgroup("scratch", 0, DataType::U32)` must panic with an actionable `Fix:` hint (currently untested). Direct mutation `b.count = 0` on a workgroup buffer must be rejected by `Program::validate` or a dedicated `BufferDecl::validate`.

---

## Summary

**New findings by severity:**
- **CRITICAL:** 1
- **HIGH:** 4
- **MEDIUM:** 3
- **LOW:** 0

**Top-3 to escalate first:**

1. **Finding 01 (DFA cache poison corruption claim)**  -  The false safety comment around `into_inner()` is the most dangerous because it tells future maintainers the cache is safe when it is not. A corrupted DFA cache causes silent false negatives in a security scanner. Escalate to P0 and replace the stdlib `Mutex` with `parking_lot` before the next release.
2. **Finding 02 (incomplete FINDING-06 fix)**  -  A single bad clause still hides all later clauses for the same file. In a security context, this means a parser error on clause 0 can conceal every finding from clauses 1..N. Escalate to P1 and move the per-clause `continue` pattern inside `scan_collected_file`.
3. **Finding 03 (streaming path lacks MAX_SCAN_FILES cap)**  -  The cap was added to the legacy path but not the live streaming path. A hostile deep tree still reads an unlimited number of files. Escalate to P1 and unify both paths under a single capped enumerator.
