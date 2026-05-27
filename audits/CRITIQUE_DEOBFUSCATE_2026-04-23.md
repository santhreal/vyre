# CRITIQUE  -  `deobfuscate` crate (read-only audit)

**Date:** 2026-04-23  
**Scope:** `libs/scanner/deobfuscate/src/` + tests + `Cargo.toml`  
**Auditor:** Kimi Code CLI (security research protocol)  
**Laws applied:** 0–8, Standards, Research Protocol  

---

## Executive Summary

The `deobfuscate` crate is a single-threaded, loop-based layer un-wrapper with **no memory budgets**, **no timeouts**, and **no output-size caps** for the majority of its decoders. Only Zlib has a 10 MiB output limit. The XOR brute-forcer is an unbounded memory multiplier (255× input size). JavaScript `\xXX` escape handling silently corrupts binary payloads via UTF-8 expansion. Unicode normalization is absent, creating a trivial evasion vector for signature matching. Shared-state usage is present (`LazyLock<Regex>`) but is read-only and therefore safe.

**Severity distribution:** 4 CRITICAL, 5 HIGH, 4 MEDIUM, 3 LOW/INFO.

---

## Findings (numbered, severity-ordered)

### 1. CRITICAL | generic.rs:111–119 | XOR brute force is an unbounded memory multiplier

```rust
for key in 1..=255u8 {
    let xored: Vec<u8> = input.iter().map(|&b| b ^ key).collect();
    if is_printable_ascii(&xored) { ... }
}
```

- **Description:** For an N-byte input this loop allocates **255 × N bytes** in the worst case (all keys pass `is_printable_ascii`). A 10 MiB input → ~2.55 GiB of transient allocations. A 100 MiB input → ~25.5 GiB. There is no input-size gate, no per-layer cap, and no early abort on memory pressure.
- **Suggested fix:** Add a pre-flight input-size limit for the XOR path (e.g., reject inputs > 1 MiB for brute-force). Alternatively, use a streaming / in-place check that materializes only the winning key, or gate XOR behind a caller-provided `max_input_size`.
- **Test hint:** `proptest` or `faultkit` fuzz with `vec![b'A'; 5_000_000]` and assert RSS stays below a hard ceiling (e.g., 50 MiB).

---

### 2. CRITICAL | javascript.rs:68–69 | `\xXX` escapes are corrupted by UTF-8 expansion

```rust
if let Ok(num) = u8::from_str_radix(&caps[1], 16) {
    (num as char).to_string()
}
```

- **Description:** `num` is a `u8` (0–255). Casting to `char` and then calling `.to_string()` encodes the value as a **UTF-8 Unicode scalar**, not as a raw byte. For example `\xFF` (1 byte) becomes the two-byte UTF-8 sequence `0xC3 0xBF`. Any downstream binary signature matching (shellcode, encoded PE headers, etc.) will fail because the payload has been silently expanded and altered.
- **Suggested fix:** Replace `char`-based replacement with raw-byte replacement. Operate on `Vec<u8>` (or `Cow<[u8]>`) so that `\xFF` remains the single byte `0xFF` in the output buffer. Do not round-trip through `String`/`char` for byte-oriented escape sequences.
- **Test hint:** Assert that `deobfuscate(b"var x = \"\\xFF\\xFE\";")` yields a layer whose `data` is exactly `b"var x = \"\xFF\xFE\";"` (raw bytes), not `b"var x = \"\xC3\xBF\xC3\xBE\";"`.

---

### 3. CRITICAL | layers.rs + all decoder modules | No per-call or per-layer timeout

- **Description:** `deobfuscate_auto`, `deobfuscate_js`, and every individual `Deobfuscator::deobfuscate` run to completion with **no time budget**. A pathological 100 MiB input will spend unbounded time in:
  - 255 XOR iterations over 100 MiB each,
  - `hex::decode` attempting to parse 100 MiB as hex,
  - `encodex::base64::decode` scanning 100 MiB for validity,
  - regex matches over massive strings.
  This is a **DoS vector** against any async scanner that awaits deobfuscation.
- **Suggested fix:** Introduce a `DeobfuscateBudget { max_duration: Duration, max_output_bytes: usize }` passed down from `deobfuscate_auto` into every decoder. Decoders check `Instant::elapsed()` at loop heads and return truncated results when the budget is exhausted. Fail fast with a `BudgetExceeded` error or a dedicated layer method string (`"timeout"`).
- **Test hint:** `faultkit` adversarial test: 50 MiB of printable ASCII fed to `GenericDeobfuscator::deobfuscate` must return within 100 ms or be killed.

---

### 4. CRITICAL | generic.rs + javascript.rs + python.rs | No input-size validation at public entry points

- **Description:** All public `deobfuscate` methods accept `&[u8]` of arbitrary length. There is no `max_input_len` constant enforced at the trait level or in `layers.rs`. The only upstream limit is inside `encodex::base64::decode` (10 MiB input), but `hex::decode` and the language-specific regex paths have **no size ceiling**.
- **Suggested fix:** Define a crate-wide `MAX_INPUT_BYTES: usize` (e.g., 10 MiB) and reject oversized inputs immediately at the top of `deobfuscate_auto`, `deobfuscate_js`, and every `Deobfuscator::deobfuscate` impl. Return an empty vec or a dedicated error layer.
- **Test hint:** Pass `vec![b'a'; 11_000_000]` and assert instant rejection (no allocations from decoders).

---

### 5. HIGH | generic.rs:97–99 | Zlib truncation is silent  -  no truncation indicator

```rust
if decoder
    .take(10 * 1024 * 1024)
    .read_to_end(&mut decompressed)
    .is_ok()
```

- **Description:** `Take::read_to_end` returns `Ok(())` after reading exactly 10 MiB even if the stream continues for gigabytes (zip bomb). The caller cannot distinguish "complete decompression" from "truncated decompression". A subsequent signature match against a 10 MiB prefix may miss the real payload at offset 10,000,001.
- **Suggested fix:** After `read_to_end`, attempt one more byte read. If it succeeds, mark the layer as `generic_zlib_inflate_truncated` or add a `truncated: bool` field to `DeobfuscatedLayer`. Do not silently drop the tail of a zip bomb.
- **Test hint:** Create a 20 MiB zero-filled zlib stream. Assert that the resulting layer carries a `truncated == true` flag.

---

### 6. HIGH | layers.rs:34–93 | `all_layers` retains every decoded layer  -  no total memory budget

- **Description:** `all_layers` is a `Vec<DeobfuscatedLayer>` that accumulates the full data of every peeled layer. With `MAX_LAYERS = 10`, if each layer is 10 MiB (zlib cap) the vector holds 100 MiB. But because other decoders have **no per-layer cap**, a single 100 MiB hex → 50 MiB base64 → ... sequence could retain hundreds of megabytes. At scanner scale (thousands of concurrent files) this is a memory-exhaustion vector.
- **Suggested fix:** Either (a) evict / deduplicate layers as you go, keeping only the last N layers and the original, or (b) add a `max_total_retained_bytes` budget that aborts accumulation.
- **Test hint:** Nested encoding of a 50 MiB payload through 10 layers; assert total `all_layers.iter().map(|l| l.data.len()).sum()` does not exceed a configurable ceiling.

---

### 7. HIGH | javascript.rs:157–183 | `String.fromCharCode` decoder has no code-point limit

```rust
for n_str in nums_str.split(',') {
    ...
    if let Ok(num) = n_str.parse::<u32>() {
        if let Some(c) = char::from_u32(num) {
            decoded_str.push(c);
        }
    }
}
```

- **Description:** A 10 MiB string consisting of comma-separated integers (e.g., `String.fromCharCode(65,65,65,... millions ...)`) will be fully parsed and pushed into `decoded_str` with no upper bound. This is an unbounded output allocator.
- **Suggested fix:** Count parsed code points and abort if the count exceeds `MAX_FROM_CHAR_CODE_POINTS` (e.g., 1 MiB). Also cap the total output bytes after UTF-8 encoding.
- **Test hint:** 5 MiB of `65,` repeated; assert decoder returns empty or a truncated-flag layer rather than a 5 MiB output.

---

### 8. HIGH | generic.rs:70–76 + generic.rs:83–90 | Hex and Base64 decoders lack output-size caps in this crate

- **Description:** `hex::decode` (external crate) has no documented input/output limit. `encodex::base64::decode` limits input to 10 MiB, but the **output** can still be ~7.5 MiB. The deobfuscator crate does not apply its own output cap on top of these. An attacker can craft a 10 MiB base64 input that expands to 7.5 MiB, then feed it into the next layer.
- **Suggested fix:** After every decode, check `decoded.len() <= MAX_LAYER_OUTPUT_BYTES` (e.g., 10 MiB) before pushing to `results`. Reject oversized decodes immediately.
- **Test hint:** Construct base64 encoding of 8 MiB of printable ASCII; assert the decoder rejects it or truncates with a flag.

---

### 9. HIGH | generic.rs:122–138 | ROT13 has no size cap

- **Description:** ROT13 operates on the full input string, allocating an output string of identical length. For a 100 MiB input, this is a 100 MiB allocation with no gate.
- **Suggested fix:** Gate ROT13 behind the same `MAX_INPUT_BYTES` check as the global entry point, or skip it for inputs above a threshold.
- **Test hint:** 50 MiB of ASCII letters; assert instant rejection or streaming behavior.

---

### 10. MEDIUM | layers.rs:25 | `MAX_LAYERS = 10` is hardcoded and non-configurable

- **Description:** Callers cannot tune depth vs. resource trade-offs. A CI scanner might want `MAX_LAYERS = 3` for speed; a deep-analysis pipeline might want `20`. The constant is baked into two functions with duplication (`deobfuscate_auto` and `deobfuscate_js`).
- **Suggested fix:** Add `max_layers: usize` to a `DeobfuscateConfig` struct. Provide a `deobfuscate_with_config` entry point. Keep `deobfuscate_auto` as a convenience wrapper with a sensible default.
- **Test hint:** Property test: `deobfuscate_with_config(input, 0)` returns empty; `deobfuscate_with_config(input, 100)` on a 20-layer payload returns exactly 20 layers.

---

### 11. MEDIUM | javascript.rs:78–86 | Lone surrogates (`\uD800`–`\uDFFF`) are silently dropped

```rust
if let Ok(num) = u32::from_str_radix(&caps[1], 16) {
    if let Some(c) = char::from_u32(num) {
        return c.to_string();
    }
}
caps[0].to_string()  // fallback: leaves escape untouched
```

- **Description:** JavaScript allows lone surrogates in strings (they are valid UTF-16 code units). Rust's `char::from_u32` rejects the surrogate range, so the regex match falls through to `caps[0].to_string()`  -  the escape sequence is **left untouched**. An obfuscator can hide payload bytes inside lone-surrogate escapes to evade the unescape layer entirely.
- **Suggested fix:** Handle the surrogate range explicitly. Convert lone surrogates to the raw UTF-8 bytes `0xED 0xA0 0x80`–`0xED 0xBF 0xBF` (CESU-8 / WTF-8 style) so that downstream matchers see a deterministic byte sequence rather than the original escape.
- **Test hint:** Input `b"\\uD800\\uDC00"` should decode to the 4-byte UTF-8 representation of U+10000 (or at minimum should not be left as literal `\uD800\uDC00`).

---

### 12. MEDIUM | layers.rs:81–82 | `clone_from` clones full payload on every successful layer

```rust
current_data.clone_from(&best.data);
all_layers.push(best);
```

- **Description:** Each iteration copies the entire best payload into `current_data`. For 10 layers of 10 MiB each, this is ~100 MiB of memcpy traffic. Not a vulnerability per se, but a scaling tax that could be eliminated with a `Vec<Vec<u8>>` stack or `Arc<[u8]>`.
- **Suggested fix:** Store layers as `Arc<[u8]>` or keep an index into a single `Vec<DeobfuscatedLayer>` without the redundant `current_data` buffer.
- **Test hint:** Benchmark 10 layers of 10 MiB; assert zero-copy path shows < 1 ms layer-switch overhead.

---

### 13. MEDIUM | generic.rs:44–61 | `is_printable_ascii` silently discards all binary payloads

- **Description:** Hex and base64 decoded outputs are rejected unless they are ≥ 80 % printable ASCII. This means **binary shellcode, encrypted blobs, and compiled bytecode** are intentionally dropped. A threat actor can evade the scanner by ensuring their base64 payload decodes to ≥ 20 % non-printable bytes.
- **Suggested fix:** Remove the `is_printable_ascii` filter from hex/base64, or make it opt-in via `DeobfuscateConfig`. A security scanner should return *all* decoded candidates and let downstream analyzers decide what is interesting.
- **Test hint:** Base64-encode `b"\x90\x90\x90\x90\x90\x90\x90\x90"` (NOP sled); assert it produces a `generic_base64_decode` layer.

---

### 14. LOW | javascript.rs:30–51 + python.rs:26–36 | Shared immutable state via `LazyLock<Regex>`

- **Description:** Regex objects are stored in `static` `LazyLock` variables. They are immutable after initialization and thread-safe. **Per-call isolation is NOT broken** because `Regex::is_match` / `captures_iter` do not mutate global state. However, the pattern does centralize regex compilation; a future maintainer might accidentally add a `static mut` or a cache here.
- **Suggested fix:** Document the invariants with a `// SAFETY: Regex is immutable; no per-call state.` comment above each static. Audit quarterly.
- **Test hint:** Concurrent stress test already exists in `tests/concurrent/`; ensure it asserts deterministic result counts across 1_000_000 iterations.

---

### 15. LOW | All string decoders | Unicode normalization is absent

- **Description:** No decoder applies NFC, NFD, NFKC, or NFKC. An attacker can obfuscate a string by encoding it in NFD (e.g., `e` + combining acute) and the scanner will emit raw bytes that differ from the NFC form used in signatures. This is a standard evasion technique.
- **Suggested fix:** After any string-based decode (JS unescape, JS eval payload, Python compile string, ROT13), normalize to **NFC** before returning bytes. NFC is the W3C recommendation for text interchange and matches most signature databases.
- **Test hint:** Input containing `\u0065\u0301` (NFD `é`) should decode to the NFC bytes `0xC3 0xA9`.

---

### 16. LOW / INFO | generic.rs | No dead code / commented-out code observed

- **Observation:** The crate is clean of TODOs, FIXMEs, and commented-out logic. Lint preamble is strict (`forbid(unsafe_code)`, `deny(unwrap_used, expect_used, todo, unimplemented)`). This is exemplary.

---

## Architecture Observations (non-blocking)

1. **Duplication in `layers.rs`:** `deobfuscate_auto` and `deobfuscate_js` share ~90 % of their logic. Extract a private `deobfuscate_with_set(deobfuscators, max_layers)` helper to satisfy LAW 2 (single responsibility, < 500 lines  -  currently 204 lines, so acceptable, but DRY still applies).
2. **`Deobfuscator` trait signature:** `fn deobfuscate(&self, input: &[u8]) -> Vec<DeobfuscatedLayer>` returns an owned `Vec` with no error channel. Consider changing to `fn deobfuscate(&self, input: &[u8], budget: &mut Budget) -> Vec<DeobfuscatedLayer>` so decoders can report timeout/budget exhaustion without panicking.
3. **`DeobfuscatedLayer` missing metadata:** No `truncated`, `elapsed`, or `output_len_limit_reached` flags. Callers cannot reason about result quality.

---

## Competitor Comparison

| Capability | `deobfuscate` (current) | Best-in-class expectation (e.g., `binwalk` / `CAPE` / `JSDetox`) |
|---|---|---|
| Per-layer output cap | Only Zlib (10 MiB) | All decoders capped |
| Timeout / budget | None | Per-layer wall-clock + memory budget |
| Binary payload support | Filtered out (printable-ASCII gate) | Returned raw for downstream analysis |
| Configurable depth | Hardcoded `MAX_LAYERS = 10` | Caller-configurable |
| UTF-8 expansion safety | `\xXX` corrupted via `char` | Raw-byte pipeline |
| Unicode normalization | None | NFC on all text exits |
| Cancellation / async | Blocking only | `CancellationToken` or async |

---

## Test Coverage Gaps

1. **Memory ceiling test:** No test asserts a hard RSS limit during decoding.
2. **Timeout test:** No test verifies that a pathological input is aborted within a deadline.
3. **Binary payload test:** No test checks that non-printable base64/hex decodes are returned.
4. **UTF-8 expansion test:** No test catches the `\xFF` → `0xC3 0xBF` bug.
5. **Truncation-flag test:** No test checks that zlib zip bombs are flagged as truncated.
6. **Normalization test:** No test verifies NFC output for combining-character sequences.

---

## File Checksum Reference (for reproducibility)

| File | Lines | SHA-256 (first 16 hex chars) |
|---|---|---|
| `src/lib.rs` | 36 | `a3f1…` *(not computed)* |
| `src/layers.rs` | 204 |  -  |
| `src/generic.rs` | 211 |  -  |
| `src/javascript.rs` | 285 |  -  |
| `src/python.rs` | 218 |  -  |

---

*End of audit. All findings are actionable. No band-aids recommended  -  fix the engine, never weaken the test.*
