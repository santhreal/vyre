# polyglot/src Security & Architecture Critique

**Date:** 2026-04-23  
**Scope:** `libs/scanner/polyglot/src/*.rs` (read-only audit)  
**Methodology:** Static analysis against magic-byte conflicts, confidence scoring, chain depth, memory budgets, cross-invocation state corruption, and malformed header handling. Compared against best-in-class file-type detection (libmagic, file(1), Apache Tika).

---

## Executive Summary

The `polyglot` crate implements flat substring matching via Aho-Corasick but lacks all architectural primitives required for robust polyglot detection: no offset-aware magic-byte validation, no confidence scoring, no container recursion, no input size enforcement, and a **critically broken TOML configuration path for binary signatures**. At internet scale, every finding here translates to false negatives (evasion) or false positives (alert fatigue / DoS).

---

## Findings

### 1. CRITICAL | config.rs:22 | Binary signatures corrupted by UTF-8 TOML string encoding

**Description:** `RuleSignature::pattern` is a `String`. When deserialized from TOML, Unicode escapes such as `\u0089` are decoded into Rust `String` (valid UTF-8). `as_bytes()` therefore returns the **UTF-8 encoding** of the code point, not the raw byte. Example: `\u0089` → `[0xC2, 0x89]` instead of `[0x89]`. This makes the community-extensible TOML ruleset **fundamentally incapable** of expressing the PNG, JPEG, or any non-ASCII binary signature correctly. The doc-comment example in `lib.rs:103` itself demonstrates the broken pattern.

**Fix:** Change `pattern` to a `Vec<u8>` and accept TOML arrays of integers (e.g., `pattern = [0x89, 0x50, 0x4E, 0x47]`) or base64-encoded strings. Reject String-based patterns for binary data.

**Test hint:** Load a TOML rule with `pattern = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]` and assert the compiled bytes match the raw PNG signature exactly.

---

### 2. CRITICAL | detect.rs:86 | No offset-aware magic-byte validation (magic bytes match anywhere)

**Description:** The scanner uses `find_overlapping_iter` over the **entire** input. A JPEG signature (`\xFF\xD8\xFF`) embedded at offset 500,000 inside a 1 GB PE resource section marks the file as `Jpeg+Pe` → `Risk::High`. This is not a polyglot; it is a binary containing an image resource. Competitors (libmagic) check magic bytes at offset 0 or known trailer offsets, not via global substring search.

**Fix:** Distinguish **header signatures** (offset 0) from **heuristic keywords**. Only flag binary formats (PNG, JPEG, GIF, ZIP, ELF, PE, PDF) if the magic byte sequence appears at offset 0. For container formats, optionally parse structure to confirm validity before marking.

**Test hint:** Craft a PE file with `GIF89a` embedded in `.rsrc` at offset > 0x1000. Assert `detect()` returns only `Pe`, not `Gif+Pe`.

---

### 3. HIGH | signatures.rs:90 | PE signature `MZ` is only 2 bytes  -  extreme false-positive rate

**Description:** The PE signature is `b"MZ"` (2 bytes). English text contains "MZ" constantly ("amazing", "zoom", "MZero"). At internet scale, a 2-byte prefix is indistinguishable from noise. libmagic uses `MZ` at offset 0 **plus** a valid DOS stub and PE header pointer.

**Fix:** Require at least the DOS stub length check (`e_lfanew` pointer within bounds) and the `PE\0\0` signature at the pointed offset. Validate at least 64 bytes of DOS header structure.

**Test hint:** Feed the detector `b"amazing MZ explosion"` and assert it does **not** flag `Pe`.

---

### 4. HIGH | detect.rs:86 / lib.rs:122 | `PolyglotError::SizeLimitExceeded` defined but never enforced  -  DoS vector

**Description:** `error.rs:18-24` defines a size-limit error, yet `PolyglotDetector::detect(&self, data: &[u8])` accepts arbitrarily large slices with no length check. An attacker can submit a multi-gigabyte stream, forcing an O(N) Aho-Corasick pass with no cancellation point.

**Fix:** Add a `max_input_size: usize` field to `PolyglotDetector` (default e.g., 100 MiB). Return `PolyglotError::SizeLimitExceeded` before scanning if `data.len() > max_input_size`. Expose a builder method `with_max_input_size()`.

**Test hint:** Call `detect()` on a `vec![0u8; 101 * 1024 * 1024]` and assert `SizeLimitExceeded` is returned. Update `detect()` to return `Result<PolyglotResult, PolyglotError>`.

---

### 5. HIGH | signatures.rs:94-104 | JS keywords `var `, `let `, `const ` are natural-language substrings

**Description:** The JS heuristic keywords include `b"var "`, `b"let "`, and `b"const "`. These are common English words. A plain-text README containing "let us begin" or "constant variable" will be flagged as `Js+Html` or `Js+Pdf`, elevating risk to CRITICAL or HIGH falsely.

**Fix:** Require a minimum of **2 distinct** JS keyword matches **and** at least one execution-context keyword (`eval(`, `document.write`, `window.`, `setTimeout`, `console.log`). Or, require a structural JS parse context (e.g., `function foo() {` with balanced braces).

**Test hint:** `detector.detect(b"let us consider a constant value")` must not contain `Format::Js`.

---

### 6. HIGH | detect.rs:86-97 | No confidence score  -  single substring = full format conviction

**Description:** There is no confidence metric. One occurrence of `<html` anywhere marks the file as HTML with 100% certainty. There is no position weighting (header vs body vs trailer), no minimum match density, and no structural corroboration. Competitors (libmagic) use a weighted scoring system with multiple tests per format.

**Fix:** Introduce a `Confidence` enum (`None`, `Weak`, `Strong`, `Certain`). Header magic at offset 0 = `Certain`. Heuristic keywords require multiple matches or structural validation for `Strong`. Only formats with `Confidence >= Strong` should contribute to `is_polyglot` and risk calculation.

**Test hint:** Scan a 1 MiB file with exactly one `<html` at a random offset. Assert `Html` confidence is `Weak` and the file is **not** flagged as a polyglot.

---

### 7. MEDIUM | detect.rs:86 / lib.rs:122 | No container parsing  -  flat matching with zero chain-detection depth

**Description:** ZIP and PDF are container formats. The scanner performs a single flat pass with no recursion, no embedded stream extraction, and therefore no depth limit. A PDF with an embedded ZIP stream is detected as `Pdf+Zip` (HIGH risk) even though it is a single-format container carrying an attachment. True polyglot detection requires understanding whether two formats occupy the **same** byte ranges or are nested.

**Fix:** Implement a `FormatParser` trait with `parse(data: &[u8]) -> Vec<EmbeddedStream>`. For ZIP, enumerate local file headers. For PDF, enumerate object streams. Recursively scan embedded streams with a configurable `max_depth` (default 3). Increment depth counter per recursion; abort with `PolyglotError::MaxDepthExceeded`.

**Test hint:** Create a valid PDF with an embedded `PK\x03\x04` object stream. Assert `detect()` returns only `Pdf` (or `Pdf` with an `embedded: [Zip]` metadata field), not a polyglot.

---

### 8. MEDIUM | risk.rs:45-89 | SVG treated as benign image  -  ignores executable vector

**Description:** `calculate_risk()` does not treat SVG as a code-bearing format. SVG supports inline `<script>` tags, `onload` event handlers, and XSS vectors natively. `Svg + Js` should be `CRITICAL`; `Svg + Html` should be at least `HIGH`.

**Fix:** Add `has_svg` to the risk matrix. Treat SVG as both image and code vector: `svg && (js || html)` → `Critical`.

**Test hint:** `calculate_risk(&[Format::Svg, Format::Js])` must equal `Risk::Critical`.

---

### 9. MEDIUM | config.rs:69-78 | Unknown format strings silently dropped

**Description:** `into_compiled_patterns()` ignores unknown format names via `if let Some(format) = Format::parse(&format_str)`. A typo like `jpegg` or a user-defined `wasm` rule is swallowed without error. This violates "Never swallow errors."

**Fix:** Return `Result<Vec<(Format, Vec<u8>)>, PolyglotError>` and emit `PolyglotError::UnknownFormat(String)` for unrecognised keys.

**Test hint:** Parse TOML with `[signatures]\nunknownfmt = [{ pattern = "AB" }]` and assert `UnknownFormat("unknownfmt")` error.

---

### 10. MEDIUM | detect.rs:47-49 | `Scanner::new()` panics on invariant violation instead of returning `Result`

**Description:** If the static default signatures are somehow corrupted, `Scanner::new()` panics via `.expect("Failed to build default Aho-Corasick automaton")`. Library constructors should not panic on build failures; they should return `Result` so callers can handle the error gracefully.

**Fix:** Change `Scanner::new()` to return `Result<Self, PolyglotError>` and propagate the `BuildError`. Update `PolyglotDetector::new()` accordingly. (Note: this is an API-breaking change; do it now, not later.)

**Test hint:** Unit-test that `Scanner::new()` returns `Ok` under normal conditions. Mock an invalid pattern set and assert `PatternCompileError`.

---

### 11. MEDIUM | signatures.rs:85-86 | ZIP signature missing empty-archive and spanned-archive variants

**Description:** Only `PK\x03\x04` (local file header) is recognised. Empty ZIP archives start with `PK\x05\x06`, and spanned archives start with `PK\x07\x08`. These valid ZIP files will evade detection entirely.

**Fix:** Add signatures for `b"PK\x05\x06"` and `b"PK\x07\x08"`, mapping to `Format::Zip`.

**Test hint:** Create a 22-byte empty ZIP archive (EOCD only, starting with `PK\x05\x06`). Assert `Format::Zip` is detected.

---

### 12. MEDIUM | detect.rs:86-97 | No per-format or per-scan memory budget / cancellation token

**Description:** The scanner iterates over the entire `&[u8]` with no way to abort mid-scan. For streaming or chunked inputs, there is no `max_scanned_bytes` or timeout. Aho-Corasick itself does not allocate per match, but the caller cannot cap the work done.

**Fix:** Add a `scan_budget: ScanBudget` struct (max bytes, max matches, max elapsed time). Return a `BudgetExceeded` error or partial result if limits are hit.

**Test hint:** Scan a 1 GiB file of `0x00` with `max_matches = 1`. Assert the scan terminates immediately after the first match without iterating the remaining bytes.

---

### 13. LOW | signatures.rs:77 | PDF signature `%PDF-` not validated beyond 5 bytes

**Description:** The PDF magic is `%PDF-` (5 bytes). No version number validation (`1.0`–`2.0`), no trailer `%%EOF` check, no cross-reference table validation. A 5-byte file `%PDF-` is treated as a valid PDF.

**Fix:** Require at least `%PDF-1.[0-9]` or `%PDF-2.[0-9]` at offset 0. Optionally verify `%%EOF` exists within the last 1024 bytes.

**Test hint:** `detector.detect(b"%PDF-")` must not contain `Format::Pdf`.

---

### 14. LOW | signatures.rs:75-91 | No malformed-header validation for any binary format

**Description:** Every binary format is triggered by magic bytes alone with no structural sanity checks:
- **JPEG:** No `SOF0`/`SOF2` marker, no `EOI` (`\xFF\xD9`), no dimension validation.
- **PNG:** No `IHDR` chunk, no CRC32 check.
- **GIF:** No Logical Screen Descriptor, no `0x3B` trailer.
- **ELF:** No `e_ident[EI_CLASS]`, `e_machine`, or `e_version` validation.
- **ZIP:** No local file header structure validation (compression method, file name length).

At scale, random data or truncated files will falsely trigger these formats if the magic bytes happen to appear.

**Fix:** Implement minimal `Validator` trait per format. Example: `JpegValidator` checks for `\xFF\xD8` at 0 and `\xFF\xD9` somewhere after offset 2. Run validators **after** Aho-Corasick flagging to filter false positives.

**Test hint:** `detector.detect(b"\xFF\xD8\xFF\x00\x00")` (JPEG SOI with no valid markers) must not contain `Format::Jpeg`.

---

### 15. LOW | signatures.rs:106-114 | HTML keyword `<script>` triggers inside non-HTML contexts

**Description:** `<script>` is a valid HTML signature, but it also appears inside SVG, XML, and XHTML documents. The flat matcher will flag `Html` for an SVG file containing `<script>`, potentially masking the true format or elevating risk incorrectly.

**Fix:** If SVG is detected, require HTML-specific structural markers (`<!DOCTYPE html>`, `<html>`, `<head>`) before also marking `Html`. Do not conflate generic XML tags with HTML.

**Test hint:** `detector.detect(b"<svg><script>alert(1)</script></svg>")` must return only `Svg`, not `Html`.

---

## Cross-Cutting Architectural Gaps

| Concern | Status | Notes |
|---------|--------|-------|
| **Magic prefix conflicts** | Partial | No two signatures share identical prefixes, but `MZ` (2 bytes) is effectively noise. |
| **Inner vs outer confidence** | **Missing entirely** | No scoring system; all matches are boolean. |
| **Chain-detection depth limit** | **Missing entirely** | No container parsing, no recursion, no depth concept. |
| **Per-format memory budget** | **Missing entirely** | No streaming, no match limits, no cancellation. |
| **Static state corruption** | Clean | All statics are immutable `&'static` slices. No `Mutex`, `OnceLock`, or `lazy_static` with interior mutability found. |
| **Malformed header handling** | **Missing entirely** | Zero structural validation beyond magic-byte substring presence. |

---

## Competitor Comparison

| Feature | `polyglot` (current) | `libmagic` / `file(1)` | Apache Tika |
|---------|----------------------|------------------------|-------------|
| Offset-aware magic | ❌ | ✅ (offset 0, plus indirect offsets) | ✅ |
| Confidence scoring | ❌ | ✅ (strength/weakness ratings) | ✅ (mime-confidence) |
| Container recursion | ❌ | ✅ (zip, tar, ole2, etc.) | ✅ |
| Depth limits | ❌ | ✅ | ✅ |
| Header validation | ❌ | ✅ (multi-test per type) | ✅ (parsers) |
| Configurable size limits | ❌ | ✅ (via `MAGIC_PARAM_BYTES_MAX`) | ✅ |
| Binary-safe config | ❌ (UTF-8 TOML strings) | ✅ (compiled magic db) | ✅ (Java API) |

---

## Conclusion

The crate is a functional **prototype** but not production-grade polyglot detection. The most urgent fix is the **TOML binary-signature corruption** (Finding 1), followed by **offset-aware magic validation** (Finding 2) and **size-limit enforcement** (Finding 4). Without these, the crate will generate false positives at scale and fail to correctly load community signatures.

*No mutable static state was found; cross-invocation corruption risk is zero under the current immutable design.*
