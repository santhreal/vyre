# CRITIQUE  -  `attackstr/src/` Payload Generator

**Date:** 2026-04-23  
**Scope:** `libs/offensive/attackstr/src/` (read-only audit)  
**Rules:** LAWS 0–8, STANDARDS, RESEARCH PROTOCOL  
**Method:** Static analysis of source + tests + `Cargo.toml` + examples. Competitor baseline: `fuzzdb`, `secLists`, `payloadbox` payload generators.

---

## Executive Summary

`attackstr` has a solid TOML-driven grammar engine and good expansion safety (recursion limits, payload count caps). However, it fails the six hunt criteria across metadata architecture, payload classification, length bounding, encoding chain verification, WAF provenance, and exploit gating. Additionally, two dead-code files containing real exploit strings live in the repo, the `Payload` type lacks embedding-context tags, and three source files exceed the 500-line ceiling.

**Severity counts:** Critical 5 | High 8 | Medium 9 | Low 4

---

## Critical

### 1. `Payload` has no embedding-context tag  -  consumers cannot escape correctly
**SEVERITY:** CRITICAL | `src/lib.rs:566` | `src/grammar.rs:139`  
**Description:** The `Payload` struct carries `category`, `technique`, `context`, `encoding`, but **no field** indicates whether the payload is intended for a URL query string, JSON body, HTML attribute, JavaScript string, SQL statement, or shell command. A consumer receiving `"<script>alert(1)</script>"` cannot know whether to JSON-escape it (`\u003c`), URL-encode it, or inject it raw. This forces every downstream tool to guess, leading to false negatives when payloads are over-escaped or under-escaped.  
**Fix:** Add `target_media_type: Option<String>` or `escape_context: Vec<EmbeddingContext>` to `Payload` and `Context`. Values: `url_query`, `json_value`, `html_attribute`, `html_text`, `js_string`, `sql_string`, `shell_argument`, `header_value`, `xml_text`.  
**Test hint:** Load a grammar with `context.target_media_type = "json_value"`; assert the emitted `Payload` carries the tag; assert a downstream formatter JSON-escapes `<` to `\u003c`.

### 2. SQLi payloads are not classified by technique class
**SEVERITY:** CRITICAL | `src/ports/sqli.rs:8` | `src/grammar.rs:162`  
**Description:** `legacy_sqli_payloads()` returns a flat `Vec<&'static str>` with no classification. The generic `Technique` has `tags: Vec<String>` (unstructured), but there is **no enum or guaranteed vocabulary** for `time-based`, `error-based`, `boolean-based`, `union-based`, `stacked-query`, or `out-of-band`. A scanner cannot programmatically pick the right class for a given injection point (e.g., boolean-based for blind, union-based for visible output).  
**Fix:** Add `technique_class: Option<SqliClass>` to `Technique` (or a generic `attack_class: Option<String>` with a documented vocabulary). Port `legacy_sqli_payloads()` into a grammar with explicit `technique_class` tags.  
**Test hint:** Load a grammar with `technique_class = "time-based"`; assert `db.payloads("sql-injection")` can be filtered by that field.

### 3. No per-payload length ceiling  -  malformed corpus can DoS downstream
**SEVERITY:** CRITICAL | `src/grammar.rs:268` | `src/mutate.rs:228` | `src/lib.rs:354`  
**Description:** `MAX_TEMPLATE_LENGTH` caps template expansion at 256 KB, but:
- Encodings can **expand** that (hex triples size; `mutate_encoding_mix` concatenates two encoded halves).
- `mutate_all()` accepts arbitrary input and produces output ≥ input length; the 1 MB test in `break_it.rs` explicitly validates that a 1 MB input survives mutation without truncation.
- `PayloadConfig` has `max_per_category` (count limit, 0 = unlimited) but **no `max_payload_length`**.
A malicious grammar with a 256 KB template + hex encoding + `mutate_all` can emit a multi-MB payload that exhausts downstream HTTP client buffers, WAF regex engines, or logging pipelines.  
**Fix:** Add `max_payload_length: usize` to `PayloadConfig` (default 100 KB). Enforce it in `GrammarExpansionIter::next()` after encoding and again after marker injection. Error with `TemplateExpansionError::ExpansionLengthExceeded`.  
**Test hint:** Grammar with 200 KB template + hex encoding; config `max_payload_length = 100_000`; assert expansion errors.

### 4. Encoding round-trip chains are untested and unsupported
**SEVERITY:** CRITICAL | `src/encoding.rs:135` | `tests/unit/encoding.rs:157`  
**Description:** The crate exposes 19 single-step encodings but **no chaining API** and **no round-trip tests**. The user specifically asked: does every payload survive URL→base64→gzip→URL without corruption? There is no `chain_encoding` function, no gzip transform, and no test verifying composite pipelines. The `encodex` dependency (internal path `../../performance/compression/encodex`) is only used for base64; gzip is not exposed. Competitors (`fuzzdb`, `SecLists`) ship payloads pre-tested through common proxy/WAF chains.  
**Fix:** Add `chain_encoding(payload: &str, transforms: &[&str]) -> String` and a `gzip` built-in. Add adversarial tests for `url→base64→gzip→url` and `hex→unicode→url` with assert-eq on decoded output.  
**Test hint:** Proptest: for all `s` in `\PC*`, `decode(chain(encode(s))) == s` for every reversible chain.

### 5. Real exploit payloads shipped without `exploits` feature flag
**SEVERITY:** CRITICAL | `src/ports/cmdi.rs:8` | `src/ports/sqli.rs:9` | `examples/toml_config.rs:18` | `examples/basic.rs:24`  
**Description:** The repo ships working exploit strings:
- `ports/cmdi.rs`: `; id`, `| id`, `$(id)`, `` `id` ``
- `ports/sqli.rs`: `' OR 1=1--`, `" OR 1=1--`, `admin' --`
- `examples/toml_config.rs`: `; ping -c 1 {host}`
- `examples/basic.rs`: `<svg/onload=alert(1)>`
There is **no `exploits` feature flag** in `Cargo.toml`. Accidental inclusion of this crate into a library corpus (e.g., a build pipeline, a dependency tree, a container image scan) ships working attacks. Competitor crates (e.g., `rust-fuzz`, `boofuzz`) gate real exploit strings behind non-default features.  
**Fix:** Add `exploits` feature to `Cargo.toml` (non-default). Wrap `legacy_cmdi_payloads()`, `legacy_sqli_payloads()`, and any example/grammar containing real PoC strings in `#[cfg(feature = "exploits")]`. Gate the `ports` module behind the feature.  
**Test hint:** `cargo test` without `exploits` feature skips exploit payloads; `cargo test --features exploits` includes them.

---

## High

### 6. `ports/cmdi.rs` and `ports/sqli.rs` are dead code in the source tree
**SEVERITY:** HIGH | `src/ports/cmdi.rs:1` | `src/ports/sqli.rs:1`  
**Description:** `lib.rs` never declares `mod ports;`, so these files are **not compiled** but remain in the repo. They violate LAW 1 (no stubs  -  these are literal unreferenced stubs) and LAW 7 (no dead code). Worse, they contain real exploit strings (see Finding 5) that are invisible to `cargo test` but visible to grep, SCA scanners, and compliance audits.  
**Fix:** Either (a) wire `mod ports` into `lib.rs` and gate behind `feature = "exploits"`, or (b) delete the files.  
**Test hint:** `find src -name "*.rs" | xargs grep -l "mod ports"` must return a result after fix.

### 7. `PayloadError` serialization is lossy  -  round-trip breaks
**SEVERITY:** HIGH | `src/lib.rs:696`  
**Description:** `Serialize` writes `{"kind":"...", "message":"..."}`. `Deserialize` reconstructs **only** `PayloadError::Io` with a concatenated string. A `GrammarValidation` error round-tripped through JSON becomes an opaque `Io` error. This violates the principle that serialization must be reversible and breaks any distributed system caching `PayloadError`.  
**Fix:** Implement `Deserialize` as a tagged enum that matches the `kind` field and reconstructs the correct variant with parsed fields. Or remove `Serialize`/`Deserialize` from `PayloadError` if round-tripping is not needed.  
**Test hint:** Serialize `PayloadError::GrammarValidation { ... }` to JSON, deserialize back, assert matches original variant.

### 8. `apply_encoding` silently swallows unknown transforms
**SEVERITY:** HIGH | `src/encoding.rs:135`  
**Description:** Unknown transform names log a `tracing::warn!` and return the input unchanged. This is **error swallowing**. A grammar author typing `"transform = "url_enocde"` (typo) will generate raw payloads thinking they are URL-encoded. WAF bypasses will fail silently. Standard says: "Never swallow errors. Fail fast with context."  
**Fix:** Change `apply_encoding` to return `Result<String, EncodingError>` where `EncodingError::UnknownTransform` is returned for unrecognized names. Update `GrammarExpansionIter` to propagate the error.  
**Test hint:** Grammar with `transform = "typo"`; assert `load_toml` returns `PayloadError` containing "unknown transform".

### 9. Deduplication ignores payload metadata  -  loses technique/confidence diversity
**SEVERITY:** HIGH | `src/loader.rs:567`  
**Description:** `PayloadIter` deduplicates using `HashSet<String>` keyed **only** on `expanded_payload.text`. Two techniques producing the same text but different `confidence` or `expected_pattern` will be silently dropped. A scanner that wants high-confidence payloads first may receive the wrong metadata.  
**Fix:** Deduplicate on the full `ExpandedPayload` (text + technique + context + encoding + confidence + expected_pattern), not just text. If dedup-by-text is desired, make it explicit in config (`deduplicate_strategy`).  
**Test hint:** Two techniques with same template but `confidence = 0.9` and `confidence = 0.5`; assert both payloads emitted when dedup is full-metadata.

### 10. Three source files exceed 500-line ceiling
**SEVERITY:** HIGH | `src/lib.rs:802` | `src/loader.rs:641` | `src/grammar.rs:661`  
**Description:** LAW 2 mandates every file < 500 lines, single responsibility. `lib.rs` (802) mixes public API, `Payload` struct, `PayloadConfig`, `StaticPayloads`, `PayloadError`, and trait definitions. `loader.rs` (641) mixes directory loading, TOML parsing, grammar validation, expansion, caching, marker injection, and iteration. `grammar.rs` (661) mixes data types, template expansion, and iterator state machines.  
**Fix:** Split `lib.rs` into `payload.rs`, `error.rs`, `config.rs` (config already exists but lib.rs still has config logic). Split `loader.rs` into `db.rs`, `expand.rs`, `cache.rs`. Split `grammar.rs` into `types.rs` and `expand.rs`.  
**Test hint:** `find src -name "*.rs" -exec wc -l {} + | awk '$1 > 500 {print}'` must be empty.

### 11. `StaticPayloads::add()` sorts entire vector on every insertion
**SEVERITY:** HIGH | `src/lib.rs:235`  
**Description:** `add()` pushes one element then calls `sort_payloads_by_category` on the entire `Vec`, rebuilding `category_ranges` from scratch. This is O(n log n) per insertion. A loop adding 10,000 payloads is ~O(n² log n).  
**Fix:** Maintain a `BTreeMap<String, Vec<Payload>>` or batch-insert then sort once. If single-add is needed, insert into the correct category vector directly.  
**Test hint:** Benchmark `add()` in a loop of 10,000 iterations; assert < 50 ms.

### 12. `load_single_grammar_file` reads unbounded files into memory
**SEVERITY:** HIGH | `src/loader.rs:252`  
**Description:** `std::fs::read_to_string(file_path)` is called without a file-size check. A 10 GB `.toml` file will be read entirely into RAM before parsing, causing OOM. There is no `max_grammar_file_size` in `PayloadConfig`.  
**Fix:** Check `std::fs::metadata(file_path)?.len()` against a config limit (default 10 MB) before reading. Return `PayloadError::Io` with "Fix: grammar file exceeds N bytes" if oversized.  
**Test hint:** Create a temp file of 20 MB; assert `load_dir` rejects it with a descriptive error.

### 13. `builtin-grammars` feature is an explicit no-op stub
**SEVERITY:** HIGH | `Cargo.toml:16`  
**Description:** The feature is documented as "currently a no-op in this crate version until embedded grammars are added, so enabling it is forward-compatible but has no effect." This is a **stub**. LAW 1: "If you can't implement it fully, DELETE it." Forward-compatible no-ops confuse consumers and CI pipelines that enable the feature expecting behavior.  
**Fix:** Delete the `builtin-grammars` feature from `Cargo.toml` and all documentation. Re-introduce it only when grammars are actually embedded.  
**Test hint:** `grep -r "builtin-grammars" .` must return nothing.

---

## Medium

### 14. WAF-evasion mutations return naked strings  -  no provenance tags
**SEVERITY:** MEDIUM | `src/mutate.rs:228`  
**Description:** `mutate_all()` returns `Vec<String>`. The caller cannot tell whether a variant came from `mutate_case`, `mutate_whitespace`, `mutate_null_bytes`, `mutate_sql_comments`, `mutate_html`, or `mutate_unicode`. When a bypass succeeds, the consumer cannot report *which* evasion class worked, hindering WAF fingerprinting and rule generation.  
**Fix:** Return `Vec<MutatedPayload>` where `MutatedPayload { text: String, strategy: MutationStrategy }` and `MutationStrategy` is an enum (`CaseLower`, `WhitespaceTab`, `NullBytePrefix`, `SqlComment`, `HtmlTagUpper`, `UnicodeFullwidth`, …).  
**Test hint:** `mutate_all("UNION SELECT")`; assert every variant has a non-empty `strategy` tag.

### 15. `expected_pattern` is stored as raw string  -  not validated as regex
**SEVERITY:** MEDIUM | `src/grammar.rs:178`  
**Description:** `Technique.expected_pattern: Option<String>` accepts any string. An invalid regex like `[unclosed` will be stored silently and will panic at the consumer when compiled with `regex::Regex::new()`.  
**Fix:** Validate `expected_pattern` at grammar load time using `regex::Regex::new()` (or a lightweight regex parser). If invalid, return `GrammarValidation` error.  
**Test hint:** Grammar with `expected_pattern = "[invalid"`; assert `load_toml` returns validation error.

### 16. `html_encode` escapes only 5 characters  -  misses backtick and slash
**SEVERITY:** MEDIUM | `src/encoding.rs:389`  
**Description:** `html_encode` escapes `& < > " '`. It does **not** escape backtick `` ` `` (legacy IE vector) or forward slash `/` (used to break out of `<script>` contexts when `</script>` is filtered). Modern XSS lists (e.g., OWASP XSS Filter Evasion Cheat Sheet) treat both as dangerous.  
**Fix:** Escape `` ` `` to ``&#96;`` and `/` to `&#47;` (or at minimum provide a `html_encode_strict` variant).  
**Test hint:** `apply_encoding("`</script>", "html_entities")` must not contain raw backtick or raw `</script>`.

### 17. `CustomEncoder` uses function pointer  -  closures cannot capture state
**SEVERITY:** MEDIUM | `src/encoding.rs:67`  
**Description:** `CustomEncoder` stores `fn(&str) -> String`, not `Box<dyn Fn(&str) -> String>`. This prevents closures from capturing environment (e.g., a salt, a key, or a configuration flag). The trait `Encoder` is correctly generic (`impl<F> Encoder for F where F: Fn(&str) -> String`), but the concrete `CustomEncoder` type is needlessly restrictive.  
**Fix:** Change `CustomEncoder` to hold `Box<dyn Fn(&str) -> String + Send + Sync>`.  
**Test hint:** `let salt = "abc".to_string(); db.register_encoding("salted", |s| format!("{salt}{s}"));` must compile.

### 18. `validate_encodings` warns instead of erroring on unknown transforms
**SEVERITY:** MEDIUM | `src/validate.rs:193`  
**Description:** An unknown encoding transform produces a `Warning`-level issue, but `load_single_grammar_file` only rejects `Error`-level issues. The grammar loads successfully and the payload passes through **raw**, while the author believes it is encoded. This is a silent security regression.  
**Fix:** Upgrade unknown encoding transforms to `IssueLevel::Error` in `validate_encodings`.  
**Test hint:** Grammar with `transform = "rot47"`; assert `load_toml` fails with `GrammarValidation` error.

### 19. `depluralize` is broken for common English words
**SEVERITY:** MEDIUM | `src/grammar.rs:653`  
**Description:** `depluralize("ss")` → `"s"`, `depluralize("status")` → `"statu"`, `depluralize("children")` → `"childre"`. While documented as "simple," it is used for variable name matching. A grammar using `"payloads"` vs `"payload"` may fail to resolve.  
**Fix:** Use a correct English depluralization table or switch to requiring exact variable names and remove automatic pluralization.  
**Test hint:** `assert_eq!(depluralize("status"), "status"); assert_eq!(depluralize("children"), "children");`

### 20. `tracing::warn!` for unknown encodings may be silently swallowed
**SEVERITY:** MEDIUM | `src/encoding.rs:140`  
**Description:** Libraries should not rely on a global `tracing` subscriber being installed. In many consumer apps (tests, CLIs without subscriber setup), the warning is dropped into the void. The error is effectively swallowed.  
**Fix:** Return `Result<String, EncodingError>` (see Finding 8) so the caller is forced to handle the failure.

### 21. `GrammarExpansionIter::new` eagerly validates all context×technique combinations
**SEVERITY:** MEDIUM | `src/grammar.rs:334`  
**Description:** The constructor iterates `contexts × techniques` and instantiates a `TemplateExpansionIter` for each pair to validate it. For a grammar with 100 contexts and 100 techniques, this creates 10,000 iterators at load time. While not a bug per se, it is an unexpected O(n×m) cost at load time that could stall startup.  
**Fix:** Move validation to a separate `validate_expansion()` method, or perform lightweight validation (brace balance check only) in the constructor and defer full expansion validation to the first iteration.  
**Test hint:** Load a grammar with 100×100 context/technique pairs; assert load time < 100 ms.

### 22. `Cargo.toml` contains stray `[workspace]` declaration
**SEVERITY:** MEDIUM | `Cargo.toml:60`  
**Description:** A `[workspace]` key in a leaf crate's `Cargo.toml` can conflict with the root workspace or cause unexpected resolver behavior when the crate is used as a path dependency.  
**Fix:** Remove `[workspace]` from `attackstr/Cargo.toml` unless this crate is intentionally a workspace root.

---

## Low

### 23. `mutate_html` hardcodes HTML tag list
**SEVERITY:** LOW | `src/mutate.rs:154`  
**Description:** The tag list `["script", "img", "svg", ...]` is hardcoded. New tags (e.g., `video`, `source`, `portal`) require a code change. No TOML configuration or extensibility hook exists.  
**Fix:** Accept `tags: &[&str]` as a parameter, or load tag lists from `PayloadConfig`.  
**Test hint:** `mutate_html_with_tags("<video>", &["video"])` produces variants.

### 24. `load_dir` skips `.TOML` (uppercase) on case-sensitive filesystems
**SEVERITY:** LOW | `src/loader.rs:214`  
**Description:** The extension check is `entry.path().extension().and_then(|s| s.to_str()) == Some("toml")`. Files named `GRAMMAR.TOML` are silently skipped on Linux.  
**Fix:** Compare lowercase extension: `s.to_str().map(|e| e.eq_ignore_ascii_case("toml")) == Some(true)`.  
**Test hint:** Create `GRAMMAR.TOML` in temp dir; assert it loads.

### 25. `confidence: f64` lacks `NaN` guard on `Payload` deserialization
**SEVERITY:** LOW | `src/lib.rs:566`  
**Description:** `Technique` validates confidence on deserialization (`deserialize_confidence`), but `Payload` (which also has `confidence: f64`) does not. A `Payload` serialized from an external source could carry `NaN`, breaking `PartialEq` and `Hash` (which rely on `to_bits()`).  
**Fix:** Add a custom `deserialize_confidence` to `Payload` or normalize `NaN` to `0.0` in a constructor.

### 26. Multiple `#[allow(clippy::...)]` suppressions hide real issues
**SEVERITY:** LOW | `src/lib.rs:12`  
**Description:** `missing_errors_doc`, `needless_borrows_for_generic_args`, `unused_self`, `iter_filter_is_ok`, `float_cmp`, `cast_sign_loss`, `unnecessary_wraps` are all suppressed crate-wide. Many of these point to genuine code-quality issues (e.g., `float_cmp` on `confidence` comparison, `missing_errors_doc` on public functions).  
**Fix:** Remove blanket allows. Fix individual lints or add targeted `#[allow(...)]` at the specific line with a comment justifying it.

---

## Competitor Comparison

| Capability | `attackstr` | `fuzzdb` | `SecLists` | `payloadbox` |
|---|---|---|---|---|
| TOML grammar engine | ✅ | ❌ | ❌ | ❌ |
| Context tagging (URL/JSON/HTML/SQL) | ❌ | ❌ | ❌ | ❌ |
| SQLi class enum (time/error/boolean/union) | ❌ | ❌ | ❌ | ❌ |
| Per-payload length cap | ❌ | ❌ | ❌ | ❌ |
| Encoding chain + round-trip tests | ❌ | N/A | N/A | N/A |
| Mutation provenance tags | ❌ | N/A | N/A | N/A |
| Exploit feature gating | ❌ | ✅ (text files, not compiled) | ✅ (text files) | ❌ |
| Grammar validation at load | ✅ | N/A | N/A | N/A |
| Custom encoding registration | ✅ | ❌ | ❌ | ❌ |

`attackstr` leads in architecture (grammar engine, custom encodings) but lags in **consumer safety** (no context tags, no length caps, no exploit gating). The gap is architectural: metadata on `Payload` is too thin for a security tool that needs to know *where* and *how* to inject.

---

## Recommended Priority Order

1. **Delete or gate `ports/` dead code** (Finding 5 + 6)  -  compliance risk, immediate.
2. **Add `exploits` feature flag** (Finding 5)  -  compliance risk.
3. **Add `target_media_type` / `escape_context` to `Payload`** (Finding 1)  -  unlocks correct downstream usage.
4. **Add `max_payload_length` to `PayloadConfig`** (Finding 3)  -  prevents DoS.
5. **Return `Result` from `apply_encoding`** (Finding 8)  -  fail fast.
6. **Split files > 500 lines** (Finding 10)  -  enables long-term maintainability.
7. **Implement encoding chains + round-trip tests** (Finding 4)  -  closes the test gap.
8. **Add structured `technique_class`** (Finding 2)  -  enables intelligent scanner selection.
9. **Add mutation provenance tags** (Finding 14)  -  enables WAF fingerprinting.
10. **Fix `PayloadError` serialization** (Finding 7)  -  correctness.
