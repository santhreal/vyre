# CRITIQUE  -  pocgen src/ (2026-04-23)

**Scope:** `libs/offensive/pocgen/src/` (read-only audit)  
**Hunt targets:** (1) exploit PoC gating, (2) template injection escape, (3) reproducibility, (4) auth/session handling, (5) retry + timeout, (6) CVE database staleness.

---

## 1. Exploit PoC Gating Behind Feature Flag

### Finding 1.1  -  CRITICAL | `Cargo.toml:1` / `src/lib.rs:36`
**Exploit payloads ship unconditionally; no compile-time or runtime safety gate.**

The `exploits` module is `pub mod exploits` (always compiled), `default_exploits.toml` is baked into the binary via `include_str!`, and the public API exposes `SqliExploiter`, `OpenRedirectExploiter`, and SSRF-oriented chain rules (`169.254.169.254` cloud metadata probes) with zero opt-out mechanism. There is no `dangerous-exploits`, `unsafe-poc`, or `skip-exploits` Cargo feature, and no runtime permission gate.

**Fix:** Introduce a `dangerous-exploits` Cargo feature (disabled by default). Wrap the entire `exploits` module, `default_exploits.toml` embedding, and all attack-chain helpers in `#[cfg(feature = "dangerous-exploits")]`. Expose a stub that returns `PocError::GatedFeature` when the feature is off.

**Test hint:** Build with `--no-default-features` and assert that `SqliExploiter` is unavailable and `generate_attack_chain` returns `GatedFeature` for exploit-oriented rules.

---

## 2. Template Injection Escape

### Finding 2.1  -  CRITICAL | `src/chain.rs:168`
**Unescaped template substitution in `ExtractedPath` enables path traversal and request-line injection.**

```rust
let path = path_template.replace(&format!("{{{{{field}}}}}"), &val);
```

`val` comes from `finding.extracted()[field]` and is inserted raw. A crafted extracted value of `../../../etc/passwd` or `\r\nPOST /evil` bypasses any URL validation performed later by `base.join(&path)` because `join` only normalizes *path* segments; a value starting with `?` or containing `\r\n` corrupts the resulting request line or injects additional HTTP requests.

**Fix:** After substitution, re-parse the result as a `Url` path segment and reject any string containing `\r`, `\n`, or NUL. Alternatively, percent-encode the substituted value before insertion, or validate `val` against `[^\x00-\x1f\x7f]` before replacement.

**Test hint:** Proptest with extracted values containing `../../../`, `\r\n`, `\x00`, and `?query=1`. Assert `build_chain_step` returns `None` or an error.

### Finding 2.2  -  HIGH | `src/exploits.rs:225`
**Naive `String::replace` acts as a global template engine with no occurrence limit.**

```rust
let replaced = redirect_url.replace("https://evil.com", attacker_domain);
```

If `redirect_url` contains the placeholder string in multiple positions (e.g., `https://evil.com?ref=https://evil.com`), both are replaced. There is no single-target guarantee. Additionally, `attacker_domain` is not validated as a valid origin, allowing injection of fragments or query strings that alter the semantic meaning of the final URL.

**Fix:** Use a single-replacement API (`replacen(..., 1)`) or a typed template struct. Validate `attacker_domain` with `Url::parse` and ensure it has a host and scheme before substitution.

**Test hint:** Pass `redirect_url = "https://evil.com?ref=https://evil.com"` and assert only the first occurrence is replaced.

### Finding 2.3  -  HIGH | `src/exploits.rs:236`
**Template replacement in path templates uses URL-encoding that may double-encode or mis-encode.**

```rust
let final_path = path.replace("REDIRECT", &urlencoding::encode(&redirect));
```

`urlencoding::encode` uses `application/x-www-form-urlencoded` rules (space → `+`). When the substituted value is placed into a URL path or query string inside a path template, pre-existing percent-encoded sequences in `redirect` may be double-encoded, breaking the PoC.

**Fix:** Parse the final URL with `Url::parse` after substitution, or use `Url::query_pairs_mut()` for query parameters and `utf8_percent_encode` with an explicit ASCII set for paths. Add a round-trip parse test.

**Test hint:** Pass `redirect = "https://a.com?x=%2F"` and assert the output round-trips through `Url::parse` without double-encoding `%252F`.

### Finding 2.4  -  HIGH | `src/python.rs:56-62`
**`python_string` fails to escape `\r`, allowing Python script breakage and potential injection.**

```rust
fn python_string(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('\n', "\\n");
    format!("'{escaped}'")
}
```

A header value or URL containing `\r` (e.g., `value\r\n# inject`) is emitted as a literal carriage return inside a Python single-quoted string. This terminates the line prematurely, corrupting the script or enabling comment injection on the following line.

**Fix:** Add `.replace('\r', "\\r")`. Consider also escaping `\x00`–`\x1f` control characters comprehensively.

**Test hint:** `RequestSeed::new("https://a.com").with_header("X", "a\rb").render_python(...)` must contain `\r` escape sequence, not a literal carriage return. Compile the output with `python3 -m py_compile`.

### Finding 2.5  -  HIGH | `src/raw_http.rs:78-86`
**HTTP request splitting via unvalidated method and headers in raw HTTP renderer.**

```rust
lines.push(format!("{} {} HTTP/1.1", request.method.to_ascii_uppercase(), path));
// ...
for (key, value) in self.filtered_headers(&request.headers) {
    lines.push(format!("{key}: {value}"));
}
```

`render_raw_http_bytes` only validates URL and host. It does **not** validate `request.method` or headers for CRLF/CTL characters when the caller used the unchecked `with_method` / `with_header` constructors. A method of `GET / HTTP/1.1\r\nHost: evil.com\r\n\r\n` produces a valid-looking first line followed by a second HTTP request. Header values with `\r\n` injected via `with_header` are emitted raw.

**Fix:** Reject any method or header key/value containing `\r`, `\n`, or `\x00` inside `render_raw_http_bytes` before serialization, returning `PocError::HeaderInjection` or a new `PocError::RequestInjection`.

**Test hint:** Property test: generate random methods and header values, render raw HTTP, then assert the output contains at most one request line (zero `\r\n\r\n` sequences before the intended end).

### Finding 2.6  -  MEDIUM | `src/curl_helpers.rs:66`
**`generate_race_curl` injects unbounded `concurrency` into a shell loop with no upper limit.**

```rust
format!("# Send {concurrency} concurrent requests\nfor i in $(seq 1 {concurrency}); do {single} & done; wait")
```

A `concurrency` value of `usize::MAX` or even `1_000_000` generates a shell script that will fork-bomb the host. There is no `MAX_CONCURRENCY` constant or runtime check.

**Fix:** Enforce `concurrency <= 10_000` (or a configurable limit) and return `PocError::GenerationError` if exceeded.

**Test hint:** Assert `generate_race_curl(..., 1_000_000, ...)` returns `Err`.

---

## 3. Reproducibility (Deterministic Output?)

### Finding 3.1  -  MEDIUM | `src/generator.rs:14-18`
**No reproducibility contract, versioning, or content-addressable identity.**

`GeneratedPoc` contains only `title`, `format`, and `content`. It lacks:
- A generator version or schema version.
- A canonical timestamp or deterministic build identifier.
- A content hash (SHA-256) for integrity verification.

While output is *implicitly* deterministic today because `HeaderMap` is a `BTreeMap`, this is an implementation detail, not a contract. Future refactors (e.g., switching to `HashMap` for performance) would silently break determinism.

**Fix:** Add `version: String` and `content_hash: String` to `GeneratedPoc`. Document determinism as a public guarantee. Add snapshot tests (e.g., `insta`) that hash the full output.

**Test hint:** Generate the same `RequestSeed` 100 times and assert all fields (including `content_hash`) are identical.

### Finding 3.2  -  LOW | `src/generator.rs:101`
**Title is derived from raw URL, which may contain secrets.**

```rust
title: format!("Reproduce {}", request.url),
```

If the URL contains an API key in the query string, the title leaks it into logs, reports, and CI artifacts.

**Fix:** Strip query parameters from the title, or provide a `title_mask` option in `GeneratorOptions`.

**Test hint:** `RequestSeed::new("https://a.com?key=SECRET")` must not produce a title containing `SECRET`.

---

## 4. Auth / Session Handling in Generated PoCs

### Finding 4.1  -  CRITICAL | `src/curl_helpers.rs:108-118` / `src/generator.rs:165-181`
**Live credentials are baked into generated PoCs with no redaction, masking, or warning.**

`generate_idor_curl` accepts `cookie: Option<&str>` and emits it verbatim. The general `render_curl` / `render_python` / `render_raw_http` paths pass `Authorization`, `Cookie`, `X-Api-Key`, and other sensitive headers through unchanged. `GeneratorOptions` offers `strip_header_prefixes` but no `strip_auth_headers` or `mask_secrets` flag.

This means a CI pipeline running `pocgen` can accidentally commit live session tokens to git, or a PDF report can contain reusable bearer tokens.

**Fix:** Add a `mask_secrets: bool` field to `GeneratorOptions` (default `true`). Maintain a `SENSITIVE_HEADERS` list (`authorization`, `cookie`, `x-api-key`, `proxy-authorization`, etc.). When masking is enabled, replace header values with `[REDACTED]` and strip query parameters matching `token`, `api_key`, `session`, etc. Emit a `// WARNING: contains authentication data` comment when masking is disabled.

**Test hint:** Generate a PoC with `Authorization: Bearer sekrit` and assert `mask_secrets=true` yields `[REDACTED]`, while `mask_secrets=false` emits a warning comment.

### Finding 4.2  -  HIGH | `examples/basic.rs:23` / `tests/unit/types_tests.rs:75`
**Examples and tests model unsafe credential handling.**

The primary example embeds `Cookie: session=SESSION_TOKEN_PLACEHOLDER` directly in a `verification_request`, teaching consumers to put session material into finding structs. There is no documentation advising ephemeral tokens, short-lived JWTs, or placeholder substitution.

**Fix:** Update the example to use a placeholder token and a comment explaining that production findings should never contain live credentials. Add a doc comment on `FindingLike::verification_request` warning against real secrets.

**Test hint:** Add a doc-test that demonstrates placeholder-based cookie replacement.

---

## 5. Retry + Timeout on Network Calls

### Finding 5.1  -  HIGH | `src/curl.rs:34-68`
**Generated `curl` commands lack connection timeouts, max-time limits, and retry logic.**

The renderer produces bare `curl` invocations. For long-running or hanging endpoints, the generated PoC will block indefinitely. Race-condition scripts (`generate_race_curl`) are especially dangerous because they spawn many background processes without any timeout, creating a resource exhaustion vector on the machine executing the PoC.

**Fix:** Add `curl_timeout_secs: Option<u64>` and `curl_retry_count: u32` to `GeneratorOptions`. When set, emit `--connect-timeout {t} --max-time {t}` and, if retries > 0, `--retry {n} --retry-delay 1`. Default to sensible values (e.g., 30s timeout, 0 retries) rather than leaving them absent.

**Test hint:** Assert `render_curl` output contains `--connect-timeout 30` when `curl_timeout_secs = Some(30)`.

### Finding 5.2  -  MEDIUM | `src/python.rs:44-52`
**Generated Python has `timeout` but no retry adapter.**

```python
response = requests.request(
    ...,
    timeout={timeout},
    verify={verify},
)
```

A single transient failure (connection reset, 503) requires manual re-execution. For reproducibility in automated pipelines, the script should retry idempotent methods at least once.

**Fix:** When `python_timeout_secs` is set and the method is idempotent (`GET`, `HEAD`, `OPTIONS`, `TRACE`), prepend a `requests.adapters.HTTPAdapter` with `urllib3.util.retry.Retry(total=3, backoff_factor=0.5)` to the session.

**Test hint:** Compile generated Python for a GET request and assert it contains `Retry(` or `HTTPAdapter`.

### Finding 5.3  -  MEDIUM | `src/curl_helpers.rs:88`
**CORS verification helper pipes curl to `grep` without timeout guards.**

```rust
format!("{} -v 2>&1 | grep -i 'access-control'", generator.render_curl(&request)?)
```

If the target is unresponsive, the subprocess hangs forever. The `-v` flag adds verbosity but no time bound.

**Fix:** Append the same `--connect-timeout` / `--max-time` flags to the CORS helper output. Consider replacing the `grep` pipe with a curl `--write-out` format string to avoid spawning a second process.

**Test hint:** Parse the CORS output with `shell_words::split` and assert presence of `--max-time`.

---

## 6. CVE Database Staleness Detection

### Finding 6.1  -  CRITICAL | `src/config.rs:1-19` / `default_exploits.toml:1-26`
**Complete absence of CVE database, versioning, or staleness detection.**

The crate ships a static TOML file (`default_exploits.toml`) containing undated, unversioned SQLi payloads and OAuth paths. There is:
- No `schema_version` field in `ExploitsConfig`.
- No `last_updated` timestamp.
- No `cve_id` mapping for payloads.
- No update mechanism (HTTP fetch, git submodule, or embedded manifest).
- No signature or checksum for config integrity.
- No warning when payloads are older than N days.

At internet scale, stale payloads generate false negatives (WAFs have learned them) and false positives (outdated signatures). This is a total gap.

**Fix:** Add `schema_version: u32` and `last_updated: chrono::DateTime<Utc>` to `ExploitsConfig`. Embed a manifest checksum at build time. Provide an optional `update()` async function (behind an `online` feature) that fetches a signed manifest and validates freshness. If `last_updated` is older than 90 days, return `PocError::StaleDatabase` on `init_config`.

**Test hint:** Load a TOML with `last_updated = "2020-01-01"` and assert `init_config` returns `StaleDatabase`. Load a valid current manifest and assert `schema_version >= 1`.

---

## Cross-Cutting Findings

### Finding 7.1  -  CRITICAL | `src/exploits.rs:243-248`
**`sanitize_identifier` silently mutates input instead of failing.**

```rust
fn sanitize_identifier(ident: &str) -> String {
    ident
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}
```

`users;drop table` becomes `usersdroptable`. Rather than rejecting malicious input, the function produces a different, potentially valid identifier. This violates "fail fast" and can mislead operators into thinking the input was accepted as-is.

**Fix:** Rename to `validate_identifier` and return `Result<String, PocError>`, failing on any non-alphanumeric/non-underscore character.

**Test hint:** `SqliExploiter::row_count_payload("users;drop table")` must return `Err`, not a sanitized string.

### Finding 7.2  -  HIGH | `src/exploits.rs:6-22`
**Global `OnceLock` config is immutable after first write.**

```rust
static CONFIG: OnceLock<ExploitsConfig> = OnceLock::new();
```

`init_config` can only succeed once per process. There is no way to hot-reload rules, rotate payloads, or update CVE mappings without restarting the process. This conflicts with long-running scanners.

**Fix:** Replace `OnceLock` with `RwLock<Arc<ExploitsConfig>>` (or `arc-swap`) and provide a `reload_config` function. Validate the new config against the current `schema_version` before swapping.

**Test hint:** Call `init_config`, then `reload_config` with a new manifest, and assert `SqliExploiter::version_payloads()` reflects the updated values.

### Finding 7.3  -  MEDIUM | `src/request.rs:82-85`
**`try_with_method` rejects valid RFC 7230 token characters.**

```rust
if method.chars().any(|c| !c.is_ascii_alphanumeric() && c != '-') {
    return Err(crate::error::PocError::InvalidMethod);
}
```

RFC 7230 §3.1.1 permits `!#$%&'*+-.^_`|~` in methods. The gap test (`test_release_gap_try_with_method_must_accept_valid_rfc7230_tokens`) already documents this failure.

**Fix:** Validate against the RFC 7230 `tchar` set: `!#$%&'*+-.^_`|~` plus alphanumerics.

**Test hint:** `try_with_method("CUSTOM~!")` must return `Ok`.

### Finding 7.4  -  MEDIUM | `src/request.rs:132`
**`try_with_header` only rejects CRLF, not spaces or control chars in field names.**

```rust
if key.contains('\r') || key.contains('\n') || value.contains('\r') || value.contains('\n')
```

A header name of `Bad Header\x00` passes validation but is illegal per RFC 7230 and will break HTTP parsers.

**Fix:** Reject header names containing spaces, null bytes, or any character outside `tchar`. Reject header values containing `\r`, `\n`, or `\x00`.

**Test hint:** `try_with_header("Bad Header\x00", "v")` must return `HeaderInjection`.

### Finding 7.5  -  LOW | `src/lib.rs:15,18`
**Duplicate `#![allow(clippy::must_use_candidate)]`.**

**Fix:** Remove the redundant directive.

### Finding 7.6  -  LOW | `src/utils/url.rs:5`
**`is_http_url` allocates a new `String` for every check.**

```rust
let url = url.to_ascii_lowercase();
```

**Fix:** Use `url.get(..7).map(|s| s.eq_ignore_ascii_case("http://")).unwrap_or(false)` or `url::Url::parse` scheme check.

---

## Summary Table

| # | Severity | File:Line | Category | Description |
|---|----------|-----------|----------|-------------|
| 1.1 | CRITICAL | `Cargo.toml:1` / `lib.rs:36` | Gating | No feature flag gates exploit payloads |
| 2.1 | CRITICAL | `chain.rs:168` | Template Injection | Unescaped `{{field}}` substitution |
| 2.2 | HIGH | `exploits.rs:225` | Template Injection | Global `replace` without occurrence limit |
| 2.3 | HIGH | `exploits.rs:236` | Template Injection | Potential double-encoding in path templates |
| 2.4 | HIGH | `python.rs:56-62` | Template Injection | `\r` not escaped in `python_string` |
| 2.5 | HIGH | `raw_http.rs:78-86` | Template Injection | HTTP request splitting via unchecked method/headers |
| 2.6 | MEDIUM | `curl_helpers.rs:66` | Template Injection | Unbounded `concurrency` in race shell loop |
| 3.1 | MEDIUM | `generator.rs:14-18` | Reproducibility | No versioning or content hash in `GeneratedPoc` |
| 3.2 | LOW | `generator.rs:101` | Reproducibility | Title leaks URL secrets |
| 4.1 | CRITICAL | `curl_helpers.rs:108` / `generator.rs:165` | Auth | Credentials rendered verbatim with no masking |
| 4.2 | HIGH | `examples/basic.rs:23` | Auth | Example models unsafe credential embedding |
| 5.1 | HIGH | `curl.rs:34-68` | Timeout/Retry | Generated curl lacks timeouts and retry |
| 5.2 | MEDIUM | `python.rs:44-52` | Timeout/Retry | Python script lacks retry adapter |
| 5.3 | MEDIUM | `curl_helpers.rs:88` | Timeout/Retry | CORS helper pipes to grep without timeout |
| 6.1 | CRITICAL | `config.rs:1` / `default_exploits.toml` | CVE/Staleness | No CVE database, versioning, or staleness detection |
| 7.1 | CRITICAL | `exploits.rs:243-248` | Input Validation | `sanitize_identifier` silently mutates instead of failing |
| 7.2 | HIGH | `exploits.rs:6-22` | Config | Immutable `OnceLock` prevents hot-reload |
| 7.3 | MEDIUM | `request.rs:82-85` | Standards | Rejects valid RFC 7230 method tokens |
| 7.4 | MEDIUM | `request.rs:132` | Standards | Header validation too narrow (CRLF only) |
| 7.5 | LOW | `lib.rs:15,18` | Quality | Duplicate allow directive |
| 7.6 | LOW | `utils/url.rs:5` | Performance | Unnecessary allocation in `is_http_url` |
