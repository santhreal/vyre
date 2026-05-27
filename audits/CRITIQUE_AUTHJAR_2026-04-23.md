# CRITIQUE: authjar  -  Security & Compliance Audit

**Date:** 2026-04-23  
**Scope:** `libs/runtime/authjar/src/` (read-only)  
**Auditor:** Kimi Code CLI (security researcher)  
**Ruleset:** RFC 6265 / 6265bis, modern browser cookie policies, CHIPS, crate LAWS  

---

## Executive Summary

`authjar` implements a functional cookie-jar primitive with domain/path scoping, serialization, and browser extraction. However, it **fails to implement multiple security-critical requirements** from RFC 6265bis and modern browser standards. The most severe gaps are: **no SameSite default (violates RFC 6265bis)**, **no Public Suffix List enforcement (super-cookie vulnerability)**, **no `__Host-` / `__Secure-` prefix enforcement**, and a **per-session cookie limit that is trivially bypassed** when ingesting `Set-Cookie` headers. Several architectural choices (silent rejection, shelling out to `openssl`, predictable temp-file names) create additional attack surface.

---

## Findings

### 1. CRITICAL | session/cookie.rs:72 | SameSite defaults to `None`  -  violates RFC 6265bis

**Description:**  
`Cookie::new` sets `same_site: None`. The parser (`from_header_line_with_domain`) also leaves `same_site` as `None` when the attribute is absent. RFC 6265bis **mandates** that cookies without an explicit `SameSite` attribute be treated as `Lax`. The `matches()` function does not filter cookies by `SameSite` at all, meaning a cookie parsed as `None` will be sent in cross-site contexts exactly like a pre-2016 cookie.

**Suggested Fix:**  
Change the default in `Cookie::new` and in the parser fallback to `Some(SameSite::Lax)`. Implement `SameSite` enforcement in `Cookie::matches`: reject `Strict` cookies for cross-site requests, reject `Lax` cookies for unsafe cross-site methods (POST/PUT/etc.), and require `Secure` for `SameSite=None`.

**Test Hint:**  
```rust
let cookie = Cookie::new("sid", "1", "example.com");
assert_eq!(cookie.same_site, Some(SameSite::Lax));
```

---

### 2. CRITICAL | session/cookie/scope.rs:19 | No Public Suffix List check  -  super-cookie vulnerability

**Description:**  
`validate_domain` only checks character set, length ≤253, and leading/trailing dot/hyphen rules. It does **not** reject public suffixes such as `.co.uk`, `.com`, `.github.io`, or `.cloudfront.net`. An attacker controlling `evil.github.io` can set a cookie with `Domain=github.io` and have it sent to **every** `*.github.io` origin.

**Suggested Fix:**  
Integrate a Public Suffix List (PSL) lookup (e.g., `publicsuffix` crate or an embedded list). Reject any `Domain=` attribute that matches a public suffix or is a bare TLD. Also reject `Domain` values that are exact IP addresses unless compared exactly.

**Test Hint:**  
```rust
assert!(Cookie::from_set_cookie("sid=1; Domain=.co.uk", "x.co.uk").is_none());
assert!(Cookie::from_set_cookie("sid=1; Domain=.github.io", "evil.github.io").is_none());
```

---

### 3. CRITICAL | session/middleware.rs:132 | `add_set_cookie` bypasses `MAX_COOKIES_PER_SESSION`

**Description:**  
`add_cookie_with_path` (line 87–116) enforces `MAX_COOKIES_PER_SESSION` (4096) before insertion. However, `add_set_cookie`, `add_cookie_header_line`, and `add_cookie_header_line_for_host` all call `Cookie::from_*` and then do an **unconditional** `self.cookies.insert(key, cookie)`. A malicious server can flood a session with unlimited cookies via `Set-Cookie` responses.

**Suggested Fix:**  
Centralize all insertions through a single `insert_cookie(&mut self, cookie: Cookie) -> Result<(), AuthJarError>` method that enforces the limit and returns an error instead of silently swallowing. Call this from every public `add_*` method.

**Test Hint:**  
```rust
let mut session = AuthSession::new("test");
for i in 0..5000 {
    session.add_set_cookie(&format!("c{i}=v; Path=/; Domain=example.com"), "example.com");
}
assert_eq!(session.cookie_count(), 4096);
```

---

### 4. CRITICAL | session/cookie.rs:22 | `__Host-` / `__Secure-` prefix enforcement is absent

**Description:**  
The codebase contains **zero** references to `__Host-` or `__Secure-`. Per RFC 6265bis Section 4.1.3:
- Names prefixed with `__Secure-` MUST be accompanied by the `Secure` attribute.
- Names prefixed with `__Host-` MUST be accompanied by `Secure`, `Path=/`, and MUST NOT have a `Domain` attribute.
Without enforcement, a server can set `__Host-session=xyz` with `Domain=evil.com; Path=/api`, violating the prefix contract and breaking security assumptions of downstream code.

**Suggested Fix:**  
In `apply_cookie_attribute` and `add_cookie_with_path`, add prefix checks:
- If name starts with `__Secure-` and `!secure`, reject.
- If name starts with `__Host-` and `(!secure || path != "/" || !host_only)`, reject.

**Test Hint:**  
```rust
assert!(Cookie::from_set_cookie("__Secure-session=1; Path=/", "example.com").is_none());
assert!(Cookie::from_set_cookie("__Host-session=1; Path=/; Secure; Domain=example.com", "example.com").is_none());
```

---

### 5. HIGH | session/cookie.rs:247 | Secure context inference missing  -  insecure cookies accepted over HTTPS

**Description:**  
There is no mechanism to tell the jar “this cookie was received over HTTPS.” `add_cookie_header_line_for_host` and `add_set_cookie` accept cookies with `secure: false` even when the transport was TLS. Modern browsers and RFC 6265bis expectations are that a jar bound to an HTTPS context should reject non-Secure cookies (or at least provide an opt-in policy). The `matches()` function only filters on read (`Secure` cookie → no HTTP), not on write.

**Suggested Fix:**  
Add `require_secure: bool` to `SessionSettings` (default `false` for back-compat, but document the risk). In `from_header_line_with_domain`, if `request_is_secure` is true and the cookie lacks `Secure`, reject it or flag it. Alternatively, add an `HttpsJar` wrapper that enforces this at ingestion time.

**Test Hint:**  
```rust
let settings = SessionSettings { require_secure: true, ..Default::default() };
let mut session = AuthSession::new("test");
session.add_set_cookie("sid=1; Path=/; Domain=example.com", "example.com"); // no Secure
assert!(session.is_empty());
```

---

### 6. HIGH | session/mod.rs:24 | Total serialized cookie size limit not enforced

**Description:**  
`MAX_COOKIE_NAME_LEN = 256` and `MAX_COOKIE_VALUE_LEN = 4096` are checked independently. A cookie with a 256-byte name and 4096-byte value serializes to >4350 bytes, exceeding the de-facto 4096-byte total limit that browsers enforce. The parser rejects oversized values, but there is **no check on the total `name=value` pair size**, nor on the total `Set-Cookie` header length.

**Suggested Fix:**  
Add `MAX_COOKIE_TOTAL_LEN = 4096` and validate `name.len() + value.len() + 1 <= MAX_COOKIE_TOTAL_LEN` in `is_valid_cookie_name`/`is_valid_cookie_value` or a new `validate_cookie_size` function.

**Test Hint:**  
```rust
let name = "A".repeat(256);
let val = "B".repeat(4096);
assert!(Cookie::from_set_cookie(&format!("{name}={val}; Path=/"), "example.com").is_none());
```

---

### 7. HIGH | browser/chrome.rs:8 | Browser temp file leaks on error paths

**Description:**  
`read_chrome_cookies` copies the browser DB to a temp file, then performs SQL operations that may return `Err` via `?`. The cleanup `let _ = std::fs::remove_file(&temp);` is at the very end of the function. If `conn.prepare()` or `stmt.query_map()` fails, the function returns early and the temp SQLite file (containing all cookies) is left in `/tmp/`.

**Suggested Fix:**  
Use `tempfile::NamedTempFile` or wrap the `PathBuf` in a RAII guard that deletes on `Drop`. Do not rely on manual cleanup at the end of a fallible function.

**Test Hint:**  
Simulate a read-only temp dir or corrupt DB copy and verify no `karyx-chrome-cookies-*.sqlite` remains in `std::env::temp_dir()` after the function returns `Err`.

---

### 8. HIGH | browser/chrome.rs:145 | Chrome cookie decryption shells out to `openssl`

**Description:**  
`aes_128_cbc_decrypt` spawns an `openssl` child process for every encrypted cookie. This is:
- **Brittle**: fails silently if `openssl` is not installed.
- **Slow**: process-per-cookie is unacceptable at scale.
- **Security risk**: hardcoded IV (`[b' '; 16]`), hardcoded password `"peanuts"`, and reliance on an external binary.
- **Outdated**: modern Chrome/Chromium use v11 encryption or OS keyrings (DPAPI/Keychain); the `"peanuts"` scheme is legacy Linux-only.

**Suggested Fix:**  
Replace the `Command::new("openssl")` call with the `aes` crate (e.g., `aes::Aes128` + `cbc` crate) for pure-Rust decryption. Document that only legacy v10 Linux cookies are supported, and return a clear error for unsupported encryption schemes.

**Test Hint:**  
Run on a system without `openssl` in `$PATH`; assert that extraction returns a clear error rather than silently returning empty values.

---

### 9. HIGH | session/cookie.rs:182 | `SameSite=None` accepted without `Secure` attribute

**Description:**  
The parser accepts `SameSite=None` even when the `Secure` flag is absent. RFC 6265bis requires that `SameSite=None` MUST be accompanied by `Secure`. Browsers reject such cookies. `authjar` silently stores and will later serialize them, creating cookies that would be dropped by any compliant user-agent.

**Suggested Fix:**  
In `apply_cookie_attribute`, when `SameSite=None` is set, immediately check `cookie.secure`. If false, return `None` (reject the cookie). Likewise during `to_set_cookie_string`, assert the invariant or emit `Secure` automatically.

**Test Hint:**  
```rust
assert!(Cookie::from_set_cookie("sid=1; SameSite=None; Path=/", "example.com").is_none());
```

---

### 10. HIGH | browser/matching.rs:17 | No third-party / first-party context policy

**Description:**  
There is no concept of "site for cookies" or first-party vs. third-party context. The jar will send any matching cookie regardless of the initiator of the request. Modern browsers block third-party cookies by default (Chrome, Firefox, Safari). A security-focused cookie jar should at minimum provide an opt-in `block_third_party` setting that rejects cookies without `SameSite=None` (or rejects all) when the request origin differs from the cookie's registrable domain.

**Suggested Fix:**  
Add `block_third_party: bool` to `SessionSettings`. When enabled, `matches()` should reject cookies for cross-site requests unless `SameSite=None` and `Secure` are present (and even then, warn that this is deprecated behavior). Add `Partitioned` attribute support for CHIPS.

**Test Hint:**  
```rust
let settings = SessionSettings { block_third_party: true, ..Default::default() };
let header = session.cookie_header_for("third-party.com", "/", true, &settings);
assert!(header.is_empty());
```

---

### 11. MEDIUM | session/middleware.rs:57 | Silent rejection instead of loud errors

**Description:**  
`add_cookie`, `add_cookie_with_path`, and store `add*` methods reject invalid input by logging via `tracing::warn!` and returning `()`. At internet scale, silent rejection means a monitoring system will never see the failure unless someone is watching logs. The task requirements state: "Beyond-limit cookies need a loud reject, not silent truncation."

**Suggested Fix:**  
Change all `add_*` methods to return `Result<(), AuthJarError>`. Keep convenience wrappers if necessary, but the primary API must surface errors. Never swallow validation failures.

**Test Hint:**  
```rust
let err = session.add_cookie("", "", "").unwrap_err();
assert!(err.to_string().contains("invalid cookie"));
```

---

### 12. MEDIUM | session/cookie/scope.rs:70 | `domain_matches` allows IP-address suffix matching

**Description:**  
`validate_domain` permits IP addresses (`192.168.1.1` passes because it contains only digits and dots). `domain_matches` then uses `ends_with(".{cookie_domain}")`. A cookie with `Domain=168.1.1` would match request `192.168.1.1`, which is nonsensical and dangerous for internal networks.

**Suggested Fix:**  
Detect IP addresses (IPv4 and IPv6) in `validate_domain` and in `domain_matches`. For IPs, require exact equality; never apply suffix matching.

**Test Hint:**  
```rust
assert!(!domain_matches("192.168.1.1", "168.1.1"));
assert!(domain_matches("192.168.1.1", "192.168.1.1"));
```

---

### 13. MEDIUM | session/token.rs:147 | HTTP-date parser lacks strict bounds checking

**Description:**  
`to_timestamp` does not validate that `day` is within the month, that `hour < 24`, `minute < 60`, or `second < 60`. The adversarial tests confirm that day `99` and hour `25` parse successfully and produce a timestamp. This can cause cookies to have incorrect lifetimes (too long or miscalculated).

**Suggested Fix:**  
Add bounds checks in `to_timestamp` before computing the result: `day <= 31`, `hour < 24`, `minute < 60`, `second < 60`. Also validate month/day combinations (e.g., no 30th of February). Return `None` on out-of-range values.

**Test Hint:**  
```rust
assert!(Cookie::from_set_cookie("sid=1; Expires=Sun, 99 Nov 1994 08:49:37 GMT", "ex.com").is_none());
```

---

### 14. MEDIUM | safety.rs:28 | `use rand::RngExt;`  -  compilation error in rand 0.10.1

**Description:**  
`Cargo.toml` pins `rand = "=0.10.1"`. The trait `RngExt` does not exist in rand 0.10.x (it was removed/reorganized after earlier versions). While `gaussian_delay` only uses `rand::Rng` methods (`random_range`), the unused import will cause a **compile-time error**, breaking the build.

**Suggested Fix:**  
Remove `use rand::RngExt;`. Verify with `cargo check`.

**Test Hint:**  
Run `cargo check --all-features`; assert zero errors.

---

### 15. MEDIUM | safety.rs:122 | `is_safe_endpoint` uses substring matching, causing false positives

**Description:**  
`is_safe_endpoint` checks `path_only.contains(&blocked_lower)`. A blocked endpoint `/logout` will therefore block `/api/logout/user`, `/blog/logout-guide`, etc. The intent is to block exact or glob-matched endpoints, not arbitrary substrings.

**Suggested Fix:**  
Change substring matching to exact path matching (or anchored prefix matching). Keep glob support for explicit patterns, but `/logout` should match `/logout` and `/logout/` exactly, not `/api/logout`.

**Test Hint:**  
```rust
let config = SafetyConfig::default();
assert!(is_safe_endpoint("/api/logout/user", &config)); // currently false, should be true
```

---

### 16. MEDIUM | browser/matching.rs:64 | `sanitize_cookie_value` strips non-graphic ASCII, but caller trusts original value

**Description:**  
`cookies_header_for_domain` calls `sanitize_cookie_value` when formatting the header, but `BrowserCookie` values are stored unsanitized. If a later code path reads `cookie.value` directly (not through the sanitizer), it may encounter control characters. Defense in depth requires sanitizing at ingestion or using a validated newtype.

**Suggested Fix:**  
Sanitize browser cookie values at extraction time (in `read_firefox_cookies` / `read_chrome_cookies`), or store them in a validated `CookieValue` newtype that guarantees printable ASCII on construction.

**Test Hint:**  
Directly construct a `BrowserCookie` with `value: "a\x01b"`, pass it through `cookies_header_for_domain`, and assert the control byte is stripped.

---

### 17. MEDIUM | session/cookie.rs:310 | `to_set_cookie_string` does not re-validate fields at serialization

**Description:**  
`to_set_cookie_string` trusts `self.name` and `self.value`. If a `Cookie` struct is mutated after creation (e.g., `cookie.value.push(';')`), the serialized output will contain an invalid `Set-Cookie` header (`name=val;ue; Path=/`). This is a header-injection risk for any code that mutates cookies programmatically.

**Suggested Fix:**  
In `to_set_cookie_string`, assert `is_valid_cookie_name(&self.name)` and `is_valid_cookie_value(&self.value)` before formatting, and panic (or return `Result`) if the invariant is violated. In non-test builds, the crate already `#![deny(clippy::unwrap_used)]`, so return a `Result<String, AuthJarError>`.

**Test Hint:**  
```rust
let mut c = Cookie::new("sid", "abc", "example.com");
c.value.push(';');
assert!(c.to_set_cookie_string().is_err());
```

---

### 18. LOW | browser/chrome.rs:8 | Predictable temp-file name with project name leak

**Description:**  
The temp file is named `karyx-chrome-cookies-{pid}.sqlite`. The string `karyx` appears to be a leftover from a different project. Predictable names in `/tmp` allow a local attacker to know the filename and potentially race or read the copied database.

**Suggested Fix:**  
Use `tempfile::NamedTempFile` with a random suffix, or at minimum rename the pattern to `authjar-chrome-cookies-{random}.sqlite`.

**Test Hint:**  
Verify that two successive calls produce different temp file names.

---

### 19. LOW | session/cookie.rs:260 | Empty `Domain=` attribute falls back to request host instead of rejecting

**Description:**  
When parsing `Set-Cookie: sid=1; Domain=`, the code normalizes the empty string to `""`, fails `validate_domain`, and returns `None`. This is correct. However, when `Domain` is **absent**, it falls back to `default_domain`. The RFC says a missing `Domain` makes the cookie host-only. The implementation does this correctly (`host_only = true`), but the distinction between "empty Domain attribute" and "missing Domain attribute" could be clearer in docs.

**Suggested Fix:**  
Document the behavior explicitly: empty `Domain=` is rejected; omitted `Domain` creates a host-only cookie. Add a dedicated test for this distinction.

**Test Hint:**  *(already covered in existing tests; ensure doc comment is updated)*

---

### 20. LOW | csrf/extract.rs:6 | CSRF HTML scan truncates at 1 MB without warning

**Description:**  
`extract_from_html` silently truncates HTML to `MAX_HTML_SCAN_BYTES` (1 MB). A token placed after 1 MB will be missed without any indication to the caller. This is a silent truncation, which LAW 0 and the standards explicitly forbid.

**Suggested Fix:**  
Return a `Result` or a scan report struct that includes `truncated: bool`. Alternatively, stream-parse the HTML instead of loading it all.

**Test Hint:**  
```rust
let html = format!("{}<meta name=\"csrf-token\" content=\"found\">", "A".repeat(2_000_000));
let tokens = extract_csrf_tokens(&html, &[], &[]);
assert!(tokens.is_empty() /* or check truncation flag */);
```

---

## Positive Observations

1. **No panics on malformed input.** The parser consistently uses `Option` returns and avoids unwraps in production code.
2. **CRLF injection defense.** Both name and value validators reject `\r`, `\n`, and control characters.
3. **Session/cookie limits exist** (4096 cookies/session, 1024 sessions/store, 8 MB store file)  -  but enforcement is incomplete (see Finding 3).
4. **Host-only vs domain-scoped cookies are correctly distinguished** in storage keys and matching logic.
5. **Plaintext persistence is explicitly warned** in `save_to_file`.

---

## Competitor Comparison

| Feature | `authjar` | `cookie_store` (rust) | `reqwest` cookie jar |
|---|---|---|---|
| SameSite default `Lax` | ❌ `None` | ✅ | ✅ |
| Public Suffix List | ❌ | ✅ (via `publicsuffix`) | ✅ |
| `__Host-` / `__Secure-` | ❌ | ❌ | ❌ |
| Third-party blocking | ❌ | ❌ | ❌ |
| Secure-context enforcement | ❌ (read-only) | ✅ | ✅ |
| `Partitioned` / CHIPS | ❌ | ❌ | ❌ |
| Total cookie size limit | ❌ (name+value only) | ✅ | ✅ |
| Error-return API | ❌ (silent) | ✅ | ✅ |

---

## Remediation Priority

1. **Fix `MAX_COOKIES_PER_SESSION` bypass** (Finding 3)  -  single insertion path.
2. **Implement SameSite=Lax default + enforcement** (Finding 1).
3. **Add Public Suffix List validation** (Finding 2).
4. **Add `__Host-` / `__Secure-` prefix rules** (Finding 4).
5. **Replace `openssl` subprocess with Rust crypto** (Finding 8).
6. **Fix temp-file leak with RAII** (Finding 7).
7. **Enforce `Secure` required for `SameSite=None`** (Finding 9).
8. **Return `Result` from all mutation APIs** (Finding 11).
9. **Fix `rand::RngExt` compile error** (Finding 14).
10. **Add total cookie size check** (Finding 6).
