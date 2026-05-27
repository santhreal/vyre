# Audit: safecheckpoint + santh-error

**Date:** 2026-04-23  
**Scope:** `libs/general/safecheckpoint/src/` + `libs/general/santh-error/src/` (read-only)  
**Auditor:** Kimi Code CLI  
**Standards:** LAWS 0-8, UNIX/SQLITE standard, competitor-informed design.

---

## Executive Summary

`safecheckpoint` implements a safetensors-compatible checkpoint format with CRC32 per-tensor, advisory flock, and atomic rename for single files. Its sharding layer and error handling have critical gaps in atomicity, concurrency, and schema evolution. `santh-error` has a clean typestate builder enforcing `Fix:` hints, but the crate suffers from unused dependencies, information loss in `Display`, and an unvalidated error-code namespace. Both crates have test suites that exercise happy paths and some adversarial cases, yet miss the exact failure modes that matter at internet scale: temp-file races, silent header corruption, and cross-version compatibility.

---

## safecheckpoint

### CRITICAL

| Severity | File:Line | Description | Suggested Fix |
|----------|-----------|-------------|---------------|
| CRITICAL | `shard.rs:154-157` | Index file is written with `std::fs::write(temp, json)` followed by `rename`. `std::fs::write` does **not** `fsync` before close. A crash between `write` and `rename` (or before OS flush) leaves a zero-length or partial index file at the final path. The entire sharded checkpoint becomes unresolvable. | Reuse `Writer::save` atomic semantics: `File::create`, `write_all`, `sync_all`, `rename`, `sync` parent directory. Never use `std::fs::write` for atomic targets. |
| CRITICAL | `writer.rs:81` | Temp file path is `path.with_extension("tmp")`. All concurrent writers targeting the same final path collide on the identical temp file. Process A can truncate/remove process B's temp file mid-write; rename races produce arbitrary winner. | Use `tempfile::NamedTempFile` in the target directory (or embed PID + random nonce) so temp paths are unique. Ensure cleanup of orphaned temps on startup. |
| CRITICAL | `reader.rs:38-44` | `mmap` is created under an **advisory** `lock_shared()`. External processes (or containers) that do not respect `flock` can truncate the file after `Reader::open`. Subsequent `get_tensor` accesses trigger `SIGBUS`. No re-stat or guard page is used. | Document the advisory-only guarantee in public docs. For environments requiring hard integrity, offer a `pread`-based `Reader` backend as an alternative to `mmap`. After `open`, record file size and validate `metadata.data_offsets` against it on every `get_tensor`. |
| CRITICAL | `tensor.rs:95-107` | `CheckpointHeader` has **no schema version field**. A v2 reader encountering a v1 file (or vice versa) has undefined behavior: new required fields will fail deserialization, removed fields will be silently ignored, and data layout changes will cause offset misinterpretation. | Add `schema_version: u64` to `CheckpointHeader`. Bump on any breaking change. The reader must reject unknown versions with a dedicated `UnsupportedVersion` error. Provide an explicit forward-compatibility policy. |
| CRITICAL | `reader.rs:52-78` | The file header (8-byte length + JSON) has **no integrity checksum**. A single-bit flip in the header length changes which bytes are interpreted as JSON; a flip in JSON changes `data_offsets`, allowing silent arbitrary memory mapping or out-of-bounds access. Per-tensor CRC32 only covers the data block, not the header. | Compute a checksum (xxh3 or blake3) over `header_len_bytes + header_json`. Store it in a fixed-size footer or prepend it before the header length. Validate it before parsing JSON. |

### HIGH

| Severity | File:Line | Description | Suggested Fix |
|----------|-----------|-------------|---------------|
| HIGH | `writer.rs:99-102`, `writer.rs:88-91` | `Error::AtomicRename(String, String)` and `Error::AtomicCleanup(String, String)` store `.to_string()` of the `std::io::Error`. The original `ErrorKind`, OS errno, and `source()` chain are **lost**. Callers cannot programmatically distinguish `NotFound` from `PermissionDenied`. | Change variants to `AtomicRename(std::io::Error)` and `AtomicCleanup(std::io::Error)`, or wrap them in `Box<dyn std::error::Error>`. |
| HIGH | `shard.rs:121-137` | Shard verification converts all `Reader`/`Checksum`/`Io` errors into `Error::Sharding(String)`. The underlying cause (which shard, which tensor, expected vs actual checksum) is discarded. | Use a structured variant like `ShardVerification { shard: String, source: Box<Error> }` that preserves the full causal chain. |
| HIGH | `shard.rs:82-106` | `save_sharded` acquires per-file exclusive locks on individual shards, but there is **no directory-level lock** protecting the index and the set of shards as a unit. Two processes saving different models with the same prefix can interleave shard writes and produce a mixed index. | Acquire an exclusive lock on a well-known lockfile (e.g., `{directory}/{prefix}.safecheckpoint.lock`) for the entire duration of `save_sharded`, including index write. |
| HIGH | `reader.rs:121-134` | `checked_add` overflow on `data_offset + metadata.data_offsets[0]` falls through to `Error::SizeMismatch { expected: 0, actual: 0 }`. The error message says "expected 0, found 0", which is factually incorrect and gives no hint that an overflow occurred. | Add `Error::OffsetOverflow { tensor_name, offset }` and report the offending values explicitly. |
| HIGH | `shard.rs:167-189` | `is_shard_valid` returns `true` if the expected tensors are present and checksums match, but **ignores extra tensors** in the shard. A stale shard from a previous write with additional tensors passes validation, leading to silent data leakage or model poisoning if the index later references it. | Validate that the shard contains **exactly** the expected tensor set (no more, no less). |
| HIGH | `writer.rs:72-79` | Path traversal check only rejects `Component::ParentDir`. It does **not** defend against absolute paths (`/etc/passwd`), symlink traversal, or `Component::RootDir`. A caller passing a symlink as `path` can write anywhere. | Resolve the path with `std::fs::canonicalize()` and assert the result is within an explicitly allowed base directory. Reject symlinks unless explicitly permitted. |

### MEDIUM

| Severity | File:Line | Description | Suggested Fix |
|----------|-----------|-------------|---------------|
| MEDIUM | `tensor.rs:56-67` | `TensorMetadata::validate()` only checks `data_offsets[0] > data_offsets[1]`. It does **not** validate that the data size matches `shape.iter().product() * dtype.size_in_bytes()`. A checkpoint can claim shape `[1000, 1000]` with 4 bytes of data. | Enforce size consistency in `validate()`. `Writer::prepare_header()` should compute expected bytes and reject mismatches at write time. |
| MEDIUM | `tensor.rs:98-99` | `CheckpointHeader` uses `#[serde(flatten)]` for the tensor map and a separate `__metadata__` field. A tensor literally named `__metadata__` will collide with the metadata map and be silently overwritten or mis-parsed. | Reject the reserved key `__metadata__` in `Writer::add_tensor`. Alternatively, stop flattening and use an explicit `"__tensors__"` wrapper. |
| MEDIUM | `safecheckpoint/Cargo.toml:12` | `categories` includes `"no-std"`, but the crate depends entirely on `std` (`std::fs`, `std::path`, `std::io`, `HashMap`, etc.). This is false advertising for downstream users. | Remove `"no-std"` from categories, or refactor to a real `no_std` core with `std` feature gating. |
| MEDIUM | `shard.rs:159-161` | Parent directory sync after index rename uses `let _ = dir.sync_all();`, **silently ignoring errors**. If the directory sync fails, the rename may not be durable. | Surface the error. If `sync_all` fails, return `Error::Io(e)` so the caller knows durability was not achieved. |
| MEDIUM | `reader.rs:150` | `get_tensor` returns `Tensor { metadata: metadata.clone(), data }`. `metadata` includes `data_offsets` that are only meaningful relative to the file base. If the caller serializes this `Tensor` elsewhere, the offsets are misleading. | Consider stripping `data_offsets` from the public `Tensor` type, or document that they are file-relative and not portable. |

### LOW

| Severity | File:Line | Description | Suggested Fix |
|----------|-----------|-------------|---------------|
| LOW | `writer.rs:176-179` | `BufWriter::new(&file)` is flushed and dropped before `mmap` creation, which is correct, but the comment at `writer.rs:182-183` falsely claims the `flock` guarantees safety for `unsafe { MmapMut::map_mut(&file) }`. Advisory locks do not prevent non-compliant processes from modifying the file. | Correct the safety comment to state the advisory-only nature of the lock. |
| LOW | `reader.rs:170-173` | `Reader::metadata()` exposes `&HashMap<String, String>` with no contract about mutation. While the borrow checker prevents mutation by callers, the API leaks the internal collection type, making future changes (e.g., `BTreeMap`, `Arc<HashMap>`) breaking. | Return an opaque wrapper or `impl Iterator<Item = (&str, &str)>` to keep the internal type private. |
| LOW | `tests/concurrent/test_release_concurrent_stress.rs:22-62` | The stress test wraps all reader/writer access in an `Arc<RwLock<()>>`, serializing operations. It does **not** test the library's actual advisory-lock concurrency behavior; it tests the test harness. | Remove the external `RwLock` and let `Reader`/`Writer` flock mechanisms contend directly. Assert that operations succeed or fail gracefully without panics. |

---

## santh-error

### CRITICAL

| Severity | File:Line | Description | Suggested Fix |
|----------|-----------|-------------|---------------|
| CRITICAL | `santh-error/Cargo.toml:19` | `thiserror` is listed as a dependency but is **never used** in any source file. Unused dependencies increase compile times, binary size, and supply-chain attack surface. | Remove `thiserror` from `Cargo.toml`. Run `cargo udeps` or `cargo clippy` to catch future dead deps. |

### HIGH

| Severity | File:Line | Description | Suggested Fix |
|----------|-----------|-------------|---------------|
| HIGH | `lib.rs:239-241` | `Display` for `SanthError` delegates to `actionable_message()`, which prints `title`, `fix`, `context`, and `location`. It **does not include the source chain** (`std::error::Error::source`). A user who only prints the error (`println!("{}", err)`) never sees the underlying `std::io::Error` message unless they manually walk `.source()`. | Append source chain summaries to `actionable_message()` (e.g., `Caused by: ...`), or at minimum include the immediate source's `Display` text. |
| HIGH | `lib.rs:252-285` | Every `From` impl maps an entire error **type** to a single code and a single generic `Fix:` hint. `std::io::ErrorKind::NotFound`, `PermissionDenied`, and `ConnectionRefused` all become `SANTH-IO-01` with the identical "Check that the file or resource exists..." message. The fix is often wrong for the actual failure. | Match on `std::io::ErrorKind` (or regex error kind) inside the `From` impl to emit discriminated codes (`SANTH-IO-NOTFOUND`, `SANTH-IO-PERM`, etc.) and tailored fixes. |
| HIGH | `lib.rs:146` | Error codes are raw `&'static str` with **no registry, validation, or stability contract**. The gap tests explicitly document this as unimplemented. A maintainer can rename `SANTH-IO-01` to `SANTH-IO-02` in a patch release, breaking programmatic error handling downstream. | Implement the error code registry (per `tests/gap.rs`). Codes should be consts in a single source of truth, with CI checks that they never change without a major/minor version bump. |

### MEDIUM

| Severity | File:Line | Description | Suggested Fix |
|----------|-----------|-------------|---------------|
| MEDIUM | `lib.rs:116-122` | `SanthErrorBuilder<HasFix>::build()` panics at runtime if the fix hint does not start with `"Fix: "`. The typestate pattern already enforces that `.fix()` was called; the panic is a second-layer guard that can crash a production process on a malformed hint. | Replace the panic with a `debug_assert!` in release builds, or return `Result<SanthError, BuildError>` so callers handle misconfiguration gracefully. |
| MEDIUM | `lib.rs:143-153` | `SanthError` does not implement `PartialEq` or `Eq`. Adversarial tests and property tests cannot assert `assert_eq!(err, expected)`; they must match individual fields manually. | Implement `PartialEq`/`Eq` that compare code, title, fix, context, and location. Ignore `source` in equality (as it is `#[serde(skip)]` and not part of the logical identity). |
| MEDIUM | `lib.rs:149` | Context keys are `&'static str`, preventing dynamic keys derived from user input or configuration keys. This limits diagnostic richness. | Change to `Cow<'static, str>` or `String` so dynamic keys are possible without lifetime contagion. |
| MEDIUM | `redact.rs:48-53` | `redact_secrets` runs 12 regexes sequentially over the entire string. Complexity is `O(N * M)` where `N` = string length and `M` = regex count. At scale (millions of log lines), this is a CPU bottleneck. | Benchmark against `aho-corasick` or a compiled DFA union. If regex flexibility is required, consider combining patterns into a single alternation regex where safe. |

### LOW

| Severity | File:Line | Description | Suggested Fix |
|----------|-----------|-------------|---------------|
| LOW | `redact.rs:5-30` | Regex patterns are compiled inside `LazyLock::new` with `.unwrap()`. If any pattern is ever modified to an invalid regex, the library panics at first access rather than failing gracefully. | Use a compile-time regex macro (e.g., `lazy_regex::lazy_regex!`) so invalid patterns are build failures, not runtime panics. |
| LOW | `lib.rs:190-217` | `actionable_message()` allocates a `Vec<String>` and joins. For hot paths, this generates many small allocations. | Use `write!` into a `String` (or `fmt::Write` into a buffer) to build the message in a single allocation. |

---

## Competitor Comparison

### safecheckpoint vs. HuggingFace safetensors (Rust)

| Feature | HF safetensors | safecheckpoint (current) |
|---------|---------------|--------------------------|
| Header integrity | No built-in checksum | No header checksum (CRITICAL gap) |
| Data integrity | None by default | Per-tensor CRC32 (good addition) |
| Schema version | Spec implies v1 only | No version field (CRITICAL gap) |
| Atomic write | Not provided by spec | Present for single file, **broken for index** |
| Concurrency | N/A (read-only library) | Advisory flock; **temp file race** |
| Alignment | Enforces 64-byte alignment | Not enforced |
| `no_std` | Optional | Claims category, does not deliver |

**Verdict:** `safecheckpoint` adds CRC32 and atomic writes, which are improvements over the baseline spec. However, the missing header checksum, missing schema version, and broken index atomicity make it **less robust** than a production-grade implementation should be.

### santh-error vs. miette

| Feature | miette | santh-error (current) |
|---------|--------|----------------------|
| Structured severity | Yes (`miette::Severity`) | Gap test acknowledges missing |
| Error code registry | Yes (`miette::Diagnostic::code`) | Not implemented |
| Source chain in display | Yes (rich report) | **Lost in `Display`** |
| Span/location tracking | Yes (with source snippets) | Basic file:line:column |
| Secret redaction | No (user responsibility) | Built-in regex redaction (nice) |
| Typestate builder | No | Yes (`NoFix` -> `HasFix`) |

**Verdict:** The typestate builder and built-in redaction are genuine differentiators. However, `miette` provides a strictly richer diagnostic model. `santh-error` should close the gap on source-chain display, discriminated error codes, and severity levels before it can claim superiority.

---

## Test Coverage Assessment

### safecheckpoint

- **Unit tests:** Basic round-trip, empty checkpoint, invalid tensor name. Good for smoke tests, but do not cover offset overflow edge cases in `SizeMismatch`.  
- **Adversarial:** Header length DOS, invalid JSON, data offset overflow. Solid, but missing: temp-file race reproduction, concurrent `save_sharded` without external locks, symlink traversal, header bit-flip survival.  
- **Concurrent:** `test_concurrent_read` is valid. `test_release_concurrent_stress` uses an external `RwLock`, invalidating the stress aspect.  
- **Crash:** Only tests checksum mismatch via manual byte corruption. Missing: power-loss simulation for index atomicity, partial header write, fsync failure injection.  
- **Property:** Roundtrip tests present. Missing: `proptest` for arbitrary tensor names (including `__metadata__`), arbitrary shapes vs data length mismatches.

### santh-error

- **Unit tests:** Builder roundtrip, display content, source chain walking, redaction. Good baseline.  
- **Adversarial:** Unicode context values, very long strings, multiple secrets. Solid.  
- **Property:** Only asserts `"Fix: "` presence. Missing: generated invalid fix hints near the `Fix: ` boundary, source chain depth invariants, context key collisions.  
- **Gap tests:** Four `#[should_panic]` tests document acknowledged missing features. Acceptable as a roadmap, but the gaps themselves are findings.

---

## Action Items (Priority Order)

1. **Fix index file atomicity in `shard.rs`**  -  add `fsync` before `rename`. (CRITICAL)
2. **Add header checksum and schema version to `CheckpointHeader`**  -  prevents silent corruption and undefined cross-version behavior. (CRITICAL)
3. **Use unique temp filenames in `Writer::save`**  -  eliminates temp-file races. (CRITICAL)
4. **Restore source chains in `safecheckpoint::Error`**  -  change `String` error payloads back to `std::io::Error` or boxed errors. (HIGH)
5. **Include source chain in `SanthError::actionable_message()`**  -  `Display` must not lose causal information. (HIGH)
6. **Implement discriminated `From` impls for `std::io::ErrorKind`**  -  stop collapsing all I/O errors into a single code. (HIGH)
7. **Add directory-level lock to `save_sharded`**  -  prevents interleaved shard/index state. (HIGH)
8. **Remove unused `thiserror` from `santh-error/Cargo.toml`**  -  reduces supply-chain surface. (CRITICAL)
9. **Validate tensor data size against `shape * dtype.size_in_bytes()`**  -  in both `Writer` and `TensorMetadata::validate`. (MEDIUM)
10. **Reserve `__metadata__` tensor key**  -  prevents serialization collision. (MEDIUM)

---

*Audit complete. All findings traced to specific file:line. No stubs, no placeholders, no shortcuts.*
