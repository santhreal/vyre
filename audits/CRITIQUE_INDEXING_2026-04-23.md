# Security & Architecture Critique  -  flashsieve + hashkit

**Date:** 2026-04-23  
**Scope:** `libs/performance/indexing/flashsieve/src/` + `libs/performance/indexing/hashkit/src/` (read-only)  
**Protocol:** LAW 0–8, STANDARDS, RESEARCH PROTOCOL  

---

## Executive Summary

- **flashsieve** is a **buffered (O(corpus))** index. A 100 GB corpus with default 256 KiB blocks produces ~400 k blocks, consuming ~3.6 GB RAM for the in-memory `BlockIndex`. There is **no streaming writer** that emits blocks directly to disk/mmap without buffering all metadata.
- The bloom filter uses a **hardcoded k = 3** hash functions. For small filters (< 4096 bits, where the exact-pair table is absent) this is far from optimal and can drive FPR toward ~85 % when m ≈ n. The exact-pair table is a clever zero-FPR optimization for 2-byte queries, but it is **lost during `NgramBloom::union_of`**, degrading `FileBloomIndex` to hash-only lookups.
- **Merge semantics are lossy for boundary state**: `BlockIndex::merge` and `remove_blocks` fail to update `last_byte`, causing **false negatives** on subsequent incremental appends.
- **hashkit** provides both cryptographic (BLAKE3, SHA-256) and fast non-cryptographic (FNV-1a, SplitMix, WyHash) hashes. The fast hashes are **not DoS-resistant** and are used without keyed seeding. The crate docs incorrectly claim FNV-1a is "flashsieve-compatible", but flashsieve uses WyHash.
- Neither crate uses `HashMap` in library code, so capacity questions are N/A for hashkit src.

---

## flashsieve Findings

### Architecture & Memory Model

| Severity | Location | Finding | Fix |
|----------|----------|---------|-----|
| **CRITICAL** | `builder.rs:124-137` | `BlockIndexBuilder::build` buffers **all** block histograms and blooms in `Vec`s (O(corpus)). Default settings yield ~3.6 GB RAM for a 100 GB corpus. No streaming writer API exists. | Add a `BlockIndexWriter<W: std::io::Write>` that serializes each block immediately, keeping only a rolling 1-byte boundary state in memory. |
| **CRITICAL** | `incremental.rs:69-116` | `IncrementalBuilder::append_blocks` deserializes the **entire** existing index into RAM, appends, and re-serializes. Incremental updates are O(index_size), not O(new_data). | Implement append-in-place on the serialized byte stream (append new blocks and rewrite only the header + CRC). |
| **HIGH** | `mmap_query.rs:139-144` | `MmapBlockIndex::candidate_blocks` allocates temporary `Vec<ByteHistogramRef>` and `Vec<NgramBloomRef>` inside the per-block hot loop for multi-block window checks. At scale (10K patterns × 400K blocks) this causes billions of small heap allocations. | Use a fixed-size stack array (e.g., `smallvec::ArrayVec` or `[_; 16]`) for the window refs, or inline the multi-check without collecting. |
| **MEDIUM** | `mmap_index.rs:44-51` | `MmapBlockIndex` stores redundant `block_offsets: Vec<usize>` when `block_metas` already holds the same offset. Wastes ~8 bytes per block (~3 MB on a 400K-block index). | Remove `block_offsets`; derive offsets from `block_metas[block_id].offset` in `try_histogram`. |

### Merge Semantics & Correctness

| Severity | Location | Finding | Fix |
|----------|----------|---------|-----|
| **CRITICAL** | `incremental.rs:250-258` | `BlockIndex::merge_with_boundary` appends blocks and updates `total_len`, but **never updates `self.last_byte`**. After merge, `append_block` uses the stale last byte, producing false negatives for boundary-spanning patterns. | After successful merge, set `self.last_byte = other.last_byte`. |
| **HIGH** | `incremental.rs:330-337` | `BlockIndex::remove_blocks` removes trailing blocks but **never updates `self.last_byte`**. The field now points to deleted data, corrupting future boundary n-gram inserts. | After removal, set `self.last_byte` to the new final block's last byte (or `None` if empty). |
| **MEDIUM** | `bloom/builder.rs:207-237` | `NgramBloom::union_of` discards `exact_pairs` tables. `FileBloomIndex::try_new` calls `union_of`, so the file-level bloom loses zero-FPR exact lookups even when every per-block bloom had them. | When all input blooms have `exact_pairs`, bitwise-OR the tables into the union result. Document the fallback when inputs are mixed. |

### Bloom Filter Tuning & FPR

| Severity | Location | Finding | Fix |
|----------|----------|---------|-----|
| **HIGH** | `bloom/filter.rs:1` | `NUM_HASHES = 3` is hardcoded globally. No dynamic `k` based on m/n. For small filters (< 4096 bits, no exact_pairs), k=3 is far above optimal. When m ≈ n, FPR ≈ (1 − e^(−3))^3 ≈ **85 %**. | Compute optimal `k = max(1, ((m as f64) / (n as f64) * std::f64::consts::LN_2).round() as u32)` at construction, unrolling probes with a `match` on `k`. |
| **MEDIUM** | `bloom/filter.rs:15-20` | `EXACT_PAIR_THRESHOLD_BITS = 4096` is a hardcoded cliff. A filter at 4095 bits is 512 B; at 4096 bits it jumps to 8.5 KB (16×). No intermediate sizes or shared arenas exist. | Make the threshold configurable via `NgramBloom::with_exact_pair_threshold`, or store exact-pair tables in a bump allocator shared across blocks. |
| **MEDIUM** | `bloom/builder.rs:91-97` | `from_block_compact` sets `compact_bits = block_size / 2`. For small block sizes this can fall below the exact-pair threshold, silently losing zero-FPR. The `.max(EXACT_PAIR_THRESHOLD_BITS)` saves it, but the interaction is undocumented. | Explicitly document that compact mode preserves exact-pair acceleration by forcing the threshold. |

### Serialization, Transport & Error Precision

| Severity | Location | Finding | Fix |
|----------|----------|---------|-----|
| **MEDIUM** | `index/codec.rs:118-230` | `from_bytes_checked` returns generic `Error::TruncatedBlock` for zero bloom bits and non-power-of-two bit counts, instead of the precise `ZeroBloomBits` / `InvalidBlockSize` variants. | Map `num_bits == 0` → `ZeroBloomBits`; map `!num_bits.is_power_of_two()` → a new `InvalidBloomBits` variant (do not reuse `InvalidBlockSize`). |
| **MEDIUM** | `bloom/serde.rs:56-58` | `from_raw_parts` returns `Error::InvalidBlockSize { size: num_bits }` when bloom bits are not a power of two. The error message says "block size must be a power of two and at least 256 bytes", which is nonsensical for a bit count. | Introduce `Error::InvalidBloomBits { bits: usize }` with a message like "bloom bits must be a power of two; got {bits}". |
| **MEDIUM** | `transport.rs:271-284` | `crc32_simple` in the transport module uses a bit-serial O(n×8) algorithm, while `index/codec.rs` uses a table-driven O(n) CRC. Transport encode/decode of large indexes is unnecessarily slow. | Extract the fast table-driven `crc32_simple` from `index/codec.rs` into a shared `crc32` module and reuse it in transport. |
| **LOW** | `transport.rs:185-222` | `rle_compress` unnecessarily escapes `0xFE` bytes (only `0xFF` is the RLE marker). The format docs also describe a `0xFE` literal-run opcode that is **never emitted or decoded**. | Remove the `0xFE` escape branch; update the doc comment to match the actual wire format. |
| **LOW** | `lib.rs:104-108` | Doc comment claims "keeps `unsafe` usage narrowly scoped", but the crate has `#![forbid(unsafe_code)]`. The statement is stale and misleading. | Update to "This crate contains no `unsafe` code." |
| **LOW** | `incremental_watch.rs:183-207` | `walk_dir_inner` silently ignores subdirectory read errors (`let _ = walk_dir_inner(...)`). The caller cannot distinguish "no files" from "permission denied". | Propagate or collect walk errors and expose them in `ChangeSet` or return `Result`. |

### Histogram & Block Size

| Severity | Location | Finding | Fix |
|----------|----------|---------|-----|
| **MEDIUM** | `histogram.rs:62-113` | `ByteHistogram` stores `u32` counts with `saturating_add`. `validate_block_size` does not cap `block_size`, so a 4 GiB block of identical bytes saturates at `u32::MAX`, returning a truncated count. | Enforce `block_size <= u32::MAX` in `validate_block_size`, or switch counts to `u64`. |

---

## hashkit Findings

### Hash Function Security & DoS Resistance

| Severity | Location | Finding | Fix |
|----------|----------|---------|-----|
| **HIGH** | `lib.rs:94-98` | `bloom_hash_pair` returns `(fnv1a_pair, splitmix::pair)`. Both are **unkeyed, deterministic, and vulnerable to collision attacks**. An adversary can craft 2-byte n-grams to maximize bloom FPR or force HashMap degradation. No SipHash, aHash, or keyed variant is offered. | Add a `bloom_hash_pair_seeded(seed: u64)` variant using SipHash-1-3 or aHash for security-sensitive bloom filters. Document the attack vector for the unseeded path. |
| **HIGH** | `lib.rs:14-15` | Crate docs claim FNV-1a is "flashsieve-compatible" and `bloom_hash_pair` is "the flashsieve-compatible FNV-1a hash". **flashsieve uses wyhash**, not FNV-1a. The hashes are incompatible. | Update docs to remove stale "flashsieve-compatible" claims. If cross-crate compatibility is desired, add a wyhash-based pair hash that matches `flashsieve::bloom::hash::hash_pair`. |
| **MEDIUM** | `lib.rs:20-23` | Security note warns against content-addressed use, but does **not** warn that FNV/SplitMix/WyHash must not be used as `BuildHasher` for `HashMap`/`HashSet` with untrusted keys** (HashDoS). | Add explicit warning: "Do not use these hashes as `BuildHasher` for hash tables with adversarial input. Use `std::collections::HashMap` (SipHash) or `ahash` instead." |

### `no_std` Metadata Mismatch

| Severity | Location | Finding | Fix |
|----------|----------|---------|-----|
| **MEDIUM** | `Cargo.toml` categories | `hashkit` lists `categories = ["algorithms", "no-std"]`, but the crate uses `std::error::Error`, `String`, `Vec`, and does **not** declare `#![no_std]`. It will not compile in a `no_std` environment. | Either add `#![no_std]` + `extern crate alloc` and polyfill error handling, or remove `"no-std"` from categories. |

### Error Messages

| Severity | Location | Finding | Fix |
|----------|----------|---------|-----|
| **MEDIUM** | `hex.rs:22-31` | `DecodeError` display messages are plain descriptions ("odd number of digits in hex string"). They lack the required "Fix: ..." actionable suffix. | Append fixes, e.g. "Fix: ensure the hex string has an even number of characters." |

### Code Quality

| Severity | Location | Finding | Fix |
|----------|----------|---------|-----|
| **LOW** | `wyhash.rs:27-34` | `wymum` uses `unreachable!()` (a panic macro) for `u64::try_from` on bitmasks that are provably in-bounds. While unreachable in practice, it is a panic path in production code. | Replace with `as u64` (the cast is safe by construction) or add a `// SAFETY:` comment. |
| **INFO** | `hashkit/src/*.rs` | **No `HashMap` or `HashSet` usage exists in library source code.** Questions about `HashMap::with_capacity` estimates and `HashMap::new()` in hot loops are not applicable to the src tree. | Note for audit trail. |

---

## Competitor Comparison Notes

| Competitor | What they do better | flashsieve/hashkit gap |
|------------|--------------------|------------------------|
| **fastbloom** / **sbbf-rs** | Blocked SIMD bloom filters, streaming construction, memory-mapped query without heap deserialization. | flashsieve lacks SIMD, true streaming construction, and mmap-native query (mmap mode still allocates per-query Vecs). |
| **bloomfilter** crate | Counting & scalable blooms, configurable `k`, and `BuildHasher` generics. | flashsieve hardcodes k=3 and offers no counting bloom or configurable hasher. |
| **ahash** / **seahash** / **highwayhash** | Provide DoS-resistant or cryptographically strong non-cryptographic hashing. | hashkit only provides attackable FNV/SplitMix/WyHash; no SipHash, aHash, or HighwayHash equivalent. |
| **zstd** / **lz4** transport | Standard, high-speed compression with bounded memory. | flashsieve transport uses a custom RLE with ad-hoc escaping and a slow bit-serial CRC. |

---

## Audit Trail

- All source files in `flashsieve/src/` and `hashkit/src/` were read in full.
- `cargo test` and `cargo clippy -- -D warnings` were **not** run because the task is read-only.
- No `HashMap`/`HashSet` usage was found in `hashkit/src/` (verified with ripgrep).
- flashsieve does **not** depend on hashkit; it re-implements its own wyhash-based bloom hashing.
