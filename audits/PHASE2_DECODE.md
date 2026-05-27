# PHASE2_DECODE  -  Decode-stage hot-path audit

**Scope:** `vyre-libs/src/decode`, `encodex`, `ziftsieve`, `surgec scan::decode`, `keyhog scanner decode`
**Auditor:** Kimi Code CLI  
**Date:** 2026-04-24  
**Historical audit scope:** decode hot-path source findings.

---

## Specific-ask summary

| # | Question | Answer |
|---|----------|--------|
| 1 | Does every decoder emit a vyre Program? | **No.** `vyre-libs` base64/hex/inflate emit Programs, but `surgec scan::decode`, `encodex`, `ziftsieve`, and `keyhog` are pure CPU Rust loops. |
| 2 | Is decoded-bytes buffer always Storage (DRAM)? | **No.** The *standalone* decode paths (`base64_decode`, `hex_decode`, `inflate`) use `BufferDecl::output` / `BufferDecl::read_write` → **DRAM**. Only the fused `*_then_aho_corasick` paths optionally promote to workgroup via `streaming.rs` / `decode_scan_fuse.rs`. |
| 3 | Is inflate/lz4 a cooperative-thread-block kernel? | **Scoped fixed for vyre-libs.** The shipped op is now named `vyre-libs::decode::inflate_stored_block`, so the BTYPE=0-only contract is explicit. Cross-project `ziftsieve::lz4` rows remain outside this vyre decode/parser pass. |
| 4 | Do decoders pre-filter with magic-byte check? | **Partially.** `surgec` gzip/zip/tar and `ziftsieve` LZ4 frame do magic checks. `vyre-libs::inflate` does **not** pre-filter; it allocates an output buffer before reading the 3-bit BTYPE header. |
| 5 | Any O(n²) scans over decoded bytes? | **Scoped fixed for fused decoders.** `base64_decode_then_aho_corasick`, `hex_decode_then_aho_corasick`, and `inflate_stored_block_then_aho_corasick` use the shared `decode::scan::linear_aho_scan_body`, which walks the decoded stream once and preserves Aho-Corasick output parity. |
| 6 | Are decode/scan Programs cached (G8 content-hash)? | **Partially.** `vyre-runtime::pipeline_cache` and `vyre-driver-wgpu::pipeline_disk_cache` cache compiled pipelines by content hash. However, `surgec scan::decode` has **zero** Program caching  -  it rebuilds CPU-side decode state per file. The `vyre-libs` benchmarks also reconstruct Programs every benchmark iteration. |

---

## Findings

### CRITICAL

| SEVERITY | file:line | description | suggested fix |
|----------|-----------|-------------|---------------|
| FIXED | `vyre-libs/src/decode/inflate.rs` | The generic `inflate` registry op no longer ships. The registered op is `inflate_stored_block`, and compatibility builders route to that explicit BTYPE=0 contract. | Verified by `cargo test -p vyre-libs --features decode,matching-dfa decode:: --lib -- --nocapture`. |
| CRITICAL | `libs/tools/surgec/src/scan/decode.rs:162-178` | `decode_layers` is **100% CPU Rust**  -  no vyre Program, no GPU dispatch, no pipeline cache. It rebuilds regex-like extraction loops for every encoding on every file. | Replace the hot-path encodings (base64, hex, gzip, zlib) with vyre Program dispatch. Retain CPU fallback only for archive formats (zip, tar) that need host-side directory traversal. |
| FIXED | `vyre-libs/src/decode/base64.rs`, `vyre-libs/src/decode/hex.rs`, `vyre-libs/src/decode/inflate.rs`, `vyre-libs/src/decode/scan.rs` | The duplicated per-output prefix replay was replaced by one shared linear scan body. | Verified by decode unit tests and `hex_decode_scan_fused` parity. |
| CRITICAL | `vyre-libs/src/decode/inflate.rs:164-179` | Standalone `inflate` declares output as `BufferDecl::output(&output, 1, DataType::U32).with_count(input_len)`  -  worst-case allocation even for tiny stored blocks. No workgroup promotion. | Add `inflate_then_scan` variant and promote the decoded buffer to workgroup memory via `streaming::fuse_decode_scan`. For standalone use, size the output buffer from the `LEN` field in the stored block header, not `input_len`. |
| CRITICAL | `encodex/src/base64.rs:8-44` | `encodex::base64::decode` is a pure CPU helper allocating `Vec<u8>` on every call. It is used by `keyhog` and `surgec` but has **no GPU path**. | Deprecate or rewrite as a vyre Program builder. If encodex must support both CPU and GPU, split into `encodex::base64::decode_cpu` and `encodex::base64::decode_gpu` with the latter returning a `vyre::Program`. |
| CRITICAL | `ziftsieve/src/lz4.rs:69-166` | `extract_literals` is sequential CPU Rust. The module comment claims it skips decompression for speed, but it still runs on a **single thread** with no SIMD, no GPU kernel, and no vyre IR emission. | Emit a vyre Program for LZ4 literal extraction. Literal-length parsing is a finite-state walk perfect for GPU threads (one thread per LZ4 sequence). |
| CRITICAL | `software/keyhog/crates/scanner/src/decode/base64.rs:12-23` | `Base64Decoder::decode_chunk` runs `find_base64_strings` (regex-like scan) followed by `base64_decode` (CPU allocation) **per chunk**. No vyre, no cache, no streaming. | Replace with vyre pipeline: one dispatch for base64 candidate detection + decode, second dispatch for pattern matching. Reuse `vyre-libs::decode::base64_decode_then_aho_corasick`. |

### HIGH

| SEVERITY | file:line | description | suggested fix |
|----------|-----------|-------------|---------------|
| HIGH | `vyre-libs/src/decode/inflate.rs:40-44` | `inflate_body` reads the 3-bit BTYPE from the first word **after** the output buffer has already been declared and allocated. No magic-byte / BTYPE pre-filter means DRAM is reserved for payloads that will trap. | Reorder: read header first, branch on BTYPE, and only declare the output buffer size for BTYPE=0. For BTYPE=1/2/3 either dispatch to a separate Program or return an error before buffer allocation. |
| HIGH | `libs/tools/surgec/src/scan/decode.rs:196-250` | `decode_recursive` iterates **every** encoding in `config.encodings` for **every** layer. A file with 10 layers and 15 encodings = 150 extraction attempts, most of which are guaranteed to fail (e.g. tar on a JSON file). | Add a fast magic-byte classifier at the layer level. Only attempt encodings whose magic signature matches the leading bytes of the current layer. |
| HIGH | `vyre-libs/benches/decode_scan.rs:99-107` | Benchmark constructs `base64_decode_then_aho_corasick` **inside** the benchmark loop (line 99 is outside, but the program is passed by reference; however the CPU baseline rebuilds `aho_corasick` inside the loop on line 120). | Hoist Program construction out of the benchmark loop. Add a `criterion` comparison that measures cached vs uncached dispatch to quantify G8 savings. |
| HIGH | `vyre-foundation/src/optimizer/passes/decode_scan_fuse.rs:44-77` | `decode_scan_fuse::run` promotes **every** `ReadWrite` buffer with `count > 0` and `!pipeline_live_out`. This is over-aggressive: a decoder might have a legitimate `ReadWrite` sidecar (e.g. histogram) that must stay in DRAM. | Require an explicit marker (e.g. `BufferDecl::with_decode_handoff(true)`) before promoting to workgroup. Default should be conservative  -  only promote buffers the caller explicitly tags as handoff. |
| FIXED | `vyre-libs/src/decode/streaming.rs` | `fuse_decode_scan` returns `DecodeScanFuseError::ZeroHandoff` instead of asserting on zero capacity. | Covered by `decode::streaming::tests::zero_handoff_byte_count_returns_structured_error`. |
| HIGH | `encodex/src/detect.rs:72-100` | `detect_regions` tokenises the entire input and then runs `detect_chain` on every candidate. The tokeniser has **no** SIMD or GPU acceleration, and the decode attempts are performed sequentially. | Compose a vyre Program that does byte-classification (candidate vs non-candidate) in parallel, then stream-compact the candidates to a second kernel that attempts decode. |
| HIGH | `ziftsieve/src/detect.rs:38-66` | `CompressionFormat::detect` is a pure CPU `starts_with` chain. It is never called from any GPU path, so ziftsieve formats cannot be detected on-device. | Expose a vyre Program that reads the first 16 bytes and returns a format tag. Wire it into the decode pipeline so the GPU can skip entire decompression kernels when the magic doesn't match. |
| HIGH | `vyre-driver-wgpu/src/pipeline_disk_cache.rs:346-372` | `normalized_compile_wire` sets `buffer.count = 1` for all non-Shared buffers before hashing. This means a `base64_decode` for 8 bytes and a `base64_decode` for 8 MiB share the **same** pipeline cache key. While correct for shader compilation, it is **incorrect** for workgroup-size tuning and shared-memory sizing. | Include the original `count` in the cache key for workgroup buffers, or add a separate tuning cache keyed by `(program_hash, max_buffer_counts)`. |

### MEDIUM

| SEVERITY | file:line | description | suggested fix |
|----------|-----------|-------------|---------------|
| FIXED | `vyre-libs/src/decode/base64.rs` | `base64_decode` rejects non-multiple-of-four lengths with an actionable `Fix:` message. | Covered by decode unit tests. |
| MEDIUM | `libs/tools/surgec/src/scan/decode.rs:277-364` | `extract_base64_regions` uses a `BTreeSet<(usize, usize, Vec<u8>)>` to dedupe candidates. For large files with many candidate spans this is O(k log k) in CPU with repeated `Vec<u8>` clones as the set key. | Replace with a `FxHashSet` of `(start, end)` only, then compare bytes after decode. Or, better: move candidate extraction to a GPU prefix scan. |
| FIXED | `vyre-libs/src/decode/hex.rs` | Hex decode now uses `HEX_DECODE_TABLE_BUFFER` / `hex_decode_table()` and a single table load per nibble. | Covered by decode unit tests and fused hex parity. |
| MEDIUM | `vyre-libs/src/decode/mod.rs:10-13` | Module doc claims streaming fusion keeps bytes in "workgroup-shared memory instead of a DRAM round-trip", but the standalone decode APIs (`base64_decode`, `hex_decode`, `inflate`) are the ones most users import and they **all** use DRAM. | Update docs to warn that standalone decode paths round-trip through DRAM. Push users toward the `*_then_aho_corasick` fused variants or `streaming::fuse_decode_scan`. |
| MEDIUM | `vyre-runtime/src/pipeline_cache.rs:258-289` | `canonical_wire` sorts buffers by `(binding, name)` before hashing, but it does **not** canonicalise the entry-node body (e.g. `a + 1` vs `1 + a`). The comment claims canonicalisation happens via `vyre_foundation::transform::optimize::canonicalize::run`, but that pass is not imported or called in this file. | Verify that `canonicalize::run` is actually invoked. If it is a no-op, implement commutative-expression sorting in the canonicaliser so structurally identical decode Programs share a cache key. |

---

## Architectural recommendations

1. **Single source of truth for decode.** Today there are at least five independent decode implementations (vyre-libs GPU, surgec CPU, encodex CPU, ziftsieve CPU, keyhog CPU). Unify on vyre Programs for all formats that are amenable to GPU execution (base64, hex, deflate literals, LZ4 literals). Keep CPU fallbacks only for formats that require host-side I/O (zip directory, tar headers).

2. **Linear fused scan by default.** The duplicated O(n²) decode scan body was removed from fused decode→scan composition. The replacement walks the decoded stream once and preserves byte-for-byte Aho-Corasick output parity.

3. **Magic-first dispatch.** Every decode kernel should begin with a 4-byte magic check in vyre IR (a single `Node::if_then` on `load(input, 0..3)`). If the magic doesn't match, the thread returns immediately without touching DRAM. This eliminates dispatch overhead for non-matching files.

4. **Cache discipline.** `surgec scan::decode` must cache compiled Programs per `(encoding, max_input_size, pattern_set_hash)`. Rebuilding CPU-side regex tables and GPU Programs for every file is unacceptable for a 100 GiB corpus scan.

5. **Stored-block naming discipline.** The shipped DEFLATE op is `inflate_stored_block`; callers cannot discover it as a general compressed-block decoder through the registry.
