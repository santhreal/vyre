# VYRE_MEM_LAYOUT  -  Buffer Layout vs GPU Cacheline Audit

**Auditor:** kimik-code-cli  
**Scope:** `vyre-primitives/src`, `vyre-libs/src`  -  BufferDecl layouts, ProgramGraph, hit-buffers, NFA/DFA transition tables, exploded supergraph, bitset word order, u64 atomicity.  
**Date:** 2026-04-24  
**Findings:** 18  

---

## Summary

The vyre memory layout has systemic issues against GPU cache-line geometry (16B–64B). AoS dominates where SoA would coalesce, row-major transition tables stride by 256–1024 bytes, cross-module layout mismatches exist between composition and primitive layers, `MemoryHints` are dead fields, and U64 is silently split into two u32 loads with no atomicity contract. Every finding below is fixable without API breakage.

---

## Findings

### 1. CRITICAL | `vyre-libs/src/matching/nfa.rs:362` | `bit_in_word` is undefined  -  transition table build is uncompileable

`build_transition_table` calls `bit_in_word(word_idx, bit)`, but no such function exists in the repository. The table is supposed to be a flat `[num_states × 256]` of u32 bitsets; `bit` is already `1 << (dst % 32)`. The call should be `table[idx] |= bit;`. As written, the crate fails to compile and the NFA scan path is completely dead.

**Suggested fix:** Delete the `bit_in_word` call and OR `bit` directly into `table[idx]`.

---

### 2. CRITICAL | `vyre-libs/src/matching/nfa.rs:278` vs `vyre-primitives/src/nfa/subgroup_nfa.rs:227` | Transition table layout mismatch between composition and primitive

`vyre-libs::matching::nfa::build_transition_table` emits a flat `[num_states × 256]` u32 table.  
`vyre-primitives::nfa::subgroup_nfa::nfa_step` declares `transition_buf` as `[num_states × 256 × LANES_PER_SUBGROUP]` with indexing `src_state * 256 * LANES + byte * LANES + lane`.  
These layouts are byte-incompatible. A caller that builds a table with the helper and dispatches the primitive will read garbage.

**Suggested fix:** Unify on one canonical layout. If the primitive needs the lane-major expansion, provide a host-side `expand_for_subgroup` helper in `vyre-libs` that converts the flat table to the primitive shape, and document the ABI contract.

---

### 3. HIGH | `vyre-libs/src/matching/hit_buffer.rs:79-94` | Hit buffer `emit_hit` uses AoS  -  4-word stride breaks coalescing

The output layout is `(rule_id, file_id, span_start, span_len)` stored interleaved:
```
out_hits[base+0] = rule_id
out_hits[base+1] = file_id
out_hits[base+2] = span_start
out_hits[base+3] = span_len
```
With 4 lanes active, each lane writes to offsets 0, 4, 8, 12  -  contiguous, but the *fields* of the same hit are scattered. If downstream kernels access only `rule_id` or only `span_start`, they stride by 4 words (16 bytes). On a 32-byte cache line, two hits share one line but only one field is used, wasting 75% of fetched bytes.

**Suggested fix:** Provide a SoA variant (four separate buffers, or one buffer with four contiguous regions) when the consumer is field-selective. Keep the AoS path only when the consumer needs the full tuple sequentially.

---

### 4. HIGH | `vyre-libs/src/matching/nfa.rs:251-260` | NFA hit triples are AoS not SoA

Accept-state hits are written as:
```
hit_buf[1 + 3*slot_idx + 0] = pattern_id
hit_buf[1 + 3*slot_idx + 1] = start
hit_buf[1 + 3*slot_idx + 2] = end
```
Three-word stride. If the compaction kernel later scans only `pattern_id` to filter duplicates, every load pulls 12 bytes of unused data.

**Suggested fix:** Switch to SoA: `[counter][pattern_id…pattern_id][start…start][end…end]`. The counter already lives at index 0; extend the pattern to three contiguous arrays starting after the counter.

---

### 5. HIGH | `vyre-libs/src/matching/nfa.rs:148` | NFA transition table row-major with 1024-byte row stride

The per-byte lookup index is `src * 256 + byte`. Each row is 256 u32s = 1024 bytes. When a workgroup processes multiple source states in parallel (e.g. one source per lane in the unrolled loop), consecutive lanes access memory 1024 bytes apart. This exceeds GPU cache-line size (32–64 bytes) by 16–32×, causing every load to miss L1 and hit L2/DRAM.

**Suggested fix:** Transpose to byte-major: `byte * num_states + src`. Then lanes processing consecutive `src` values on the same `byte` access contiguous memory (4 bytes apart), which coalesces perfectly.

---

### 6. HIGH | `vyre-libs/src/matching/aho_corasick.rs:60` / `cooperative_dfa.rs:129` | DFA transition table repeats the 1024-byte row-stride mistake

Both DFA scanners use `state * 256 + byte`. The same cache-line thrashing applies. `cooperative_dfa_scan` additionally does `subgroup_shuffle` correction rounds; the shuffled state is then used to index the same strided table, compounding the waste.

**Suggested fix:** Same transpose as finding #5, or tile the table into 32×32 blocks so a subgroup loads a cache-line-friendly tile.

---

### 7. HIGH | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:272` | U64 mapped to `vec2<u32>`  -  split-word atomicity violation

`DataType::U64` is lowered to `vec2<u32>`. The array stride is 8 bytes, but WGSL `vec2<u32>` loads/stores are not guaranteed atomic on all Vulkan targets (e.g., some mobile GPUs lack 64-byte-aligned vector atomicity). The reference interpreter (`vyre-reference`) handles U64 as a single `u64`, but the WGSL backend splits the semantic type into two u32 words. A concurrent write to the same U64 element from two invocations can tear: one word from the old value, one from the new.

**Suggested fix:** Reject U64 buffers in the wgpu backend until a full U64 emulation pass (carry-propagation, 64-bit CAS loop) is implemented, or document the torn-read risk and require `atomic<u64>` where the hardware supports it.

---

### 8. MEDIUM | `vyre-foundation/src/ir_inner/model/program/buffer_decl.rs:20` | `BufferDecl` is an AoS CPU struct with `Arc<str>` indirection

The struct contains:
- `name: Arc<str>` (16 bytes + heap allocation)
- `binding: u32`, `access: BufferAccess`, `kind: MemoryKind`, `element: DataType`
- `count: u32`, `is_output: bool`, `pipeline_live_out: bool`
- `output_byte_range: Option<Range<usize>>` (24 bytes)
- `hints: MemoryHints` (8 bytes)
- `bytes_extraction: bool`

Total size is >80 bytes with poor alignment. When the optimizer iterates over `program.buffers()` (e.g. in `compute_stats`), every buffer declaration may span two cache lines, and the `Arc<str>` causes a second cache miss to fetch the string data.

**Suggested fix:** Store buffer names in a side-car interner (e.g. `Arc<str>` table in `Program`) and replace `BufferDecl.name` with a `u32` name-id. This shrinks the struct to ~32 bytes and removes the indirection.

---

### 9. MEDIUM | `vyre-primitives/src/graph/program_graph.rs:104-148` | ProgramGraph CSR uses 5 disjoint buffers  -  no prefetch, no cache-line packing

The canonical ProgramGraph splits nodes, edge_offsets, edge_targets, edge_kind_mask, and node_tags into five separate GPU buffers. A traversal kernel (`csr_forward_traverse`) touches at least three of them per source node: `edge_offsets`, `edge_targets`, `edge_kind_mask`. Each buffer is a separate binding with independent base address and allocation alignment. The GPU memory controller cannot prefetch across binding boundaries, and each buffer may start at an unrelated cache-line offset, causing up to 5× cache-line fetches for what could be one fused fetch.

**Suggested fix:** Provide an optional fused-buffer ABI that interleaves or concatenates the five arrays with 64-byte padding between them, so the prefetcher can stream through one contiguous region. Keep the split ABI for callers that need sparse edges.

---

### 10. MEDIUM | `vyre-primitives/src/fixpoint/bitset_fixpoint.rs:62-63` | Every lane does `atomic_or` on the same `changed[0]` word

```rust
Expr::atomic_or(changed, Expr::u32(0), Expr::u32(1))
```
All 256 lanes (or however many are active) contend for the same memory location. On NVIDIA/AMD this serializes through the L2 atomic unit and causes cache-line bouncing across SMs.

**Suggested fix:** Use a subgroup ballot (`subgroupBallot(c != n)`) followed by a single lane atomic_or, or use a per-lane scratch flag in shared memory and reduce with a subgroup OR before writing one atomic to global memory.

---

### 11. MEDIUM | `vyre-primitives/src/graph/adaptive_traverse.rs:139` | Dense adjacency rows are not padded to cache-line boundary

`adap_adj_rows_dense` is `node_count × bitset_words(node_count)` u32s. Each row is `words` u32s long. If `words` is odd (e.g. 33 nodes → 2 words, 65 nodes → 3 words), the next row starts at an 8-byte or 12-byte offset, not a 32-byte or 64-byte cache-line boundary. When the workgroup iterates `for w in 0..words` loading `adj_rows_dense[row_start + w]`, the first load of the next row may span two cache lines.

**Suggested fix:** Pad `words` up to the next cache-line multiple (e.g. `words.div_ceil(16) * 16` for 64-byte lines). Add a `padded_bitset_words` helper and use it for all dense row-major layouts.

---

### 12. MEDIUM | `vyre-primitives/src/graph/exploded.rs:145-148` | Exploded supergraph dense index is proc-major  -  cross-proc BFS strides by `blocks × facts`

The dense index is:
```rust
idx = p * blocks * facts + b * facts + f
```
In the IFDS BFS (`ifds_gpu_step`), each invocation handles one source node. If the frontier spans multiple procedures, consecutive invocations may access indices `blocks × facts` apart (up to 1024×1024 = 1M floats / 4M bytes). This is a worst-case strided access pattern for the CSR `row_ptr` and `col_idx` arrays built from the dense space.

**Suggested fix:** For GPU BFS, use a tiled or Z-ordered mapping of `(p, b, f)` so that spatially nearby nodes are also nearby in memory. Alternatively, keep the dense index for CPU reference but emit a GPU-specific tiled index during `build_cpu_reference`.

---

### 13. MEDIUM | `vyre-foundation/src/ir_inner/model/program/mod.rs:68-75` | `MemoryHints.preferred_alignment` is a dead field  -  no backend reads it

`MemoryHints` carries `preferred_alignment: u32` and `cache_locality: CacheLocality`. The wire format encodes both (`to_wire.rs:301`), and `buffer_decl_canonical_key` hashes them. However, `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs` (`add_buffer`) never consults `buffer.hints.preferred_alignment`; it derives stride solely from `buffer.element.size_bytes()`. The field is therefore decorative and gives callers a false sense of control.

**Suggested fix:** Either honor `preferred_alignment` in the wgpu backend by padding the Naga array stride, or remove the field from the public API to avoid lying to users.

---

### 14. MEDIUM | `vyre-libs/src/decode/streaming.rs:103` | Decode-scan fuse handoff uses `DataType::U32` for byte stream  -  4× shared-memory bloat

`promote_to_workgroup` redeclares the handoff buffer as:
```rust
BufferDecl::workgroup(handoff_buf, count, DataType::U32)
```
If the decoder emits bytes (e.g. from `base64` or `inflate`), each byte occupies a 4-byte u32 slot in workgroup memory. On GPUs with 48 KB shared memory per workgroup, this reduces the maximum decode chunk from 48 KB to 12 KB, cutting occupancy and throughput.

**Suggested fix:** Support `DataType::U8` (or `DataType::Bytes`) in workgroup buffers, or pack four bytes per u32 and unpack in the scanner. The wire format already allows `Bytes` but the wgpu backend rejects it; fixing the backend unlocks this path.

---

### 15. MEDIUM | `vyre-libs/src/compiler/types_layout.rs:34` | `c11_compute_alignments` alignment is a mock (`base_size % 8`)  -  never produces cache-line alignment

The kernel computes:
```rust
Expr::rem(Expr::var("base_size"), Expr::u32(8))
```
For a 4-byte type this yields 4; for an 8-byte type it yields 0. It never emits 16, 32, or 64, which are the actual GPU cache-line and shared-memory bank sizes. The docstring claims "strict CPU cache-line compliance" but the implementation contradicts it.

**Suggested fix:** Replace the mock with a real C11-layout evaluator that respects `alignof(T)` up to 64 bytes, or delete the function until it is implemented correctly.

---

### 16. LOW | `vyre-primitives/src/graph/program_graph.rs:133-140` | Zero-edge graphs allocate 1-word placeholder buffers  -  allocator overhead dwarfs payload

`read_only_buffers()` uses `edge_count.max(1)` for `edge_targets` and `edge_kind_mask`. A graph with 0 edges still ships two 1-word buffers. GPU memory allocators typically round minimum allocations to 256 bytes or 4 KB. The kernel never indexes these placeholders (every `edge_offsets[i+1] == edge_offsets[i]`), but the allocator still reserves full pages, wasting TLB entries and cache tags.

**Suggested fix:** Allow zero-count edge buffers when `edge_count == 0` and teach the validator and backends to skip binding empty storage buffers. Vulkan and WebGPU both permit zero-sized bindings in some configurations.

---

### 17. LOW | `vyre-primitives/src/graph/motif.rs:51-77` | Motif edge fields in three separate buffers  -  3 cache-line fetches per edge on small motifs

`motif_from`, `motif_kind`, and `motif_to` are separate bindings. A motif edge check loads one word from each buffer. If the motif has few edges (e.g. 3–5), each buffer is tiny but lives in its own allocation, so the three loads likely hit three different cache lines. The combined working set is ~12 bytes but the cache footprint is ~192 bytes.

**Suggested fix:** Fuse the three fields into one interleaved buffer `(from, kind, to)` or use a single `vec3<u32>` array. This reduces cache-line pressure from 3 lines to 1 line per edge.

---

### 18. LOW | `vyre-libs/src/compiler/object_writer.rs:72-93` | ELF header stores 64-bit fields as split u32 pairs with no atomicity guard

The ELF64 header writes 64-bit fields (e.g. `e_entry`, `e_phoff`, `e_shoff`) as two separate u32 stores:
```rust
Node::store(target_object_bytes, Expr::u32(6), Expr::u32(0x00000000));
Node::store(target_object_bytes, Expr::u32(7), Expr::u32(0x00000000));
```
Indices 6 and 7 are adjacent u32 slots. A reader loading a 64-bit quantity from the same buffer concurrently could observe a torn value (one word updated, the other not yet). While this is inside a single thread's initialization block, the pattern is replicated in the loop body where multiple threads atomic-add into `.text` and then store machine code. If the object buffer is ever read while being written, 64-bit fields tear.

**Suggested fix:** Use a single 64-bit write expression (when U64 lowering is fixed) or insert an explicit `Node::Barrier` between the header write and the loop-body writes, and document that the output buffer must not be read until the dispatch completes.

---

## Competitor Comparison

| System | Cache-line alignment | AoS/SoA default | U64 atomicity | Transition table layout |
|--------|---------------------|-----------------|---------------|------------------------|
| **vyre (current)** | None enforced | AoS (hit buffers, NFA) | Split vec2<u32>, no contract | Row-major 1024B stride |
| **TensorFlow/XLA** | 64B aligned alloc | SoA (tiled) | Rejected on TPU; emulated on GPU | Tiled 32×32 blocks |
| **Triton** | 128B shared mem banking | SoA (blocked) | Native via PTX `mov.u64` | Tiled & swizzled |
| **WebGPU best practice** | `minStorageBufferOffsetAlignment` | SoA for coalescing | `atomic<u64>` optional | Transpose or blocked |

vyre is behind the state of the art on every axis in this table. The good news: all fixes are local and additive.

---

## Action Priority

1. **Fix #1 and #2 immediately**  -  they block the NFA scan path from compiling or running correctly.  
2. **Fix #5 and #6**  -  transition-table transposition is the single biggest perf win for the matching pipeline.  
3. **Fix #3 and #4**  -  SoA hit buffers unlock coalesced compaction and filtering.  
4. **Fix #10**  -  atomic contention on the fixpoint flag limits graph-analysis scalability.  
5. **Fix #7, #13, #14**  -  remove dead/misleading APIs and unlock dense byte packing.

---

*End of audit.*
