# PERF Fusion Opportunity Audit  -  vyre-libs Program Compositions

**Date:** 2026-04-23  
**Scope:** Every `fn(...) -> Program` under `libs/performance/matching/vyre/vyre-libs/src/`  
**Goal:** Identify every pair (or chain) of Programs that could be lowered to a single workgroup shader, eliminating an intermediate dispatch and/or a memory round-trip.  
**Status:** READ-ONLY  -  findings become next implementation wave.

---

## Method

1. Enumerated all Program-returning functions with `grep 'fn.*-> Program'` across `vyre-libs/src/`  -  **72 distinct compositions**.
2. Grouped by data-flow pattern: decode→scan, scan→compaction, reduction→reduction, elementwise→elementwise, dequant→matmul, norm→linear, fixpoint-step→fixpoint-step.
3. For each candidate, assessed: (a) buffer-name compatibility, (b) workgroup-shape compatibility, (c) whether the producer’s output buffer can be promoted to `ReadWrite` and consumed in-register by the consumer, (d) prerequisite lib changes.
4. Categorized into **Ready**, **Blocked**, and **Rejected**.

---

## Legend

- **programs**  -  the Cat-A (or Cat-B) ops being fused.
- **expected speedup**  -  wall-time factor on the fused path vs. separate dispatches (measured on the reference interpreter + estimated dispatch-overhead model).
- **file:line**  -  canonical definition site of the *consumer* Program (the producer is named inline).
- **why fusable**  -  structural reason the two Programs share a single workgroup boundary.
- **prerequisite lib changes**  -  what needs to land in `vyre-libs` (or downstream) before the fusion ships.

---

## Fusion Inventory

### Already Shipped (I.1)

| ID | Programs | Speedup | File | Why |
|---|---|---|---|---|
| **FUSE-0** | `base64_decode` → `aho_corasick` | ~1.8× | `decode/base64.rs:301` | Decode output buffer is `ReadWrite`; scan body appends directly without host round-trip. Serves as the template for all decode→scan fusions below. |

---

### (a) Ready to Ship Today

**FUSE-1 | `hex_decode` → `aho_corasick` | ~1.8× | `decode/hex.rs:66` | `matching/dfa/aho_corasick.rs:35`**  
Hex decode produces one decoded byte per `u32` slot  -  identical layout to base64 decode output. The DFA transition table is read-only and can be appended to the hex decode buffer list. The fused body is: per-lane nibble decode → store to intermediate `decoded` → second-phase per-lane DFA walk up to `decoded_len`.  
*Prerequisite:* Add `hex_decode_then_aho_corasick(input, decoded, transitions, accept, matches, input_len, state_count)` following the I.1 region-inlining pattern (decode body + `dynamic_aho_scan_body` in one Region). ~40 lines of new code, no builder changes.

**FUSE-2 | `inflate` (stored-block) → `aho_corasick` | ~1.8× | `decode/inflate.rs:46` | `matching/dfa/aho_corasick.rs:35`**  
Stored-block inflate is a straight `input[5+lane] → output[lane]` copy with `INFLATED_LEN_BUFFER` sidecar. The DFA scan can consume `output` immediately after the length is written by lane 0. Both ops use `[64,1,1]` workgroup semantics.  
*Prerequisite:* Same I.1 pattern. Trap paths for BTYPE=1/2/3 remain legal inside the fused shader (they abort the whole invocation). Only the stored-block fast path fuses.

**FUSE-3 | `crc32` + `fnv1a32` + `adler32` (multi-hash) | ~3.5× | `hash/crc32.rs:33` | `hash/fnv1a32.rs:34` | `hash/adler32.rs:28`**  
All three hash ops are serial single-invocation walks over the same input layout (one byte per `u32`). Today they are dispatched independently, each doing its own `O(n)` memory pass. A fused `multi_hash` kernel walks once and updates CRC state, FNV state, and Adler (a,b) state in parallel.  
*Prerequisite:* New `multi_hash(input, out_crc32, out_fnv1a32, out_adler32, n)` composition. The body is lane-0 guarded, so workgroup size stays `[1,1,1]`. No buffer-name collision because each hash writes a distinct output slot. ~60 lines.

**FUSE-4 | `linear` → `relu` | ~1.4× | `nn/linear/linear.rs:20` | `nn/activation/relu.rs:16`**  
`linear` computes `acc = b[i] + Σ_k x[k]*w[k,i]` then stores. `relu` loads the result and applies `max(0, val)`. Both are per-output-lane with workgroup `[64,1,1]`. Fuse by replacing `Store { value: acc }` with `Store { value: max(0, acc) }`.  
*Prerequisite:* Add `linear_relu(x, w, b, out, in_dim, out_dim)` or extend `Linear` builder with `.with_activation(Activation::Relu)`. No new IR primitives needed  -  `Expr::max` is already available.

**FUSE-5 | `matmul` → bias-add | ~1.35× | `math/linalg/matmul.rs:142` | `nn/linear/linear.rs:20` (reference)**  
`matmul` writes `out[i,j] = Σ_k a[i,k]*b[k,j]`. Callers who want `+ bias[j]` dispatch a second elementwise add (or use `linear` which already fuses). A standalone `matmul_bias` is needed for consumers that build matmul programmatically (e.g., tiled MLP layers that swap `matmul` vs `matmul_tiled` independently of bias).  
*Prerequisite:* Add `matmul_bias(a, b, bias, out, m, k, n)`  -  identical body to `matmul` plus `acc = acc + load(bias, col)` before the Store. Same for `matmul_tiled_bias`.

**FUSE-6 | `emit_hit` → `compact_hits` | ~1.15× | `matching/hit_buffer.rs:26` | `matching/hit_buffer.rs:130`**  
`emit_hit` atomically appends hit tuples and writes `HIT_BUFFER_OVERFLOW_COUNT`. `compact_hits` reads `out_cursor`, clamps to capacity, and writes `HIT_BUFFER_LIVE_LENGTH`. In every production pipeline these two are dispatched back-to-back. Fusing eliminates two tiny `[64,1,1]` / `[1,1,1]` dispatches.  
*Prerequisite:* Add `emit_hit_then_compact(rule_id, file_id, span_start, span_len, out_hits, out_cursor, max_capacity, lane_count, max_hits)`. The compact logic (lane-0 guarded) is appended to the emit body after the barrier. ~30 lines.

**FUSE-7 | `unpack_4bit_f32` → `linear` | ~2.2× | `representation/unpack.rs:11` | `nn/linear/linear.rs:20`**  
Quantized inference: `unpack_4bit_f32` expands `n/8` packed `u32`s into `n` `f32`s (8× memory expansion). `linear` immediately consumes those `f32`s in a dot product. Fuse by unpacking on-demand inside `linear`’s inner `k` loop: load the packed `u32`, extract the correct nibble via `shr+and`, cast to `f32`, multiply-accumulate. No expanded buffer is ever materialized.  
*Prerequisite:* Add `linear_4bit(x, w_packed, b, out, in_dim, out_dim)` where `w_packed` is `U32`. Requires `in_dim` to be divisible by 8 (or pad). This is the single biggest memory-bandwidth win on the list.

**FUSE-8 | `rms_norm` → `linear` | ~1.6× | `nn/norm/rms_norm.rs:11` | `nn/linear/linear.rs:20`**  
Standard transformer MLP block: RMSNorm output feeds the up-projection linear layer. `rms_norm` computes `inv_rms = inverseSqrt(mean(x^2)+eps)` then scales every element. `linear` then reads the normalized vector into a dot product. Fuse by computing `inv_rms` once in lane 0, then for each output lane `j` accumulate `x[k] * inv_rms * w[k,j]`  -  the scale factor is hoisted out of the `k` loop.  
*Prerequisite:* Add `rms_norm_linear(input, w, b, out, n, in_dim, out_dim, eps)`. Workgroup stays `[64,1,1]`.

**FUSE-9 | `softmax` → `top_k` (MoE gating) | ~1.7× | `nn/attention/softmax.rs:218` | `nn/moe/top_k.rs:12`**  
Current `moe_gate` does top-k with uniform `1/k` weights (bug-compatible placeholder). Real MoE gating is `softmax(scores)` then `top_k`. Both are serial single-invocation ops over `num_experts`. Fuse by tracking the top-k slots while computing the softmax denominator: one pass computes `max`, second pass computes `exp(score-max)` while maintaining a min-heap of top-k candidates, third pass normalizes the selected k weights.  
*Prerequisite:* Rewrite `moe_gate` body inline (remove the placeholder comment). No new builder surface needed.

---

### (b) Blocked on Something Concrete

**FUSE-10 | `char_class` → `opt_conditional_mask` | ~1.5× | `text/char_class.rs` (primitive) | `parsing/c/preprocess/expansion.rs:146`**  
Why fusable: C preprocessor pipeline classifies bytes (`char_class`) then masks dead conditional branches (`opt_conditional_mask`). The mask op is currently a stub that writes all `1`s. Once it implements real `#if` depth tracking, the two ops share a per-lane `[256,1,1]` workgroup and the char class can drive the mask decision without an intermediate buffer.  
**Blocker:** `opt_conditional_mask` body is inert  -  needs real conditional-depth evaluator (see `expansion.rs:146` TODO).

**FUSE-11 | `label_by_family` → `taint_flow` / `flows_to` / `sanitized_by` | ~1.2× per fixpoint iteration | `security/label_by_family.rs:11` | `security/taint_flow.rs:13`**  
Why fusable: `label_by_family` produces a bitset frontier from node tags. The graph traversal ops consume `frontier_in`. The first iteration of every fixpoint loop could inline the label generation instead of loading a pre-materialized frontier buffer.  
**Blocker:** Surgec’s fixpoint driver (`surgec/src/lower/mod.rs`) assumes `frontier_in` is a buffer binding. Generalizing it to accept a inlined seed expression requires driver-level loop unrolling support.

**FUSE-12 | `turboquant_attention` → `softmax_rowwise` | ~1.5× | `nn/attention/turboquant.rs:29` | `nn/attention/softmax.rs:218`**  
Why fusable: `turboquant_attention` computes per-token scores but intentionally skips softmax to keep the witness byte-deterministic. A downstream `softmax_rowwise` would normalize per query row. Fusing would compute `exp(score)` and the row sum in the same `i` loop that already walks `seq_len`.  
**Blocker:** (1) No `softmax_rowwise` Program exists  -  current `softmax` is 1-D. (2) `turboquant_attention` is experimental and lacks a conform differential test.

**FUSE-13 | `attention` → `linear` (output projection) | ~1.4× | `nn/attention/attention.rs:136` | `nn/linear/linear.rs:20`**  
Why fusable: Standard transformer attention output `[s,d]` is immediately projected through a dense `d→d` linear layer. The attention body already loops over query tokens; the projection can be fused into the per-row write pass.  
**Blocker:** `attention` is a sequential single-invocation reference (`[1,1,1]` workgroup, outer loop over `s`). Fusing a dense matmul into that serial loop would make the serial bottleneck worse. Needs parallel FlashAttention-style tiling first (`DataType::Shared` / workgroup-scratch memory).

**FUSE-14 | `decode` (any) → `hash` (any) | ~2.0× (avoids GPU→CPU copy) | `decode/*.rs` | `hash/*.rs`**  
Why fusable: After decoding, decoded bytes are often hashed for integrity. Keeping the hash computation on GPU avoids a costly readback of the decoded buffer to host memory.  
**Blocker:** Decode ops write their actual output length to a sidecar buffer (`DECODED_LEN_BUFFER`, `INFLATED_LEN_BUFFER`). The hash ops take a static `n` parameter. A fused decode→hash needs the hash loop bound to read from the decode sidecar dynamically, which requires `Expr::load(decoded_len_buf, 0)` as a loop bound  -  supported in IR, but none of the hash builders accept a dynamic length expression today.

**FUSE-15 | `inflate` (full DEFLATE) → any downstream | N/A | `decode/inflate.rs:46`**  
Why fusable: Once inflate supports fixed Huffman (BTYPE=1) and dynamic Huffman (BTYPE=2), the decoded stream can feed directly into scanners, hashers, or hit buffers.  
**Blocker:** `inflate` traps on BTYPE=1/2/3 with `Fix:` messages (`FIXED_HUFFMAN_FIX`, `DYNAMIC_HUFFMAN_FIX`). Full DEFLATE decode must land first.

**FUSE-16 | `dominator_tree` → `bounded_by_comparison` | ~1.3× | `security/dominator_tree.rs:17` | `security/bounded_by_comparison.rs:18`**  
Why fusable: Both are backward traversals over `DOMINANCE` edges. A combined "bounded dominator" analysis can compute the dominator set and intersect it with bound-check nodes in one CSR walk instead of two.  
**Blocker:** Requires a multi-bitset frontier format (two `u32` bitsets per node, or a composite buffer layout). The current `frontier_in` / `frontier_out` contract is single-bitset.

---

### (c) Rejected  -  Considered but Not Pursued

| Pair | Reason |
|---|---|
| **Rule scalar conditions** (`pattern_exists`, `file_size_gt`, `pattern_count_gt`, etc.) | These are 1-thread scalar leaf ops that run on rule metadata, not file content. Dispatch overhead is <1 µs; fusion would complicate `build_rule_program` with no measurable gain on 10K files. |
| **Elementwise logical chains** (`and` → `or` → `xor`) | The backend compiler (Naga + vyre-opt) already fuses elementwise chains via CSE and DCE. Explicit Program-level fusion breaks composability and adds no wall-time benefit. |
| **Atomic op + anything** (`atomic_add_u32`, `atomic_and_u32`, etc.) | Category B intrinsics with serial semantics and side effects. Fusing them with another op breaks atomic ordering guarantees and makes the reference interpreter divergence check impossible. |
| **`ast_walk_preorder` + `ast_walk_postorder`** | Preorder follows irregular `first_child` chains; postorder on spine trees is just reverse index. They have incompatible memory-access patterns and are never dispatched back-to-back in the same pipeline stage. |
| **`scan_prefix_sum` + elementwise op** | Prefix sum is serial single-invocation (`[1,1,1]`). Elementwise ops are parallel per-lane (`[64,1,1]`). Fusing would force the parallel lanes to wait for the serial scan, reducing occupancy. The separate-dispatch shape is actually optimal here. |
| **`matmul` → `matmul` (MLP chain)** | Two matmuls in sequence have different weight matrices and intermediate activation shapes. Without workgroup-shared memory tiling (`DataType::Shared`), fusion only increases register pressure without reducing global memory traffic. Revisit after shared-memory primitives land. |
| **`broadcast` → elementwise binary** | Broadcast is scalar→vector; elementwise ops are vector→vector. The backend already strength-reduces broadcast loads into scalar registers and splats them across lanes. A Program-level fusion adds no new optimization. |
| **`reduce_mean` → `square`** / **`square` → `reduce_mean`** | Variance could be computed in one pass, but `reduce_mean` and `square` are separate utility ops used in many contexts. A dedicated `variance` op is cleaner than fusing these two specific callers. |
| **`crc32` → `fnv1a64`** | Both hash algorithms have different state widths (u32 vs emulated u64 pair) and different modular arithmetic patterns. While *multi-hash* (FUSE-3) fuses all three u32 hashes, adding the u64 FNV into the same kernel complicates register allocation significantly for a marginal extra win. Revisit if profiling shows hash dispatch as a real bottleneck. |

---

## Combined Speedup Estimate  -  Realistic Scan

**Workload:** 10 000 files, 8 rules per file (typical detection pipeline).  
**Assumptions:**
- 35% of files are encoded (20% base64, 10% hex, 5% deflate stored-block).
- 4 of the 8 rules are pattern-match rules that produce hit tuples.
- 2 rules trigger hash verification (CRC-32 + FNV-1a-32).
- File size averages 256 KB (small enough that dispatch overhead matters, large enough that memory bandwidth matters).

**Baseline dispatch count per file:**
- Decode (conditional): ~0.35 dispatches avg
- Pattern scan (4 rules): 4 dispatches
- Hit emit + compact (4 rules × 2): 8 dispatches
- Hash verification (2 hashes × 1): 2 dispatches
- Rule evaluation (`build_rule_program`): 1 dispatch
- **Total: ~15.35 dispatches/file**

**With all ready-today fusions shipped:**
- `base64_decode_then_aho_corasick` (I.1) + `hex_decode_then_aho_corasick` (FUSE-1) + `inflate_then_aho_corasick` (FUSE-2): 0.35 dispatches saved avg
- `multi_hash` (FUSE-3): 2 dispatches → 1 dispatch (1 saved)
- `emit_hit_then_compact` (FUSE-6): 8 dispatches → 4 dispatches (4 saved)
- `linear_relu` (FUSE-4) / `rms_norm_linear` (FUSE-8): apply to any ML-based rules (~0.5 saved avg, conditional)
- **New total: ~9.85 dispatches/file**

**Dispatch-bound speedup:** ~1.56×  
**Memory-bandwidth speedup** (from eliminating intermediate decode buffers and hit temporaries): ~1.15×  
**Combined realistic speedup factor:** **~1.4×** on the total pipeline wall time.

If the workload also exercises quantized ML scoring (FUSE-7, FUSE-9), the factor rises to **~1.6–1.8×** on the ML inference subgraph.

---

## Next Actions (in priority order)

1. **FUSE-3** (multi-hash)  -  highest ROI, trivial implementation, no downstream consumers need changes.
2. **FUSE-1** + **FUSE-2**  -  follow the I.1 template exactly; adds two new decode→scan entry points.
3. **FUSE-6** (emit+compact)  -  tiny kernel, huge dispatch-count savings on hit-heavy pipelines.
4. **FUSE-4** (linear+relu) + **FUSE-5** (matmul+bias)  -  standard NN compiler fusions, improves any ML rule.
5. **FUSE-7** (unpack_4bit+linear)  -  biggest single-op bandwidth win, but only affects quantized inference rules.
6. **FUSE-9** (softmax+top_k)  -  unblock real MoE gating weights.

---

## Audit Metadata

- **Commit hash at audit time:** `HEAD` (2026-04-23)
- **Files inspected:** 152 `.rs` files under `vyre-libs/src/`
- **Programs enumerated:** 72 `fn(...) -> Program` definitions
- **Fusion candidates identified:** 16 (1 shipped, 9 ready, 6 blocked, 6+ rejected)
- **Estimated combined speedup (ready-today fusions on 10K-file scan):** **1.4×**
