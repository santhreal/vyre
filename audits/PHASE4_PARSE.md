# PHASE4_PARSE  -  Language Parser Audit

**Scope:** `libs/shared/surgec-grammar-gen`, `libs/tools/surgec/src/compile` (parser-related),
`libs/surge/src`, tree-sitter lowering into vyre Programs,
`libs/performance/matching/vyre/vyre-libs/src/parsing/*`,
`vyre-foundation/src/transform/compiler/*`,
`vyre-primitives/src/nfa/subgroup_nfa.rs`,
`vyre-libs/src/matching/nfa.rs`

**Auditor:** Kimi Code CLI (security-researcher mode)
**Date:** 2026-04-24

---

## Architecture Answers (the six specific asks)

1. **Tree-sitter state table on GPU?**  
   **No.** There is no tree-sitter runtime on the GPU hot path. Tree-sitter appears only in test fixtures (`go_frontend_corpus.rs`) and in the `surge-source` AP-1 adapter surface (trait + enum, no grammar wired). GPU parsers use hand-written DFA lexers expressed as vyre `Program` values, with transition tables uploaded as `BufferAccess::ReadOnly` storage buffers.

2. **AST build: single kernel dispatch or per-node loop?**  
   **Per-node / per-chunk loop.** The C11 GPU lexer dispatches one chunk per lane and loops over `chunk_size` bytes (`vyre-libs/src/parsing/c/lex/lexer.rs`). The visitor walk pops **one node per dispatch** and requires a host loop until the stack drains (`vyre-foundation/src/transform/compiler/visitor_walk.rs`). The recursive-descent primitive processes **one token per dispatch** with workgroup size `[1,1,1]`. There is no whole-AST single-kernel build today.

3. **LR(1) parse table: CSR or heap tree?**  
   **Dense flat array.** `surgec-grammar-gen/src/lr.rs` stores `action` as `Vec<u32>` of length `num_states * num_tokens` and `goto` as `num_states * num_nonterminals`. The generator serializes LR blobs from explicit caller-supplied tables and no longer emits a synthetic C11 LR blob.

4. **Preprocessor / macro expansion: fused or separate CPU stage?**  
   **Separate CPU stage.** `host_preprocess.rs` runs line splicing, comment stripping, `#if 0` folding, and conservative object-like `#define` / `#undef` expansion before lexing.

5. **String literal intern: CHD (G9) or linear scan?**  
   **FNV-1a + linear probing for interning; CHD is used only for label families.** The workgroup-local string interner (`vyre-foundation/src/transform/compiler/string_interner.rs`) uses FNV-1a 32-bit hash with linear probing and `atomicCompareExchangeWeak`. The CHD perfect hash (`vyre-libs/src/intern/perfect_hash.rs`) is reserved for Tier-B TOML label-family lookups (60k+ function names), not for lexer/parser string interning.

6. **Parser-level memoisation benefiting from G8 content-hash cache?**  
   **None.** There is no parse-node cache, no memoisation table, and no content-hash deduplication of AST subtrees. The only G8-style caching in the stack is the pipeline/shader compilation cache (`pipeline_disk_cache.rs`, `aot.rs`), which caches lowered WGSL and driver pipeline blobs, not parse results.

---

## Findings

### CRITICAL

| # | SEVERITY | file:line | description | suggested fix |
|---|----------|-----------|-------------|---------------|
| 1 | **CRITICAL** | `vyre-libs/src/parsing/c/lex/lexer.rs:135-148` | C11 GPU lexer ignores the DFA action field. It extracts `next_state` via `div(packed, 65536)` but never reads `packed & 0xFFFF` (the action). `EmitToken`, `PushBack`, and `Error` actions are silently discarded. Tokens requiring pushback or explicit emits will have wrong boundaries. | Decode the action field and branch on it inside the walk loop: `EmitToken` resets state and emits; `PushBack` rewinds `pos` by one; `Error` breaks the loop. |
| 2 | **CRITICAL** | `vyre-libs/src/matching/nfa.rs:230-236` | NFA scan hit emission hardcodes `start = input_len - pattern_len` and `end = input_len`, completely ignoring `cursor`. Every match reports the same global end-of-buffer position regardless of where it actually fired. | Emit the actual byte offset: `start = cursor + 1 - pattern_len`, `end = cursor + 1`. |
| 3 | **CRITICAL** | `vyre-libs/src/matching/nfa.rs:116-199` | The composed NFA scan uses nested `loop_for` over `num_states` for both transition and epsilon closure, giving **O(num_states²)** work per byte. At the 1024-state limit this is ~1M inner iterations per byte  -  the GPU will hang or timeout. | Replace with the subgroup-shuffle based `nfa_step` primitive from `vyre-primitives/src/nfa/subgroup_nfa.rs`, or at least bound the epsilon loop to the true ε-diameter. |

### HIGH

| # | SEVERITY | file:line | description | suggested fix |
|---|----------|-----------|-------------|---------------|
| 4 | **HIGH** | `vyre-foundation/src/transform/compiler/visitor_walk.rs:76-83` | GPU `visit_step_program` pushes children onto the stack without checking stack capacity. A deep or malicious tree increments `stack[0]` and writes past the buffer end, corrupting adjacent GPU memory. | Add a capacity check: if `stack[0] + 1 >= max_stack`, set a `stack_overflow` flag and skip the push. Mirror the CPU reference's bound. |
| 5 | **HIGH** | `libs/surge/src/lexer.rs:13` | `Span::offset` is `u32`, capping source files at 4 GiB. At internet scale (firmware blobs, concatenated corpora), offsets wrap silently, destroying diagnostic accuracy. | Change `offset` to `u64`, or add an explicit overflow check in `Tracker::next` that aborts with a `Fix: ...` error. |
| 6 | **FIXED** | `libs/shared/surgec-grammar-gen/src/dfa.rs` | `DfaBuilder::build` now defaults to `MatchKind::LeftmostFirst`; `MatchKind::All` remains an explicit opt-in through `build_with_match_kind`. | Verified by `cargo test -p surgec-grammar-gen`. |
| 7 | **HIGH** | `vyre-foundation/src/transform/compiler/string_interner.rs:239` | `intern_all` hardcodes `byte_capacity = slot_capacity * 64`. No empirical basis is given. For long identifiers (C++ mangled names, Java package paths) the byte pool overflows while slots remain empty. | Expose `byte_capacity` as a caller parameter, or sample the input distribution and derive a robust bound (e.g., 95th-percentile length × slots). |
| 8 | **HIGH** | `vyre-driver-wgpu/src/pipeline_disk_cache.rs:412-415` | Cache integrity relies on CRC32 (`crc32fast`). The birthday bound for 32-bit collisions is ~77k entries; with millions of cached pipelines a collision is probable. A corrupted blob with matching CRC32 passes validation and injects bad shader code. | Replace CRC32 with a truncated cryptographic hash (e.g., `blake3` first 16 bytes) for all cache checksums. |

### MEDIUM

| # | SEVERITY | file:line | description | suggested fix |
|---|----------|-----------|-------------|---------------|
| 9 | **FIXED** | `libs/shared/surgec-grammar-gen/src/host_preprocess.rs` | Host preprocessing now expands conservative object-like macros and removes stale non-scope wording. | Verified by `cargo test -p surgec-grammar-gen`. |
| 10 | **FIXED** | `libs/shared/surgec-grammar-gen/src/lr.rs`, `src/main.rs` | Public synthetic LR output was removed. The CLI still emits/dumps LR blobs, but only from concrete caller-supplied `LrTable` JSON. | Verified by `cargo test -p surgec-grammar-gen`. |
| 11 | **FIXED** | `vyre-foundation/src/transform/compiler/recursive_descent.rs` | CPU reference `parse` builds a `HashMap<(state, token_kind), Transition>` once and keeps first-match semantics for duplicate edges. | Verified by `cargo test -p vyre-foundation recursive_descent -- --nocapture`. |
| 12 | **MEDIUM** | `vyre-foundation/src/vast.rs:252-298` | `walk_preorder_indices` and `walk_postorder_indices` bound stack growth but do **not** detect cycles. A malformed VAST buffer with a cyclic sibling/child link spins until `max_stack`, producing an opaque `StackOverflow` instead of a clear `Cycle` error. | Maintain a `visited: Vec<bool>` bitset and return `VastError::Cycle` when a node index is revisited. |
| 13 | **MEDIUM** | `vyre-libs/src/intern/perfect_hash.rs:215` | CHD construction uses `candidate_slots.contains(&slot)`  -  an O(bucket_size) linear scan inside the displacement search. For buckets with dozens of keys, construction time degrades to O(n²). | Use a temporary `HashSet<usize>` or a `Vec<bool>` of length `table_size` to test slot occupancy in O(1). |
| 14 | **MEDIUM** | `libs/surge/src/parser/mod.rs:452` | `bump()` clones the entire `Token` payload on every consume. `Token::String` and `Token::Regex` hold owned `String`s; large repeated literals cause quadratic allocation pressure. | Replace `String` payloads with `Arc<str>` or a string-interned `Symbol` type so clones are refcount bumps. |

### LOW

| # | SEVERITY | file:line | description | suggested fix |
|---|----------|-----------|-------------|---------------|
| 15 | **FIXED** | `libs/shared/surgec-grammar-gen/src/wire.rs` | `SGGC` version 1 appends and verifies a BLAKE3-128 payload tag for DFA and LR blobs. | Verified by checksum-corruption tests in `cargo test -p surgec-grammar-gen`. |
| 16 | **LOW** | `vyre-libs/src/parsing/c/lex/lexer.rs:35-36` | Magic numbers `200` and `201` are used for whitespace/comment token IDs without named constants in the lexer file. | Define and import named constants (e.g., `TOK_WHITESPACE = 200`, `TOK_COMMENT = 201`) so the code is self-describing. |
| 17 | **LOW** | `vyre-primitives/src/nfa/subgroup_nfa.rs:209` | GPU epsilon closure is capped at `num_states.min(32).max(1)` iterations. For NFAs with ε-chains longer than 32, the GPU result diverges from the CPU reference which runs to true fixpoint. | Cap at `num_states` (up to 1024) or add a convergence loop using `subgroup_ballot` to detect when no new states were added. |
| 18 | **LOW** | `vyre-foundation/src/transform/compiler/recursive_descent.rs:112` | `consume_step_program` uses workgroup size `[1, 1, 1]`. Only one lane parses; 31+ subgroup lanes are idle. Batch parsing of many small token streams wastes 97% of SIMD throughput. | Add a batch dimension: dispatch N token streams in parallel, one per lane, each with its own state/output buffers. |

---

## Competitor Comparison

| Capability | Vyre (today) | Best-in-class (tree-sitter / ANTLR / hand-written) |
|------------|--------------|-----------------------------------------------------|
| GPU lexer | Hand-written DFA, dense 256-column table, chunk-parallel |  -  (no mainstream GPU lexer) |
| GPU parser | Per-node host-loop visitor walk; 1-token-per-dispatch recursive descent |  -  (research prototypes only) |
| String interning | FNV-1a + linear probing in workgroup SRAM | `string_cache` (Rust), `intern` (Go) use perfect hashing or arena dedup |
| LR table format | Dense flat array | Bison/Yacc use compressed row/column (LALR), some use bit-matrix |
| Preprocessor | None (C subset only) | Clang preprocessor is fused but separable; every production C parser has one |
| Error recovery | None (fail-fast) | Tree-sitter has robust error-recovery and partial parse trees |
| Parse caching | None | `tree-sitter` WASM builds cache parse trees in JS engines; ANTLR has DFA cache for LL(*) prediction |

Key gaps vs. competitors:
- **Tree-sitter** provides error-tolerant parsing and incremental re-parsing. Vyre has no error recovery at any layer.
- **Clang** preprocesses, lexes, and parses in a single fused pipeline with full macro expansion. Vyre's C frontend skips preprocessing entirely.
- **rustc** uses string interning (`Symbol`) everywhere, making AST clones nearly free. Vyre's SURGE parser clones `String` on every token consume.

---

## Summary

The parser stack still has non-scoped findings in this historical audit (#1, #2, #3, #4, #8, #12, #14, #17, #18). The scoped rows touched by this decode/parser pass are marked `FIXED` above with source/tests.
