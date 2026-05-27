# PHASE3_SCAN  -  Audit Report

**Scope:** NFA / DFA / Aho-Corasick / gpumatch scan stage  
**Files:** `vyre-primitives/src/nfa`, `vyre-libs/src/matching` (nfa.rs, dfa/, substring/, hit_buffer.rs), `matchkit`, `simdsieve`, `dfajit`, `cudagrep`, `warpstate` GpuMatcher, `vyre-driver-wgpu` pipeline cache  
**Date:** 2026-04-24  
**Findings:** 25  

---

## Specific Questions  -  Executive Answers

| # | Question | Answer |
|---|----------|--------|
| 1 | Does `nfa::subgroup_nfa::nfa_step` use `subgroup_shuffle` or fall back to global atomics? | **Uses `subgroup_shuffle` exclusively.** No atomics in the primitive. |
| 2 | Is `nfa_scan` (vyre-libs) single-dispatch or multi-dispatch per byte? | **Single dispatch per input buffer.** One `@workgroup_size(32,1,1)` loop iterates over the whole input one byte at a time. |
| 3 | Is DFA compile budget enforced (max state count, max transition bytes)? | **Only transition-table byte budget is enforced (default 16 MiB).** No `max_states` cap. Budget check runs **after** the full trie + failure-link + dense table is built in RAM, so adversarial inputs can exhaust host memory before the check fires. |
| 4 | Does Aho-Corasick run one byte per thread or one byte per subgroup? | **Classic AC runs one byte per thread** (`[64,1,1]` workgroup, O(n²) serial replay per thread). **Cooperative DFA runs one byte per thread with subgroup shuffle forwarding** (`log2(subgroup_size)` correction rounds). Neither variant runs one byte per subgroup. |
| 5 | Does `hit_buffer` compaction use subgroup prefix-sum or atomic counter? | **Atomic counter.** `emit_hit` calls `atomic_add` on a single global `out_cursor`. `compact_hits` runs on a single invocation `[1,1,1]`  -  no prefix-sum, no subgroup ballot, no parallel compaction. |
| 6 | Any regex/literal mixing in `nfa_scan` that stalls whole lanes? | **No regex implemented yet** (literal-only). However, `nfa_scan` has inherent **32× lane divergence** in its scalar transition/epsilon loops: each lane is active for only 1/32 of loop iterations. |
| 7 | Is `gpumatch`'s compiled index regenerated per corpus or cached? | **Cached per `GpuMatcher` / `WgpuPipeline` instance, not per corpus.** `warpstate::GpuMatcher` stores compiled pipelines and pattern buffers in `GpuState`. `vyre-driver-wgpu` caches compiled `ComputePipeline` artifacts keyed by `(IR hash, adapter fingerprint)` in a `DashMap` + on-disk `DiskPipelineCache`. |

---

## Findings

### CRITICAL

**CRITICAL | `vyre-libs/src/matching/nfa.rs:336` | Undefined function `bit_in_word`  -  compilation failure**  
`build_transition_table` calls `bit_in_word(word_idx, bit)`, but no such function exists anywhere in the repository. The crate does not compile.
> **Fix:** Implement `bit_in_word` (e.g. `if word_idx == 0 { bit } else { 0 }` for the flat u32 layout) or remove the call and write the bit directly.

**CRITICAL | `vyre-libs/src/matching/nfa.rs:242-256` + `vyre-primitives/src/nfa/subgroup_nfa.rs:220-231` | Transition-table layout mismatch between primitive and composition**  
`nfa_step` (primitive) declares `transition_buf` as `num_states * 256 * LANES_PER_SUBGROUP` u32s (lane-major).  
`nfa_scan` (composition) declares `nfa_transition` as `num_states * 256` u32s (flat) and loads it with `src * 256 + byte`.  
The two modules are physically incompatible; composing them would read out-of-bounds or produce garbage.
> **Fix:** Unify on one layout. If the composition layer keeps the flat bitset, change the primitive's buffer declaration and indexing to match. Add an integration test that round-trips a multi-lane state through the primitive.

**CRITICAL | `vyre-libs/src/matching/dfa/dfa_compile.rs:88-101` | Budget check runs after full DFA construction  -  host-memory DoS vector**  
`dfa_compile_with_budget` calls `dfa_compile_inner(patterns)` first (building trie, BFS failure links, and dense transition table in `Vec`s), then checks `requested_bytes > budget_bytes`. An adversary can submit a pattern set that causes exponential DFA state explosion, exhausting host RAM before the budget gate ever fires.
> **Fix:** Enforce a `max_states` cap **during** trie construction (e.g. abort when `trie.len() > budget_states`). Wire the existing `DEFAULT_DFA_BUDGET_BYTES` into a state-cap derived from `budget_bytes / 1024`.

**CRITICAL | `vyre-libs/src/matching/hit_buffer.rs:74` | Global atomic counter serializes all hit emission across the GPU**  
`emit_hit_with_layout` allocates slots via `atomic_add(out_cursor, 0, 1)`. Every active lane in every workgroup that finds a hit contends on this single memory location. At high hit rates (e.g. common literals in large corpora) this becomes a wall-clock bottleneck.
> **Fix:** Use a two-level allocation: per-workgroup local counter + one atomic per workgroup to claim a global block, then write hits into the block without further atomics. Or implement subgroup ballot + prefix-sum for intra-subgroup compaction.

**CRITICAL | `vyre-libs/src/matching/nfa.rs:208-240` | Fixed-slot hit emission overwrites duplicate matches**  
Accept hits are written to `hit_buf[3*pattern_id .. 3*pattern_id+2]`. If the same pattern matches multiple times, each subsequent match overwrites the previous slot. The caller sees only the last (or first, depending on race timing) match.
> **Fix:** Use the atomic `emit_hit` primitive for NFA accept emission, or reserve variable-length output and append per-match triples with an atomic slot allocator.

### HIGH

**HIGH | `vyre-libs/src/matching/nfa.rs:107-205` | `nfa_scan` inlines scalar loops instead of using `subgroup_shuffle`**  
The module doc claims to compose `vyre_primitives::nfa::subgroup_nfa::nfa_step`, but the emitted IR contains scalar `loop_for` over `num_states` with lane-local `if_then` guards. Each lane is active for only `num_states/32` iterations; the other 31/32 iterations are divergent no-ops. At 1024 states this is ~32× wasted work per byte.
> **Fix:** Actually call `nfa_step` (or emit its shuffle-based body inline) so the 32 lanes cooperatively gather transitions across the subgroup.

**HIGH | `vyre-libs/src/matching/dfa/aho_corasick.rs:35-89` | Classic AC is O(n²) serial work per thread, not cooperative**  
Each invocation `i` replays the DFA from state 0 through byte `i` independently. Workgroup size is `[64,1,1]`. For a 1 MiB haystack, the last thread walks 1 MiB of transitions. Total serial work is ~O(n²/2).
> **Fix:** Default to `cooperative_dfa_scan` for contiguous inputs; use classic AC only for deliberately scattered-index workloads where prefix replay is unavoidable.

**HIGH | `vyre-libs/src/matching/hit_buffer.rs:165-174` | `compact_hits` is completely scalar (`[1,1,1]` workgroup)**  
A single invocation reads the atomic counter, computes `min(cursor, max_capacity, buffer_cap)`, and writes `live_len`. No parallel prefix-sum, no subgroup shuffle, no compaction kernel. On GPUs this is a host-round-trip for work a single CPU core could do faster.
> **Fix:** If compaction is needed (e.g. removing gaps from sparse hits), implement a parallel prefix-sum kernel. If the buffer is already dense, eliminate the separate `compact_hits` dispatch and fold the length clamp into the host readback path.

**HIGH | `vyre-driver-wgpu/src/pipeline.rs:374-382` | Pipeline-cache eviction is not LRU  -  drops arbitrary entry**  
When `pipeline_cache.len() > MAX_PIPELINE_CACHE_ENTRIES`, the code calls `pipeline_cache.iter().next()` and removes that key. `DashMap`'s iteration order is shard-stable but not recency-ordered; this evicts a random pipeline, not the coldest one.
> **Fix:** Replace the `DashMap` with the `IntrusiveLru` already present in `runtime/cache/lru.rs`, or maintain a touch-timestamp alongside each cached artifact and evict the oldest.

**HIGH | `vyre-libs/src/matching/nfa.rs:230` | Hit `start` position is computed as `input_len - pattern_len`  -  always wrong except for EOF matches**  
The comment on line 202 admits: "`start` not tracked per-state in this simple build". The emitted code computes start as `plan.input_len.saturating_sub(pattern_len)`, which places every match at the end of the buffer regardless of where it actually occurred.
> **Fix:** Track per-state start offsets (e.g. a second `u32` per state recording the match origin) or compute start from the cursor position inside the accept-check loop.

**HIGH | `vyre-primitives/src/nfa/subgroup_nfa.rs:117-158` | `nfa_step` unrolls 1024 conditional branches per byte step**  
The transition gather unrolls `LANES_PER_SUBGROUP (32) × 32 bits = 1024` `if_then` nodes. For `num_states = 1024`, WGSL lowering emits 1024 branches. This blows up shader code size and icache pressure.
> **Fix:** For dense state bitsets, use subgroup ballot operations instead of per-bit conditionals. Or tile the unroll so the compiler can vectorize the inner bit-test loop.

**HIGH | `vyre-libs/src/matching/cooperative_dfa.rs:58-137` | `cooperative_dfa_scan` does not validate hardware subgroup size**  
The kernel assumes `subgroup_size` lanes are active and does `log2(subgroup_size)` shuffle rounds. If the GPU's actual subgroup width is smaller (e.g. 16 on some mobile GPUs), state forwarding is incomplete and matches are silently lost.
> **Fix:** Query the adapter's `subgroup_size` at pipeline creation time and either reject incompatible adapters or emit a specialized kernel for the actual subgroup width.

### MEDIUM

**MEDIUM | `vyre-libs/src/matching/nfa.rs:162-199` | Epsilon closure runs `num_states²` iterations even for literal patterns**  
For literal-only patterns, `build_epsilon_table` returns an all-zero vector. The epsilon-closure nested loops still execute `num_states × num_states` iterations per byte, OR-ing zeros. At 1024 states this is ~1M no-op iterations per input byte.
> **Fix:** Skip the epsilon-closure loop when the epsilon table is all-zero (detect at compile time), or lower the iteration bound to the actual epsilon diameter.

**MEDIUM | `vyre-primitives/src/nfa/subgroup_nfa.rs:209` | Epsilon iteration cap is `num_states.min(32)`, but `MAX_EPSILON_ITERS` claims 1024**  
The docstring says "cap guards against pathological inputs" and `MAX_EPSILON_ITERS = 1024`. The emitted code uses `num_states.min(32).max(1)`. For regex with epsilon chains longer than 32 states, closure is truncated and matches are silently lost.
> **Fix:** Expose the epsilon-iteration cap as a compile-time parameter (e.g. `NfaPlan::epsilon_iters`) and default to `num_states`, not 32.

**MEDIUM | `vyre-libs/src/matching/hit_buffer.rs:122` | `emit_hit` workgroup size `[64,1,1]` with `DEFAULT_LANES=4`**  
Only 4 lanes have input data; the other 60 lanes immediately fail the `lane < buf_len(rule_id)` guard and do nothing. 93.75 % of launched invocations are idle overhead.
> **Fix:** Use `[DEFAULT_LANES, 1, 1]` or pad the input to a multiple of 32 and use subgroup-wide operations.

**MEDIUM | `cudagrep/src/hardware/cache.rs:8` | FD fast-path array hard-capped at 1024 entries**  
`FD_FAST_PATH_SIZE = 1024`. On high-throughput servers with many open files, file descriptors routinely exceed 1024, forcing every lookup into the fallback `HashMap`.
> **Fix:** Make `FD_FAST_PATH_SIZE` configurable via `CuFileHardware` constructor, or replace the array with a small LRU of the most-recently-used fds.

**MEDIUM | `cudagrep/src/hardware/cache.rs:66-75` | `get_or_register_for_bytes` calls `fstat` on every cache miss**  
To validate inode/device identity, the code `fstat`s the fd. This adds a syscall to the hot path every time a new fd is seen.
> **Fix:** Cache the `(inode, device)` tuple in a separate `HashMap<RawFd, (u64,u64)>` so repeated lookups for the same fd avoid `fstat`.

**MEDIUM | `warpstate/src/gpu/builder.rs:185-192` | `GpuMatcher` builds both specialized and buffer-based regex pipelines, then discards one**  
`build_specialized_regex_gpu` is called first; if it succeeds, `build_regex_gpu` is skipped. But the construction logic still evaluates both paths' preconditions. The specialized shader embeds the DFA as WGSL constants  -  if it succeeds, the buffer-based path is dead code.
> **Fix:** Only call `build_regex_gpu` when `specialized_regex.is_none()`, avoiding buffer creation and layout work for the discarded path.

**MEDIUM | `simdsieve/src/sieve/compiler.rs:28-34` | `MAX_PATTERNS = 16` hard limit with no automatic sharding**  
The SIMD prefilter rejects pattern sets larger than 16. There is no built-in chunking or multi-pass orchestration; the caller must manually shard.
> **Fix:** Implement an internal `MultiSieve` path (already exists in `multi.rs`) as the default backend for >16 patterns so the public API is not capped.

**MEDIUM | `dfajit/src/dfa.rs:339-373` | Trie construction has no runtime memory budget**  
`from_patterns` builds a trie and then failure links, but only the JIT code-size path (`compile_with_output_links`) has a state limit. The trie + dense table can OOM the host on adversarial pattern sets before JIT eligibility is checked.
> **Fix:** Add a `max_trie_states` parameter to `from_patterns` and abort early during trie insertion.

**MEDIUM | `vyre-driver-wgpu/src/runtime/cache/tiered_cache.rs:103-109` | `eviction_candidate_per_tier` fallback is non-deterministic**  
If the tier LRU tail is stale, the fallback returns `entries.keys().next().copied()`. `FxHashMap` key order is hash-dependent and non-deterministic across runs, making eviction behavior unpredictable.
> **Fix:** Fall back to the tier's hottest entry (or a round-robin cursor) instead of an arbitrary hash-map key.

**MEDIUM | `vyre-libs/src/matching/nfa.rs:66-72` | `nfa_scan` panics on oversized patterns instead of returning `Result`**  
Both `nfa_scan` and `nfa_step` use `assert!` for bounds checks. At internet scale, a single malformed rule upload can crash the scan worker.
> **Fix:** Change the return type to `Result<Program, NfaCompileError>` and propagate the error to the caller for graceful degradation or sharding.

**MEDIUM | `vyre-libs/src/matching/dfa/dfa_compile.rs:170-189` | Dense transition table built with O(states × 256 × failure_depth) loops**  
Phase 3 walks failure links per state per byte. For patterns with long failure chains (e.g. `a`, `aa`, `aaa`, …), this is quadratic in state count during compilation.
> **Fix:** Build the dense table during the BFS phase (like standard AC construction) so each transition is resolved in O(1) amortized time.

**MEDIUM | `vyre-libs/src/matching/hit_buffer.rs:97-103` | Overflow tracking uses a second global atomic**  
When the hit buffer is full, `emit_hit` increments `HIT_BUFFER_OVERFLOW_COUNT` via a second `atomic_add`. This doubles the atomic contention on overflow scenarios.
> **Fix:** Use a single 64-bit atomic where the high 32 bits store the overflow count and the low 32 bits store the slot cursor, eliminating the second contention point.

---

## Competitor Comparison

| Capability | vyre / warpstate | dfajit | regex-automata (baseline) |
|-----------|------------------|--------|---------------------------|
| NFA subgroup shuffle | **Present** (primitive) | N/A | N/A |
| NFA composition uses shuffle | **No** (scalar fallback) | N/A | N/A |
| DFA compile budget | Byte budget only, post-construction | States + bytes + code size, pre-construction | Dense DFA builder has `state_limit` |
| AC GPU scan | O(n²) per thread or shuffle-corrected | N/A (CPU JIT) | N/A |
| Hit compaction | Global atomic + scalar clamp | Direct slice write | N/A |
| Pipeline cache | Memory + disk cache | N/A | N/A |
| Regex/literal mixing | Not implemented | DFA + AC via `regex-automata` | Full hybrid |

**Key gap:** `regex-automata`'s dense DFA builder enforces `state_limit` **during** construction (via `dfa::dense::Builder::state_limit`), preventing the host-memory DoS vector present in `vyre-libs::dfa_compile`. We should adopt the same early-abort strategy.

---

## Recommendations (Priority Order)

1. **Fix `bit_in_word` and the transition-table layout mismatch**  -  these block correct compilation and execution of the NFA scan path.
2. **Move DFA budget enforcement into trie construction**  -  prevent adversarial OOM before dense-table allocation.
3. **Replace `nfa_scan` scalar loops with actual `subgroup_shuffle` emission**  -  this is the primary performance win for the NFA path.
4. **Implement hierarchical hit allocation** (workgroup-local counters + block atomics)  -  removes the global atomic bottleneck.
5. **Add `max_states` to `NfaPlan` and `DfaCompileError`**  -  replace panics with structured errors so the orchestrator can shard.
6. **Audit `cooperative_dfa_scan` on sub-32-lane hardware**  -  add a runtime subgroup-size query and fallback.
7. **Unify pipeline cache eviction on `IntrusiveLru`**  -  deterministic, true-LRU replacement instead of random eviction.
