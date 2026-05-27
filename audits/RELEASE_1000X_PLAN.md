# Supersession notice

This document is historical evidence, not the active optimization control
plane. For current optimization ownership, layer placement, op/backend matrix,
and benchmark targets, start at `docs/optimization/START_HERE.md`.

# The 1000× plan  -  every computation surgec does, on GPU

Surgec's hot path is not just pattern matching. At internet scale every
phase below must hit the release 1000× bar. This file is the live
plan; agents add findings as audits return; new tracks open as gaps appear.

## Phase inventory (everything surgec does)

1. **File enumeration + content collection**  -  directory walk, stat,
   read, gitignore filtering, encoding detect.
2. **Decode**  -  base64 / hex / inflate / lz4 on suspicious blobs.
3. **Lexical scan**  -  literal match, regex, Aho-Corasick, subgroup NFA.
4. **Parsing**  -  AST build per language (C/C++/Rust/Go/Python/JS).
5. **AST walk / pattern match**  -  tree-sitter query engine, predicate
   evaluation on IR.
6. **Dataflow**  -  IFDS / reaching-defs / liveness / points-to / escape
   / value-range / loop-sum / proc-summary.
7. **Graph queries**  -  CFG, call graph, exploded supergraph,
   reachability, SCC, shortest path, flows-to.
8. **Rule orchestration**  -  rule-set fusion, clause dispatch,
   exemption filter, confidence scoring, provenance chain.
9. **Finding emit + dedupe + sort**  -  SARIF build, confidence sort,
   suppression, delta-replay.
10. **Differential / watch mode**  -  only scan changed closure;
    incremental re-rebuild, hot-reload.

Every one has a GPU innovation track. Below are the innovation
obligations per phase  -  each must hit its factor or the line is a
regression.

## Innovation obligations per phase

### Phase 1  -  enumeration (target 50×)
- io_uring-backed walkdir with GPU-side stat batching.
- DMA from NVMe to GPU VRAM via G105 zero-copy.
- Gitignore matcher runs as a vyre Program, one dispatch per 10k files.

### Phase 2  -  decode (target 30×)
- G5 decode-scan fused  -  no HBM round-trip for decoded bytes.
- base64/hex decoders emit bytes straight into workgroup memory.
- inflate/lz4 use a cooperative-thread-block literal-copy kernel.

### Phase 3  -  scan (target 100×)
- G1 subgroup-cooperative NFA  -  1024 states in one subgroup.
- G2 megakernel rule fusion + cross-rule CSE (`input`, `output_slots`
  shared). One dispatch per corpus tile.
- G4 adaptive CSR / dense bitmatrix switch per tile.
- gpumatch compiled-index Aho-Corasick, one byte per thread.

### Phase 4  -  parse (target 80×)
- Tree-sitter state tables uploaded as u32 tables; kernel walks with
  persistent threads.
- LR(1) parse table as a GPU-resident CSR; one lookup = one load.
- Preprocessor expansion as a fused decoder→parser stage.

### Phase 5  -  AST walk (target 50×)
- G109 GPU-resident AST buffer: one u32 per AST node, predecessor
  encoded in high bits.
- Tree-sitter query lowers to a vyre Program via `surgec compile`.
- Predicate evaluation is bitset-flattened (G59 `bitset_fixpoint`).

### Phase 6  -  dataflow (target 200×)
- G3 exploded supergraph → `csr_forward_traverse` BFS to convergence
  (already wired).
- G244 SSA construction, G245 reaching-defs, G246 Andersen points-to,
  G248 indirect-dispatch call graph, G249 backward slicer, G250
  value-range lattice, G251 escape analysis, G252 procedure summaries,
  G253 loop-summarisation acceleration  -  every one a vyre Program.

### Phase 7  -  graph queries (target 500×)
- `reachable`, `toposort`, `scc_decompose`, `path_reconstruct`, `motif`,
  `flows_to`  -  every one already a primitive; ensure G4 adaptive
  traversal + G110 warm-start fixpoint cache hit every dispatch.

### Phase 8  -  orchestration (target 20×)
- G9 CHD perfect hash for rule→slot and label family lookup.
- G8 content-hash pipeline cache  -  single-digit ms cold start after
  first run.
- G6 speculative commit/rollback around confirmer rules.
- G7 persistent engine: one dispatch amortises 10 000 tiny-file scans.

### Phase 9  -  finding emit (target 10×)
- `compact_hits` on GPU with subgroup prefix sum.
- SARIF encoding as a vyre Program (tokens → bytes).
- Confidence score as a bitset dot-product kernel.

### Phase 10  -  differential / watch (target 10× on cold, 1000× on warm)
- G10 transitive-closure differential scan (landed).
- G110 incremental bitset fixpoint warm-start.
- G8 persistent pipeline cache survives process restart.
- Hot-reload via watcher sends one WorkItem per changed file.

## Total compounded speedup

50 × 30 × 100 × 80 × 50 × 200 × 500 × 20 × 10 × 10 = absurd on paper.
Orthogonal axes only multiply inside the dispatch envelope where the
pipeline is hot. The achievable *end-to-end* 1000× is:

- Cold-start (one-file scan): 5× (G8 cache) × 20× (G1 prefilter +
  G2 fusion) = **100×** minimum over CodeQL on the same corpus.
- Warm-path (10k-file repo, typical CI):
  5× (persistent engine) × 20× (NFA prefilter)
  × 4× (rule fusion) × 3× (decode-scan fusion)
  × 2× (speculative) × 5× (content-hash cache hit) = **1200×**.

## Audit fan-out (live  -  agents write findings here as they come in)

Ten read-only audits covering each phase. Each produces
`SEVERITY | file:line | defect | fix`. Findings land under
`# Findings from <audit_name>` below.

| audit | scope | deliverable |
|------|-------|-------------|
| PHASE1_ENUM | walkkit + codewalk + io_uring integration | audits/PHASE1_ENUM.md |
| PHASE2_DECODE | decode/base64 decode/hex decode/inflate/streaming | audits/PHASE2_DECODE.md |
| PHASE3_SCAN | nfa/dfa/aho-corasick/gpumatch | audits/PHASE3_SCAN.md |
| PHASE4_PARSE | tree-sitter lowering + LR tables | audits/PHASE4_PARSE.md |
| PHASE5_ASTWALK | predicate eval + GPU AST buffer | audits/PHASE5_ASTWALK.md |
| PHASE6_DATAFLOW | ifds + reach + points-to + all dataflow | audits/PHASE6_DATAFLOW.md |
| PHASE7_GRAPH | reachable/toposort/scc/path/motif/flows_to | audits/PHASE7_GRAPH.md |
| PHASE8_ORCH | rule fusion + exemption + speculation + cache | audits/PHASE8_ORCH.md |
| PHASE9_EMIT | hit_buffer compact + SARIF + confidence | audits/PHASE9_EMIT.md |
| PHASE10_DIFF | diff_scan + watcher + warm-start | audits/PHASE10_DIFF.md |

## Rules (enforced on every commit)

- LAW 1: No stubs. A `Program::empty()` in non-test code is a bug.
- LAW 7: One module, one responsibility. Files ≥500 LOC split.
- LAW 8: Every finding is critical at internet scale. Fix all.
- LAW 9: No evasion. Remove TODO → IMPLEMENT. No "pending real kernel".

## Findings  -  live ingest

### Wave 1 audit findings ingested (2026-04-24)

Each line = one actionable item from the 10 PHASE audits. Status: [ ] open, [x] done, [W] work-in-progress.

**PHASE1_ENUM** (12 critical/high found):
- [ ] walkkit uses sync std::fs::read_dir  -  swap for io_uring batched stat.
- [ ] gitignore filter is CPU fnmatch loop; upload patterns + globset as vyre Program.
- [ ] NVMe reads bounce through host DRAM  -  use `O_DIRECT | SPLICE` pipe into GPU VRAM.
- [ ] encoding detect does a per-byte UTF-8 validate loop in Rust  -  run as vyre::matching kernel.

**PHASE2_DECODE**:
- [ ] CRITICAL inflate traps on BTYPE=1/2/3. Implement fixed + dynamic Huffman decode or rename op to inflate_stored_block_only to stop lying.
- [ ] Decode→scan chain not warmgroup-promoted except via the G5 pass; ensure every shipped pipeline runs the pass.

**PHASE3_SCAN**:
- [x] nfa_scan hit-buffer overwrites on repeat match  -  atomic_add slot counter.
- [x] Epsilon loop runs even for literal-only patterns  -  skip when table is all-zero.
- [ ] O(n²) DFA replay: every byte-position lane replays from state 0. Use Hillis-Steele parallel-scan or Aho-Corasick multi-pattern automaton.
- [ ] Subgroup size hard-coded 32  -  detect device size at runtime, fall back to vendor-specific kernel.
- [ ] `start` offset in hit-buffer is `input_len − pattern_len` (wrong for repeated matches)  -  carry start through state table.

**PHASE4_PARSE**:
- [ ] Tree-sitter state tables live on CPU; upload per-language LR(1) tables as `pg_*` ProgramGraph buffers.
- [ ] No single-kernel AST build  -  every language re-dispatches per node.
- [ ] Preprocessor / macro expansion is a separate CPU stage; fuse into parser Program.

**PHASE5_ASTWALK**:
- [ ] Predicate evaluation is partly a recursive CPU tree walk; lower every predicate to a bitset_fixpoint Program.
- [ ] VAST (AST-as-vyre-buffer) vs ProgramGraph duality  -  unify on ProgramGraphShape.
- [ ] Predicate registry uses linear scan; G9 CHD perfect hash.
- [ ] flows_to + dominates + reaches: some are per-query BFS, not the shared bitset_fixpoint.

**PHASE6_DATAFLOW** (matched 7 panicstubs):
- [x] ssa / points_to / callgraph / slice / range / escape / summary / loop_sum  -  REAL vyre Programs landed.
- [ ] Soundness marker unenforced  -  rules can compose `Sound` primitives under zero-FP contracts and pass.

**PHASE7_GRAPH**:
- [ ] path_reconstruct uses Floyd-Warshall (O(V³)); single-source Dijkstra via bitset_fixpoint.
- [ ] motif uses sequential DFS; switch to BFS with depth cap.
- [ ] flows_to sometimes per-query BFS, sometimes fixpoint  -  unify.
- [ ] adaptive_traverse density probe not always called before choosing CSR vs dense  -  gate every caller through `should_use_dense`.

**PHASE8_ORCH**:
- [ ] Scan loop runs rules one at a time; mandatorily apply G2 fuse_cse before every dispatch batch.
- [ ] Exemption filter is CPU hot-loop; lower to a vyre bitset mask kernel.
- [ ] Confidence sort is CPU quicksort; radix sort on GPU over u32 score.
- [ ] AdaptiveSpeculator never called from production  -  wire it into dispatch.rs before every fusable batch.
- [ ] Pipeline cache eviction uses random next() instead of LRU.

**PHASE9_EMIT** (6 criticals):
- [ ] CRITICAL compact_hits is [1,1,1] scalar dispatch  -  replace with CPU clamp or real subgroup-prefix-sum compaction.
- [ ] Dedupe is CPU BTreeSet in tests, absent in prod  -  wire G9 CHD hashset.
- [ ] SARIF encode is pure CPU serde  -  emit as byte-level vyre Program.
- [ ] Confidence score is CPU f32 multiply  -  bitset dot product on GPU.
- [ ] Emit path readbacks to host  -  zero-copy GPU → stdout via mapped buffer.
- [ ] Suppression filter CPU-side after readback  -  lower to pre-readback vyre filter.

**VYRE_MEM_LAYOUT**:
- [x] CRIT-2 nfa_scan transition table flat [num_states × 256]  -  now lane-major [num_states × 256 × LANES] matching subgroup_nfa::nfa_step.

**VYRE_NAGA_LOWER**:
- [x] CRIT-1 dispatch geometry ignored workgroup_size[1,2]  -  fixed (commit 346d50f4e2).
- [x] CRIT-2 in-memory pipeline cache checked after disk I/O  -  early_pipeline_cache_key now precedes lowering.
- [x] HIGH-1 unreachable!() in subgroup BinOp fallback  -  replaced with LoweringError::unsupported_op.
- [x] HIGH-2 `size_bytes() as u32` truncates  -  try_into with LoweringError::invalid.

**VYRE_IR_HOTSPOTS**:
- [x] CRIT autotune clones Program on no-op path  -  PassResult::unchanged.
- [x] CRIT spec_driven clones Program on no-op path  -  PassResult::unchanged.
- [x] CRIT canonicalize `program.entry().to_vec()` unconditional clone  -  into_entry_vec + scaffold.
- [x] CRIT canonicalize + region_inline `(*body).clone()` per Region  -  Arc::try_unwrap fast path.
- [x] CRIT to_wire per-node Vec<u8> allocation  -  scratch buffer reused across N nodes.
- [x] CRIT to_wire shape/hints per-buffer Vec alloc  -  two scratch buffers reused across B buffers.
- [x] CRIT from_wire `.to_vec()` per node payload  -  sub-slice view via Reader::take &'a [u8].
- [x] CRIT from_wire shape/hints `.to_vec()`  -  sub-slice views.
- [x] HIGH rewrite_program always allocates  -  Cow fast-path + (Program, bool) return.
- [x] HIGH graph_view.rs clones every top-level node  -  from_program_owned moves via into_entry_vec.
- [x] HIGH Program::clone deep-copies output_buffer_index + stats  -  Arc-wrapped for O(1) clone.
- [x] HIGH fusion replacement_exprs rebuilds hashmap per node  -  substitute_expr reads PendingExpr directly.
- [x] LOW canonicalize expr_sort_key hashes names per-compare  -  Ident::cached_hash().
- [ ] CRIT fuse_programs `programs[0].clone()`  -  shallow Arc clone; documented non-issue.
- [ ] CRIT cse rewrite_args (per-call impl_csectx): same Cow pattern; deferred until CseCtx refactor.

**VYRE_OPTIMIZER**:
- [x] HIGH-01 scheduler.rs:286  -  invalidated SKIPped passes dropped from next_dirty; removed available.contains guard.
- [x] HIGH-02 strength_reduce  -  extended to Div/Mod by unsigned power-of-two (-> Shr / BitAnd).
- [x] HIGH-03 execution_plan/fusion  -  OverDispatch error when axis-wise max exceeds 4× largest arm.

**VYRE_NAGA_LOWER**:
- [x] CRIT-1 dispatch geometry ignored workgroup_size[1,2]  -  multi-dim product.
- [x] CRIT-2 in-memory pipeline cache checked after disk I/O  -  early_pipeline_cache_key.
- [x] HIGH-1 unreachable!() in subgroup BinOp fallback  -  LoweringError::unsupported_op.
- [x] HIGH-2 `size_bytes() as u32` truncation  -  try_into + LoweringError::invalid.
- [x] HIGH double Naga validation on WGSL path  -  emit_module drops validator, writer validates once.
- [x] MEDIUM println! leak on validation failure  -  tracing::trace.
- [x] MEDIUM rotate_width_bits U64/I64 split-brain  -  reject at width site.
- [x] MEDIUM fold_binary_literal negative i32 shift silently reinterprets  -  now refuses to fold.
- [x] MEDIUM spirv_backend Capabilities::empty vs wgsl all  -  unified to all().
- [x] MEDIUM normalized_compile_wire full Program clone + serialize  -  normalized_cache_digest (thread-local scratch).

**VYRE_PRIMITIVES_GAPS**:
- [x] LOW quest_select_top_k sentinel f32::MIN collided with valid negatives  -  now f32::NEG_INFINITY.

**G1  -  subgroup-cooperative NFA**:
- [x] nfa_scan lane-major transition table + epsilon table match primitive contract.
- [x] Per-cursor accept emission with atomic_add slot claim (fixes post-loop-only miss).
- [x] mega_scan integrator wired as minimal real RulePipeline composer; extensible as G2-G10 land.

**PHASE10_DIFF**:
- [ ] diff_scan transitive closure is CPU BFS; run `csr_forward_traverse` on include graph.
- [ ] Include-graph not cached across invocations  -  blake3 key over source roots.
- [ ] Watcher doesn't feed G7 PersistentEngine  -  hook `notify` crate to `engine.enqueue(WorkItem)`.
- [ ] bitset_fixpoint_warm_start unused in prod.
- [ ] on_disk pipeline cache miss-every-run: verify `compute_cache_key_for` is called at pipeline build (currently likely isn't).
- [ ] diff_replay.rs uses CPU comparisons  -  vyre Program that XORs two finding bitsets.

---

### Session sweep 2026-04-24  -  additional closures

**VYRE_OPTIMIZER**:
- [x] HIGH-01 scheduler.rs:286  -  invalidated SKIPped passes dropped from next_dirty; removed `available.contains` guard.
- [x] HIGH-02 strength_reduce extended to Div/Mod by unsigned power-of-two (-> Shr/BitAnd).
- [x] HIGH-03 execution_plan/fusion  -  FusionError::OverDispatch when fused geometry > 4× largest arm.
- [x] MED-05 normalize_atomics already uses PassResult fast-path (verified stale).
- [x] MED-02 const_fold Select clone fires only on fold-success (verified stale).

**VYRE_IR_HOTSPOTS (sweep 3)**:
- [x] HIGH Program::clone deep-copies output_buffer_index + stats  -  Arc-wrapped.
- [x] HIGH rewrite_program always allocates  -  Cow + (Program, bool) fast path.
- [x] HIGH structural_eq per-call  -  Arc::ptr_eq short-circuit + in-place equality fast path before key sort.
- [x] HIGH fusion replacement_exprs rebuilt hashmap per node  -  substitute_expr reads PendingExpr directly.
- [x] HIGH graph_view clones every top-level node  -  from_program_owned via into_entry_vec.
- [x] HIGH readback busy-wait  -  recv_timeout + POLL_TICK instead of yield_now spin loop.
- [x] CRIT to_wire per-node Vec<u8>  -  scratch buffer reused across N nodes.
- [x] CRIT to_wire shape/hints per-buffer Vec  -  scratch buffers reused across B buffers.
- [x] CRIT from_wire node payload `.to_vec()`  -  sub-slice view via Reader::take<'a>.
- [x] CRIT from_wire shape/hints `.to_vec()`  -  sub-slice views.
- [x] LOW canonicalize cached_hash  -  Ident::cached_hash replaces per-compare hash_str.
- [x] MEDIUM combined_entry Vec growth in fusion  -  preallocate capacity.

**VYRE_NAGA_LOWER (sweep 3)**:
- [x] MEDIUM normalized_compile_wire clone+serialize  -  normalized_cache_digest thread-local scratch.
- [x] MEDIUM rotate U64/I64 split-brain  -  reject at width site.
- [x] MEDIUM fold_binary_literal negative shift  -  refuse to fold.
- [x] MEDIUM spirv capability split-brain  -  unified to Capabilities::all().
- [x] MEDIUM println! leak on validation failure  -  tracing::trace.
- [x] HIGH double validation on WGSL path  -  removed emit_module validator.
- [x] MEDIUM wire decode leb_string  -  str::from_utf8 replaces bytes.to_vec()+from_utf8.

**VYRE_BACKEND_WGPU**:
- [x] CRIT MegakernelDispatch trait "dead code"  -  `WgpuMegakernelDispatcher` already wraps it (audit stale; verified).
- [x] HIGH readback busy-wait loop → recv_timeout with POLL_TICK.

**PHASE2_DECODE**:
- [x] HIGH fuse_decode_scan `assert!` panic on zero handoff  -  DecodeScanFuseError::ZeroHandoff.
- [x] MEDIUM base64_decode silent input_len % 4 drop  -  assert + actionable error.
- [x] streaming promote_to_workgroup entry deep-clone  -  Arc::try_unwrap.

**PHASE5_ASTWALK**:
- [x] HIGH flows_to + reaches use restricted edge masks (FLOWS_TO_MASK / CONTROL+CALL_ARG).
- [x] HIGH sanitized_by fuses sanitizer subtraction into emitted Program.
- [x] MEDIUM emit_zone_membership fold  -  balanced-tree reduction replaces left-fold.
- [x] MEDIUM perfect_hash `candidate_slots.contains` O(bucket)  -  Vec<bool> occupancy scratchpad.

**VYRE_PRIMITIVES_GAPS**:
- [x] LOW quest_select_top_k sentinel collision  -  f32::NEG_INFINITY.

**VYRE_MEM_LAYOUT**:
- [x] CRIT-2 nfa_scan layout mismatch  -  lane-major table fused into primitive contract.
