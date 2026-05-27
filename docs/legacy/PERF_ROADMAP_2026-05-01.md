# vyre performance roadmap (2026-05-01)  -  SUPERSEDED 2026-05-02

> **Evidence-only.** The canonical executable roadmap is now
> [`docs/optimization/ROADMAP.md`](docs/optimization/ROADMAP.md). The IDs below
> (M0..M5, A1..A36, B1..B15, C1..C6, D1..D9, E1..E5, F..L) were imported into
> the canonical roadmap with the same labels so existing references still
> resolve. Read [`docs/optimization/START_HERE.md`](docs/optimization/START_HERE.md)
> before opening this file. Lane assignment for any item below lives in
> [`docs/optimization/OWNERSHIP.toml`](docs/optimization/OWNERSHIP.toml); patch
> proof requirements are in [`docs/optimization/AGENT_CONTRACT.md`](docs/optimization/AGENT_CONTRACT.md).

Comprehensive list of optimization opportunities. Grouped by layer. Each item has an effect estimate (where defensible), a likely LOC bucket, and a dependency hint. **Numbers without measurement are guesses**  -  flag them as `[guess]` in the estimate column.

The "Effect" column means **estimated end-to-end speedup on warm-batch-per-100-files** unless otherwise noted. "Cold" means it only matters on first run. "Memory-bound" means it only matters on memory-bound ops. None of these are verified yet.

Order rules:
1. Measure before committing. Build the flame-graph first (item M0).
2. Items marked **[blocker]** must land before items marked **[depends-on-X]**.
3. **Bigger structural changes (egraph, hash-cons, SoA)** unlock the smaller ones  -  but they're high-risk. Land smaller wins in parallel.

---

## M. Measurement (do this first, always)

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| M0 | **Per-stage flame-graph** for `vyre-bench` warm-batch corpus: optimizer % / lowering % / emit % / dispatch % / kernel %. | meta | ~200 | **[blocker]** for everything else. We are guessing without it. |
| M1 | **Per-rule telemetry** in the optimizer: how many times each pass/rule fires, how much cost it removes. | meta | ~150 | Lets us cull dead rules; pays off once egraph rule-DB lands. |
| M2 | **Per-op kernel-time table**: μs/op × occupancy × bytes-moved. Identify the top 10 hot ops. | meta | ~100 | Frames where to spend time. |
| M3 | **Cold vs warm separation**: time to first dispatch vs steady-state. | meta | ~50 | Cold-start fixes are different work from steady-state fixes. |
| M4 | **Memory-bandwidth utilization probe** (achieved GB/s vs peak). | meta | ~80 | Tells us memory-bound vs compute-bound per op. |
| M5 | **Tier B device signature TOML** in `vyre/devices/*.toml`. One file per arch (sm_86, sm_89, sm_120/Blackwell, gfx1100, RDNA4 once specs are public). Schema: `max_sm`, `warp_size`, `regs_per_thread_max`, `shared_mem_per_sm_kb`, `l1_kb`, `l2_kb`, `mem_bw_gbps`, `tensor_core_supported`, `tensor_core_dtypes`, `ideal_unroll_depth`, `ideal_vector_pack_bits`, `ideal_workgroup_tile`, `bank_count`, `bank_width_bytes`. Consumed by vyre-opt for unroll, vector-pack, tile, and bank-padding decisions. Aligns with global Tier-B rule: community drops in a `.toml` for an unreleased arch and gets correct tuning without recompilation. ~250 LOC for the loader/struct, plus one TOML per device shipped. **[blocker-for-A6 cost vector]**  -  egraph extraction needs the device profile to score variants. | enables hardware-aware tuning across all backends without Rust changes | ~250 + N×TOML | Tier B per CLAUDE.md global rule. CLI never accepts these as flags; community-extensible by file drop. |

---

## A. Optimizer-layer (IR rewrites, substrate-agnostic)

Current `optimize()` runs canonicalize → region_inline → PassScheduler-fixpoint → CSE → DCE → canonicalize → ConstFold (phase 4). Each pass is a tree walk. Phase 4 only exists because of phase ordering  -  egraph saturation eliminates that bug class by definition.

### A.1 Structural rewrites (high LOC, big payoff)

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| A1 | **Hash-consed Expr**: intern every Expr in a slab, refer by 32-bit id. `Expr::clone()` becomes `Copy`. CSE becomes free by construction. | 2-5x [guess] on optimizer time | ~1500 | Breaks the world. Foundation for A2, B1, C1. |
| A2 | **SoA columnar Program**: opcode column + operand-id columns instead of `Vec<Node>` of enum. Cache-friendly streaming access. Rewrites become index updates not vector copies. | 1.5-3x [guess] on optimizer + lowering walks | ~3000 | What MLIR went to. **[depends-on-A1]**. |
| A3 | **Strip Region nodes before optimize, restore as side-table for diagnostics.** Region is a tracing marker; optimizer pays to walk past every one. | 5-15% [guess] on optimizer time | ~200 | Cheap, independent. |
| A4 | **Tags as per-Program bitsets** (one bit per (node, tag)) instead of two u64 words per node. O(1) intersect/union. | small but measurable | ~400 | **[depends-on-A2]**. |
| A5 | **Validator skip-cache**: `validated: bool` Cell on Program, cleared only on raw-construct paths. | 2-8% [guess] on hot-path | ~50 | Free win. |

### A.2 The egraph / egglog rewrite

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| A6 | **Egglog-shaped saturation engine** with rule database as TOML files. Subsumes ConstFold, StrengthReduce, CSE, DCE, GVN, LICM, normalize-atomics, all future micro-opts. Whole-megakernel domain (not per-Program). | structural; depends on rules | ~5000 + rule DB | The long-tail answer. **[depends-on-A1, A2]**. See section R for why. |
| A7 | **Cost-aware extraction** with backend-tunable cost vector (op-cost × frequency × pressure). Different extraction for cold vs hot path. | structural | ~800 | Part of A6 deliverable. |
| A8 | **Saturation budget per rule family** (not global): some rules saturate in 3 waves, others need 30. | minor but tunable | ~200 | Part of A6. |
| A9 | **Rule-applicability predicates** as small expressions over e-class metadata (`bool_typed`, `gpu_only`, `if_workgroup_size_le_64`, `target_has_tensor_cores`, etc.). Enables substrate-aware and shape-aware rules without polluting the engine. | structural | ~400 | Part of A6. |
| A10 | **GPU-resident e-graph** (CSR-backed e-class storage on device, parallel union-find with path-compression rounds, parallel rule-firing waves). | only if A6 is the bottleneck | ~3000 | Phase 2 of A6. Don't build until measurement says we need it. |

### A.3 Wire existing analyses into the optimizer

These are the highest-payoff items because the analysis work is already done in the dataflow consumer. The optimizer just doesn't read it.

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| A11 | **Reaching-defs → ConstFold across control flow**. Enables `x = 5; if (cond) y = 7; z = x + 1` folding. | medium-large on real code | ~500 | dataflow consumer::reaching exists. |
| A12 | **Points-to → memory-side optimization**. `*p = 5; *q = 7; load *p` foldable when p, q proven distinct. | medium-large on memory-heavy ops | ~700 | dataflow consumer::points_to exists. |
| A13 | **Escape analysis → buffer-storage reuse across megakernel arms**. Currently each arm gets fresh scratch. | reduces allocator pressure, helps cache | ~600 | dataflow consumer::escape exists. |
| A14 | **Live-range + register-pressure model → rematerialization**. Recompute cheap pure values rather than spill. | helps on large megakernels | ~800 | dataflow consumer::live exists. |
| A15 | **Buffer aliasing → load elision**. Aliases-dataflow can prove distinct buffers; redundant loads vanish. | medium on multi-buffer ops | ~400 | aliases_dataflow exists post-fuse_programs fix. |
| A16 | **Range analysis → cast/branch elision**. If value range proven `[0, 255]`, cast `u32 → u8` is free, bounds checks vanish. | medium across the board | ~600 | dataflow consumer::range exists. |

### A.4 Classical compiler passes we lack

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| A17 | **LICM** (loop-invariant code motion). Search loop bodies for expressions with loop-invariant operands; hoist. | 10-30% on hot loops | ~500 | Subsumed by A6 if egraph lands first. |
| A18 | **GVN** (global value numbering). CSE across control flow. | medium | ~700 | Subsumed by A6. |
| A19 | **Predicate hoisting**. `if_then(lt(idx, count), body)` inside a loop where count is invariant → hoist to loop bound. | 5-15% on guarded ops | ~300 | Subsumed by A6. |
| A20 | **Dead-store elimination** (different from DCE  -  for stores whose write is overwritten before any read). | small but additive | ~250 | |
| A21 | **Dead-load elimination** (load whose result is never read by anything live). | small but additive | ~200 | |
| A22 | **Store-to-load forwarding**. `store buf[i], v; load buf[i]` → `v` directly. | medium when present | ~350 | **[depends-on-A12]** for soundness. |
| A23 | **Branch coalescing**. Multiple `if x > 0` branches share predicate evaluation. | small | ~200 | |
| A24 | **Phi/select coalescing**. Adjacent `select(cond, ...)` chains with same cond. | small | ~150 | |
| A25 | **Boolean simplification** (Karnaugh-map style for chained logic). | small | ~400 | |
| A26 | **Loop fusion** (adjacent loops with disjoint writes → single loop). | medium when present | ~500 | |
| A27 | **Loop fission** (one fat loop body → two thinner loops, often enables vectorization). | enables A39 | ~400 | |
| A28 | **Loop peeling** (special-case first/last iteration to clean up bounds inside the body). | small | ~250 | |
| A29 | **Loop strip-mining** (turn `for i in 0..N` into `for ii in 0..N step T; for i in ii..ii+T` for vectorization/tiling). | enables A39, A53 | ~300 | |
| A30 | **Polyhedral loop transformations** (tile, interchange, skew). Hard but high payoff for tensor ops. | 2-10x on dense linalg | ~3000 | Long-term. |
| A31 | **Software pipelining** (overlap loop-body iterations to hide latency). | 1.2-1.8x on latency-bound loops | ~800 | |
| A32 | **Tail duplication** for divergent branches (lets each tail be optimized in its branch context). | helps wgpu/CUDA divergence | ~400 | |
| A33 | **Algebraic identity expansion** (distributive, associative reorder for vectorization). | enables A39 | ~600 | Subsumed by A6 rule DB. |
| A34 | **Strength reduction expansion**: shift for power-of-2 mul/div, add chains for small mul, Horner's rule for polynomials. | 5-20% on math-heavy ops | ~500 | Mostly already there but incomplete. |
| A35 | **Range-based folding**: if value provably in `[0, K]`, `min(x, K) → x`, etc. | medium | ~300 | **[depends-on-A16]**. |
| A36 | **Atomic minimization**: elide atomics where the writer is provably unique. NormalizeAtomicsPass exists; needs to be checked for completeness. | medium when atomics present | ~250 | |

---

## B. Lowering / emit-layer (substrate-aware patterns)

Per the architectural rule (CUDA is just an emit target, not a source of optimization), substrate-specific micro-opts belong here as patterns matched during emit, not as IR passes.

### B.1 Naga / wgpu emit

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| B1 | **Vec2/vec4 packing**: adjacent element accesses (`load[i], load[i+1], load[i+2], load[i+3]`) fuse into one vec4 load. Naga supports it. | up to 4x on memory-bound ops | ~600 | Probably the biggest emit-layer win. **[depends-on-A2]** for clean operand-stream detection. |
| B2 | **Naga IR caching at Module level**, not Program level. Two megakernels sharing an inner kernel re-emit today. | 30-50% [guess] on cold start | ~400 | Per-arm/per-shader hash. |
| B3 | **Parallel naga emit per arm**. Arms are independent. | 2-3x [guess] on cold-start emit time | ~200 | |
| B4 | **Pipeline reflection pre-warm during canonicalize**. First-dispatch reflection is sync-blocking today. | cold-only, but visible | ~150 | |
| B5 | **wgpu disk cache key fix**: per-arm hash, not full-Program hash. | meaningful cold-start improvement | ~200 | Currently at `vyre-driver-wgpu/src/pipeline/disk_cache*`. |

### B.2 CUDA / PTX emit

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| B6 | **Tensor-core (wmma/mma) fragment promotion** when shape divides 16 and dtype is f16/bf16. | 5-30x on matmul/conv/attention | ~1000 | Pattern at emit-time, not IR. |
| B7 | **ldmatrix / cp.async** for async tile loads in CUDA emit. | 1.5-2.5x on tiled loads | ~600 | |
| B8 | **Predicated execution** for short divergent branches (avoid actual branch). | small but additive | ~300 | |
| B9 | **PTX-level instruction scheduling** for instruction-level parallelism. Compiler does some, but often not enough for vyre IR shape. | 1.1-1.3x [guess] | ~500 | |

### B.3 Cross-substrate

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| B10 | **Constant-buffer promotion**: small read-only data → const-buffer (much faster than global on every backend). | medium when applicable | ~250 | |
| B11 | **Texture-memory promotion** for read-only spatial-access patterns. | helps on visual ops | ~400 | |
| B12 | **Shared-memory promotion** for arrays that fit and are reused across workgroup. | **5-50x on tiled ops** | ~800 | Massive. **[depends-on-A12 + A16]**. |
| B13 | **Shared-memory bank-conflict avoidance** (padding, swizzling). | up to 32x on stride patterns | ~400 | **[depends-on-B12]**. |
| B14 | **Memory-coalescing analysis**: detect non-coalesced global reads/writes, rewrite or warn. | **up to 32x on memory-bound ops** | ~700 | Probably the single biggest unrealized speedup in the codebase. |
| B15 | **Workgroup-uniform branch detection**: `subgroup_uniform` perf wins where branches are workgroup-coherent. | medium-large where uniform | ~300 | |

---

## C. Megakernel / runtime layer

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| C1 | **Whole-megakernel egraph** (not per-arm-then-fuse). Cross-arm GVN, cross-arm constant prop, dead-barrier elimination. | structural; the biggest egraph payoff | part of A6 | The reason A6's domain is megakernel-wide. |
| C2 | **Scratch-buffer reuse across arms** via escape analysis. | reduces allocation, helps cache | ~500 | **[depends-on-A13]**. |
| C3 | **Shared prologue extraction**: arms with identical binding setup / constant materialization share. | small but additive | ~300 | |
| C4 | **Barrier elision for value-flow chains**: arm B reads register-only result from arm A  -  no memory barrier needed. | medium | ~400 | **[depends-on-A6]**. |
| C5 | **Three-arm-with-gid-gated-middle-arm pattern**: extraction proves the gate makes middle a no-op for most threads, elides. | targeted at the recall-bug pattern | part of A6 | See `recall-bug-bisection-2026-04-30.md`. |
| C6 | **Pipeline reuse cache hit-rate audit**: how often does dispatch hit the cache? Probably much lower than it should be. | cold → warm conversion | ~150 audit | |

---

## D. Dispatch / driver layer (no IR changes)

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| D1 | **Persistent kernel mode** for many-small-launches (analyses like ssa, ifds, reaching that run thousands of times per scan). | 2-5x on probe-style workloads | ~600 | |
| D2 | **CUDA streams / wgpu queues** for independent megakernel arms. | depends on workload graph | ~400 | |
| D3 | **Async memcpy overlap with compute** (cudaMemcpyAsync + stream dependency). | helps when memcpy is non-trivial | ~300 | |
| D4 | **CUDA graphs / wgpu command bundles** for repeated dispatch patterns (the warm-batch loop). | 1.2-2x [guess] on dispatch overhead | ~500 | |
| D5 | **Multi-kernel concurrent launch** where occupancy permits. | small but additive | ~300 | |
| D6 | **Bind-group reuse across launches** (avoid re-binding the same buffers). | small but additive | ~250 | |
| D7 | **Push-constant inlining** for tiny scalar args (avoid uniform buffer). | small but additive | ~200 | |
| D8 | **Indirect dispatch** when launch shape depends on data. | enables data-dependent kernels without CPU roundtrip | ~400 | |
| D9 | **Bindless textures / buffers** (CUDA, wgpu where supported). | medium on many-buffer ops | ~600 | |

---

## E. Compile-time / cold-start

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| E1 | **Module-level naga IR cache** (B2 again  -  duplicate listed for layer clarity). | same as B2 |  -  | |
| E2 | **LRU cache for parsed/typed C source** in `vyre-libs::parsing::c`. | cold + warm | ~300 | |
| E3 | **Incremental re-optimization**: only rewalk subtrees whose hash changed. | medium on warm-batch | ~600 | **[depends-on-A1]**. |
| E4 | **CUDA module persistent across runs** (currently reload on every process launch). | huge cold-start | ~400 | |
| E5 | **PTX cache shared across processes** (file-system or memory-mapped). | huge cold-start | ~500 | |

---

## F. Specialization

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| F1 | **Shape specialization**: compile a kernel variant per common input shape (e.g. seq_len=512, head_dim=64). | 1.5-3x on shape-stable workloads | ~800 | Cache and dispatch by shape. |
| F2 | **Buffer-content folding**: if a buffer is provably constant (e.g. embedding table), fold into the kernel. | 1.2-2x on small const buffers | ~600 | |
| F3 | **Type specialization**: kernel variant per dtype (f16 vs f32 vs bf16). | substrate-dependent | ~400 | Already partly there. |
| F4 | **Backend-capability specialization** (sm_80+ vs sm_60, wgpu features). | when targeted | ~500 | |

---

## G. Numerical / precision

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| G1 | **Mixed-precision auto-downcast f32 → f16** where range-analysis proves accuracy bound. | 1.5-2.5x on memory-bound | ~600 | **[depends-on-A16]**. |
| G2 | **Reciprocal approximation** (`1/x → fast_inv`) where accuracy bounds permit. | small but easy | ~150 | |
| G3 | **FMA pattern matching** (`a*b + c → fma(a,b,c)`). Already partly there. | small | ~200 | Subsumed by A6. |
| G4 | **Horner's rule for polynomial expressions**. | small but predictable | ~250 | |
| G5 | **Range-reduced transcendentals** (sin, cos, exp, log fast paths). | helps if used | ~500 | |
| G6 | **Welford's algorithm for sum-of-squares** (numerical stability + perf). | only where used | ~150 | |
| G7 | **Block-FMA reduction** for accumulation (numerical + perf). | helps on long reductions | ~300 | |

---

## H. Algorithm-level rewrites

These are *automatic algorithm substitution*  -  high-payoff but tricky. Egraph extraction picks the variant that fits the cost vector.

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| H1 | **Strassen-like substitution** for matmul where N is large. | 1.1-1.4x on big GEMM | ~800 | Numerically tricky. |
| H2 | **FFT for convolution** where kernel size is large. | 5-50x on large conv | ~1500 | |
| H3 | **Im2col / direct-conv decision** based on shape and memory budget. | meaningful on small conv | ~600 | |
| H4 | **Flash-attention fusion** for attention ops (we have a 3-pass softmax in gqa_attention; flash is the fused-tile rewrite). | 2-5x on attention, big memory savings | ~1200 | High-payoff. |
| H5 | **Operator fusion above current scope**: gemm + bias + activation as a single kernel (already a known TASO/PET pattern). | 1.3-2x on transformer blocks | ~800 | Egraph rule DB. |

---

## I. Profile-guided / feedback-driven

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| I1 | **PGO-style hot-path recording**: record which arms / kernels are hot, recompile with hot-path hints. | 1.2-1.5x [guess] on long-running | ~700 | |
| I2 | **Trace-based JIT specialization**: warm patterns get specialized variants. | 1.3-2x on stable workloads | ~1000 | |
| I3 | **Auto-tuning database** persisted across runs (workgroup size, unroll factor, tile size per shape). | 1.2-2x once primed | ~600 | |
| I4 | **Occupancy-aware autotuning**: empirical search per shape with small budget. | 1.2-1.8x [guess] | ~500 | Pays back in one warm run. |

---

## J. Layout / data-side

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| J1 | **Buffer layout transformation** (AoS → SoA at the data level, with a transpose pass). | 2-10x on the wrong layouts | ~700 | Hard to do automatically; opt-in hint may be enough. |
| J2 | **Padding to avoid bank conflicts** (B13 again at the data layout level). | up to 32x on stride patterns | ~300 | **[overlaps-B13]**. |
| J3 | **Buffer alignment hints** to the driver (16/64/256-byte aligned). | small but additive | ~150 | |

---

## K. Validator / sanity (free wins)

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| K1 | **Validator skip-cache** (A5 again). | same as A5 | ~50 | |
| K2 | **Many sanity checks debug-only**. Currently every Program touch validates. | 2-8% [guess] on hot path | ~200 audit | |
| K3 | **Tag-bit assertions guarded** by debug_assert (A4 unlocks). | small | ~100 | **[depends-on-A4]**. |

---

## L. Frontend / parsing layer

| # | Item | Effect | LOC | Notes |
|---|------|--------|-----|-------|
| L1 | **Single-pass C lexer** (currently multi-pass via preprocess + lex + parse). | medium on parse-heavy paths | ~800 | Big refactor. |
| L2 | **Persistent parsed-AST cache** keyed on file content hash. | huge for incremental scans | ~400 | |
| L3 | **Parallel parse across files in a corpus**. | scales with cores | ~300 | |
| L4 | **Lazy scope resolution**: don't resolve names eagerly during parse. | small but compounding | ~500 | |

---

## R. Why egglog wins for the long tail (anchor for A6)

The classical answer to "we have 1000 micro-opts of varying applicability" is **add 1000 passes**. That's how LLVM has 200+ passes whose ordering is folklore and that don't compose well.

The egglog answer is **a relational rule database**:

- Every micro-opt is a small TOML file: `LHS pattern + RHS template + cost-aware applicability tag`.
- The engine indexes LHS patterns in a discrimination tree (Bachmair/Tanguy)  -  O(node-size) match per program.
- Rules fire in waves to a budget. Conflicts resolved by deterministic minimum-id merging.
- Extraction is cost-aware DP over e-classes; cost vector tunable per backend / per cold-vs-warm.
- New rule = drop a TOML in `rules/`  -  zero engine recompile.
- Telemetry: per-rule fire-count + cost-reduction. Dead rules culled.

This is **Tier-B configurability** per CLAUDE.md: the engine is the moat, the rule database is the *community-knowledge* moat, identical pattern to consumer rule ecosystems.

Subtleties worth knowing:
- **Applicability tags** (`bool_typed`, `gpu_only`, `target_has_tensor_cores`, `if_workgroup_size_le_64`, `cold_path`, `hot_path`) let substrate-aware and shape-aware rules fire only when relevant. Solves "how do we add a CUDA-specific algebraic rule without polluting the wgpu path."
- **Whole-megakernel domain** (not per-Program). Cross-arm GVN, cross-arm constant prop, barrier elision  -  all fall out from this domain choice.
- **Region/op-id metadata is side-band**, not e-graph nodes. Solves the validator-Region-scoping bug class at the optimizer layer (Region was conflated with scope and broke 1+ ops; future similar bugs vanish).
- **GPU-resident e-graph is phase 2** (item A10), only if measurement says CPU-side saturation is the bottleneck. Don't speculate-build it.

Reference reading:
- `egg`: Willsey et al. POPL 2021 (the classic egraph + saturation paper)
- `egglog`: Zhang et al. 2023 (datalog-style; better suited to parallelism)
- `TASO`: Jia et al. SOSP 2019 (NN graph substitutions, closest spiritual predecessor)
- `PET`: Wang et al. OSDI 2021 (partially-equivalent transformations)
- GPU union-find: Jaiganesh & Burtscher PPoPP 2018; Rama & Patel 2014

---

## Anti-patterns / things NOT to do

- **Don't add 100 hand-rolled passes.** That's the LLVM trap. Build the engine, populate the rule DB.
- **Don't put substrate-specific opts in the IR layer.** They belong in emit (B section) or in egraph applicability tags.
- **Don't optimize without M0 measurement.** Every "this should be 2x" estimate above is a guess.
- **Don't speculate-build A10 (GPU-resident e-graph).** CPU-side egglog is plenty fast for current workloads; don't pay GPU complexity until measurement says we need it.
- **Don't bypass validator with `unsafe` shortcuts.** Validator skip-cache (A5) is the right answer; bypassing is the wrong one.
- **Don't merge ops just because they look similar.** A6 cost-aware extraction makes the choice; manual fusion is a maintenance trap.

---

## Summary table  -  top 10 by speedup-per-LOC (best guess)

Ordered by my best guess at expected-payoff / LOC, **only after M0 measurement**:

| Rank | Item | Layer | Why |
|------|------|-------|-----|
| 1 | M0 flame-graph | meta | Without this we are guessing. |
| 2 | B14 memory-coalescing analysis | emit | up to 32x on memory-bound ops. |
| 3 | B12 shared-memory promotion | emit | 5-50x on tiled ops. |
| 4 | A1 hash-cons + A2 SoA Program | optimizer | Foundation for everything else; 2-5x optimizer time. |
| 5 | A11/A12 wire reaching+points-to into optimizer | optimizer | Already-built analyses; just unused. |
| 6 | B6 tensor-core promotion | emit | 5-30x on matmul/conv/attention. |
| 7 | H4 flash-attention fusion | algorithm | 2-5x on attention, big memory savings. |
| 8 | B1 vec2/vec4 packing | emit | up to 4x on memory-bound ops. |
| 9 | A6 egglog rule engine | optimizer | The long-tail answer; structural. |
| 10 | C2 scratch reuse across megakernel arms | runtime | Reduces allocator pressure, helps cache. |
