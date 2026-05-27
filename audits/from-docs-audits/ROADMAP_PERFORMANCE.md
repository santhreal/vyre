# vyre  -  pure-performance roadmap

Nothing in this file is breadth. No LSP, no studio, no WASM, no textures, no documentation plays. Every item is either (a) raw wall-clock speedup or (b) a technical innovation that doesn't exist anywhere else yet.

Items are ordered by **impact per engineering-hour**, not by "which phase."

---

## Category 1  -  Free money (days, not weeks)

### P-1. Registry lookup O(n) → O(1)
LazyLock FxHashMap. 5 LOC. Kills 6.4M string compares per cert run. **30m.**

### P-2. Arena-backed Value
`Value::Bytes(Arc<[u8]>)` + `Vec<Value>` slotted locals instead of `HashMap<String, Value>`. Reference interpreter 5-10×. **Half a day.**

### P-3. Constant folding + strength reduction
`3u + 4u → 7u`, `x * 2 → x << 1`. 10-30% fewer WGSL instructions in every composed op. **1 day.**

### P-4. Node uses Ident (not String)
Halve Program clone cost. Mechanical sed. **2h.**

### P-5. Zero-copy output-slice readback
Program declares "output bytes X..Y"; backend reads back only those bytes. For a 16MB output where the consumer needs 4 bytes, transfer 4 bytes. **Every existing GPU framework misses this.** **3 days.**

**Category 1 cumulative: ~3-4 days for ~15-30% end-to-end speedup on the cert run alone, plus 5-10× on the reference interpreter.**

---

## Category 2  -  Dispatch model overhaul (the one that matters most)

### P-6. Pipeline dispatch mode
Compile WGSL + pipeline + bind-group-layout once; dispatch repeatedly with different inputs. **90% of per-call overhead gone.** Every cert run, every streaming workload, every rule-engine evaluation. Trait-additive, not breaking. **2-3 days.**

### P-7. Streaming/chunked dispatch
`push_chunk(bytes) → GPU processes chunk N while CPU stages N+1`. Sustains 100% GPU utilization on any input > GPU RAM. Unlocks 10GB+ inputs (rule scanner on whole-disk, compression on large files). **2-3 days.**

### P-8. GPU-resident dispatch graph  ⟵ FIRST-IN-CLASS
CUDA Graphs is NVIDIA-proprietary. Emit an indirect dispatch graph the GPU executes itself. One CPU→GPU launch, hundreds of sequential kernels on-device. **Ports CUDA Graphs to every backend via the vyre IR.** Saves 10-100 µs per op.
No Rust GPU lib has this. **5-7 days.**

### P-9. Temporal coalescing
If a Program is dispatched 1000×/sec with different inputs, runtime batches N dispatches into ONE compound dispatch. Invisible to the caller. Rule scanner processing a packet stream suddenly goes 10× faster. **3-5 days.**

### P-10. Async copy + multi-stream execution
`Node::AsyncLoad { tag } ... Node::AsyncWait { tag }`. CUDA/Metal/Vulkan all expose overlap; vyre's IR abstracts it. Compute + memcpy overlap, permanent 100% GPU util. **3 days.**

**Category 2 cumulative: ~2-3 weeks. The cert run goes from ~90s to <5s. Every downstream workload that dispatches in a tight loop (rules, ML inference, streaming codecs) gets 10-100× throughput.**

---

## Category 3  -  GPU primitive unlock (raw ceiling)

### P-11. Subgroup / SIMD-lane ops  ⟵ DEFINING FEATURE
`Broadcast, ShuffleXor, ShuffleUp/Down, Ballot, Any, All, Reduce, InclusiveScan, ExclusiveScan`. Every reduction / prefix_sum / histogram / sort becomes 4-8× faster.
IR stays backend-agnostic: WGSL `subgroupBroadcast`, PTX `shfl.sync`, MSL `simd_shuffle`, Vulkan `subgroupShuffleXor`. **3-5 days.**

### P-12. Cooperative matrix (tensor cores)
`CoopMatrix::{Load, Store, MulAdd}`. WMMA on CUDA, simdgroup_matrix on Metal, coopmat on Vulkan. Opens ML inference primitives: attention, linear projection, batched matmul. **10-100× on matmul shapes.** **3-5 days.**

### P-13. Indirect/conditional dispatch
Dispatch count determined by prior kernel's output. Foundational for sparse workloads, stream compaction, rule-match fan-out. **2 days.**

### P-14. Workgroup-shared auto-sizing
IR-level dataflow analysis proves a bound on SRAM requirement. Emits tightest-possible shader. Current workgroup primitives (stack, hashmap, queue) force user to declare SRAM capacity; vyre derives it. **3-4 days.**

**Category 3 cumulative: ~2 weeks. Unlocks every modern GPU performance idiom. Closes the gap to CUDA/Metal/Vulkan on raw primitives.**

---

## Category 4  -  Optimizer 2 → 9 passes + unique innovations

### P-15. Standard optimizer passes
Beyond const-fold + strength-reduce (P-3): LICM, copy propagation, GVN, loop unrolling, induction-var simplification, barrier elimination, buffer coalescing. 1 pass = 1 file with a `#[vyre_pass]` attribute. Pass scheduler runs to fixpoint. **~1500 LOC across 7 passes. 1 week.**

### P-16. Kernel fusion at IR level  ⟵ DEFIES PRIOR ART
XLA/TVM/Triton do kernel fusion for ML. *None of them prove it correct*  -  they rely on the developer finding miscompilations via regression testing. vyre's kernel fusion uses the conform gate to **prove the fused kernel is byte-identical to the split version**. First provably correct GPU kernel fusion in any Rust GPU compute library.
Detects `op_A(x) → op_B(result)` patterns in the IR; emits a single shader with the combined body. **2×-10× speedup on every composed pipeline (which is every non-trivial workload).**
**1-2 weeks.**

### P-17. Spec-driven optimizer  ⟵ UNIQUE APPROACH
LLVM/XLA use hardcoded transform rules. vyre reads the OpSpec declarations:
- "op_A is commutative → reassociate"
- "op_B is idempotent → fold duplicate consecutive calls"
- "op_C has identity element I → `op_C(x, I) → x`"
- "op_D is involutive → `op_D(op_D(x)) → x`" (xor, bitwise-not)
Optimizer strength grows with catalog. **3-5 days.**

### P-18. Proof-carrying dispatches  ⟵ FIRST-IN-CLASS
Every dispatch emits a proof cert showing which IR-level laws were used during optimization. If a backend diverges from the cert, that's a backend bug with a specific law it broke. **Whole runtime becomes auditable, not just the spec. Nobody else has this.**
**3-4 days.**

**Category 4 cumulative: ~3 weeks. Vyre becomes the only GPU compute optimizer that can prove every transform it makes.**

---

## Category 5  -  Correctness innovations that defy what was possible

### P-19. Cross-vendor bit-identical determinism  ⟵ THE MOAT
Nobody guarantees this. Not NVIDIA, not Apple, not AMD. Same IR on two vendors today → different bytes, vendor-dependent "approximately IEEE 754." vyre's strict-mode cert gate **forces bit-identity** across NVIDIA/AMD/Apple/Intel. Runs in daily CI. First in the world. Adoption argument for any ML platform where reproducibility matters.
**Engineering: ~1 week** (cert format already supports vendor diffing; need real CI infra + real ieee754.rs with CR-Math CR-level transcendentals).
**CI infra: 2 weeks to stand up 2-vendor (NVIDIA + Apple); AMD/Intel as they become available.**

### P-20. ULP-budget approximate compilation
Call site declares `max_error: 2ulp`. vyre emits the fastest shader that provably meets that bound (uses fast approximations where safe, falls back to strict where not). Dial to `0ulp` → strict. **No GPU compiler does this at op level.** **1 week.**

### P-21. Multi-GPU work stealing
Run ONE `vyre::Program` across N GPUs. Runtime partitions at buffer boundaries, schedules shards, gathers results. Transparent to caller. No Rust GPU lib has this. **1-2 weeks.**

### P-22. GPU-native fuzzing
`vyre::fuzz::gen_program` random valid IR + differential test reference vs backend. Weekend run finds more driver bugs than a year of hand-written tests. **~600 LOC. 3-4 days.**

### P-23. SMT proof per optimizer pass
For each `#[vyre_pass]`, emit a bounded SMT problem: "∀ Program p, depth ≤ 8, dispatch(p) == dispatch(pass(p))." Z3 proves it. Ship the Z3 proof as part of the release. **Makes every optimizer pass provably safe.** **~500 LOC + per-pass effort. 1 week.**

**Category 5 cumulative: ~3-4 weeks. These are the items competitors can't catch up on quickly  -  they're all either cert-gated (require vyre's invariant infrastructure) or architecturally novel (work-stealing Program partitioning, SMT-verified passes).**

---

## Category 6  -  Items I missed on first pass (deep-pass additions)

### P-24. Persistent kernels  ⟵ STREAMING WORKLOAD KILLER
A kernel that stays resident on the GPU and pulls work from a GPU-side queue. Rule engine scanning a packet stream, inference serving a batch pipeline  -  one kernel launch, indefinite work. **Eliminates ALL per-batch launch overhead.** CUBLAS/FlashAttention use this; no Rust GPU lib exposes it.
**5-7 days.** Pairs naturally with P-8 GPU-resident dispatch graph.

### P-25. Ahead-of-time kernel specialization + on-disk cache
Compile specialized versions of a kernel for observed argument shapes. `scan<N=1024>` gets a kernel specialized for exactly N=1024. Cache on disk keyed by spec hash + backend fingerprint. **Next process start: zero WGSL compilation time.** Node.js v8's warmup snapshot applied to GPU compute.
**5 days.**

### P-26. Profile-guided backend routing (PGO for GPU)
Cert gate measures each op's latency on every backend as a side effect. Runtime uses those measurements to route at call time: "sha256 is 2× faster on cuda-native with subgroup ops → route there." Not hardcoded priority; measured.
**3-4 days.** Depends on cert gate producing latency histograms per op+backend.

### P-27. Program hash → pipeline cache
Program blake3 hash → compiled pipeline on disk. Same IR = instant load. Not WGSL text cache (which drivers already do); pre-validated Program cache. Saves IR validation + WGSL codegen + Naga pass on every re-run.
**2-3 days.**

### P-28. Constant-buffer folding + shader monomorphization
Buffer declared with compile-time-constant contents (e.g., a 256-entry LUT) → inlined into shader as an immediate array. Saves buffer upload + bind group + one indirection per access. Rust `const` semantics for GPU buffers. Applies to every table-driven algorithm (AES S-boxes, hash LUTs, CRC tables).
**3-4 days.**

### P-29. Dead buffer elimination
IR dataflow proves a declared buffer feeds no output. Skip the allocation + upload + bind. Saves memory + bandwidth + binding-table slot.
**2 days.** Pairs with P-15 DCE pass.

### P-30. Shared-nothing parallelism detection
Analyze the IR for write-after-write conflicts. Prove two ops share no writable state → emit them as concurrent dispatches on separate command queues. CUDA streams, automatically. No manual stream management in Rust GPU today.
**3-4 days.**

### P-31. Distribution-aware algorithm selection  ⟵ SELF-OPTIMIZING
Record observed data distribution per op call site (via cert gate sampling). Next call, pick the algorithm that fits: small N → insertion sort, uniform → bitonic, skewed → radix. Conform verifies all variants are byte-identical; runtime picks the fastest one for this distribution. **Adaptive GPU compute is niche even in research.**
**1 week.**

### P-32. Backend capability fingerprinting
Hash actual backend behavior (subgroup size, rounding mode, transcendental impl quality, ULP on known witnesses) into the cert. If the fingerprint drifts between runs (driver update silently changed a rounding rule), the next cert mismatches and surfaces the drift. **Driver-regression detector built into every cert.**
**2 days.**

### P-33. Numerical-stress determinism verification
Every op cert includes runs against NaN-producing / infinity-producing / overflow-at-boundary / subnormal-entry / denormal-exit witnesses. Proves deterministic outcome (even if outcome is "Error"). Competing suites cert happy path + random-ish; vyre certs the floating-point underworld.
**3-4 days.**

### P-34. Replay-based bisection
Every cert run emits a replay log of dispatches + bytes. `cargo vyre replay cert-2026-04-17.log` reruns exactly that sequence on any machine. When a regression appears, `git bisect` + replay narrows to the exact commit + exact dispatch that broke. **Time-travel debugging for GPU compute.**
**3 days.**

**Category 6 cumulative: 11 items, 3-4 weeks of engineering, largely parallelizable.**

---

## Timeline (3 Codex + Kimi swarm + orchestrator)

| Sprint | Duration | Items | Ship |
|---|---|---|---|
| 1 | 3-4 days | P-1..P-5 (Category 1) + P-6 Pipeline | v0.4.1 |
| 2 | 3-4 days | P-7 streaming, P-8 GPU-resident dispatch graph, P-10 async copy, P-13 indirect dispatch | v0.4.2 |
| 3 | 3-5 days | P-11 subgroup ops, P-14 workgroup auto-size | v0.5.0 (spec bump) |
| 4 | 3-5 days | P-12 coop-matrix, P-15 optimizer 2→9 | v0.5.1 |
| 5 | 1-2 weeks | P-16 kernel fusion, P-17 spec-driven optimizer, P-18 proof-carrying, P-23 SMT | v0.5.2 |
| 6 | 3-5 days | P-9 temporal coalescing, P-22 fuzzer | v0.5.3 |
| 7 | 1-2 weeks | P-19 cross-vendor determinism, P-20 ULP-budget, P-21 multi-GPU | v0.6.0 |

**Total: 6-10 weeks sustained to v0.6.0  -  every item above delivered.**

No LSP. No studio. No WASM. No textures. No rendering. No community features. No marketing work. Only raw speed + innovations no other GPU compute library can demonstrate today.

---

## Parallelization analysis  -  why 6-10 weeks, not 1

The 6-10 week number assumes serial-ish review. Here's what's actually parallelizable vs blocked.

### Trivially parallel (any number of Codex, no dependencies)
- **Cat 1** (P-1..P-5): 5 Codex, 1 day. All independent files.
- **P-11, P-12, P-13, P-14** (Cat 3 GPU primitives): 4 Codex parallel. IR-additive, non-overlapping.
- **P-15** optimizer passes: 7 Codex parallel, 1 pass each. Each a single file with `#[vyre_pass]`.
- **P-20, P-21, P-22** (Cat 5 pieces): 3 Codex parallel.
- **P-24, P-25, P-26, P-27, P-28, P-29, P-30, P-32, P-34** (Cat 6): 9 Codex parallel, mostly independent.

**If we ran every one of those in parallel**, engineering time collapses to ~**5-7 days** of wall-clock code-writing.

### Sequential dependencies that can't be compressed
- **P-16 kernel fusion** needs the optimizer pass framework (P-15 scheduler) to land first. +3-4 days sequential.
- **P-17 spec-driven optimizer** builds on the P-15 pass framework. +2-3 days sequential.
- **P-18 proof-carrying dispatches** needs cert-gate hooks to emit proofs. +3 days sequential.
- **P-23 SMT proofs** needs the passes to exist (P-15) and needs their invariants in the spec (requires designing the SMT encoding). +5-7 days sequential.
- **P-8 GPU-resident dispatch graph** builds on P-6 Pipeline mode. +3 days sequential.
- **P-9 temporal coalescing** builds on P-6 and P-8. +2 days sequential.
- **P-10 async copy** builds on P-6. +2 days sequential.
- **P-31 distribution-aware algorithm selection** needs P-26 PGO measurements. +3 days sequential.

**Critical path (longest chain)**:
P-6 Pipeline → P-8 GPU graph → P-9 temporal → P-15 passes → P-16 fusion → P-23 SMT on fusion.

Best case if EVERY step moves the day it unblocks: **~3 weeks wall clock**. Not 1 week because you can't have someone write P-16 fusion before you have a pass framework to register fusion as a pass.

### Things that literally can't be compressed (need real time to elapse)
- **P-19 cross-vendor bit-identical determinism**  -  needs real CI hardware running real cert runs across NVIDIA + Apple Silicon + AMD + Intel. Code is ~1 week to write; **stability data burn-in** to prove "daily cert holds 30 days running" is 30 days. Can't skip.
- **Bench stabilization**  -  every Cat 1 / Cat 4 change needs perf variance characterization. Running the same benchmark 20 times to establish confidence intervals takes real wall clock; can't run 20 iterations in parallel without 20 GPUs.
- **Review bandwidth**  -  me reading every diff. 34 items × 30 min average per review = 17 hours of my focused time. Can compress by batching diffs per day (6h/day of reviews, 3 days). But I'm also writing code + directing agents + hunting rot. Realistically 1 week of my time sustained.
- **Integration conflicts**  -  when P-16 fusion landing conflicts with P-15 DCE pass rewrite, someone has to resolve. Can't parallelize.
- **Codex rate limits**  -  I can run 3-4 Codex concurrent before 429s hit. Beyond that, Kimi fills in at single-file scope. Total concurrency ceiling ~8 agents; above that, quality drops.

### Honest bucketed ETA

| Scenario | ETA |
|---|---|
| **Code written, compile green** (every item coded, no validation) | 10-15 days |
| **Code + benches stable + CI green** (no cross-vendor determinism data) | 3-4 weeks |
| **Everything above + cross-vendor determinism proven via 30-day burn-in** | 6-10 weeks |
| **Theoretical absolute minimum** (infinite perfect agents, no review latency, no CI burn-in) | ~5-7 days |

So the honest answer: **13-23 items can ship "in the next 10 days" if we hammer it.** The remaining ~11 items (most of Cat 4 + all of Cat 5 burn-in) need the critical-path work to settle first.

### What pegs wall clock longer than engineering days
1. **Cross-vendor CI burn-in**  -  owns a month of calendar regardless of code speed. Can be running in the background while other work ships.
2. **Integration + review cycles**  -  every merge rebases + re-benchmarks. Adds 1-2 days per sprint.
3. **One Codex redispatch per sprint on average**  -  quality rollbacks ~25% of time.
4. **Benchmark variance**  -  CI needs 10+ runs to establish confidence on perf claims. Real wall clock.

### What could take it to 5 days flat
- Accept "code compiles + tests green" without multi-day perf stabilization (fine for a pre-1.0 line)
- Defer P-19 cross-vendor determinism infra to v0.7 (ship cert format but don't claim 30-day data yet)
- Pre-stage 4 Codex lanes + Kimi swarm, no idle time between items
- Direct commit to main, no review gate (accept post-hoc review via the cert gate itself)

If we do all four: the "code green, innovations shipped, no 30-day burn-in" version is **1 week of hard work.** The rest is waiting for CI to prove the determinism claims hold.

---

## Benchmarks we publish on the way (technical only  -  no vanity)

1. `cargo bench --bench cert_run_time`  -  full registry wall-clock from v0.4.0 baseline → each sprint.
2. `cargo bench --bench subgroup_reduce`  -  prefix_sum barrier vs subgroup, 1M u32.
3. `cargo bench --bench coop_matrix_gemm`  -  f32 GEMM 1024×1024 vs hand-tuned CUDA.
4. `cargo bench --bench fused_pipeline`  -  5-op composition, fused vs split.
5. `cargo bench --bench streaming_throughput`  -  10GB input, sustained GB/s.
6. `cargo bench --bench graph_dispatch`  -  100-op program, CPU-driven vs GPU-resident.
7. `cargo bench --bench multi_gpu_scan`  -  prefix_sum on 1/2/4/8 GPUs.

CI-gated: any sprint that regresses these reverts.

---

## What defines "defies what was previously possible"

Pick any of these items and try to find a prior implementation:

- **Proof-carrying GPU kernel fusion (P-16)**  -  XLA/Triton have fusion; none prove correctness.
- **Proof-carrying dispatch (P-18)**  -  no GPU runtime emits dispatch-level proofs.
- **Spec-driven optimizer (P-17)**  -  LLVM uses hardcoded rules; vyre uses runtime-discoverable OpSpec laws.
- **Cross-vendor bit-identical determinism (P-19)**  -  IEEE 754 "approximate parity" is industry status quo; vyre forces strict bit-identity via daily cert.
- **SMT-verified optimizer passes (P-23)**  -  LLVM alive but limited to small lowerings; vyre verifies every pass.
- **GPU-resident dispatch graph for every backend (P-8)**  -  CUDA Graphs is NVIDIA-exclusive; vyre ports the concept.
- **Multi-GPU work stealing at Program granularity (P-21)**  -  nothing in Rust GPU space.
- **ULP-budget approximate compilation (P-20)**  -  no GPU compiler exposes this.

Any one of these is enough to matter. Shipping all ten in one release is the entire point.
