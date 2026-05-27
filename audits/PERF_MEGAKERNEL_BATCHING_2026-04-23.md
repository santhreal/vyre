# PERF_MEGAKERNEL_BATCHING_2026-04-23

**Scope:** `vyre-driver-megakernel` + `vyre-runtime::megakernel` + `surgec::compile::fuse` + `surgec::compile::specialize`  
**Claim under audit:** "Megakernel fusion coalesces many rule programs into one dispatch, amortizing PCIe overhead and beating per-rule serial dispatch."  
**Method:** Read-only source audit. No runtime measurements were taken; claims are evaluated against code that exists today.

---

## Answers

### 1. How many Programs does the megakernel batch per call today, and what is the hard cap? Is the cap driven by device limits or a guess?

| Layer | Cap | Source | Nature |
|---|---|---|---|
| Compile-time fuse | 1,048,576 rules (`MAX_FUSED_RULES = 1 << 20`) | `libs/tools/surgec/src/compile/fuse.rs:103` | **Guess/contract**  -  doc-comment says "well below the available opcode range; a document that trips this needs to be split into multiple megakernel tenants anyway." No device query, no occupancy model. |
| Runtime work-queue | `u32::MAX` (~4.29 B) work items | `libs/performance/matching/vyre/vyre-runtime/src/megakernel/batch.rs:173` | Address-space limit, not a performance-aware bound. |
| Legacy `WgpuMegakernelDispatcher` | Unlimited items, but **1 workgroup** (≤256 lanes) | `libs/performance/matching/vyre/vyre-runtime/src/megakernel/wgpu_dispatch.rs:69-76` | Dispatches `[1, 1, 1]` grid regardless of queue length. Slot count is rounded to workgroup multiple, but grid size never grows. |
| `BatchDispatcher` (DFA batch path) | `worker_groups: 1` default × `workgroup_size_x: 64` default = **64 total threads** | `libs/performance/matching/vyre/vyre-runtime/src/megakernel/dispatcher.rs:77-78` | Caller-configurable, yet defaults and benchmark (`megakernel_batch.rs:41-43`) both keep `worker_groups = 1`. |

**Verdict:** The theoretical cap is ~1 M fused rules / 4 B work items, but the actual GPU parallelism is tiny  -  64–256 threads looping to claim multiple items. The hard cap is **not** driven by device limits; it is an arbitrary compile-time constant justified by opcode-range convenience, not by register-file size, LDS capacity, or warp-occupancy probing.

---

### 2. Is rule-independent work (input buffer upload, decode, tokenize) deduplicated across the batch, or does every rule re-upload + re-decode?

**Input buffer upload  -  YES, deduplicated.**  
`FileBatch::upload` packs all file bytes into one contiguous haystack buffer and uploads it once (`batch.rs:181-186`). Offsets and metadata tables are also uploaded once.

**DFA transition/accept tables  -  YES, deduplicated.**  
`BatchDispatcher::ensure_rule_buffers` fingerprints each rule's DFA with BLAKE3 and shares identical tables across the catalog (`dispatcher.rs:253-284`, `pack_rule_catalog` at line 582).

**Decode / tokenize  -  NO, NOT deduplicated across the batch.**  
- `specialize.rs:168-181` fuses `base64_decode → aho_corasick` into a single per-rule `Program` at compile time. This is **per-rule** fusion, not batch-level deduplication.  
- `fuse.rs:261-264` splices each rule body into an independent `if opcode == X { body }` arm.  
- The optimization pipeline (`canonicalize → hoist_common_arm_lets → cse → region_inline → dce`) only hoists let-bindings that are **byte-identical across ALL arms** (`fuse.rs:351-409`). If rule A decodes base64 and rule B decodes hex, or even if both decode base64 but at different offsets, the decode work is replicated inside each arm.  
- There is no batch-level "decode once, share decoded bytes" pass in the megakernel path today.

**Verdict:** The megakernel deduplicates PCIe upload and DFA tables, but **every rule arm that needs decode/tokenize recomputes it independently**. The release "GPU-resident pure-VRAM pipeline" (Innovation I.1 in `RELEASE_GATE.md`) is a plan, not a realized batch deduplication.

---

### 3. Does the batching pass emit a single WGSL entry point or N entry points sharing a command encoder?

**Single WGSL entry point.**  
- `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:104-105` pushes exactly one `EntryPoint` named `"main"` with `ShaderStage::Compute`.  
- `vyre-runtime/src/megakernel/builder.rs:77-88` (`build_program_jit`) constructs one `Program` containing a single `Node::forever(persistent_body_jit(...))` loop.  
- End-to-end WGSL emit test confirms one `@compute @workgroup_size(...)` entry (`vyre-driver-wgpu/tests/megakernel_emit.rs:154-156`).  
- The `BatchDispatcher` calls `pipeline.dispatch_persistent` once per batch (`dispatcher.rs:205`), which records **one** compute pass into a single command encoder (`pipeline_persistent.rs:143-149`).

**Verdict:** One WGSL compute shader, one `compute pass`, one command encoder submission. The "N entry points" model does not exist in the current path.

---

### 4. Is there a benchmark proving megakernel > loop-dispatch at the chosen batch size? Cite it.

**No.**

- `vyre-runtime/benches/megakernel_batch.rs` measures megakernel throughput at 1 / 64 / 1,024 / 4,096 / 16,384 files. It has **no serial-dispatch baseline**. The only assertion is an RTX 5090 SLO for 4,096 files (< 50 ms, line 76-82).
- `vyre-libs/benches/cooperative_dfa_vs_classic.rs` compares cooperative DFA against classic Aho-Corasick. It is a **scan-algorithm** benchmark, not a **dispatch-strategy** benchmark.
- The serial loop-dispatch path lives in `surgec/src/scan/dispatch.rs:96-149` (`dispatch_rules`), which calls `backend.dispatch` once per rule. No criterion bench compares `dispatch_rules` against `BatchDispatcher::dispatch`.

**Verdict:** The claim that megakernel beats loop-dispatch is **unproven in code**. There is no `criterion` comparison, no `#[bench]` pairing, and no CI gate enforcing the win.

---

### 5. What's the first regression point where megakernel starts to LOSE to serial dispatch (register pressure, stack depth, LDS contention)?

**Unknown  -  the codebase contains no measurement or model of this crossover.**

What we *do* know:
- The **serial** dispatch path has explicit register-pressure awareness: `optimal_workgroup_size` counts `Node::Let` bindings and drops workgroup size from 256 → 64 when `let_count ≥ 24` (`surgec/src/scan/dispatch.rs:323-354`).
- The **megakernel** path has **zero** equivalent analysis. `build_program_jit` and `FusionPlan::optimize` accept a caller-provided `workgroup_size_x` (`builder.rs:77`, `fuse.rs:333`) but never inspect the fused body to adapt it.
- As more rules are fused, the if-tree grows linearly. Every arm's live temporaries compete for the same register file; there is no per-arm register windowing or spilling analysis.
- The default `BatchDispatchConfig` runs 64 total threads (`worker_groups=1`, `workgroup_size_x=64`). Each thread loops `claim_budget = ceil(queue_len / 64)` times. At 16,384 files × 1 rule, that's 256 iterations per thread  -  well within typical GPU timeouts, but the loop body holds `state`, `file_start`, `file_end`, `rule_base`, `transition_base`, `accept_base`, `byte_pos`, `byte`, `accepting`, `hit_slot` simultaneously. No register budget is checked.

**Inferred first regression point:**  
When the fused rule count pushes the persistent body's live-variable count past the adapter's register-file budget per thread (typically 64–128 vector registers on modern GPUs), the driver will spill to scratch memory. Because the megakernel has no adaptive workgroup shrink, it cannot trade occupancy for register pressure the way the serial path does. The crossover likely occurs **before** `MAX_FUSED_RULES` (1 M) and probably in the low-hundreds of fused rules on current hardware, but this is an **unmeasured hypothesis**.

---

## MEGA-N Findings

**MEGA-1** | **BUSTED** | `libs/tools/surgec/src/compile/fuse.rs:103` | `MAX_FUSED_RULES = 1 << 20` is justified as "well below the available opcode range," not by any device limit or occupancy probe. A 1 M-rule document would overflow any real GPU register file long before it overflows the opcode counter. | Fix: Cap `MAX_FUSED_RULES` by a runtime register-pressure model, or at least split at an occupancy-proven threshold (≤256 rules for current hardware). Measure `build_program_jit` compile time and emitted WGSL register count vs. fused rule count.

**MEGA-2** | **BUSTED** | `libs/performance/matching/vyre/vyre-runtime/src/megakernel/wgpu_dispatch.rs:69-76` | `WgpuMegakernelDispatcher` clamps `workgroup_size_x` to 256, computes `slot_count`, then dispatches `[1, 1, 1]`  -  exactly one workgroup no matter how many work items are queued. | Fix: Scale `workgroups` with `slot_count / workgroup_size_x` (or use `BatchDispatcher` semantics). Add a test asserting `grid_size > 1` when `item_count > workgroup_size_x`.

**MEGA-3** | **BUSTED** | `libs/performance/matching/vyre/vyre-runtime/src/megakernel/dispatcher.rs:77-78` | `BatchDispatchConfig::default()` sets `worker_groups: 1`. The benchmark (`megakernel_batch.rs:41-43`) also hard-codes `worker_groups: 1`. This means 64 threads process up to 16,384 files by looping 256×  -  serializing the inner scan inside each thread instead of parallelizing across the GPU. | Fix: Benchmark `worker_groups = file_count / workgroup_size_x` (saturating at SM count) and prove it beats `worker_groups = 1`. Default `worker_groups` to `min(SM_count, ceil(file_count / workgroup_size_x))`.

**MEGA-4** | **UNPROVEN** | `vyre-runtime/benches/megakernel_batch.rs` | The benchmark measures megakernel wall time at varying file counts but has **no serial-dispatch baseline**. The SLO assertion (4096 files < 50 ms on RTX 5090) proves nothing about relative speedup. | Fix: Add a paired `bench_serial_dispatch` that calls `surgec::scan::dispatch_rules` (or an equivalent per-rule `backend.dispatch` loop) on the same corpus and rules. Fail the benchmark if megakernel is not faster by a documented margin (e.g., 2×) at the chosen batch size.

**MEGA-5** | **CONFIRMED** | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:104-105` | The batching pass emits exactly **one** WGSL `@compute` entry point. `builder.rs:77-88` wraps the fused payload in a single `forever` loop. | No fix needed  -  this is the correct release architecture. Keep it.

**MEGA-6** | **BUSTED** | `libs/tools/surgec/src/compile/fuse.rs:351-409` | `hoist_common_arm_lets` only deduplicates let-bindings when **every** opcode arm starts with the exact same `Node::Let { name, value }`. Two rules that both load `file_bytes` at the same offset will still duplicate the load if their second statements differ. True cross-arm CSE is blocked by the `Region` wrapping that happens before CSE runs. | Fix: Run global CSE **before** `rewrap_rule_arms`, as described in `RELEASE_GATE.md` Innovation I.2. Add a test fusing 100 rules that all `load("haystack", offset)` and assert the optimized payload contains exactly one `load`.

**MEGA-7** | **BUSTED** | `libs/performance/matching/vyre/vyre-runtime/src/megakernel/dispatcher.rs:287-296` + `libs/tools/surgec/src/scan/dispatch.rs:323-354` | The serial dispatch path adapts workgroup size to register pressure (`let_count ≥ 24 → 64 lanes`). The megakernel path has **no** equivalent analysis. A megakernel fused from 200 rules will likely spill registers, yet it still runs at the caller's fixed workgroup size (64–256) with no occupancy feedback. | Fix: Port `optimal_workgroup_size` logic to `FusionPlan::optimize`. Count live temporaries across the fused payload; if pressure exceeds a device-dependent threshold, emit a smaller workgroup size or split the fusion plan into multiple tenants.

**MEGA-8** | **BUSTED** | `libs/tools/surgec/src/compile/specialize.rs:168-181` | `specialize_decode_chain` calls `fuse_base64_decode_scan_program` per-rule. The decode→scan fusion is **rule-local**; if 50 rules process the same base64-encoded file, the GPU decodes it 50 times (once per rule arm). | Fix: Move decode to a pre-batch pass that writes a shared `decoded` layer buffer into `FileBatch`, then reference that layer in rule arms. Alternatively, emit a megakernel preamble that decodes once per file slot before the opcode switch, and pass decoded handles in the work-item triple.

**MEGA-9** | **MISSING** | `libs/performance/matching/vyre/vyre-runtime/src/megakernel/dispatcher.rs:205` | `BatchDispatcher::dispatch` calls `dispatch_persistent` with `[worker_groups, 1, 1]`. There is no metric for **occupancy** (active warps / max warps), **register spills**, or **LDS usage** returned to the caller. The first regression point is therefore invisible. | Fix: Add a `BatchDispatchReport` field for `occupancy_estimate` (computed from rule count × state variables / register file size) and a warning threshold. Run a parameterized benchmark sweeping `rule_count = 1..1000` at fixed file count to find the knee where wall time stops being flat and starts growing super-linearly.

---

## Summary

The megakernel architecture **is** structurally correct  -  single WGSL entry point, persistent work queue, one command encoder  -  but the current implementation is **not release-scale** today. The hard cap is an arbitrary 1 M rules with no occupancy model; the actual GPU parallelism defaults to 64 threads; rule-independent decode work is not batch-deduplicated; and there is **no benchmark proving it beats serial dispatch**. The serial path has better register-pressure awareness than the megakernel path, which is a glaring inversion. Fix order: (1) add the serial baseline benchmark, (2) measure the regression knee, (3) wire cross-arm CSE + batch-level decode deduplication, (4) scale `worker_groups` with batch size.
