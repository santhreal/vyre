# GPU-beats-Hyperscan-by-OOM: the segmentation thesis (code-grounded)

Goal: on a single 8 MiB scan, the wgpu megakernel must beat Hyperscan (~1.5 GB/s,
~5.6 ms) by an order of magnitude. RTX 5090 ≈ 1.7 TB/s, so a fully
bandwidth-bound scan of 8 MiB is ~5 µs — a ~1000× ceiling physically exists. The
gap today is **parallel decomposition + per-lane inner-loop cost**, not physics.

## Root cause (read, not theorized)

1. **Work-item geometry is (file × rule), never (segment × rule).**
   `vyre-driver-wgpu/src/megakernel/dispatcher.rs:1385-1417`:
   ```
   file_idx = claim / rule_count
   rule_idx = claim % rule_count
   scan file_offsets[file_idx] .. file_offsets[file_idx+1]   // the WHOLE file
   ```
   For one 8 MiB file, work-items = `rule_count` (~hundreds). Each lane walks the
   **entire 8 MiB sequentially** for its one rule. So occupancy is bounded by the
   rule count, not by the 21,760 cores, and every busy lane still does an 8 MiB
   byte-at-a-time DFA walk. The `segment_count` / `command_encoder_count`
   machinery (`dispatcher.rs:77-196`) splits the **batch across command
   encoders** for dispatch; it does **not** split a single input into windows.

2. **The portable reference kernel is single-threaded.**
   `vyre-driver-wgpu/src/shaders/aho_corasick_scan.wgsl` is
   `@workgroup_size(1,1,1)` gated to `id.x == 0`, and re-copies all 256
   transition entries into workgroup memory **per byte** (a `workgroupBarrier`
   with one thread). It is the conformance/parity reference, not the megakernel,
   but anything that dispatches it pays a 1-lane sequential scan.

3. **Per-lane transition reads are uncached global loads.** Byte-at-a-time
   `transitions[state*256 + byte]` against a ~1 GB catalog is latency-bound
   (~400–800 cyc/load), ~100× slower per byte than Hyperscan's SIMD shuffle.

Net: rule-parallelism (×rule_count) cannot overcome a slow, 1-lane-per-rule,
full-file sequential walk. That is why GPU loses to Hyperscan on 8 MiB.

## The fix (two parts, both required)

### A. Input segmentation → (segment × rule) work-items
Split each file into overlapping windows so work-items = `rule_count ×
ceil(file_len / seg_len)`, saturating all cores even for a single file.
- `seg_len` ≈ 512 B → 8 MiB ⇒ 16,384 segments × rule_count ⇒ full occupancy.
- Each window scans `[seg_start − OVERLAP, seg_start + seg_len)` from state 0 and
  **emits only matches at `pos ≥ seg_start`**.
- `OVERLAP = max_pattern_len`. **Soundness:** an Aho-Corasick / failure-function
  DFA state at position *i* is a function of at most the last `max_pattern_len`
  bytes (bounded suffix history), so after an `OVERLAP`-byte warm-up each window's
  state is identical to a full-file scan's state at `seg_start`, regardless of the
  state it started from. No match is missed and none double-counted (the
  `pos ≥ seg_start` guard dedups the overlap region). This is the parallelism
  Hyperscan structurally cannot exploit — its per-stream input scan is sequential.
- Overhead = `OVERLAP / seg_len` (e.g. 64/512 = 12.5%); tune `seg_len` for the
  occupancy/overhead knee. Expose it Tier-A (default derived from
  `max_pattern_len` and core count).

Geometry change: `claim → (seg_idx, rule_idx)`; derive `(file_idx, seg_start)`
from `seg_idx` via a segment table built next to `file_offsets`. The hit-dedup
already keys on `(rule, start, end)`; the `pos ≥ seg_start` guard makes overlap
emission idempotent.

### B. Fast per-lane inner loop (so each window is bandwidth-bound)
- **Byte-class compression — ALREADY DONE.** `dispatcher.rs:1438-1500`
  (`dfa_byte_scanner`) already folds each byte through the rule's 256→class map
  (`rule_meta` words 3/4 = `class_map_base`, `num_classes`) and indexes the
  compressed `state * num_classes + class` row, byte-for-byte identical to the
  dense table. Do NOT redo this. What remains:
  - **Haystack word reload**: the loop reloads `haystack[byte_pos/4]` every byte
    even though 4 consecutive bytes share one `u32` word — load the word once per
    4 bytes (or hoist + shift). ~4× fewer haystack loads, free.
  - **Transition residency**: `class_maps` / `transitions` are global storage
    reads (latency-bound). Stage the small per-rule class map (256 B) and hot DFA
    rows into shared/constant memory; reserve global for the precise large DFA on
    coarse-filter survivors (hierarchical).
- **Resident transition table**: small per-rule (or coarse-filter) DFAs go in
  shared/constant memory; only the precise large DFA touches global, and only on
  survivors of a coarse filter (hierarchical: cheap shared-mem filter DFA →
  precise DFA on the rare hit).
- **Vectorized input loads** (uint4 / 128-bit) + coalesced access across lanes;
  delete the per-byte 256-row workgroup copy in the reference kernel.
- **Warp-cooperative option**: 32 lanes scan 32 consecutive windows in lockstep
  for coalesced loads, or a single window via `__shfl` state hand-off.

## Order of attack
1. Land the canonical, artifact-free 8 MiB GPU-vs-Hyperscan harness + CPU↔GPU
   finding-parity oracle (the number every change is measured against).
2. (A) segment-×-rule work-item geometry + segment table + `pos ≥ seg_start`
   guard. This alone should move occupancy from ~rule_count to full and is the
   single biggest lever.
3. (B) byte-class-resident inner loop + vectorized loads + coarse/precise
   hierarchy until the kernel is bandwidth-bound.
4. Transfer: pinned/async upload overlapped with compute, catalog kept
   GPU-resident across files (upload once, scan many), CUDA-graph/persistent
   launch to kill per-dispatch latency.

Every step ships with the finding-parity oracle green (Law 6/9) and a measured
before/after on the 8 MiB harness. Do not claim OOM until measured.

---

## Session findings log (vyre greening for the keyhog segmentation API)

These are fixes landed while greening vyre at version 0.6.3 so keyhog can consume
the intra-file segmentation dispatch API. Each is verified by a real test.

### F1 — Megakernel silent under-claim → DRAINED (Law 10) [SUPERSEDED by F6]
`vyre-driver-wgpu/src/megakernel/dispatcher.rs`. The persistent claim loop used a
FIXED `claim_budget = ceil(queue_len / total_workers)`. When fewer GPU lanes are
resident than `total_workers`, the budget under-provisions and leaves
`segment_count * rule_count` work-items UNCLAIMED — `found < expected`,
`dropped_hits == 0`: a SILENT recall loss. The first fix was a fail-closed guard
(error if `HEAD < queue_len`); **F6 then superseded it with the real drain**, so
under-claim is now impossible by construction rather than merely surfaced. The
guard remains as a cheap drain-completion check (it now fires only on a
timed-out/incomplete drain, never on a complete one).
Proof: `tests/megakernel_segmentation_conservation_and_throughput.rs`.

### F6 — Megakernel queue DRAIN: every geometry conserves + 15× Hyperscan
The sound fix for F1's under-claim is to DRAIN the queue: each resident lane keeps
claiming until the queue is exhausted, independent of the resident-lane count. The
summary had recorded this as "needs a new while/break IR construct the foundation
lacks (only `loop_for`)" — that was WRONG. `vyre-foundation` already ships
`Node::forever(body)` (a persistent loop to `u32::MAX` whose body terminates via
`Node::Return`); `forever` + `if claim >= QUEUE_LEN { Return }` IS a bounded
persistent drain. `build_batch_program` now emits:
```
forever([ let claim = atomicAdd(HEAD, 1);
          if claim >= QUEUE_LEN { Return };
          execute_batch_claim_body(claim) ])
```
Safe because the drain loop is the kernel's only top-level statement (Return =
clean exit, no skipped finalization) and `execute_batch_claim_body` has no
workgroup barrier (no divergence deadlock). Overhead is one past-the-end
`atomicAdd` per resident lane — a rounding error, not a per-work-item cost.
`compute_claim_budget` is deleted; `worker_groups` now sizes only the dispatch
grid. Live-GPU result on an RTX 5090 (8 MiB, 8 rules, 137 planted markers):

      seg_len  wgroups  found  dropped     GB/s   vs HS   conserves?
         1024     1024    137        0   16.205  10.80x   conserves   (was 64/128!)
         1024     2048    137        0   23.017  15.34x   conserves
          512     1024    137        0   13.038   8.69x   conserves
          512     2048    137        0   16.971  11.31x   conserves
          256     2048    137        0   17.076  11.38x   conserves
          128     4096    137        0   20.178  13.45x   conserves

EVERY geometry now conserves all 137 markers with 0 dropped (the `1024×1024`
geometry that silently dropped half before now conserves), and the best beats the
Hyperscan 1.5 GB/s floor by 15.34×. Proof: the conservation+throughput oracle
(green on live GPU) + 219 wgpu lib unit tests.

### F2 — Dense-permutation index validation triple-duplicated → single-sourced
Three sites independently re-implemented "is this sorted index slice a dense
permutation of `0..N`": `resident_dispatch/helpers.rs::validate_dense_resident_indices`,
`cuda_graph_replay.rs::validate_cached_graph_slot_index_map`, and the readback
fusion cardinality check. Worse, two conflated *duplicate* `[0,0,2]` with *sparse*
`[0,2,3]` — both emitted a generic "not dense" message, so an operator could not
tell whether a slot was aliased or skipped. Fix: one backend-neutral classifier
`vyre_driver::ordering::classify_dense_permutation(sorted, expected_len) ->
Result<(), DensePermutationDefect>` (Duplicate / Sparse / LengthMismatch); each
call site formats its own context-specific remediation from the defect. One
algorithm, no fork.
Proof: `ordering.rs::classify_dense_permutation_*` (incl. a generated reference
oracle) + the resident/graph-replay behavioral tests that assert "duplicate" vs
"dense" vs "expected N" per defect.

### F3 — F-IR-34 runtime category check gutted into a silent no-op (Law 2/6/10)
`vyre-intrinsics/src/category_check.rs::check_opdef`. Doc + `#[should_panic]` test
both require it to PANIC when a `Category::Composite` (pure-IR) op carries a
backend-specific `primary_text` builder arm — the exact A/B/C drift F-IR-34
exists to catch. The body had been hollowed to `if category == Composite &&
primary_text.is_some() { return; }` — it silently ACCEPTED the violation and
discarded `id`. Fix: restore the panic with the actionable `Fix:` message.
Proof: `composite_with_primary_text_panics` (now green as `- should panic`).
NOT a coverage hole: F-IR-34 is actually enforced at BUILD time by
`vyre-intrinsics/build.rs`, which fails the build when a source block carries
`category: Category::Composite` together with `primary_text: Some(`. The runtime
`check_all_inventory_opdefs` walk is complementary defense-in-depth (a type-level
inventory walk vs. the build-time source scan), exercised by the test binary; it
has no production caller, which is acceptable since the build gate is the real
enforcement. The two are intentional redundancy, not a cancerous silent fork —
the fix only restored the runtime path to AGREE with the build gate instead of
silently disagreeing with it.

### F4 — Workspace versioned 0.6.2 → 0.6.3
All crate `Cargo.toml`s + root bumped (0 remaining 0.6.2 pins). Dropped dev-deps
re-added to vyre-libs / vyre-primitives / vyre-self-substrate so the lib-test
build resolves. Publish remains user-gated (ordered multi-crate crates.io push).

### F5 — Silent-fallback / fail-closed sweep across the workspace (Law 10)
Re-adding the dropped dev-deps (F4) let the full `cargo test --workspace --lib`
compile for the first time post-migration, surfacing 27 RED contract tests. The
dominant family was the silent-fallback cancer — checked `try_*` builders whose
infallible legacy wrappers swallowed the structured error and returned a
degenerate result with no signal. All fixed to fail closed (panic naming the
violated contract; callers needing structured handling use the `try_*` variant):

  - `vyre-primitives matching::dfa_compile::dfa_compile` — on a budget-overflowing
    pattern set returned `CompiledDfa::empty()`, the automaton that REJECTS ALL
    INPUT. Any scanner built on it (aho_corasick / classic_ac / literal_set /
    cooperative_dfa all call it) would silently drop every match. The single
    worst hole — a scanner-recall silent fallback. Now panics.
  - `vyre-primitives graph::*` legacy builders (`program_graph::read_only_buffers`,
    `vast_tree_walk::ast_walk_{pre,post}order`, `dominator_frontier`,
    `csr_forward_or_changed_parallel_batch{,_global_slot}`, `persistent_bfs_batch`,
    `csr_bidirectional::merge_frontier_or_changed`) — all returned an inert empty
    `Program::wrapped(Vec::new(), [1,1,1], Vec::new())` (or `false`) on an invalid
    launch shape: a GPU kernel that walks/scans nothing. The SAME inert-program
    literal was copy-pasted across 7 sites (duplicate AND silent). Now fail closed.
  - `vyre-self-substrate graph::vast_tree_walk::build_trusted_{pre,post}order_walk`
    — same inert fallback behind the "trusted = prevalidated" contract; now panics
    naming the broken prevalidation promise.
  - Poisoned-lock silent recovery (`unwrap_or_else(PoisonError::into_inner)` /
    `error.into_inner()`) handing back half-mutated state: `vyre-runtime`
    pipeline_cache shard lock, `vyre-libs` C-sema lazy_scope read/write locks, and
    `vyre-reference oob::Buffer` (9 copy-pasted sites — the CPU reference oracle,
    where a laundered poison means corrupt GOLDEN values the conform gate trusts).
    All fail closed; oob.rs collapsed to two shared `read_bytes`/`write_bytes`
    helpers + a new poison regression test. Repo-wide sweep confirms no other
    production `into_inner`-on-poison sites remain (the `#[cfg(test)]` disk-cache
    fixture and the deliberate reset-to-default emit-naga module cache are not it).
  - `vyre-libs c::lower::semantic_edges::resolved_semantic_edges` — resolved edges
    over a truncated VAST row buffer via a `u32::MAX` sentinel, silently dropping
    semantic/control-flow edges; now prevalidates row coverage (`assert_vast_rows_present`).
  - `vyre-libs c::parse::structure_statement::c11_statement_bounds` — a non-literal
    token count silently defaulted to 1 (`_ => 1`), mis-sizing build-time output
    buffers; now panics requiring a literal window count.

Every fix is proven by the pre-existing fail-fast / `should_panic` / poison
contract tests going green (they were the roadmap; not weakened — Law 6/9).

### OPEN — release-evidence gates red on stale/regressed committed artifacts
The remaining ~14 RED tests are all `vyre-self-substrate integration::{evidence,
quality,coverage,release}` gates that validate COMMITTED measurement/evidence
artifacts (e.g. `release/evidence/benchmarks/cuda-ptx-patterns.json`) and named
evidence ledgers ("Dataflow analysis DSE family", "docs matrix schema", "CUDA
family count", "resident CSR queue API test", "dataflow crate surface", "Dataflow
consumer adversarial coverage"). These are NOT code-logic bugs and NOT silent
fallbacks: the migration regenerated/partially-updated these artifacts such that
required evidence entries were zeroed or dropped — PROVEN: `cuda-ptx-patterns.json`
`cuda_ptx_source_cache_entries.p50` went **8 → 0** in git history, failing the
gate's `>= 1` requirement.

These artifacts are MEASURED evidence; hand-editing the JSON to pass is Law-9
fabrication and is refused. The correct remediation is to RE-RUN the evidence
pipeline that writes them — `xtask release-benchmarks` → `vyre-bench` case
`cuda_ptx_patterns` (and the sibling dataflow/docs/CUDA-family evidence
generators) — on a CUDA host so the committed evidence reflects current reality.
Scope note: vyre-self-substrate is OFF keyhog's consumption path (keyhog depends
on `vyre` / `vyre_libs` / `vyre-driver-wgpu` / `vyre-driver-cuda` / `vyre-runtime`,
NOT vyre-self-substrate), so this blocks a clean vyre *publish*, not keyhog's use
of the segmentation API. Publish is user-gated regardless.

### RESOLVED — 15/15 gates green (root cause: release-train.toml + validator coherence)
The ~14 reds collapsed once `release/release-train.toml` was completed (the
missing `weir` version + `weir_rc`/`combined_release_train` tags had been aborting
`release_train::data()`, skewing every version-derived generator). Down to **4**
CUDA-evidence gates, resolved as follows — each a COHERENCE sync (validator needle
vs. the actual committed-artifact key/value), NOT a test weakening:

- **release_gpu `family_count`** — `cuda-release-suite.json` carries 16 real,
  distinct `family_id` rows (ifds-witness, alias-reaching-def, callgraph-reachability,
  … the weir dataflow additions) at HEAD; validator pinned a stale `13`. Synced
  needle + returned literal + assert to `16` (16 ≥ 13 = stronger contract).
- **optimization `dataflow-*` families** — the dataflow families were namespaced
  under the weir crate split: `dataflow-dse` → `weir-dataflow-dse` (and `-licm`,
  `-loop-fusion`, `-loop-fission`) in `optimization-family-manifest.json`. Synced
  the four validator needles to the weir- prefix.
- **optimization corpus dataflow keys** — corpus emits `dataflow_analysis_cases` /
  `dataflow_analysis_optimized_cases`; validator needled the abbreviated
  `dataflow_cases` / `dataflow_optimized_cases`. Synced to the real keys.
- **cuda_ptx `cache_entries`** — a real Law-10 SILENT-DROP bug, not a measured
  regression. The artifact showed cache `0/0/0` with `samples:1` while sibling
  metrics had `samples:30` — proof the value came from a different path. Root
  cause: `cuda_ptx_source_cache_{entries,hits,misses}` were absent from the
  `custom_metric_key` allow-list in `vyre-bench/src/runner/execute/metric_keys.rs`,
  so `collect_custom_metrics` SILENTLY DROPPED the codegen corpus's own emitted
  values (entries=8 unique kernels). The cuda driver's `backend_metric_snapshot`
  then filled the key via `or_insert_with` with `0` — correct for the driver's
  real module cache (this codegen-only case calls `emit_with_target` and never
  dispatches a module) but wrong for the metric the gate validates. Fix: register
  the three keys (so the corpus's 30-sample values are collected and the driver's
  0 no longer clobbers them) + a `custom_metric_key_keeps_cuda_ptx_source_cache_visible`
  regression test. The artifact is regenerated by the single-case bench
  (`vyre-bench run --suite release --case cuda.ptx.patterns.release.corpus
  --backend cuda --output …`), which mirrors `run_named_benchmark` but bypasses the
  workload-matrix gate — so it does NOT require the unrelated INT4 100x perf case.

The three suite-merge recorders (`record_{required,observed}_metric_percentile`)
were audited for Law-10: both push a loud blocker on a missing metric — no silent
default-to-zero. Publish remains user-gated.

### OPEN (coverage note, NOT a confirmed regression) — INT4 100x
A full `xtask release-benchmarks --backend cuda` run observed
`nn.linear_4bit_affine_grouped.1m` at **96.55x** vs its `cpu_sota_100x` contract.
This was a SINGLE sample taken while a heavy `cargo` build contended the CPU; the
ratio is `T_cpu_oracle / T_gpu`, and CPU contention inflates `T_cpu_oracle` → the
true idle ratio is ≤ 96.55x. So this MIGHT be a real ~3.5% shortfall — but one
contended data point is weak; it needs a clean isolated measurement
(`vyre-bench run --suite release --case nn.linear_4bit_affine_grouped.1m
--backend cuda --measured-samples 30` on an idle host) before claiming a
regression. It does NOT block any of the 4 evidence gates (gate 4 reads the
committed cuda-release-suite; gate 1 uses the single-case ptx bench). It only
fails a FULL release-benchmarks run and gates the optimization-manifest emission
via `should_write_optimization_manifest(workload_failures_empty=…)`. Kernel:
`vyre-libs/src/nn/linear/inner/linear_4bit.rs` (256-wide grouped INT4, 32
lanes/output, 8 outputs/workgroup).
