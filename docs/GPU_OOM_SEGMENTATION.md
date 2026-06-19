# GPU-beats-Hyperscan-by-OOM: the segmentation thesis (code-grounded)

Goal: on a single 8 MiB scan, the wgpu megakernel must beat Hyperscan (~1.5 GB/s,
~5.6 ms) by an order of magnitude. RTX 5090 ≈ 1.7 TB/s, so a fully
bandwidth-bound scan of 8 MiB is ~5 µs — a ~1000× ceiling physically exists. The
gap today is **parallel decomposition + per-lane inner-loop cost**, not physics.

## STATUS: GOAL MET — per-rule segmentation path (verified 2026-06-18, RTX 5090)
The intra-file `(segment × rule)` geometry is wired end to end (host `segments`
table at binding 9 + `seg_idx = claim/rule_count` decode + warm-up prefix
`[scan_start, emit_start)` + emit-guard `byte_pos >= emit_start` mirroring the
`segmentation.rs` CPU oracle) and the drain loop (`forever` + `claim>=QUEUE_LEN`)
removed the fixed-budget under-claim silent-drop. Re-measured on the 5090 via
`cargo test -p vyre-driver-wgpu --features megakernel-batch,wgpu --test
megakernel_segmentation_conservation_and_throughput -- --ignored --nocapture`
(8 MiB, 8 rules, 137 planted markers):

| seg_len | wgroups | found/137 | dropped | GB/s   | vs HS  |
|---------|---------|-----------|---------|--------|--------|
| 1024    | 1024    | 137       | 0       | 13.594 | 9.06×  |
| 1024    | 2048    | 137       | 0       | 18.519 | 12.35× |
| 512     | 1024    | 137       | 0       | 13.425 | 8.95×  |
| 512     | 2048    | 137       | 0       | 17.752 | 11.83× |
| 256     | 2048    | 137       | 0       | 18.247 | 12.16× |
| 128     | 4096    | 137       | 0       | 18.686 | 12.46× |

Best conserving geometry **18.686 GB/s = 12.46× the 1.5 GB/s Hyperscan floor**;
every geometry conserves all markers with 0 dropped. The pre-segmentation
baseline was 4.1× SLOWER at 17% occupancy — segmentation flipped a 4.1× loss
into a 12.46× win.

## STATUS: COMBINED-AC path BUILT + GPU-verified (2026-06-18, RTX 5090)
The combined-Aho-Corasick path — one dense automaton, `queue_len =
segment_count` (NO `rule_count` multiplier), per-state multi-emit via
`output_offsets`/`output_records` CSR — is wired end to end and verified:
`build_combined_batch_program` + `CombinedBatch` (host upload, takes raw
flattened automaton arrays; `classic_ac_compile` stays in the caller because
vyre-libs is ABOVE vyre-driver-wgpu) + `CombinedDispatcher`. Verified on the
5090 via `cargo test -p vyre-driver-wgpu --features megakernel-batch,wgpu
--test megakernel_combined_scan -- --ignored --nocapture` — a DIFFERENTIAL test
whose ground truth is `classic_ac_scan` over the whole 8 MiB buffer (32-pattern
catalog, 2115 oracle matches):

| seg_len | found/2115 | dropped | GB/s   | vs HS  |
|---------|------------|---------|--------|--------|
| whole   | 2115       | 0       | 0.009  | 0.01×  |
| 65536   | 2115       | 0       | 0.873  | 0.58×  |
| 16384   | 2115       | 0       | 3.144  | 2.10×  |
| 4096    | 2115       | 0       | 10.202 | 6.80×  |
| 1024    | 2115       | 0       | 19.266 | 12.84× |
| 512     | 2115       | 0       | 20.565 | 13.71× |

Every geometry reproduces the oracle hit set EXACTLY (no miss / dup /
fabrication, 0 dropped). Best conserving geometry **20.565 GB/s = 13.71× the
Hyperscan floor** with 32 patterns in ONE automaton — the per-rule path would
scan every byte 32× (`queue_len = segment_count * 32`). The whole-file→512
curve (0.009 → 20.565 GB/s) shows segmentation, not raw compute, saturates the
device. Commits: `c2b82986b7` (kernel IR), `46f6b93087` (host + test).

Direct per-rule-vs-combined head-to-head (DONE 2026-06-18, commit c4df9300d8,
`tests/megakernel_combined_vs_perrule.rs`): same 8 MiB buffer, same 64
single-byte patterns, both conserve all 2048 oracle matches exactly — per-rule
best **3.913 GB/s** (queue = segment_count × 64), combined best **18.009 GB/s**
(queue = segment_count) ⇒ **combined 4.60× the per-rule path**. The 4.60× (not
64×) gap is because per-rule is partly occupancy/memory-bound, not purely
compute-bound; the advantage is real, conserving, and widens with catalog size.

REMAINING combined-AC depth (not yet built): byte-class compression of the
(larger) combined transition table (shrink each row from 256 cols to
`num_classes`, the per-rule `class_maps` machinery — LOSSLESS, fewer global
loads); the literal/regex split (regex rules with no single required literal
stay on a per-rule path or literal-factor prefilter — bound and LOG the split,
never silently drop them). The per-rule path remains the win for small catalogs;
see "Concrete kernel plan" below for the build record.

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

### RESOLVED (was a contention artifact, NOT a regression) — INT4 100x
A full `xtask release-benchmarks --backend cuda` run observed
`nn.linear_4bit_affine_grouped.1m` at **96.55x** vs its `cpu_sota_100x` contract,
while a heavy `cargo` build contended the host. A clean isolated re-measure on the
idle host (`vyre-bench run --suite release --case nn.linear_4bit_affine_grouped.1m
--backend cuda --measured-samples 30`) gives **116.8x** (pass) — GPU p50 6144 ns,
CPU oracle p50 717312 ns. The kernel is fine; the dip was a measurement artifact:
the GPU time is tiny (~6 µs), so host-side contention inflated the GPU-side
measurement far more than the large (~717 µs) CPU oracle, depressing the ratio
below 100x. (My earlier "contention inflates the ratio" reasoning was wrong — for
a microsecond-scale GPU kernel, host contention inflates the *denominator* more.)
Lesson: budget-margin perf cases must be measured on an idle host; a contended
full-suite run is not authoritative for sub-10 µs GPU kernels. Kernel:
`vyre-libs/src/nn/linear/inner/linear_4bit.rs` (256-wide grouped INT4, 32
lanes/output, 8 outputs/workgroup).

---

## END-TO-END RESULT (the thesis MEASURED on the real keyhog catalog)

The segmentation API shipped in **vyre 0.6.3** (`FileBatch::set_segmentation` +
`segmentation::catalog_sync_overlap`, kernel decode + emit-guard in
`dispatcher.rs`). keyhog 0.6.3 now drives it from `MegakernelCatalog::scan`:
compute the catalog sync-distance overlap once (cached; `None` ⇒ an unbounded
rule ⇒ whole-file, surfaced loudly), pick a device-saturating `seg_len`
(`choose_seg_len`), `set_segmentation`, dispatch. Measured on an **RTX 5090**,
`gpu_vs_hs_8mib` (902 detectors → **3124** GPU rules, 8 MiB, sparse real hits):

| backend | wall (median) | notes |
|---|---|---|
| SimdCpu (Hyperscan) | **255.95 ms** | literal prefilter phase-1 ~2 ms + shared phase-2 ~254 ms |
| Gpu (vyre megakernel) | **807.09 ms** | phase-1 dispatch 489 ms + shared phase-2 254 ms + overhead |

**Ratio 3.15× SLOWER (was 4.1×). Recall parity held (254 == 254).**

The thesis was HALF right. Segmentation did exactly what it promised — `overlap`
came back **Some(42)** on the real catalog (the `BudgetExceeded`/`None` fear was
unfounded), 21 segments × 3124 rules = 65604 work-items, and **occupancy went
17% → 100%** (`occupancy_bps=10000`). But that only moved the ratio 4.1× → 3.15×,
not past 1.0×. **Occupancy was never the gating factor for a literal-rich
catalog.** Phase-1 telemetry: the GPU brute-forces *every rule over every byte*
(3124 × 8 MiB ≈ 24 GB of DFA stepping → 489 ms, **memory-latency-bound on the
per-rule transition tables**; more warps just contend on that bandwidth).
Hyperscan's phase-1 is ~2 ms because its literal prefilter (FDR/Teddy) runs the
expensive DFA confirm ONLY at the few candidate offsets a literal hit. The shared
CPU phase-2 (~254 ms) then dominates both totals.

**Corrected order of attack.** Part B ("fast inner loop") is not enough either —
the real lever is a **GPU-side multi-pattern literal prefilter** (Teddy/FDR
equivalent) so per-rule DFA work happens only at candidate positions, OR a single
combined automaton (one pass over the input, not 3124). Size won't save it: both
phase-1s are linear in bytes, so the ~244× phase-1 work ratio is roughly
size-invariant; dispatch overhead amortizes but the brute-force multiplier does
not. The GPU megakernel's genuine niche is **low-literal / regex-heavy catalogs**
where Hyperscan's prefilter is also weak (and very large inputs). Segmentation is
a necessary, recall-preserving, now-landed foundation a prefilter builds on — not,
by itself, a Hyperscan-beater on this workload. (See memory
`gpu-megakernel-prefilter-bound`; keyhog wiring commit `d23e70af`.)

---

## COMBINED-AUTOMATON path (the real lever after segmentation) — design, grounded 2026-06-17

> STATUS CORRECTION (2026-06-17): the combined-automaton lever below is **already
> implemented in keyhog**, not pending. The 3.15x in the END-TO-END section above
> is STALE (pre-grouping). keyhog `megakernel.rs` dedups identical literals
> (3124->1643) then GROUPS the 1643 unique literals into 32 COMBINED multi-pattern
> DFAs (`GPU_LITERAL_RULE_GROUPS=32`, each `build_regex_dfa_unanchored(&[group
> literals])`) scanned ONCE: kernel_wall 278ms->7-9ms (~35x), ratio 2.29x->**1.15x**,
> recall parity 254==254 (keyhog commit fb466fc7). So "walk once, not 3124 times"
> is DONE. The remaining 1.15x gap is the SHARED phase2 floor (generic/fallback/
> preprocess, ~252ms, run by BOTH backends), beatable only by keyhog's phase2-skip
> (fold the GPU-confirmed firings into one anchored extraction pass) — a KEYHOG
> change, not a vyre kernel change. See memory `gpu-megakernel-prefilter-bound`.
> The vyre role is the segmented combined-scan PRIMITIVE + the soundness oracle
> below; the design text that follows documents that primitive (it is what licensed
> keyhog's grouped scan), NOT an un-started TODO.

The END-TO-END finding above proved segmentation (occupancy 17%->100%) does NOT
beat Hyperscan on a literal-rich 3124-rule catalog: GPU phase-1 = 489 ms because
the geometry is `(seg, rule)` — every rule's per-rule DFA is walked over every
byte, so 8 MiB is scanned ~3124 times (~24 GB of latency-bound transition reads).
The lever is to walk the input ONCE with a COMBINED automaton, not 3124 times
(landed in keyhog via literal grouping — see the status correction above).

### What already exists (do NOT rebuild — NO DUPLICATION)
- `vyre_libs::scan::classic_ac::classic_ac_compile(&[&[u8]]) -> ClassicAcAutomaton`
  builds ONE multi-pattern Aho-Corasick `CompiledDfa` over a pattern SET:
  dense `transitions[state*256+byte]`, `output_offsets[state]`, and a flat
  `output_records` array = the SET of pattern_ids accepting at each state
  (incl. via failure links). `classic_ac_scan` is the linear CPU oracle emitting
  every `(pattern_id, end)`; `classic_ac_program` is a vyre GPU-IR program that
  multi-emits via `output_records`. (`vyre-primitives::matching::dfa_compile`.)
- The megakernel segmentation geometry, `plan_segments` window tiling, overlap
  warm-up, and emit-guard are landed + proven (F6 + segmentation.rs proptests).

### What was MISSING, now landed (CPU soundness foundation)
`vyre-driver-wgpu/src/megakernel/segmentation.rs` test module gains
`combined_segmented_scan` + `combined_segmented_equals_linear_classic_ac_scan`
(proptest) + a known-case test: the PRODUCTION combined AC (`classic_ac_compile`,
dense transitions + `output_records` multi-emit), scanned in `plan_segments`
windows with `overlap >= max_pattern_len` and the `i >= emit_start` guard,
produces EXACTLY the linear `classic_ac_scan` `(pattern_id, end)` set, for any
seg_len. This is the soundness contract the combined kernel mirrors — analogous
to the existing per-rule `segmented_scan == dense_scan` oracle, but on the real
`CompiledDfa`/`output_records` automaton the kernel will actually run.
(`end` = 0-based byte index, classic_ac_scan's convention — NOT the model
oracle's `i+1`.)

### The build (self-contained in vyre — NO keyhog dependency)
KEY ARCHITECTURAL FINDING: rules enter the megakernel as PRE-BUILT PER-RULE DFAs
(`vyre_runtime::megakernel::rule_catalog::pack_rule_catalog(&[BatchRuleProgram])`,
each carrying `transitions/accept/state_count`). There is NO pattern-level entry.
A combined automaton CANNOT be reconstructed from minimized per-rule DFAs (the
patterns are gone; a 3124-way DFA product is astronomically large). So the
combined path needs the PATTERNS. That is NOT a keyhog blocker: vyre ships a
self-contained combined-scan capability that takes patterns, builds the combined
DFA via `classic_ac_compile`, and scans once. keyhog adopts later by handing vyre
its literal pattern set (or per-rule required-literal factors) instead of, or
alongside, the per-rule DFAs.

Concrete kernel plan (all in vyre-driver-wgpu — does NOT touch vyre-runtime's
per-rule `rule_catalog`, so no collision with the cycle-3 swarm):
1. `CombinedBatch::upload(files, patterns: &[&[u8]], hit_capacity)` — build one
   `classic_ac_compile(patterns)`; flatten `transitions` (state_count*256),
   `output_offsets` (state_count+1), `output_records` into device buffers;
   `overlap = dfa.max_pattern_len`; reuse `plan_segments`/`segment_table`.
2. Kernel geometry: `(seg)` only — `queue_len = segment_count` (NOT
   `* rule_count`). The persistent drain loop (F6 `forever` + `claim>=QUEUE_LEN`)
   is unchanged.
3. `combined_dfa_byte_scanner`: state=0; for byte_pos in [scan_start, emit_end):
   `state = transitions[state*256 + byte]`; if `byte_pos >= emit_start`:
   `for out_idx in output_offsets[state]..output_offsets[state+1]:
   emit HitRecord{ file_idx, rule_idx = output_records[out_idx], layer_idx,
   match_offset = byte_pos }`. HitRecord needs NO new field — `rule_idx` carries
   the pattern_id directly (pattern_id == rule_idx when patterns are the rules;
   else a pattern_id->rule_idx map applied host-side at readback).
4. Optional: byte-class compress the combined transitions (the per-rule path's
   `class_maps` machinery) to shrink the (larger) combined table; dense first.
5. GPU measurement: extend the throughput oracle with a MANY-pattern catalog
   (hundreds of literals) comparing `(seg,rule)` per-rule vs `(seg)` combined
   phase-1 time on the 5090. The per-rule multiplier disappears: combined does
   ONE transition read per byte regardless of pattern count.

Regex rules (no single required literal) cannot join the literal AC and must
stay on a per-rule path or a literal-factor prefilter (Hyperscan's design); the
combined path is the win for the literal-coverable majority. Bound the split and
log it (Law 10) — never silently drop the regex rules from the combined pass.

## Combined-AC build: the code-grounded integration seam (next, 2026-06-18)
The per-rule path above is the verified OOM win for small catalogs; the
combined-AC path removes the `rule_count` queue multiplier so a many-pattern
catalog scans in ONE transition read per byte. Foundation is already green:
- CPU oracle: `segmentation.rs::combined_segmented_scan` +
  `combined_segmented_equals_linear_classic_ac_scan` (proptest) prove the
  combined `classic_ac_compile` automaton, scanned in `plan_segments` windows
  with `overlap >= max_pattern_len` and the `byte_pos >= emit_start` guard,
  equals the linear `classic_ac_scan` `(pattern_id, end)` set for any seg_len.
- Combined automaton + buffer surface already exist in
  `vyre-libs/src/scan/classic_ac.rs`: `classic_ac_compile(patterns)` →
  `ClassicAcAutomaton { dfa: { transitions[state*256+byte], output_offsets
  [state_count+1], output_records[len], state_count } }`. (`classic_ac_program`
  there is a TEST-ONLY O(n^2) per-position reference, NOT the segmented kernel.)

MAXIMAL-REUSE seam (do NOT fork the FileBatch machinery):
1. Host `CombinedBatch::upload(device, files, patterns, hit_capacity)` mirrors
   `FileBatch::upload` EXACTLY for haystack / file_offsets / file_metadata /
   segments / queue_state / hit_ring, with `overlap = max_pattern_len` and
   `queue_len = segment_count` (i.e. `rule_count = 1`: one work item per
   segment). It REPLACES the four per-rule automaton buffers
   (`class_maps`/`rule_meta`/`transitions`/`accept`) with three combined ones:
   `transitions` (state_count*256), `output_offsets` (state_count+1),
   `output_records` (len). Keep `RULE_COUNT` queue word = 1 so the existing
   `seg_idx = claim / rule_count`, drain loop, and DONE/HEAD accounting are
   byte-for-byte reused.
2. `dispatcher.rs::build_combined_batch_program` = `build_batch_program` with
   `batch_program_buffers` swapped for the combined buffer list and
   `execute_batch_claim_body` → `execute_combined_claim_body` (same segment-row
   decode; drops rule_base/transition_base/accept_base/class_map; binds
   `out_begin/out_end` from output_offsets).
3. `combined_dfa_byte_scanner`: `state=0; for byte_pos in [scan_start, emit_end):
   state = transitions[state*256 + byte]; if byte_pos >= emit_start { for out_idx
   in output_offsets[state]..output_offsets[state+1]: emit HitRecord{ file_idx,
   rule_idx = output_records[out_idx], match_offset = byte_pos } }`. Reuses the
   exact warm-up + emit-guard the per-rule scanner proved.
4. Verify: (a) naga emit of the combined program compiles to valid WGSL
   (emit-level unit test, no GPU); (b) extend the throughput oracle with a
   many-pattern catalog (hundreds of literals) and assert the combined `(seg)`
   conserves the planted markers AND beats the per-rule `(seg,rule)` phase-1
   time — the per-rule multiplier should vanish (one transition read per byte).
5. Regex rules with no single required literal cannot join the literal AC: keep
   them on the per-rule path or a literal-factor prefilter; bound the split and
   LOG it (Law 10) — never silently drop regex rules from the combined pass.
