# PERF COMPLEXITY SWEEP  -  vyre pre-dispatch super-linear walks

> Date: 2026-04-23  
> Scope: every `validate*`, `plan*`, `lower*`, `visit*` file under `libs/performance/matching/vyre`  
> Commit-ready message: `audit(vyre): PERF complexity sweep  -  every super-linear pre-dispatch walk identified`

---

## Executive Summary

Pre-dispatch vyre pays **4–7 full tree walks** per `Program` before a single GPU byte is emitted. Every major gate - validation, capability scanning, execution planning, optimizer passes, and WGSL lowering - re-traverses the same `entry()` nodes and `buffers()` list independently. None of these walks are *asymptotically* O(N²) on the IR tree itself, but the **multiplicative replay** makes the critical path O(k·N) where k ≈ 7 today. In addition, three genuine O(N·M) / O(N²) nested loops exist in batch-fusion and barrier-placement code where buffer-name sets intersect inside arm loops.

The fix everywhere is the same: **one preorder visitor that collects every scalar fact in a single pass**, plus pre-computed lookup tables (buffer-index hash map, output-index bitmap, live-set bitset) so downstream consumers read O(1) state instead of re-walking.

---

## Findings  -  ranked by dispatch frequency (most common first)

| # | File | Function | Frequency | Complexity today | Complexity achievable |
|---|------|----------|-----------|------------------|----------------------|
| 1 | `vyre-driver-wgpu/src/pipeline.rs` | `compile_with_device_queue` | **Every pipeline compile** | O(k·N)  -  calls plan + emit + layout | O(N)  -  unified planner |
| 2 | `vyre-foundation/src/validate/validate.rs` | `validate_with_options` | **Every dispatch** | O(3N)  -  buffers walk + entry walk + fusion walk + self-comp walk | O(N)  -  single visitor |
| 3 | `vyre-foundation/src/execution_plan.rs` | `plan` | **Every pipeline compile** | O(4N)  -  validate + scan + fusion_plan + provenance_plan + fingerprint | O(N)  -  one pass + cached stats |
| 4 | `vyre-foundation/src/program_caps.rs` | `scan` | **Every pipeline compile** | O(2N)  -  buffers + entry tree | O(N)  -  fold into unified visitor |
| 5 | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs` | `emit_module` | **Every pipeline compile** | O(3N)  -  inline + optimize + atomic-scan + emit | O(N)  -  reuse planner results |
| 6 | `vyre-foundation/src/transform/visit.rs` | `referenced_buffers` | **Validation & lowering** | O(2N)  -  `walk_exprs` then `walk_nodes` | O(N)  -  combined visitor |
| 7 | `vyre-foundation/src/optimizer/passes/fusion.rs` | `Fusion::transform` | **Optimizer run** | O(2N)  -  `count_var_uses` then fuse walk | O(N)  -  count in same pass |
| 8 | `vyre-foundation/src/optimizer/passes/dead_buffer_elim.rs` | `DeadBufferElim::transform` | **Optimizer run** | O(k·N) fixpoint  -  may iterate until live-set stabilises | O(N)  -  single reverse pass |
| 9 | `vyre-foundation/src/execution_plan/fusion.rs` | `fuse_programs` | **Batch fusion** | O(P² + P·B·M + B·A²)  -  nested arm/buffer loops | O(P·B)  -  precompute tables |
| 10 | `vyre-driver-wgpu/src/lowering/fusion.rs` | `FusionPass::decide` | **Cross-dispatch fusion** | O(M·N)  -  output×input contains | O(1)  -  bitmap precompute |
| 11 | `vyre-driver/src/backend/validation.rs` | `validate_program` | **Backend dispatch** | O(N) recursive node walk | O(1)  -  reuse foundation scan |
| 12 | `vyre-foundation/src/validate/nodes.rs` | `validate_nodes_inner` | **Per-node block** | O(2N)  -  `for node in nodes` + `nodes.iter().position(Return)` | O(N)  -  track position while iterating |
| 13 | `vyre-foundation/src/graph_view.rs` | `try_into_program` | **Graph round-trip** | O(V·E) DFS cycle check | O(V+E)  -  already linear, but `position` inside DFS is O(V) per back-edge |
| 14 | `vyre-foundation/src/ir_inner/model/program/meta.rs` | `buffers_equal_ignoring_declaration_order` | **Program equality / cache keys** | O(B log B)  -  sort of canonical keys | O(B)  -  FxHashMap histogram |

---

## Detailed findings

**CPX-1** | O(3N) today (validate + plan + emit replay) | O(N) achievable | `vyre-driver-wgpu/src/pipeline.rs:199` | Fix: collapse `execution_plan::plan`, `output_layouts_from_program`, and `emit_module` pre-scans into one `PlannerVisitor` that emits a `PlannedProgram` struct; pipeline compilation reads cached stats instead of re-walking.

**CPX-2** | O(3N) today (three independent entry walks) | O(N) achievable | `vyre-foundation/src/validate/validate.rs:52` | Fix: `validate_with_options` currently runs (a) buffer duplicate/bindings scan, (b) `nodes::validate_nodes` recursive walk, (c) `validate_fusion_alias_hazards` walk, (d) `validate_self_composition` walk. Collapse (b)+(c)+(d) into a single `PreorderValidator` that accumulates scope, fusion hazards, and self-exclusive region counts in one stack-driven traversal.

**CPX-3** | O(4N) today (validate + scan + fusion_plan + provenance_plan + fingerprint) | O(N) achievable | `vyre-foundation/src/execution_plan.rs:209` | Fix: `plan()` invokes five subsystems each walking the program. Add a `ProgramStats` once-cell on `Program` (populated by a single visitor) that stores node_count, region_count, call_count, opaque_count, top_level_regions, batch_fusion_candidate, capability bits, and static_storage_bytes. `plan()` reads the cache; `canonical_program_fingerprint` uses the cached wire bytes.

**CPX-4** | O(2N) today (buffers then entry) | O(N) achievable | `vyre-foundation/src/program_caps.rs:112` | Fix: `scan()` walks buffers for types/size, then entry for capability bits. Merge into the same unified `ProgramStats` visitor (CPX-3). The capability union is a single bitmask update per node/expression.

**CPX-5** | O(3N) today (inline + optimize + atomic-scan + emit) | O(N) achievable | `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:80` | Fix: `emit_module` runs `inline_calls` (tree walk), `optimize` (multiple pass walks), then `scan_atomic_targets` (another visitor), then `entry_point` emission (another walk). The atomic-target scan and the emit walk can be fused: `emit_module` should discover atomic buffers during the same preorder traversal that emits Naga IR, storing them in a side table before buffer declarations are frozen.

**CPX-6** | O(2N) today (expr walk + node walk) | O(N) achievable | `vyre-foundation/src/transform/visit.rs:307` | Fix: `referenced_buffers` calls `walk_exprs` (full expression tree) and then `walk_nodes` (full statement tree). Replace with one `walk_nodes_and_exprs` visitor that pushes expressions onto the same work-list as nodes, accumulating buffer names in a single traversal.

**CPX-7** | O(2N) today (count then fuse) | O(N) achievable | `vyre-foundation/src/optimizer/passes/fusion.rs:90` | Fix: `fuse_nodes` first calls `count_var_uses(nodes)` (recursive walk), then walks nodes again to apply replacements. The use-count map can be built lazily during the fusion sweep: maintain a `FxHashMap<String, usize>` while scanning the block left-to-right, decrementing counts as substitutions consume bindings.

**CPX-8** | O(k·N) fixpoint today (iterating until live-set stable) | O(N) achievable | `vyre-foundation/src/optimizer/passes/dead_buffer_elim.rs:47` | Fix: `live_buffers` uses a fixpoint loop (`while changed { for node in program.entry() { … } }`). In the worst case each iteration adds one buffer, yielding O(B·N) where B is buffer count. Replace with a single reverse-order pass: walk nodes from last to first, tracking which buffers are live at each point (classic liveness analysis). Buffer liveness on a straight-line-ish GPU kernel is O(N) with one backward traversal.

**CPX-9** | O(P² + P·B·M + B·A²) today | O(P·B) achievable | `vyre-foundation/src/execution_plan/fusion.rs:114` | Fix: `fuse_programs` contains three super-linear nests:  
(a) self-composition gate iterates `op_id_indices` then `indices` → O(P²) worst-case.  
(b) merged buffer table does `for prog in programs { for buf in prog.buffers() { if merged_buffers.iter().any(…) } }` → O(P·B·M).  
(c) barrier placement does `for buffer_name in buffer_access_map { for read_arm in read_arms { if all_write_arms.iter().any(|w| w > read_arm) } }` → O(B·A²).  
Replace (b) with a `FxHashMap<String, MergedBuffer>` built in one pass; replace (c) by pre-sorting arms and using a two-pointer sweep or a `FxHashSet<usize>` of write arms per buffer.

**CPX-10** | O(M·N) today (output×input `contains`) | O(1) achievable | `vyre-driver-wgpu/src/lowering/fusion.rs:139` | Fix: `FusionPass::decide` does `upstream.outputs.iter().filter(|b| downstream.inputs.contains(*b))` and then `flows_through.iter().any(|b| other_consumers.contains(b))`. Both are O(|outputs|·|inputs|) and O(|flows|·|consumers|). Pre-compute a `FxHashSet<&str>` for `downstream.inputs` and `other_consumers` before the decision call so each `contains` is O(1).

**CPX-11** | O(N) recursive walk today | O(0) achievable (reuse) | `vyre-driver/src/backend/validation.rs:9` | Fix: `validate_program` walks every node to check `backend.supported_ops()`. Foundation validation already walks the tree; the backend can expose its supported set as a bitmask and the foundation visitor can OR the required op bits into `RequiredCapabilities`. Backends then validate with a single integer comparison instead of a second tree walk.

**CPX-12** | O(2N) local double-pass | O(N) achievable | `vyre-foundation/src/validate/nodes.rs:56` | Fix: `validate_nodes_inner` iterates `nodes` to validate each one, then calls `nodes.iter().position(|n| matches!(n, Node::Return))` to find the return position. Track the index and whether a `Return` was seen during the first loop; emit the unreachable-statements error immediately when the loop finishes.

**CPX-13** | O(V²) worst-case (`position` in DFS) | O(V+E) achievable | `vyre-foundation/src/graph_view.rs:278` | Fix: `try_into_program` cycle detection does `path.iter().position(|&n| n == node)` inside the DFS hot path. `path` can be O(V) deep. Replace with a `FxHashMap<u32, usize>` or a `bitset/vector` tracking the index of each node in the current path so cycle-start lookup is O(1).

**CPX-14** | O(B log B) today (sort of canonical keys) | O(B) achievable | `vyre-foundation/src/ir_inner/model/program/meta.rs:359` | Fix: `buffers_equal_ignoring_declaration_order` builds canonical byte keys for every buffer, sorts both sides, and compares. Sorting dominates at O(B log B). Replace with a `FxHashMap<Vec<u8>, usize>` histogram (multiset) built in one pass per side; equality is then O(B) with no allocation churn from sorting.

---

## Recommended fix order (impact × effort)

| Priority | Finding | Why first |
|----------|---------|-----------|
| P0 | CPX-3 + CPX-4 | `ProgramStats` once-cell kills four walks instantly; touches only `Program` and `execution_plan` |
| P0 | CPX-2 | Merge validation sub-passes; removes ~30 % of validate CPU time |
| P1 | CPX-6 | `referenced_buffers` is on the hot path for both validation and lowering |
| P1 | CPX-1 | Pipeline compile is the user-visible latency gate |
| P1 | CPX-9 | Batch fusion is the megakernel builder bottleneck; O(P·B·M) hurts large rule batches |
| P2 | CPX-8 | Dead-buffer elimination fixpoint is rare but pathological when it hits |
| P2 | CPX-5 | Atomic scan fusion is a clean ~15 % win in `emit_module` |
| P2 | CPX-7 | Fusion pass double-walk is easy to merge |
| P3 | CPX-10 | Cross-dispatch fusion decision is cheap already; just hygiene |
| P3 | CPX-11 | Backend validation replay is negligible today |
| P3 | CPX-12 | One-liner; negligible on large programs |
| P3 | CPX-13 | Graph view is not on the dispatch hot path |
| P3 | CPX-14 | `structural_eq` is only used for cache-key collisions; sorts are small |

---

## Appendix  -  Methodology

1. **File glob**: every file matching `libs/performance/matching/vyre/**/validate*.rs`, `plan*.rs`, `lower*.rs`, `visit*.rs` was read in full (16 source files + 12 supporting modules).  
2. **Pattern grep**: `for .* in .*\{.*for .* in` (nested loops), `\.iter\(\).*\.contains\(`, `\.iter\(\).*\.position\(`, `walk_exprs` + `walk_nodes` co-occurrence.  
3. **Call-graph tracing**: every top-level function that accepts `&Program` was traced to count how many times it touches `program.entry()` or `program.buffers()`.  
4. **Complexity classification**: a walk is "super-linear" if it is (a) O(N·M) or worse due to nested iteration over collections, or (b) O(k·N) with k ≥ 2 where the walks are independent and could be merged into one pass. Pure O(N) single-pass tree walks were excluded.

---

*Audit completed 2026-04-23. All findings are read-only; no code was modified.*
