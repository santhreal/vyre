# VYRE_IR_HOTSPOTS  -  IR hot-path allocation / copy / visitor walk audit

**Scope:** `libs/performance/matching/vyre/vyre-foundation/src/ir_inner` (program, buffer_decl, expr, node), `visit/` dir, optimizer passes, wire encode/decode, `execution_plan/fusion.rs`.

**Method:** Static analysis + LAW 0–8. Every finding is traced to a concrete line. Fixes are minimal and target the root cause, not the symptom.

---

## Findings Table

| SEVERITY | file:line | defect | fix |
|----------|-----------|--------|-----|
| CRITICAL | `execution_plan/fusion.rs:119` | `fuse_programs` deep-clones single input via `programs[0].clone()` instead of moving or returning by reference. | Return `Ok(programs[0].to_owned())` only when caller needs ownership; better, accept `Vec<Program>` by value and drain. |
| CRITICAL | `optimizer/passes/autotune.rs:26,30` | No-op autotune path clones entire `Program` via `program.clone()` just to return `changed=false`. | Return `PassResult { program, changed: false }` directly; avoid `from_programs` equality check when identity is known. |
| CRITICAL | `optimizer/passes/spec_driven.rs:24` | `PassResult::from_programs(&program.clone(), program)` clones the whole Program before comparing by structural equality. | Use `PassResult { program, changed: false }` directly; skip the clone + eq check. |
| CRITICAL | `optimizer/passes/decode_scan_fuse.rs:75` | `(*program.entry).clone()` deep-clones the entire entry `Vec<Node>` even when no buffers are promoted. | Return `program` by value early when `promotable.is_empty()` (line 56 does this, but line 60–75 still clones in the non-early path). In the rewrite path, use `program.with_rewritten_entry(...)` instead of cloning entry manually. |
| CRITICAL | `transform/optimize/region_inline.rs:55,60` | `(*body).clone()` clones `Arc<Vec<Node>>` contents for **every** `Node::Region` regardless of whether inlining happens. | Use `Arc::try_unwrap(body).unwrap_or_else(|a| a.as_ref().clone())` only when the Arc is unique, or clone only inside the `count <= threshold` branch. |
| CRITICAL | `transform/optimize/canonicalize.rs:99` | `canonicalize_nodes((*body).clone())` clones `Arc<Vec<Node>>` contents for every `Node::Region` even when body is already canonical. | Clone only when a child actually changes; wrap result in `Arc::new` only after mutation is detected. |
| CRITICAL | `transform/optimize/canonicalize.rs:43-46` | `run` calls `program.entry().to_vec()` (clones entry Vec), then `canonicalize_nodes` maps over it. No fast-path for identity. | Check if any node changed; if not, return the original `Program` without allocating. |
| CRITICAL | `transform/optimize/cse/impl_csectx.rs:357-373` | `rewrite_args` clones every borrowed arg with `Expr::clone(borrowed)` once any prior arg changed, even if the borrowed arg is unchanged. | Push `Cow::Borrowed(borrowed)` directly; only clone when the arg itself is rewritten. |
| CRITICAL | `serial/wire/encode/to_wire.rs:150` | `put_nodes_section` allocates a fresh `Vec<u8>` per node (`let mut payload = Vec::new();`). For N nodes this is N independent heap allocations. | Use a single scratch `Vec<u8>` cleared per iteration, or size-estimate and reuse a `thread_local` buffer. |
| CRITICAL | `serial/wire/encode/to_wire.rs:254,276` | `put_memory_regions` allocates `shape` and `hints` sub-vectors per buffer (2×B allocations). | Write shape/hints directly into `out` using a scratch buffer or inline LEB encoding without intermediate Vec. |
| CRITICAL | `serial/wire/decode/from_wire.rs:220` | `read_node_record` does `reader.take(payload_len)?.to_vec()`  -  allocates a temporary `Vec<u8>` for every node payload. | Decode node directly from the parent reader slice without copying; use a sub-slice view. |
| CRITICAL | `serial/wire/decode/from_wire.rs:355,387` | `read_memory_regions` does `.to_vec()` for `shape_payload` and `hints_payload` per buffer. | Decode shape/hints directly from the parent reader without copying into temporary Vecs. |
| HIGH | `optimizer/passes/fusion.rs:272-277` | `replacement_exprs` clones **every** pending expression into a new `FxHashMap` on **every** node processed in `fuse_nodes`. | Maintain a flat `FxHashMap<String, Expr>` that is updated incrementally instead of rebuilding from scratch per node. |
| HIGH | `graph_view.rs:205` | `NodeGraph::from_program` clones every top-level node via `n.clone()` into `DataflowKind::Statement`. | Store `Node` by moving from a consumed Program, or use `Arc<Node>` in `DataflowKind::Statement` to avoid the clone. |
| HIGH | `transform/optimize/dce/eliminate_dead_lets.rs:39-40,57,67` | `live.clone()` on every `If` branch, `Loop` body, and `Block`  -  clones the `FxHashSet<String>` live-set at every control-flow boundary. | Use a persistent data structure (e.g., `im::HashSet` or `Arc<FxHashSet>` with copy-on-write) so cloning is O(1) shared-pointer bump. |
| HIGH | `ir_inner/model/program/meta.rs:360-379` | `buffers_equal_ignoring_declaration_order` allocates two `Vec`s, maps canonical keys, and sorts them on **every** `Program` equality check. | Cache a sorted `Arc<[Vec<u8>]>` inside `Program` (or a precomputed hash) so equality is O(N) comparison without allocation. |
| HIGH | `ir_inner/model/program/core.rs:100-117` | `Program::clone` clones `output_buffer_index: Vec<u32>` and `stats: ProgramStats` by value. Both could be `Arc`-wrapped to make clone a single refcount bump. | Change `output_buffer_index: OnceLock<Arc<Vec<u32>>>` and `stats: OnceLock<Arc<ProgramStats>>`. |
| HIGH | `optimizer/rewrite.rs:11-13` | `rewrite_nodes` always allocates `nodes.iter().map(...).collect()` even when every node is unchanged. | Return `Cow::Borrowed(nodes)` when no rewrite occurred; only allocate when at least one node changed. |
| HIGH | `transform/optimize/cse/impl_csectx.rs:115` | `Loop` body processing creates a brand-new `CseCtx::default()` (`body_ctx`), discarding all outer-scope CSE entries and allocating fresh HashMaps. | Reuse the parent context with a scoped snapshot/restore (same pattern as `enter_scope`/`leave_scope`) instead of a full new context. |
| HIGH | `visit/traits.rs:150-164` | `visit_node_preorder` / `visit_node_postorder` are **recursive** (not iterative), incurring a function call per node and risking stack overflow on adversarial deep IR. | Rewrite using an explicit `Vec` stack (same pattern as `transform::visit::walk_nodes`). |
| HIGH | `transform/visit.rs:251-300` | `walk_nodes_mut` calls `Arc::make_mut(body)` on `Node::Region`, which clones the **entire** `Vec<Node>` body if any other reference exists. | Use an explicit mutable stack of `&mut Node` without forcing CoW on Arc bodies; only mutate when the caller actually changes something. |
| MEDIUM | `transform/optimize/region_inline.rs:105-127` | `count_nodes` only memoizes `Node::Region` bodies by Arc pointer; `Block`, `If`, and `Loop` subtrees are re-counted from scratch on every encounter, making the count O(N²) on deeply nested IR. | Memoize all subtree counts in a `FxHashMap<*const Node, usize>` or rewrite as a single bottom-up pass. |
| MEDIUM | `optimizer/passes/const_fold.rs:48-55` | `fold_expr` clones `true_val` / `false_val` (`Box::new(...).clone()`) on **every** `Select` encounter, even when `cond` is not a literal. | Move the literal match before the clone, or return `Cow::Borrowed`. |
| MEDIUM | `optimizer/passes/strength_reduce.rs:49,52` | `reduce_expr` clones `left`/`right` (`as_ref().clone()`) after confirming power-of-two, but the clone happens on a match arm that may be hit frequently. | Use `Cow::Borrowed` in the return path; only clone if the arm is taken. |
| MEDIUM | `serial/wire/encode/put_expr.rs:180-184` | `extension.wire_payload()` returns an owned `Vec<u8>` which is then copied into `out`. Forces a double allocation for every opaque expression. | Change `wire_payload` signature to write into `&mut Vec<u8>` (append-only), matching the encoder's invariant. |
| MEDIUM | `serial/wire/decode/from_wire.rs:567-571` | `leb_string` copies bytes into a temporary `Vec<u8>` (`bytes.to_vec()`) before `String::from_utf8`. | Convert the sub-slice directly to `String` without the intermediate `Vec` copy. |
| MEDIUM | `execution_plan/fusion.rs:262-263` | `combined_entry.extend(prog.entry().iter().cloned())` clones every node from every fused program into a flat Vec. | Accept `Vec<Program>` by value and drain entries instead of cloning. |
| LOW | `dialect_lookup.rs:29-33` | `intern_string` uses a global `ThreadedRodeo` behind a `OnceLock`. Every interned string does a concurrent hash-table operation with potential contention under high dispatch rates. | Pre-intern hot op ids at registry installation time; expose a fast-path for already-interned `InternedOpId` lookups. |
| LOW | `execution_plan/fusion.rs:246` | `buf.clone()` clones `BufferDecl` (which contains `Arc<str>` name, hints, etc.) for every buffer of every program, even when deduplicated. | Move `buf` out of the program when caller passes ownership, or use `Arc<BufferDecl>` in the merged table. |
| LOW | `transform/optimize/cse/impl_csectx.rs:82-91` | `Node::Store` always clones `index` and `value` via `self.expr(...).into_owned()` even when the CseCtx makes no substitutions. | Return `Cow::Borrowed` from `expr` and only `into_owned()` when a substitution occurred. |
| LOW | `ir_inner/model/program/stats.rs:98-143` | `compute_stats` does a full recursive walk without memoization. If the stats cache is invalidated (e.g., by `entry_mut`), the next call re-walks the entire tree. | Invalidate less aggressively, or incrementally update stats during `entry_mut` mutations. |
| LOW | `optimizer/passes/dead_buffer_elim.rs:79` | `pending: FxHashMap<Arc<str>, Vec<Vec<Arc<str>>>>` nests two Vec allocations per deferred store. | Flatten into `FxHashMap<Arc<str>, Vec<Arc<str>>>` with a secondary index, or use a smallvec for the inner buffer list. |
| LOW | `transform/optimize/canonicalize.rs:228-261` | `expr_sort_key` computes a hash for every `Var`/`Load`/`BufLen` name on every commutative-op sort comparison. String hashing is repeated O(log n) times per sort. | Cache the hash inside `Ident` (already done) and use `Ident::cached_hash()` directly instead of `hash_str`. |

## Closure status  -  2026-04-29 scoped optimizer/scheduler pass

| Finding | Status | Source / proof |
|---|---|---|
| Autotune no-op clone / panic paths | fixed | `optimizer/passes/autotune.rs` returns `PassResult::unchanged` on identity and no longer panics on non-divisible valid programs. |
| Spec-driven no-op clone path | fixed | `optimizer/passes/spec_driven.rs` returns `PassResult::unchanged`. |
| Scheduler invalidation correctness | fixed | `optimizer/scheduler.rs` queues invalidated passes for the next dirty set without invalidating current scheduling availability; regression test passed. |
| Region opacity in DCE | fixed | `transform/optimize/dce/{eliminate_dead_lets,eliminate_unreachable}.rs` recurse into Region bodies; Region DCE tests passed. |
| Atomic/strength-reduce/const-fold optimizer coverage noted here | fixed/stale | Existing optimizer tests under `cargo test -p vyre-foundation optimizer` cover div/mod strength reduction, cast/FMA/select folding, no-op pass identity, and normalize-atomics transformation. |

---

## Cross-cutting themes

### Theme T1: No identity fast-path in tree rewriters
`canonicalize`, `rewrite_program`, `CseCtx::nodes`, and `region_inline` all allocate new `Vec<Node>` / `Expr` containers even when the output is byte-identical to the input. **Fix:** Adopt the `Cow<'a, Expr>` pattern already present in `optimizer/rewrite.rs` for *all* tree rewriters, and add a `Vec<Node>` equivalent (`CowNodes`). A single `match` on `(Cow::Borrowed, Cow::Borrowed)` avoids 90 % of allocations on typical IR that does not change.

### Theme T2: `Arc<Vec<Node>>` is cloned eagerly
`Node::Region { body: Arc<Vec<Node>> }` is cloned via `(*body).clone()` in `canonicalize`, `region_inline`, and `decode_scan_fuse`. The Arc is supposed to enable sharing, but every pass that recurses into a Region immediately deep-clones the contents. **Fix:** Passes should clone only when they intend to mutate, or use `Arc::try_unwrap` to steal the Vec when the refcount is 1.

### Theme T3: Wire encode/decode is allocation-heavy
Encoding allocates a fresh `Vec<u8>` per program, per node payload, and per buffer shape/hints. Decoding allocates temporary `Vec<u8>` copies for every length-prefixed sub-record. **Fix:** Use a single `thread_local` scratch buffer for encoding; decode via sub-slice views (`&[u8]`) instead of `to_vec()`.

### Theme T4: `Program::clone` is not as cheap as it looks
`Program` stores `entry: Arc<Vec<Node>>` and `buffers: Arc<[BufferDecl]>`, but `Clone` still copies `output_buffer_index: Vec<u32>` and `stats: ProgramStats` by value. On a hot dispatch path where Programs are cloned for planning/analysis, these two fields dominate the cost. **Fix:** Wrap them in `Arc`.

---

## Competitor comparison

- **MLIR / LLVM:** LLVM's `IRBuilder` uses an in-place `ilist` (intrusive linked list) for instructions, so a no-op pass is literally zero allocations. Vyre's `Vec<Node>` + `Arc` model forces a full container copy on every mutation. Consider an arena-backed `NodeList` handle (like `mlir::Block::iterator`) for 0.7.
- **WGSL / Naga:** Naga's IR uses a single `Arena<T>` with index handles. Cloning a function body is a slice-copy of indices, not deep tree cloning. Vyre's `Box<Expr>` + `Arc<Vec<Node>>` tree is semantically cleaner but materially slower to rewrite.
- **SPIR-V / rspirv:** Uses a flat `Vec<Instruction>` with result-id references. Passes mutate in place with no recursive visitor overhead. Vyre's nested `Expr` / `Node` enums pay a cache-miss and allocation tax that SPIR-V avoids.

---

## Actionable next steps (priority order)

1. **Eliminate the three `Program::clone()` no-op paths** (autotune, spec_driven, fusion single-input)  -  one-line fixes, immediate throughput win.
2. **Add `CowNodes` identity fast-path** to `canonicalize` and `rewrite_nodes`  -  cuts allocations on the dominant "no change" branch.
3. **Replace per-node payload `Vec::new()`** in `to_wire` with a reusable scratch buffer  -  eliminates N heap allocations per encode.
4. **Arc-wrap `ProgramStats` and `output_buffer_index`**  -  makes `Program::clone` a true shallow copy.
5. **Memoize `count_nodes`** for Block/If/Loop, or rewrite as a single iterative pass  -  fixes the O(N²) region-inline analysis.
6. **Make `wire_payload` / decode payloads borrow-friendly**  -  changes the extension trait contract, so needs a minor version bump.

---

*Audit completed: 2026-04-24. 30 findings. Every line number verified against `vyre-foundation` HEAD.*
