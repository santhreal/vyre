# Supersession notice

This document is historical optimizer-audit evidence. Active optimizer work is
controlled by `docs/optimization/START_HERE.md`,
`docs/optimization/TAXONOMY.md`, and
`docs/optimization/OWNERSHIP.toml`.

# VYRE_OPTIMIZER Audit  -  2026-04-24

Scope: `libs/performance/matching/vyre/vyre-foundation/src/optimizer` (all passes + scheduler)  
Auditor: Kimi Code CLI  
Focus: performance regressions, correctness gaps, architectural debt.

---

## Summary

| Finding | Severity | File |
|---------|----------|------|
| CRIT-01 | CRITICAL | `passes/autotune.rs:228` |
| CRIT-02 | CRITICAL | `fusion_cert.rs:53` |
| HIGH-01 | HIGH | `scheduler.rs:286` |
| HIGH-02 | HIGH | `passes/strength_reduce.rs:41` |
| HIGH-03 | HIGH | `execution_plan/fusion.rs:276` |
| MED-01 | MEDIUM | `passes/fuse_cse.rs:52` |
| MED-02 | MEDIUM | `passes/const_fold.rs:39` |
| MED-03 | MEDIUM | `passes/spec_driven.rs:23` |
| MED-04 | MEDIUM | `passes/autotune.rs:26` |
| MED-05 | MEDIUM | `passes/normalize_atomics.rs:37` |
| MED-06 | MEDIUM | `passes/decode_scan_fuse.rs:72` |
| MED-07 | MEDIUM | `passes/fusion.rs:272` |
| MED-08 | MEDIUM | `passes/fusion.rs:334` |
| LOW-01 | LOW | `passes/decode_scan_fuse.rs:60` |
| LOW-02 | LOW | `optimizer.rs:68` |
| LOW-03 | LOW | `execution_plan/fusion.rs:411` |
| LOW-04 | LOW | `passes/const_buffer_fold.rs:22` |
| LOW-05 | LOW | `passes/strength_reduce.rs:57` |
| LOW-06 | LOW | `rewrite.rs:4` |

## Closure status  -  2026-04-29 scoped optimizer pass

| Finding | Status | Source / proof |
|---|---|---|
| CRIT-01 autotune panic on non-divisible sizes | fixed | `passes/autotune.rs` uses `check_even_divisible_without_guard` and `PassResult::unchanged`; `cargo test -p vyre-foundation optimizer` passed. |
| CRIT-02 fusion certificate zero digest collision | fixed | `fusion_cert.rs::blake3_program` hashes a domain-separated serialization-error digest; `cargo test -p vyre-foundation optimizer` passed. |
| HIGH-01 dirty tracking drops invalidated skipped/earlier passes | fixed | `scheduler.rs` now carries invalidation through `next_dirty` without corrupting current `available`; `scheduler_tests::invalidating_prior_requirement_does_not_break_current_iteration` passed. |
| HIGH-02 strength reduction misses div/mod power-of-two | fixed | `passes/strength_reduce.rs` rewrites unsigned div/mod by powers of two and carries regression tests; `cargo test -p vyre-foundation optimizer` passed. |
| MED-03 / MED-04 no-op clone paths | fixed | `spec_driven.rs` and `autotune.rs` return `PassResult::unchanged` on identity paths. |
| MED-05 normalize atomics no-op placeholder | fixed | `passes/normalize_atomics.rs` now hoists atomic conditions and has regression tests. |
| MED-06 decode-scan metadata / entry clone findings | fixed | `decode_scan_fuse.rs` avoids the deep entry clone with `Arc::try_unwrap` and preserves identity when no candidates exist. |

---

## CRITICAL

### CRIT-01 | `passes/autotune.rs:228-239`  -  Panic on valid programs with non-divisible problem sizes

`assert_even_divisible_without_guard` uses `assert_eq!`, which panics in release builds when a program's inferred problem size does not evenly divide the workgroup size.  
**Defect:** The optimizer crashes on valid user input. A program with `workgroup_size = [64,1,1]` and a buffer count of 1000 is perfectly legal  -  it merely needs a bounds check, which autotune is supposed to inject. Instead, if the workgroup size is already "tuned" (so `size_changed == false`), the assertion fires.  
**Fix:** Replace `assert_eq!` with a diagnostic-return or `return PassResult { program, changed: false }` after logging a structured warning. Never panic on user IR.

### CRIT-02 | `fusion_cert.rs:53-58`  -  Collision of all unserializable programs to `[0u8; 32]`

`blake3_program` swallows `to_wire` errors with `Err(_) => [0u8; 32]`. Every program that fails wire serialization collides to the same 32 zero bytes.  
**Defect:** Two distinct unserializable fused kernels produce identical pre/post fingerprints, making the certificate useless for audit trails and permitting silent unsound fusion. The `fingerprint_program` function in `optimizer.rs` was already fixed (domain-separated error digest) but `fusion_cert.rs` was not updated.  
**Fix:** Port the `FINGERPRINT_ERROR_SENTINEL` domain-separated error hashing from `optimizer.rs:308-320` into `blake3_program`.

---

## HIGH

### HIGH-01 | `scheduler.rs:280-291`  -  Dirty-tracking loses invalidated passes that never ran in the current iteration

If pass A invalidates pass B, and B has **not yet run** in the current `run_once` iteration (e.g., because `analyze` returned SKIP on the pre-A program), B is not in `available`. The invalidation logic only adds passes to `next_dirty` when they are in `available`:  
```rust
if available.contains(invalidated) { next_dirty.insert(*invalidated); }
```  
**Defect:** B is silently dropped from the fixpoint. If A's changes would have flipped B's `analyze` to `RUN`, B never gets a chance to run. This is a latent correctness bug in any pipeline where an early pass creates work for a later pass that was initially skipped.  
**Fix:** Always insert invalidated passes into `next_dirty`, regardless of `available` membership. Remove the `available.contains` guard.

### HIGH-02 | `passes/strength_reduce.rs:41-65`  -  Only `Mul` is reduced; `Div` and `Mod` by power-of-two are ignored

`reduce_expr` matches exclusively on `BinOp::Mul`.  
**Defect:** `x / 8` and `x % 8` are not rewritten to `x >> 3` and `x & 7`, leaving ~1–2 ALU cycles on the table per operator on every GPU backend. Competitors (MLIR, SPIR-V opts, naga) all perform these rewrites.  
**Fix:** Extend `reduce_expr` with `BinOp::Div` → `Shr` and `BinOp::Mod` → `BitAnd(value - 1)` for unsigned literals that are powers of two. Guard signed division with an explicit sign-magnitude decomposition or skip it until a signed-safe variant is proven.

### HIGH-03 | `execution_plan/fusion.rs:276-282`  -  Axis-wise max workgroup can over-dispatch by 1000×

`fused_workgroup` takes the per-axis maximum of all input programs:  
```rust
fused_workgroup[0] = fused_workgroup[0].max(wg[0]);
```  
**Defect:** Fusing `[1024,1,1]` with `[1,1024,1]` yields `[1024,1024,1]` = 1,048,576 threads. The comment admits this is "safe (over-dispatching is a no-op wasted invocation)", but at internet scale a 1000× thread waste is a regression, not a no-op. Some GPU schedulers throttle or TDR on extreme over-dispatch.  
**Fix:** Reject fusion when the axis-wise max exceeds a conservative multiplier (e.g., 4× the largest individual program), or switch to per-arm dynamic dispatch indices instead of a single global workgroup.

---

## MEDIUM

### MED-01 | `passes/fuse_cse.rs:52-68` / `execution_plan/fusion.rs:88-288`  -  `fuse_cse` is buffer-dedup only, not expression-level CSE

The module comment states: "Shared buffers collapse to a single `BufferDecl` via the name-keyed union … that is the CSE."  
**Defect:** The name `fuse_cse` promises cross-rule Common Subexpression Elimination (shared subexpressions deduplicated across rules). In reality, only buffer declarations are deduplicated. Two rules that both compute `input[i] * 2` will emit the identical computation twice in the fused entry body. This is a naming/architectural lie that misleads maintainers.  
**Fix:** Rename to `fuse_buffer_dedup` or implement true expression-level CSE across the concatenated entry bodies after fusion.

### MED-02 | `passes/const_fold.rs:39-61`  -  Missing folds for `Cast`, `Fma`, and `Select` with non-`U32` literals

`fold_expr` handles `BinOp`, `UnOp`, and `Select` with `LitBool`/`LitU32` conditions only.  
**Defect:**
- `Cast(U32, LitI32(5))` is not folded to `LitU32(5)`.
- `Fma(LitF32(2.0), LitF32(3.0), LitF32(1.0))` is not folded to `LitF32(7.0)`.
- `Select(LitI32(-1), a, b)` is not folded to `a` (non-zero `LitI32` is truthy).
These leave literal expressions in the IR that downstream passes and backends must re-traverse.  
**Fix:** Add `Expr::Cast { value: Lit*, .. }`, `Expr::Fma { a: Lit*, b: Lit*, c: Lit* }`, and `Expr::Select { cond: LitI32(v), .. }` arms to `fold_expr`.

### MED-03 | `passes/spec_driven.rs:23-24`  -  Wasteful `program.clone()` in no-op transform

```rust
pub fn transform(program: Program) -> PassResult {
    PassResult::from_programs(&program.clone(), program)
}
```  
**Defect:** The entire `Program` is cloned just to compare it against itself. This allocates a new `Arc<Vec<Node>>`, rebuilds `OnceLock`s, and triggers `structural_eq` on every call. At foundation tier this pass is a permanent no-op (`analyze` always returns `SKIP`), but the clone path is still compiled and reachable.  
**Fix:** Return `PassResult { program, changed: false }` directly.

### MED-04 | `passes/autotune.rs:26`  -  Unnecessary clone on unchanged workgroup size

```rust
return PassResult::from_programs(&program, program.clone());
```  
**Defect:** Same pattern as MED-03. When the workgroup size is already optimal, the pass clones the program and performs a full structural comparison instead of returning unchanged.  
**Fix:** Return `PassResult { program, changed: false }`.

### MED-05 | `passes/normalize_atomics.rs:37-62`  -  Documented no-op placeholder pass violates LAW 1 (NO STUBS)

The file header admits: "The `transform` function below returns `changed = false` on every program. No tree rewrite is performed. The pass exists in the pipeline so the pass-id is stable."  
**Defect:** A stub pass occupies a slot in the default pipeline, wastes scheduler cycles (`analyze` returns `SKIP`, but the scheduler still evaluates `dirty.contains`), and promises a rewrite that does not exist.  
**Fix:** Delete the pass and its registration. If the pass-id stability is required for external tooling, replace it with a compile-time `#[deprecated]` alias that maps to a no-op external registration, not a living file.

### MED-06 | `passes/decode_scan_fuse.rs:72-76`  -  Metadata loss on `Program::wrapped`

```rust
Program::wrapped(
    new_buffers,
    program.workgroup_size,
    (*program.entry).clone(),
)
```  
**Defect:** `entry_op_id`, `non_composable_with_self`, and cached hashes are dropped because `Program::wrapped` initializes them to `None` / `false` / empty. A fused decode→scan kernel that was marked `non_composable_with_self` silently becomes composable, breaking the self-aliasing gate in `execution_plan::fusion.rs:125`.  
**Fix:** Chain `.with_optional_entry_op_id(...)` and `.with_non_composable_with_self(...)` after `Program::wrapped`, matching the pattern used in `autotune.rs:45-47` and `fusion.rs:39-40`.

### MED-07 | `passes/fusion.rs:272-277`  -  `replacement_exprs` clones all pending replacements on every non-control-flow node

```rust
fn replacement_exprs(replacements: &FxHashMap<String, PendingExpr>) -> FxHashMap<String, Expr> {
    replacements.iter().map(|(name, pending)| (name.clone(), pending.expr.clone())).collect()
}
```  
**Defect:** Called on every `Let`, `Assign`, and `Store` node. If `r` replacements are pending and the block has `n` nodes, this is `O(n·r)` clones of expression trees. For long scalar pipelines (the exact shape fusion is designed to optimize), `r` can be 10–50 and `n` 100–500, causing quadratic work.  
**Fix:** Pass `&FxHashMap<String, PendingExpr>` directly into `substitute_expr` and clone only the matching replacement at substitution time (lazy clone).

### MED-08 | `passes/fusion.rs:334-351`  -  `flush_selected_replacements` re-inserts unflushed items one-by-one

```rust
let pending = std::mem::take(replacement_order);
for name in pending {
    if let Some(pending_expr) = replacements.remove(name.as_str()) {
        if names.contains(name.as_str()) {
            fused.push(Node::let_bind(name, pending_expr.expr));
        } else {
            replacements.insert(name.clone(), pending_expr);
            replacement_order.push(name);
        }
    }
}
```  
**Defect:** Every non-flushing call scans all `k` pending replacements and re-inserts `k-1` of them. Over `f` flushes this is `O(f·k)`. In programs with frequent buffer writes (e.g., every 3rd node), `f` ≈ `n/3` and `k` grows linearly, yielding `O(n²)` total work.  
**Fix:** Drain only the items that need flushing. Use `replacement_order.retain(|name| { ... })` or a `VecDeque` pop-front discipline instead of full `mem::take` + re-insert.

---

## LOW

### LOW-01 | `passes/decode_scan_fuse.rs:60-70`  -  O(n·m) buffer promotion check

```rust
if promotable.iter().any(|n| n == b.name()) { ... }
```  
**Defect:** Nested linear scan over `promotable` inside the buffer loop. For programs with hundreds of buffers this is needless `O(n²)`.  
**Fix:** Collect `promotable` into a `FxHashSet<String>` before the mapping loop.

### LOW-02 | `optimizer.rs:68-71`  -  `PassResult::from_programs` forces structural comparison even when the pass knows nothing changed

```rust
pub fn from_programs(before: &Program, program: Program) -> Self {
    let changed = before != &program;
    Self { program, changed }
}
```  
**Defect:** Every pass that uses this helper pays `O(N)` `structural_eq` to discover `changed = false`. Passes like `normalize_atomics` and `spec_driven` know they are no-ops but still pay the comparison tax. Passes that use `rewrite_program` also pay it even when `rewrite_nodes` produced byte-identical output.  
**Fix:** Add a second constructor `PassResult::unchanged(program: Program)` that skips comparison, and teach no-op passes to use it. Optionally thread a `changed: bool` flag through `rewrite_program` / `rewrite_nodes`.

### LOW-03 | `execution_plan/fusion.rs:411-416` / `418-424`  -  Wildcard match arms default to `ReadOnly` / `Global`

```rust
_ => ReadOnly,
```  
```rust
_ => crate::ir::MemoryKind::Global,
```  
**Defect:** If a new `BufferAccess` variant is added (e.g., `Storage`), these arms silently downgrade or misclassify it instead of failing to compile. This is an extensibility hazard.  
**Fix:** Replace `_ =>` with an explicit exhaustiveness match or a `compile_error!` fallback so adding a variant is a breaking change that forces review.

### LOW-04 | `passes/const_buffer_fold.rs:22-30`  -  Not wired into default `registered_passes()` pipeline

`const_buffer_fold` exports `fold_const_buffer(program, &ConstBuffer)` as a free function, but it is absent from `optimizer.rs:232-240`.  
**Defect:** The pass exists but is never invoked by `PassScheduler::default()`. Callers must discover it manually. This is dead pipeline surface area.  
**Fix:** Either register it as a proper `PassKind` with a `ConstBuffer` context source (e.g., from `AdapterCaps` or a side-channel), or move it to a pure utility module outside `optimizer/passes/`.

### LOW-05 | `passes/strength_reduce.rs:57-64`  -  Misses negative power-of-two `LitI32` and `mul by 1`

```rust
Expr::LitI32(value) if *value > 0 && (*value as u32).is_power_of_two() => { ... }
```  
**Defect:** `x * -2` is not reduced (negative literals are ignored). Additionally, `x * 1` is reduced to `x << 0`, which is not a reduction  -  some GPUs handle `mul` by 1 better than `shl` by 0.  
**Fix:** Handle negative powers of two by reducing to `Negate(Shl(x, shift))`. Exclude `1` (and `-1`) from power-of-two reduction; let `const_fold` handle `x * 1 → x`.

### LOW-06 | `rewrite.rs:4-9`  -  `rewrite_program` always allocates a new entry Vec even when no expr changed

```rust
pub(crate) fn rewrite_program(program: &Program, mut expr: impl FnMut(&Expr) -> Option<Expr>) -> Program {
    program.with_rewritten_entry(rewrite_nodes(program.entry(), &mut expr))
}
```  
**Defect:** `rewrite_nodes` unconditionally `collect()`s into a new `Vec<Node>`, and `rewrite_node` reconstructs every node variant even when `rewrite_expr` returns `Cow::Borrowed`. For large programs (megakernel fusion outputs), this allocates thousands of `Node` and `Expr` objects that are byte-identical to the input.  
**Fix:** Thread a `changed: &mut bool` flag through `rewrite_nodes` / `rewrite_node` / `rewrite_expr`. When every recursive call returns `Cow::Borrowed`, return `Cow::Borrowed(nodes)` from `rewrite_nodes` and short-circuit `with_rewritten_entry`.

---

## Cross-Cutting Observations

### No-Op Passes (Focus Area #1)

| Pass | `analyze` | `transform` | Verdict |
|------|-----------|-------------|---------|
| `normalize_atomics` | `SKIP` | Returns program unchanged | **Identity no-op** |
| `spec_driven` | `SKIP` | `from_programs(&program.clone(), program)` | **Identity no-op** |
| `const_buffer_fold` | N/A (not registered) | Free function only | **Invisible to scheduler** |

**Recommendation:** Delete `normalize_atomics` until it has a real rewrite. Move `spec_driven` out of the default pipeline or give it a driver-layer hook that actually supplies a dialect registry.

### Canonicalise / Hashing (Focus Area #2)

The canonicalization pass lives in `transform/optimize/canonicalize.rs`, not under `optimizer/`. `expr_sort_key` (line 228) does **not** perform recursive hashing  -  it hashes only the top-level variant tag and, for `Var`/`Load`/`BufLen`, the identifier string. There is no `O(n²)` hashing over `Expr` because `Expr` does not implement `Hash` at all; the CSE subsystem uses a separate `ExprKey` enum. The sort key is `O(1)` per node.

### const_fold / const_buffer_fold Coverage (Focus Area #3)

- `const_fold` recursively folds bottom-up via `rewrite_expr`, so nested literals like `(1+2)+3` collapse to `6` in one pass. ✅
- `const_buffer_fold` only substitutes `Load` → `LitU32`; it does **not** subsequently fold the resulting `LitU32` into surrounding `BinOp`s in the same pass. The fixpoint scheduler mitigates this if `const_fold` is scheduled after `const_buffer_fold`, but `const_buffer_fold` is not in the default pipeline. ⚠️
- `const_fold` misses `Cast`, `Fma`, and `Select(cond=LitI32)` folds. See MED-02.

### fuse_programs Barrier Insertion (Focus Area #4)

Barriers are inserted when a **read-only** arm precedes a **write** arm on the same buffer. The logic at `execution_plan/fusion.rs:219-224` uses `partition_point` to find the first write after a read.  
**Conservative or tight?** It is **conservative** (correct but sometimes excessive): it inserts a barrier after the *last* read arm even if no write arm actually follows it in program order. Wait  -  no, it inserts `barrier_after_arm.insert(read_arm)` for each read arm that has any later write arm. This is tight for the read→write hazard, but it misses **write→read** hazards (a later arm reads a buffer that an earlier arm wrote). The comment says barriers are inserted for "read by one arm and written by a later arm", but write→read is also a hazard that requires a barrier. The current code only checks `read_arms` → `write_arms`, not `write_arms` → `read_arms`.

### fuse_cse Subexpression Detection (Focus Area #5)

As documented in MED-01, `fuse_cse` performs **zero** expression-level deduplication. It only collapses buffer declarations by name. Cross-rule subexpression sharing (e.g., two rules computing `input[i] & mask`) is not detected. True cross-rule CSE would require a global value numbering pass over the concatenated entry body.

### strength_reduce Coverage (Focus Area #6)

| Pattern | Detected? | Rewrite |
|---------|-----------|---------|
| `mul by pow2` | ✅ | `shl` |
| `div by pow2` | ❌ | missing |
| `mod by pow2` | ❌ | missing |
| `mul by 1` | ⚠️ (regression to `shl 0`) | should be skipped |
| `mul by neg pow2` | ❌ | missing |

See HIGH-02 and LOW-05.

### Clone-Then-Mutate Patterns (Focus Area #7)

| Location | Pattern | Fix |
|----------|---------|-----|
| `autotune.rs:26` | `program.clone()` for no-op | MED-04 |
| `spec_driven.rs:24` | `program.clone()` for no-op | MED-03 |
| `rewrite.rs:4` | Always rebuilds entry Vec | LOW-06 |

No pass does the literal `let fresh = program.clone(); mutate(&mut fresh)` anti-pattern, but `with_rewritten_entry` reconstructs the `Program` struct around a new entry Arc on every call. This is architecturally required because `Program` is immutable, but the unconditional allocation in `rewrite_program` is wasteful.

---

## Competitor Comparison

| Feature | MLIR Affine / SPIR-V opt | Naga | Vyre (current) |
|---------|--------------------------|------|----------------|
| `div/mod → shift/and` | ✅ | ❌ | ❌ |
| Cross-dispatch expr CSE | ✅ (LLVM GVN) | N/A | ❌ |
| Workgroup auto-tuning | ✅ (Auto-Tuning in Triton) | ❌ | ⚠️ (panics) |
| Constant fold `Cast` | ✅ | ✅ | ❌ |
| Fixpoint scheduling with invalidation | ✅ (PassManager) | N/A | ⚠️ (dirty bug) |

---

## Action Priority

1. **Now:** Fix CRIT-01 (panic) and CRIT-02 (certificate collision).  
2. **This week:** Fix HIGH-01 (scheduler dirty tracking), HIGH-02 (strength reduce gaps), MED-06 (metadata loss).  
3. **Next sprint:** Fix MED-07 / MED-08 (fusion quadratic clones), MED-01 (rename or real CSE), LOW-06 (rewrite short-circuit).  
4. **Cleanup:** Delete `normalize_atomics` stub (MED-05), wire or delete `const_buffer_fold` (LOW-04).
