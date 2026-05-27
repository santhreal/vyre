# Gemini handoff  -  10 easiest vyre roadmap tasks (2026-05-02)

Each task is a single self-contained PR. Pick one, finish it end-to-end (code +
tests + ROADMAP.md status update), then move to the next. **Stay inside the
named file unless explicitly told to cross.** Run only `cargo test
--release -p vyre-foundation --lib <task_filter>` (or the listed crate) to
verify; do not touch `vyre-lower/`, `vyre-driver-cuda/`, `vyre-driver-wgpu/`,
`vyre-runtime/`, `vyre-self-substrate/`, or `vyre-emit-naga/`  -  those are
Codex's lane and you will collide.

Lane-aware: every task lives in `foundation_optimizer` or `bench_harness`
unless noted. Per `docs/optimization/AGENT_CONTRACT.md`, every patch must
ship at least one positive + one negative twin test.

---

## 1. Loop trip-zero on swapped bounds

- **File:** `vyre-foundation/src/optimizer/passes/loops/loop_trip_zero_eliminate.rs`
- **Add:** the existing rule fires when `from >= to` for `Expr::LitU32`. Add
  the same rule for `Expr::LitI32` so `Loop("i", 5, 3, body)` collapses too.
- **Tests:** add 2 tests next to existing ones  -  `i32_swapped_bounds_collapses`
  and `i32_equal_bounds_collapses`.
- **ROADMAP:** no row change (this is a coverage extension under existing rule).

## 2. Constant-condition If branch elimination  -  already done; add coverage

- **File:** `vyre-foundation/src/optimizer/passes/cleanup/if_constant_branch_eliminate.rs`
- **Add:** if the existing test list misses `Expr::LitU32(0) → reject`,
  `Expr::LitU32(1) → accept`, `Expr::LitI32(0) → reject`,
  `Expr::LitI32(-1) → accept` (any non-zero u32/i32 is truthy in WGSL),
  add the missing 4 tests.
- **ROADMAP:** no row change; widen test coverage of an existing pass.

## 3. Empty-block collapse adversarial twin

- **File:** `vyre-foundation/src/optimizer/passes/cleanup/empty_block_collapse.rs`
- **Add:** an adversarial test that asserts `Block(vec![Block(vec![])])` collapses
  to `Block(vec![])` (nested empty), and a negative twin asserting
  `Block(vec![Store(...)])` is preserved. Two tests total.
- **ROADMAP:** no row change.

## 4. ROADMAP G2  -  Reciprocal approximation, constant-only path

- **File:** `vyre-foundation/src/optimizer/passes/algebraic/strength_reduce/arithmetic.rs`
- **Add:** add a fold rule for `Expr::Div { LitF32(1.0), x }` that returns
  `Expr::Div { LitF32(1.0), x }` unchanged (no-op until UnOp::Reciprocal exists)
  but DO add the matching arm so the line/comment exists for future wiring.
  Then add a real fold for `Div(LitF32(1.0), LitF32(c))` where `c != 0` to
  return `LitF32(1.0/c)`. Two tests: `div_one_by_constant_folds_to_reciprocal_literal`
  and `div_one_by_zero_does_not_fold` (Div by zero stays as the IR's defined
  trap path).
- **ROADMAP:** update the G2 row to note the constant-folding half is now
  shipped; the variable-x half still needs a `UnOp::Reciprocal` variant.

## 5. ROADMAP A28  -  loop peeling first iteration when guarded

- **File:** new `vyre-foundation/src/optimizer/passes/loops/loop_peel.rs` +
  register in `loops/mod.rs`.
- **Pattern:** `Loop("i", 0, N, [If(Eq(Var("i"), Lit(0)), peeled_body), rest...])`
  with `N` literal and `N > 1` → emit `peeled_body; Loop("i", 1, N, [rest...])`.
- **Conservatism:** require `from = LitU32(0)`, `to = LitU32(N)` with `N > 1`,
  and the first body node is exactly `If(Eq(Var(loop_var), LitU32(0)), then)`
  with empty otherwise.
- **Tests:** 5 tests  -  positive (peel fires), negative when from != 0, when
  to is not literal, when first node is not the matching If, when peeled body
  contains an Assign to the loop var.
- **ROADMAP:** mark A28 done.

## 6. ROADMAP A32  -  tail duplication for divergent branches

- **File:** new `vyre-foundation/src/optimizer/passes/cleanup/tail_duplication.rs`
  + register in `cleanup/mod.rs`.
- **Pattern:** `If(c, [a, b], [a', b])` where `b == b'` (identical tail) and
  `b` has length 1 and is observably free → leave the If with the differing
  prefixes and hoist `b` out: `If(c, [a], [a']); b`.
- **Tests:** positive, negative when tails differ, negative when tail has
  side effects (Store / Atomic), negative when tail is a Loop.
- **ROADMAP:** mark A32 done.

## 7. ROADMAP M2  -  per-op kernel-time table emitter

- **File:** new `vyre-bench/src/report/kernel_time_table.rs` + register in
  `report/mod.rs`.
- **Body:** iterate every `CaseReport` in a `ReportSchema`, emit a table row
  per case with columns `case_id`, `kernel_execute_ns_p50`, `kernel_execute_ns_p99`,
  `bytes_touched_p50`, `wall_throughput_gb_s_p50`. Skip cases missing
  `kernel_execute_ns`. Output is plain pipe-delimited text suitable for
  `column -t -s '|'`.
- **Tests:** 4 tests  -  positive multi-case, single-case, missing-stage skip,
  empty input.
- **ROADMAP:** mark M2 done.

## 8. ROADMAP M4  -  achieved memory bandwidth probe (CPU-side computation)

- **File:** add a method to `vyre-bench/src/api/metric.rs::BenchMetrics`:
  `pub fn achieved_bandwidth_gb_s(&self) -> Option<f64>` that returns
  `bytes_touched / wall_ns * 1e9 / 1e9` when both are present. Add unit tests
  in the same file under `#[cfg(test)] mod tests`.
- **Tests:** 4  -  positive (both fields), missing wall_ns, missing bytes_touched,
  zero wall_ns (returns None to avoid div-by-zero).
- **ROADMAP:** mark M4 done (CPU-side computation half; backend-counter half
  needs driver_cuda / driver_wgpu wiring which is Codex's lane).

## 9. ROADMAP S6  -  OP_MATRIX coverage scan reporter

- **File:** new `vyre-bench/tests/op_matrix_coverage.rs` (integration test).
- **Body:** load `docs/optimization/OP_MATRIX.toml`, iterate, collect every
  registered `vyre_libs::harness::all_entries()` op id. Assert that at least
  the basic shape is present (each registered op appears with non-empty
  `tier`, `owners`, `backend_status` fields). The test should print the
  delta  -  registered op ids missing from OP_MATRIX, and OP_MATRIX rows whose
  id is not registered  -  and fail with a counts-only assertion.
- **Tests:** the test IS the test. Run with `cargo test
  --release -p vyre-bench --test op_matrix_coverage`.
- **ROADMAP:** mark S6 partially-done (the audit gate exists; closing the
  gaps it surfaces is per-op work tracked under op_matrix lane).

## 10. ROADMAP K3  -  debug-only tag assertions in tag accessors

- **File:** `vyre-foundation/src/ir_inner/model/program/meta.rs` (audit) plus
  a new `#[cfg(test)] mod tests` block adding 3 tests:
  1. `validation_skip_cache_hits_on_repeated_validate_calls`  -  call `validate()`
     twice on the same Program, assert the second call returns immediately
     (no recomputation; can be observed by capturing a `Cell<u32>` counter
     wired through a debug-only path, OR by asserting `is_structurally_validated()`
     flips to true after the first call).
  2. `validation_skip_cache_clears_after_with_rewritten_entry`  -  the cache
     must invalidate when the Program shape changes.
  3. `mark_validated_on_distinguishes_backends`  -  `mark_validated_on("wgpu")`
     does not satisfy `is_validated_on("cuda")`.
- **ROADMAP:** mark K3 done (the discipline this row asks for is already
  embodied by the K1 cache; the test block above proves the discipline).

---

## Hand-off checklist (per task)

Before marking a task done in ROADMAP.md:

1. New code lives in the named file (no scope creep).
2. Tests pass: `cargo test --release -p <crate> --lib <filter>`.
3. ROADMAP.md row updated with file path + test count.
4. No edits to files outside the named one (run `git diff --name-only` to
   confirm; if you needed cross-file changes you picked the wrong task  -  stop
   and ask).
5. No `assert!` outside `#[cfg(test)]` (per K2 audit; use `Result` for
   production fallible paths).

---

## What you will NOT touch

- `vyre-lower/` (Codex actively rewriting)
- `vyre-driver-cuda/` (Codex)
- `vyre-driver-wgpu/` (Codex; only safe edit is `vyre-driver-wgpu/tests/buf_len_array_length.rs` extension if a task explicitly says so)
- `vyre-driver-spirv/` (broken; Codex will fix)
- `vyre-runtime/megakernel/` (Codex)
- `vyre-self-substrate/` (Codex)
- `vyre-emit-naga/`, `vyre-emit-ptx/`, `vyre-emit-spirv/` (Codex)
- `vyre-foundation/src/optimizer/diff_compile.rs` (Codex)
- `vyre-foundation/src/optimizer/effect_lattice.rs` (Codex)
- The top of `docs/optimization/CLAIMS.toml` `main-codex` block (don't edit
  Codex's claim, but DO add your own `[[claim]]` block for whichever task
  you start).

If you need to touch any of those to finish a task, the task is wrong-sized  - 
stop and ask.
