# Gemini handoff #2  -  every non-complex remaining task (2026-05-02)

Status of batch 1: 10/10 done.

This batch is **everything that does not require genuinely new
infrastructure or cross-crate refactor**. Pick any row, finish
end-to-end, commit, move on. CC takes only items marked **(CC)** below
because they cross-cut the dataflow consumer / vyre-lower / new IR variants.

## Hard rules (unchanged)

- Stay inside the named file unless explicitly told to cross.
- Do NOT touch: `vyre-lower/`, `vyre-driver-cuda/`, `vyre-driver-spirv/`,
  `vyre-driver-wgpu/src/**` (tests are OK), `vyre-runtime/megakernel/`,
  `vyre-self-substrate/`, `vyre-emit-naga/`, `vyre-emit-ptx/`,
  `vyre-emit-spirv/`, `vyre-foundation/src/optimizer/diff_compile.rs`,
  `vyre-foundation/src/optimizer/effect_lattice.rs`, top of
  `docs/optimization/CLAIMS.toml` `main-codex` block.
- Add your own `[[claim]]` block to `CLAIMS.toml` for whichever task you
  start. Update `ROADMAP.md` row when done.
- One file per task unless it explicitly says otherwise.

---

## Hygiene / dedup batch

### Task 11  -  H1d byte-pack helper consolidation

- **File:** new `vyre-libs/src/test_support/byte_pack.rs` + register module
  in `vyre-libs/src/test_support/mod.rs` (create if missing) + `vyre-libs/src/lib.rs`.
- **Body:** lift the helpers that appear ≥9 times across `vyre-libs/src`
  inventory blocks: `pub fn u32_bytes(words: &[u32]) -> Vec<u8>` and
  `pub fn bytes_to_u32(slice: &[u8]) -> Vec<u32>`.
- **Replace each duplicate** with `crate::test_support::byte_pack::u32_bytes`.
- **Tests:** 2 (round-trip, empty input).
- **ROADMAP:** mark H1d done.

### Task 12  -  H2d wgpu C-front fixture helper

- **File:** new `vyre-driver-wgpu/tests/common/c_fixture.rs` + add
  `mod common;` to each `vyre-driver-wgpu/tests/c_*.rs` file that uses it.
- **Body:** factor the 7 duplicated fixture builders into one shared
  helper. Touch each test only to swap the local helper for the shared one.
- **Tests:** the 7 affected test files stay green.
- **ROADMAP:** mark H2d done.

### Task 13  -  H3d RecordAndReadback constructor

- **File:** `vyre-driver-wgpu/src/engine/record_and_readback.rs` only
  (this is the one wgpu src exception  -  the change is a single helper
  inside a file CC has not edited recently).
- **Body:** add `RecordAndReadback::for_dispatch(...)` builder helper;
  collapse the two existing call sites onto it.
- **ROADMAP:** mark H3d done.

### Task 14  -  H4d canonical walker replacement

- **File:** find via `git grep -n 'fn walk_node\|fn visit_node\b' vyre-foundation/src/optimizer`.
- **Body:** replace each with `crate::visit::node_map::map_children` /
  `node_map::any_descendant`.
- **Tests:** add `walker_matches_canonical_on_corpus` per file (run both
  the kept-inline private old walker and the new one on the same
  Program; assert identical node-set).
- **ROADMAP:** mark H4d done.

### Task 15  -  H5d ExprVisitor default no-op stubs

- **File:** locate via `git grep -n 'pub trait ExprVisitor'` (likely
  `vyre-foundation/src/visit/expr.rs`).
- **Body:** give every `visit_*` method a `default { ControlFlow::Continue(()) }`
  body. Then delete the no-op stubs from each of the 6 implementors.
- **Tests:** adversarial  -  a `#[cfg(test)] mod` introduces a fake new
  ExprVisitor that overrides only one method; assert the others compile
  via the default.
- **ROADMAP:** mark H5d done.

### Task 16  -  H7d M5 device-signature path verification

- **File:** `docs/optimization/ROADMAP.md` (M5 docstring) + possibly
  `vyre/devices/*.toml`.
- **Body:** find the new path of the deleted `devices/sm_120.toml`
  (likely `vyre/devices/blackwell.toml`). Update M5 docstring; if no
  replacement exists, re-create it from
  `vyre-driver/src/device_signature.rs::BUILTIN_SM_120`.
- **Tests:** add `vyre-driver/tests/device_signature_path.rs` that reads
  the new path and asserts deserialisation succeeds.
- **ROADMAP:** mark H7d done.

### Task 17  -  H9d JSON variant of the M0 flame emitter

- **File:** `vyre-bench/src/report/flame.rs` (add `pub fn collapse_report_json`).
- **Body:** mirror `collapse_report` but emit JSON array
  `[{ "case": "...", "stage": "...", "p50_ns": N }, ...]`.
- **Tests:** 4 (multi-case, single-case, empty, missing-stage).
- **ROADMAP:** mark H9d (M0 half) done.

### Task 18  -  H9d JSON variant of the M2 kernel-time table

- **File:** `vyre-bench/src/report/kernel_time_table.rs` (add `_json` variant).
- **Body:** same JSON shape as task 17; one row per case.
- **Tests:** 4.
- **ROADMAP:** mark H9d (M2 half) done.

### Task 19  -  H10d analyze-SKIP coverage (per-pass)

- **One file at a time.** Pick any pass file under
  `vyre-foundation/src/optimizer/passes/**` that has a
  `pub fn analyze(...) -> PassAnalysis` but lacks an `analyze_skips_*`
  / `analyze_runs_*` pair.
- **Body:** add the two tests.
- **Repeat** for each remaining pass file (~30 of them). Each one is its
  own small commit.
- **ROADMAP:** widen H10d row's progress count after each.

### Task 20  -  Empty-block collapse adversarial twins

- **File:** `vyre-foundation/src/optimizer/passes/cleanup/empty_block_collapse.rs`.
- **Body:** add 3 adversarial tests: empty Region inside Block; three
  levels nested; empty Block sibling alongside a Store.
- **ROADMAP:** widen the existing row.

---

## Optimizer-side small wins

### Task 21  -  A36 atomic minimization (single-writer literal-buffer case)

- **File:** new `vyre-foundation/src/optimizer/passes/algebraic/atomic_minimize.rs`
  + register in `passes/algebraic/mod.rs`.
- **Body:** when a `BufferDecl` is touched by exactly ONE `Expr::Atomic`
  in the entire program AND that atomic is `AtomicOp::Add` with
  `expected: None` AND no other Store/Load touches the buffer, replace
  the atomic with a non-atomic `Node::Store(buffer, index, current_load + value)`.
  Conservative: if any other access exists, do nothing.
- **Tests:** 6 (positive single-atomic, negative when 2 atomics exist,
  negative when Load also reads, negative when buffer is shared with
  Store, AtomicOp::CompareExchange not eligible, MemoryOrdering::SeqCst
  not eligible).
- **ROADMAP:** mark A36 done.

### Task 22  -  A33 algebraic expansion (distribute mul over add for literal RHS)

- **File:** `vyre-foundation/src/optimizer/passes/algebraic/const_fold/binop_identities.rs`.
- **Body:** add `(a + b) * K → a*K + b*K` only when `K` is a small
  literal (u32 ≤ 256, i32 with absolute value ≤ 256). Limits avoid blow-up.
- **Tests:** 3 (positive small-K, negative large-K, negative non-literal RHS).
- **ROADMAP:** widen A33 / mark A25 supplement.

### Task 23  -  A33 sign-preserving distribution for subtraction

- **File:** same as task 22.
- **Body:** `(a - b) * K → a*K - b*K` for the same small-K constraint.
- **Tests:** 3.
- **ROADMAP:** same row.

### Task 24  -  A35 stronger range fold: `Mod(x, N)` where x.max < N

- **File:** `vyre-foundation/src/optimizer/passes/algebraic/const_fold/binop_identities.rs`.
- **Body:** when the LHS of `BinOp::Mod` is a `Var` bound by a `Let`
  whose value is a `LitU32(c)` with `c < N`, fold to `Var(name)`. This
  needs a tiny single-block lookbehind; conservative: walk back through
  the immediately preceding sibling Lets only.
- **Tests:** 3 (positive, negative when c >= N, negative when binding
  is not a literal).
- **ROADMAP:** strengthen A35 row.

### Task 25  -  G2 reciprocal constant-folding pass arm

- **File:** `vyre-foundation/src/optimizer/passes/algebraic/strength_reduce/arithmetic.rs`.
- **Body:** existing strength_reduce already does `x / Lit(c) → x * Lit(1/c)`.
  Add the symmetric arm: `Lit(1.0) / Lit(c) → Lit(1/c)` (constant folds
  the reciprocal). Trivial.
- **Tests:** 2.
- **ROADMAP:** widen G2.

### Task 26  -  G4 Horner rewrite (degree-2 only, conservative)

- **File:** new `vyre-foundation/src/optimizer/passes/algebraic/horner.rs`
  + register in `algebraic/mod.rs`.
- **Body:** detect `a*x*x + b*x + c` (where x is a Var, a/b/c are
  observably-free expressions) and rewrite to `(a*x + b)*x + c`. Only
  fires when all three coefficients are visible and `x` is a simple Var.
- **Tests:** 4 (positive, negative when x is not simple, negative when
  any coefficient depends on x, negative when degree > 2).
- **ROADMAP:** mark G4 done (degree-2 case).

### Task 27  -  A14 register-pressure stub (cost-model only)

- **File:** new `vyre-foundation/src/optimizer/cost_model/register_pressure.rs`
  + add module to `optimizer/cost.rs` if missing.
- **Body:** `pub fn estimate_register_pressure(program: &Program) -> u32`
  that returns the maximum live-Let count at any point in the entry
  body. Conservative single-pass scan, no rematerialization. Used as
  a cost input for future passes; no rewrite yet.
- **Tests:** 4 (empty program → 0; one Let → 1; nested If → max of arms;
  Loop body → max within body).
- **ROADMAP:** mark A14 (estimate half) done.

### Task 28  -  H6 Welford sum-of-squares primitive

- **File:** new `vyre-libs/src/math/welford.rs` + register in
  `vyre-libs/src/math/mod.rs`.
- **Body:** `pub fn welford_sum_of_squares(input: &str, sum_out: &str,
  sum_sq_out: &str, n: u32) -> Program` that emits the Welford-stable
  recurrence as a single-invocation loop. Real IR builder.
- **Tests:** 3 (positive against a small dataset, length-1 input,
  empty input rejected with a Fix message).
- **ROADMAP:** mark G6 done.

---

## Handoff checklist

Before marking a task done in ROADMAP.md:

1. New code lives in the named file (no scope creep).
2. `cargo test --release -p <crate> --lib <filter>` passes.
3. ROADMAP.md row updated with file path + test count.
4. CLAIMS.toml has a `[[claim]]` block from you for the task; status
   moves to `done` on commit.
5. `git diff --name-only` shows only the named file; cross-file edits =
   wrong-sized task, stop and ask.

---

## What CC keeps

These are NOT in this handoff because they cross dataflow consumer, vyre-lower, or
need new IR variants:

- A1 hash-cons Expr (foundational, breaks the world)
- A2 SoA Program (depends on A1)
- A3 Region side-table (depends on A1/A2)
- A4 bitset tags (depends on A2)
- A10 GPU e-graph (depends on M0 measurement first)
- A11–A18, A22 (need the dataflow consumer reaching-def / points-to / range / alias / live)
- A30 polyhedral, A31 software pipelining (heavy algorithm)
- E2 parsed C source LRU cache (libs_parsing rework)
- H1 Strassen, H2 FFT, H3 im2col, H4 flash-attention, H5 GEMM-bias-act
  (heavy algorithm)
- I1, I2, I4 (PGO needs telemetry plumbing)
- L1–L4 (parsing perf, libs_parsing rework)
- P1–P10 (consumer phase obligations, multi-crate)
- All S1–S5 (Codex's lane)
- B / C / D series (Codex's lane)
