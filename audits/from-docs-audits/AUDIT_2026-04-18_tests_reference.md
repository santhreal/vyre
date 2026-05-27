# AUDIT: TEST QUALITY — vyre-reference (the oracle)

**Date:** 2026-04-18  
**Scope:** `vyre-reference/src/**/*.rs`, `vyre-reference/tests/**/*.rs` (none exist)  
**Auditor:** Kimi Code CLI  
**Severity scale:** CRITICAL / HIGH / MEDIUM / LOW  

> The reference interpreter is the ORACLE — every backend's correctness is measured against it. If the reference's tests are weak, every conformance claim is weak.

---

## Executive Summary

`vyre-reference` has **no dedicated `tests/` directory** and only **three inline `#[cfg(test)]` modules** (`interp.rs`, `eval_expr.rs`, `value.rs`). The vast majority of source modules — including the entire `primitives/` subtree (~20 canonical evaluators), `atomics.rs`, `oob.rs`, `eval_node.rs`, `workgroup.rs`, `typed_ops.rs`, and `eval_call.rs` — contain **zero tests**. Where tests do exist, they are either **self-referential** (comparing two implementations that both live in this crate) or **happy-path-only** (single-value stores, masked floats, trivial workgroup sizes). No adversarial cases, no external oracles, no property tests on multi-dimensional input spaces, and no verification of error paths.

**Bottom line:** A bug in the reference interpreter will not be caught by its own test suite. It will only be caught when a backend diverges — at which point the reference is assumed correct and the backend is blamed. This is a conformance pipeline built on sand.

---

## Findings

### TEST-01 — `src/interp.rs:301` — CRITICAL — Self-referential oracle

**Current:** `prop_arena_value_matches_hashmap_baseline` compares `run_arena_reference` against `eval_hashmap_reference`. Both implementations live in this crate and share the same author-maintainer mental model. A conceptual error (e.g., identical misinterpretation of workgroup barrier semantics) will pass silently.

**Fix:** Add property tests that compare interpreter output against **independent external oracles**: stdlib `hash` for `hash_fnv1a`, the `blake3` crate's own test vectors for `hash_blake3`, `regex-automata` for `pattern_match_dfa`, and IEEE-754 reference tables for float operations.

---

### TEST-02 — `src/interp.rs:298` — HIGH — Happy-path-only interpreter proptest

**Current:** The only interpreter-level proptest builds a trivial program (`Let` + `Store` u32) with workgroup size `[1, 1, 1]`. It does not exercise `Barrier`, `Loop`, `If`/`Else`, `Atomic`, `Return`, `Block`, `IndirectDispatch`, `AsyncLoad`/`AsyncWait`, workgroup memory, or multi-invocation dispatch.

**Fix:** Add adversarial IR programs that exercise **every `Node` variant** and run with `workgroup_size > 1`. Include programs designed to fail (uniform-control-flow violations, missing inputs, zero workgroup size).

---

### TEST-03 — `src/eval_expr.rs:716` — CRITICAL — Self-referential expression oracle

**Current:** `prop_flat_evaluator_matches_frame_oracle` compares `eval` (flat opcode stack) against `eval_frame_oracle` (recursive frame stack). Both are in the same crate and are written by the same authors. A shared bug in expression lowering (e.g., `Expr::Atomic` operand order) will be identical in both paths.

**Fix:** Validate expression results against independent ground truth: Rust's built-in `wrapping_add`, `wrapping_mul`, etc. for integers; `libm` or hardware-independent reference for floats.

---

### TEST-04 — `src/eval_expr.rs:712` — HIGH — Happy-path-only expression proptest

**Current:** The expression proptest only covers `Add`, `Mul`, `Sub`, `Select`, and `FMA`. It does **not** cover:
- `Div` / `Mod` (including division-by-zero)
- `Shl` / `Shr` (including shift amount ≥ bit-width)
- `BitAnd` / `BitOr` / `BitXor` on signed types
- `Cast` between any type pairs
- `Atomic` load/store/compare-exchange
- `Load` / `BufLen` / `Store` (buffer expressions)
- `Call` (primitive dispatch)
- `Opaque` expressions

**Fix:** Expand proptest to cover every `Expr` variant and every `BinOp`/`UnOp` with adversarial inputs (zero, max, NaN, negative).

---

### TEST-05 — `src/eval_expr.rs:739` — MEDIUM — `prop_assert_eq!` without diagnostic message

**Current:** `prop_assert_eq!(flat, frame);` gives no context on divergence. When the property test fails in CI, the developer sees only "assertion failed: `(left == right)`" with no expression tree or generated inputs.

**Fix:** Add a descriptive message: `prop_assert_eq!(flat, frame, "flat vs frame oracle diverged for expr={expr:?} a={a} b={b} c={c} pick_left={pick_left}");`.

---

### TEST-06 — `src/value.rs:253-269` — MEDIUM — Happy-path-only truthiness tests

**Current:** `truthy()` tests check `-0.0`, `0.0`, `1.0`, `-1.0`, `INFINITY`, `NEG_INFINITY`. They omit:
- `NaN` (all bit payloads)
- Subnormal floats
- `f64::MAX`, `f64::MIN`, `f32::MIN_POSITIVE`
- Mixed-type truthiness (`Value::Array(empty)`, `Value::Bytes(empty)`, `Value::U64(0)`, `Value::I32(-1)`)

**Fix:** Add exhaustive truthiness tests for every `Value` variant and every float edge case.

---

### TEST-07 — `src/value.rs:255` — MEDIUM — `assert!` without panic message

**Current:** `assert!(!Value::Float(-0.0).truthy());` provides no diagnostic on failure. A regression that makes `-0.0` truthy will produce a generic assertion failure with no explanation of WGSL semantics.

**Fix:** `assert!(!Value::Float(-0.0).truthy(), "WGSL bool(-0.0) must be false");`.

---

### TEST-08 — `src/value.rs:272` — HIGH — Missing property tests on multi-dimensional Value space

**Current:** The `neg_zero_select_branches_to_false` proptest only generates `positive_sign in bool::ANY` to produce `0.0` vs `-0.0`. It never exercises `NaN`, `INFINITY`, subnormal, non-zero, or cross-variant values (`Value::U32`, `Value::Bool`, `Value::Array`).

**Fix:** Add `proptest` over all `Value` constructors and all `f64` bit patterns (including NaN payloads, generated via `f64::from_bits(any::<u64>())`).

---

### TEST-09 — `src/typed_ops.rs` (module-level) — CRITICAL — Zero tests for ~50 generated binop/unop functions

**Current:** `typed_ops.rs` contains macro-generated helpers for `BinOp` and `UnOp` across `u32`, `i32`, `u64`, `bool`, and `f32`. There are **no unit tests, no proptests, and no edge-case coverage** for:
- Division-by-zero (`div_u32`, `div_i32`, `div_u64`)
- Modulo-by-zero (`rem_u32`, `rem_i32`, `rem_u64`)
- Shift amount overflow (`shift_u32`, `shift_i32`, `shift_u64`)
- Float NaN propagation in `binop_f32`
- `AbsDiff` on all integer types

**Fix:** Add property tests for every `(BinOp, type)` and `(UnOp, type)` pair, including adversarial operands.

---

### TEST-10 — `src/atomics.rs` (module-level) — CRITICAL — Zero tests for all atomic operations

**Current:** `atomic_add`, `atomic_or`, `atomic_and`, `atomic_xor`, `atomic_min`, `atomic_max`, `atomic_exchange`, and `atomic_compare_exchange` are completely untested. No verification of:
- `atomic_add` wrapping behavior (`u32::MAX + 1`)
- `atomic_compare_exchange` success vs failure paths
- Semantics matching `std::sync::atomic` reference behavior

**Fix:** Add exhaustive unit tests for every `AtomicOp` and property tests for compare-exchange races.

---

### TEST-11 — `src/oob.rs` (module-level) — CRITICAL — Zero tests for OOB semantics

**Current:** Out-of-bounds `load`, `store`, `atomic_load`, and `atomic_store` define the GPU-matching behavior (zero-fill on load, no-op on store, `None` on atomic OOB). These semantics are **untested**.

**Fix:** Add adversarial tests for OOB indices on all `IrDataType` variants (`U32`, `U64`, `F32`, `Bytes`, `Vec2U32`, `Vec4U32`). Assert exact byte output.

---

### TEST-12 — `src/eval_node.rs` (module-level) — CRITICAL — Zero tests for statement execution

**Current:** The statement executor (`step`, `execute_node`) handles `Let`, `Assign`, `Store`, `If`, `Loop`, `Return`, `Block`, `Barrier`, `IndirectDispatch`, `AsyncLoad`/`AsyncWait`, and `Opaque`. **None** of these are tested. No verification of:
- Uniform-control-flow violation detection
- Barrier synchronization across invocations
- Nested loop scoping and variable shadowing rejection
- `eval_return` clearing frames

**Fix:** Add unit tests for every `Node` variant and integration tests for adversarial control-flow programs.

---

### TEST-13 — `src/primitives/arith_add.rs` (and all 20 primitives) — CRITICAL — Zero tests for every canonical primitive

**Current:** The `primitives/` subtree contains 20 reference evaluators (`arith_add`, `arith_mul`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `clz`, `compare_eq`, `compare_lt`, `gather`, `hash_blake3`, `hash_fnv1a`, `pattern_match_dfa`, `pattern_match_literal`, `popcount`, `reduce`, `scan`, `scatter`, `shift_left`, `shift_right`, `shuffle`). **Every single one has zero tests.**

**Fix:** Add unit tests and property tests for every primitive. The SQLite-quality bar demands exhaustive testing on small domains (e.g., all `u16 × u16` for `arith_add`, all shift amounts `0-31` for `shift_left`/`shift_right`, known test vectors for `hash_blake3`).

---

### TEST-14 — `src/primitives/hash_blake3.rs:6` — HIGH — No external test-vector verification

**Current:** `HashBlake3::evaluate` delegates to the `blake3` crate but has no test asserting the output matches known vectors. A crate upgrade, miswired byte slice, or endianness bug would go undetected.

**Fix:** Add BLAKE3 official test vectors (empty input, "abc", 1 MiB all-zeros) asserting exact byte-for-byte output.

---

### TEST-15 — `src/primitives/hash_fnv1a.rs:4-15` — HIGH — No external oracle for FNV-1a

**Current:** The FNV-1a constants and loop are replicated inline but never compared against a canonical reference (e.g., Python's `fnvhash`, known IETF vectors). A single-byte typo in `FNV_OFFSET` or `FNV_PRIME` would silently corrupt every downstream conformance test.

**Fix:** Add canonical FNV-1a test vectors for empty, single-byte (`0x00`), and multi-byte inputs.

---

### TEST-16 — `src/primitives/pattern_match_dfa.rs:34-99` — CRITICAL — Complex parser, zero tests, ~10 error paths

**Current:** `ParsedDfa::parse` validates magic header, state count, start state, accept count, transition table size, accept state ranges, and transition target ranges. **None** of these error paths are exercised. No tests for:
- Truncated header / wrong magic
- `state_count == 0` or `start >= state_count`
- Accept state out of range
- Transition target out of range
- Byte-length mismatch

**Fix:** Add unit tests for every `EvalError` path and a property test comparing `pattern_match_dfa` output against `regex-automata` on random regexes.

---

### TEST-17 — `src/primitives/pattern_match_literal.rs:7-22` — HIGH — Literal matcher untested

**Current:** `PatternMatchLiteral::evaluate` has zero tests. No coverage for:
- Empty haystack
- Literal longer than haystack
- Overlapping matches (e.g., `"aa"` in `"aaa"`)
- `u32::MAX` offset overflow path
- Unicode / non-ASCII bytes

**Fix:** Add adversarial tests for all edge cases and compare against `memchr` or `regex` crate.

---

### TEST-18 — `src/primitives/gather.rs:5-14` — HIGH — Gather primitive untested

**Current:** `Gather` has zero tests. No coverage for empty `values`/`indices`, out-of-bounds indices (which should hit `checked_index`), or single-element arrays.

**Fix:** Add property tests comparing `gather` output against a manual Rust indexing loop, plus adversarial index inputs.

---

### TEST-19 — `src/primitives/scatter.rs:5-32` — HIGH — Scatter primitive untested

**Current:** `Scatter` has zero tests. The "last write wins" semantics for duplicate indices is undefined in docs and unverified. No coverage for:
- Empty inputs
- `max_index == u32::MAX` → `usize` overflow path
- Duplicate indices
- Mismatched value/index lengths

**Fix:** Add property tests and adversarial tests for index overflow and duplicate scattering. Assert exact output bytes.

---

### TEST-20 — `src/primitives/reduce.rs:5-15` — MEDIUM — Reduce primitive untested

**Current:** `Reduce` returns `0` on empty input, but this path is untested. No property test verifies associativity or correctness against `Iterator::fold` for any `CombineOp`.

**Fix:** Add unit tests for empty, single-element, and multi-element inputs for all `CombineOp` variants. Add property tests.

---

### TEST-21 — `src/primitives/scan.rs:5-19` — MEDIUM — Scan primitive untested

**Current:** `Scan` returns empty on empty input, but this path is untested. No property test verifies prefix-sum correctness for any `CombineOp`.

**Fix:** Add unit tests and property tests for all `CombineOp` variants, comparing against a manual prefix-sum loop.

---

### TEST-22 — `src/primitives/shift_left.rs:5-9` — MEDIUM — Shift-left masking untested

**Current:** `shift_left` masks the shift amount with `& 31` to match WGSL semantics, but this behavior is completely untested. No test for shift amount `31`, `32`, `33`, or `0`.

**Fix:** Add exhaustive tests for all shift amounts `0-63` and compare against Rust's `wrapping_shl`.

---

### TEST-23 — `src/primitives/shift_right.rs:5-9` — MEDIUM — Shift-right masking untested

**Current:** Same as TEST-22 — zero tests for shift-right behavior.

**Fix:** Add exhaustive tests for all shift amounts `0-63` and compare against Rust's `wrapping_shr`.

---

### TEST-24 — `src/primitives/compare_eq.rs` / `compare_lt.rs` — MEDIUM — Comparison primitives untested

**Current:** `CompareEq` and `CompareLt` have zero tests. No coverage for `u32::MAX` vs `0`, signed vs unsigned boundaries, or equality of identical bit patterns.

**Fix:** Add exhaustive small-domain property tests (all `u8 × u8` pairs promoted to `u32`).

---

### TEST-25 — `src/workgroup.rs` (module-level) — HIGH — Workgroup simulation untested

**Current:** `create_invocations`, `workgroup_memory`, `LocalSlots::for_program`, and `Invocation` locals/binding/assignment have **zero tests**. No verification of:
- Workgroup ID arithmetic overflow (`global_dim`)
- `MAX_WORKGROUP_BYTES` boundary
- `LocalSlots` duplicate-name handling
- `Invocation::bind` / `assign` / `bind_loop_var` error paths

**Fix:** Add unit tests for every public and `pub(crate)` function in `workgroup.rs`.

---

### TEST-26 — `src/eval_call.rs` (module-level) — HIGH — Primitive call dispatch untested

**Current:** `eval_call`, `encode_inputs`, and `execute_spec` have zero tests. No coverage for:
- Arity mismatch (`validate_arity`)
- Input byte budget overflow (`MAX_CALL_INPUT_BYTES`)
- Unsupported `Compose` variants
- `spec_output_value` type coercion

**Fix:** Add adversarial tests for each error path and golden-path tests for every registered op in `vyre::ops::registry`.

---

### TEST-27 — `src/eval_expr_cast.rs` (module-level) — HIGH — Cast matrix untested

**Current:** `cast_value` handles `U32`, `I32`, `U64`, `Bool`, `Bytes`, `Vec2U32`, `Vec4U32`, and fallback `_`. Zero tests for:
- `u64 → u32` narrowing overflow
- `i32 → u32` signed→unsigned reinterpretation
- `f32 → u32` truncation
- `Vec2U32` / `Vec4U32` widening behavior

**Fix:** Add exhaustive cast matrix tests for every `(source Value, target DataType)` pair.

---

### TEST-28 — `src/primitive/bitwise/xor/reference_a.rs:4-11` — MEDIUM — Dual-reference XOR untested

**Current:** `reference_a` (word-oriented XOR) and `reference_b` (bit-by-bit XOR) are the **only** dual-reference pair in the crate. They have **zero dedicated tests** verifying they produce identical output. The only differential test is the generic interpreter proptest, which never exercises `primitive.bitwise.xor` directly.

**Fix:** Add property tests directly comparing `reference_a` and `reference_b` for all `u16 × u16` pairs and for short/long byte inputs (including `< 8` bytes).

---

### TEST-29 — `src/interp.rs:311-313` — MEDIUM — `prop_assert_eq!` without diagnostic context

**Current:** The interpreter proptest uses `.expect("Fix: arena interpreter succeeds")` (okay) and `prop_assert_eq!(arena, hashmap);` (not okay). On failure, the developer cannot see the generated `value` or the program IR.

**Fix:** `prop_assert_eq!(arena, hashmap, "interpreter divergence for value={value} program={program:?}");`.

---

### TEST-30 — `src/lib.rs:46-48` — MEDIUM — Test-only export indicates architectural coupling

**Current:** `#[cfg(test)] pub use interp::eval_hashmap_reference;` exposes an internal test oracle as public API solely for the convenience of inline tests. This reveals that tests are coupled to internals rather than exercising the public `run` interface.

**Fix:** Remove test-only public exports. Move integration-style tests to a `tests/` directory that uses only the public `run` API.

---

### TEST-31 — `vyre-reference/tests/` — HIGH — Missing integration test directory

**Current:** The crate has **no `tests/` directory**. All tests are inline `#[cfg(test)]` modules embedded in source files. This prevents:
- Black-box testing of the public API
- Test isolation from `pub(crate)` visibility
- Separation of test code from shipping code

**Fix:** Create `vyre-reference/tests/` with integration tests for `run`, `flat_cpu::run_flat`, and every primitive re-export.

---

### TEST-32 — `benches/eval_throughput.rs:5-18` — LOW — Benchmark covers trivial program only

**Current:** The sole benchmark measures a `let` + `store` of a `u32` literal. It provides no signal for expression-heavy programs, primitive call dispatch, barrier synchronization, or multi-invocation throughput.

**Fix:** Expand benchmarks to cover expression evaluation (nested arithmetic), primitive dispatch (hash, pattern match), and workgroup barrier round-trip latency.

---

## Category Summary

| Category | Count | Notes |
|----------|-------|-------|
| Happy-path-only | 6 | Interpreter, expression, value, float masking, trivial programs |
| Self-referential | 3 | Arena↔HashMap, Flat↔Frame, reference_a↔reference_b (indirectly) |
| Missing adversarial cases | 10 | Div-by-zero, OOB, UCF violations, zero workgroup, empty inputs |
| `assert_eq!` without message | 3 | `interp.rs`, `eval_expr.rs`, `value.rs` |
| `let _ = something()` | 0 | *None found — good* |
| Missing property tests | 8 | `typed_ops`, `atomics`, `primitives/*`, `workgroup`, `oob` |
| Coverage gaps | 12 | Entire `primitives/` subtree, `eval_node`, `eval_call`, `eval_expr_cast` |
| `println!` in tests | 0 | *None found — good* |
| Test helpers duplicating production code | 1 | `empty_memory()` is minor; the larger issue is `hashmap_interp.rs` duplicating the whole interpreter without dedicated sync tests |

> **Total findings:** 32 (minimum requested: 20)

---

## Recommendations (Priority Order)

1. **Stop adding primitives without tests.** Every new `ReferenceEvaluator` impl must include unit tests and, where feasible, property tests before merge.
2. **Create `vyre-reference/tests/` integration tests** that exercise the public `run` API against known-good inputs for every primitive.
3. **Add external oracles.** The reference interpreter must not be the only oracle for itself. Use `blake3` test vectors, `regex-automata`, `libm`, and Rust stdlib ops as independent ground truth.
4. **Proptest every `BinOp`/`UnOp`/`AtomicOp`/`CombineOp`.** The macros in `typed_ops.rs` and `atomics.rs` generate dozens of functions; each needs property coverage.
5. **Add adversarial IR programs.** Programs designed to fail (OOB, UCF violation, div-by-zero, missing buffer) must be tested to ensure error messages are actionable and deterministic.
