# CRITIQUE_FIX_REVIEW_2026-04-23

Date: 2026-04-23
Scope: Second-pass read-only audit of the vyre-foundation fix wave committed since `57de6d4c7e` (F-IR-21/22/23/25-28/29-33/37/41/44-48, plus opaque_payload, dialect_lookup provider_id, canonicalisation helpers, execution_plan, and fuse_batch).

Method: Static read-only audit focused on what the first critique wave (CRITIQUE_IR_SOUNDNESS_2026-04-22.md) missed. No source files modified. Every finding names a concrete path, describes the hole, proposes a fix, and names an adversarial test shape.

Pre-flight check: `grep -rn 'unwrap_or_else(|poison| poison.into_inner())' src/` returned **zero matches**  -  the specific poison-recovery pattern was not introduced in this fix wave.

---

## Findings

### 1. CRITICAL | src/ir_inner/model/program.rs:601 + src/serial/wire/encode/to_wire.rs + src/serial/wire/decode/from_wire.rs
`non_composable_with_self` is compared in `structural_eq` and preserved in `Clone`, but `to_wire()` does **not** encode it and `from_wire()` always decodes to `false`. A program with the flag set to `true` round-trips as `false`, violating the fundamental `decode(encode(p)) == p` invariant. It also causes execution-plan fingerprint collisions: a composable and a non-composable variant of the same IR produce identical wire bytes and therefore identical `ExecutionPlan::program_fingerprint`, defeating cache isolation.

**Fix:** Add the flag to the metadata payload in `to_wire.rs` (`metadata_payload`) and `from_wire.rs` (`read_metadata`), or remove it from `structural_eq` if it is strictly a transient fusion hint. If retained in the wire format, bump `WIRE_FORMAT_VERSION` and add a regression test.

**Test hint:** Generate a `Program` with `.with_non_composable_with_self(true)`, encode, decode, and assert `decoded.is_non_composable_with_self()` is `true` and `decoded.structural_eq(&original)` holds.

---

### 2. CRITICAL | src/optimizer/fuse_batch.rs:125-142
The self-composition gate (`F-IR-23`) only checks programs that carry `Some(entry_op_id)`. A program with `non_composable_with_self = true` but `entry_op_id = None` bypasses the gate entirely. Two copies of such a program can be fused silently, corrupting shared workgroup-local state.

**Fix:** Reject the batch if any two programs both have `is_non_composable_with_self() == true`, regardless of `entry_op_id`. Use a fallback key (e.g., a hash of buffer names + workgroup size) when `entry_op_id` is `None`.

**Test hint:** `Program::wrapped(...).with_non_composable_with_self(true)` (deliberately omit `with_entry_op_id`), fused with itself  -  must return `FusionError::SelfAliasing`, not `Ok`.

---

### 3. HIGH | src/ir_inner/model/program.rs:389-403
`BufferDecl::workgroup(name, count, element)` assigns `count` directly without the `assert!(count > 0)` that `with_count` enforces. The F-IR-05 fix added the panic to `with_count` but left the `workgroup` constructor as an unguarded bypass. A parser builder can construct a zero-count workgroup buffer that passes construction, fails at wire-encode time, or crashes the GPU with a cryptic backend error instead of the actionable construction-time message.

**Fix:** Add the same `assert!(count > 0, "Fix: ...")` inside `BufferDecl::workgroup` that `with_count` carries.

**Test hint:** `BufferDecl::workgroup("scratch", 0, DataType::U32)` must panic at construction with an actionable `Fix:` hint naming the zero-count offence.

---

### 4. HIGH | src/execution_plan.rs:434-441  -  **[FIXED]**
`output_visible_bytes` silently falls back to `full_size` when the declared `output_byte_range` is invalid (`start > end` or `end > full_size`). No error is surfaced; the caller gets full readback instead of a loud failure that would reveal the bug. The docstring on `BufferDecl::with_output_byte_range` does not document this fallback, and there is no `Fix:` path.

**Fix:** Return `Option<u64>` or a dedicated `InvalidOutputRange` error from `output_visible_bytes`, propagate it through `memory_plan` to `PlanError`, and add a validation check in `to_wire()` that rejects invalid ranges before encoding.

**Test hint:** `BufferDecl::output("out", 0, DataType::U32).with_count(1024).with_output_byte_range(12..4)`  -  `plan()` must fail with an error naming the invalid range.

---

### 5. HIGH | src/optimizer/fuse_batch.rs:173-175 + 185-195
Barrier insertion only considers `BufferAccess::ReadOnly` as a "read" arm. If an arm declares a buffer as `BufferAccess::Uniform` (read-only by definition) and a later arm writes it as `ReadWrite`, no barrier is inserted because `Uniform` is not matched by the read-arm filter. This is a write-after-read hazard on a buffer that the first arm legitimately reads.

**Fix:** Change the read-arm filter to match any read-only access mode: `*a == BufferAccess::ReadOnly || *a == BufferAccess::Uniform`.

**Test hint:** arm0 with `BufferDecl::uniform("params", 0, DataType::U32)` followed by arm1 with `BufferDecl::read_write("params", 0, DataType::U32)`  -  fused output must contain a `Node::Barrier` between the arms.

---

### 6. HIGH | src/transform/optimize/canonicalize.rs:1-33  -  **[FIXED]**
Module docstring lists three canonicalization rules, including: "**Fold `x == x` and `x != x`** where both operands are syntactically identical `Var` references to literal identities." The `canonicalize_expr` implementation has no such fold  -  it only swaps commutative operands. The docstring promises behaviour the code does not deliver, which misleads extension authors who read the canonicalisation contract.

**Fix:** Either implement the fold in `canonicalize_expr` for `Eq`/`Ne` with `Expr::Var` on both sides (replacing with `Expr::LitBool(true/false)`), or remove the claim from the docstring and file a follow-up task.

**Test hint:** `Program` containing `Node::let_bind("t", Expr::eq(Expr::var("a"), Expr::var("a")))`  -  after `canonicalize::run`, the wire bytes should match a program containing `Expr::LitBool(true)`.

---

### 7. HIGH | src/ir_inner/model/program.rs:1-1230
`program.rs` is 1,230 lines  -  more than 2× the 500-line limit. The fix wave added `non_composable_with_self`, `reconcile_runnable_top_level`, and related methods to an already-god file instead of extracting a `program_meta.rs` or `program_builder.rs` module. Backwards compatibility is not an excuse: the file should have been split as part of the refactor.

**Fix:** Extract `BufferDecl` impl block, `Scope`, and `Program` constructors into `program/builder.rs` or similar; keep `program.rs` under 500 lines as the type-definition root.

**Test hint:** `find src -name '*.rs' -exec wc -l {} + | sort -n | tail -5`  -  no file may exceed 500 lines.

---

### 8. HIGH | src/visit/mod.rs:1-782
`visit/mod.rs` swelled to 782 lines after the fix wave added ~139 lines of visitor exhaustiveness tests. The actual module surface is only ~40 lines; the remaining ~740 lines are a monolithic `#[cfg(test)]` block. This violates the single-responsibility and <500-lines rules.

**Fix:** Extract all `#[cfg(test)]` blocks from `visit/mod.rs` into a new `visit/tests.rs` file that is conditionally included with `#[cfg(test)] mod tests;`.

**Test hint:** `wc -l src/visit/mod.rs` must be < 500 after extraction; `cargo test` must still compile and pass all visitor tests.

---

### 9. HIGH | src/opaque_payload.rs:88-130
The module mixes two distinct responsibilities: (1) endian-fixed wire helpers (`push_u16`, `read_f32`, etc.) and (2) semantic canonicalisation for hash equality (`canonical_regex_flags`, `canonical_f32_zero`). The module name implies only the first concern. A new developer reading the file for wire-format guidance is surprised to find regex-flag sorting.

**Fix:** Split `canonical_regex_flags` and `canonical_f32_zero` into a new `opaque_payload/canonicalize.rs` submodule (or `canonicalize_payload.rs` at the crate root), re-exporting from `opaque_payload` for backward compatibility during a migration window.

**Test hint:** `grep -c 'canonical_' src/opaque_payload.rs` should be 0 after extraction; existing tests should continue to compile via the re-export.

---

### 10. HIGH | src/optimizer/fuse_batch.rs:1-475
`fuse_batch.rs` lives under `optimizer/` but its docstring explicitly states it is **not** expression-level optimization; it is cross-dispatch fusion  -  a runtime/dispatch concern. `execution_plan.rs` already owns fusion planning (`FusionPlan`, `batch_fusion_candidate`). Placing cross-dispatch fusion inside the optimizer layer leaks a runtime concept into the wrong architectural boundary.

**Fix:** Move `fuse_batch.rs` to `src/execution_plan/fusion.rs` (or a new `src/dispatch/fusion.rs`), updating all imports. Keep `optimizer::passes::fusion` for intra-program expression fusion only.

**Test hint:** `grep -rn 'optimizer::fuse_batch' src/ tests/` must return zero hits after the move.

---

### 11. HIGH | src/opaque_payload.rs:109-130
`canonical_f32_zero` normalises `-0.0 → +0.0` for f32 opaque payloads, but no `canonical_f64_zero` exists. An extension author encoding an f64 literal has no canonicalisation helper, so two programs differing only by f64 sign-of-zero will hash distinctly, defeating CSE and cache lookups for f64-bearing opaque extensions.

**Fix:** Add `canonical_f64_zero` with the same IEEE-754 sign-of-zero normalisation logic, reference it from the `canonical_f32_zero` docstring, and add regression coverage.

**Test hint:** `canonical_f64_zero(-0.0f64).to_bits() == 0u64` and `canonical_f64_zero(f64::from_bits(0x8000000000000001)).to_bits() == 0x8000000000000001`.

---

### 12. MEDIUM | src/optimizer.rs:85-86 + src/dialect_lookup.rs:335
`DialectLookup` now requires `provider_id()` for debuggability (F-IR-33), but the sealed `Pass` trait in the same crate does not follow the same pattern. When the scheduler hits `MaxIterations`, it can only report `metadata().name`  -  there is no instance-level identity for inventory-registered external passes, making root-cause tracing harder than 60 seconds.

**Fix:** Add a `pass_id(&self) -> &'static str` method to the `Pass` trait with a default implementation returning `metadata().name`, so external passes can override it for richer diagnostics.

**Test hint:** Register an external pass via `inventory::submit!` that returns a custom `pass_id`; assert the scheduler error includes the custom id during an oscillation loop.

---

### 13. MEDIUM | src/opaque_payload.rs:1-18
Docstring states "extension authors MUST NOT use `to_ne_bytes`" and "MUST be written with `to_le_bytes`", but there is zero compile-time or runtime enforcement. A non-compliant extension author can still append host-endian bytes manually, and the wire decoder has no way to detect the violation on the current host.

**Fix:** Provide a `Writer` wrapper around `Vec<u8>` that ONLY exposes little-endian methods and is required by `ExprNode::wire_payload` / `NodeExtension::wire_payload`. Alternatively, add a post-encode lint in `to_wire` that samples opaque payloads for byte patterns consistent with `to_ne_bytes` on the current host and warns.

**Test hint:** Write a custom `ExprNode` whose `wire_payload` appends `42u32.to_ne_bytes()`; assert that `Program::to_wire` either panics or emits a diagnostic naming the non-compliant extension.

---

### 14. MEDIUM | src/ir_inner/model/program.rs:635-638  -  **[FIXED]**
`reconcile_runnable_top_level` is a new public method added in this fix wave. It has **zero in-tree callers** and **zero unit tests**. The docstring claims "The standard optimizer run ends with this helper," but no optimizer pass or scheduler calls it. This is dead code masquerading as a live contract.

**Fix:** Either wire the method into the standard `optimize()` run immediately after `region_inline`, or delete it and re-introduce it when there is an actual caller. If kept, add a test proving it re-wraps a flattened entry into a root `Node::Region`.

**Test hint:** Build a program via `Program::wrapped`, run `region_inline` to flatten the root region, then call `reconcile_runnable_top_level` and assert `is_top_level_region_wrapped()` is true.

---

### 15. MEDIUM | src/execution_plan.rs:329-349  -  **[FIXED]**
`autotune_plan` hardcodes magic thresholds (`node_count >= 16`, `static_storage_bytes >= (1 << 20)`) without named constants or docstring explanation. These values are unconfigurable and invisible to callers.

**Fix:** Extract `AUTOTUNE_NODE_COUNT_THRESHOLD: usize = 16` and `AUTOTUNE_STORAGE_THRESHOLD: u64 = 1 << 20` as `pub const` items in `execution_plan.rs` with docstrings explaining how they were chosen.

**Test hint:** Not applicable  -  pure refactoring with no behavioural change.

---

### 16. MEDIUM | src/optimizer/fuse_batch.rs:256
`fuse_programs` hardcodes the output workgroup size to `[1, 1, 1]` without explanation. The original arms may have had larger workgroup sizes, and the fused kernel's launch geometry is silently discarded. This could cause under-dispatching or correctness issues if the caller assumes the fused program inherits the original geometry.

**Fix:** Derive the output workgroup size from the maximum of each axis across all input programs, or document why `[1, 1, 1]` is correct and add an assertion that all input programs share the same workgroup size.

**Test hint:** Fuse two programs with workgroup sizes `[64, 1, 1]` and `[128, 1, 1]`; assert the result's workgroup size is `[128, 1, 1]` (or that fusion rejects mismatched sizes with a `Fix:` hint).

---

### 17. MEDIUM | src/execution_plan.rs:186-215  -  **[FIXED]**
`plan()` calls `program.to_wire()` to build a fingerprint without first validating the program. `to_wire()` succeeds for some invalid programs (e.g., an unwrapped top-level entry), so `plan()` can produce a strategy for garbage IR that the validator would reject.

**Fix:** Call `program.validate()?` (or its error equivalent) before `to_wire()`, or document the pre-condition loudly and add a debug assertion.

**Test hint:** `Program::new(vec![], [1, 1, 1], vec![Node::Return])` (unwrapped, deprecated constructor)  -  `plan()` should reject with `PlanError` or a validation error, not produce a `StrategyPlan`.

---

### 18. MEDIUM | tests/wire_roundtrip_proptest.rs
The `arb_program()` strategy never generates programs with `non_composable_with_self = true`. This leaves the wire-roundtrip of that flag completely uncovered by proptest. Combined with Finding #1, this means the round-trip regression would not be caught by the existing property-based suite.

**Fix:** Add a weighted coin-flip in the program generator that sets `with_non_composable_with_self(true)` on ~10% of generated programs once the wire format encodes the flag.

**Test hint:** Proptest configuration that generates `non_composable_with_self` true/false randomly; assert round-trip preserves the flag.

---

### 19. MEDIUM | tests/dialect_lookup_install.rs:101-151
Concurrent race test uses only 2 threads and 2 unique provider ids. It does not exercise the case where N threads with M different ids race, nor the case where all threads share the same id.

**Fix:** Parameterize the test with a `thread_count` const (e.g., 8) and a `unique_ids` slice; spawn threads in a loop. Assert exactly one winner when ids differ, and zero panics when all ids match.

**Test hint:** 8 threads racing `install_dialect_lookup` with 4 unique ids  -  assert 1 winner, 7 panics, and the global `dialect_lookup()` remains in a valid readable state afterwards (no torn `Arc`).

---

### 20. MEDIUM | tests/fusion_atomic_aliasing.rs
Tests only cover 2-arm fusion. Three-arm batches with cyclic read-write-read patterns on the same buffer are not exercised, nor is the `Uniform`-then-`ReadWrite` barrier need (Finding #5).

**Fix:** Add a test with arm0 `ReadOnly` "x", arm1 `ReadWrite` "x", arm2 `ReadOnly` "x"  -  assert barriers after arm0 and arm1. Add another test for the Uniform→ReadWrite path.

**Test hint:** 3-arm cyclic dependency: arm0 reads "a", arm1 writes "a" and reads "b", arm2 writes "b". Assert exactly 2 barriers in the fused entry, and assert no spurious barriers when arms are independent.

---

### 21. MEDIUM | tests/opaque_payload_endian.rs
`canonical_regex_flags` adversarial coverage is limited to small Unicode strings (≤ a few codepoints). There is no test for very large inputs that could expose O(n²) deduplication behaviour or allocator stress.

**Fix:** Add a test that generates a 1MB string of random flag chars, calls `canonical_regex_flags`, and asserts it completes in < 100ms and the output length is ≤ unique codepoint count.

**Test hint:** `canonical_regex_flags` on a 10MB string of alternating 'a' and 'b'  -  must not stack-overflow, must not take > 100ms, and output length must be 1.

---

## Summary

**Finding count by severity:** 2 CRITICAL, 9 HIGH, 10 MEDIUM.

**Top-3 highest-impact holes to escalate first:**

1. **Wire format omits `non_composable_with_self` (Finding #1).** This breaks the fundamental `decode(encode(p)) == p` invariant and causes execution-plan / fingerprint cache collisions between composable and non-composable programs. Every program using the self-composition flag is silently corrupted on round-trip.

2. **Self-composition gate bypassed when `entry_op_id = None` (Finding #2).** Two copies of a parser program that forgot to set an `entry_op_id` will fuse together and corrupt shared workgroup memory. The safety gate is present in the code but has a trivial bypass.

3. **`BufferDecl::workgroup` constructor allows zero-count without assertion (Finding #3).** The F-IR-05 fix added a panic to `with_count` but left the `workgroup` constructor as an unguarded backdoor. Parser builders can construct invalid zero-count workgroup buffers that crash at GPU dispatch time instead of at the call site.
