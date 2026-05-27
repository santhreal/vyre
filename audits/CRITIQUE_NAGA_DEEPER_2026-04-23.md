# Deep Audit  -  Naga Lowering Silent Correctness Hazards (Deeper Pass)

**Date:** 2026-04-23  
**Scope:** `vyre-driver-wgpu/src/lowering/naga_emit/*`  
**Goal:** Surface every silent correctness hazard the 2026-04-22 audit missed, focusing on the six mandated angles. A "silent" hazard is one that (a) passes `cargo test`, (b) passes naga validation, and (c) produces wrong runtime results or drops semantic guarantees without crashing.

---

## Meta-observation: `#[non_exhaustive]` makes compile-time exhaustiveness impossible in downstream crates

Every IR enum used by the emitter (`Expr`, `Node`, `BinOp`, `UnOp`, `AtomicOp`, `DataType`, `MemoryKind`, `BufferAccess`) is marked `#[non_exhaustive]` in its defining crate. Rust **requires** a `_ =>` catch-all in any `match` inside `vyre-driver-wgpu`. The 2026-04-22 audit listed several catch-all removals as "fixed" (FINDING-01, 10, 11, 21). Those removals are **structurally impossible** without removing `#[non_exhaustive]` from the upstream enums. The correct audit target is therefore not "does a catch-all exist?" (it must) but "does the catch-all silently default to a wrong value instead of returning a loud error?"  
**Verdict:** All existing catch-alls in `naga_emit/` now return `Err(...)` with `Fix:` messaging. No silent default remains. The architectural tension itself is noted below as a separate finding.

---

## Angle 1  -  `_ =>` catch-all on non_exhaustive enums

### FINDING-51: Constant-folding catch-alls (`fold_expr`, `fold_binary_literal`, `fold_unary_literal`, `fold_cast_literal`) silently skip new variants
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:57, 1223, 1251, 1258, 1260, 1300, 1331, 1339, 1367, 1369, 1392`  
**Description:** Every folding function uses `_ => None` for `Expr`, `BinOp`, `UnOp`, and `Cast` targets. Because the enums are `#[non_exhaustive]`, these arms are required, but they mean any newly-added variant (e.g. a future `Expr::LitU64` or `BinOp::DotProduct`) will silently bypass constant folding. For integer ops this is especially dangerous: WGSL's shader-creation rules reject constant-expression overflow. If a new wrapper variant prevents a literal `u32 + u32` from being folded, the shader may fail naga validation with an opaque overflow error that points at the wrong line.  
**Fix:** Replace `_ => None` in fold helpers with an explicit `panic!("unfoldable variant ...")` or move the fold tables into `vyre-spec`/`vyre-foundation` (same crate as the enum definitions) where exhaustiveness is enforced at compile time and no catch-all is required.  
**Test hint:** Add a proc-macro or `static_assertions` compile test that every `BinOp`/`UnOp` variant appears in `fold_binary_literal`/`fold_unary_literal`.

---

## Angle 2  -  `unwrap_or_else(|| naga::Expression::<some default>)`

### FINDING-52: No `naga::Expression` default patterns exist, but `mod.rs:369` silently defaults array stride to 4 bytes
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:369`  
**Description:** `buffer.element.size_bytes().unwrap_or(4)` chooses 4 bytes whenever `size_bytes()` returns `None`. All types currently accepted by `scalar_type` have known sizes, so this path is dead code today. If `scalar_type` is ever extended to accept `DataType::Opaque` or `DataType::Tensor` without updating this line, the emitted array will have the wrong stride. Because the error is suppressed by `unwrap_or`, naga validation may pass (the type is still sized) but the shader will access memory at the wrong offsets.  
**Fix:** Replace `unwrap_or(4)` with `ok_or_else(|| LoweringError::invalid("cannot determine array stride for ..."))`.  
**Test hint:** Construct a `BufferDecl` with a type whose `size_bytes()` is `None` and assert the lowering returns a stride-related error before naga validation.

---

## Angle 3  -  `Expr::Cast` target coverage

### FINDING-53: `DataType::U64` cast has no lowering arm despite being a valid scalar type
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:417-471` (`Expr::Cast` match)  
**Description:** `scalar_type` accepts `DataType::U64` and maps it to `vec2_u32_ty`. A program can declare a U64 buffer, load from it, and even pass the value through `BinOp::Add` (see FINDING-55). But `Expr::cast(DataType::U64, some_expr)` hits the `other => unsupported_type` catch-all. This is inconsistent: the type is valid for declaration and load, yet cast-to-U64 is rejected.  
**Fix:** Add a `DataType::U64` arm to `emit_expr` for `Cast` that emits the two-component `vec2<u32>` construction (low word from `As { kind: Uint }`, high word zero-extended) or reject with a precise "U64 cast requires vec2<u32> emulation pass" error until the emulation is implemented.  
**Test hint:** `Program::new(..., vec![Node::let_bind("x", Expr::cast(DataType::U64, Expr::u32(1)))])` must lower successfully or emit an actionable error containing "U64 cast".

### FINDING-54: `emit_bool_from_handle` rejects `DataType::F32`, blocking float-to-bool casts and float predicates
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:70-97`  
**Description:** `emit_bool_from_handle` handles `Bool`, `I8/16/32`, and `U8/16/32`, but not `F32`. This means:  
1. `Expr::cast(DataType::Bool, Expr::f32(1.0))` (non-literal) fails lowering.  
2. Any future use of `emit_bool_expr` on an F32 expression (e.g. `Node::If { cond: Expr::f32(1.0), ... }`) fails.  
WGSL supports `f32 != 0.0` for bool coercion. The omission forces front-ends to insert manual comparison nodes, which is a leaking abstraction.  
**Fix:** Add `DataType::F32 => { let zero = self.append_expr(Expression::Literal(Literal::F32(0.0))); ... Binary { op: NotEqual, left: value, right: zero } }`.  
**Test hint:** `emit_expr(&Expr::cast(DataType::Bool, Expr::f32(1.0)))` must produce a valid naga handle.

### FINDING-55: `fold_cast_literal` silently loses vector and wide-integer literal folds
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:1373-1394`  
**Description:** `fold_cast_literal` has explicit arms for `U32`, `I32`, `F32`, and `Bool` scalar literals only. Casts to `Vec2U32`, `Vec4U32`, or `U64` fall through to `_ => None`. An optimizer that splats a scalar literal into a vector and then casts it will leave constant-evaluable work in the shader. While harmless for correctness today, it wastes instruction slots and, for U64, could mask the fact that no U64 cast path exists (FINDING-53).  
**Fix:** Add explicit `Vec2U32` / `Vec4U32` arms that construct `Expr::Opaque` or reject, and a `U64` arm that rejects with a precise message.  
**Test hint:** Assert `fold_cast_literal(&DataType::Vec2U32, &Expr::LitU32(1))` returns `None` with a logged reason, not silent fall-through.

---

## Angle 4  -  `Expression::Load` ordering / `MemoryOrdering`

### FINDING-56: `Expr::Atomic` carries no `MemoryOrdering` field, contradicting the architecture contract and silently dropping all ordering guarantees
**Severity:** CRITICAL  
**Location:** `vyre-foundation/src/ir_inner/model/generated.rs:47` (IR definition) and `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:500-338` (lowering)  
**Description:** `docs/memory-model.md:33` states: "`Expr::Atomic` carries a `MemoryOrdering`" and lists `Relaxed`, `Acquire`, `Release`, `AcquireRelease`, `SeqCst`. The actual IR definition has **no such field**. The emitter therefore has no ordering to honor; every atomic is lowered to `naga::Statement::Atomic` with no memory-semantics annotation.  
In WGSL, `atomicLoad`/`atomicStore`/`atomicAdd` have **implicit relaxed semantics** (the WGSL spec does not expose ordering parameters). In SPIR-V, naga maps these to `OpAtomicIAdd` with `MemorySemantics None` or `Relaxed`, depending on the backend writer. A vyre program assuming `SeqCst` (the architecture-promised default) will observe reordered atomic operations on drivers that optimize aggressively. This is a **silent semantics mismatch**: the IR documentation promises strong ordering, the emitted shader provides weak ordering, and no error is raised at any stage.  
**Fix:** Add `ordering: MemoryOrdering` to `Expr::Atomic`, thread it through every IR visitor/transform in `vyre-foundation`, and map it in the emitter:  
- `Relaxed` → lower as today.  
- `Acquire` / `Release` / `AcqRel` / `SeqCst` → reject with a clear error until the WGSL backend gains `atomicLoad(weak, storage, workgroup, acquire/release)` support (WGSL future) or route through the SPIR-V backend which can express `MemorySemantics` directly.  
**Test hint:** Construct a program with `Expr::Atomic { ordering: MemoryOrdering::SeqCst, .. }` and assert that lowering either emits the correct SPIR-V `MemorySemantics` mask or returns an actionable `Fix:` error explaining the WGSL limitation.

---

## Angle 5  -  `Node::Barrier` lowered with correct scope

### FINDING-57: `Node::Barrier` always emits `STORAGE | WORK_GROUP`, over-synchronizing workgroup-local-only code and precluding subgroup barriers
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:118-129`  
**Description:** The emitter unconditionally emits `Statement::Barrier(naga::Barrier::STORAGE | naga::Barrier::WORK_GROUP)`. For a program that only touches `MemoryKind::Shared` buffers, this is safe for correctness (it synchronizes more memory than necessary) but it is a performance regression. The silent hazard is **forward-compatibility**: if vyre ever introduces a `subgroup_barrier` intrinsic that maps to `Node::Barrier`, the combined barrier will **omit** `Barrier::SUB_GROUP` (naga 25) / `Barrier::SUBGROUP` (naga 24), causing invocations within a subgroup to diverge without synchronization.  
**Fix:** Add a `scope: BarrierKind` field to `Node::Barrier` (or a dedicated `Node::SubgroupBarrier`) and map each to the precise naga flag:  
- `Workgroup` → `Barrier::WORK_GROUP`  
- `Storage` → `Barrier::STORAGE`  
- `Subgroup` → `Barrier::SUB_GROUP`  
**Test hint:** Emit a program with only `BufferDecl::workgroup` and a `Node::Barrier`, then inspect the WGSL output and assert it contains only `workgroupBarrier()` (not `storageBarrier()`).

---

## Angle 6  -  `Expr::Opaque` dispatch to registered extension lowering

### FINDING-58: `Expr::Opaque` and `Node::Opaque` downcast to `&dyn WgpuEmitExpr` / `&dyn WgpuEmitNode` is structurally impossible and can never succeed
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:528` and `node.rs:227`  
**Description:** The dispatch code does:  
```rust
if let Some(wgpu_ext) = ext.as_any().downcast_ref::<&dyn WgpuEmitExpr>() {
    return wgpu_ext.wgpu_emit_expr(self);
}
```
`as_any()` returns `&dyn Any` for the **concrete struct** stored in the `Arc<dyn ExprNode>`. `Any::downcast_ref::<T>()` checks whether the concrete TypeId matches `T`. The concrete TypeId is `MyExtensionStruct`, whereas `T` is `&dyn WgpuEmitExpr` (a fat-pointer type). These TypeIds are never equal, so the downcast **always returns `None`**.  
There are **zero** implementations of `WgpuEmitExpr` or `WgpuEmitNode` anywhere in the workspace, and no tests exercise the success path. The bug is latent: any extension author who implements the trait and calls `Expr::opaque(my_ext)` will receive the misleading error:  
> "unsupported opaque expression `my_ext` in wgpu lowering. Fix: implement WgpuEmitExpr for this extension."  
even though the trait **is** implemented.  
**Fix:** Change the extension contract. Options:  
1. Add `fn as_wgpu_emit_expr(&self) -> Option<&dyn WgpuEmitExpr>` to the `ExprNode` trait (default `None`).  
2. Use the `inventory` registry (already used for `WgpuBinOpRegistration`, `WgpuUnOpRegistration`, etc.) keyed by `extension_kind()`.  
**Test hint:** Register a dummy `WgpuEmitExpr` implementation in a test-only inventory entry, construct `Expr::Opaque(dummy)`, and assert that `emit_expr` invokes `wgpu_emit_expr` instead of returning the unsupported error.

---

## Cross-cutting finding: U64 emulation is declared but not implemented  -  arithmetic is silently wrong

### FINDING-59: `DataType::U64` is accepted by `scalar_type` but all arithmetic lowerings treat it as native `vec2<u32>`, producing component-wise instead of carry-propagation results
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:272` (type mapping) and `expr.rs:933-1148` (binop lowering)  
**Description:** `DataType::U64` maps to `vec2_u32_ty`. `expr_type` for `Expr::Load` from a U64 buffer returns `DataType::U64`. When the program performs `BinOp::Add` on two U64 values, `emit_binop` emits `Expression::Binary { op: Add, left, right }` where both handles are `vec2<u32>`. In WGSL this is component-wise vector addition: `(low1, high1) + (low2, high2)` with **no carry from low to high word**. The result is mathematically wrong for every addition where `low1 + low2 >= 2^32`.  
The same breakage affects: `Sub`, `Mul`, `Div`, `Mod`, `Shl`, `Shr`, `RotateLeft`, `RotateRight`, `Min`, `Max`, `AbsDiff`, `SaturatingAdd`, `SaturatingSub`, `SaturatingMul`. Only bitwise ops (`BitAnd`, `BitOr`, `BitXor`) and equality (`Eq`, `Ne`) are correct because they are naturally component-wise.  
Because naga validation accepts vector arithmetic and the WGSL shader runs without error, this is a **silent correctness hazard**  -  the program produces wrong numeric results.  
**Fix:** Either reject `BinOp` on `DataType::U64` in `emit_binop` with "U64 arithmetic requires emulation pass before wgpu lowering", or implement the full `vec2<u32>` emulation with carry/overflow handling.  
**Test hint:**  
```rust
let program = Program::new(
    vec![BufferDecl::storage("a", 0, BufferAccess::ReadWrite, DataType::U64)],
    [1, 1, 1],
    vec![Node::store("a", Expr::u32(0), Expr::add(Expr::load("a", Expr::u32(0)), Expr::load("a", Expr::u32(0))))],
);
```
Run the shader with inputs `0x00000001_00000000` and `0x00000001_00000000`. Correct U64 result: `0x00000002_00000000`. Component-wise vec2 result: `0x00000002_00000000` (coincidentally correct). Now test with `0x00000000_FFFFFFFF` + `0x00000000_00000001`. Correct: `0x00000001_00000000`. Component-wise: `0x00000001_00000000` (still correct by accident). Test with `0xFFFFFFFF_00000000` + `0x00000001_00000000`. Correct: `0x00000000_00000000` with overflow. Component-wise: `0x00000000_00000000` (high word correct, low word overflows to 0  -  actually correct again!). Hmm, let me pick a better case: `0x00000000_80000000` + `0x00000000_80000000`. Correct U64: `0x00000001_00000000`. Component-wise vec2: `0x00000001_00000000` (high word 0+0=0, but we got 1? No, 0x80000000 + 0x80000000 = 0x100000000, which in u32 wraps to 0x00000000 with carry 1. But component-wise vector add doesn't carry to the high component. So result is `(0x00000000, 0x00000000)` = 0. Correct result is `0x00000001_00000000`. The test would show 0 vs correct value.  
Assert the backend produces the correct value or rejects the program before dispatch.

---

## Summary Table

| # | Severity | File:Line | Issue | Escalate |
|---|----------|-----------|-------|----------|
| 51 | MEDIUM | `expr.rs:57, 1223, 1251, 1258, 1260, 1300, 1331, 1339, 1367, 1369, 1392` | Fold-helper `_ => None` catch-alls silently skip new variants | No |
| 52 | MEDIUM | `mod.rs:369` | `unwrap_or(4)` silently defaults array stride | No |
| 53 | HIGH | `expr.rs:417-471` | `Cast` missing `DataType::U64` arm | No |
| 54 | HIGH | `expr.rs:70-97` | `emit_bool_from_handle` rejects `F32` | No |
| 55 | MEDIUM | `expr.rs:1373-1394` | `fold_cast_literal` loses vector/wide folds | No |
| 56 | CRITICAL | `generated.rs:47` + `expr.rs:500-338` | `Expr::Atomic` lacks `MemoryOrdering`; architecture contract violated | **Yes** |
| 57 | MEDIUM | `node.rs:118-129` | `Node::Barrier` always combined `STORAGE \| WORK_GROUP` | No |
| 58 | CRITICAL | `expr.rs:528` + `node.rs:227` | `Opaque` downcast to trait-object reference is impossible | **Yes** |
| 59 | CRITICAL | `mod.rs:272` + `expr.rs:933-1148` | `DataType::U64` arithmetic silently wrong (component-wise vec2) | **Yes** |

**New finding count by severity:**  
- CRITICAL: 3  
- HIGH: 2  
- MEDIUM: 4  

**Top-3 to escalate:**
1. **FINDING-58**  -  Broken `Expr::Opaque` / `Node::Opaque` downcast makes the entire extension-expression pipeline unusable. The error message actively gaslights extension authors.
2. **FINDING-56**  -  `MemoryOrdering` is documented but absent from the IR; every atomic silently gets backend-default (likely relaxed) semantics instead of the promised `SeqCst`.
3. **FINDING-59**  -  `DataType::U64` arithmetic compiles and runs but produces mathematically incorrect results because carry is never propagated between the emulated low/high `vec2<u32>` words.

---

*Skipped: All findings marked `[fixed 2026-04-23]` in the prior audit were verified in source; where the fix description claimed removal of a `_ =>` arm that is mandated by `#[non_exhaustive]`, the arm was confirmed to still exist but now returns a loud `Err` instead of a silent default. Those changes are accepted as mitigated.*
