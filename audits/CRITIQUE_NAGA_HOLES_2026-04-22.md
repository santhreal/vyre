# Deep Audit  -  Naga Lowering Coverage Holes (Redispatch)

**Date:** 2026-04-22  
**Scope:** `vyre-driver-wgpu/src/lowering/naga_emit/*`, `vyre-foundation/src/ir_inner/model/*`, `vyre-intrinsics/src/hardware/subgroup_*`  
**Goal:** Every IR construct used by `surgec` must lower to a correct naga `Statement`/`Expression`. Any `_ =>` catch-all, no-op, string-WGSL arm, or missing variant is a finding.

---

## 1. Expr Coverage

### FINDING-01: `emit_expr` catch-all swallows future variants silently [fixed 2026-04-23]
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:240-244`  
**Variant:** `_ =>` fallback on `Expr`  
**Description:** The `_ =>` arm returns a generic error for any expression variant not explicitly matched. Because `Expr` is `#[non_exhaustive]`, new variants added upstream (e.g. `Expr::SubgroupBroadcast`) compile successfully but silently hit this catch-all at runtime instead of failing at compile time.  
**Minimal Program:**
```rust
// Hypothetical future variant
Expr::SubgroupBroadcast { value: Box::new(Expr::u32(1)) }
```
**Fix:** Removed the fallback arm in `FunctionBuilder::emit_expr` so exhaustiveness over every `Expr` variant is enforced at compile time in this same match. Unknown variants now cannot be silently accepted, and extension-aware `Expr::Opaque` still follows the actionable path through `WgpuEmitExpr` dispatch with the `unsupported opaque expression` diagnostic.

---

### FINDING-02: `Expr::SubgroupBallot` unconditionally rejected
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:206-217`  
**Variant:** `Expr::SubgroupBallot`  
**Description:** The arm returns a hard error citing Naga 25+ gating. The workspace is pinned to naga 24.0.0, so the expression has *no* structural lowering path in the default build. `vyre-intrinsics` ships `subgroup_ballot` and it is **not** feature-gated out of the default build (only the test inventory is gated).  
**Minimal Program:**
```rust
Program::new(
    vec![
        BufferDecl::storage("cond", 0, BufferAccess::ReadOnly, DataType::U32),
        BufferDecl::output("out", 1, DataType::U32),
    ],
    [64, 1, 1],
    vec![Node::store("out", Expr::u32(0), Expr::subgroup_ballot(Expr::eq(Expr::load("cond", Expr::u32(0)), Expr::u32(1))))],
)
```
**Fix:** Bump workspace naga pin to `>=25.0.0`, lower to `Expression::SubgroupOperation { op: SubgroupOperation::Ballot, .. }`, and gate on `Capabilities::SUBGROUP`.

---

### FINDING-03: `Expr::SubgroupShuffle` unconditionally rejected
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:218-224`  
**Variant:** `Expr::SubgroupShuffle`  
**Description:** Same as FINDING-02. No structural AST arm. The `vyre-intrinsics::hardware::subgroup_shuffle` builder is callable in the default build and will hit this error at lowering time.  
**Minimal Program:**
```rust
Expr::subgroup_shuffle(Expr::load("values", Expr::var("idx")), Expr::load("lanes", Expr::var("idx")))
```
**Fix:** Same as FINDING-02  -  use naga 25+ `SubgroupOperation::Shuffle`.

---

### FINDING-04: `Expr::SubgroupAdd` unconditionally rejected
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:225-230`  
**Variant:** `Expr::SubgroupAdd`  
**Description:** Same as FINDING-02. `vyre-intrinsics::hardware::subgroup_add` is callable without the `subgroup-ops` feature and will fail here.  
**Minimal Program:**
```rust
Expr::subgroup_add(Expr::load("values", Expr::var("idx")))
```
**Fix:** Same as FINDING-02  -  use naga 25+ `SubgroupOperation::Add`.

---

### FINDING-05: `Expr::Cast` missing arms for Bool, vectors, and sub-32-bit integers
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:95-110`  
**Variant:** `Expr::Cast { target: DataType::Bool | DataType::Vec2U32 | DataType::Vec4U32 | DataType::U8 | DataType::U16 | DataType::I8 | DataType::I16, .. }`  
**Description:** Only `U32`, `I32`, and `F32` are handled. Every other target hits `LoweringError::unsupported_type`.  
**Minimal Program:**
```rust
Expr::cast(DataType::Bool, Expr::u32(1))
```
**Fix:** Add explicit arms:
- `Bool` → emit `expr != 0u` via `BinaryOperator::NotEqual`.
- `Vec2U32` / `Vec4U32` → emit `Expression::As` with appropriate vector type (or reject with a clear message until vector lowering is complete).
- `U8`/`U16`/`I8`/`I16` → emit `As { kind: Uint/Sint, convert: Some(1/2) }` if naga/WGSL supports them, otherwise reject with "8/16-bit scalar requires emulation pass".

---

### FINDING-06: `Expr::Atomic` result type hardcoded to `u32_ty` [fixed 2026-04-23]
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:178-204`  
**Variant:** `Expr::Atomic { op: AtomicOp::Add | And | Or | Xor | Min | Max | Exchange | LruUpdate, .. }` on an `i32` buffer  
**Description:** `Expression::AtomicResult` is created with `ty: self.module.types.u32_ty` for every op except `CompareExchange`. If the buffer element is `DataType::I32`, the pointer points to `atomic<i32>`, but the result type is `u32`. Naga validation rejects the type mismatch (`AtomicResult type does not match atomic scalar type`).  
**Minimal Program:**
```rust
Program::new(
    vec![BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::I32)],
    [1, 1, 1],
    vec![Node::let_bind("x", Expr::atomic_add("buf", Expr::u32(0), Expr::i32(1)))],
)
```
**Fix:** `FunctionBuilder::emit_atomic_expr` now derives the return type in `atomic_result_type` from the target buffer's declared element (`U32` -> `AtomicResult<...>` with `u32_ty`, `I32` -> `i32_ty`) and `ModuleBuilder::add_buffer` wraps atomic buffers as `atomic<...>` by element, so this is now enforced at declaration and lowering sites with precise `Fix:` diagnostics when unsupported element types are used.

---

### FINDING-07: `AtomicOp::CompareExchangeWeak` rejected despite being lowerable [fixed 2026-04-23]
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:169-176`  
**Variant:** `AtomicOp::CompareExchangeWeak`  
**Description:** Naga 24 has no separate "weak" compare-exchange, but the strong `AtomicFunction::Exchange { compare: Some(..) }` is semantically equivalent for GPU purposes. The emitter rejects it with `unsupported_op`.  
**Minimal Program:**
```rust
Expr::Atomic {
    op: AtomicOp::CompareExchangeWeak,
    buffer: Ident::from("buf"),
    index: Box::new(Expr::u32(0)),
    expected: Some(Box::new(Expr::u32(42))),
    value: Box::new(Expr::u32(7)),
}
```
**Fix:** `FunctionBuilder::emit_atomic_expr` now maps `AtomicOp::CompareExchangeWeak` onto the same `AtomicFunction::Exchange { compare: Some(...) }` path as `CompareExchange`, so lowering proceeds through the same compare-exchange structuring with an unchanged return contract.

---

### FINDING-08: `AtomicOp::FetchNand` rejected with generic error [fixed 2026-04-23]
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:169-176`  
**Variant:** `AtomicOp::FetchNand`  
**Description:** Naga 24 does not expose `FetchNand`. The emitter returns the generic `unsupported_op` error instead of explaining that this op requires emulation (`atomicAnd(ptr, ~value)`) or is unavailable on the current naga pin.  
**Minimal Program:**
```rust
Expr::Atomic { op: AtomicOp::FetchNand, buffer: Ident::from("buf"), index: Box::new(Expr::u32(0)), expected: None, value: Box::new(Expr::u32(1)) }
```
**Fix:** The atomic arm now emits an explicit `invalid` diagnostic for `FetchNand` in `FunctionBuilder::emit_atomic_expr`, with a concrete `Fix:` message directing users to emulate with `compare_exchange`/`atomicAnd` or upgrade naga.

---

### FINDING-09: `AtomicOp::Opaque` rejected  -  extension atomics have no dispatch path [fixed 2026-04-23]
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:169-176`  
**Variant:** `AtomicOp::Opaque(...)`  
**Description:** Extension atomic ops cannot be lowered because there is no `WgpuEmitAtomic` trait analogous to `WgpuEmitExpr`/`WgpuEmitNode`.  
**Minimal Program:** Any extension registering a custom `AtomicOp`.  
**Fix:** `FunctionBuilder::emit_atomic_expr` now dispatches `AtomicOp::Opaque(id)` to `emit_registered_atomic_op`, and `extension_ops.rs` now provides a matching extension registry path that returns the extension-specific handler or a clear `Fix:` error when no `WgpuAtomicOpRegistration` is present.

---

### FINDING-10: `emit_unary` catch-all swallows `Unpack4Low`, `Unpack4High`, `Unpack8Low`, `Unpack8High`, `Opaque` [fixed 2026-04-23]
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs: around emit_unary`  
**Status:** Fixed in `emit_unary` by adding explicit lowering branches for `UnOp::Unpack4Low`, `UnOp::Unpack4High`, `UnOp::Unpack8Low`, `UnOp::Unpack8High`, plus direct extension dispatch for `UnOp::Opaque`; regression coverage exists in `unary_unpack_ops_are_lowered` in `naga_findings_followup.rs`, with actionable `Fix:` diagnostics for unsupported input types.

---

### FINDING-11: `binary_operator` catch-all swallows subgroup `BinOp` variants [fixed 2026-04-23]
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/utils.rs:88`  
**Status:** Fixed by replacing the generic fallback with explicit subgroup/op-code-specific handling in `binary_operator` (returning actionable `Fix:` errors) and by implementing subgroup emission in `emit_binop`; this is guarded by `subgroup_binops_are_lowered_as_subgroup_statements`.

---

### FINDING-12: `expr_type` returns `Bool` for `Expr::Load` from a bool buffer, but emitter produces `u32` [fixed 2026-04-23]
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/utils.rs:106-117` and `expr.rs:70-73`  
**Variant:** `Expr::Load { buffer: "flags", .. }` where buffer element is `DataType::Bool`  
**Description:** `storage_scalar_type` stores bool buffers as `u32` arrays. `emit_expr` for `Load` emits `Expression::Load` from that `u32` array, so the handle is `u32`. However, `expr_type` reads the buffer declaration and returns `DataType::Bool`. Any downstream unary op that branches on type (e.g. `UnOp::LogicalNot`) will emit `UnaryOperator::LogicalNot` on a `u32` expression, causing naga validation to fail.  
**Minimal Program:**
```rust
Program::new(
    vec![BufferDecl::storage("flags", 0, BufferAccess::ReadOnly, DataType::Bool)],
    [1, 1, 1],
    vec![Node::let_bind("b", Expr::not(Expr::load("flags", Expr::u32(0))))],
)
```
**Fix:** The same bool-load normalization is now enforced in `FunctionBuilder::emit_expr`: after `Expression::Load` from a bool buffer, it now executes `emit_bool_from_handle` with `u32 != 0` and therefore `emit_bool_from_handle` contract in one place now aligns IR type expectations. `Fix:` messages on unknown buffers remain explicit in the same arm.

---

### FINDING-13: `Expr::BufLen` on workgroup buffer emits invalid `ArrayLength`
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:74-78`  
**Variant:** `Expr::BufLen { buffer: "scratch" }` where scratch is `MemoryKind::Shared`  
**Description:** `Expression::ArrayLength` is only valid for runtime-sized arrays. Workgroup arrays are emitted with `ArraySize::Constant`, so naga validation rejects `ArrayLength` on them.  
**Minimal Program:**
```rust
Program::new(
    vec![BufferDecl::workgroup("scratch", 64, DataType::U32)],
    [64, 1, 1],
    vec![Node::let_bind("n", Expr::buf_len("scratch"))],
)
```
**Fix:** For non-runtime-sized arrays, emit the static `count` as a `Literal::U32(buffer.count)` instead of `ArrayLength`.

---

### FINDING-14: `Expr::Fma` assumes `F32` regardless of operand types
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:111-122` and `utils.rs:128`  
**Variant:** `Expr::Fma { a, b, c }` with non-F32 operands  
**Description:** `expr_type` hardcodes `DataType::F32` for `Fma`. The emitter uses `MathFunction::Fma` without checking operand scalar kinds. If the IR ever carries `f16` or `f64` FMA, the emitted naga expression will have a type mismatch.  
**Minimal Program:** (Hypothetical) `Expr::fma(Expr::f16(1.0), Expr::f16(2.0), Expr::f16(3.0))`.  
**Fix:** Validate operand types in `expr_type` and reject non-F32 `Fma` with a precise error, or extend lowering to emit the correct `MathFunction::Fma` with typed literals.

---

## 2. Node Coverage

### FINDING-15: `Node::Region` silently flattens into parent block
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:99-110`  
**Variant:** `Node::Region { body, .. }`  
**Description:** The emitter calls `self.emit_nodes(body.as_slice())` directly, inlining the region's statements into the current `self.function.body`. No new naga `Block` is created. If `Region` is ever used for lexical scoping or boundary-based barrier semantics, those semantics are lost.  
**Minimal Program:**
```rust
Node::Region {
    generator: Ident::from("gen"),
    source_region: None,
    body: Arc::new(vec![Node::let_bind("x", Expr::u32(1))]),
}
```
**Fix:** Wrap the region body in a dedicated `Block` so it forms a lexical boundary in the naga IR.

---

### FINDING-16: `Node::Barrier` emits combined `STORAGE | WORK_GROUP` for every barrier
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:111-122`  
**Variant:** `Node::Barrier`  
**Description:** Both `workgroup_barrier` and `storage_barrier` from `vyre-intrinsics` produce `Node::Barrier`. The emitter always emits `Barrier::STORAGE | Barrier::WORK_GROUP`, making the two intrinsics indistinguishable. It also never emits `Barrier::SUBGROUP`, which is required for subgroup synchronization.  
**Minimal Program:**
```rust
// From vyre-intrinsics::hardware::storage_barrier
Node::barrier() // inside a storage_barrier program
```
**Fix:** Add a scope tag to `Node::Barrier` (e.g. `BarrierKind::Workgroup | Storage | Subgroup`) and map each to the precise naga `Barrier` flag.

---

### FINDING-17: `Node::Loop` double-evaluates `to` expression per iteration
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:127-184` and `248-317`  
**Variant:** `Node::Loop { to, .. }` where `to` contains a side-effect expression  
**Description:** The loop bound is evaluated once in the guard block (`emit_loop_guard_block`) and again in the continuing block (`emit_loop_continuing_block`). Because `Expr` is supposed to be pure this is usually benign, but `Expr::Atomic` is *not* pure  -  it performs a read-modify-write. A loop bound containing an atomic will execute the atomic **twice per iteration**.  
**Minimal Program:**
```rust
Program::new(
    vec![BufferDecl::storage("counter", 0, BufferAccess::ReadWrite, DataType::U32)],
    [1, 1, 1],
    vec![Node::Loop {
        var: "i",
        from: Expr::u32(0),
        to: Expr::atomic_add("counter", Expr::u32(0), Expr::u32(1)),
        body: vec![],
    }],
)
```
**Fix:** Hoist the `to` expression into a temporary local before the loop header:
```rust
let _vyre_to = self.emit_expr(to)?;
// use _vyre_to in both guard and continuing
```

---

### FINDING-18: `Node::Loop` error path loses saved function body
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:248-317`  
**Variant:** `Node::Loop` where `to` or body emission fails  
**Description:** `emit_loop_guard_block` and `emit_loop_continuing_block` use `std::mem::take` to swap `self.function.body`. If an intermediate `?` returns `Err`, the original body is never restored, leaving the builder in a corrupted state.  
**Minimal Program:**
```rust
Node::Loop {
    var: "i",
    from: Expr::u32(0),
    to: Expr::var("UNKNOWN"), // unknown local → emit_expr errors
    body: vec![],
}
```
**Fix:** Use a RAII guard or `match`/`finally` pattern to ensure `self.function.body = saved` before returning `Err`.

---

### FINDING-19: `Node::AsyncLoad / AsyncStore / AsyncWait` silently skipped [fixed 2026-04-23]
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:190-200`  
**Status:** Fixed by replacing the silent branch with a hard `invalid` return that is explicit about host-scheduling requirements, validated by `async_nodes_are_rejected_in_naga_emit`.

---

### FINDING-20: `Node::Trap / Node::Resume` silently skipped [fixed 2026-04-23]
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:201-208`  
**Status:** Fixed by rejecting this branch with an actionable `invalid` error and explicit `Fix:` messaging; regression test `trap_resume_nodes_are_rejected_in_naga_emit` validates this behavior.

---

### FINDING-21: `emit_node` catch-all swallows future Node variants [fixed 2026-04-23]
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:218-222`  
**Status:** Fixed by removing the `_` fallback and requiring explicit variant matches; unknown `Node` kinds now fail at compile-time.

---

### FINDING-22: `Node::If` does not validate boolean condition [fixed 2026-04-23]
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:81-97`  
**Status:** Already fixed by `Node::If` using `emit_bool_expr(cond)` before emission; that enforces predicate coercion and returns actionable diagnostics for unsupported condition types.

---

## 3. Type Lowering

### FINDING-23: `scalar_type` rejects every type except Bool, U32, I32, F32, Bytes, Array
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:204-211`  
**Variant:** `DataType::U8 | U16 | I8 | I16 | I64 | U64 | Vec2U32 | Vec4U32 | Vec {..} | F16 | BF16 | F64 | Tensor | Handle | TensorShaped | SparseCsr | SparseCoo | SparseBsr | F8E4M3 | F8E5M2 | I4 | FP4 | NF4 | DeviceMesh | Opaque`  
**Description:** The IR supports ~30 data types, but the naga emitter only knows how to build naga `TypeInner` for 5 of them. Any buffer or local declared with a missing type fails lowering.  
**Minimal Program:**
```rust
BufferDecl::storage("v", 0, BufferAccess::ReadOnly, DataType::Vec4U32)
```
**Fix:** Add explicit mappings:
- `Vec2U32` / `Vec4U32` → `TypeInner::Vector { size: Bi/Quad, scalar: u32 }`.
- `U64` → `TypeInner::Vector { size: Bi, scalar: u32 }` (low/high word emulation) with a doc comment.
- `F16` → `TypeInner::Scalar { kind: Float, width: 2 }` (requires `Capabilities::FLOAT16`).
- `F64` → reject with "F64 is not representable in WGSL 1.0; use F32 or emulate via vec2<u32>".
- All others → domain-specific rejection messages instead of generic `unsupported_type`.

---

### FINDING-24: `scalar_type` maps `DataType::Bytes` to `u32_ty`
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:207`  
**Variant:** `DataType::Bytes`  
**Description:** A byte buffer is emitted as `array<u32>` with stride 1. This is invalid  -  the array base type size (4) does not match the stride (1). Naga validation may reject, or the WGSL writer may emit nonsensical code.  
**Minimal Program:**
```rust
BufferDecl::storage("b", 0, BufferAccess::ReadOnly, DataType::Bytes)
```
**Fix:** Map `Bytes` to `array<u8>` if naga/WGSL supports it, otherwise reject with "Bytes buffer requires pack-to-u32 pre-pass before wgpu lowering."

---

### FINDING-25: `scalar_type` maps `DataType::Array { element_size }` to `u32_ty`
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:207`  
**Variant:** `DataType::Array { element_size: 16 }`  
**Description:** An array of 16-byte elements is emitted as `array<u32, stride=16>`. Naga may accept this for SPIR-V but the WGSL writer will emit `array<u32>` and ignore the stride, producing invalid shader text.  
**Minimal Program:**
```rust
BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::Array { element_size: 16 })
```
**Fix:** Create a struct type with the correct member layout and use it as the array base, or reject non-4-byte arrays until struct lowering is complete.

---

### FINDING-26: Only `vec3_u32_ty` is pre-registered; `Vec2U32` and `Vec4U32` have no handles
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:154-166`  
**Variant:** `DataType::Vec2U32 | DataType::Vec4U32`  
**Description:** The `TypeHandles` struct only contains `vec3_u32_ty` (used for builtins). Vector buffers cannot be declared because there is no handle to assign.  
**Minimal Program:**
```rust
BufferDecl::storage("v", 0, BufferAccess::ReadOnly, DataType::Vec4U32)
```
**Fix:** Pre-register `vec2_u32_ty` and `vec4_u32_ty` in `ModuleBuilder::new`.

---

## 4. Buffer Binding

### FINDING-27: `storage_access` catch-all mis-handles future access modes
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/utils.rs:55-62`  
**Variant:** `_ => StorageAccess::LOAD | StorageAccess::STORE`  
**Description:** The catch-all grants read-write access to any unrecognized `BufferAccess` variant. The doc comment in `vyre-spec/src/lib.rs:32` claims a `WriteOnly` variant exists, but it is missing from the enum. If it is ever added, it would incorrectly receive `LOAD` permission here.  
**Minimal Program:** (Once `WriteOnly` exists) `BufferAccess::WriteOnly` → gets `LOAD | STORE`.  
**Fix:** Remove `_ =>` and enumerate every variant explicitly. Add `BufferAccess::WriteOnly` to the spec and map it to `StorageAccess::STORE`.

---

### FINDING-28: `BufferAccess::WriteOnly` is documented but missing from the enum
**Severity:** MEDIUM  
**Location:** `vyre-spec/src/buffer_access.rs` (enum definition) vs `vyre-spec/src/lib.rs:32` (doc comment)  
**Variant:** `BufferAccess::WriteOnly`  
**Description:** The spec doc claims `WriteOnly` exists, but the enum only has `ReadOnly`, `ReadWrite`, `Uniform`, `Workgroup`. This forces users to use `ReadWrite` for write-only output buffers, losing cache-coherence hints.  
**Minimal Program:** N/A  -  cannot construct a non-existent variant.  
**Fix:** Add `WriteOnly` to `BufferAccess` and map it to `StorageAccess::STORE` in `storage_access`.

---

### FINDING-29: `address_space` rejects `MemoryKind::Persistent`
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/utils.rs:23-39`  
**Variant:** `MemoryKind::Persistent`  
**Description:** Persistent memory is intended for host-side async I/O. If a Program accidentally includes a persistent buffer, the emitter returns a generic error.  
**Minimal Program:**
```rust
BufferDecl::storage("p", 0, BufferAccess::ReadOnly, DataType::U32).with_kind(MemoryKind::Persistent)
```
**Fix:** Either map `Persistent` to `AddressSpace::Storage { access: LOAD }` with a warning, or reject with "Persistent buffers must be resolved to async loads before wgpu lowering."

---

### FINDING-30: `binding` returns `None` for `MemoryKind::Push`
**Severity:** LOW  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/utils.rs:42-53`  
**Variant:** `MemoryKind::Push`  
**Description:** Push constants in WGSL require a binding. The emitter assigns `AddressSpace::PushConstant` but sets `binding: None`. Naga may accept this for SPIR-V but the WGSL writer may produce invalid output.  
**Minimal Program:**
```rust
BufferDecl::storage("pc", 0, BufferAccess::ReadOnly, DataType::U32).with_kind(MemoryKind::Push)
```
**Fix:** Assign a binding for push constants or reject with a clear message that push constants are not yet supported in the WGSL path.

---

## 5. Subgroup Ops

### FINDING-31: `subgroup_intrinsics.rs` contains string-based WGSL emission helpers
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/subgroup_intrinsics.rs:110-125` and `134-146`  
**Variant:** `emit_wgsl_for`, `emit_shuffle_xor`, `emit_sram_scan_fallback`  
**Description:** These functions assemble raw WGSL source strings. The 0.7 gate mandates **naga-AST only**  -  no string WGSL may survive inside `vyre-driver-wgpu/src/lowering/**`. Even though these helpers are currently unused in the naga path, their presence is a liability for a future refactor accident.  
**Minimal Program:** Any code path calling `emit_wgsl_for(SubgroupOp::Add, "x")`.  
**Fix:** Delete the string-based helpers. If a fallback path is needed, build it using `FunctionBuilder` and naga `Statement`/`Expression` directly.

---

### FINDING-32: `SubgroupCaps` is probed but never consumed by the emitter
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/subgroup_intrinsics.rs:86-103`  
**Variant:** `SubgroupCaps::from_adapter`  
**Description:** The adapter capability check is dead code. The naga emitter never receives `SubgroupCaps` and therefore never routes between native subgroup AST and SRAM-scan fallback. It always rejects subgroup expressions.  
**Minimal Program:** Run on an RTX 5090 with `Features::SUBGROUP` enabled  -  `subgroup_add` still fails.  
**Fix:** Pass `SubgroupCaps` into `emit_module` and use it to choose between `Expression::SubgroupOperation` and a fallback block built from naga `Statement`s.

---

### FINDING-33: IR lacks `Expr` variants for `SubgroupMax`, `SubgroupMin`, `SubgroupInclusiveAdd`, etc.
**Severity:** HIGH  
**Location:** `vyre-foundation/src/ir_inner/model/generated.rs` (Expr definition) vs `vyre-driver-wgpu/src/lowering/subgroup_intrinsics.rs:26-41` (`SubgroupOp` enum)  
**Variant:** `SubgroupOp::Max | Min | InclusiveAdd | ExclusiveAdd | ShuffleXor | Broadcast`  
**Description:** `SubgroupOp` defines 7 ops, but the IR only has `Expr::SubgroupAdd`, `SubgroupShuffle`, and `SubgroupBallot`. The remaining 4 ops cannot be expressed in vyre IR at all.  
**Minimal Program:** N/A  -  cannot construct.  
**Fix:** Extend the `Expr` enum with the missing subgroup variants, or consolidate on a single `Expr::Subgroup { op: SubgroupOp, .. }` node.

---

## 6. Control Flow

### FINDING-34: `Node::Loop` bound expression evaluated twice per iteration
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:248-317`  
**Variant:** `Node::Loop { to, .. }`  
**Description:** (Same as FINDING-17, framed under control flow.) The `to` expression is evaluated in both the guard block and the continuing block. Side-effect expressions (atomics) execute twice.  
**Minimal Program:** (See FINDING-17.)  
**Fix:** (See FINDING-17.)

---

### FINDING-35: `emit_child_block` loses saved body on error
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:319-325`  
**Variant:** `emit_child_block`  
**Description:** Uses the same `std::mem::take` pattern as the loop emission. If `emit_nodes` fails, the saved body is lost.  
**Minimal Program:** Any `Node::If` or `Node::Loop` whose inner body references an unknown variable.  
**Fix:** Restore `self.function.body = saved` in the error path, or use a RAII guard.

---

## 7. Atomics

### FINDING-36: `scan_atomic_targets` does not recurse into `Node::Opaque` [fixed 2026-04-23]
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:365-410`  
**Variant:** `Node::Opaque` containing `Expr::Atomic`  
**Description:** The pre-pass that marks buffers as atomic only scans explicit `Node`/`Expr` variants. If an extension node wraps an `Expr::Atomic`, the buffer is not wrapped in `atomic<...>`, and naga validation rejects the atomic statement.  
**Minimal Program:** An opaque node that internally builds `Expr::atomic_add("buf", ..)`.  
**Fix:** `scan_atomic_targets` now routes `visit_opaque_node` through `visit_preorder` into `WgpuScanAtomicNode` via `scan_atomic_targets_expr` and `Expr` extension downcast, so extension-backed atomics are discovered before `ModuleBuilder::add_buffer`; this is covered by `atomic_scan_collects_targets_from_opaque_*` tests with `Fix:`-style failures for unsupported opaque extensions.

---

### FINDING-37: `scan_atomic_targets_expr` uses `_ => {}` catch-all [fixed 2026-04-23]
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:415-458`  
**Variant:** `_ => {}` in `scan_atomic_targets_expr`  
**Description:** Any future `Expr` variant that contains nested expressions (e.g. a new vector constructor) will not be recursed into, potentially missing atomic targets.  
**Minimal Program:** Any future Expr variant holding sub-expressions.  
**Fix:** The scan visitor implementation now covers every current `Expr` callback path in `AtomicTargetScanner` and does not leave a generic `_` arm, making the traversal exhaustive for the current IR shape and preventing silent non-recursion from future extension nodes.

---

### FINDING-38: `AtomicOp::CompareExchangeWeak` not lowered
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:169-176`  
**Variant:** `AtomicOp::CompareExchangeWeak`  
**Description:** (Same as FINDING-07.)  
**Fix:** (See FINDING-07.)

---

### FINDING-39: `AtomicOp::FetchNand` not lowered
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:169-176`  
**Variant:** `AtomicOp::FetchNand`  
**Description:** (Same as FINDING-08.)  
**Fix:** (See FINDING-08.)

---

### FINDING-40: `AtomicOp::Opaque` not lowered
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:169-176`  
**Variant:** `AtomicOp::Opaque`  
**Description:** (Same as FINDING-09.)  
**Fix:** (See FINDING-09.)

---

## 8. Region Chain

### FINDING-41: `Node::Region` flattens without a `Block` wrapper
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/node.rs:99-110`  
**Variant:** `Node::Region`  
**Description:** (Same as FINDING-15.) Region nodes are inlined into the parent statement list. If the region defines locals, they leak into the parent scope.  
**Fix:** (See FINDING-15.)

---

## 9. Catch-alls & Forward Compatibility

### FINDING-42: `binary_operator` catch-all
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/utils.rs:88`  
**Variant:** `BinOp` catch-all  
**Description:** (Same as FINDING-11.)  
**Fix:** (See FINDING-11.)

---

### FINDING-43: `storage_access` catch-all
**Severity:** HIGH  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/utils.rs:61`  
**Variant:** `BufferAccess` catch-all  
**Description:** (Same as FINDING-27.)  
**Fix:** (See FINDING-27.)

---

### FINDING-44: `address_space` catch-all
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/utils.rs:35-38`  
**Variant:** `MemoryKind` catch-all  
**Description:** (Same as FINDING-29.)  
**Fix:** (See FINDING-29.)

---

### FINDING-45: `expr_type` catch-all
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/utils.rs:139-141`  
**Variant:** `Expr` catch-all in type inference  
**Description:** Returns a generic error for unknown expression variants. Also masks the fact that `Expr::Atomic` type inference is broken for i32 (returns U32).  
**Fix:** Remove `_ =>`, add explicit arms for every variant, and fix `Expr::Atomic` to derive type from buffer element.

---

## 10. Miscellaneous

### FINDING-46: `add_buffer` clamps zero-sized static arrays to size 1
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:270`  
**Variant:** `BufferDecl::workgroup("scratch", 0, DataType::U32)`  
**Description:** `count == 0` is silently replaced with `1`. A user requesting a zero-sized workgroup array gets a 1-element array instead of an error.  
**Minimal Program:**
```rust
BufferDecl::workgroup("scratch", 0, DataType::U32)
```
**Fix:** Reject zero-sized static arrays with: "Fix: workgroup/local buffer count must be > 0."

---

### FINDING-47: `emit_module` validates with `Capabilities::all()`
**Severity:** LOW  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:96-106`  
**Variant:** N/A  
**Description:** The emitter validates the module against every naga capability. If the target adapter lacks a capability (e.g. `SUBGROUP`), the shader passes here but fails at pipeline creation time.  
**Minimal Program:** Any shader using subgroup ops on an adapter without `Features::SUBGROUP`.  
**Fix:** Accept a `Capabilities` mask derived from the live adapter and validate against that subset.

---

### FINDING-48: `fold_expr` misses `Fma` and other pure literal variants [fixed 2026-04-23]
**Severity:** LOW  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:14-43`  
**Variant:** `Expr::Fma { a: LitF32, b: LitF32, c: LitF32 }`  
**Description:** Constant folding does not handle `Fma`, leaving constant-evaluable arithmetic in the shader. This is harmless but wastes instruction slots.  
**Minimal Program:**
```rust
Expr::fma(Expr::f32(1.0), Expr::f32(2.0), Expr::f32(3.0))
```
**Fix:** `fold_expr` now has a dedicated `Expr::Fma` branch that folds all three `LitF32` operands to a single constant `Expr::LitF32(a.mul_add(b, c))`, reducing backend work and preventing constant-folding-related validation edge cases.

---

### FINDING-49: `BinOp::RotateLeft/RotateRight` mask is hardcoded to 31 [fixed 2026-04-23]
**Severity:** MEDIUM  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:601-642`  
**Variant:** `BinOp::RotateLeft | BinOp::RotateRight`  
**Description:** The shift-count mask is `Literal::U32(31)`. This is correct for 32-bit types but will be wrong if 64-bit rotate is ever added.  
**Minimal Program:** (Hypothetical U64 rotate) `Expr::rotate_left(Expr::u64(1), Expr::u64(1))`.  
**Fix:** `FunctionBuilder::emit_binop` now computes rotate width through `rotate_width_bits(&DataType)` and builds the mask from that width, so the shift mask is `width_bits - 1` for `U32/I32/U64/I64` and vectors are rejected with a clear `Fix:` message.

---

### FINDING-50: `emit_expr` for `Expr::Load` from bool buffer does not cast back to bool [fixed 2026-04-23]
**Severity:** CRITICAL  
**Location:** `vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:70-73`  
**Variant:** `Expr::Load` from `DataType::Bool` buffer  
**Description:** (Same as FINDING-12.)  
**Fix:** `Expr::Load` now checks `DataType::Bool` buffer elements and immediately wraps loaded `u32` values in `u32 != 0`, enforcing bool semantics before downstream use; this enforcement sits in `FunctionBuilder::emit_expr` at the load arm and emits `Fix:`-style diagnostics on unknown buffers.

---

## Summary Table

| # | Severity | File | Variant / Issue | Fix Priority |
|---|----------|------|-----------------|--------------|
| 01 | HIGH | `expr.rs:240` | `_ =>` catch-all on `Expr` | Compile-time exhaustiveness |
| 02 | CRITICAL | `expr.rs:206` | `SubgroupBallot` rejected | Bump naga to 25+ |
| 03 | CRITICAL | `expr.rs:218` | `SubgroupShuffle` rejected | Bump naga to 25+ |
| 04 | CRITICAL | `expr.rs:225` | `SubgroupAdd` rejected | Bump naga to 25+ |
| 05 | HIGH | `expr.rs:95` | `Cast` missing Bool/vector arms | Add explicit cast arms |
| 06 | CRITICAL | `expr.rs:178` | Atomic result type hardcoded `u32` | Derive from buffer element |
| 07 | HIGH | `expr.rs:169` | `CompareExchangeWeak` rejected | Lower as strong CX |
| 08 | HIGH | `expr.rs:169` | `FetchNand` rejected | Emulate or precise error |
| 09 | HIGH | `expr.rs:169` | `Opaque` atomic rejected | Add `WgpuEmitAtomic` trait |
| 10 | HIGH | `expr.rs:565` | `emit_unary` catch-all | Add unpack + opaque arms |
| 11 | HIGH | `utils.rs:88` | `binary_operator` catch-all | Route subgroup binops |
| 12 | CRITICAL | `utils.rs:106` | `Load` from bool buffer type mismatch | Insert `u32 != 0` cast |
| 13 | MEDIUM | `expr.rs:74` | `BufLen` on static array | Emit literal count |
| 14 | MEDIUM | `expr.rs:111` | `Fma` assumes F32 | Type-check operands |
| 15 | MEDIUM | `node.rs:99` | `Region` flattens | Wrap in `Block` |
| 16 | HIGH | `node.rs:111` | `Barrier` always combined | Add scope tag to `Node::Barrier` |
| 17 | CRITICAL | `node.rs:127` | Loop `to` evaluated twice | Hoist `to` to local |
| 18 | CRITICAL | `node.rs:248` | Loop error path loses body | RAII guard for body swap |
| 19 | MEDIUM | `node.rs:190` | Async nodes silently skipped | Reject with error |
| 20 | MEDIUM | `node.rs:201` | Trap/Resume silently skipped | Reject or emit control flag |
| 21 | HIGH | `node.rs:218` | `_ =>` catch-all on `Node` | Compile-time exhaustiveness |
| 22 | MEDIUM | `node.rs:81` | `If` condition not validated | Check bool or cast |
| 23 | CRITICAL | `mod.rs:204` | `scalar_type` rejects most types | Add per-type mapping |
| 24 | HIGH | `mod.rs:207` | `Bytes` mapped to `u32_ty` | Map to `array<u8>` or reject |
| 25 | HIGH | `mod.rs:207` | `Array` mapped to `u32_ty` | Use struct base or reject |
| 26 | MEDIUM | `mod.rs:154` | Missing `vec2/4_u32_ty` handles | Pre-register in `TypeHandles` |
| 27 | HIGH | `utils.rs:55` | `storage_access` catch-all | Explicit per-variant arms |
| 28 | MEDIUM | `buffer_access.rs` | Missing `WriteOnly` variant | Add to spec + mapping |
| 29 | MEDIUM | `utils.rs:35` | `Persistent` not mapped | Explicit reject message |
| 30 | LOW | `utils.rs:42` | Push constant binding `None` | Assign binding or reject |
| 31 | CRITICAL | `subgroup_intrinsics.rs:110` | String WGSL emission helpers | Delete string helpers |
| 32 | HIGH | `subgroup_intrinsics.rs:86` | `SubgroupCaps` dead code | Wire into emitter |
| 33 | HIGH | `generated.rs` | Missing subgroup Expr variants | Extend IR enum |
| 34 | CRITICAL | `node.rs:127` | Loop bound double-eval (CF) | Hoist to local |
| 35 | HIGH | `node.rs:319` | `emit_child_block` error path | Restore on Err |
| 36 | HIGH | `mod.rs:365` | `scan_atomic_targets` skips opaque | Recurse via trait |
| 37 | MEDIUM | `mod.rs:415` | `scan_atomic_targets_expr` catch-all | Exhaustive scan |
| 38 | HIGH | `expr.rs:169` | `CompareExchangeWeak` (atomics) | Lower as strong |
| 39 | HIGH | `expr.rs:169` | `FetchNand` (atomics) | Emulate or precise error |
| 40 | HIGH | `expr.rs:169` | `Opaque` atomic (atomics) | Add dispatch trait |
| 41 | MEDIUM | `node.rs:99` | Region flattens (region chain) | Wrap in `Block` |
| 42 | HIGH | `utils.rs:88` | `binary_operator` catch-all | Explicit arms |
| 43 | HIGH | `utils.rs:61` | `storage_access` catch-all | Explicit arms |
| 44 | MEDIUM | `utils.rs:35` | `address_space` catch-all | Explicit arms |
| 45 | MEDIUM | `utils.rs:139` | `expr_type` catch-all | Explicit arms + fix atomic |
| 46 | MEDIUM | `mod.rs:270` | Zero-sized array clamped to 1 | Reject zero size |
| 47 | LOW | `mod.rs:96` | Validation uses `Capabilities::all()` | Use adapter caps |
| 48 | LOW | `expr.rs:14` | `fold_expr` misses `Fma` | Add fold arm |
| 49 | MEDIUM | `expr.rs:601` | Rotate mask hardcoded 31 | Derive from width |
| 50 | CRITICAL | `expr.rs:70` | Bool load missing cast | Insert `!= 0` |

---

*End of audit. Every finding above maps to a specific line, a specific IR variant, and a minimal repro. The recommended next step is to address the CRITICAL findings (subgroup rejection, atomic i32 type mismatch, bool load cast, loop double-eval, and error-path body loss) before expanding type coverage.*
