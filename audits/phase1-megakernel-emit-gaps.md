# Phase 1  -  Megakernel emit gaps

This report enumerates the IR node variants the runtime megakernel builds, what the wgpu Naga emitter already covers, missing arms/shape-cases, buffer binding expectations, required WGSL features, and validator / wgpu risks. Citations use file:line.

---

## 1) IR Node constructors used by vyre-runtime/megakernel/builder.rs
(only constructors as written in source; line refs in builder.rs)

- Node::forever(...)  -  builder: vyre-runtime/src/megakernel/builder.rs:29,43 (creates the persistent body via Node::forever(...)).
- Node::let_bind(...)  -  many sites; examples: vyre-runtime/src/megakernel/builder.rs:56,65,72,77,112 (shutdown_flag, lane_id, slot_base, tenant_id, prev_status CAS).
- Node::if_then(...)  -  vyre-runtime/src/megakernel/builder.rs:60,121,195 (shutdown check, CAS success branch, JIT branch).
- Node::Return  -  vyre-runtime/src/megakernel/builder.rs:62,140 (early exit on shutdown).
- Node::store(...)  -  JIT tail writes and status store: vyre-runtime/src/megakernel/builder.rs:212-216,216-217 (store DONE/status words and stores used in handlers via claimed bodies).
- Node::loop_for / Node::Loop (memcpy uses loop_for)  -  handlers: vyre-runtime/src/megakernel/handlers.rs:110-123 (memcpy loop_for body). Note: Node::forever is sugar for Loop { from=0, to=u32::MAX }  -  vyre-foundation impl: vyre-foundation/src/ir_inner/model/node/impl_node.rs:152-154.
- Node::let_bind wrapping atomics (atomic_compare_exchange, atomic_add)  -  builder: vyre-runtime/src/megakernel/builder.rs:112-120 (atomic_compare_exchange), vyre-runtime/src/megakernel/builder.rs:210-211 (atomic_add done counter).

(These are all Node constructors that compose the interpreted and JIT megakernel flows.)

---

## 2) Which of those variants are emitted by vyre-driver-wgpu (naga_emit)
(List each covered variant with the emitter file:line of the match arm)

From vyre-driver-wgpu/src/lowering/naga_emit/node.rs:

- Node::Let   -  node.rs:15-37 (emit_node match arm for `Node::Let`).
- Node::Assign  -  node.rs:38-49 (present; assignment lowering).  
- Node::Store   -  node.rs:50-80 (handles buffer pointer/Store; includes bool→u32 cast)  -  node.rs:55-79.
- Node::If     -  node.rs:81-97 (If → Statement::If emission).
- Node::Block  -  node.rs:98 (delegates to emit_nodes).
- Node::Region  -  node.rs:99-110 (treated as wrapper → lowers body).
- Node::Barrier  -  node.rs:111-115 (emits Statement::Barrier).
- Node::Return   -  node.rs:115-118 (emits Return statement).
- Node::Loop (bounded loops)  -  node.rs:119-176 (full bounded-loop lowering: create local, emit guard, continuing, body).
- Node::Opaque  -  node.rs:182-190 (extension hook via WgpuEmitNode).
- Unknown / default rejection  -  node.rs:191-195 returns an explicit LoweringError for unknown Node variants.

From vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:

- Literal / Var / Builtins  -  Expr::LitU32/I32/F32/Bool, Expr::Var, InvocationId/WorkgroupId/LocalId  -  expr.rs:12-28.
- Expr::Load / BufLen  -  expr.rs:29-37 (load pointers, ArrayLength).
- Expr::BinOp / UnOp / Select / Cast / Fma  -  expr.rs:38-81 and 193+ (many arithmetic, relational operators mapped to Naga ops).
- Expr::Atomic  -  expr.rs:87-150 (handles AtomicOp variants: Add, And, Or, Xor, Min, Max, Exchange, CompareExchange). CompareExchange builds an `Exchange { compare: Some(cmp) }` and emits an Atomic Statement plus returns AccessIndex for old value; rust-side result type uses module.types.atomic_compare_exchange_u32_ty: expr.rs:123-131,134-149.
- Subgroup ops / some intrinsics rejected or gated  -  expr.rs:151-175 (SubgroupBallot/Shuffle/Add are rejected/gated on Naga 25+).

Summary: every Node constructor used by the megakernel body (Let, If, Return, Store, Loop) has a direct lowering arm in node.rs (see arms above). Core atomic expressions used by the kernel (atomic_add, atomic_compare_exchange, atomic_exchange) are handled in expr.rs:87-150.

---

## 3) Missing variants / missing arm combinations (concrete gaps)
(arm + file:line citations and explanation)

1. Atomic storage element type emission: ModuleBuilder.add_buffer creates an array type whose element is `storage_scalar_type` (u32 for DataType::U32) and attaches it as the GlobalVariable type  -  vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:216-256 (array_ty creation) and 249-256 (global variable append). BUT expr.rs emits `Statement::Atomic` operations that in WGSL/Naga are expected to target an `atomic<...>` element type. There is NO creation of an `atomic<u32>` element type in add_buffer (mod.rs:216-256). This mismatch is likely to trigger Naga validator errors or WGSL semantics issues. (See expr.rs Atomic handling: expr.rs:123-141.)

2. CompareExchangeWeak / FetchNand / some atomic flavors rejected: expr.rs explicitly rejects AtomicOp::CompareExchangeWeak and AtomicOp::FetchNand (expr.rs:118-121). If any future opcode/handler uses these flavors, Naga emitter will error. The megakernel currently uses CompareExchange (supported)  -  handlers.rs:96-106 (compare_swap_body).

3. Subgroup / collective intrinsics gating: emitter rejects SubgroupBallot/Shuffle/Add and returns an explanatory LoweringError pointing to Naga 25+ (expr.rs:151-161). If megakernel code ever uses subgroup intrinsics (not currently present), the emitter will fail unless Naga/toolchain upgraded.

4. Loop shape: emitter assumes bounded loops with matching `from` and `to` types and enforces index_ty ∈ {u32,i32} (node.rs:125-136 and emit_loop_bound_condition checks types: node.rs:125-136 and 278-290). Node::forever lowers to Loop { from=0, to=u32::MAX } (vyre-foundation impl: impl_node.rs:152-154)  -  that fits the bounded-loop lowering, but care: extremely large `to` is supported by the emitter because it generates a >= comparison (node.rs:285-289). No change needed, but note the emitter enforces the bound types and will reject non-u32 loop bounds.

5. Atomics on non-u32 element types: expr.rs maps all atomic ops to ScalarKind::Uint and returns DataType::U32 for Expr::Atomic (expr.rs:96-132 and expr_type: expr.rs:132). Validation rule in IR: atomic-buffer-element-must-be-u32 (docs and validation rules referenced in lowering/utils: lowering/naga_emit/utils.rs:92-98 and the docs grep results). If the kernel ever declared atomics targeting i32/f32/bool storage directly (instead of u32), emitter rejects or mis-emits them.

6. Missing write of atomic reservation vs store ordering for PRINTF: handlers.rs prints reserves with atomic_add on debug_log then stores four stores sequentially (handlers.rs:51-79). This pattern relies on atomic reservation + non-atomic stores to those reserved indices. The emitter supports the atomic reservation (expr.rs:55-61 atomic_add mapping) and Store (node.rs:50-80), but Naga/WGSL validation may require the Debug buffer element type to be suitable for atomic+non-atomic accesses (atomic<> vs scalar). The emitter must ensure the buffer's element type and memory layout permit `atomic_add` and then scalar stores to adjacent elements  -  this is an emitter-level correctness gap (mod.rs:216-256 + expr.rs atomic handling 87-150).

---

## 4) Buffer kinds the megakernel expects vs current Naga emission

From runtime builder: control, ring_buffer, debug_log are declared as read_write U32 buffers: vyre-runtime/src/megakernel/builder.rs:22-24 (BufferDecl::read_write("control", 0, DataType::U32), etc.).

- MemoryKind / access: these are storage (read-write) buffers. ModuleBuilder.add_buffer maps MemoryKind::Global/Readonly → AddressSpace::Storage with StorageAccess derived from buffer.access (naga_emit/utils.rs:23-35 and mod.rs:add_buffer:216-256). Binding assignment uses bind_group_for(...) → currently group 0 (lowering/mod.rs:171-206).

- Are Naga bindings correct? Partial: the emitter emits a Storage array with element = u32 (mod.rs:216-256) and sets ResourceBinding with group/binding (utils.rs:42-51). However: for atomic operations the WGSL spec expects atomic<T> element types (and Naga usually requires the pointed-to type for Atomic to be an atomic). There is no emission of an `atomic<u32>` element type in add_buffer (mod.rs:155-177 shows `atomic_compare_exchange_u32_ty` *result* struct but not atomic element types). Therefore:
  - Storage read/write buffers are emitted and bound (OK): mod.rs:249-256 + utils.binding:42-51.
  - Atomic operations target u32 scalar elements (expr.rs:96-132) but the module does not declare atomic<u32> element types  -  GAP: need to emit atomic element types for buffers used with atomics (mod.rs:216-256 + expr.rs:87-150). Without that change validation may fail.

---

## 5) WGSL / runtime features required for persistent-entry-point shader

Required language/runtime features (concrete):

- Storage buffer LOAD/STORE support (core)  -  checked via limiting `max_storage_buffer_binding_size` in adapter_caps_probe: vyre-driver-wgpu/src/runtime/adapter_caps_probe.rs:41-42 reports `max_storage_buffer_binding_size`.
- Atomics on storage buffers (atomic_add, atomic_exchange, atomic_compare_exchange)  -  emitter uses `Statement::Atomic` (expr.rs:135-141) and relies on storage element typing to be atomic. adapter_caps_probe DOES NOT explicitly probe for atomic-support feature  -  it currently only checks SUBGROUP and INDIRECT_FIRST_INSTANCE (adapter_caps_probe.rs:25-26). Add explicit probe or capability/limit check for atomic availability if backend requires a feature flag.
- Workgroup builtins: GlobalInvocationId / WorkGroupId / LocalInvocationId  -  emitter pushes builtins in mod.rs:272-289 (naga_emit/mod.rs:272-289) and the entry point uses them (mod.rs:299-306)  -  required by megakernel (builder uses Expr::workgroup_x/local_x). No extra adapter probe done; these are standard.
- Sufficient workgroup and storage limits: max_compute_workgroup_size and max_compute_workgroup_storage_size already probed in adapter_caps_probe.rs:35-41 (max_workgroup_size & max_shared_memory_bytes).
- (Optional) Subgroup ops: NOT required by current megakernel, but if used would require Feature::SUBGROUP which adapter_caps_probe already checks (adapter_caps_probe.rs:25).

Actionable gap: adapter_caps_probe currently probes SUBGROUP and INDIRECT_FIRST_INSTANCE (adapter_caps_probe.rs:25-26) and reports many limits (lines 35-43), but it does NOT explicitly gate/check atomic-on-storage support. Consider adding explicit capability probe or a `supports_storage_atomics` boolean derived from adapter/features or backend API.

---

## 6) Risks  -  Naga validator rules and wgpu ceiling blockers

1. Atomic element typing: Naga/WGSL requires atomic types for atomic operations. ModuleBuilder currently emits storage array elements as plain u32 (mod.rs:216-256) while Statement::Atomic is used (expr.rs:135-141). Validator may reject atomic statements whose pointer type is not an atomic element. Fix: when a buffer is used with atomics (control, debug_log, ring status), emit element type as a Naga atomic<u32> or otherwise ensure correct atomic pointer typing.

2. Naga version ceiling (24.x): subgroup intrinsics lowering is gated on Naga 25+ (expr.rs:151-161). If future megakernel features rely on subgroup ops, the toolchain must upgrade; otherwise emitter will reject.

3. Validation of combined atomic + scalar stores to same array (PRINTF reservation pattern): WGSL semantics and validator may require atomic elements to be used consistently; performing atomic_add to an atomic element that yields a reservation index and then performing scalar stores to surrounding array slots must map to a layout Naga/WGSL accepts. This is an emitter correctness surface (handlers.rs:51-79 + mod.rs:add_buffer 216-256).

4. Workgroup/shared memory limits: megakernel's expected worker_count/workgroup_size combined with per-lane local storage could exceed `max_compute_workgroup_storage_size`  -  adapter_caps_probe reports that limit (adapter_caps_probe.rs:41). Ensure code checks `MegakernelConfig.worker_count` vs backend caps before dispatch.

5. Storage binding size: ring buffer size may be very large; ensure `max_storage_buffer_binding_size` (adapter_caps_probe.rs:42) is sufficient and emit a compile-time check/fail with an actionable Fix if exceeded.

6. Validator strictness: `emit_module` uses Validator::new(ValidationFlags::all(), Capabilities::all()) (naga_emit/mod.rs:84-87)  -  this is permissive in capabilities but the writer/validator will still enforce WGSL typing rules (atomic types, array strides, push-constant rules). Errors surface as LoweringError::validation(e) with function/ep dumps (mod.rs:88-94)  -  the emitter must modify module construction (atomic element types, correct result structs) to pass.

---

## Implementation checklist to land engine/megakernel_emit.rs (actionable)

1. Emit atomic element types for buffers used with atomics (control, debug_log, ring status): create TypeInner::Atomic or appropriate Naga atomic wrapper and use it as the array base for those buffers (naga_emit/mod.rs:add_buffer:216-256; expr.rs:87-150).
2. Ensure PRINTF reservation pattern yields a valid memory layout: either use an atomic<u32> cursor and adjacent u32 scalar elements with correct stride, or use a single atomic struct layout the validator accepts (handlers.rs:51-79 + mod.rs:add_buffer:238-256).
3. Add adapter cap probe for `supports_storage_atomics` or equivalent, and assert `max_storage_buffer_binding_size` & `max_compute_workgroup_storage_size` early (runtime/adapter_caps_probe.rs:25-43).
4. Add tests that compile the emitted persistent shader through Naga validator and produce WGSL (naga_emit/mod.rs:84-97)  -  assert no validation errors and that the emitted WGSL uses atomic element types.
5. If subgroup intrinsics are ever required, plan a Naga/toolchain upgrade to 25+ (expr.rs:151-161).

---

### Key file citations (representative)
- runtime megakernel builder + handlers: vyre-runtime/src/megakernel/builder.rs:22-31,53-63,112-124,180-218
- protocol/opcodes: vyre-runtime/src/megakernel/protocol.rs:9-19,27-36,74-83,86-96,116-125
- handlers opcode bodies: vyre-runtime/src/megakernel/handlers.rs:28-36,52-61,85-106,110-123,128-140
- node lowering arms: vyre-driver-wgpu/src/lowering/naga_emit/node.rs:14-36,38-49,50-80,81-97,119-176,182-191
- atomic expr lowering: vyre-driver-wgpu/src/lowering/naga_emit/expr.rs:87-150 (atomics), 12-37 (loads/literals), 38-81 (binops)
- buffer addition + type creation: vyre-driver-wgpu/src/lowering/naga_emit/mod.rs:216-256,249-256 (global var append)
- binding/address space helpers: vyre-driver-wgpu/src/lowering/naga_emit/utils.rs:23-35 (address_space),42-51 (binding)
- adapter cap probe: vyre-driver-wgpu/src/runtime/adapter_caps_probe.rs:21-26,35-43
- forever → Loop lowering reference: vyre-foundation/src/ir_inner/model/node/impl_node.rs:152-154

---

If desired, next step is a focused patch that:
- changes ModuleBuilder::add_buffer to emit atomic element types for u32 buffers when they are used in atomic expressions (detect via Program analysis or conservative rule: all read_write u32 buffers used with Expr::Atomic → atomic<u32> elements), and
- adds adapter capability probe for storage-atomics where appropriate.

(End of Phase 1 gap report.)
