# Subgroup reduction generalization: `SubgroupAdd` → `SubgroupReduce { op }`

## The gap (real-path, multi-axis)
`vyre-foundation/src/lower/subgroup_lowering.rs:179-188` explicitly REFUSES to lower
`workgroup_max_*` reductions to subgroup ops ("for simplicity ... we just keep the
original body — the task focuses on sum reductions"). So `workgroup_max_f32`
(a SHIPPED primitive, `vyre-primitives/src/reduce/workgroup_tree.rs:108`, dispatched
via `vyre-self-substrate/.../dispatch_workgroup_max_f32`) runs as the slow
shared-memory tree even on subgroup-capable hardware, while sum reductions get the
fast subgroup path.

Root cause: the IR can only express `Expr::SubgroupAdd { value }` (one op). The
driver taxonomy (`vyre-driver/src/subgroup.rs::SubgroupOp`) and wgpu capability
diagnostics (`vyre-driver-wgpu/tests/wgpu_subgroup_*`) already promise
`subgroup_min`/`subgroup_max`, but the IR/emit cannot produce them.

Axes: CAPABILITY (IR can't express max/min), PERF (Law 7 — max stays on slow tree),
COHERENCE (driver promises ops the IR lacks), WIRING (the lowering's max branch is a
dead `None`).

## Design
Mirror the existing `Atomic { op: AtomicOp }` shape exactly.

- NEW `vyre-spec::subgroup_reduce_op::SubgroupReduceOp` (closed `#[non_exhaustive]`
  enum, `builtin_wire_tag`/`from_wire_tag` like `CollectiveOp`). The complete
  hardware/naga set: `Add, Mul, Min, Max, And, Or, Xor`. Wire tags
  Add=0x01,Mul=0x02,Min=0x03,Max=0x04,And=0x05,Or=0x06,Xor=0x07.
- `Expr::SubgroupAdd { value }` → `Expr::SubgroupReduce { op: SubgroupReduceOp, value }`
  (generated.rs). Wire tag 17 keeps its slot; encoding becomes `[17][op_byte][value]`.
- `KernelOpKind::SubgroupAdd` → `KernelOpKind::SubgroupReduce { op: SubgroupReduceOp }`
  (vyre-lower/src/descriptor.rs).
- Constructors (vyre-foundation expr.rs): keep `subgroup_add(v)` = `SubgroupReduce{Add}`
  for back-compat (Law 3); add `subgroup_reduce(op,v)`, `subgroup_max`, `subgroup_min`,
  `subgroup_mul`, `subgroup_and`, `subgroup_or`, `subgroup_xor`.

## Per-backend op mapping (all 7 natively supported on the warp)
- **naga** (`emit_subgroup_add` → generalize): `naga::SubgroupOperation::{Add,Mul,Min,Max,And,Or,Xor}`.
- **spirv**: `OpGroupNonUniform{IAdd/FAdd, IMul/FMul, SMin/UMin/FMin, SMax/UMax/FMax, BitwiseAnd, BitwiseOr, BitwiseXor}` (signed/unsigned/float by operand dtype).
- **metal**: `simd_sum / simd_product / simd_min / simd_max / simd_and / simd_or / simd_xor`.
- **ptx/cuda**: integer ops via `redux.sync.{add,min,max,and,or,xor}.{u32,s32}` (sm_80+);
  float + mul via the shuffle-butterfly tree with the op swapped in the combine step.
  Any (op,dtype) a target cannot lower must FAIL CLOSED LOUDLY (EmitError) — never
  silent (Law 10).
- **reference oracle** (`eval_subgroup_add` → `eval_subgroup_reduce(op)`): fold the
  lane snapshots with the op; neutral = 0(Add)/1(Mul)/+inf(Min)/-inf(Max)/!0(And)/0(Or/Xor).

## Lowering (the payoff)
`try_lower_workgroup_reduction`: the `WORKGROUP_MAX_PREFIX` branch stops returning
`None` and emits `subgroup_reduce(Max, ...)`, with the two-level neutral = f32::-inf
(vs 0 for sum). Only enable for a backend once its emit supports Max (else loud reject,
no regression vs today's slow-but-correct tree).

## Test plan (assert real values, Law 6)
- lowering: `lowers_every_workgroup_max_to_subgroup_reduce_max` (body uses SubgroupReduce{Max}).
- reference: per-op fold over a known lane vector → exact expected scalar (e.g. max([3,1,4,1,5])=5).
- naga: each op emits the matching `SubgroupOperation`.
- wire: roundtrip every `SubgroupReduceOp` (encode→decode identity).
- typecheck: And/Or/Xor reject f32 operands; Add/Mul/Min/Max preserve value dtype.

## Status
Implementation COMPLETE; whole `cargo build --workspace` is GREEN (exit 0, ~30 crates).
Migration touched the whole workspace (every `Expr::SubgroupAdd {` and
`KernelOpKind::SubgroupAdd` site, compiler-guided). keyhog pins crates.io `vyre =0.6.3`
(no path override) so the dirty tree never blocked the concurrent keyhog agent.

Landed:
- vyre-spec: `SubgroupReduceOp` {Add,Mul,Min,Max,And,Or,Xor} + wire tags + `reduce_u32`
  / `f32_identity` / `combine_f32` (single source of truth) + 6 unit tests.
- IR: `Expr::SubgroupReduce { op, value }` (generated.rs), constructors `subgroup_reduce`
  + `subgroup_{add,mul,min,max,and,or,xor}`; wire encode/decode carries the op byte;
  structural hash (meta.rs) folds the op tag (so subgroup_max ≠ subgroup_add for CSE).
- vyre-lower: `KernelOpKind::SubgroupReduce { op }`; both lowering sites thread the op;
  re-exports `SubgroupReduceOp`.
- Backends: naga maps op→`naga::SubgroupOperation` (all 7); ptx uses `redux.sync.{op}`
  for integer add/min/max/and/or/xor and a shfl butterfly with the op's combine instr
  for f32 add/mul/min/max — integer-mul and f32-bitwise FAIL CLOSED LOUD (no silent
  wrong code). spirv = capability-only (arithmetic bit for all ops); metal has no
  subgroup path. Reference oracle folds via the spec helpers.
- Lowering payoff: `workgroup_max_*` now lowers to `subgroup_reduce(Max)` with the
  `-inf` two-level neutral (was a dead `None` keeping the slow shared tree).

Tests added (assert real values): spec reduce_u32 exact values + wrap + neutrals;
foundation lowering max (single + two-level -inf neutral); naga per-op→SubgroupOperation
mapping; reference oracle u32 max/xor, f32 -inf max, f32-bitwise fail-loud.

Open follow-ons (NOT regressions; the lowering only emits Add/Max today):
- PTX f32 shfl-down butterfly yields the full reduction only at lane 0 (pre-existing
  for Add; redux.sync integer path is all-lane-uniform). If a consumer needs the f32
  result broadcast to every lane, the butterfly needs an up-sweep or a final broadcast.
- integer subgroup Mul has no PTX redux; add a shfl butterfly if a generator needs it.
