//! Substrate-neutral kernel descriptor.
//!
//! This is the type that lives BETWEEN the optimizer and the emitters.
//! Every emitter takes a `KernelDescriptor` and produces a backend
//! artifact.
//!
//! ## Design principles
//!
//! - **Faithful to vyre IR**: embeds the same `BinOp`, `UnOp`,
//!   `AtomicOp`, `MemoryOrdering`, and `DataType` enums as the IR. No
//!   re-enumeration that would force the lowering to map "vyre IR op X"
//!   to "descriptor op Y" with a translation table; the descriptor
//!   carries the same op identity.
//! - **SSA-shaped**: every value-producing op assigns a unique 32-bit
//!   `result` id. Operands reference earlier results by id. No named
//!   variables at this layer  -  the lowering pass converts vyre IR's
//!   named bindings (`Node::Let`, `Node::Assign`, `Expr::Var`) into
//!   id references.
//! - **Structured control flow only**: `StructuredIfThen`,
//!   `StructuredIfThenElse`, `StructuredForLoop` carry indices into
//!   `KernelBody::child_bodies`. There is no goto / arbitrary jump;
//!   that's an explicit constraint required by structured compute
//!   emitters and low-level instruction emitters alike.
//! - **Substrate-neutral**: nothing in this module names any specific
//!   backend. Substrate-specific assumptions live in emitter crates.
//! - **Round-trippable**: serde-derived for every value; emitters can
//!   cache descriptors on disk.
//!
//! Every public descriptor type is declared in this file: the public API
//! snapshot pins their path at `vyre_lower::*`. Each child
//! module owns the behavior for one concern.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{AtomicOp, BinOp, DataType, SubgroupReduceOp, UnOp};
use vyre_foundation::schedule::{
    AxisMapping, PipelineRoleGroup, ScheduleResourceBounds, SynchronizationScope,
};

mod async_transaction;
mod binding_layout;
mod intent;
mod kernel;
mod kernel_op;
mod physical_schedule;
mod storage_layout;
mod tensor_access;
// Inline: supplies descriptor fixtures to the crate-private `descriptor` tests that stay inline.
#[cfg(test)]
pub(crate) mod test_descriptors;

pub(crate) use async_transaction::stage_transactions;
pub use async_transaction::AsyncTransactionError;
pub use binding_layout::{
    descriptor_trap_tags, DescriptorTrapTag, TRAP_SIDECAR_NAME, TRAP_SIDECAR_WORDS,
};
pub use intent::{
    scan_construct_intent_mapping, DESCRIPTOR_INTENT_SCHEMA_VERSION, SCAN_CONSTRUCT_INTENT_MAPPINGS,
};
pub use physical_schedule::PHYSICAL_SCHEDULE_VERSION;
pub use storage_layout::{
    StorageLayout, StorageLayoutError, StorageLifetime, StorageRegion, STORAGE_LAYOUT_VERSION,
};

/// One synchronization boundary the selected schedule placed on a phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BarrierPhase {
    /// Zero-based position of this boundary among the phase's boundaries.
    pub index: u32,
    /// Scope the boundary synchronizes.
    pub scope: SynchronizationScope,
}

/// Every execution fact the selected schedule froze for one phase.
///
/// A target reads this instead of inferring geometry, role assignment or
/// resource ceilings from the op stream. Nothing here is a suggestion: a
/// backend that cannot honor a field rejects the kernel rather than
/// substituting its own value.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PhysicalSchedule {
    /// Projection schema version.
    pub version: u16,
    /// Schedule schema version this was projected from.
    pub schedule_version: u16,
    /// Identity of the logical algorithm the schedule transforms.
    pub logical_identity: [u8; 32],
    /// Projected phase.
    pub phase: u32,
    /// Exact logical coverage selected for the phase.
    pub logical_coverage: [u64; 3],
    /// Exact workgroup shape selected for the phase.
    pub workgroup: [u32; 3],
    /// Selected vector width.
    pub vector_width: u32,
    /// Selected axis mappings, in schedule order.
    pub mappings: Vec<AxisMapping>,
    /// Role groups the phase participates in, empty when it is not pipelined.
    pub roles: Vec<PipelineRoleGroup>,
    /// Ring slots of the pipeline the phase participates in, zero when it is
    /// not pipelined.
    pub ring_slots: u32,
    /// Synchronization boundaries placed on the phase, in schedule order.
    pub barriers: Vec<BarrierPhase>,
    /// Persistent queue capacity selected for the phase, zero when it is not
    /// persistent.
    pub queue_capacity: u32,
    /// Checked resource ceiling of the phase.
    pub resources: ScheduleResourceBounds,
}

/// Where a binding's storage lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryClass {
    /// Global / device memory; visible to every workgroup.
    Global,
    /// Workgroup-shared memory.
    Shared,
    /// Read-only constant memory backed by a storage buffer
    /// (`BufferDecl::storage(.., ReadOnly, ..)`). Bind in group 0
    /// alongside `Global` writers.
    Constant,
    /// True uniform-buffer memory backed by `BufferDecl::uniform`.
    /// Maps to a uniform-buffer descriptor in the emitted dialect and binds
    /// in group 1 per `bind_group_for`. Distinct from
    /// `Constant` so the emitter can pick `AddressSpace::Uniform` and
    /// the layout builder can reserve the second bind group.
    Uniform,
    /// Backend-managed scratch storage.
    Scratch,
}

/// Read/write visibility for a binding slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BindingVisibility {
    /// Kernel may only read the binding.
    ReadOnly,
    /// Kernel may only write the binding.
    WriteOnly,
    /// Kernel may read and write the binding.
    ReadWrite,
}

/// One bound buffer at the kernel boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BindingSlot {
    /// Bind-group-slot index, stable across emitters.
    pub slot: u32,
    /// Element type. Carries the full vyre IR DataType so emitters can
    /// reproduce the exact type information (lane counts, sparse
    /// layouts, etc.).
    pub element_type: DataType,
    /// Element count. `None` means runtime-sized.
    pub element_count: Option<u32>,
    /// Storage location for the binding.
    pub memory_class: MemoryClass,
    /// Kernel access permissions for the binding.
    pub visibility: BindingVisibility,
    /// Caller-friendly identifier (for debug; does NOT participate in
    /// kernel hashing).
    pub name: String,
}

/// Full binding layout for a kernel.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BindingLayout {
    /// Bindings in stable slot order.
    pub slots: Vec<BindingSlot>,
}

/// A literal value that can sit in the literal pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LiteralValue {
    /// Unsigned 32-bit literal.
    U32(u32),
    /// Signed 32-bit literal.
    I32(i32),
    /// 32-bit floating-point literal.
    F32(#[serde(with = "literal_f32")] f32),
    /// Boolean literal.
    Bool(bool),
}

/// Serde representation of an f32 literal.
///
/// A descriptor travels to a materializer inside a target-module bundle, and
/// that bundle is JSON, which has no non-finite number. `serde_json` writes
/// `f32::NEG_INFINITY` as `null` and then refuses to read `null` back as an
/// f32, so every op whose literal pool holds an infinity produced a bundle no
/// backend could decode: `vyre-libs::nn::top_k` and `nn::softmax_top_k` seed a
/// running maximum with negative infinity and failed target-module decode with
/// `invalid type: null, expected f32`.
///
/// Only the values the plain encoding could not represent change shape. A
/// finite literal is still written as a number, byte for byte what the derived
/// impl wrote, so no other serde surface that carries a descriptor changes at
/// all. A non-finite literal is written as its IEEE-754 bit pattern in hex,
/// which is exact for every f32 including each NaN payload, and reads back
/// through `f32::from_bits`.
///
/// The escape is asked for only where the format is self-describing and the
/// plain encoding is lossy. A compact format cannot answer `deserialize_any`.
/// Descriptor dumps and descriptor hashes both use compact binary encoding,
/// which carries all 32 bits in a number and wants no escape at all. It reads
/// and writes exactly what the derived implementation did, byte for byte: a
/// descriptor hash keeps its value and a dumped descriptor stays readable
/// across this change.
mod literal_f32 {
    use std::fmt;

    use serde::de::{Unexpected, Visitor};
    use serde::{Deserializer, Serializer};

    /// Radix prefix for the non-finite escape. Present so a reader can tell an
    /// escaped bit pattern from a decimal literal someone wrote by hand.
    const BITS_PREFIX: &str = "0x";

    pub(super) fn serialize<S: Serializer>(value: &f32, serializer: S) -> Result<S::Ok, S::Error> {
        if value.is_finite() || !serializer.is_human_readable() {
            return serializer.serialize_f32(*value);
        }
        serializer.serialize_str(&format!("{BITS_PREFIX}{:08x}", value.to_bits()))
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
        if !deserializer.is_human_readable() {
            return deserializer.deserialize_f32(LiteralVisitor);
        }
        deserializer.deserialize_any(LiteralVisitor)
    }

    struct LiteralVisitor;

    impl Visitor<'_> for LiteralVisitor {
        type Value = f32;

        fn expecting(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                out,
                "a finite f32 number, or a non-finite one as `{BITS_PREFIX}` and eight hex digits"
            )
        }

        fn visit_f32<E: serde::de::Error>(self, value: f32) -> Result<f32, E> {
            Ok(value)
        }

        fn visit_f64<E: serde::de::Error>(self, value: f64) -> Result<f32, E> {
            Ok(value as f32)
        }

        fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<f32, E> {
            Ok(value as f32)
        }

        fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<f32, E> {
            Ok(value as f32)
        }

        fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<f32, E> {
            let digits = value
                .strip_prefix(BITS_PREFIX)
                .ok_or_else(|| E::invalid_value(Unexpected::Str(value), &self))?;
            let bits = u32::from_str_radix(digits, 16)
                .map_err(|_| E::invalid_value(Unexpected::Str(value), &self))?;
            let value = f32::from_bits(bits);
            if value.is_finite() {
                // A finite value has a number encoding, so accepting it here too
                // would give one literal two spellings and break the bundle's
                // canonical-bytes check on re-encode.
                return Err(E::invalid_value(
                    Unexpected::Float(f64::from(value)),
                    &"a non-finite f32 bit pattern; write a finite literal as a number",
                ));
            }
            Ok(value)
        }
    }
}

/// Stable identifier for a named entity (variable, region label, async
/// tag, trap tag). Mirrors vyre-foundation's `Ident` shape so the
/// lowering can preserve names for diagnostics.
pub type Name = Arc<str>;

/// Matrix multiply-accumulate tile extents, in elements.
///
/// These are mathematical extents, not backend instruction names. A target
/// lowers the extents it has a native form for and rejects the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatrixTileShape {
    /// Rows of the accumulator tile.
    pub m: u16,
    /// Columns of the accumulator tile.
    pub n: u16,
    /// Reduction extent shared by both input tiles.
    pub k: u16,
}

/// Element type used by a matrix MMA fragment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatrixMmaElement {
    /// 16-bit floating-point elements.
    F16,
    /// Brain floating-point 16-bit elements.
    BF16,
    /// Tensor floating-point 32-bit elements.
    TF32,
    /// 32-bit floating-point elements.
    F32,
}

/// Matrix fragment layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatrixMmaLayout {
    /// Row-major fragment layout.
    RowMajor,
    /// Column-major fragment layout.
    ColMajor,
}

/// Which operand of the multiply-accumulate a fragment carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FragmentOperand {
    /// Left input tile, `m` by `k`.
    Left,
    /// Right input tile, `k` by `n`.
    Right,
    /// Accumulator tile, `m` by `n`.
    Accumulator,
}

/// Typed access map for the storage a fragment tile is staged through.
///
/// The map states the addressing the tile requires. It never states how a
/// target satisfies it: bank permutation, vector width and transfer mechanism
/// are backend choices made under this map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorAccessMap {
    /// Storage the tile is addressed in.
    pub storage: MemoryClass,
    /// Element stride between consecutive rows of the tile, in elements.
    ///
    /// Zero states the rows are packed at the tile's own column extent.
    pub row_stride: u32,
    /// Guaranteed alignment of the tile base, in elements.
    pub alignment: u16,
}

/// One matrix operand of a multiply-accumulate, distributed across a
/// subgroup.
///
/// The fragment declares the facts a target needs to place it: element type,
/// tile orientation, how many invocations hold it, and the access map of the
/// storage it is staged through when it is not already register-resident.
/// Operand arity and result span are derived from these facts rather than
/// pinned to one native form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FragmentValue {
    /// Element type of the fragment's elements.
    pub element: MatrixMmaElement,
    /// Orientation of the tile the fragment holds.
    pub layout: MatrixMmaLayout,
    /// Invocations the fragment is distributed across.
    pub lanes: u16,
    /// Access map of the staging storage, or `None` when the fragment is
    /// register-resident.
    pub access: Option<TensorAccessMap>,
}

/// Complete typed specification of one matrix multiply-accumulate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatrixMmaSpec {
    /// Accumulator and reduction extents.
    pub tile: MatrixTileShape,
    /// Left input fragment.
    pub left: FragmentValue,
    /// Right input fragment.
    pub right: FragmentValue,
    /// Accumulator fragment, which is also the result fragment.
    pub accumulator: FragmentValue,
}

/// Why a matrix specification cannot be carried to a target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatrixSpecError {
    /// A tile extent is zero.
    ZeroExtent,
    /// A fragment is distributed across zero invocations.
    ZeroLanes,
    /// The fragment's element count is not divisible by its lane count.
    UnevenDistribution {
        /// Which operand the fragment carries.
        operand: FragmentOperand,
        /// Elements the tile holds.
        elements: u32,
        /// Invocations the fragment is distributed across.
        lanes: u16,
    },
    /// The per-lane element count does not fill whole 32-bit operand words.
    PartialWord {
        /// Which operand the fragment carries.
        operand: FragmentOperand,
        /// Bits each invocation holds for this fragment.
        bits_per_lane: u32,
    },
    /// A staged tile declares a row stride narrower than the tile itself.
    ShortRowStride {
        /// Which operand the fragment carries.
        operand: FragmentOperand,
        /// Declared stride in elements.
        stride: u32,
        /// Columns the tile occupies.
        columns: u16,
    },
    /// A staged tile declares zero base alignment.
    ZeroAlignment {
        /// Which operand the fragment carries.
        operand: FragmentOperand,
    },
}

/// Widest set of invocations that observes a completed transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransactionScope {
    /// Only the issuing invocation reads the destination.
    Invocation,
    /// The issuing subgroup reads the destination.
    Subgroup,
    /// Every invocation of the workgroup reads the destination.
    Workgroup,
    /// Invocations outside the workgroup read the destination.
    Device,
}

/// Fence a generic-proxy read needs after an asynchronous transfer lands.
///
/// An asynchronous transfer reaches memory through a proxy of its own, so the
/// write is not ordered against an ordinary read by completion alone. The
/// descriptor states which fence restores that order; the instruction that
/// carries it is a backend choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryProxyFence {
    /// The issuing invocation is the only reader, so completion orders it.
    None,
    /// Order the transfer against workgroup-scoped reads.
    Workgroup,
    /// Order the transfer against reads from outside the workgroup.
    Device,
}

/// One slot of a bounded stage ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StageSlot {
    /// Zero-based slot this transfer occupies.
    pub slot: u16,
    /// Depth of the ring the slot belongs to.
    pub ring_slots: u16,
}

/// One asynchronous data-movement transaction.
///
/// A backend chooses the transfer mechanism, bulk or scalar, and the wait form
/// under these facts. Nothing here names an instruction or a device.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AsyncTransaction {
    /// Identity pairing the transfer with its wait.
    pub tag: Name,
    /// Scope the completed transfer becomes readable at.
    pub visibility: TransactionScope,
    /// Ring slot the transfer occupies when a schedule staged it.
    pub stage: Option<StageSlot>,
}

/// A wait for one asynchronous transaction and the fence that follows it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AsyncWaitSpec {
    /// Transaction this wait completes.
    pub transaction: AsyncTransaction,
    /// Fence a subsequent generic-proxy read needs.
    pub fence: MemoryProxyFence,
}

/// One lowered op in the kernel body. Operands are referenced by
/// 32-bit id; the id space is per-`KernelBody`. SoA-friendly: an
/// emitter walks `body.ops` linearly and looks up operand ops by id
/// when needed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelOp {
    /// Operation semantics.
    pub kind: KernelOpKind,
    /// Operand ids into the same `KernelBody.ops` (or the literal pool
    /// for `Literal*` kinds  -  see the per-kind documentation).
    pub operands: Vec<u32>,
    /// Result id this op assigns. `None` for ops with no value
    /// (stores, barriers, returns, structured-control-flow markers).
    pub result: Option<u32>,
}

/// Lowered op kinds. Closed enum but covers the entire vyre IR
/// surface. Adding a new vyre IR variant requires a matching variant
/// here AND emit rules in every `vyre-emit-*` crate  -  that's the cost
/// of substrate parity.
///
/// Operand semantics are documented per variant. Reading a kind without
/// reading its operand contract gives wrong code.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KernelOpKind {
    // ---------- Literals ----------
    /// Operand 0 = index into `KernelBody.literals`. Result is the
    /// literal value typed per the literal pool entry.
    Literal,
    /// Snapshot a result value. Operand 0 = source result id. Result is
    /// a fresh SSA value with the source value at this program point.
    /// This is required when a source-level `Let` captures a mutable
    /// loop carrier: aliasing the carrier result id would read the
    /// later carrier value after a subsequent `Assign`.
    Copy,
    // ---------- Variable binding (lowered from Node::Let/Assign and Expr::Var) ----------
    //
    // The lowering pass converts vyre IR's named variables into SSA
    // form. `Node::Let` becomes "the result id of the bound expression
    // is now what `Var(name)` refers to in subsequent ops". `Node::Assign`
    // becomes a fresh result id that supersedes the earlier one. Names
    // are erased at this layer; the emitter never sees them.

    // ---------- Builtins ----------
    /// `LocalInvocationId.x/y/z`. Operand 0 = axis (0/1/2) as a small
    /// inline literal (NOT a literal-pool reference  -  emit picks the
    /// builtin directly).
    LocalInvocationId,
    /// `GlobalInvocationId.x/y/z`.
    GlobalInvocationId,
    /// `WorkgroupId.x/y/z`.
    WorkgroupId,
    /// Subgroup local invocation id (a.k.a. lane id).
    SubgroupLocalId,
    /// Subgroup size.
    SubgroupSize,
    /// Current induction value for the nearest structured loop that
    /// declared this variable. Produced as the first op in that loop's
    /// child body so uses of `Expr::Var(loop_var)` remain SSA-shaped
    /// instead of resolving to the loop's lower bound.
    LoopIndex {
        /// Source-level induction-variable name.
        loop_var: Name,
    },

    /// Initialize the loop-carrier slot for `name` from the pre-loop
    /// SSA value. Emitted ONCE in the PARENT body before the
    /// `StructuredForLoop` op. Operands: `[seed_value_id]`. No result.
    /// Emitters allocate a function-scope `LocalVariable` keyed by
    /// `name` (if not already allocated) and `Store(local, seed_value)`
    /// in the parent block.
    LoopCarrierInit {
        /// Source-level carrier name.
        name: Name,
    },

    /// Pure read of the carrier slot for `name`. Operands: `[]`.
    /// Result: the SSA id that in-loop reads of the source-level
    /// variable resolve to. Emit semantics: `Load` from the
    /// function-local allocated by the matching `LoopCarrierInit`.
    /// Used in three places per loop: (a) once at the top of each
    /// iteration so per-iteration reads resolve to the latest stored
    /// value; (b) in the parent body AFTER the loop so post-loop
    /// readers observe the loop's final value. Without this op,
    /// `Node::Assign` inside a loop body would have no observable
    /// effect on subsequent iterations  -  name resolution would always
    /// pick the pre-loop SSA, which is baked at lowering time.
    LoopCarrier {
        /// Source-level carrier name.
        name: Name,
    },

    /// Loop-carried-variable write at iteration end. Operands:
    /// `[final_value_id]`. No result. Pairs with `LoopCarrier { name }`
    /// to commit the iteration's final value of `name` back to the
    /// carrier local so the next iteration (or the post-loop reader)
    /// observes it.
    LoopCarrierEnd {
        /// Source-level carrier name.
        name: Name,
    },

    // ---------- Buffer access ----------
    /// `load(buf, index)`. Operands: [binding_slot, index_op_id].
    /// Result is the loaded value, dtype = binding's element type.
    LoadGlobal,
    /// `load(buf, index)` for a workgroup-shared binding.
    LoadShared,
    /// `load(buf, index)` for a constant/uniform binding.
    LoadConstant,
    /// Buffer length (number of elements). Operand 0 = binding_slot
    /// inline. Result is u32.
    BufferLength,
    /// `store(buf, index, value)`. Operands: [binding_slot, index_op_id, value_op_id].
    /// Result: None.
    StoreGlobal,
    /// `store(buf, index, value)` for a workgroup-shared binding.
    StoreShared,
    /// `load_vec<width>(buf, index)`. Operands: [binding_slot, index_op_id].
    /// Result is the loaded vector value of width 2 or 4.
    VectorLoadGlobal {
        /// Vector width (2 or 4).
        width: u8,
    },
    /// `store_vec<width>(buf, index, v0, v1, ...)`.
    /// Operands: [binding_slot, index_op_id, val0_id, val1_id, ...] (width values).
    /// Result: None.
    VectorStoreGlobal {
        /// Vector width (2 or 4).
        width: u8,
    },
    /// Extract a scalar lane from a vector value. Operands: `[vector_op_id]`.
    /// Result is the scalar value at `lane` (0..width).
    ExtractLane {
        /// Lane index (0..width).
        lane: u8,
    },

    // ---------- Arithmetic / logic ----------
    /// Binary op. Operands: [left_op_id, right_op_id]. Result has the
    /// dtype dictated by the operand dtypes (per vyre-spec rules).
    BinOpKind(BinOp),
    /// Unary op. Operands: `operand_op_id`. Result dtype per spec.
    UnOpKind(UnOp),

    // ---------- Composite ops ----------
    /// Fused multiply-add: `a * b + c`. Operands: [a_id, b_id, c_id].
    Fma,
    /// Matrix multiply-accumulate fragment op.
    ///
    /// The specification states the tile extents and one typed fragment per
    /// operand. Operands are the fragment's 32-bit words in operand order
    /// (left, then right, then accumulator), and `result` is the first of the
    /// accumulator fragment's words. Both counts are derived from the
    /// specification by `MatrixMmaSpec::operand_words`, so a tile the
    /// descriptor can state is a tile the verifier can check.
    MatrixMma(Box<MatrixMmaSpec>),
    /// Conditional select: `if cond { true_val } else { false_val }`.
    /// Operands: [cond_id, true_val_id, false_val_id].
    Select,
    /// Type cast. Operands: `value_id`. The target dtype is on the op.
    Cast {
        /// Target value type.
        target: DataType,
    },
    /// Atomic op. Operands: [binding_slot, index_op_id, value_op_id]
    /// for most ops. CompareExchange variants prepend `expected_op_id`:
    /// [binding_slot, index_op_id, expected_op_id, value_op_id].
    Atomic {
        /// Atomic operation.
        op: AtomicOp,
        /// Memory ordering.
        ordering: MemoryOrdering,
    },

    // ---------- Subgroup ops ----------
    /// Operand 0 = bool-typed cond_op_id. Result is u32 ballot mask.
    SubgroupBallot,
    /// Operands: [value_op_id, lane_op_id]. Result has the value's dtype.
    SubgroupShuffle,
    /// Operands: [value_op_id, lane_op_id]. Broadcasts `value` from the lane
    /// named by `lane` (uniform) to every lane; result has the value's dtype.
    /// Distinct from `SubgroupShuffle` (per-lane source), broadcast requires a
    /// uniform source lane and emits `subgroupBroadcast`.
    SubgroupBroadcast,
    /// Operand 0 = value_op_id. Reduces across the subgroup with `op`;
    /// result has the value's dtype.
    SubgroupReduce {
        /// Subgroup reduction operation.
        op: SubgroupReduceOp,
    },

    // ---------- Structured control flow ----------
    /// `if (cond) { body }`. Operands: [cond_op_id, child_body_index].
    /// `child_body_index` references `KernelBody.child_bodies`.
    /// Result: None.
    StructuredIfThen,
    /// `if (cond) { then } else { otherwise }`. Operands:
    /// [cond_op_id, then_body_index, otherwise_body_index].
    StructuredIfThenElse,
    /// `for (var = lo; var < hi; ++var) { body }`. Operands:
    /// [lo_op_id, hi_op_id, body_index]. The loop variable name is
    /// embedded on the op (preserved for debug, not for codegen).
    StructuredForLoop {
        /// Source-level induction-variable name.
        loop_var: Name,
    },
    /// Inline statement block  -  explicit grouping; semantically a
    /// no-op (body is flattened during emit). Operand 0 = body_index.
    StructuredBlock,
    /// Function/kernel return. Operands: empty. Result: None.
    Return,
    /// Memory barrier with explicit ordering.
    Barrier {
        /// Barrier memory ordering.
        ordering: MemoryOrdering,
    },
    /// Tracing/grouping marker (vyre IR `Node::Region`). Operand 0 =
    /// body_index. Carries no execution semantics; emitters MAY pass
    /// through as a comment or annotation. SEPARATION_AUDIT S5 plans
    /// to move this to a sidecar; until then it's an op so the
    /// descriptor preserves it round-trip.
    Region {
        /// Stable region generator identifier.
        generator: Name,
    },

    // ---------- Async ----------
    /// Asynchronous global-to-shared transfer. Operands:
    /// [src_binding, dst_binding, offset_op_id, size_op_id].
    /// The transaction pairs the transfer with its wait and states the ring
    /// slot and visibility a target must honor. Boxed to keep the common-case
    /// op small.
    AsyncLoad(Box<AsyncTransaction>),
    /// Mirror of `AsyncLoad` for shared-to-global. Operands:
    /// [src_binding, dst_binding, offset_op_id, size_op_id].
    AsyncStore(Box<AsyncTransaction>),
    /// Wait for one previously-issued transfer. Operands: empty.
    AsyncWait(Box<AsyncWaitSpec>),

    // ---------- Effect handlers ----------
    /// Trap into a host-side effect handler. Operands: `address_op_id`.
    Trap {
        /// Effect-handler identifier.
        tag: Name,
    },
    /// Resume from a previously-trapped effect.
    Resume {
        /// Effect-handler identifier.
        tag: Name,
    },

    // ---------- Indirect dispatch ----------
    /// Indirect-dispatch hint. The dispatch shape comes from
    /// `count_buffer[count_offset]`. Operand 0 = count_buffer
    /// binding_slot. Result: None.
    IndirectDispatch {
        /// Byte offset of the dispatch count.
        count_offset: u64,
    },

    // ---------- Calls ----------
    /// Call into a known op-id (e.g., a vyre-primitives builder
    /// surface). Operand list is the call's args. The op_id picks the
    /// callee at emit time.
    Call {
        /// Stable callee operation identifier.
        op_id: Name,
    },

    // ---------- Extension escape hatches ----------
    /// Opaque expression extension. The extension id resolves through
    /// the foundation extension registry. Emitters that do not recognize the
    /// extension MUST surface an error rather than silently emit nothing.
    ///
    /// Boxed to keep the common-case `KernelOpKind` small: most ops
    /// are Literal/BinOp/Load/Store at ≤16 bytes; without boxing,
    /// every op in the `ops` Vec pays the 52-byte OpaqueExpr tax.
    OpaqueExpr(Box<OpaqueExprData>),
    /// Opaque statement-node extension.
    OpaqueNode(Box<OpaqueNodeData>),
}

/// Heap-allocated payload for [`KernelOpKind::OpaqueExpr`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpaqueExprData {
    /// Numeric extension identifier.
    pub extension_id: u32,
    /// Stable extension-kind name.
    pub extension_kind: String,
    /// Opaque extension payload.
    pub payload: Vec<u8>,
}

/// Heap-allocated payload for [`KernelOpKind::OpaqueNode`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OpaqueNodeData {
    /// Stable extension-kind name.
    pub extension_kind: String,
    /// Opaque extension payload.
    pub payload: Vec<u8>,
}

/// Workgroup dispatch shape. `[x, y, z]` matches every modern
/// compute backend. `(1, 1, 1)` is a single invocation per workgroup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Dispatch {
    /// Local invocation dimensions along the x, y, and z axes.
    pub workgroup_size: [u32; 3],
}

/// One kernel body. Flat op stream + child bodies for nested
/// structured control flow. The entry point is `KernelDescriptor.body`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelBody {
    /// Operations in execution order.
    pub ops: Vec<KernelOp>,
    /// Child bodies referenced by `StructuredIfThen` etc. operand
    /// indices. Indexed from 0 within this body's child_bodies vec.
    pub child_bodies: Vec<KernelBody>,
    /// Literal pool referenced by `KernelOpKind::Literal` ops.
    pub literals: Vec<LiteralValue>,
}

/// The full kernel descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelDescriptor {
    /// Stable kernel identifier (for caching). Computed from the
    /// content hash by `lower::lower`. Empty string until lowering
    /// assigns it.
    pub id: String,
    /// Host and workgroup binding layout.
    pub bindings: BindingLayout,
    /// Workgroup dispatch geometry.
    pub dispatch: Dispatch,
    /// Root structured kernel body.
    pub body: KernelBody,
}

/// Iterator returned by [`KernelDescriptor::ops_iter`].
pub struct KernelOpsIter<'a> {
    /// Stack of (body, next_op_index) frames. Pushed as we descend
    /// into child bodies; popped when a body is exhausted.
    stack: Vec<(&'a KernelBody, usize)>,
}

/// Backend-neutral scan intent attached beside a [`KernelDescriptor`].
///
/// These intents describe why descriptor regions exist, not how a backend must
/// implement them. Every emitter, device or CPU, can route from these strategy
/// classes without owning scan compiler policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DescriptorIntentKind {
    /// Literal candidate prefiltering.
    LiteralPrefilter,
    /// Automata state transition.
    AutomataTransition,
    /// Candidate verification.
    Verifier,
    /// Output stream compaction.
    OutputCompaction,
    /// Relation seed construction.
    RelationSeed,
    /// Persistent streaming state.
    StreamingState,
}

/// Routing class derived from [`DescriptorIntentKind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DescriptorIntentStrategy {
    /// Literal-prefilter strategy.
    Prefilter,
    /// Automata-transition strategy.
    Automata,
    /// Verification strategy.
    Verifier,
    /// Output-compaction strategy.
    Compaction,
    /// Relation-seed strategy.
    RelationSeed,
    /// Streaming-state strategy.
    Streaming,
}

/// Semantic class of one scan-language construct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum ScanConstructIntentClass {
    /// Exact literal matching.
    Literal,
    /// Finite-state automata evaluation.
    Automata,
    /// Candidate verification.
    Verifier,
    /// Derivative-based matching.
    Derivative,
    /// External accelerator integration.
    ExternalAccelerator,
}

/// Required descriptor intents for one scan construct.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct ScanConstructIntentMapping {
    /// Stable construct identifier.
    pub construct_id: &'static str,
    /// Support tier exposed to operators.
    pub tier: &'static str,
    /// Semantic classes covered by the construct.
    pub classes: &'static [ScanConstructIntentClass],
    /// Descriptor intents required to execute the construct.
    pub required_intents: &'static [DescriptorIntentKind],
    /// Optional verifier fragment identifier.
    pub verifier_fragment_id: Option<&'static str>,
}

/// One intent annotation for a descriptor region, binding, or result id.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DescriptorIntent {
    /// Intent behavior.
    pub kind: DescriptorIntentKind,
    /// Optional referenced binding slot.
    pub binding_slot: Option<u32>,
    /// Optional referenced operation result.
    pub op_result: Option<u32>,
    /// Required persistent stream-state bytes.
    pub stream_state_bytes: u32,
    /// Arity of the seeded relation.
    pub relation_arity: u16,
    /// Stable digest of the source section.
    pub section_digest: u64,
}

/// Intent sidecar for a descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DescriptorIntentSet {
    /// Sidecar schema version.
    pub schema_version: u32,
    /// Descriptor intents in stable order.
    pub intents: Vec<DescriptorIntent>,
}

/// Descriptor plus validated intent sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IntentAnnotatedDescriptor {
    /// Lowered kernel descriptor.
    pub descriptor: KernelDescriptor,
    /// Validated intent sidecar.
    pub intents: DescriptorIntentSet,
}

/// Validated routing evidence derived from a descriptor intent sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DescriptorIntentEvidence {
    /// Validated sidecar schema version.
    pub schema_version: u32,
    /// Stable descriptor identifier.
    pub descriptor_id: String,
    /// Number of validated intents.
    pub intent_count: usize,
    /// Whether literal-prefilter intent is present.
    pub has_literal_prefilter: bool,
    /// Whether automata-transition intent is present.
    pub has_automata_transition: bool,
    /// Whether verifier intent is present.
    pub has_verifier: bool,
    /// Whether output-compaction intent is present.
    pub has_output_compaction: bool,
    /// Whether relation-seed intent is present.
    pub has_relation_seed: bool,
    /// Whether streaming-state intent is present.
    pub has_streaming_state: bool,
    /// Stable digest of routing-relevant intent data.
    pub strategy_digest: u64,
}

/// Invalid descriptor intent sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DescriptorIntentError {
    /// Sidecar schema version does not match this library.
    UnsupportedSchemaVersion {
        /// Supported schema version.
        expected: u32,
        /// Received schema version.
        actual: u32,
    },
    /// Sidecar contains no intents.
    EmptyIntentSet,
    /// An intent omits its source-section digest.
    MissingSectionDigest {
        /// Intent missing the digest.
        kind: DescriptorIntentKind,
    },
    /// An intent references an undeclared binding slot.
    UnknownBindingSlot {
        /// Referencing intent.
        kind: DescriptorIntentKind,
        /// Missing binding slot.
        slot: u32,
    },
    /// An intent references an unknown operation result.
    UnknownOpResult {
        /// Referencing intent.
        kind: DescriptorIntentKind,
        /// Missing result identifier.
        result: u32,
    },
    /// Relation-seed intent declares zero arity.
    MissingRelationArity,
    /// Streaming-state intent declares zero bytes.
    MissingStreamingStateBytes,
}
