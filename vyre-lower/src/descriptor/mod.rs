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
//! snapshot pins their path at `vyre_lower::descriptor::*`. Each child
//! module owns the behavior for one concern.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use vyre_foundation::ir::{AtomicOp, BinOp, DataType, SubgroupReduceOp, UnOp};
use vyre_foundation::memory_model::MemoryOrdering;

mod binding_layout;
mod intent;
mod kernel;
mod kernel_op;
#[cfg(test)]
pub(crate) mod test_descriptors;

pub use binding_layout::{TRAP_SIDECAR_NAME, TRAP_SIDECAR_WORDS};
pub use intent::{
    scan_construct_intent_mapping, DESCRIPTOR_INTENT_SCHEMA_VERSION, SCAN_CONSTRUCT_INTENT_MAPPINGS,
};

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
    /// Maps to WGSL `var<uniform>` / Vulkan `uniform_buffer` descriptor
    /// and binds in group 1 per `bind_group_for`. Distinct from
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
    F32(f32),
    /// Boolean literal.
    Bool(bool),
}

/// Stable identifier for a named entity (variable, region label, async
/// tag, trap tag). Mirrors vyre-foundation's `Ident` shape so the
/// lowering can preserve names for diagnostics.
pub type Name = Arc<str>;

/// Matrix multiply-accumulate tile shape for descriptor-level MMA ops.
///
/// These are mathematical fragment shapes, not backend instruction names.
/// Emitters map supported shapes to their native substrate and reject shapes
/// they cannot lower.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatrixMmaShape {
    /// 16 rows × 8 columns × 16 reduction lanes.
    M16N8K16,
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
    /// Operand contract for `M16N8K16/F16/F16/F32`:
    /// `[a0,a1,a2,a3, b0,b1, c0,c1,c2,c3]`, where `a*` and `b*` are
    /// packed 16-bit fragment words and `c*` are f32 accumulators. `result`
    /// is the first of four consecutive result ids (`result..result+4`).
    /// This keeps the descriptor SSA-shaped without adding backend-specific
    /// register-fragment objects to the neutral IR.
    MatrixMma {
        /// Matrix tile shape.
        shape: MatrixMmaShape,
        /// Left fragment layout.
        a_layout: MatrixMmaLayout,
        /// Right fragment layout.
        b_layout: MatrixMmaLayout,
        /// Left fragment element type.
        a_type: MatrixMmaElement,
        /// Right fragment element type.
        b_type: MatrixMmaElement,
        /// Accumulator element type.
        accum_type: MatrixMmaElement,
    },
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
    /// `cp.async`-style global-to-shared copy. Operands:
    /// [src_binding, dst_binding, offset_op_id, size_op_id].
    /// `tag` ties the load to a matching `AsyncWait`.
    AsyncLoad {
        /// Identifier shared with the matching wait.
        tag: Name,
    },
    /// Mirror of AsyncLoad for shared-to-global. Operands:
    /// [src_binding, dst_binding, offset_op_id, size_op_id].
    AsyncStore {
        /// Identifier shared with the matching wait.
        tag: Name,
    },
    /// Wait on a previously-issued AsyncLoad/Store. Operands: empty.
    AsyncWait {
        /// Identifier of the asynchronous operation to await.
        tag: Name,
    },

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
/// implement them. CUDA, WGPU, Metal, SPIR-V, and CPU emitters can route from
/// these strategy classes without owning scan compiler policy.
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
