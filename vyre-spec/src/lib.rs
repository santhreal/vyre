//! vyre-spec is the machine-checkable frozen data contract for the vyre GPU
//! compute IR. Any backend may depend on vyre-spec alone to prove conformance
//! without depending on vyre itself.
//!
//! This crate is intentionally data-only. It has no dependency on downstream
//! crates; backend vendors can use these types as the stable contract
//! for conformance proofs. Example: a conformance runner can read an
//! [`OpSignature`] and verify the byte width expected by a backend primitive.

/// Adversarial input descriptors  -  hostile payloads every op must reject or handle.
/// Specification element.
mod adversarial_input;
/// Algebraic law primitives  -  associativity, identity, commutativity declarations.
/// Specification element.
mod algebraic_law;
/// Canonical catalog of every algebraic law tagged to operations.
/// Specification element.
mod all_algebraic_laws;
#[macro_use]
mod op_wire;
/// Versioned cross-engine analysis fact records.
pub mod analysis;
/// Atomic operation enum  -  the bounded set of read-modify-write primitives.
/// Specification element.
mod atomic_op;
/// Binary operator enum  -  all element-wise two-operand primitives.
/// Specification element.
mod bin_op;
/// Buffer access mode (ReadOnly / WriteOnly / ReadWrite) + enforcement helpers.
/// Specification element.
mod buffer_access;
/// Iterator returning op ids grouped by their `Category`.
/// Specification element.
mod by_category;
/// Reverse index from op id string to its canonical descriptor.
/// Specification element.
mod by_id;
/// Terminal ids for the LR(1) arithmetic expression grammar.
/// Specification element.
pub mod c11_expr_token;
/// C11 lexer token ids shared by the host table generator and the GPU parser.
/// Specification element.
pub mod c11_token;
/// Conformance invariant: the op catalog enumerates every known id.
/// Specification element.
mod catalog_is_complete;
mod catalog_slices;
/// Category enum (A/B/C) + backend-availability predicates.
/// Specification element.
mod category;
/// Collective communication operators and communicator handles.
/// Specification element.
mod collective_op;
/// Calling conventions between CPU host and GPU kernels.
/// Specification element.
mod convention;
/// Primitive data-type enum (U32/F32/Bool/etc.) + size helpers.
/// Specification element.
mod data_type;
/// Invariants the engine itself must preserve (wire round-trip, CSE stability, …).
/// Specification element.
mod engine_invariant;
/// Frozen catalog of core `Expr` variant names used by coverage tests.
/// Specification element.
mod expr_variant;
/// Dialect extension descriptor  -  marks non-core ops carried by extensions.
/// Specification element.
pub mod extension;
/// Floating-point type subset (F16/F32/F64) with associated properties.
/// Specification element.
mod float_type;
/// Go lexer token ids shared by the GPU lexer program and its host matchers.
/// Specification element.
pub mod go_token;
/// Golden reference samples  -  tiny fixtures every backend must reproduce exactly.
/// Specification element.
mod golden_sample;
/// Table of hardware intrinsics exposed by `vyre-primitives::hardware`.
/// Specification element.
mod intrinsic_table;
/// Abstract invariant type + provenance tracking.
/// Specification element.
mod invariant;
/// Classification buckets grouping related invariants (numeric, memory, …).
/// Specification element.
mod invariant_category;
/// Catalog of invariants every registered op is checked against.
/// Specification element.
mod invariants;
/// Known-answer test vector type  -  deterministic input/output pairs.
/// Specification element.
mod kat_vector;
/// Canonical catalog of algebraic laws exposed via `law_catalog()`.
/// Specification element.
mod law_catalog;
/// Layer enum (IR / backend / runtime)  -  coarse module placement.
/// Specification element.
mod layer;
/// Metadata classification for `OpMetadata` entries.
/// Specification element.
mod metadata_category;
/// Monotonicity direction (increasing / decreasing / none) for op outputs.
/// Specification element.
mod monotonic_direction;
/// Versioned numeric semantics table and datatype conversion helpers.
/// Specification element.
mod numeric_semantics;
/// Operation contract: capability requirements, determinism, cost hints.
/// Specification element.
mod op_contract;
/// Op metadata struct  -  human-facing description and discoverability hooks.
/// Specification element.
mod op_metadata;
/// Op signature  -  stable type profile every backend lowers against.
/// Specification element.
mod op_signature;
/// Packed graph node kinds for language-agnostic analysis.
/// Specification element.
mod pg_node_kind;
/// Python lexer token ids shared by the GPU lexer program and its host matchers.
/// Specification element.
pub mod python_token;
/// Canonical semiring selector for dataflow and algebraic kernels.
mod semiring;
/// Soundness markers and precision contracts for cross-engine analysis data.
pub mod soundness;
/// Subgroup (warp) reduction operator enum  -  add/mul/min/max/and/or/xor.
/// Specification element.
mod subgroup_reduce_op;
/// Ternary operator enum  -  select, FMA, mask-merge.
/// Specification element.
mod ternary_op;
/// Structured test descriptor  -  op id, input sampler, expected shape.
/// Specification element.
mod test_descriptor;
#[cfg(test)]
#[path = "../tests/internal/mod.rs"]
mod tests;
/// Unary operator enum  -  single-operand element-wise primitives.
/// Specification element.
mod un_op;
/// Conformance verification driver  -  runs the law + invariant battery.
/// Specification element.
mod verification;

/// See [`adversarial_input::AdversarialInput`].
/// Specification element.
pub use adversarial_input::AdversarialInput;
/// See [`algebraic_law::AlgebraicLaw`].
/// Specification element.
pub use algebraic_law::{AlgebraicLaw, LawCheckFn};
/// See [`all_algebraic_laws::all_algebraic_laws`].
/// Specification element.
pub use all_algebraic_laws::all_algebraic_laws;
/// See [`atomic_op::AtomicOp`].
/// Specification element.
pub use atomic_op::AtomicOp;
/// See [`bin_op::BinOp`].
/// Specification element.
pub use bin_op::BinOp;
pub use bin_op::{BinOpResult, OpIntensity};
/// See [`buffer_access::BufferAccess`].
/// Specification element.
pub use buffer_access::BufferAccess;
/// See [`by_category::by_category`].
/// Specification element.
pub use by_category::by_category;
/// See [`by_id::by_id`].
/// Specification element.
pub use by_id::by_id;
/// See [`catalog_is_complete::catalog_is_complete`].
/// Specification element.
pub use catalog_is_complete::catalog_is_complete;
/// See [`category::Category`] + backend-availability helpers.
/// Specification element.
pub use category::{BackendAvailability, BackendAvailabilityPredicate, Category};
/// See [`collective_op::{CollectiveOp, CommGroup}`].
/// Specification element.
pub use collective_op::{CollectiveOp, CommGroup};
/// See [`convention::Convention`].
/// Specification element.
pub use convention::Convention;
/// See [`data_type::DataType`].
/// Specification element.
pub use data_type::{DataType, QuantizationScale, QuantizationZeroPoint, TypeId};
/// See [`engine_invariant::EngineInvariant`].
/// Specification element.
pub use engine_invariant::{EngineInvariant, InvariantId};
/// See [`expr_variant::expr_variants`].
/// Specification element.
pub use expr_variant::expr_variants;
/// See [`float_type::FloatType`].
/// Specification element.
pub use float_type::FloatType;
/// See [`golden_sample::GoldenSample`].
/// Specification element.
pub use golden_sample::GoldenSample;
/// See [`intrinsic_table::IntrinsicTable`].
/// Specification element.
pub use intrinsic_table::{IntrinsicLowering, IntrinsicTable};
/// See [`invariant::Invariant`].
/// Specification element.
pub use invariant::Invariant;
/// See [`invariant_category::InvariantCategory`].
/// Specification element.
pub use invariant_category::InvariantCategory;
/// See [`invariants::invariants`].
/// Specification element.
pub use invariants::{empty_test_family, invariants};
/// See [`kat_vector::KatVector`].
/// Specification element.
pub use kat_vector::KatVector;
/// See [`law_catalog::law_catalog`].
/// Specification element.
pub use law_catalog::law_catalog;
/// See [`layer::Layer`].
/// Specification element.
pub use layer::Layer;
/// See [`metadata_category::MetadataCategory`].
/// Specification element.
pub use metadata_category::MetadataCategory;
/// See [`monotonic_direction::MonotonicDirection`].
/// Specification element.
pub use monotonic_direction::MonotonicDirection;
/// See [`numeric_semantics::NumericSemantics`].
/// Specification element.
pub use numeric_semantics::{
    dequantize_grouped_f32, f32_to_f8e4m3, f32_to_f8e5m2, f32_to_fp4, f32_to_nf4,
    f8e4m3_decode_table, f8e4m3_to_f32, f8e5m2_decode_table, f8e5m2_to_f32, fp4_to_f32, i32_to_i4,
    i4_to_i32, nf4_to_f32, numeric_semantics_for, InfinityBehavior, NanBehavior, NumericFormat,
    NumericSemantics, OverflowBehavior, RoundingMode, SaturationBehavior, SignedZeroBehavior,
    SubnormalBehavior, FP4_DECODE_TABLE, I4_DECODE_TABLE, NF4_QUANTILE_TABLE,
    NUMERIC_SEMANTICS_SCHEMA_VERSION,
};
/// See [`op_contract::OperationContract`] and its component types.
pub use op_contract::{
    CapabilityId, CostHint, DeterminismClass, OperationContract, SideEffectClass,
};
/// See [`op_metadata::OpMetadata`].
/// Specification element.
pub use op_metadata::OpMetadata;
/// See [`op_signature::OpSignature`].
/// Specification element.
pub use op_signature::OpSignature;
pub use op_signature::SignatureParam;
/// See [`pg_node_kind::PgNodeKind`].
/// Specification element.
pub use pg_node_kind::PgNodeKind;
/// See [`semiring::Semiring`].
pub use semiring::Semiring;
/// See [`subgroup_reduce_op::SubgroupReduceOp`].
/// Specification element.
pub use subgroup_reduce_op::SubgroupReduceOp;
/// See [`ternary_op::TernaryOp`].
/// Specification element.
pub use ternary_op::TernaryOp;
/// See [`test_descriptor::TestDescriptor`].
/// Specification element.
pub use test_descriptor::TestDescriptor;
/// See [`un_op::UnOp`].
/// Specification element.
pub use un_op::UnOp;
/// See [`verification::Verification`].
/// Specification element.
pub use verification::Verification;

/// Intrinsic descriptors.
/// Specification element.
mod intrinsic_descriptor;
/// See [`intrinsic_descriptor::IntrinsicDescriptor`] and its identifying types.
pub use intrinsic_descriptor::{Backend, BackendId, CpuFn, IntrinsicDescriptor};
