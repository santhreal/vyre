//! vyre-spec is the machine-checkable frozen data contract for the vyre GPU
//! compute IR. Any backend may depend on vyre-spec alone to prove conformance
//! without depending on vyre itself.
//!
//! This crate is intentionally data-only. It has no dependency on downstream
//! crates; backend vendors can use these types as the stable contract
//! for conformance proofs. Example: a conformance runner can read an
//! [`OpSignature`] and verify the byte width expected by a backend primitive.

extern crate alloc;

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
/// Specification element.
mod category;
/// Collective communication operators and communicator handles.
/// Specification element.
mod collective_op;
/// Combine kinds shared by atomics, subgroup reductions and collectives.
/// Specification element.
mod combine;
/// Category enum (A/B/C) + backend-availability predicates.
/// Compatibility and rollout matrix contracts.
mod compatibility;
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
mod extension;
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
/// Specification element.
mod invariants;
/// Compiler level a declaration owns.
/// Specification element.
mod ir_level;
/// Known-answer test vector type  -  deterministic input/output pairs.
/// Specification element.
mod kat_vector;
/// Canonical catalog of algebraic laws exposed via `law_catalog()`.
/// Specification element.
mod law_catalog;
/// Layer enum (IR / backend / runtime)  -  coarse module placement.
/// Specification element.
mod layer;
/// Closed linear types and resource consumption states.
/// Specification element.
mod linear_type;
/// Closed memory effect classifications and semantics.
/// Specification element.
mod memory_effect;
/// Metadata classification for `OpMetadata` entries.
/// Specification element.
mod metadata_category;
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
/// Declarative law families a rewrite may cite.
/// Specification element.
mod region_law;
/// Domain-neutral resource, image, view, plane, sampler, external memory, and timeline sync capabilities.
mod resource_capability;
/// Catalog of invariants every registered op is checked against.
/// Declarative schema registry for all persisted and wire formats.
pub mod schema_registry;
/// Canonical semiring selector for dataflow and algebraic kernels.
mod semiring;
/// Soundness markers and precision contracts for cross-engine analysis data.
pub mod soundness;
/// Subgroup reduction operator enum  -  add/mul/min/max/and/or/xor.
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
/// Specification element for algebraic laws.
pub use algebraic_law::{
    AlgebraicLaw, CounterexampleGenerator, GuardedLaw, LawCheckFn, LawCounterexample, LawDirection,
    LawGuard, LawValidationError, ProofMethod,
};
/// See [`all_algebraic_laws::all_algebraic_laws`].
/// Specification element.
pub use all_algebraic_laws::all_algebraic_laws;
/// See [`atomic_op::AtomicOp`].
/// Specification element.
pub use atomic_op::AtomicOp;
/// See [`bin_op::BinOp`].
/// Specification element.
pub use bin_op::BinOp;
pub use bin_op::{BinOpResult, OpIntensity, OperandSwap};
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
/// See [`combine::CombineKind`].
/// Specification element.
pub use combine::CombineKind;
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
/// See [`ExtensionSchema`].
/// Specification element.
pub use extension::{
    ExtensionAtomicOp, ExtensionAtomicOpId, ExtensionBinOp, ExtensionBinOpId, ExtensionDataType,
    ExtensionDataTypeId, ExtensionField, ExtensionFieldType, ExtensionIdentity, ExtensionNamespace,
    ExtensionNumericalContract, ExtensionOperand, ExtensionOperandKind, ExtensionProofFieldKind,
    ExtensionProofFields, ExtensionResourceBounds, ExtensionRuleConditionId, ExtensionSchema,
    ExtensionSchemaDigest, ExtensionSemVer, ExtensionShapeRule, ExtensionTernaryOp,
    ExtensionTernaryOpId, ExtensionUnOp, ExtensionUnOpId,
};
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
/// See [`ir_level::IrLevel`].
/// Specification element.
pub use ir_level::IrLevel;
/// See [`kat_vector::KatVector`].
/// Specification element.
pub use kat_vector::KatVector;
/// See [`law_catalog::law_catalog`].
/// Specification element.
pub use law_catalog::law_catalog;
/// See [`layer::Layer`].
/// Specification element.
pub use layer::Layer;
/// See [`linear_type::{LinearResourceKind, LinearState}`].
/// Specification element.
pub use linear_type::{LinearResourceKind, LinearState};
/// See [`memory_effect::MemoryEffect`].
/// Specification element.
pub use memory_effect::MemoryEffect;
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
    AliasingContract, CapabilityId, ContractValidationError, CostHint, DeterminismClass,
    NumericBehavior, OperationContract, RangeContract, RangePrecondition, ResourceBoundsContract,
    SemanticContractRecord, ShapeIndexContract, ShapeIndexRelation, SideEffectClass,
    TransformDecision,
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
/// See [`region_law::RegionLawFamily`].
/// Specification element.
pub use region_law::RegionLawFamily;
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
pub use compatibility::{
    derive_artifact_identity, CacheNamespace, CompatibilityCell, CompatibilityDisposition,
    CompatibilityMatrix, GenerationId, NegotiatedContract, NegotiationError, ProtocolDomain,
    ProtocolVersion, RetainedSessionScope, RolloutManager, SessionScopeError, SessionStatus,
    StaleGenerationError, CANONICAL_COMPATIBILITY_CELLS,
};
/// See [`intrinsic_descriptor::IntrinsicDescriptor`] and its identifying types.
pub use intrinsic_descriptor::{Backend, BackendId, CpuFn, IntrinsicDescriptor};
pub use schema_registry::{
    CanonicalField, DefaultsPolicy, FieldType, SchemaBounds, SchemaDefinition, SchemaId,
    SchemaRegistry, CANONICAL_SCHEMA_REGISTRY,
};

pub use resource_capability::{
    all_address_modes, all_alias_set_kinds, all_border_colors, all_color_interpretations,
    all_compare_functions, all_external_event_kinds, all_external_memory_kinds, all_filter_modes,
    all_format_classes, all_image_formats, all_image_view_kinds, all_layout_states,
    all_lifetime_state_kinds, all_mipmap_filter_modes, all_plane_kinds, all_provenance_kinds,
    all_swizzle_components, all_sync_protocols, all_usage_flags, AddressMode,
    AdmittedResourceRecord, BorderColor, ColorInterpretation, CompareFunction, ComponentSwizzle,
    ExternalEventCapability, ExternalEventKind, ExternalMemoryCapability, ExternalMemoryKind,
    FilterMode, FormatClass, ImageDimensions, ImageFormat, ImagePlane, ImageViewDescriptor,
    ImageViewKind, MipmapFilterMode, PlaneKind, ResourceAbiError, ResourceAliasSet,
    ResourceLayoutState, ResourceLifetimeState, ResourceOwnershipState, ResourcePermittedUsages,
    ResourceProvenance, ResourceUsageTransition, SamplerCapability, SamplerDescriptor,
    SubresourceRange, SwizzleComponent, TimelineSyncProtocol,
};
