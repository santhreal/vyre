//! Substrate-neutral verified lowering for Vyre.
//!
//! `lower_scheduled` applies a validated selected phase to schedule-free
//! `Program` IR and delegates to `lower_physical`, which expands and optimizes
//! already physical IR before building verified physical kernel IR. The result
//! is the only type accepted by megakernel target compilation. Raw descriptor
//! fixtures use `verify_descriptor`, whose bounded canonicalization only orders
//! pure same-body dependencies needed by emitters.
//!
//! ```text
//! vyre-foundation Program
//!         ↓ semantic optimize once
//! verified KernelDescriptor
//!         ↓
//! concrete emitter strategy
//!         ↓
//! backend artifact
//! ```
//!
//! `KernelDescriptor` carries binding layout, dispatch shape, and a lowered
//! kernel body. Descriptor analyses are read-only. Semantic rewrites belong in
//! `vyre-foundation`; target strategy belongs in concrete emitters and drivers.

pub mod analyses;
/// Byte-stability harness for emitted backend artifacts. Test-only, like
/// `descriptor_builder`: enable `test-fixtures` to reach it.
#[cfg(any(test, feature = "test-fixtures"))]
/// Golden-corpus framing for emitted artifacts. Every consumer is a byte-stability
/// test, so this is not part of the shipped surface: enable `test-fixtures` to reach it.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod artifact_golden;
pub(crate) mod canonicalize;
pub(crate) mod descriptor;
/// Fixture builders for kernel descriptors. Every consumer is a test, so this
/// is not part of the shipped surface: enable `test-fixtures` to reach it.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod descriptor_builder;
/// Hostile descriptor corpus shared by the backend emit matrices. Every consumer
/// is a test, so this is not part of the shipped surface: enable `test-fixtures`
/// to reach it.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod emit_adversarial_corpus;
pub(crate) mod equivalence;
pub(crate) mod error;
mod level_stage;
pub(crate) mod lower;
pub(crate) mod op_facts;
pub mod operand_class;
/// Backend-neutral `Program` corpus shared by byte-stability goldens.
/// Test-only, like `descriptor_builder`: enable `test-fixtures` to reach it.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod program_stability_corpus;
pub(crate) mod target;
/// The physical-IR variant space every emitter contract suite enumerates.
/// Test-only, like `descriptor_builder`: enable `test-fixtures` to reach it.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod variant_space;
mod verified_lowering;
pub(crate) mod verify;

/// Verify a raw descriptor, apply bounded representation canonicalization, and
/// verify the emitter-ready result.
///
/// This boundary does not perform semantic optimization. Production semantic
/// `Program` callers use [`lower_scheduled`]; descriptor fixtures and tooling
/// use this function before invoking a pure emitter.
pub fn verify_descriptor(desc: &KernelDescriptor) -> Result<KernelDescriptor, VerifyFailure> {
    if let Err(errors) = verify::verify(desc) {
        return Err(VerifyFailure::Input(errors));
    }
    let canonical = canonicalize::canonicalize_for_emit(desc);
    if let Err(errors) = verify::verify(&canonical) {
        return Err(VerifyFailure::Output(errors));
    }
    Ok(canonical)
}

/// Which descriptor verification step failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyFailure {
    /// The descriptor was invalid before canonicalization.
    Input(Vec<verify::VerifyError>),
    /// Bounded representation canonicalization produced an invalid descriptor.
    Output(Vec<verify::VerifyError>),
}

impl VerifyFailure {
    /// Return the verifier errors carried by either failure stage.
    pub fn errors(&self) -> &[verify::VerifyError] {
        match self {
            VerifyFailure::Input(e) | VerifyFailure::Output(e) => e,
        }
    }
}

pub use descriptor::{
    body_contains_grid_fence, descriptor_trap_tags, dispatch_segments,
    scan_construct_intent_mapping, AsyncTransaction, AsyncTransactionError, AsyncWaitSpec,
    BarrierPhase, BindingLayout, BindingSlot, BindingVisibility, DescriptorIntent,
    DescriptorIntentError, DescriptorIntentEvidence, DescriptorIntentKind, DescriptorIntentSet,
    DescriptorIntentStrategy, DescriptorTrapTag, Dispatch, DispatchSplitError, FragmentOperand,
    FragmentValue, GridIndexSpace, IntentAnnotatedDescriptor, KernelBody, KernelDescriptor,
    KernelOp, KernelOpKind, LiteralValue, MatrixMmaElement, MatrixMmaLayout, MatrixMmaSpec,
    MatrixSpecError, MatrixTileShape, MemoryClass, MemoryProxyFence, OpaqueExprData,
    OpaqueNodeData, PhysicalSchedule, ScanConstructIntentClass, ScanConstructIntentMapping,
    StageSlot, StorageLayout, StorageLayoutError, StorageLifetime, StorageRegion, TensorAccessMap,
    TransactionScope, DESCRIPTOR_INTENT_SCHEMA_VERSION, PHYSICAL_SCHEDULE_VERSION,
    SCAN_CONSTRUCT_INTENT_MAPPINGS, STORAGE_LAYOUT_VERSION, TRAP_SIDECAR_NAME, TRAP_SIDECAR_WORDS,
};
pub use descriptor::{KernelOpsIter, Name};
pub use equivalence::{check_effects, BindingEffects, EffectSignature, EquivalenceError};
pub use error::LowerError;
pub use level_stage::registered_level_stage;
/// Re-exported so a caller building a `KernelDescriptor` by hand through
/// `descriptor_builder` can place a Shared or Scratch binding in the range
/// `verify` accepts. A shared slot below this value is rejected with
/// `VerifyErrorKind::WorkgroupBindingInHostRange`.
pub use lower::{lower, WORKGROUP_SLOT_BASE};
pub use op_facts::{facts_for, NestedBodies, NestedBodyControl, OpFacts};
pub use target::{
    required_subgroup_capabilities, validate_workgroup_size, EmissionTargetCapabilities,
    SubgroupCapabilities, WorkgroupLimitViolation, WorkgroupLimits,
};
pub use verified_lowering::{
    lower_baseline, lower_physical, lower_scheduled, PhysicalKernel, PhysicalLowering,
    PhysicalLoweringError,
};
pub use verify::format_verify_errors;
pub use verify::{verify, VerifyError, VerifyErrorKind, VerifyResult};
/// Re-exported so consumers matching/constructing `KernelOpKind::SubgroupReduce`
/// can name the reduction operator without depending on `vyre-foundation`.
pub use vyre_foundation::ir::SubgroupReduceOp;
