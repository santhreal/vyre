//! # vyre
//!
//! Stable facade for frontend IR, whole-program compilation, authenticated
//! artifact materialization, typed submission, and reference semantics.
//!
//! Production execution starts from a validated `ProgramGraph`, compiles one
//! immutable artifact through [`compiler`], and materializes its authenticated
//! target payload through [`ArtifactSession`]. Raw `Program` dispatch remains
//! available only in explicit reference and conformance adapters.

// Feature-selected drivers submit backend registrations at link time. These
// private function pointers make the linker retain each provider without
// adding a second public backend API or a dependency-gate exemption.
#[cfg(feature = "cuda")]
#[used]
static PRIMARY_PROVIDER_LINK: fn() -> Option<&'static str> =
    vyre_driver_cuda::registered_backend_id;
#[cfg(feature = "wgpu")]
#[used]
static PORTABLE_PROVIDER_LINK: fn() -> Option<&'static str> =
    vyre_driver_wgpu::registered_backend_id;

/// The vyre Program and Graph model.
///
/// This module defines `Program` and `ProgramGraph`, the frozen, serializable
/// model that every frontend emits and every backend consumes. It has zero
/// external dependencies so that spec tools can parse it without pulling in GPU
/// libraries.
/// Public API re-export.
pub use vyre_foundation::ir;

/// Deterministic framing and content-addressed hashing helpers.
pub use vyre_foundation::hashing;

/// Domain-neutral schema and dialect translation contracts.
pub use vyre_foundation::dialect;

/// Numerical contracts, quantization schemas, and floating-point precision policies.
pub use vyre_foundation::numeric;

/// Canonical semantic operation registration and target facet views.
pub use vyre_foundation::operation;

/// Soundness markers and precision contracts from the frozen specification.
/// Public API re-export.
pub use vyre_spec::soundness;

/// Whole-program compiler request, artifact, payload, receipt, and target-facet APIs.
pub mod compiler {
    pub use vyre_megakernel::specialization::{
        compile_specialized_portfolio, compile_specialized_portfolio_measured, AxisDomain,
        AxisValue, CoverageProof, GuardTerm, PortfolioEnvelope, PortfolioVariant, RemainderKind,
        SpecializationAxis, SpecializationContract, SpecializedPortfolio, SpecializedRemainder,
        TargetCapabilityAxis, TargetResourceAxis, VariantGuard, MAX_COVERAGE_CELLS,
        MAX_PROPOSED_VARIANTS, PORTFOLIO_ENVELOPE_SCHEMA_VERSION, SPECIALIZATION_SCHEMA_VERSION,
    };
    pub use vyre_megakernel::{
        attach_target, compile, compile_measured, compile_portfolio, compile_portfolio_measured,
        compile_selected_modules, target_identity, AbiAccess, Artifact, ArtifactAbi,
        ArtifactEnvelope, ArtifactNodeId, ArtifactPortfolio, ArtifactValueId, BarrierPhaseRecord,
        BarrierRecord, BoundViolation, CompileError, CompileObjective, CompileRequest,
        CoveragePolicy, DeclaredConstraints, DependencyEdge, DependencyEndpoint, DependencyKind,
        DerivationStep, DerivedFamily, DeviceFacts, Digest, EmittedResources, EmittedTargetModule,
        EntryAbiRecord, EntryPersistence, EntryResourceBinding, ExecutionMode, ExternalFacts,
        FinalistEvaluator, FrontierTopology, FusionGroupId, FusionRecord, FusionRejection,
        GeometryRecord, LaunchObservation, LaunchResourceIntent, LawCitation,
        MaterializationReason, MaterializationRecord, MetricFigures, MetricSequence,
        ModuleNumericRecord, NodeRecord, NumericRecord, ObjectiveBounds, ObjectiveMetric,
        PlanMeasurement, PortfolioPolicy, Provenance, PruneReason, PrunedFamily, PrunedLaw,
        RealTimeDeadline, RealTimeObjective, RealTimeViolation, RequiredFact, RequiredSchedule,
        ResourceAbiRecord, ResourceEnvelope, ResourceLifetime, ResourceNameCollision,
        ResourceRecord, RiskStatistic, ScheduleProduction, SearchBudget, SearchCertificate,
        SearchWork, SelectedLowering, SelectedModule, SelectedPlan, TargetArmAssignment,
        TargetCompileError, TargetCompiler, TargetEntryPoint, TargetModuleBundle,
        TargetModuleImage, TargetPayload, TargetPayloadFormat, TargetProfile, TargetResourceAccess,
        TargetResourceBinding, TargetResourceMemory, ValidatedCompileRequest, WorkloadAggregation,
        WorkloadArrivalTrace, WorkloadClass, WorkloadProfile, ARTIFACT_ENVELOPE_SCHEMA_VERSION,
        ARTIFACT_SCHEMA_VERSION, OBJECTIVE_SCHEMA_VERSION, REAL_TIME_OBJECTIVE_SCHEMA_VERSION,
        SCHEDULE_GRAMMAR_VERSION, TARGET_MODULE_BUNDLE_SCHEMA_VERSION,
        TARGET_PAYLOAD_SCHEMA_VERSION,
    };
}

/// Shared structured diagnostic protocol.
pub use vyre_foundation::diagnostics;
/// Retry classification shared by every diagnostic that can be retried.
pub use vyre_foundation::diagnostics::RetryClass;
/// Domain-neutral tagged byte-range contract.
pub use vyre_foundation::match_result;
/// Authenticated artifact admission, materialization, and recovery.
pub use vyre_runtime::artifact_admission::{
    admit_artifact, admit_envelope, ArtifactAdmissionError, ArtifactSession, ArtifactSessionError,
    GeneratedDataSource, ResourceDataSource, ResourceIngestionError, ResourceManifest,
    ResourceManifestEntry, ResourceManifestSource, RetainedArtifactSession, TypedResource,
    TypedResourceDataset,
};
/// Resident-queue submission against an admitted artifact.
pub use vyre_runtime::persistent_executor::PersistentExecutor;

pub use vyre_driver::{
    registered_backends, ArtifactInstance, BackendRegistration, BindingSet, Completion,
    DeviceIdentity, Submission,
};

/// Canonical frontend IR program and validation entry point.
pub use ir::{Program, ProgramGraph};
pub use vyre_foundation::validate::validate;

/// Domain-neutral tagged byte range shared by source-processing products.
pub use vyre_foundation::match_result::ByteRange;

/// Typed configuration schema, precedence resolution, and credential secrecy.
pub use vyre_foundation::{
    render_cli_help, render_configuration_reference_markdown, ConfigFieldDef, ConfigLayer,
    ConfigMutability, ConfigPartition, ConfigSecrecy, ConfigType, ConfigValue, IdentityImpact,
    ResolvedConfiguration, CANONICAL_CONFIG_FIELDS,
};
#[cfg(test)]
mod tests {
    // Both cases that call this are feature-selected, so a default build links
    // neither and the helper is dead code the workspace lint floor rejects.
    #[cfg(any(feature = "cuda", feature = "wgpu"))]
    fn backend_is_registered(id: &str) -> bool {
        vyre_driver::registered_backends()
            .expect("feature-selected backend registrations must not conflict")
            .iter()
            .any(|registration| registration.id == id)
    }

    /// WHY: optional facade features promise to link their inventory provider.
    /// Merely listing an optional dependency does not keep its registration.
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_feature_links_the_cuda_registration() {
        assert!(backend_is_registered("cuda"));
    }

    /// The WGPU feature carries the same link-time registration contract.
    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_feature_links_the_wgpu_registration() {
        assert!(backend_is_registered("wgpu"));
    }
}
