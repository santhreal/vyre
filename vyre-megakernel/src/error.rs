//! Compiler failure classification and the diagnostic every failure carries.

use thiserror::Error;
use vyre_foundation::diagnostics::{Diagnostic, DiagnosticStage, OpLocation, RetryClass};

/// Compiler-internal failure classification projected into the shared diagnostic protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum CompilerFailureKind {
    /// A program failed structural validation.
    InvalidProgram,
    /// A symbolic extent had no exact binding.
    MissingSymbol,
    /// A binding was supplied for no graph symbol.
    UnknownSymbol,
    /// An ordering constraint made the dependency graph cyclic.
    DependencyCycle,
    /// Checked size arithmetic overflowed.
    ResourceOverflow,
    /// A value representation has no static packed size.
    UnsizedResource,
    /// The canonical artifact exceeded the caller's bound.
    ArtifactLimit,
    /// Artifact framing or canonical body data was malformed.
    MalformedArtifact,
    /// Artifact schema is not supported by this compiler version.
    VersionSkew,
    /// Artifact content identity did not match its body.
    DigestMismatch,
    /// Target payload framing or metadata was malformed.
    MalformedTargetPayload,
    /// Target payload schema or format version is incompatible.
    TargetPayloadVersionSkew,
    /// Target payload content identity did not match its metadata and bytes.
    TargetPayloadDigestMismatch,
    /// Target payload metadata names a different neutral artifact record.
    TargetPayloadAssociationMismatch,
    /// No attached target payload satisfies the required format identity.
    IncompatibleTargetPayload,
    /// Mandatory schedule-search bounds are zero or otherwise invalid.
    InvalidSearchBudget,
    /// A constant graph value has no verified content identity.
    MissingConstantIdentity,
    /// A constant identity was supplied for a non-constant graph value.
    UnknownConstantIdentity,
    /// Device facts contradict what any device can report.
    InvalidDeviceFacts,
    /// A finalist could not be compiled for the target or timed on the device.
    FinalistEvaluation,
    /// A representative input was supplied for an unknown or graph-produced value.
    UnknownRepresentativeInput,
    /// A representative input byte length does not match the graph value's static size.
    RepresentativeInputLengthMismatch,
    /// The stated compile objective is internally inconsistent.
    InvalidObjective,
    /// A stated objective metric is priced by a fact this device never reported.
    MissingCalibratedFact,
    /// Every legal candidate exceeds a hard bound the objective states.
    ObjectiveBoundViolated,
    /// One artifact cannot satisfy the artifact coverage policy the objective
    /// states.
    PortfolioCoverageUnsatisfied,
    /// The stated specialization contract declares an axis or domain nothing can
    /// read.
    InvalidSpecializationContract,
    /// A variant guard reads an undeclared axis, states values its domain does
    /// not hold, or conjoins terms that cannot hold at once.
    InvalidVariantGuard,
    /// Two variant guards admit the same facts at one precedence.
    GuardOverlap,
    /// Part of the declared domain is served by no variant and no remainder.
    GuardCoverageGap,
    /// A guarded artifact set holds artifacts from more than one compile.
    PortfolioProvenanceMismatch,
    /// The authenticated target is not the one a guarded set was compiled for.
    TargetIdentityMismatch,
    /// No admitted variant serves the stated workload and the remainder is
    /// declared unsupported.
    UnsupportedWorkload,
    /// The allocation and layout plan states physical storage no runtime could
    /// allocate and bind exactly.
    InvalidAllocationPlan,
    /// The device reports holding fewer bytes than the selected allocation plan
    /// requires, so the plan measured is not the plan that ran.
    UnreconciledResidentBytes,
    /// Mesh facts contradict what any device mesh can report, or their identity
    /// does not cover them.
    InvalidMeshFacts,
    /// The mesh topology plan states placement no mesh could carry.
    InvalidMeshTopology,
    /// A device of the mesh holds fewer bytes than its share of the plan.
    MeshCapacityExceeded,
    /// The caller required a schedule family and no legal candidate exercises
    /// it, so the compile is refused rather than served a different schedule.
    RequiredScheduleUnreachable,
}

impl CompilerFailureKind {
    /// Stable ASCII code for logs and serialized evidence.
    #[must_use]
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidProgram => "MKC001_INVALID_PROGRAM",
            Self::MissingSymbol => "MKC002_MISSING_SYMBOL",
            Self::UnknownSymbol => "MKC003_UNKNOWN_SYMBOL",
            Self::DependencyCycle => "MKC010_DEPENDENCY_CYCLE",
            Self::ResourceOverflow => "MKC011_RESOURCE_OVERFLOW",
            Self::UnsizedResource => "MKC012_UNSIZED_RESOURCE",
            Self::ArtifactLimit => "MKC013_ARTIFACT_LIMIT",
            Self::MalformedArtifact => "MKC014_MALFORMED_ARTIFACT",
            Self::VersionSkew => "MKC015_VERSION_SKEW",
            Self::DigestMismatch => "MKC016_DIGEST_MISMATCH",
            Self::MalformedTargetPayload => "MKC017_MALFORMED_TARGET_PAYLOAD",
            Self::TargetPayloadVersionSkew => "MKC018_TARGET_PAYLOAD_VERSION_SKEW",
            Self::TargetPayloadDigestMismatch => "MKC019_TARGET_PAYLOAD_DIGEST_MISMATCH",
            Self::TargetPayloadAssociationMismatch => "MKC020_TARGET_PAYLOAD_ASSOCIATION_MISMATCH",
            Self::IncompatibleTargetPayload => "MKC021_INCOMPATIBLE_TARGET_PAYLOAD",
            Self::InvalidSearchBudget => "MKC022_INVALID_SEARCH_BUDGET",
            Self::MissingConstantIdentity => "MKC023_MISSING_CONSTANT_IDENTITY",
            Self::UnknownConstantIdentity => "MKC024_UNKNOWN_CONSTANT_IDENTITY",
            Self::InvalidDeviceFacts => "MKC025_INVALID_DEVICE_FACTS",
            Self::FinalistEvaluation => "MKC026_FINALIST_EVALUATION",
            Self::UnknownRepresentativeInput => "MKC027_UNKNOWN_REPRESENTATIVE_INPUT",
            Self::RepresentativeInputLengthMismatch => {
                "MKC028_REPRESENTATIVE_INPUT_LENGTH_MISMATCH"
            }
            Self::InvalidObjective => "MKC029_INVALID_OBJECTIVE",
            Self::MissingCalibratedFact => "MKC030_MISSING_CALIBRATED_FACT",
            Self::ObjectiveBoundViolated => "MKC031_OBJECTIVE_BOUND_VIOLATED",
            Self::PortfolioCoverageUnsatisfied => "MKC032_PORTFOLIO_COVERAGE_UNSATISFIED",
            Self::InvalidSpecializationContract => "MKC033_INVALID_SPECIALIZATION_CONTRACT",
            Self::InvalidVariantGuard => "MKC034_INVALID_VARIANT_GUARD",
            Self::GuardOverlap => "MKC035_GUARD_OVERLAP",
            Self::GuardCoverageGap => "MKC036_GUARD_COVERAGE_GAP",
            Self::PortfolioProvenanceMismatch => "MKC037_PORTFOLIO_PROVENANCE_MISMATCH",
            Self::TargetIdentityMismatch => "MKC038_TARGET_IDENTITY_MISMATCH",
            Self::UnsupportedWorkload => "MKC039_UNSUPPORTED_WORKLOAD",
            Self::InvalidAllocationPlan => "MKC040_INVALID_ALLOCATION_PLAN",
            Self::UnreconciledResidentBytes => "MKC041_UNRECONCILED_RESIDENT_BYTES",
            Self::InvalidMeshFacts => "MKC042_INVALID_MESH_FACTS",
            Self::InvalidMeshTopology => "MKC043_INVALID_MESH_TOPOLOGY",
            Self::MeshCapacityExceeded => "MKC044_MESH_CAPACITY_EXCEEDED",
            Self::RequiredScheduleUnreachable => REQUIRED_SCHEDULE_UNREACHABLE,
        }
    }
}

const fn diagnostic_stage(code: CompilerFailureKind) -> DiagnosticStage {
    match code {
        CompilerFailureKind::InvalidProgram
        | CompilerFailureKind::MissingSymbol
        | CompilerFailureKind::UnknownSymbol
        | CompilerFailureKind::InvalidSearchBudget
        | CompilerFailureKind::MissingConstantIdentity
        | CompilerFailureKind::UnknownConstantIdentity
        | CompilerFailureKind::InvalidDeviceFacts
        | CompilerFailureKind::UnknownRepresentativeInput
        | CompilerFailureKind::RepresentativeInputLengthMismatch
        | CompilerFailureKind::InvalidObjective
        | CompilerFailureKind::MissingCalibratedFact
        | CompilerFailureKind::InvalidSpecializationContract
        | CompilerFailureKind::InvalidMeshFacts
        | CompilerFailureKind::InvalidVariantGuard => DiagnosticStage::Validate,
        CompilerFailureKind::DependencyCycle
        | CompilerFailureKind::FinalistEvaluation
        | CompilerFailureKind::ObjectiveBoundViolated
        | CompilerFailureKind::PortfolioCoverageUnsatisfied
        | CompilerFailureKind::GuardOverlap
        | CompilerFailureKind::GuardCoverageGap
        | CompilerFailureKind::InvalidAllocationPlan
        | CompilerFailureKind::InvalidMeshTopology
        | CompilerFailureKind::MeshCapacityExceeded
        | CompilerFailureKind::RequiredScheduleUnreachable
        | CompilerFailureKind::UnreconciledResidentBytes => DiagnosticStage::Plan,
        CompilerFailureKind::ResourceOverflow | CompilerFailureKind::UnsizedResource => {
            DiagnosticStage::Lower
        }
        CompilerFailureKind::ArtifactLimit => DiagnosticStage::Emit,
        CompilerFailureKind::MalformedArtifact
        | CompilerFailureKind::VersionSkew
        | CompilerFailureKind::DigestMismatch
        | CompilerFailureKind::MalformedTargetPayload
        | CompilerFailureKind::TargetPayloadVersionSkew
        | CompilerFailureKind::TargetPayloadDigestMismatch
        | CompilerFailureKind::TargetPayloadAssociationMismatch
        | CompilerFailureKind::IncompatibleTargetPayload
        | CompilerFailureKind::PortfolioProvenanceMismatch
        | CompilerFailureKind::TargetIdentityMismatch
        | CompilerFailureKind::UnsupportedWorkload => DiagnosticStage::Admit,
    }
}

const fn diagnostic_retry(code: CompilerFailureKind) -> RetryClass {
    match code {
        CompilerFailureKind::VersionSkew
        | CompilerFailureKind::TargetPayloadVersionSkew
        | CompilerFailureKind::IncompatibleTargetPayload => RetryClass::RecompileSource,
        _ => RetryClass::Never,
    }
}

/// Compilation or artifact-validation failure.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("{diagnostic}")]
pub struct CompileError {
    /// Structured stable diagnostic.
    pub diagnostic: Diagnostic,
}

pub(crate) fn serialization_failure(error: serde_json::Error) -> CompileError {
    failure(
        CompilerFailureKind::MalformedArtifact,
        "artifact.body",
        error.to_string(),
        "use values representable by the canonical artifact schema",
    )
}

pub(crate) fn overflow(path: impl Into<String>, message: impl Into<String>) -> CompileError {
    failure(
        CompilerFailureKind::ResourceOverflow,
        path,
        message,
        "reduce resolved extents or split the graph before compilation",
    )
}

/// The rejection a consumer reports when no admitted variant serves a workload.
///
/// A guarded set whose remainder is declared unsupported has one legal answer
/// for facts nothing admits, and it is a failure with a corrective action, not a
/// nearby variant.
#[must_use]
pub fn unsupported_workload(message: String) -> CompileError {
    failure(
        CompilerFailureKind::UnsupportedWorkload,
        "specialization.remainder",
        message,
        "state facts a retained guard admits, or compile a set with a generic remainder",
    )
}

/// Diagnostic code a compile carries when the caller required a schedule family
/// no legal candidate plan exercises.
///
/// A caller iterating schedule families reads this instead of the message text,
/// so a graph that cannot be fused is told apart from a graph that failed to
/// compile.
pub const REQUIRED_SCHEDULE_UNREACHABLE: &str = "MKC045_REQUIRED_SCHEDULE_UNREACHABLE";

/// Whether `error` is the refusal of a required schedule family.
#[must_use]
pub fn is_required_schedule_unreachable(error: &CompileError) -> bool {
    error.diagnostic.code.as_str() == REQUIRED_SCHEDULE_UNREACHABLE
}

pub(crate) fn failure(
    code: CompilerFailureKind,
    path: impl Into<String>,
    message: impl Into<String>,
    fix: impl Into<String>,
) -> CompileError {
    let stage = diagnostic_stage(code);
    let retry = diagnostic_retry(code);
    CompileError {
        diagnostic: Diagnostic::error(code.as_str(), message.into())
            .with_stage(stage)
            .with_location(OpLocation::op("vyre-megakernel").with_path(path))
            .with_fix(fix.into())
            .with_retry(retry),
    }
}
