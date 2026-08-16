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
        | CompilerFailureKind::InvalidDeviceFacts => DiagnosticStage::Validate,
        CompilerFailureKind::DependencyCycle | CompilerFailureKind::FinalistEvaluation => {
            DiagnosticStage::Plan
        }
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
        | CompilerFailureKind::IncompatibleTargetPayload => DiagnosticStage::Admit,
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
