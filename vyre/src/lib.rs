//! # vyre
//!
//! Stable facade for frontend IR, whole-program compilation, authenticated
//! artifact materialization, typed submission, and reference semantics.
//!
//! Production execution starts from a validated `ProgramGraph`, compiles one
//! immutable artifact through [`compiler`], and materializes its authenticated
//! target payload through [`ArtifactSession`]. Raw `Program` dispatch remains
//! available only in explicit reference and conformance adapters.

/// The vyre Program model.
///
/// This module defines `Program`, the frozen, serializable model that every
/// frontend emits and every backend consumes. It has zero external
/// dependencies so that spec tools can parse it without pulling in GPU
/// libraries.
/// Public API re-export.
pub use vyre_foundation::ir;

/// Soundness markers and precision contracts from the frozen specification.
/// Public API re-export.
pub use vyre_spec::soundness;

/// Whole-program compiler request, artifact, payload, and target-facet APIs.
pub use vyre_megakernel as compiler;

/// Canonical compiler artifact and request types.
pub use vyre_megakernel::{
    Artifact, ArtifactEnvelope, CompileRequest, ExternalFacts, SearchBudget, TargetPayload,
    TargetPayloadFormat, TargetProfile, ValidatedCompileRequest,
};

/// Retry classification shared by every diagnostic that can be retried.
pub use vyre_foundation::diagnostics::RetryClass;
/// Domain-neutral tagged byte-range contract.
pub use vyre_foundation::match_result;
/// Authenticated artifact admission, materialization, and recovery.
pub use vyre_runtime::artifact_admission::{
    admit_artifact, admit_envelope, ArtifactAdmissionError, ArtifactSession, ArtifactSessionError,
    RetainedArtifactSession,
};
/// Resident-queue submission against an admitted artifact.
pub use vyre_runtime::persistent_executor::PersistentExecutor;

pub use vyre_driver::{ArtifactInstance, BindingSet, Completion, DeviceIdentity, Submission};

/// Canonical frontend IR program and validation entry point.
pub use ir::Program;
pub use vyre_foundation::validate::validate;

/// Domain-neutral tagged byte range shared by source-processing products.
pub use vyre_foundation::match_result::ByteRange;
