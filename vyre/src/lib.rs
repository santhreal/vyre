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

#[cfg(test)]
mod tests {
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
