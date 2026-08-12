#![forbid(unsafe_code)]
//! Ahead-of-time packaging for canonical Vyre compiler artifacts.
//!
//! The crate compiles frontend `Program` adapters through `vyre-megakernel`,
//! attaches bytes from a registered pure target compiler facet, and packages
//! the authenticated [`ArtifactEnvelope`] with deployment metadata.
#![deny(missing_docs)]

pub mod artifact;
pub mod bundle;
/// Runtime-cache compatibility for AOT-emitted artifacts (audit P0 #26).
pub mod cache;
pub mod compile;
pub mod launcher;
pub mod manifest;

pub use artifact::TargetId;
pub use bundle::{bundle, package_artifact, read_bundle_artifact, BundleError, DeploymentBundle};
pub use compile::{compile, compile_with_resolver, CompileError};
pub use launcher::{emit_launcher_rust, LauncherError, LauncherOpts};
pub use manifest::Manifest;
pub use vyre_megakernel::{
    Artifact, ArtifactEnvelope, TargetEntryPoint, TargetPayload, TargetPayloadFormat,
    TargetProfile, TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};

/// Crate version surfaced into emitted artifacts and manifests.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Snapshot the driver-tier observability surface from inside vyre-aot.
/// Exposes substrate counters + decision histograms so callers
/// emitting AOT bundles can include them in their build provenance.
#[must_use]
pub fn observability_snapshot() -> vyre_driver::observability::DriverObservability {
    vyre_driver::observability::DriverObservability::snapshot()
}
