//! Driver projections and policy keyed by canonical operation identity.
//!
//! The driver owns no operation identity. Host-side capabilities such as
//! indirect dispatch, NVMe ingest, and zero-copy mapping are reached through
//! the backend capability surface and `vyre-runtime`, never through an
//! operation registration.

/// Policy enforcement contracts.
pub(crate) mod enforce;
/// Target-facet semantic identity validation.
pub(crate) mod intrinsic_adapter;
/// Operation identifier migrations and deprecations.
pub(crate) mod migration;
/// Operation mutation classification.
pub(crate) mod mutation;

pub use enforce::{Chain, EnforceGate, EnforceVerdict};
pub use intrinsic_adapter::{validate_intrinsic_lowering, IntrinsicRegistrationError};
pub use migration::DEPRECATED_OP_CODE;
pub use migration::{
    deprecation_diagnostic, AttrMap, AttrValue, Deprecation, Migration, MigrationError,
    MigrationRegistry, Semver,
};
pub use mutation::MutationClass;
