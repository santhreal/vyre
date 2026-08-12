//! Driver projections and policy keyed by canonical operation identity.

/// Canonical indirect-dispatch operation registration.
pub mod core_indirect;
/// Policy enforcement contracts.
pub mod enforce;
/// Target-facet semantic identity validation.
pub mod intrinsic_adapter;
/// Canonical target-only I/O operation registrations.
pub mod io;
/// Operation identifier migrations and deprecations.
pub mod migration;
/// Operation mutation classification.
pub mod mutation;

pub use core_indirect::INDIRECT_DISPATCH_OP_ID;
pub use enforce::{Chain, EnforceGate, EnforceVerdict};
pub use intrinsic_adapter::{validate_intrinsic_lowering, IntrinsicRegistrationError};
pub use migration::{
    deprecation_diagnostic, AttrMap, AttrValue, Deprecation, Migration, MigrationError,
    MigrationRegistry, Semver,
};
pub use mutation::MutationClass;
