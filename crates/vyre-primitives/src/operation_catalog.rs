//! Canonical semantic-operation view for this crate's Category C intrinsics.
//!
//! Builders submit foundation-owned [`vyre_foundation::operation::OperationRegistration`] values. This
//! module retains the feature-gated view without owning a second operation
//! identity or fixture schema.

use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};

/// Iterate every registration in the intrinsic tier.
///
/// Selects on [`OperationTier::Intrinsic`], so a library composition
/// registered through `OperationRegistration::library_unconstrained` is not
/// returned here. Library-tier readers use
/// `vyre_libs_builder::plumbing::registration::operation_catalog::library_entries`.
pub fn intrinsic_entries() -> impl Iterator<Item = SemanticOperation> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Intrinsic)
}
