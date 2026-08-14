//! Canonical semantic-operation view for this crate's Category C intrinsics.
//!
//! Builders submit foundation-owned [`OperationRegistration`] values. This
//! module retains the feature-gated view without owning a second operation
//! identity or fixture schema.

use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};

/// Iterate over canonical registrations owned by this crate.
pub fn all_entries() -> impl Iterator<Item = SemanticOperation> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Intrinsic)
}
