//! Canonical semantic-operation view for reusable primitives.
//!
//! Primitive builders submit foundation-owned [`OperationRegistration`] values.
//! This module retains the feature-gated primitive view without owning a second
//! operation identity or fixture schema.

use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};

/// Deterministic fixture input cases.
pub type InputsFn = vyre_foundation::operation::OperationFixtures;

/// Deterministic expected-output fixtures.
pub type ExpectedFn = vyre_foundation::operation::OperationFixtures;

/// Iterate over canonical reusable-primitive registrations.
pub fn all_entries() -> impl Iterator<Item = SemanticOperation> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == OperationTier::Primitive)
}
