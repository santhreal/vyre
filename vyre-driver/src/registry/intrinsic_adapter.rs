//! Validation helpers for target-owned operation facets.

use thiserror::Error;
use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};

/// Failure while attaching a target facet to canonical semantic identity.
#[derive(Debug, Clone, Eq, PartialEq, Error)]
pub enum IntrinsicRegistrationError {
    /// The facet names an operation without an intrinsic semantic owner.
    #[error("unknown intrinsic id `{id}` in target facet; register the semantic intrinsic first")]
    UnknownId {
        /// Unrecognized stable intrinsic id.
        id: &'static str,
    },
}

/// Validate that one target-owned intrinsic facet references a canonical semantic owner.
///
/// Target facets never repeat the callable signature, so signature drift is
/// impossible at this boundary.
pub fn validate_intrinsic_lowering(
    operation_id: &'static str,
) -> Result<SemanticOperation, IntrinsicRegistrationError> {
    OperationRegistry::global()
        .get(operation_id)
        .filter(|operation| operation.tier == OperationTier::Intrinsic)
        .ok_or(IntrinsicRegistrationError::UnknownId { id: operation_id })
}
