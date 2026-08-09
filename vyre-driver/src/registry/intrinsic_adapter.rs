//! Driver projection for canonical semantic operation registrations.
//!
//! Foundation owns identity, signatures, builders, effects, fixtures, and
//! tolerance policy. The driver projects signed intrinsic and runtime entries
//! into its lowering lookup without defining another semantic registry.

use thiserror::Error;
use vyre_foundation::dialect_lookup::{Category, LoweringTable, OpDef};
use vyre_foundation::operation::{
    OperationRegistration as OpEntry, OperationRegistry, OperationTier,
};

/// Failure while joining a backend lowering definition to the intrinsic catalog.
#[derive(Debug, Clone, Eq, PartialEq, Error)]
pub enum IntrinsicRegistrationError {
    /// The lowering names an id that has no intrinsic semantic owner.
    #[error(
        "unknown intrinsic id `{id}` in driver lowering registration; register the intrinsic in vyre-intrinsics first"
    )]
    UnknownId {
        /// Unrecognized stable intrinsic id.
        id: &'static str,
    },
    /// The lowering duplicated the id but changed its callable contract.
    #[error(
        "driver lowering signature for intrinsic `{id}` does not match the canonical vyre-intrinsics signature"
    )]
    SignatureMismatch {
        /// Stable intrinsic id whose signature diverged.
        id: &'static str,
    },
}

/// Project signed intrinsic and runtime operations into driver lookup records.
pub(crate) fn canonical_op_definitions() -> impl Iterator<Item = OpDef> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| {
            matches!(
                entry.tier,
                OperationTier::Intrinsic | OperationTier::Runtime
            ) && entry.signature.is_some()
        })
        .map(op_definition)
}

fn op_definition(entry: &'static OpEntry) -> OpDef {
    OpDef {
        id: entry.id,
        dialect: match entry.tier {
            OperationTier::Intrinsic => "intrinsic",
            OperationTier::Runtime => entry.category.unwrap_or("runtime"),
            _ => unreachable!("only intrinsic and runtime entries are projected"),
        },
        category: Category::Intrinsic,
        signature: entry
            .signature
            .clone()
            .expect("canonical intrinsic registration must provide a signature"),
        lowerings: LoweringTable::empty(),
        laws: &[],
        compose: entry.build,
    }
}

/// Verify that a concrete lowering definition uses the canonical intrinsic id
/// and signature, returning the semantic owner on success.
///
/// Concrete drivers call this at registration time before attaching their own
/// lowering table. Builders and fixtures never cross into the driver contract.
pub fn validate_intrinsic_lowering(
    lowering: &OpDef,
) -> Result<&'static OpEntry, IntrinsicRegistrationError> {
    let entry = OperationRegistry::global()
        .get(lowering.id)
        .filter(|entry| entry.tier == OperationTier::Intrinsic)
        .ok_or(IntrinsicRegistrationError::UnknownId { id: lowering.id })?;
    if entry.signature.as_ref() != Some(&lowering.signature) {
        return Err(IntrinsicRegistrationError::SignatureMismatch { id: lowering.id });
    }
    Ok(entry)
}
