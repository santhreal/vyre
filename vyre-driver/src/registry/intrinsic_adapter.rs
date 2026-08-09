//! Driver-facing adapter for the neutral intrinsic catalog.
//!
//! Intrinsic identity, signatures, builders, semantics, and fixtures remain
//! owned by `vyre-intrinsics`. This module only projects canonical catalog
//! entries into driver `OpDef` records and verifies concrete lowering records
//! against that same source.

use thiserror::Error;
use vyre_foundation::dialect_lookup::{Category, LoweringTable, OpDef};
use vyre_intrinsics::harness::{all_entries, OpEntry};

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

/// Project every canonical intrinsic into the shared driver definition schema.
///
/// No driver crate owns or repeats an intrinsic descriptor. The returned
/// definitions clone the neutral `Signature` directly from the catalog entry
/// and retain its canonical builder as the reference-interpreter path.
pub(crate) fn intrinsic_op_definitions() -> impl Iterator<Item = OpDef> {
    all_entries().map(op_definition)
}

fn op_definition(entry: &'static OpEntry) -> OpDef {
    OpDef {
        id: entry.id,
        dialect: "intrinsic",
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
    let entry = all_entries()
        .find(|entry| entry.id == lowering.id)
        .ok_or(IntrinsicRegistrationError::UnknownId { id: lowering.id })?;
    if entry.signature.as_ref() != Some(&lowering.signature) {
        return Err(IntrinsicRegistrationError::SignatureMismatch { id: lowering.id });
    }
    Ok(entry)
}
