//! Dialect schema version validation and error types.

use crate::dialect::descriptor::{DialectDescriptor, DialectOpDescriptor};

/// Errors arising from dialect version incompatibility.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DialectVersionError {
    /// The requested dialect version exceeds the known dialect schema version.
    #[error("Unsupported version `{found}` for dialect `{dialect}` (current schema version `{current}`). Fix: update the dialect reference to schema version `{current}` or update the compiler.")]
    UnsupportedVersion {
        /// Dialect identifier.
        dialect: &'static str,
        /// Requested version.
        found: u32,
        /// Current supported dialect version.
        current: u32,
    },
    /// The requested dialect version is below the minimum supported dialect version.
    #[error("Stale dialect version `{found}` for dialect `{dialect}` is below minimum supported version `{min_supported}`. Fix: migrate the dialect program to schema version `{current}`.")]
    StaleVersion {
        /// Dialect identifier.
        dialect: &'static str,
        /// Requested version.
        found: u32,
        /// Minimum supported version.
        min_supported: u32,
        /// Current supported dialect version.
        current: u32,
    },
    /// The requested operation was introduced in a later version than the target version.
    #[error("Operation `{op_id}` was introduced in version `{introduced_in}` but target version is `{target_version}`. Fix: raise the dialect target version to at least `{introduced_in}`.")]
    OperationVersionMismatch {
        /// Operation identifier.
        op_id: &'static str,
        /// Version where operation was introduced.
        introduced_in: u32,
        /// Target dialect version.
        target_version: u32,
    },
}

/// Validate whether a requested dialect version is compatible with the dialect descriptor.
///
/// # Errors
///
/// Returns [`DialectVersionError::UnsupportedVersion`] when `requested > descriptor.version`.
/// Returns [`DialectVersionError::StaleVersion`] when `requested < descriptor.min_supported_version`.
pub fn validate_dialect_version(
    descriptor: &DialectDescriptor,
    requested: u32,
) -> Result<(), DialectVersionError> {
    if requested > descriptor.version {
        return Err(DialectVersionError::UnsupportedVersion {
            dialect: descriptor.id,
            found: requested,
            current: descriptor.version,
        });
    }
    if requested < descriptor.min_supported_version {
        return Err(DialectVersionError::StaleVersion {
            dialect: descriptor.id,
            found: requested,
            min_supported: descriptor.min_supported_version,
            current: descriptor.version,
        });
    }
    Ok(())
}

/// Validate whether an operation is available at the target dialect version.
///
/// # Errors
///
/// Returns [`DialectVersionError::OperationVersionMismatch`] when `op.version > target_version`.
pub fn validate_op_version(
    op: &DialectOpDescriptor,
    target_version: u32,
) -> Result<(), DialectVersionError> {
    if op.version > target_version {
        return Err(DialectVersionError::OperationVersionMismatch {
            op_id: op.id,
            introduced_in: op.version,
            target_version,
        });
    }
    Ok(())
}
