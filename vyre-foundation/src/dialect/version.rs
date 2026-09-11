//! Dialect schema version validation and error types.

use std::collections::BTreeMap;

use crate::dialect::descriptor::{DialectDescriptor, DialectOpDescriptor, DialectRegistry};

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

/// A program's declared dialect schema versions are not admissible.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SemanticVersionRejection {
    /// A declared version is outside the dialect's supported window, or a
    /// called operation postdates the version declared for its dialect.
    #[error(transparent)]
    Version(#[from] DialectVersionError),
    /// A declared dialect is not registered by any linked crate.
    #[error("Dialect `{dialect}` is declared but not registered. Fix: link the crate that registers dialect `{dialect}`.")]
    UnregisteredDialect {
        /// Declared dialect identifier.
        dialect: String,
    },
}

/// Admit the dialect schema versions a program declares, and every operation it
/// calls at those versions.
///
/// A dialect the program does not declare is compiled at its registered
/// version, which is the migration the caller performs by rebuilding against
/// the current schema. A dialect the program does declare is held to that
/// version: a stale one is rejected rather than silently read as current, and an
/// operation introduced after it is rejected rather than emitted.
///
/// # Errors
///
/// Returns [`SemanticVersionRejection::UnregisteredDialect`] for a declared
/// dialect no linked crate registers, [`DialectVersionError::StaleVersion`] or
/// [`DialectVersionError::UnsupportedVersion`] for a declared version outside
/// the dialect's window, and [`DialectVersionError::OperationVersionMismatch`]
/// for a called operation the declared version predates.
pub fn admit_program_versions(
    declared: &BTreeMap<String, u32>,
    called_op_ids: &[&str],
) -> Result<(), SemanticVersionRejection> {
    for (dialect_id, version) in declared {
        let Some(descriptor) = DialectRegistry::get(dialect_id) else {
            return Err(SemanticVersionRejection::UnregisteredDialect {
                dialect: dialect_id.clone(),
            });
        };
        validate_dialect_version(descriptor, *version)?;
    }
    for op_id in called_op_ids {
        let Some(descriptor) = DialectRegistry::find_by_op_id(op_id) else {
            continue;
        };
        let Some(version) = declared.get(descriptor.id) else {
            continue;
        };
        if let Some(op) = descriptor.find_op(op_id) {
            validate_op_version(op, *version)?;
        }
    }
    Ok(())
}

/// Admit every registered dialect's own version declarations.
///
/// A dialect whose minimum supported version exceeds its schema version admits
/// nothing, and an operation declaring a version its dialect has not reached is
/// unreachable at every declarable version. Both are declaration defects, and
/// both are rejected before any program is compiled.
///
/// # Errors
///
/// Returns [`DialectVersionError::UnsupportedVersion`] for a minimum above the
/// schema version and [`DialectVersionError::OperationVersionMismatch`] for an
/// operation above it.
pub fn admit_registered_versions() -> Result<(), SemanticVersionRejection> {
    for descriptor in DialectRegistry::global().values() {
        admit_descriptor_versions(descriptor)?;
    }
    Ok(())
}

/// Admit one dialect descriptor's own version declarations.
///
/// # Errors
///
/// As [`admit_registered_versions`], for this descriptor alone.
pub fn admit_descriptor_versions(
    descriptor: &DialectDescriptor,
) -> Result<(), SemanticVersionRejection> {
    validate_dialect_version(descriptor, descriptor.min_supported_version)?;
    for op in descriptor.operations {
        validate_op_version(op, descriptor.version)?;
    }
    Ok(())
}
