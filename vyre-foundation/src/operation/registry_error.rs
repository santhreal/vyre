//! Catalog validation error and identity validation.

use super::{operation_id_namespace, IdNamespace, OperationRegistration, OperationTier};

/// Catalog validation failure.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OperationRegistryError {
    /// Two linked registrations claimed one stable identity.
    #[error("duplicate operation registration `{id}`; keep exactly one semantic owner")]
    DuplicateId {
        /// Duplicated stable operation id.
        id: &'static str,
    },
    /// A registration used the reserved zero semantic version.
    #[error("operation `{id}` uses semantic version zero; use a positive schema version")]
    InvalidVersion {
        /// Invalid operation id.
        id: &'static str,
    },
    /// A registration supplied neither a neutral program nor an explicit signature.
    #[error("operation `{id}` supplies neither a neutral program nor an explicit signature")]
    MissingSemantics {
        /// Invalid operation id.
        id: &'static str,
    },
    /// A registration id names no crate.
    #[error(
        "operation `{id}` names no minting crate; an id is `<crate>::<path>` and the crate is the one that published the identity"
    )]
    UnknownNamespace {
        /// Invalid operation id.
        id: &'static str,
    },
    /// Registration tier does not match the kind of crate that minted the id.
    #[error("operation `{id}` declares tier {declared:?}, which no {origin} identity can carry")]
    InvalidTier {
        /// Invalid operation id.
        id: &'static str,
        /// Tier supplied by the registration.
        declared: OperationTier,
        /// Whether the minting crate is inside the workspace.
        origin: &'static str,
    },
}

pub(super) fn validate_identity(
    entry: &OperationRegistration,
) -> Result<(), OperationRegistryError> {
    match operation_id_namespace(entry.id) {
        IdNamespace::Unknown => Err(OperationRegistryError::UnknownNamespace { id: entry.id }),
        IdNamespace::Workspace(_) => {
            if matches!(
                entry.tier,
                OperationTier::Intrinsic | OperationTier::Library | OperationTier::Foundation
            ) {
                Ok(())
            } else {
                Err(OperationRegistryError::InvalidTier {
                    id: entry.id,
                    declared: entry.tier,
                    origin: "workspace",
                })
            }
        }
        IdNamespace::External(_) => {
            if entry.tier == OperationTier::External {
                Ok(())
            } else {
                Err(OperationRegistryError::InvalidTier {
                    id: entry.id,
                    declared: entry.tier,
                    origin: "external",
                })
            }
        }
    }
}
