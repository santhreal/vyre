//! Security and capability errors (Row 119).

use thiserror::Error;

use super::label::{ConfidentialityLevel, TenantId};

/// Error conditions in security, capability validation, and tenant isolation.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SecurityError {
    /// Capability token MAC failed validation (forged or corrupted handle).
    #[error("capability handle forgery detected: cryptographic MAC mismatch")]
    ForgeryDetected,
    /// Generation counter does not match current resource generation.
    #[error("stale capability generation: expected {expected}, found {found}")]
    StaleGeneration {
        /// Expected active generation.
        expected: u64,
        /// Stale generation in token.
        found: u64,
    },
    /// Tenant in capability handle does not match executing tenant.
    #[error("tenant isolation mismatch: expected {expected}, found {found}")]
    TenantMismatch {
        /// Expected tenant id.
        expected: TenantId,
        /// Token tenant id.
        found: TenantId,
    },
    /// Device identifier does not match admitted device.
    #[error("device capability mismatch: expected device {expected}, found {found}")]
    DeviceMismatch {
        /// Expected device id.
        expected: u64,
        /// Token device id.
        found: u64,
    },
    /// Required permission not granted by capability.
    #[error("security permission denied: requires `{required}`")]
    PermissionDenied {
        /// Required permission description.
        required: String,
    },
    /// Confidentiality flow violates lattice security policy (e.g. Secret -> Public).
    #[error("confidentiality flow violation: cannot transfer data from {from:?} to {to:?}")]
    ConfidentialityLatticeViolation {
        /// Source level.
        from: ConfidentialityLevel,
        /// Destination level.
        to: ConfidentialityLevel,
    },
    /// Resource quota exceeded by tenant or compile session.
    #[error("tenant quota exceeded for resource `{resource}`: current {current} > limit {limit}")]
    QuotaExceeded {
        /// Resource name.
        resource: String,
        /// Current usage.
        current: u64,
        /// Configured limit.
        limit: u64,
    },
    /// Serialization error.
    #[error("security serialization error: {0}")]
    Serialization(String),
}
