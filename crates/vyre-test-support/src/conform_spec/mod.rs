//! Witness sets, worker protocol, and replay schemas for conformance testing.

pub mod fingerprint;
pub mod protocol;
pub mod schema;
pub mod witness;

/// Compatibility alias for [`fingerprint`].
pub mod cert {
    pub use super::fingerprint;
}

pub use protocol::{
    derive_auth_key, hash_outputs, ulp_distance_f32, verify_receipts_for_certificate, CasePayload,
    CertificateRejection, DeviceLease, NumericalMismatch, NumericalPolicy, WorkerBudget,
    WorkerMode, WorkerReceipt, WorkerRequest, WorkerStatus,
};
pub use schema::{
    ConformanceCase, ConformanceResult, ReplayCapsule, ReplayMinimization, ReplayMismatch,
    SchemaVersionError, REPLAY_CAPSULE_SCHEMA_VERSION,
};
pub use witness::{U32Witness, WitnessSet};
