//! Witness sets + composition laws for vyre conformance testing.
//!
//! Canonical, deterministic witness enumeration per DataType. Consumers
//! use these to drive backend-parity testing and algebraic-law verification.

pub mod cert;
pub mod protocol;
pub mod schema;
pub mod witness;

pub use protocol::{
    derive_auth_key, hash_outputs, ulp_distance_f32, verify_receipts_for_certificate, CasePayload,
    CertificateRejection, DeviceLease, NumericalMismatch, NumericalPolicy, WorkerBudget,
    WorkerMode, WorkerReceipt, WorkerRequest, WorkerStatus,
};
pub use schema::{
    BundleCertificate, Certificate, ConformanceCase, ConformanceResult, ReplayCapsule,
    ReplayMinimization, ReplayMismatch, SchemaVersionError, CERTIFICATE_SCHEMA_VERSION,
    REPLAY_CAPSULE_SCHEMA_VERSION,
};
pub use witness::{U32Witness, WitnessSet};
