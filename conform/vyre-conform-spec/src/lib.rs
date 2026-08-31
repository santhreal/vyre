//! Witness sets + composition laws for vyre conformance testing.
//!
//! Canonical, deterministic witness enumeration per DataType. Consumers
//! use these to drive backend-parity testing and algebraic-law verification.

pub mod cert;
pub mod schema;
pub mod witness;

pub use schema::{
    BundleCertificate, Certificate, ConformanceCase, ConformanceResult, ReplayCapsule,
    ReplayMinimization, ReplayMismatch, SchemaVersionError, CERTIFICATE_SCHEMA_VERSION,
    REPLAY_CAPSULE_SCHEMA_VERSION,
};
pub use witness::{U32Witness, WitnessSet};
