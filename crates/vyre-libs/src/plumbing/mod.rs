//! Shared plumbing infrastructure: host, operand, program, and registration.

/// Host-side buffer allocation, dispatch scratch, and telemetry.
pub mod host;
/// Operand shapes, names, and tensor references.
pub mod operand;
/// Program descriptors and outputs.
pub mod program;
/// Operation registration and catalog definitions.
pub mod registration;
