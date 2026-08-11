//! Structured, machine-readable diagnostics.
//! Shared diagnostics are defined by `vyre-foundation` and re-exported here.

pub use vyre_foundation::diagnostics::{
    Diagnostic, DiagnosticCause, DiagnosticCode, DiagnosticStage, OpLocation, RetryClass, Severity,
};
