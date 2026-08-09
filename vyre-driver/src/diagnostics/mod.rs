//! Structured, machine-readable diagnostics.
//!
//! Every fallible operation in vyre eventually surfaces a failure. The
//! legacy `Error` enum carried prose (and `Fix:` hints inside formatted
//! messages). `Diagnostic` is the structured form consumed by IDEs, language
//! servers, CI annotators, and terminal renderers.

mod legacy;
pub use legacy::from_legacy_error;

pub use vyre_foundation::diagnostics::{
    Diagnostic, DiagnosticCause, DiagnosticCode, DiagnosticStage, OpLocation, RetryClass, Severity,
};

#[cfg(test)]
mod tests;
