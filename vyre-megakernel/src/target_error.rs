//! Target compilation errors and diagnostic conversions.

use thiserror::Error;
use vyre_foundation::diagnostics::{
    CauseKind, CompilerLevel, Diagnostic, DiagnosticStage, RetryClass,
};

use crate::CompileError;

/// Failure produced by a registered target compiler facet.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TargetCompileError {
    /// The neutral artifact could not be decoded into selected modules.
    #[error("target compiler rejected the neutral artifact: {0}")]
    InvalidArtifact(String),
    /// The target cannot represent one selected module or ABI contract.
    #[error("target capability rejected the selected plan: {0}")]
    Unsupported(String),
    /// Verified target lowering or emission failed.
    #[error("target emission failed: {0}")]
    Emission(String),
    /// Canonical target-module bundle encoding or decoding failed.
    #[error("target module bundle failed: {0}")]
    ModuleBundle(String),
    /// The emitted payload violated the canonical payload contract.
    #[error("target payload construction failed: {0}")]
    Payload(#[from] CompileError),
}

impl TargetCompileError {
    /// Project this error into the versioned structured diagnostic contract.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            Self::InvalidArtifact(msg) => Diagnostic::error(
                "MKC_TARGET_INVALID_ARTIFACT",
                format!("target compiler rejected neutral artifact: {msg}"),
            )
            .with_stage(DiagnosticStage::Admit)
            .with_compiler_level(CompilerLevel::Lowering)
            .with_fix("ensure the neutral artifact satisfies canonical graph schema invariants")
            .with_cause(CauseKind::InvalidInput, "invalid_artifact", msg.clone())
            .with_retry(RetryClass::RecompileSource),
            Self::Unsupported(msg) => Diagnostic::error(
                "MKC_TARGET_UNSUPPORTED",
                format!("target capability rejected selected plan: {msg}"),
            )
            .with_stage(DiagnosticStage::Lower)
            .with_compiler_level(CompilerLevel::Lowering)
            .with_fix(
                "select a target with capabilities matching the selected plan or compile with generic schedule",
            )
            .with_cause(CauseKind::UnsupportedCapability, "unsupported_capability", msg.clone())
            .with_retry(RetryClass::RecompileSource),
            Self::Emission(msg) => Diagnostic::error(
                "MKC_TARGET_EMISSION_FAILED",
                format!("target emission failed: {msg}"),
            )
            .with_stage(DiagnosticStage::Emit)
            .with_compiler_level(CompilerLevel::Emission)
            .with_fix("inspect emitter error detail and lowered shader instructions")
            .with_cause(CauseKind::Emission, "emission_failure", msg.clone())
            .with_retry(RetryClass::RecompileSource),
            Self::ModuleBundle(msg) => Diagnostic::error(
                "MKC_TARGET_MODULE_BUNDLE",
                format!("target module bundle failed: {msg}"),
            )
            .with_stage(DiagnosticStage::Emit)
            .with_compiler_level(CompilerLevel::Emission)
            .with_fix("ensure module images and signatures are valid and non-empty")
            .with_cause(CauseKind::Encoding, "module_bundle", msg.clone())
            .with_retry(RetryClass::RecompileSource),
            Self::Payload(err) => err.diagnostic.clone(),
        }
    }
}

vyre_foundation::diagnostic_conversions!(TargetCompileError, diagnostic);
