//! Target compilation errors and diagnostic conversions.

use thiserror::Error;
use vyre_foundation::diagnostics::{
    CompilerLevel, Diagnostic, DiagnosticCode, DiagnosticStage, RetryClass, Severity,
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
            Self::InvalidArtifact(msg) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("MKC_TARGET_INVALID_ARTIFACT"),
                stage: DiagnosticStage::Admit,
                compiler_level: Some(CompilerLevel::Lowering),
                message: format!("target compiler rejected neutral artifact: {msg}").into(),
                location: None,
                artifact_id: None,
                target: None,
                device: None,
                suggested_fix: Some(
                    "ensure the neutral artifact satisfies canonical graph schema invariants".into(),
                ),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "invalid_artifact".to_string(),
                    detail: msg.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "invalid_artifact".to_string(),
                    detail: msg.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::Unsupported(msg) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("MKC_TARGET_UNSUPPORTED"),
                stage: DiagnosticStage::Lower,
                compiler_level: Some(CompilerLevel::Lowering),
                message: format!("target capability rejected selected plan: {msg}").into(),
                location: None,
                artifact_id: None,
                target: None,
                device: None,
                suggested_fix: Some(
                    "select a target with capabilities matching the selected plan or compile with generic schedule".into(),
                ),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unsupported_capability".to_string(),
                    detail: msg.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unsupported_capability".to_string(),
                    detail: msg.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::Emission(msg) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("MKC_TARGET_EMISSION_FAILED"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("target emission failed: {msg}").into(),
                location: None,
                artifact_id: None,
                target: None,
                device: None,
                suggested_fix: Some("inspect emitter error detail and lowered shader instructions".into()),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "emission_failure".to_string(),
                    detail: msg.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "emission_failure".to_string(),
                    detail: msg.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::ModuleBundle(msg) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("MKC_TARGET_MODULE_BUNDLE"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("target module bundle failed: {msg}").into(),
                location: None,
                artifact_id: None,
                target: None,
                device: None,
                suggested_fix: Some("ensure module images and signatures are valid and non-empty".into()),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "module_bundle".to_string(),
                    detail: msg.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "module_bundle".to_string(),
                    detail: msg.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::Payload(err) => err.diagnostic.clone(),
        }
    }
}

vyre_foundation::diagnostic_conversions!(TargetCompileError, diagnostic);
