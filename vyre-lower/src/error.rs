//! Errors surfaced by the lowering pass.

use thiserror::Error;
use vyre_foundation::diagnostics::{
    CompilerLevel, Diagnostic, DiagnosticCode, DiagnosticStage, RetryClass, Severity,
    ToDiagnostic,
};

/// Failure produced while lowering Vyre IR.
#[derive(Debug, Error)]
pub enum LowerError {
    /// The input contains an IR construct unsupported by the descriptor layer.
    #[error("unsupported IR construct in lowering: {0}")]
    UnsupportedConstruct(String),

    /// The input program violates a lowering invariant.
    #[error("invalid program: {0}")]
    InvalidProgram(String),

    /// A kernel requires more result identifiers than `u32` can represent.
    #[error("operand id space exhausted (over u32::MAX values in one kernel)")]
    OperandIdOverflow,

    /// Nested structured bodies exceed the supported recursion depth.
    #[error("nested body depth exceeded reasonable limit ({0})")]
    NestingTooDeep(usize),

    /// A node references a buffer absent from the program declaration table.
    #[error("buffer not declared but referenced: {0}")]
    UndeclaredBuffer(String),
}

impl LowerError {
    /// Project this error into the versioned structured diagnostic contract.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            Self::UnsupportedConstruct(msg) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("LWR001_UNSUPPORTED_CONSTRUCT"),
                stage: DiagnosticStage::Lower,
                compiler_level: Some(CompilerLevel::Lowering),
                message: format!("unsupported IR construct in lowering: {msg}").into(),
                location: None,
                artifact_id: None,
                target: None,
                device: None,
                suggested_fix: Some(
                    "rewrite the unsupported construct using standard scalar or buffer operations before lowering"
                        .into(),
                ),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unsupported_construct".to_string(),
                    detail: msg.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unsupported_construct".to_string(),
                    detail: msg.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::InvalidProgram(msg) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("LWR002_INVALID_PROGRAM"),
                stage: DiagnosticStage::Lower,
                compiler_level: Some(CompilerLevel::Lowering),
                message: format!("invalid program in lowering: {msg}").into(),
                location: None,
                artifact_id: None,
                target: None,
                device: None,
                suggested_fix: Some(
                    "validate the Program with vyre_foundation::validate before lowering".into(),
                ),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "invalid_program".to_string(),
                    detail: msg.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "invalid_program".to_string(),
                    detail: msg.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::OperandIdOverflow => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("LWR003_OPERAND_ID_OVERFLOW"),
                stage: DiagnosticStage::Lower,
                compiler_level: Some(CompilerLevel::Lowering),
                message: "operand id space exhausted (over u32::MAX values in one kernel)".into(),
                location: None,
                artifact_id: None,
                target: None,
                device: None,
                suggested_fix: Some(
                    "split the kernel into smaller dispatches to keep operand count under u32::MAX"
                        .into(),
                ),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "overflow".to_string(),
                    detail: "operand id space exhausted".to_string(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "overflow".to_string(),
                    detail: "operand id space exhausted".to_string(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::NestingTooDeep(depth) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("LWR004_NESTING_TOO_DEEP"),
                stage: DiagnosticStage::Lower,
                compiler_level: Some(CompilerLevel::Lowering),
                message: format!("nested body depth {depth} exceeded limit").into(),
                location: None,
                artifact_id: None,
                target: None,
                device: None,
                suggested_fix: Some("flatten nested loop or block structures before lowering".into()),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "nesting_depth".to_string(),
                    detail: format!("depth {depth} exceeds limit"),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "nesting_depth".to_string(),
                    detail: format!("depth {depth} exceeds limit"),
                }],
                retry: RetryClass::RecompileSource,
                context_values: vec![("depth".to_string(), depth.to_string())],
                doc_url: None,
                notes: Vec::new(),
            },
            Self::UndeclaredBuffer(name) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("LWR005_UNDECLARED_BUFFER"),
                stage: DiagnosticStage::Lower,
                compiler_level: Some(CompilerLevel::Lowering),
                message: format!("buffer not declared but referenced: {name}").into(),
                location: None,
                artifact_id: None,
                target: None,
                device: None,
                suggested_fix: Some(
                    "declare the buffer in the program's BufferDecl table before referencing it"
                        .into(),
                ),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "undeclared_buffer".to_string(),
                    detail: format!("buffer `{name}` not found"),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "undeclared_buffer".to_string(),
                    detail: format!("buffer `{name}` not found"),
                }],
                retry: RetryClass::RecompileSource,
                context_values: vec![("buffer".to_string(), name.clone())],
                doc_url: None,
                notes: Vec::new(),
            },
        }
    }
}

impl ToDiagnostic for LowerError {
    fn to_diagnostic(&self) -> Diagnostic {
        self.diagnostic()
    }
}

impl From<&LowerError> for Diagnostic {
    fn from(error: &LowerError) -> Self {
        error.diagnostic()
    }
}

impl From<LowerError> for Diagnostic {
    fn from(error: LowerError) -> Self {
        error.diagnostic()
    }
}
