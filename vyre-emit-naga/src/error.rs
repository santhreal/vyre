use thiserror::Error;
use vyre_foundation::diagnostics::{
    CompilerLevel, Diagnostic, DiagnosticCode, DiagnosticStage, RetryClass, Severity,
    ToDiagnostic,
};

/// Failure produced while emitting a Naga module.
#[derive(Debug, Error)]
pub enum EmitError {
    /// Descriptor operation is unsupported by the Naga emitter.
    #[error("unsupported KernelOp kind in naga emit: {0:?}")]
    UnsupportedOp(vyre_lower::KernelOp),

    /// The requested target lacks a subgroup feature required by the descriptor.
    #[error("unsupported emission capability `{0}`")]
    UnsupportedCapability(&'static str),

    /// The descriptor's workgroup shape exceeds the requested target limits.
    #[error("unsupported emission capability `workgroup`: {0}")]
    UnsupportedWorkgroup(vyre_lower::WorkgroupLimitViolation),

    /// Naga module assembly failed.
    #[error("naga module construction failed: {0}")]
    NagaConstructionFailed(String),

    /// Binding metadata cannot be represented in Naga.
    #[error("binding slot {slot}: {reason}")]
    InvalidBinding {
        /// Invalid binding slot.
        slot: u32,
        /// Binding validation failure.
        reason: String,
    },

    /// Kernel descriptor violates an emitter precondition.
    #[error("invalid descriptor: {0}")]
    InvalidDescriptor(String),
}

impl EmitError {
    /// Project this error into the versioned structured diagnostic contract.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            Self::UnsupportedOp(op) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("NAGA001_UNSUPPORTED_OP"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("unsupported KernelOp kind in naga emit: {op:?}").into(),
                location: None,
                artifact_id: None,
                target: Some("naga".to_string()),
                device: None,
                suggested_fix: Some(
                    "rewrite the unsupported kernel op into supported scalar/buffer operations".into(),
                ),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unsupported_op".to_string(),
                    detail: format!("{op:?}"),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unsupported_op".to_string(),
                    detail: format!("{op:?}"),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::UnsupportedCapability(cap) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("NAGA002_UNSUPPORTED_CAPABILITY"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("unsupported emission capability `{cap}`").into(),
                location: None,
                artifact_id: None,
                target: Some("naga".to_string()),
                device: None,
                suggested_fix: Some(
                    "select a target that supports this capability or disable optional shader feature".into(),
                ),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unsupported_capability".to_string(),
                    detail: (*cap).to_string(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unsupported_capability".to_string(),
                    detail: (*cap).to_string(),
                }],
                retry: RetryClass::Never,
                context_values: vec![("capability".to_string(), (*cap).to_string())],
                doc_url: None,
                notes: Vec::new(),
            },
            Self::UnsupportedWorkgroup(v) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("NAGA003_UNSUPPORTED_WORKGROUP"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("unsupported emission capability `workgroup`: {v}").into(),
                location: None,
                artifact_id: None,
                target: Some("naga".to_string()),
                device: None,
                suggested_fix: Some("reduce workgroup dimensions to fit target limits".into()),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "workgroup_limit".to_string(),
                    detail: format!("{v}"),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "workgroup_limit".to_string(),
                    detail: format!("{v}"),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::NagaConstructionFailed(msg) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("NAGA004_CONSTRUCTION_FAILED"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("naga module construction failed: {msg}").into(),
                location: None,
                artifact_id: None,
                target: Some("naga".to_string()),
                device: None,
                suggested_fix: Some("check kernel descriptor validity and binding layouts".into()),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "naga_construction".to_string(),
                    detail: msg.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "naga_construction".to_string(),
                    detail: msg.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::InvalidBinding { slot, reason } => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("NAGA005_INVALID_BINDING"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("binding slot {slot}: {reason}").into(),
                location: None,
                artifact_id: None,
                target: Some("naga".to_string()),
                device: None,
                suggested_fix: Some(
                    "ensure binding slots are sequentially mapped within Naga target limits".into(),
                ),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "invalid_binding".to_string(),
                    detail: reason.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "invalid_binding".to_string(),
                    detail: reason.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: vec![("slot".to_string(), slot.to_string())],
                doc_url: None,
                notes: Vec::new(),
            },
            Self::InvalidDescriptor(msg) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("NAGA006_INVALID_DESCRIPTOR"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("invalid descriptor: {msg}").into(),
                location: None,
                artifact_id: None,
                target: Some("naga".to_string()),
                device: None,
                suggested_fix: Some("validate kernel descriptor preconditions before emission".into()),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "invalid_descriptor".to_string(),
                    detail: msg.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "invalid_descriptor".to_string(),
                    detail: msg.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
        }
    }
}

impl ToDiagnostic for EmitError {
    fn to_diagnostic(&self) -> Diagnostic {
        self.diagnostic()
    }
}

impl From<&EmitError> for Diagnostic {
    fn from(error: &EmitError) -> Self {
        error.diagnostic()
    }
}

impl From<EmitError> for Diagnostic {
    fn from(error: EmitError) -> Self {
        error.diagnostic()
    }
}
