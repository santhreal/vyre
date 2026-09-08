use thiserror::Error;
use vyre_foundation::diagnostics::{
    CompilerLevel, Diagnostic, DiagnosticCode, DiagnosticStage, RetryClass, Severity,
    ToDiagnostic,
};

/// Failure produced while emitting PTX.
#[derive(Debug, Error)]
pub enum EmitError {
    /// Descriptor operation is unsupported by the PTX emitter.
    #[error("unsupported KernelOp kind in PTX emit: {0:?}")]
    UnsupportedOp(vyre_lower::KernelOp),

    /// PTX module assembly failed.
    #[error("PTX module construction failed: {0}")]
    PtxConstructionFailed(String),

    /// Binding metadata cannot be represented in PTX.
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

    /// Scalar data type is unsupported by the PTX emitter.
    #[error("unsupported data type for PTX scalar emit: {0}")]
    UnsupportedDataType(String),
}

impl EmitError {
    /// Project this error into the versioned structured diagnostic contract.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            Self::UnsupportedOp(op) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("PTX001_UNSUPPORTED_OP"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("unsupported KernelOp kind in PTX emit: {op:?}").into(),
                location: None,
                artifact_id: None,
                target: Some("ptx".to_string()),
                device: None,
                suggested_fix: Some(
                    "rewrite the unsupported kernel op into PTX-compatible instructions".into(),
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
            Self::PtxConstructionFailed(msg) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("PTX002_CONSTRUCTION_FAILED"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("PTX module construction failed: {msg}").into(),
                location: None,
                artifact_id: None,
                target: Some("ptx".to_string()),
                device: None,
                suggested_fix: Some("check kernel descriptor structure and register allocation".into()),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "ptx_construction".to_string(),
                    detail: msg.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "ptx_construction".to_string(),
                    detail: msg.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: Vec::new(),
                doc_url: None,
                notes: Vec::new(),
            },
            Self::InvalidBinding { slot, reason } => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("PTX003_INVALID_BINDING"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("binding slot {slot}: {reason}").into(),
                location: None,
                artifact_id: None,
                target: Some("ptx".to_string()),
                device: None,
                suggested_fix: Some(
                    "ensure parameter table entries correspond to valid kernel buffer bindings".into(),
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
                code: DiagnosticCode::new("PTX004_INVALID_DESCRIPTOR"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("invalid descriptor: {msg}").into(),
                location: None,
                artifact_id: None,
                target: Some("ptx".to_string()),
                device: None,
                suggested_fix: Some("validate kernel descriptor before PTX emission".into()),
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
            Self::UnsupportedDataType(dt) => Diagnostic {
                severity: Severity::Error,
                code: DiagnosticCode::new("PTX005_UNSUPPORTED_DATA_TYPE"),
                stage: DiagnosticStage::Emit,
                compiler_level: Some(CompilerLevel::Emission),
                message: format!("unsupported data type for PTX scalar emit: {dt}").into(),
                location: None,
                artifact_id: None,
                target: Some("ptx".to_string()),
                device: None,
                suggested_fix: Some("cast the value to a supported PTX scalar type (e.g. f32, f16, u32, i32, u64)".into()),
                cause: Some(vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unsupported_data_type".to_string(),
                    detail: dt.clone(),
                }),
                cause_chain: vec![vyre_foundation::diagnostics::DiagnosticCause {
                    kind: "unsupported_data_type".to_string(),
                    detail: dt.clone(),
                }],
                retry: RetryClass::RecompileSource,
                context_values: vec![("data_type".to_string(), dt.clone())],
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
