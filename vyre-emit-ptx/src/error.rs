use thiserror::Error;
use vyre_foundation::diagnostics::{Diagnostic};

/// Target identity every diagnostic this emitter raises carries.
const TARGET: &str = "ptx";

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
            Self::UnsupportedOp(op) => Diagnostic::emission_error(
                TARGET,
                "PTX001_UNSUPPORTED_OP",
                format!("unsupported KernelOp kind in PTX emit: {op:?}"),
            )
            .with_fix("rewrite the unsupported kernel op into PTX-compatible instructions")
            .with_cause("unsupported_op", format!("{op:?}")),
            Self::PtxConstructionFailed(msg) => Diagnostic::emission_error(
                TARGET,
                "PTX002_CONSTRUCTION_FAILED",
                format!("PTX module construction failed: {msg}"),
            )
            .with_fix("check kernel descriptor structure and register allocation")
            .with_cause("ptx_construction", msg.clone()),
            Self::InvalidBinding { slot, reason } => Diagnostic::emission_error(
                TARGET,
                "PTX003_INVALID_BINDING",
                format!("binding slot {slot}: {reason}"),
            )
            .with_fix("ensure parameter table entries correspond to valid kernel buffer bindings")
            .with_cause("invalid_binding", reason.clone())
            .with_context_value("slot", slot.to_string()),
            Self::InvalidDescriptor(msg) => Diagnostic::emission_error(
                TARGET,
                "PTX004_INVALID_DESCRIPTOR",
                format!("invalid descriptor: {msg}"),
            )
            .with_fix("validate kernel descriptor before PTX emission")
            .with_cause("invalid_descriptor", msg.clone()),
            Self::UnsupportedDataType(dt) => Diagnostic::emission_error(
                TARGET,
                "PTX005_UNSUPPORTED_DATA_TYPE",
                format!("unsupported data type for PTX scalar emit: {dt}"),
            )
            .with_fix("cast the value to a supported PTX scalar type (e.g. f32, f16, u32, i32, u64)")
            .with_cause("unsupported_data_type", dt.clone())
            .with_context_value("data_type", dt.clone()),
        }
    }
}

vyre_foundation::diagnostic_conversions!(EmitError, diagnostic);
