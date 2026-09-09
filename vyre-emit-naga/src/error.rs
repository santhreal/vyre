use thiserror::Error;
use vyre_foundation::diagnostics::{Diagnostic, RetryClass};

/// Target identity every diagnostic this emitter raises carries.
const TARGET: &str = "naga";

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
    /// This diagnostic as a downstream emitter raises it.
    ///
    /// An emitter that reaches naga through this crate reports the naga
    /// failure under its own target with a note naming the stage it happened
    /// in, so an operator reading the diagnostic sees which artifact was being
    /// produced rather than only that naga refused.
    #[must_use]
    pub fn retargeted_diagnostic(&self, target: &str, stage_note: &'static str) -> Diagnostic {
        let mut diagnostic = self.diagnostic();
        diagnostic.target = Some(target.to_string());
        diagnostic.notes.push(std::borrow::Cow::Borrowed(stage_note));
        diagnostic
    }

    /// Project this error into the versioned structured diagnostic contract.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            Self::UnsupportedOp(op) => Diagnostic::emission_error(
                TARGET,
                "NAGA001_UNSUPPORTED_OP",
                format!("unsupported KernelOp kind in naga emit: {op:?}"),
            )
            .with_fix("rewrite the unsupported kernel op into supported scalar/buffer operations")
            .with_cause("unsupported_op", format!("{op:?}")),
            Self::UnsupportedCapability(cap) => Diagnostic::emission_error(
                TARGET,
                "NAGA002_UNSUPPORTED_CAPABILITY",
                format!("unsupported emission capability `{cap}`"),
            )
            .with_fix("select a target that supports this capability or disable optional shader feature")
            .with_cause("unsupported_capability", (*cap).to_string())
            .with_context_value("capability", (*cap).to_string())
            .with_retry(RetryClass::Never),
            Self::UnsupportedWorkgroup(v) => Diagnostic::emission_error(
                TARGET,
                "NAGA003_UNSUPPORTED_WORKGROUP",
                format!("unsupported emission capability `workgroup`: {v}"),
            )
            .with_fix("reduce workgroup dimensions to fit target limits")
            .with_cause("workgroup_limit", format!("{v}")),
            Self::NagaConstructionFailed(msg) => Diagnostic::emission_error(
                TARGET,
                "NAGA004_CONSTRUCTION_FAILED",
                format!("naga module construction failed: {msg}"),
            )
            .with_fix("check kernel descriptor validity and binding layouts")
            .with_cause("naga_construction", msg.clone()),
            Self::InvalidBinding { slot, reason } => Diagnostic::emission_error(
                TARGET,
                "NAGA005_INVALID_BINDING",
                format!("binding slot {slot}: {reason}"),
            )
            .with_fix("ensure binding slots are sequentially mapped within Naga target limits")
            .with_cause("invalid_binding", reason.clone())
            .with_context_value("slot", slot.to_string()),
            Self::InvalidDescriptor(msg) => Diagnostic::emission_error(
                TARGET,
                "NAGA006_INVALID_DESCRIPTOR",
                format!("invalid descriptor: {msg}"),
            )
            .with_fix("validate kernel descriptor preconditions before emission")
            .with_cause("invalid_descriptor", msg.clone()),
        }
    }
}

vyre_foundation::diagnostic_conversions!(EmitError, diagnostic);
