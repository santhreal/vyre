use thiserror::Error;

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
