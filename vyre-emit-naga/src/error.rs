use thiserror::Error;

/// Failure produced while emitting a Naga module.
#[derive(Debug, Error)]
pub enum EmitError {
    /// Descriptor operation is unsupported by the Naga emitter.
    #[error("unsupported KernelOp kind in naga emit: {0:?}")]
    UnsupportedOp(vyre_lower::KernelOp),

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
