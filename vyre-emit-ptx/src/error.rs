use thiserror::Error;

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
