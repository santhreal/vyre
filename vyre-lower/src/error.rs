//! Errors surfaced by the lowering pass.

use thiserror::Error;

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
