//! Errors surfaced by the lowering pass.

use thiserror::Error;
use vyre_foundation::diagnostics::{
    CauseKind, CompilerLevel, Diagnostic, DiagnosticStage, RetryClass,
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
            Self::UnsupportedConstruct(msg) => Diagnostic::error(
                "LWR001_UNSUPPORTED_CONSTRUCT",
                format!("unsupported IR construct in lowering: {msg}"),
            )
            .with_stage(DiagnosticStage::Lower)
            .with_compiler_level(CompilerLevel::Lowering)
            .with_fix(
                "rewrite the unsupported construct using standard scalar or buffer operations before lowering",
            )
            .with_cause(CauseKind::UnsupportedCapability, "unsupported_construct", msg.clone())
            .with_retry(RetryClass::RecompileSource),
            Self::InvalidProgram(msg) => Diagnostic::error(
                "LWR002_INVALID_PROGRAM",
                format!("invalid program in lowering: {msg}"),
            )
            .with_stage(DiagnosticStage::Lower)
            .with_compiler_level(CompilerLevel::Lowering)
            .with_fix("validate the Program with vyre_foundation::validate before lowering")
            .with_cause(CauseKind::InvalidInput, "invalid_program", msg.clone())
            .with_retry(RetryClass::RecompileSource),
            Self::OperandIdOverflow => Diagnostic::error(
                "LWR003_OPERAND_ID_OVERFLOW",
                "operand id space exhausted (over u32::MAX values in one kernel)",
            )
            .with_stage(DiagnosticStage::Lower)
            .with_compiler_level(CompilerLevel::Lowering)
            .with_fix("split the kernel into smaller dispatches to keep operand count under u32::MAX")
            .with_cause(CauseKind::NumericOverflow, "operand_id", "operand id space exhausted")
            .with_retry(RetryClass::RecompileSource),
            Self::NestingTooDeep(depth) => Diagnostic::error(
                "LWR004_NESTING_TOO_DEEP",
                format!("nested body depth {depth} exceeded limit"),
            )
            .with_stage(DiagnosticStage::Lower)
            .with_compiler_level(CompilerLevel::Lowering)
            .with_fix("flatten nested loop or block structures before lowering")
            .with_cause(
                CauseKind::ResourceExhausted,
                "nesting_depth",
                format!("depth {depth} exceeds limit"),
            )
            .with_retry(RetryClass::RecompileSource)
            .with_context_value("depth", depth.to_string()),
            Self::UndeclaredBuffer(name) => Diagnostic::error(
                "LWR005_UNDECLARED_BUFFER",
                format!("buffer not declared but referenced: {name}"),
            )
            .with_stage(DiagnosticStage::Lower)
            .with_compiler_level(CompilerLevel::Lowering)
            .with_fix("declare the buffer in the program's BufferDecl table before referencing it")
            .with_cause(
                CauseKind::InvalidInput,
                "undeclared_buffer",
                format!("buffer `{name}` not found"),
            )
            .with_retry(RetryClass::RecompileSource)
            .with_context_value("buffer", name.clone()),
        }
    }
}

vyre_foundation::diagnostic_conversions!(LowerError, diagnostic);
