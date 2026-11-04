//! Typed failures for semantic IR transformation and versioned Program wire data.

use thiserror::Error;

use crate::diagnostics::{
    CauseKind, CompilerLevel, Diagnostic, DiagnosticStage, OpLocation, RetryClass,
};

/// Result for foundation-owned IR and Program wire operations.
pub type IrResult<T, E = IrError> = std::result::Result<T, E>;

/// Failure produced by foundation-owned IR transformation or Program wire boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum IrError {
    /// A recursive composition cycle was found during operation inlining.
    #[error(
        "IR inlining cycle at operation `{op_id}`. Fix: remove the recursive Expr::Call chain or split the recursive algorithm into an explicit bounded Loop."
    )]
    InlineCycle {
        /// The operation identifier that closed the cycle.
        op_id: String,
    },

    /// Operation inlining could not resolve an operation id.
    #[error(
        "IR inlining could not resolve operation `{op_id}`. Fix: register a Category A operation with this id before lowering or replace the call with inline IR."
    )]
    InlineUnknownOp {
        /// The missing operation identifier.
        op_id: String,
    },

    /// Operation inlining rejected an operation that must dispatch separately.
    #[error(
        "IR inlining rejected non-inlinable operation `{op_id}`. Fix: this op processes buffer inputs and must be dispatched as a separate kernel, not composed via Expr::Call."
    )]
    InlineNonInlinable {
        /// The operation identifier that cannot be inlined.
        op_id: String,
    },

    /// The number of arguments passed to an inlined operation did not match.
    #[error(
        "IR inlining argument count mismatch for operation `{op_id}`: expected {expected}, got {got}. Fix: pass exactly one argument for each ReadOnly or Uniform input buffer declared by the callee program."
    )]
    InlineArgCountMismatch {
        /// The operation identifier being expanded.
        op_id: String,
        /// The number of arguments the callee expects.
        expected: usize,
        /// The number of arguments the caller provided.
        got: usize,
    },

    /// The inlined operation never wrote to its declared output buffer.
    #[error(
        "IR inlining found no output write for operation `{op_id}`. Fix: Ensure the op's program() body writes to its output buffer at least once."
    )]
    InlineNoOutput {
        /// The operation identifier being expanded.
        op_id: String,
    },

    /// The inlined operation declared an invalid number of output buffers.
    #[error(
        "IR inlining found {got} declared output buffers for operation `{op_id}`. Fix: mark exactly one result buffer with BufferDecl::output(...)."
    )]
    InlineOutputCountMismatch {
        /// The operation identifier being expanded.
        op_id: String,
        /// The actual number of buffers marked as outputs.
        got: usize,
    },

    /// Structural validation rejected the Program with typed issues.
    #[error("IR validation rejected the Program: {issues:?}")]
    Validation {
        /// Foundation-owned validation issues in deterministic emission order.
        issues: Vec<crate::validate::ValidationError>,
    },

    /// Wire-format payload failed validation checks.
    #[error(
        "Wire-format validation failed: {message}. Fix: recompile the frontend program set and ensure the compiler only emits valid instructions."
    )]
    WireFormatValidation {
        /// Human-readable description of the validation failure.
        message: String,
    },

    /// target-text lowering failed before a shader could be emitted.
    #[error(
        "vyre target-text lowering: {message}. Fix: inspect the Program shape, backend capability report, and emitted shader diagnostics before retrying."
    )]
    Lowering {
        /// Human-readable description of the lowering failure.
        message: String,
    },

    /// Wire-format schema version mismatch.
    #[error(
        "Wire-format version mismatch: expected {expected}, found {found}. Fix: re-encode with a matching vyre version or upgrade this runtime."
    )]
    VersionMismatch {
        /// The schema version this runtime understands.
        expected: u32,
        /// The schema version present on the wire.
        found: u32,
    },

    /// Unknown dialect on the wire.
    #[error(
        "Unknown dialect `{name}` (requested version `{requested}`). Fix: link the dialect crate providing `{name}` into this runtime or drop the op that uses it before encoding."
    )]
    UnknownDialect {
        /// The dialect identifier on the wire (e.g. `"workgroup"`).
        name: String,
        /// The version string the encoder recorded for the dialect.
        requested: String,
    },

    /// Unknown op inside a known dialect.
    #[error(
        "Unknown op `{op}` in dialect `{dialect}`. Fix: upgrade the runtime to a version that includes this op, or drop the op before encoding."
    )]
    UnknownOp {
        /// The dialect that should contain the op.
        dialect: String,
        /// The op identifier that could not be resolved.
        op: String,
    },
}

impl IrError {
    /// Build a target-text lowering error with actionable guidance.
    #[must_use]
    pub fn lowering(message: impl Into<String>) -> Self {
        Self::Lowering {
            message: message.into(),
        }
    }

    /// Project this error into the versioned structured diagnostic contract.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        match self {
            Self::InlineCycle { op_id } => {
                Diagnostic::error("IRC001_INLINE_CYCLE", format!("IR inlining cycle at operation `{op_id}`"))
                    .with_stage(DiagnosticStage::Optimize)
                    .with_compiler_level(CompilerLevel::Optimizer)
                    .with_location(OpLocation::op(op_id.clone()))
                    .with_fix("remove the recursive Expr::Call chain or split the recursive algorithm into an explicit bounded Loop")
                    .with_cause(CauseKind::InvalidInput, "inlining", format!("cyclic expansion of `{op_id}`"))
                    .with_context_value("op_id", op_id.clone())
            }
            Self::InlineUnknownOp { op_id } => {
                Diagnostic::error(
                    "IRC002_INLINE_UNKNOWN_OP",
                    format!("IR inlining could not resolve operation `{op_id}`"),
                )
                .with_stage(DiagnosticStage::Optimize)
                .with_compiler_level(CompilerLevel::Optimizer)
                .with_location(OpLocation::op(op_id.clone()))
                .with_fix("register a Category A operation with this id before lowering or replace the call with inline IR")
                .with_cause(CauseKind::Configuration, "inlining", format!("unresolved operation `{op_id}`"))
                .with_context_value("op_id", op_id.clone())
            }
            Self::InlineNonInlinable { op_id } => {
                Diagnostic::error(
                    "IRC003_INLINE_NON_INLINABLE",
                    format!("IR inlining rejected non-inlinable operation `{op_id}`"),
                )
                .with_stage(DiagnosticStage::Optimize)
                .with_compiler_level(CompilerLevel::Optimizer)
                .with_location(OpLocation::op(op_id.clone()))
                .with_fix("this op processes buffer inputs and must be dispatched as a separate kernel, not composed via Expr::Call")
                .with_cause(CauseKind::InvalidInput, "inlining", format!("op `{op_id}` cannot be inlined"))
                .with_context_value("op_id", op_id.clone())
            }
            Self::InlineArgCountMismatch {
                op_id,
                expected,
                got,
            } => Diagnostic::error(
                "IRC004_INLINE_ARG_COUNT",
                format!(
                    "IR inlining argument count mismatch for operation `{op_id}`: expected {expected}, got {got}"
                ),
            )
            .with_stage(DiagnosticStage::Optimize)
            .with_compiler_level(CompilerLevel::Optimizer)
            .with_location(OpLocation::op(op_id.clone()))
            .with_fix("pass exactly one argument for each ReadOnly or Uniform input buffer declared by the callee program")
            .with_cause(
                CauseKind::InvalidInput,
                "inlining",
                format!("argument count mismatch: expected {expected}, got {got}"),
            )
            .with_context_value("op_id", op_id.clone())
            .with_context_value("expected", expected.to_string())
            .with_context_value("got", got.to_string()),
            Self::InlineNoOutput { op_id } => Diagnostic::error(
                "IRC005_INLINE_NO_OUTPUT",
                format!("IR inlining found no output write for operation `{op_id}`"),
            )
            .with_stage(DiagnosticStage::Optimize)
            .with_compiler_level(CompilerLevel::Optimizer)
            .with_location(OpLocation::op(op_id.clone()))
            .with_fix("ensure the op's program() body writes to its output buffer at least once")
            .with_cause(CauseKind::InvalidInput, "inlining", format!("no output write in `{op_id}`"))
            .with_context_value("op_id", op_id.clone()),
            Self::InlineOutputCountMismatch { op_id, got } => Diagnostic::error(
                "IRC006_INLINE_OUTPUT_COUNT",
                format!("IR inlining found {got} declared output buffers for operation `{op_id}`"),
            )
            .with_stage(DiagnosticStage::Optimize)
            .with_compiler_level(CompilerLevel::Optimizer)
            .with_location(OpLocation::op(op_id.clone()))
            .with_fix("mark exactly one result buffer with BufferDecl::output(...)")
            .with_cause(
                CauseKind::InvalidInput,
                "inlining",
                format!("output count mismatch: got {got}"),
            )
            .with_context_value("op_id", op_id.clone())
            .with_context_value("got", got.to_string()),
            Self::Validation { issues } => {
                if let Some(first) = issues.first() {
                    let mut diag = first.diagnostic();
                    if issues.len() > 1 {
                        for extra in &issues[1..] {
                            diag = diag.with_note(format!(
                                "additional validation issue [{}]: {}",
                                extra.code(),
                                extra.cause()
                            ));
                        }
                    }
                    diag
                } else {
                    Diagnostic::error(
                        "V000_GENERIC_VALIDATION",
                        "IR validation rejected the Program",
                    )
                    .with_stage(DiagnosticStage::Validate)
                    .with_compiler_level(CompilerLevel::FoundationIr)
                    .with_fix("inspect validation rules and program structure")
                }
            }
            Self::WireFormatValidation { message } => Diagnostic::error(
                "WIRE001_VALIDATION_FAILED",
                message.clone(),
            )
            .with_stage(DiagnosticStage::Validate)
            .with_compiler_level(CompilerLevel::Spec)
            .with_fix("recompile the frontend program set and ensure the compiler only emits valid instructions")
            .with_cause(CauseKind::Encoding, "wire_format", message.clone())
            .with_retry(RetryClass::RecompileSource),
            Self::Lowering { message } => Diagnostic::error("LOWER001_TARGET_TEXT_FAILED", message.clone())
                .with_stage(DiagnosticStage::Lower)
                .with_compiler_level(CompilerLevel::Lowering)
                .with_fix("inspect the Program shape, backend capability report, and emitted shader diagnostics before retrying")
                .with_cause(CauseKind::Lowering, "lowering", message.clone())
                .with_retry(RetryClass::RecompileSource),
            Self::VersionMismatch { expected, found } => Diagnostic::error(
                "WIRE002_VERSION_MISMATCH",
                format!("Wire-format version mismatch: expected {expected}, found {found}"),
            )
            .with_stage(DiagnosticStage::Admit)
            .with_compiler_level(CompilerLevel::Spec)
            .with_fix("re-encode with a matching vyre version or upgrade this runtime")
            .with_cause(
                CauseKind::VersionSkew,
                "wire_version",
                format!("expected wire version {expected}, got {found}"),
            )
            .with_context_value("expected", expected.to_string())
            .with_context_value("found", found.to_string()),
            Self::UnknownDialect { name, requested } => Diagnostic::error(
                "WIRE003_UNKNOWN_DIALECT",
                format!("Unknown dialect `{name}` (requested version `{requested}`)"),
            )
            .with_stage(DiagnosticStage::Admit)
            .with_compiler_level(CompilerLevel::Spec)
            .with_fix(format!(
                "link the dialect crate providing `{name}` into this runtime or drop the op that uses it before encoding"
            ))
            .with_cause(
                CauseKind::Configuration,
                "unknown_dialect",
                format!("dialect `{name}` v{requested}"),
            )
            .with_context_value("dialect", name.clone())
            .with_context_value("requested_version", requested.clone()),
            Self::UnknownOp { dialect, op } => Diagnostic::error(
                "WIRE004_UNKNOWN_OP",
                format!("Unknown op `{op}` in dialect `{dialect}`"),
            )
            .with_stage(DiagnosticStage::Admit)
            .with_compiler_level(CompilerLevel::Spec)
            .with_fix("upgrade the runtime to a version that includes this op, or drop the op before encoding")
            .with_cause(
                CauseKind::Configuration,
                "unknown_op",
                format!("op `{op}` in dialect `{dialect}`"),
            )
            .with_context_value("dialect", dialect.clone())
            .with_context_value("op", op.clone()),
        }
    }
}

crate::diagnostic_conversions!(IrError, diagnostic);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowering_helper_contains_fix_hint() {
        let err = IrError::lowering("buffer too large");
        let msg = err.to_string();
        assert!(msg.contains("buffer too large"));
        assert!(msg.contains("Fix:"));
    }

    #[test]
    fn inline_cycle_display() {
        let err = IrError::InlineCycle {
            op_id: "math::add".into(),
        };
        assert!(err.to_string().contains("math::add"));
        assert!(err.to_string().contains("cycle"));
    }

    #[test]
    fn version_mismatch_display() {
        let err = IrError::VersionMismatch {
            expected: 6,
            found: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("6"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn unknown_dialect_display() {
        let err = IrError::UnknownDialect {
            name: "my-dialect".into(),
            requested: "1.0".into(),
        };
        assert!(err.to_string().contains("my-dialect"));
    }

    #[test]
    fn error_is_clone_and_eq() {
        let a = IrError::lowering("test");
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn inline_arg_count_mismatch_display() {
        let err = IrError::InlineArgCountMismatch {
            op_id: "test::op".into(),
            expected: 3,
            got: 1,
        };
        let msg = err.to_string();
        assert!(msg.contains("expected 3"));
        assert!(msg.contains("got 1"));
    }
}
