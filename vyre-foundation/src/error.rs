//! Typed failures for semantic IR transformation and versioned Program wire data.

use thiserror::Error;

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
}

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
