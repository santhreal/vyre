//! Validation rules for atomic memory operations.
//!
//! Atomic operations are among the most error-prone primitives in GPU
//! compute: they require a read-write buffer, a `u32` element type, and
//! (for compare-exchange) a correctly supplied expected value. This
//! module checks all of those preconditions so that malformed atomics
//! are caught at IR validation time rather than producing silent data
//! races on the GPU.

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::program::BufferDecl;
use crate::ir_inner::model::op_signature::{AtomicOp, BufferAccess, DataType};
use crate::memory_model::MemoryOrdering;
use crate::validate::typecheck::{expr_type, ScopeTypes};
use crate::validate::{err, Binding, ValidationError};
use crate::validate::{ValidationLocation, ValidationPhase};
use rustc_hash::FxHashMap;

/// Validate an `Expr::Atomic` against buffer and type rules.
///
/// The validator enforces four invariants:
/// 1. The target buffer is declared with `BufferAccess::ReadWrite`.
/// 2. The buffer's element type is `U32` (atomics do not support `Bytes`
///    or other scalar types).
/// 3. The value operand is `U32`.
/// 4. For `AtomicOp::CompareExchange`, an expected operand is present and
///    is also `U32`; for all other ops, no expected operand is present.
///
/// Violations are appended to `errors` as `ValidationError` values with
/// actionable `Fix:` hints.
///
/// # Examples
///
/// `validate_atomic` is `pub(crate)`; it runs as part of
/// [`crate::validate::rule_pipeline::validate`] whenever a program contains an
/// `Expr::Atomic`. See the unit tests on that function for a runnable
/// example covering the U32 / U64 atomic surface and the invalid-op
/// rejection paths.
///
/// # Errors
///
/// Appends a `ValidationError` when any of the invariants above is
/// violated.
#[inline]
pub(crate) fn validate_atomic(
    op: AtomicOp,
    buffer: &str,
    index: &Expr,
    expected: Option<&Expr>,
    value: &Expr,
    ordering: MemoryOrdering,
    buffers: &FxHashMap<&str, &BufferDecl>,
    scope: &FxHashMap<crate::ir::Ident, Binding>,
    errors: &mut Vec<ValidationError>,
) {
    if !ordering.is_valid_for_atomic_rmw() {
        errors.push(err(
    "V042",
    ValidationPhase::Memory,
    ValidationLocation::Program,
    format!(
            "atomic `{op:?}` on buffer `{buffer}` uses invalid memory ordering `{ordering:?}`"
        ),
    "use Relaxed, Acquire, Release, AcqRel, or SeqCst for atomic read-modify-write operations.".to_string()
));
    }
    // VAL-001: the atomic index must be u32. target-text `atomicLoad`/`atomicStore`
    // and friends are indexed by `u32`; an f32 or i32 index slips past
    // validation today and then crashes the backend at dispatch time.
    if let Some(index_ty) = expr_type(index, &mut ScopeTypes::new(buffers, scope)) {
        if index_ty != DataType::U32 {
            errors.push(err(
                "V027",
                ValidationPhase::Memory,
                ValidationLocation::Program,
                format!("atomic index on buffer `{buffer}` has type `{index_ty}`, must be `u32`"),
                "cast the index to U32 before the atomic operation.".to_string(),
            ));
        }
    }
    if let Some(buf) = buffers.get(buffer) {
        // L.1.36 / audit finding #5: split the "non-writable" check so
        // Workgroup buffers get their own V025 code. The vyre atomic
        // memory model is defined for `ReadWrite` storage buffers;
        // Workgroup atomics require additional OOB and ordering
        // specification before they can be validated. V009 stays
        // reserved for `ReadOnly`/`Uniform` targets only.
        match &buf.access {
            BufferAccess::ReadWrite => {}
            BufferAccess::Workgroup => {
                errors.push(err(
    "V025",
    ValidationPhase::Memory,
    ValidationLocation::Program,
    format!(
                    "atomic `{op:?}` on workgroup buffer `{buffer}` is rejected by the current memory model"
                ),
    "use a storage ReadWrite buffer for atomics.".to_string()
));
            }
            BufferAccess::ReadOnly => {
                errors.push(err(
                    "V009",
                    ValidationPhase::Memory,
                    ValidationLocation::Program,
                    format!("atomic `{op:?}` targets read-only buffer `{buffer}`"),
                    format!(
                        "declare `{buffer}` with BufferAccess::ReadWrite before issuing `{op:?}`."
                    ),
                ));
            }
            BufferAccess::Uniform => {
                errors.push(err(
                    "V009",
                    ValidationPhase::Memory,
                    ValidationLocation::Program,
                    format!("atomic `{op:?}` targets uniform buffer `{buffer}`"),
                    format!("move `{buffer}` to BufferAccess::ReadWrite before issuing `{op:?}`."),
                ));
            }
            other => {
                errors.push(err(
                    "V009",
                    ValidationPhase::Memory,
                    ValidationLocation::Program,
                    format!(
                    "atomic `{op:?}` targets unsupported buffer access `{other:?}` on `{buffer}`"
                ),
                    "use BufferAccess::ReadWrite storage buffers for atomics.".to_string(),
                ));
            }
        }
        if buf.element == DataType::Bytes {
            errors.push(err(
                "V013",
                ValidationPhase::Memory,
                ValidationLocation::Program,
                format!(
                    "operation on buffer `{buffer}` with element type `bytes` is not supported"
                ),
                "use a typed buffer.".to_string(),
            ));
        }
        if buf.element != DataType::U32 {
            errors.push(err(
                "V014",
                ValidationPhase::Memory,
                ValidationLocation::Program,
                format!(
                    "atomic on buffer `{buffer}` with non-u32 element type `{elem}`",
                    elem = buf.element
                ),
                "atomics only support U32 elements",
            ));
        }
        if let Some(val_ty) = expr_type(value, &mut ScopeTypes::new(buffers, scope)) {
            if val_ty != DataType::U32 {
                errors.push(err(
                    "V057",
                    ValidationPhase::Memory,
                    ValidationLocation::Program,
                    format!("atomic value type `{val_ty}` does not match required `u32`"),
                    "ensure the atomic operand is U32.".to_string(),
                ));
            }
        }
        match (op, expected) {
            (AtomicOp::CompareExchange, Some(expected_expr)) => {
                if let Some(expected_ty) =
                    expr_type(expected_expr, &mut ScopeTypes::new(buffers, scope))
                {
                    if expected_ty != DataType::U32 {
                        errors.push(err(
    "V058",
    ValidationPhase::Memory,
    ValidationLocation::Program,
    format!(
                            "compare-exchange expected type `{expected_ty}` does not match required `u32`"
                        ),
    "ensure Expr::Atomic.expected is U32.".to_string()
));
                    }
                }
            }
            (AtomicOp::CompareExchange, None) => errors.push(err(
                "V059",
                ValidationPhase::Memory,
                ValidationLocation::Program,
                "compare-exchange atomic is missing expected value".to_string(),
                "set Expr::Atomic.expected for AtomicOp::CompareExchange.".to_string(),
            )),
            (_, Some(_)) => errors.push(err(
                "V060",
                ValidationPhase::Memory,
                ValidationLocation::Program,
                "non-compare-exchange atomic includes an expected value".to_string(),
                "use Expr::Atomic.expected only with AtomicOp::CompareExchange.".to_string(),
            )),
            (_, None) => {}
        }
    } else {
        errors.push(err(
            "V061",
            ValidationPhase::Memory,
            ValidationLocation::Program,
            format!("atomic on unknown buffer `{buffer}`"),
            "declare it in Program::buffers.".to_string(),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BufferDecl, DataType};

    fn buf_map(decl: &BufferDecl) -> FxHashMap<&str, &BufferDecl> {
        let mut m = FxHashMap::default();
        m.insert(decl.name(), decl);
        m
    }

    fn empty_scope() -> FxHashMap<crate::ir::Ident, Binding> {
        FxHashMap::default()
    }

    #[test]
    fn readwrite_u32_buffer_passes() {
        let decl =
            BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4);
        let buffers = buf_map(&decl);
        let mut errors = Vec::new();
        validate_atomic(
            AtomicOp::Add,
            "buf",
            &Expr::u32(0),
            None,
            &Expr::u32(1),
            MemoryOrdering::SeqCst,
            &buffers,
            &empty_scope(),
            &mut errors,
        );
        // Should have no V009, V013, or V025 errors
        let critical: Vec<_> = errors
            .iter()
            .filter(|error| matches!(error.code().as_str(), "V009" | "V013" | "V025"))
            .collect();
        assert!(
            critical.is_empty(),
            "ReadWrite U32 should have no access/type errors: {:?}",
            critical
        );
    }

    #[test]
    fn readonly_buffer_emits_v009() {
        let decl = BufferDecl::read("buf", 0, DataType::U32).with_count(4);
        let buffers = buf_map(&decl);
        let mut errors = Vec::new();
        validate_atomic(
            AtomicOp::Add,
            "buf",
            &Expr::u32(0),
            None,
            &Expr::u32(1),
            MemoryOrdering::SeqCst,
            &buffers,
            &empty_scope(),
            &mut errors,
        );
        assert!(errors.iter().any(|e| e.code().as_str() == "V009"));
    }

    #[test]
    fn workgroup_buffer_emits_v025() {
        let decl = BufferDecl::workgroup("buf", 4, DataType::U32);
        let buffers = buf_map(&decl);
        let mut errors = Vec::new();
        validate_atomic(
            AtomicOp::Add,
            "buf",
            &Expr::u32(0),
            None,
            &Expr::u32(1),
            MemoryOrdering::SeqCst,
            &buffers,
            &empty_scope(),
            &mut errors,
        );
        assert!(errors.iter().any(|e| e.code().as_str() == "V025"));
    }

    #[test]
    fn unknown_buffer_emits_error() {
        let buffers: FxHashMap<&str, &BufferDecl> = FxHashMap::default();
        let mut errors = Vec::new();
        validate_atomic(
            AtomicOp::Add,
            "missing",
            &Expr::u32(0),
            None,
            &Expr::u32(1),
            MemoryOrdering::SeqCst,
            &buffers,
            &empty_scope(),
            &mut errors,
        );
        assert!(errors
            .iter()
            .any(|e| e.message().contains("unknown buffer")));
    }

    #[test]
    fn compare_exchange_missing_expected_emits_error() {
        let decl =
            BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4);
        let buffers = buf_map(&decl);
        let mut errors = Vec::new();
        validate_atomic(
            AtomicOp::CompareExchange,
            "buf",
            &Expr::u32(0),
            None,
            &Expr::u32(1),
            MemoryOrdering::SeqCst,
            &buffers,
            &empty_scope(),
            &mut errors,
        );
        assert!(errors
            .iter()
            .any(|e| e.message().contains("missing expected")));
    }
}
