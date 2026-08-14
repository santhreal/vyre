//! Pre-emit `Expression::Binary` operand-type unification + the
//! `append_expr` shim every other emit path goes through. Plus the
//! tiny `emit_builtin_axis` / `emit_scalar_builtin` wrappers  -  kept
//! here because they're the simplest consumers of `append_expr`.

use naga::{BinaryOperator, Expression, Span, Statement};
use vyre_lower::KernelOp;

use super::BodyBuilder;
use crate::EmitError;

/// Whether naga requires both operands of `op` to share a scalar kind.
fn requires_matching_operand_kinds(op: BinaryOperator) -> bool {
    matches!(
        op,
        BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo
            | BinaryOperator::And
            | BinaryOperator::ExclusiveOr
            | BinaryOperator::InclusiveOr
            | BinaryOperator::ShiftLeft
            | BinaryOperator::ShiftRight
            | BinaryOperator::Equal
            | BinaryOperator::NotEqual
            | BinaryOperator::Less
            | BinaryOperator::LessEqual
            | BinaryOperator::Greater
            | BinaryOperator::GreaterEqual
    )
}

/// Whether `op` rejects a bool operand outright. A narrower set than
/// [`requires_matching_operand_kinds`]: the comparisons and the bitwise
/// operators accept two bools, the arithmetic and shift operators accept none.
fn rejects_bool_operand(op: BinaryOperator) -> bool {
    matches!(
        op,
        BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo
            | BinaryOperator::ShiftLeft
            | BinaryOperator::ShiftRight
    )
}

impl BodyBuilder<'_> {
    /// Pre-emit fix-up for `Expression::Binary` operand-type mismatches.
    ///
    /// Naga requires both operands of an arithmetic or comparison operator to
    /// share a scalar kind. Source builders, and this crate's own inserted
    /// comparisons, sometimes hand in mixed types.
    fn unify_binary_operand_types(&mut self, expr: Expression) -> Expression {
        let Expression::Binary { op, left, right } = expr else {
            return expr;
        };
        let (left, right) = self.unified_binary_operands(op, left, right);
        Expression::Binary { op, left, right }
    }

    /// The operand pair for `op` with naga's scalar-kind rules applied:
    /// value-preserving for bool to u32 (via select) and integer to integer
    /// (via `As`).
    fn unified_binary_operands(
        &mut self,
        op: BinaryOperator,
        left: naga::Handle<Expression>,
        right: naga::Handle<Expression>,
    ) -> (naga::Handle<Expression>, naga::Handle<Expression>) {
        if !requires_matching_operand_kinds(op) {
            return (left, right);
        }
        let left_kind = self.scalar_kind_of_expression(left, 0);
        let right_kind = self.scalar_kind_of_expression(right, 0);

        // Asymmetric bool-rescue: naga rejects a bool operand on every
        // arithmetic operator, so a bool side becomes u32 while the other side
        // keeps whatever kind it resolved to (or did not resolve to at all).
        if rejects_bool_operand(op) {
            let left_is_bool = left_kind == Some(naga::ScalarKind::Bool);
            let right_is_bool = right_kind == Some(naga::ScalarKind::Bool);
            if left_is_bool || right_is_bool {
                let u32_ty = self.types.u32_ty;
                let left = if left_is_bool {
                    self.coerce_value_to_type(left, u32_ty)
                } else {
                    left
                };
                let right = if right_is_bool {
                    self.coerce_value_to_type(right, u32_ty)
                } else {
                    right
                };
                return (left, right);
            }
        }

        // Shifts are ASYMMETRIC: `e1 >> e2` takes its result type from the
        // value `e1`, whose signedness selects arithmetic vs logical shift, and
        // REQUIRES the amount `e2` to be u32. The symmetric unification below
        // would coerce the amount to the value's type, emitting e.g.
        // `ShiftRight(i32, i32)`, which naga rejects with
        // InvalidBinaryOperandTypes, so signed arithmetic shifts and
        // signed-value rotates could not be emitted at all. Coerce ONLY the
        // amount, ONLY to u32, and never touch the value's type.
        if matches!(op, BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight) {
            if right_kind == Some(naga::ScalarKind::Uint) {
                return (left, right);
            }
            let u32_ty = self.types.u32_ty;
            return (left, self.coerce_value_to_type(right, u32_ty));
        }

        if let (Some(left_kind), Some(right_kind)) = (left_kind, right_kind) {
            if left_kind != right_kind {
                let target = self.canonical_type_for_scalar_kind(left_kind);
                return (left, self.coerce_value_to_type(right, target));
            }
        }
        (left, right)
    }

    pub(super) fn append_expr(&mut self, expr: Expression) -> naga::Handle<Expression> {
        let expr = self.unify_binary_operand_types(expr);
        let needs_emit = !expr.needs_pre_emit();
        let handle = self.function.expressions.append(expr, Span::UNDEFINED);
        if needs_emit {
            self.function.body.push(
                Statement::Emit(naga::Range::new_from_bounds(handle, handle)),
                Span::UNDEFINED,
            );
        }
        handle
    }

    pub(super) fn emit_builtin_axis(
        &mut self,
        op: &KernelOp,
        arg_index: u32,
    ) -> Result<(), EmitError> {
        let axis = self.inline_axis(op)?;
        let base = self.append_expr(Expression::FunctionArgument(arg_index));
        let value = self.append_expr(Expression::AccessIndex { base, index: axis });
        self.bind_result_typed(op, value, self.types.u32_ty)
    }

    pub(super) fn emit_scalar_builtin(
        &mut self,
        op: &KernelOp,
        arg_index: Option<u32>,
        name: &str,
    ) -> Result<(), EmitError> {
        let arg_index = arg_index.ok_or_else(|| {
            EmitError::InvalidDescriptor(format!(
                "{name} requires subgroup builtins, but descriptor scan did not enable them"
            ))
        })?;
        let value = self.append_expr(Expression::FunctionArgument(arg_index));
        self.bind_result_typed(op, value, self.types.u32_ty)
    }
}
