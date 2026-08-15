use super::LiveSet;
use crate::ir::Expr;
use crate::transform::visit::for_each_subexpr;

/// Record every variable `expr` reads, including inside every operand of every
/// nested expression.
///
/// Operand positions come from `transform::visit::expr_children`, the one
/// exhaustive owner, rather than a copy of it here. The copy this replaces had
/// to name each variant, and a variant it classified as a leaf hid a live use
/// from DCE: a `let x` read only from a subgroup operand looked dead, so DCE
/// dropped the binding and left the `Var(x)` inside the op dangling.
///
/// The walk visits every sub-expression rather than stopping at the first
/// match, because every name has to be recorded, not just the first one.
#[inline]
pub(crate) fn collect_expr_refs(expr: &Expr, refs: &mut LiveSet) {
    for_each_subexpr(expr, &mut |candidate| {
        if let Expr::Var(name) = candidate {
            refs.insert(name.clone());
        }
    });
}
