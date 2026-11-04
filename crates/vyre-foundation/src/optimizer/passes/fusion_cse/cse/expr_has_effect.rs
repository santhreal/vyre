//! Effect classification for common-subexpression elimination.

use crate::ir::Expr;
use crate::visit::any_subexpr;

/// Return true when evaluating `expr` can read or mutate external state.
///
/// Operand positions come from `visit::expr_children`, so only the
/// per-variant classification lives here. That matters most for the subgroup
/// ops: a subgroup op's EFFECT is its operand's effect, because `CseCtx::expr`
/// does not descend into a subgroup operand (the op interns to a unique key and
/// is never itself deduplicated), so the enclosing node relies on this answer
/// to decide whether to invalidate observed state. In the stream-compaction
/// idiom `SubgroupReduce(Add, Atomic(FetchAdd, ctr, 1))` the operand mutates
/// `ctr`; classifying the subgroup op as effect-free would let a prior
/// `Load(ctr)` survive as a stale CSE alias.
#[must_use]
#[inline]
pub fn expr_has_effect(expr: &Expr) -> bool {
    any_subexpr(expr, &mut |candidate| match candidate {
        Expr::Atomic { .. } | Expr::Call { .. } => true,
        Expr::Opaque(extension) => !extension.cse_safe(),
        _ => false,
    })
}
