//! Whether a strength-reduction rewrite may evaluate an operand more than once.
//!
//! Every rewrite in this pass that replaces one instruction with a chain of
//! cheaper ones has to write the original operand into each link of the chain.
//! `x * 3` becomes `(x << 1) + x`, which reads `x` twice. That is free for a
//! local, and it is wrong for `Expr::Load`: two loads of one address are two
//! reads of memory another invocation may have written between them, so the
//! chain can compute `(a << 1) + b` for a value that was never `a` nor `b`. It
//! is also slower, which is the opposite of the point.
//!
//! Whether re-evaluating an expression is observable is already owned by
//! [`expr_is_observably_free_with`], so this asks that and adds only the cost
//! question the optimizer has to answer on top of it.

use crate::ir::Expr;
use crate::optimizer::passes::expr_is_observably_free_with;

/// Extra evaluated nodes a rewrite may add by repeating an operand.
///
/// One duplicated leaf is the common case (`x * 3`, `x * 5`). Two lets a
/// three-term chain repeat a leaf, or a two-term chain repeat a one-level
/// expression. Beyond that the chain costs more than the multiply it replaces.
const MAX_DUPLICATED_NODES: u32 = 2;

/// Node budget for the walk. An operand larger than this is far past the
/// duplication limit, so the count stops rather than walking a deep tree.
const COUNT_CEILING: u32 = MAX_DUPLICATED_NODES + 1;

/// True when a rewrite may write `operand` into `copies` places.
///
/// `copies` counts total occurrences in the rewritten expression, so a
/// single-use rewrite passes `1` and never needs to ask.
pub(super) fn may_duplicate(operand: &Expr, copies: u32) -> bool {
    if copies <= 1 {
        return true;
    }
    // Buffer metadata is a per-dispatch constant, and a subgroup's lane index
    // and width are constant for the invocation doing the rewrite, so repeating
    // any of them observes the same value. A `Load` or an `Atomic` does not.
    if !expr_is_observably_free_with(operand, true, true) {
        return false;
    }
    let extra = copies - 1;
    evaluation_cost(operand).saturating_mul(extra) <= MAX_DUPLICATED_NODES
}

/// Number of evaluated nodes in `operand`, saturating at [`COUNT_CEILING`].
fn evaluation_cost(operand: &Expr) -> u32 {
    let mut stack = vec![operand];
    let mut count = 0;
    while let Some(expr) = stack.pop() {
        count += 1;
        if count >= COUNT_CEILING {
            return COUNT_CEILING;
        }
        match expr {
            Expr::BinOp { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            Expr::UnOp { operand, .. } => stack.push(operand),
            Expr::Cast { value, .. } => stack.push(value),
            Expr::Select {
                cond,
                true_val,
                false_val,
            } => {
                stack.push(cond);
                stack.push(true_val);
                stack.push(false_val);
            }
            Expr::Fma { a, b, c } => {
                stack.push(a);
                stack.push(b);
                stack.push(c);
            }
            // Every remaining variant is either a leaf or one this rewrite
            // already refused above as not re-executable.
            _ => {}
        }
    }
    count
}
