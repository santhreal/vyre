//! Induction-variable substitution for the loop passes.
//!
//! The implementation lives in [`crate::transform::subst`] so the optimizer
//! loop passes and reverse-mode autodiff share exactly one complete `var ->
//! expr` rewrite (no duplicated, drift-prone copy). This module is a local
//! alias kept so existing `super::substitution::...` imports stay stable.

pub(super) use crate::transform::subst::{substitute_node, substitute_nodes};

use crate::ir::{Expr, Ident, Node};

/// True iff `expr` contains an `Expr::Opaque` anywhere in its tree.
///
/// An opaque expression is a backend-defined escape hatch whose memory effect
/// no analysis can name: it may read or write any buffer. The loop passes that
/// reorder memory across iterations ([`super::loop_fission`] splitting one loop
/// into two, [`super::loop_fusion`] interleaving two into one) prove safety by
/// collecting the buffers a body touches and requiring the two halves to be
/// disjoint — but a buffer access hidden inside an opaque expression is
/// invisible to that collector, so it would be silently dropped from the
/// touched set and the disjointness proof would be unsound. Both passes call
/// this to fail closed: any opaque expression in the body keeps it whole. The
/// walk is exhaustive over every `Expr` operand position (including
/// `SubgroupShuffle`'s `lane`, which the buffer collectors elide) so an opaque
/// payload can never be reordered past a dependent access it cannot see.
pub(super) fn expr_contains_opaque(expr: &Expr) -> bool {
    match expr {
        Expr::Opaque(_) => true,
        Expr::Load { index, .. } => expr_contains_opaque(index),
        Expr::BufLen { .. } => false,
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => {
            expr_contains_opaque(index)
                || expr_contains_opaque(value)
                || matches!(expected.as_deref(), Some(e) if expr_contains_opaque(e))
        }
        Expr::BinOp { left, right, .. } => {
            expr_contains_opaque(left) || expr_contains_opaque(right)
        }
        Expr::UnOp { operand, .. } => expr_contains_opaque(operand),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            expr_contains_opaque(cond)
                || expr_contains_opaque(true_val)
                || expr_contains_opaque(false_val)
        }
        Expr::Cast { value, .. } | Expr::SubgroupReduce { value, .. } => {
            expr_contains_opaque(value)
        }
        Expr::Fma { a, b, c } => {
            expr_contains_opaque(a) || expr_contains_opaque(b) || expr_contains_opaque(c)
        }
        Expr::Call { args, .. } => args.iter().any(expr_contains_opaque),
        Expr::SubgroupBallot { cond } => expr_contains_opaque(cond),
        Expr::SubgroupShuffle { value, lane } => {
            expr_contains_opaque(value) || expr_contains_opaque(lane)
        }
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize => false,
    }
}

/// True iff any node in `nodes` rebinds `var` — a `Let` or `Assign` whose
/// name equals `var`. This is the precondition guard for every loop pass that
/// reasons about the induction variable: if the body rewrites `var`, then a
/// later `Var(var)` no longer denotes the loop's `[from, to)` induction value,
/// so induction-range facts (substitution validity, redundant-guard elision,
/// strip-mine splitting, unrolling) cannot be applied to it.
///
/// A nested `Loop` that reuses the same name (`inner == var`) opens a fresh
/// binding scope for `var`; writes inside it are to that inner binding and do
/// not perturb the outer induction value, so the walk does not descend into it
/// and does not count it as a write. Every loop pass that consults this must
/// therefore treat a nested same-name loop as establishing its own context
/// (which they do). `If` / `Block` / `Region` keep the surrounding context, so
/// the walk descends through them.
pub(super) fn body_writes_loop_var(nodes: &[Node], var: &Ident) -> bool {
    nodes.iter().any(|node| match node {
        Node::Let { name, .. } | Node::Assign { name, .. } => name == var,
        Node::If {
            then, otherwise, ..
        } => body_writes_loop_var(then, var) || body_writes_loop_var(otherwise, var),
        Node::Loop {
            var: inner, body, ..
        } => inner != var && body_writes_loop_var(body, var),
        Node::Block(body) => body_writes_loop_var(body, var),
        Node::Region { body, .. } => body_writes_loop_var(body, var),
        _ => false,
    })
}

/// Like [`body_writes_loop_var`] but *more* conservative about nested loops: a
/// nested `Loop` that reuses the same name (`inner == var`) is itself counted
/// as a rebind (returns `true`) rather than being skipped.
///
/// Passes that derive a numeric *range* for the loop variable and fold against
/// it (`loop_var_range_fold`, `loop_lower_bound_normalize`) use this stricter
/// form: a nested same-name loop reintroduces the name with a different range,
/// and rather than reason about which `Var(var)` site sees which range, these
/// passes simply decline whenever the name is reintroduced at all. Passes that
/// only ask "is the outer induction value still intact after this body"
/// ([`body_writes_loop_var`]: strip-mine, unroll) can safely skip the nested
/// same-name loop because its writes are scoped to the inner binding.
pub(super) fn body_rebinds_var(body: &[Node], var: &Ident) -> bool {
    body.iter().any(|node| match node {
        Node::Let { name, .. } | Node::Assign { name, .. } => name == var,
        Node::Loop {
            var: inner, body, ..
        } => inner == var || body_rebinds_var(body, var),
        Node::If {
            then, otherwise, ..
        } => body_rebinds_var(then, var) || body_rebinds_var(otherwise, var),
        Node::Block(body) => body_rebinds_var(body, var),
        Node::Region { body, .. } => body_rebinds_var(body, var),
        _ => false,
    })
}
