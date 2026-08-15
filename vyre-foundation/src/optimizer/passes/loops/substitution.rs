//! Loop-body guards the loop passes share.
//!
//! Induction-variable substitution is NOT here: [`crate::transform::subst`]
//! owns it, and the loop passes name that owner directly. This module used to
//! re-export it under a second path, which is one concept reachable by two
//! names and a reader's question ("which of the two is the real one") that has
//! no answer.

use crate::ir::{Expr, Ident, Node};
use crate::transform::visit::{any_subexpr, child_bodies};

/// True iff `expr` contains an `Expr::Opaque` anywhere in its tree.
///
/// An opaque expression is a backend-defined escape hatch whose memory effect
/// no analysis can name: it may read or write any buffer. The loop passes that
/// reorder memory across iterations ([`super::loop_fission`] splitting one loop
/// into two, [`super::loop_fusion`] interleaving two into one) prove safety by
/// collecting the buffers a body touches and requiring the two halves to be
/// disjoint, but a buffer access hidden inside an opaque expression is
/// invisible to that collector, so it would be silently dropped from the
/// touched set and the disjointness proof would be unsound. Both passes call
/// this to fail closed: any opaque expression in the body keeps it whole.
///
/// Operand positions come from `transform::visit::expr_children`, so every
/// position is covered, including `SubgroupShuffle`'s `lane`, which the buffer
/// collectors elide: an opaque payload can never be reordered past a dependent
/// access it cannot see.
pub(super) fn expr_contains_opaque(expr: &Expr) -> bool {
    any_subexpr(expr, &mut |candidate| matches!(candidate, Expr::Opaque(_)))
}

/// True iff `nodes` rebinds `var`, with `nested_same_name_rebinds` deciding
/// what a nested `Loop` that reuses the name counts as.
///
/// Child bodies come from `transform::visit::child_bodies`, the one exhaustive
/// owner. The copy this replaces ended in `_ => false`, so a `Node` variant
/// that gained a body would have read as rebinding nothing and every loop pass
/// guarded by this would have applied an induction-range fact through a scope
/// that overwrites the variable.
fn body_rebinds_var_with_nested_policy(
    nodes: &[Node],
    var: &Ident,
    nested_same_name_rebinds: bool,
) -> bool {
    nodes.iter().any(|node| {
        match node {
            Node::Let { name, .. } | Node::Assign { name, .. } if name == var => return true,
            // A nested loop reusing the name opens its own binding scope, so
            // the caller's policy decides without descending.
            Node::Loop { var: inner, .. } if inner == var => return nested_same_name_rebinds,
            _ => {}
        }
        child_bodies(node)
            .into_iter()
            .any(|body| body_rebinds_var_with_nested_policy(body, var, nested_same_name_rebinds))
    })
}

/// True iff any node in `nodes` rebinds `var`: a `Let` or `Assign` whose
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
    body_rebinds_var_with_nested_policy(nodes, var, false)
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
    body_rebinds_var_with_nested_policy(body, var, true)
}

// The guards above are covered by `tests/loop_induction_var_guards.rs`, which
// drives them through `LoopPeelPass::transform` and
// `LoopVarRangeFoldPass::transform` instead of calling the `pub(super)` helpers
// directly. Inline test modules are not allowed in new `vyre-foundation/src`
// code, and the organization contract detects them by scanning for the cfg
// attribute as text, so this note must not spell it either.
