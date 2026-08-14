//! `branch_coalesce`  -  collapse nested `Node::If` whose outer body is
//! exactly one inner `If` with no `otherwise` arm into a single `If`
//! whose condition is `And(outer_cond, inner_cond)`.
//!
//! Op id: `vyre-foundation::optimizer::passes::branch_coalesce`.
//! Soundness: `Exact`  -  both `Then` arms run only when both
//! conditions are true; both `Otherwise` arms are empty so there is no
//! else-arm semantics to preserve. Cost direction: monotone-down on
//! `node_count + control_flow_count`. Preserves: every analysis.
//! Invalidates: nothing.
//!
//! ## Rule
//!
//! ```text
//! Node::If {
//!     cond: c1,
//!     then: [Node::If { cond: c2, then: body, otherwise: [] }],
//!     otherwise: [],
//! }
//! →
//! Node::If {
//!     cond: And(c1, c2),
//!     then: body,
//!     otherwise: [],
//! }
//! ```
//!
//! Comes up frequently after region inlining and CSE: domain code
//! often writes `if (in_bounds(x)) { if (matches_pattern(x)) { ... } }`
//! and the optimizer should see one combined predicate instead of two
//! nested branches. Coalescing also unblocks downstream
//! const-fold/boolean-simplification (ROADMAP A25) since the combined
//! predicate may collapse further when one of the conditions is a
//! literal.
//!
//! Does NOT fire (deliberately):
//!   - when the outer `then` has more than one child node  -  sibling
//!     statements would otherwise be hoisted into the inner branch and
//!     change observable order.
//!   - when either `otherwise` arm is non-empty  -  would lose else-arm
//!     semantics.
//!   - when the conditions involve side-effects (Load, Atomic, Call,
//!     Opaque). Even pure-looking expression evaluation may matter when
//!     the inner cond depends on a state mutation hidden inside the
//!     outer cond's evaluation; the conservative rule keeps both
//!     conditions evaluated lexically by skipping when either touches
//!     impure constructs.

use crate::ir::{Expr, Node, Program};
use crate::optimizer::passes::driver;
use crate::optimizer::passes::expr_is_observably_free_with;
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};

/// Drop the inner `Node::If` and merge its condition into the outer's
/// via logical AND.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "branch_coalesce",
    requires = [],
    invalidates = [],
    phase = "cleanup",
    boundary_class = "abi_preserving",
    cost_model_family = "fusion"
)]
pub struct BranchCoalesce;

impl BranchCoalesce {
    /// Skip the pass when no body in the program contains a nested-If
    /// pair matching the rule.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        driver::analyze_candidates(
            program,
            &[crate::ir::stats::NODE_KIND_IF],
            &mut is_coalesceable_if,
        )
    }

    /// Walk the program; replace every coalesceable nested If with a
    /// single If carrying the conjoined predicate.
    ///
    /// The driver rewrites children first, so a deeply-nested
    /// `If(c1) { If(c2) { If(c3) { .. } } }` chain coalesces bottom-up in one
    /// run.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        driver::rewrite_entry_nodes(program, &mut coalesce_if)
    }
}

/// The coalesced form of `node`, or `None` when the rule does not apply.
///
/// Legality is [`is_coalesceable_if`]'s decision, not restated here. The two
/// used to be written out separately, once as the analysis matcher and once as
/// a ladder of guards inside the rewriter that reconstructed the node it had
/// just taken apart at every non-matching step, so the pass could be scheduled
/// on a program its rewriter then declined.
fn coalesce_if(node: &Node) -> Option<Vec<Node>> {
    if !is_coalesceable_if(node) {
        return None;
    }
    let Node::If {
        cond: outer_cond,
        then,
        otherwise,
    } = node
    else {
        return None;
    };
    let Some(Node::If {
        cond: inner_cond,
        then: inner_then,
        ..
    }) = then.first()
    else {
        return None;
    };
    Some(vec![Node::If {
        cond: Expr::and(outer_cond.clone(), inner_cond.clone()),
        then: inner_then.clone(),
        otherwise: otherwise.clone(),
    }])
}

/// True iff `node` is an outer-If whose body is a single inner-If with an empty
/// otherwise and both predicates observably free.
///
/// The one owner of the rule's legality: the analysis asks it whether the pass
/// has work, and the rewrite asks it whether to fire.
fn is_coalesceable_if(node: &Node) -> bool {
    let Node::If {
        cond: outer_cond,
        then,
        otherwise,
    } = node
    else {
        return false;
    };
    if !otherwise.is_empty() || then.len() != 1 {
        return false;
    }
    let Node::If {
        cond: inner_cond,
        otherwise: inner_otherwise,
        ..
    } = &then[0]
    else {
        return false;
    };
    if !inner_otherwise.is_empty() {
        return false;
    }
    expr_is_observably_free_with(outer_cond, true, true)
        && expr_is_observably_free_with(inner_cond, true, true)
}

#[cfg(test)]
mod tests;
