//! Induction-variable substitution over the IR tree.
//!
//! `substitute_nodes` replaces every free occurrence of a variable `var` with
//! an arbitrary `replacement` expression. It is the single canonical
//! implementation shared by the optimizer loop passes (strip-mine, unroll) and
//! the reverse-mode autodiff loop arm (which substitutes the reversed-index
//! expression into the adjoint body).
//!
//! Completeness is load-bearing: a missed `Expr`/`Node` position would silently
//! leave a stale `var` reference behind, a wrong loop tiling or a wrong
//! reversed gradient. Neither position list is restated here. Nodes come from
//! [`rewrite_walk::rewrite_node`], expressions from
//! [`crate::optimizer::rewrite::rewrite_expr`], and both are exhaustive matches
//! that fail to compile when a variant is added.
//!
//! Substitution is free when the variable does not occur: both walks report
//! "unchanged" upward, so an untouched subtree is returned as-is rather than
//! deep-cloned.

use crate::ir::{Expr, Ident, Node};
use crate::transform::rewrite_walk::{self, NodeRewrite};

/// Substitute every free occurrence of `var` with `replacement` across `nodes`.
pub(crate) fn substitute_nodes(nodes: &[Node], var: &Ident, replacement: &Expr) -> Vec<Node> {
    let mut subst = Substitution { var, replacement };
    rewrite_walk::rewrite_body(nodes, &mut subst).unwrap_or_else(|| nodes.to_vec())
}

/// Substitute every free occurrence of `var` with `replacement` in one node.
pub(crate) fn substitute_node(node: &Node, var: &Ident, replacement: &Expr) -> Node {
    let mut subst = Substitution { var, replacement };
    rewrite_walk::rewrite_node(node, &mut subst).unwrap_or_else(|| node.clone())
}

struct Substitution<'a> {
    var: &'a Ident,
    replacement: &'a Expr,
}

impl NodeRewrite for Substitution<'_> {
    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        crate::optimizer::rewrite::rewrite_operand(expr, &mut |candidate| match candidate {
            Expr::Var(name) if name == self.var => Some(self.replacement.clone()),
            _ => None,
        })
    }

    /// A `Loop` whose induction variable is `var` rebinds the name, so every
    /// occurrence inside its body is bound rather than free and the body is
    /// left verbatim. The loop bounds are still substituted: they are
    /// evaluated in the enclosing scope, where `var` is the outer binding.
    fn body(&mut self, parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
        if let Node::Loop { var, .. } = parent {
            if var == self.var {
                return None;
            }
        }
        rewrite_walk::rewrite_body(body, self)
    }
}
