//! Shared IR fragments used by megakernel builders and schedulers.

use vyre_foundation::ir::{Expr, Node};

/// Emits a relaxed atomic load.
///
/// target-text atomics are implicitly relaxed; explicit acquire/release ordering is
/// modeled by surrounding synchronization nodes, not by the atomic expression.
pub fn atomic_load_relaxed(buffer: &str, index: Expr) -> Expr {
    Expr::atomic_add(buffer, index, Expr::u32(0))
}

/// Emits a relaxed atomic store.
///
/// The returned node binds the previous value so callers can splice the store
/// into expression-only IR regions without losing the exchange result.
pub fn atomic_store_relaxed(name: &str, buffer: &str, index: Expr, value: Expr) -> Node {
    Node::let_bind(name, Expr::atomic_exchange(buffer, index, value))
}

/// Visit `nodes` and every nested body in source preorder.
///
/// Child bodies come from [`vyre_foundation::transform::visit::child_bodies`],
/// the single owner of which node variants nest, so a walk built on this cannot
/// classify a new nesting variant as a leaf. The worklist is explicit so an
/// adversarially deep body cannot overflow the native stack.
#[cfg(test)]
pub(super) fn walk_body_preorder<'a>(nodes: &'a [Node], visit: &mut impl FnMut(&'a Node)) {
    let mut stack: Vec<&'a Node> = nodes.iter().rev().collect();
    while let Some(node) = stack.pop() {
        visit(node);
        for body in vyre_foundation::transform::visit::child_bodies(node)
            .into_iter()
            .rev()
        {
            stack.extend(body.iter().rev());
        }
    }
}

/// Every name bound by a `Node::Let`, in source preorder.
#[cfg(test)]
pub(super) fn let_names_preorder<'a>(nodes: &'a [Node]) -> Vec<&'a str> {
    let mut names: Vec<&'a str> = Vec::new();
    walk_body_preorder(nodes, &mut |node: &'a Node| {
        if let Node::Let { name, .. } = node {
            names.push(name.as_str());
        }
    });
    names
}
