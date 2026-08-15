//! Source-preorder body traversal used by megakernel builder and scheduler
//! tests to assert emitted IR shape.

use vyre_foundation::ir::Node;

/// Visit `nodes` and every nested body in source preorder.
///
/// Child bodies come from [`vyre_foundation::visit::child_bodies`],
/// the single owner of which node variants nest, so a walk built on this cannot
/// classify a new nesting variant as a leaf. The worklist is explicit so an
/// adversarially deep body cannot overflow the native stack.
pub(super) fn walk_body_preorder<'a>(nodes: &'a [Node], visit: &mut impl FnMut(&'a Node)) {
    let mut stack: Vec<&'a Node> = nodes.iter().rev().collect();
    while let Some(node) = stack.pop() {
        visit(node);
        for body in vyre_foundation::visit::child_bodies(node)
            .into_iter()
            .rev()
        {
            stack.extend(body.iter().rev());
        }
    }
}

/// Every name bound by a `Node::Let`, in source preorder.
pub(super) fn let_names_preorder<'a>(nodes: &'a [Node]) -> Vec<&'a str> {
    let mut names: Vec<&'a str> = Vec::new();
    walk_body_preorder(nodes, &mut |node: &'a Node| {
        if let Node::Let { name, .. } = node {
            names.push(name.as_str());
        }
    });
    names
}
