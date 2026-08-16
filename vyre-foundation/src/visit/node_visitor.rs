//! The exhaustive `Node` visitor contract and its traversal entry points.
//!
//! Every core variant is an explicit method, so a new `Node` variant makes
//! every implementor decide rather than silently skipping it.

use crate::ir_inner::model::expr::{Expr, GeneratorRef, Ident};
use crate::ir_inner::model::generated::Node;
use crate::ir_inner::model::node::NodeExtension;
use crate::visit::node_parts::child_bodies;
use crate::visit::VisitOrder;
use smallvec::SmallVec;
use std::ops::ControlFlow;

/// Visitor over [`Node`] trees.
///
/// Implementors must handle every core node variant explicitly. Like
/// [`crate::visit::ExprVisitor`], this trait is abstract-by-default so
/// adding a new node variant forces downstream code to make a conscious
/// decision.
///
/// Traversal order is explicit:
/// - [`visit_node_preorder`] visits the current node before nested nodes.
/// - [`visit_node_postorder`] visits nested nodes before the current node.
///
/// `NodeVisitor` traverses node structure only. If a visitor also needs
/// to recurse into node-owned expressions, it should pair this trait
/// with [`crate::visit::ExprVisitor`] and call the expression entry
/// points from the relevant node hooks.
pub trait NodeVisitor {
    /// Break payload returned when traversal short-circuits.
    type Break;

    /// Variable declaration.
    fn visit_let(&mut self, node: &Node, name: &Ident, value: &Expr) -> ControlFlow<Self::Break>;
    /// Variable assignment.
    fn visit_assign(&mut self, node: &Node, name: &Ident, value: &Expr)
        -> ControlFlow<Self::Break>;
    /// Buffer store.
    fn visit_store(
        &mut self,
        node: &Node,
        buffer: &Ident,
        index: &Expr,
        value: &Expr,
    ) -> ControlFlow<Self::Break>;
    /// Conditional branch.
    fn visit_if(
        &mut self,
        node: &Node,
        cond: &Expr,
        then_nodes: &[Node],
        otherwise: &[Node],
    ) -> ControlFlow<Self::Break>;
    /// Counted loop.
    fn visit_loop(
        &mut self,
        node: &Node,
        var: &Ident,
        from: &Expr,
        to: &Expr,
        body: &[Node],
    ) -> ControlFlow<Self::Break>;
    /// Indirect dispatch source.
    fn visit_indirect_dispatch(
        &mut self,
        node: &Node,
        count_buffer: &Ident,
        count_offset: u64,
    ) -> ControlFlow<Self::Break>;
    /// Async load node.
    fn visit_async_load(
        &mut self,
        node: &Node,
        source: &Ident,
        destination: &Ident,
        offset: &Expr,
        size: &Expr,
        tag: &Ident,
    ) -> ControlFlow<Self::Break>;
    /// Async store node.
    fn visit_async_store(
        &mut self,
        node: &Node,
        source: &Ident,
        destination: &Ident,
        offset: &Expr,
        size: &Expr,
        tag: &Ident,
    ) -> ControlFlow<Self::Break>;
    /// Async wait node.
    fn visit_async_wait(&mut self, node: &Node, tag: &Ident) -> ControlFlow<Self::Break>;
    /// Trap node.
    fn visit_trap(&mut self, node: &Node, address: &Expr, tag: &Ident) -> ControlFlow<Self::Break>;
    /// Resume node.
    fn visit_resume(&mut self, node: &Node, tag: &Ident) -> ControlFlow<Self::Break>;
    /// Return node.
    fn visit_return(&mut self, node: &Node) -> ControlFlow<Self::Break>;
    /// Barrier node.
    fn visit_barrier(&mut self, node: &Node) -> ControlFlow<Self::Break>;
    /// Distributed collective node.
    fn visit_collective(&mut self, node: &Node) -> ControlFlow<Self::Break> {
        let _ = node;
        ControlFlow::Continue(())
    }
    /// Tile operation node.
    fn visit_tile(&mut self, node: &Node) -> ControlFlow<Self::Break> {
        let _ = node;
        ControlFlow::Continue(())
    }
    /// Block node.
    fn visit_block(&mut self, node: &Node, body: &[Node]) -> ControlFlow<Self::Break>;
    /// Region wrapper node.
    fn visit_region(
        &mut self,
        node: &Node,
        generator: &Ident,
        source_region: &Option<GeneratorRef>,
        body: &[Node],
    ) -> ControlFlow<Self::Break>;
    /// Downstream opaque node extension.
    fn visit_opaque_node(
        &mut self,
        node: &Node,
        extension: &dyn NodeExtension,
    ) -> ControlFlow<Self::Break>;

    /// Recursively walk this node's nested node children using the requested order.
    fn walk_children_default(&mut self, node: &Node, order: VisitOrder) -> ControlFlow<Self::Break>
    where
        Self: Sized,
    {
        walk_node_children_default(self, node, order)
    }
}

/// Visit a node tree in pre-order.
pub fn visit_node<V: NodeVisitor>(visitor: &mut V, node: &Node) -> ControlFlow<V::Break> {
    visit_node_preorder(visitor, node)
}

/// Visit a node tree in pre-order without recursive stack growth.
pub fn visit_node_preorder<V: NodeVisitor>(visitor: &mut V, node: &Node) -> ControlFlow<V::Break> {
    let mut stack = SmallVec::<[&Node; 32]>::new();
    stack.push(node);
    while let Some(current) = stack.pop() {
        dispatch_node(visitor, current)?;
        for body in child_bodies(current).into_iter().rev() {
            stack.extend(body.iter().rev());
        }
    }
    ControlFlow::Continue(())
}

/// Visit a node tree in post-order without recursive stack growth.
pub fn visit_node_postorder<V: NodeVisitor>(visitor: &mut V, node: &Node) -> ControlFlow<V::Break> {
    enum Task<'a> {
        Visit(&'a Node),
        Dispatch(&'a Node),
    }
    let mut stack = SmallVec::<[Task<'_>; 32]>::new();
    stack.push(Task::Visit(node));
    while let Some(task) = stack.pop() {
        match task {
            Task::Visit(n) => {
                stack.push(Task::Dispatch(n));
                for body in child_bodies(n).into_iter().rev() {
                    stack.extend(body.iter().rev().map(Task::Visit));
                }
            }
            Task::Dispatch(n) => {
                dispatch_node(visitor, n)?;
            }
        }
    }
    ControlFlow::Continue(())
}

/// Walk only the nested node children of `node`, leaving the current node to the caller.
pub fn walk_node_children_default<V: NodeVisitor>(
    visitor: &mut V,
    node: &Node,
    order: VisitOrder,
) -> ControlFlow<V::Break> {
    for child in child_bodies(node).into_iter().flatten() {
        visit_node_with_order(visitor, child, order)?;
    }
    ControlFlow::Continue(())
}

fn visit_node_with_order<V: NodeVisitor>(
    visitor: &mut V,
    node: &Node,
    order: VisitOrder,
) -> ControlFlow<V::Break> {
    match order {
        VisitOrder::Preorder => visit_node_preorder(visitor, node),
        VisitOrder::Postorder => visit_node_postorder(visitor, node),
    }
}

pub(crate) fn dispatch_node<V: NodeVisitor>(visitor: &mut V, node: &Node) -> ControlFlow<V::Break> {
    match node {
        Node::Let { name, value } => visitor.visit_let(node, name, value),
        Node::Assign { name, value } => visitor.visit_assign(node, name, value),
        Node::Store {
            buffer,
            index,
            value,
        } => visitor.visit_store(node, buffer, index, value),
        Node::If {
            cond,
            then,
            otherwise,
        } => visitor.visit_if(node, cond, then, otherwise),
        Node::Loop {
            var,
            from,
            to,
            body,
        } => visitor.visit_loop(node, var, from, to, body),
        Node::IndirectDispatch {
            count_buffer,
            count_offset,
        } => visitor.visit_indirect_dispatch(node, count_buffer, *count_offset),
        Node::AsyncLoad {
            source,
            destination,
            offset,
            size,
            tag,
        } => visitor.visit_async_load(node, source, destination, offset, size, tag),
        Node::AsyncStore {
            source,
            destination,
            offset,
            size,
            tag,
        } => visitor.visit_async_store(node, source, destination, offset, size, tag),
        Node::AsyncWait { tag } => visitor.visit_async_wait(node, tag),
        Node::Trap { address, tag } => visitor.visit_trap(node, address, tag),
        Node::Resume { tag } => visitor.visit_resume(node, tag),
        Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. } => visitor.visit_collective(node),
        Node::Return => visitor.visit_return(node),
        Node::Barrier { .. } => visitor.visit_barrier(node),
        Node::Block(body) => visitor.visit_block(node, body),
        Node::Region {
            generator,
            source_region,
            body,
        } => visitor.visit_region(node, generator, source_region, body),
        Node::TileLoad { .. }
        | Node::TileStore { .. }
        | Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileElementwise { .. }
        | Node::TileDecl { .. } => visitor.visit_tile(node),
        Node::Opaque(extension) => visitor.visit_opaque_node(node, extension.as_ref()),
    }
}
