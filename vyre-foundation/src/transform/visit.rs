//! Visitor for IR traversal.
//!
//! Optimization passes, lowering, and analysis use these utilities to walk
//! the IR tree without manually matching every variant. All traversals are
//! implemented with an explicit stack rather than recursion. This is a
//! critical design choice: it prevents stack overflows when processing deep
//! ASTs (e.g., highly nested `If` or `Block` nodes) during adversarial or
//! extreme workloads.
//!
//! The file split is by what is being visited. `node` owns the per-variant
//! decisions a `Node` traversal needs, `expr` owns the same for the value
//! namespace, and `walk` owns the traversals, which are written entirely
//! against those two and restate neither. All three are private and every item
//! is re-exported here, so `transform::visit` is the one path a caller names.

/// Per-variant `Node` decisions: nesting, scalar binding, operands, buffers.
mod node;

/// Per-variant `Expr` decisions: operands, buffer reference, sub-expression walks.
mod expr;

/// Explicit-worklist traversals over nodes and expressions.
mod walk;

#[cfg(test)]
pub(crate) mod fixtures;

#[cfg(test)]
mod tests;

pub use expr::{
    any_subexpr, expr_buffer_ref, expr_children, for_each_subexpr, ExprBufferRef, ExprChildren,
};
pub use node::{
    child_bodies, child_bodies_mut, node_bound_name, node_buffer_refs, node_operands, node_scalars,
    node_shape, BufferRefs, NameBinding, NodeScalars, NodeShape,
};
pub(crate) use node::map_bodies_cow;
pub use walk::{
    any_descendant, any_expr_in, collect_call_op_ids, for_each_expr, for_each_node,
    referenced_buffers, try_for_each_expr, try_for_each_node, walk_exprs, walk_nodes,
    walk_nodes_and_exprs, walk_nodes_mut, ExprVisitor, NodeVisitor,
};
pub(crate) use walk::{any_body, for_each_descendant};
