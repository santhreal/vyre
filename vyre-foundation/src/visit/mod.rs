//! IR traversal: the visitor contracts, the per-variant decisions every
//! traversal is written against, and the traversals themselves.
//!
//! # Why one module owns traversal
//!
//! Vyre's IR is `#[non_exhaustive]`. Silent default visitor bodies are
//! therefore a soundness bug: a new `Expr` or `Node` variant can compile
//! while downstream analyses quietly skip it. Every per-variant decision in
//! the crate is made once here, in `node_parts` and `expr_parts`, and every
//! traversal is written against those two, so a new variant reaches every
//! walk in the workspace from one exhaustive match.
//!
//! Two visitor shapes live here and answer different questions:
//! - [`NodeVisitor`](crate::visit::NodeVisitor) and
//!   [`ExprVisitor`](crate::visit::ExprVisitor) are the exhaustive contracts, one
//!   method per variant, abstract-by-default so rustc forces a decision at
//!   every implementation site. Traversal order is explicit: `*_preorder`
//!   visits the current node before its children, `*_postorder` after, and a
//!   visitor short-circuits by returning `ControlFlow::Break`.
//! - [`NodeSink`](crate::visit::NodeSink) and [`ExprSink`](crate::visit::ExprSink)
//!   receive every node and expression a
//!   [`walk_nodes_and_exprs`](crate::visit::walk_nodes_and_exprs) pass reaches,
//!   with no per-variant knowledge.
//!
//! Every walk is an explicit worklist rather than recursion, so an
//! adversarially deep program costs heap instead of native stack.

/// Canonical bound-name (`Let` / `Loop` variable) collector shared by the
/// scope-aware passes (`region_inline`, `tail_duplication`,
/// `read_only_load_hoist`). Internal: all helpers are `pub(crate)`, so the
/// module stays off the public API surface.
pub(crate) mod bound_names;
/// Per-variant `Expr` decisions: operands, buffer reference, sub-expression walks.
pub(crate) mod expr_parts;
/// The exhaustive `Expr` visitor contract and its traversal entry points.
pub(crate) mod expr_visitor;
/// Per-variant `Node` decisions: nesting, scalar binding, operands, buffers.
pub(crate) mod node_parts;
/// The exhaustive `Node` visitor contract and its traversal entry points.
pub(crate) mod node_visitor;
/// Explicit-worklist traversals over nodes and expressions.
pub(crate) mod walk;

/// Owning child-recursive `Node` map shared by the cleanup catalog
/// (`empty_block_collapse`, `region_promote_singleton_block`,
/// `if_constant_branch_eliminate`, `noop_assign_eliminate`,
/// `loop_trip_zero_eliminate`, `loops::loop_redundant_bound_check_elide`).
/// Descendant search lives in [`any_descendant`].
pub mod node_map;

/// The contract an evaluator implements to execute IR against an environment.
pub(crate) mod evaluatable;
/// The contract a backend implements to lower IR into its own representation.
pub(crate) mod lowerable;

#[cfg(test)]
pub(crate) mod fixtures;

/// Recursive traversal order for visitor entry points and default child walking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisitOrder {
    /// Visit the current node before its children.
    Preorder,
    /// Visit the current node after its children.
    Postorder,
}

pub use evaluatable::Evaluatable;
pub use expr_parts::{
    any_subexpr, expr_buffer_ref, expr_children, for_each_subexpr, ExprBufferRef, ExprChildren,
};
pub use expr_visitor::{
    visit_expr, visit_expr_buffer_accesses, visit_postorder, visit_preorder,
    walk_expr_children_default, ExprBufferAccess, ExprVisitor,
};
pub use lowerable::Lowerable;
pub(crate) use node_parts::map_bodies_cow;
pub use node_parts::{
    child_bodies, child_bodies_mut, node_bound_name, node_buffer_refs, node_operands, node_scalars,
    node_shape, node_tag, node_variadic_operands, BufferRefs, NameBinding, NodeScalars, NodeShape,
};
pub use node_visitor::{
    visit_node, visit_node_postorder, visit_node_preorder, walk_node_children_default, NodeVisitor,
};
pub(crate) use walk::{any_body, for_each_descendant};
pub use walk::{
    any_descendant, any_expr_in, collect_call_op_ids, for_each_expr, for_each_node,
    referenced_buffers, try_for_each_expr, try_for_each_node, walk_exprs, walk_nodes,
    walk_nodes_and_exprs, walk_nodes_mut, ExprSink, NodeSink,
};

#[cfg(test)]
#[path = "../../tests/internal/visit/mod.rs"]
mod tests;
