//! Graph and AST buffer compositions.
//!
//! Host-side packed layout lives in [`vyre_foundation::vast`]. The programs
//! here are minimal GPU-facing slices of that contract.

pub(crate) mod ast_walk;

/// Graph traversal, dominance, and dispatch-pipeline compositions.
#[cfg(feature = "graph-dispatch")]
pub mod dispatch;

pub use ast_walk::{
    ast_walk, ast_walk_postorder, ast_walk_postorder_nodes, ast_walk_preorder,
    pack_branching_fixture, pack_spine_fixture, VastWalkOrder,
};
