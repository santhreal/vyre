//! Graph / AST buffer compositions (`docs/ops-catalog.md` §1).
//!
//! Host-side packed layout lives in [`vyre_foundation::vast`]. The programs
//! here are minimal GPU-facing slices of that contract.

pub mod ast_walk;

pub use ast_walk::{
    ast_walk, ast_walk_postorder, ast_walk_postorder_nodes, ast_walk_preorder,
    pack_branching_fixture, pack_spine_fixture, VastWalkOrder,
};
