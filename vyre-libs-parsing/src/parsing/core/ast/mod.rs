//! Data-parallel generic AST building blocks.

/// Shared AST node layout definitions.
pub mod node;
/// Shunting-yard AST extraction over a C11 token vocabulary.
pub mod shunting;

/// Parallel Prefix-Scan binding map.
pub mod binding;
/// Parallel basic-block metadata for structured control flow.
pub mod blocks;
