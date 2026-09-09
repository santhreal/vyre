//! Graph algorithms, CSR traversal, AST walks, dominator trees, and topological sort.

pub mod graph;
pub mod graph_compositions;
pub mod topology;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[must_use]
pub fn link_anchor() -> usize {
    vyre_libs_builder::link_anchor()
}
