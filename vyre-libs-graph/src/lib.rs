//! Graph algorithms, CSR traversal, AST walks, dominator trees, and topological sort.

pub mod graph;
pub mod graph_compositions;
pub mod topology;

/// Ensure all feature-selected library operation registrations are retained by the linker.
#[inline(never)]
pub fn link_anchor() {
    vyre_libs_builder::link_anchor();
}
