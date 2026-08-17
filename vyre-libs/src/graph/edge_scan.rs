//! The ONE canonical CSR neighbor-expansion edge-scan, backed by
//! [`crate::builder::csr::CsrTraversalComposer`].
//!
//! Preserved at `graph/` level as the public/internal seam for graph module
//! callers, while sharing implementation with the canonical builder composer.

use vyre_foundation::ir::{Expr, Node};

use crate::builder::csr::CsrTraversalComposer;
use crate::graph::program_graph::ProgramGraphShape;

/// Emit ONLY the CSR edge walk for source node `src` (no source-activity guard):
/// load the `[edge_start, edge_end)` range, and for every edge passing
/// `edge_kind_mask`, atomic-OR the target bit into `frontier_out` at
/// `frontier_index(dst_word)`, running `on_new_bit()` when a bit flips 0→1.
#[must_use]
pub(in crate::graph) fn csr_edge_expand_nodes(
    shape: ProgramGraphShape,
    frontier_out: &str,
    src: Expr,
    frontier_index: impl Fn(Expr) -> Expr,
    on_new_bit: impl Fn() -> Vec<Node>,
    edge_kind_mask: u32,
    prefix: &str,
) -> Vec<Node> {
    CsrTraversalComposer::forward(
        "edge_expand",
        shape.node_count,
        shape.edge_count,
        edge_kind_mask,
    )
    .with_prefix(prefix)
    .emit_edge_expand(frontier_out, src, frontier_index, on_new_bit)
}

/// Emit the CSR neighbor expansion for one source node `src`, reading its frontier
/// bit INLINE and expanding only when set.
#[must_use]
pub(in crate::graph) fn csr_edge_scan_nodes(
    shape: ProgramGraphShape,
    frontier_out: &str,
    src: Expr,
    frontier_index: impl Fn(Expr) -> Expr,
    on_new_bit: impl Fn() -> Vec<Node>,
    edge_kind_mask: u32,
    prefix: &str,
) -> Vec<Node> {
    CsrTraversalComposer::forward(
        "edge_scan",
        shape.node_count,
        shape.edge_count,
        edge_kind_mask,
    )
    .with_prefix(prefix)
    .emit_edge_scan(frontier_out, src, frontier_index, on_new_bit)
}
