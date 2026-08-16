use vyre_foundation::composition::wrap_anonymous_region;

use vyre_foundation::ir::{Expr, Node, Program};

use super::body::csr_forward_or_changed_body;
use super::layout::{CSR_FORWARD_OR_CHANGED_WORKGROUP_SIZE, OP_ID};
use crate::graph::program_graph::{push_frontier_changed_buffers, ProgramGraphShape};

/// Standalone in-place expansion program for primitive conformance.
#[must_use]
pub fn csr_forward_or_changed(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
) -> Program {
    let mut body = vec![Node::let_bind("local_changed", Expr::u32(0))];
    body.extend(csr_forward_or_changed_body(
        shape,
        frontier_out,
        "local_changed",
        edge_kind_mask,
    ));
    body.push(Node::if_then(
        Expr::eq(Expr::var("local_changed"), Expr::u32(1)),
        vec![Node::let_bind(
            "_changed",
            Expr::atomic_or(changed, Expr::u32(0), Expr::u32(1)),
        )],
    ));
    let mut buffers = shape.read_only_buffers();
    push_frontier_changed_buffers(&mut buffers, frontier_out, changed, shape.node_count);
    Program::wrapped(
        buffers,
        CSR_FORWARD_OR_CHANGED_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}
