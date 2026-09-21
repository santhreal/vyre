//! Program shapes shared by the subgroup-collective integration tests.
//!
//! Every test here dispatches one invocation per lane, binds `idx` to the
//! invocation id, and gathers from a `values` buffer indexed by that lane. Two
//! test files built that prologue separately and the copies drifted apart in
//! buffer order while claiming to exercise the same shape, so it is stated once
//! and each caller supplies only the part that differs.
#![cfg(feature = "subgroup-ops")]

use vyre_foundation::ir::{BufferDecl, Expr, Node, Program};

/// One invocation per lane over `buffers`, with `idx` bound to the invocation
/// id before `body` runs.
///
/// The dispatch is single-axis because a subgroup is a lane window along x, so
/// a second axis would only change which invocation ids share a subgroup
/// without changing what any collective reads.
pub(crate) fn lane_program(lanes: u32, buffers: Vec<BufferDecl>, body: Vec<Node>) -> Program {
    let mut nodes = Vec::with_capacity(body.len() + 1);
    nodes.push(Node::let_bind("idx", Expr::InvocationId { axis: 0 }));
    nodes.extend(body);
    Program::wrapped(buffers, [lanes, 1, 1], nodes)
}

/// `name = subgroupShuffle(values[idx], lane)`.
///
/// The shuffled value is always this lane's own element, so `lane` alone
/// decides which lane the result comes from. A caller that wants a different
/// source value changes the buffer, not this node.
pub(crate) fn shuffle_values_by(name: &str, lane: Expr) -> Node {
    Node::let_bind(
        name,
        Expr::SubgroupShuffle {
            value: Box::new(Expr::load("values", Expr::var("idx"))),
            lane: Box::new(lane),
        },
    )
}
