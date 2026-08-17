use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};

use vyre_foundation::ir::Ident;
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{Expr, Node, Program};

use super::layout::{CSR_FORWARD_OR_CHANGED_PARALLEL_WORKGROUP_SIZE, OP_ID};
use crate::graph::program_graph::{push_frontier_changed_buffers, ProgramGraphShape};

/// Parallel in-place expansion program for production fixed-point drivers.
///
/// Unlike [`crate::graph::csr_forward_or_changed::csr_forward_or_changed`], this variant gives each source node its
/// own invocation instead of walking the whole CSR from one lane. The pass is
/// monotone: each dispatch may observe only the frontier bits visible at that
/// point in the dispatch, but every newly discovered destination is ORed into
/// the same resident accumulator and sets `changed[0]`. Re-dispatch until the
/// changed flag stays zero to compute the same reachability fixpoint without a
/// full frontier readback per iteration.
#[must_use]
pub fn csr_forward_or_changed_parallel(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
) -> Program {
    let body = csr_forward_or_changed_parallel_body_prefixed(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        "",
    );
    let mut buffers = shape.read_only_buffers();
    push_frontier_changed_buffers(&mut buffers, frontier_out, changed, shape.node_count);
    Program::wrapped(
        buffers,
        CSR_FORWARD_OR_CHANGED_PARALLEL_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

/// Build the parallel expansion body used by production closure drivers and
/// large persistent-BFS programs.
#[must_use]
pub fn csr_forward_or_changed_parallel_body_prefixed(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Vec<Node> {
    csr_forward_or_changed_parallel_body_prefixed_impl(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        local_prefix,
        None,
        None,
        None,
    )
}

/// Build one parallel expansion body that snapshots source-node activity
/// before any lane writes newly reached destination bits.
#[must_use]
pub fn csr_forward_or_changed_parallel_snapshot_body_prefixed(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Vec<Node> {
    csr_forward_or_changed_parallel_body_prefixed_impl(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        local_prefix,
        Some(MemoryOrdering::GridSync),
        None,
        None,
    )
}

/// Build one snapshotting parallel expansion body and skip the expensive edge
/// scan when `active_gate` is zero. Newly discovered nodes set both
/// `changed[0]` and `changed[active_changed_index]`.
#[must_use]
pub(crate) fn csr_forward_or_changed_parallel_snapshot_body_prefixed_with_active(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    active_gate: Expr,
    active_changed_index: Expr,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Vec<Node> {
    csr_forward_or_changed_parallel_body_prefixed_impl(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        local_prefix,
        Some(MemoryOrdering::GridSync),
        Some(active_gate),
        Some((changed, active_changed_index)),
    )
}

fn csr_forward_or_changed_parallel_body_prefixed_impl(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
    snapshot_barrier: Option<MemoryOrdering>,
    active_gate: Option<Expr>,
    extra_changed: Option<(&str, Expr)>,
) -> Vec<Node> {
    crate::builder::csr::CsrTraversalComposer::forward(
        OP_ID,
        shape.node_count,
        shape.edge_count,
        edge_kind_mask,
    )
    .with_prefix(local_prefix)
    .emit_parallel_forward_or_changed_body(
        frontier_out,
        changed,
        snapshot_barrier,
        active_gate,
        extra_changed,
    )
}

/// Wrap a parallel expansion body as a child Region of `parent_op_id`.
#[must_use]
pub fn csr_forward_or_changed_parallel_child_prefixed(
    parent_op_id: &str,
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Node {
    wrap_child_region(
        OP_ID,
        Ident::from(parent_op_id),
        csr_forward_or_changed_parallel_body_prefixed(
            shape,
            frontier_out,
            changed,
            edge_kind_mask,
            local_prefix,
        ),
    )
}

/// Wrap a snapshotting parallel expansion body as a child Region.
#[must_use]
pub fn csr_forward_or_changed_parallel_snapshot_child_prefixed(
    parent_op_id: &str,
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Node {
    wrap_child_region(
        OP_ID,
        Ident::from(parent_op_id),
        csr_forward_or_changed_parallel_snapshot_body_prefixed(
            shape,
            frontier_out,
            changed,
            edge_kind_mask,
            local_prefix,
        ),
    )
}

/// Wrap an active-gated snapshotting parallel expansion body as a child Region.
#[must_use]
pub fn csr_forward_or_changed_parallel_snapshot_child_prefixed_with_active(
    parent_op_id: &str,
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    active_gate: Expr,
    active_changed_index: Expr,
    edge_kind_mask: u32,
    local_prefix: &str,
) -> Node {
    wrap_child_region(
        OP_ID,
        Ident::from(parent_op_id),
        csr_forward_or_changed_parallel_snapshot_body_prefixed_with_active(
            shape,
            frontier_out,
            changed,
            active_gate,
            active_changed_index,
            edge_kind_mask,
            local_prefix,
        ),
    )
}
