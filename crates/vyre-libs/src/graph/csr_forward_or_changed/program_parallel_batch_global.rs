//! The batched form with one global convergence flag, and the slot-indexed
//! variants a changed-history loop uses.
//!
//! Writing history into a distinct slot per iteration is what lets a bounded
//! loop skip zeroing the flag between dispatches.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::layout::{CSR_FORWARD_OR_CHANGED_PARALLEL_WORKGROUP_SIZE, OP_ID};
use crate::graph::program_graph::{ProgramGraphShape, BINDING_PRIMITIVE_START};
/// Batched parallel expansion with one dynamically selected global convergence slot.
///
/// `changed_slot_input[0]` selects the convergence word to OR. The changed
/// buffer is sized for `changed_slots` and can be zeroed once before a
/// fixed-point sequence, allowing each iteration to write a fresh slot instead
/// of requiring a host zero-upload before every dispatch.
pub(crate) fn try_csr_forward_or_changed_parallel_batch_global_dynamic_slot(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    changed_slot_input: &str,
    edge_kind_mask: u32,
    query_count: u32,
    changed_slots: u32,
) -> Result<Program, String> {
    if changed_slots == 0 {
        return Err(
            "Fix: csr_forward_or_changed dynamic changed-slot dispatch requires at least one changed slot."
                .to_string(),
        );
    }
    csr_forward_or_changed_parallel_batch_global_indexed(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        query_count,
        Expr::var("changed_slot"),
        changed_slots,
        vec![Node::let_bind(
            "changed_slot",
            Expr::load(changed_slot_input, Expr::u32(0)),
        )],
        vec![BufferDecl::storage(
            changed_slot_input,
            BINDING_PRIMITIVE_START + 2,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(1)],
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn csr_forward_or_changed_parallel_batch_global_indexed(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    query_count: u32,
    changed_index: Expr,
    changed_slots: u32,
    prologue: Vec<Node>,
    extra_buffers: Vec<BufferDecl>,
) -> Result<Program, String> {
    crate::builder::csr::CsrTraversalComposer::forward(
        OP_ID,
        shape.node_count,
        shape.edge_count,
        edge_kind_mask,
    )
    .with_workgroup_size(CSR_FORWARD_OR_CHANGED_PARALLEL_WORKGROUP_SIZE)
    .build_parallel_batch_forward_or_changed(
        frontier_out,
        changed,
        query_count,
        changed_index,
        changed_slots,
        prologue,
        extra_buffers,
    )
}
