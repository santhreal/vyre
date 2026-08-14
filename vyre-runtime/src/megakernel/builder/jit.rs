use super::persistent_lane_prologue;
use super::{
    assemble_lane_body, claimed_slot_bindings, execute_published_slot_body, try_assemble_lane_body,
    wrap_persistent_megakernel_program,
};
use crate::resident_work_queue::protocol::{control, slot};
use vyre_foundation::ir::{Expr, Node, Program};

/// Build the JIT Megakernel IR where payload processor logic is fused into the body stream.
#[must_use]
pub fn build_program_jit(workgroup_size_x: u32, payload_processor: &[Node]) -> Program {
    build_program_jit_slots(workgroup_size_x, workgroup_size_x.max(1), payload_processor)
}

/// Build the JIT megakernel IR for an explicit number of ring slots.
#[must_use]
pub fn build_program_jit_slots(
    workgroup_size_x: u32,
    slot_count: u32,
    payload_processor: &[Node],
) -> Program {
    wrap_persistent_megakernel_program(
        workgroup_size_x,
        slot_count,
        persistent_body_jit(workgroup_size_x, payload_processor),
    )
}

fn execute_slot_body_jit(payload_processor: &[Node]) -> Vec<Node> {
    execute_published_slot_body(claimed_slot_body_jit(payload_processor))
}

// ---- JIT variant ----

/// The JIT body that runs once per iteration per lane.
#[must_use]
pub fn persistent_body_jit(workgroup_size_x: u32, payload_processor: &[Node]) -> Vec<Node> {
    assemble_lane_body(
        persistent_lane_prologue(workgroup_size_x),
        execute_slot_body_jit(payload_processor),
        true,
    )
}

/// Fallible JIT body builder with explicit staging-allocation reporting.
pub(super) fn try_persistent_body_jit(
    workgroup_size_x: u32,
    payload_processor: &[Node],
) -> Result<Vec<Node>, String> {
    try_assemble_lane_body(
        persistent_lane_prologue(workgroup_size_x),
        execute_slot_body_jit(payload_processor),
        true,
        "megakernel JIT body",
        "reduce fused payload/body staging before building the JIT megakernel",
    )
}

fn claimed_slot_body_jit(payload_processor: &[Node]) -> Vec<Node> {
    let mut nodes = claimed_slot_bindings();

    // Wire the statically JIT-compiled rule/payload evaluation graph.
    nodes.extend(payload_processor.iter().cloned());

    nodes.push(Node::let_bind(
        "done_prev",
        Expr::atomic_add("control", Expr::u32(control::DONE_COUNT), Expr::u32(1)),
    ));
    nodes.push(Node::store(
        "ring_buffer",
        Expr::var("status_index"),
        Expr::u32(slot::DONE),
    ));
    nodes
}
