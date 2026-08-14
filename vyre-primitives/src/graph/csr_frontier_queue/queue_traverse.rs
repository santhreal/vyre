//! Queue-driven CSR row expansion.

use vyre_foundation::ir::{DataType, Program};

use super::{CsrQueueForwardTraverseParams, CSR_QUEUE_FORWARD_OP_ID};
use crate::graph::csr_frontier_step::{
    csr_queue_step_program, CsrQueueEmit, CsrQueueInputs, CsrQueueLanes, CsrQueueRowPlan,
    CsrQueueStepSpec,
};

/// Build a GPU program that expands only queued CSR source rows.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn csr_queue_forward_traverse(
    active_queue: &str,
    queue_len: &str,
    edge_offsets: &str,
    edge_targets: &str,
    edge_kind_mask: &str,
    frontier_out: &str,
    node_count: u32,
    edge_count: u32,
    queue_capacity: u32,
    allow_mask: u32,
) -> Program {
    csr_queue_forward_traverse_with(CsrQueueForwardTraverseParams {
        active_queue,
        queue_len,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_out,
        node_count,
        edge_count,
        queue_capacity,
        allow_mask,
    })
}

/// Build a GPU program that expands only queued CSR source rows.
#[must_use]
pub fn csr_queue_forward_traverse_with(params: CsrQueueForwardTraverseParams<'_>) -> Program {
    let CsrQueueForwardTraverseParams {
        active_queue,
        queue_len,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_out,
        node_count,
        edge_count,
        queue_capacity,
        allow_mask,
    } = params;
    if node_count == 0 || queue_capacity == 0 {
        return crate::invalid_output_program(CSR_QUEUE_FORWARD_OP_ID,
        frontier_out,
        DataType::U32,
        format!(
            "Fix: csr_queue_forward_traverse requires node_count > 0 and queue_capacity > 0, got node_count={node_count} queue_capacity={queue_capacity}."
        ),);
    }
    csr_queue_step_program(&CsrQueueStepSpec {
        op_id: CSR_QUEUE_FORWARD_OP_ID,
        builder_name: "csr_queue_forward_traverse",
        prefix: "qt",
        workgroup_size: [256, 1, 1],
        inputs: CsrQueueInputs {
            active_queue,
            queue_len,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
        },
        lanes: CsrQueueLanes::Scalar,
        row_plan: CsrQueueRowPlan::ExpandAll,
        emit: CsrQueueEmit::Frontier { frontier_out },
        node_count,
        edge_count,
        queue_capacity,
        allow_mask,
    })
}

#[cfg(test)]
mod tests {
    use super::csr_queue_forward_traverse;

    #[test]
    fn csr_queue_traverse_rejects_offset_count_overflow_without_panic() {
        let result = std::panic::catch_unwind(|| {
            csr_queue_forward_traverse(
                "queue",
                "len",
                "offsets",
                "targets",
                "kinds",
                "out",
                u32::MAX,
                0,
                1,
                1,
            )
        });
        assert!(
            result.is_ok(),
            "csr_queue_forward_traverse must emit an invalid program instead of panicking"
        );

        let program = result.unwrap();
        assert_eq!(program.workgroup_size, [1, 1, 1]);
        assert_eq!(program.buffers.len(), 1);
        assert_eq!(program.buffers[0].name.as_ref(), "out");
        let entry = format!("{:?}", program.entry());
        assert!(
            entry.contains("node_count + 1 overflows u32"),
            "Fix: invalid CSR queue program must preserve the offset overflow diagnostic, got: {entry}"
        );
    }
}
