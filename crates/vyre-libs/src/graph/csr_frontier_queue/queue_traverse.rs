//! Queue-driven CSR row expansion.

use vyre_foundation::ir::Program;

use super::{
    define_csr_queue_forward_entry_point, CsrQueueForwardTraverseParams, CSR_QUEUE_FORWARD_OP_ID,
};
use crate::graph::csr_frontier_step::{csr_queue_step_program, CsrQueueLanes};

define_csr_queue_forward_entry_point! {
    /// Build a GPU program that expands only queued CSR source rows.
    csr_queue_forward_traverse -> csr_queue_forward_traverse_with
}

/// Build a GPU program that expands only queued CSR source rows.
#[must_use]
pub fn csr_queue_forward_traverse_with(params: CsrQueueForwardTraverseParams<'_>) -> Program {
    if let Some(program) =
        params.empty_shape_program(CSR_QUEUE_FORWARD_OP_ID, "csr_queue_forward_traverse")
    {
        return program;
    }
    csr_queue_step_program(&params.spec(
        CSR_QUEUE_FORWARD_OP_ID,
        "csr_queue_forward_traverse",
        "qt",
        [256, 1, 1],
        CsrQueueLanes::Scalar,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::assert_offset_overflow_traps;
    use super::csr_queue_forward_traverse;

    #[test]
    fn csr_queue_traverse_rejects_offset_count_overflow_without_panic() {
        assert_offset_overflow_traps(csr_queue_forward_traverse, "csr_queue_forward_traverse");
    }
}
