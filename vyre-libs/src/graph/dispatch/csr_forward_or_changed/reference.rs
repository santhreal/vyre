use vyre_primitives::graph::csr_forward_or_changed::{
    cpu_ref as csr_foc_cpu,
    cpu_ref_closure_into_with_step_hook as csr_foc_closure_into_with_step_hook,
};

/// Run one in-place forward-expand step over the CSR graph and
/// return both the new frontier and a 0/1 changed flag. The
/// primitive's contract: bits added to the frontier flip the flag;
/// no new bits → flag stays 0 → caller's fixpoint loop terminates.
///
/// Bumps the dataflow-fixpoint substrate counter so observability
/// logs every change-detection step.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn reference_forward_step_with_change_flag(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    allow_mask: u32,
) -> (Vec<u32>, u32) {
    use crate::telemetry::observability::{bump, graph_dispatch_calls};
    bump(&graph_dispatch_calls);
    csr_foc_cpu(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier,
        allow_mask,
    )
}

vyre_primitives::define_csr_closure_entry_points! {
    allocating: reference_forward_closure_via_change_flag {
        /// Iterate `forward_step_with_change_flag` until the change flag
        /// reads 0 or `max_iters` is reached. Returns the saturated
        /// frontier.
        ///
        /// This is the substrate path for "expand a Region set to its
        /// forward-reachable closure": the same fixpoint loop the
        /// optimizer used to write by hand, now driven by the primitive's
        /// own change flag.
    },
    borrowing: reference_forward_closure_via_change_flag_into {
        /// Iterate `forward_step_with_change_flag` using caller-owned scratch.
    },
    hooked: csr_foc_closure_into_with_step_hook,
    step_hook: |_| {
        crate::telemetry::observability::bump(
            &crate::telemetry::observability::graph_dispatch_calls,
        )
    },
}
