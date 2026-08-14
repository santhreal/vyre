//! Which queue-driven CSR traversal a skewed fixture gets, and the grid it launches on.
//!
//! The IFDS and CSR frontier families choose a traversal the same way: split the
//! queue into a low-degree pass and a high-degree pass when the active frontier
//! carries a hub tail worth its own dispatch, give every queued source a strided
//! lane team when the widest row makes one lane per source wasteful, and run one
//! lane per queue slot otherwise. Only the edge-kind mask and the split
//! threshold differ between the families, so those are what a caller supplies.

use vyre_foundation::ir::Program;
use vyre_primitives::graph::csr_frontier_queue::csr_queue_forward_traverse;
use vyre_primitives::graph::csr_queue_split::{
    csr_queue_split_low_dispatch_grid, csr_queue_split_low_forward_traverse,
    csr_queue_split_mixed_logical_lanes,
};
use vyre_primitives::graph::csr_queue_strided::{
    csr_queue_strided_forward_dispatch_grid, csr_queue_strided_forward_traverse,
    CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE,
};

/// Lanes one workgroup covers in the one-lane-per-slot traversal.
const QUEUE_TRAVERSE_WORKGROUP_X: u32 = 256;

/// Widest row a fixture may carry before every queued source earns a lane team.
///
/// A team is `CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE` lanes wide, so it pays
/// for itself only once a row has that many lanes' worth of edges for each lane.
pub(crate) const ROW_STRIDED_MIN_DEGREE: u32 = CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE
    .saturating_mul(CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE);

/// Whether a fixture's widest row justifies a strided lane team per source.
pub(crate) const fn should_use_row_strided(max_degree: u32) -> bool {
    max_degree >= ROW_STRIDED_MIN_DEGREE
}

/// Whether a high-degree tail is worth its own kernel.
///
/// A tail that covers the whole queue would give the low pass nothing to do, and
/// an empty tail would give the high pass nothing to do.
pub(crate) const fn should_use_split_high_degree(
    queue_capacity: u32,
    high_degree_queue_capacity: u32,
) -> bool {
    high_degree_queue_capacity > 0 && high_degree_queue_capacity < queue_capacity
}

/// Logical lanes a non-split traversal launches over `queue_capacity` slots.
pub(crate) const fn traverse_logical_lanes(queue_capacity: u32, row_strided: bool) -> u64 {
    if row_strided {
        (queue_capacity as u64).saturating_mul(CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE as u64)
    } else {
        queue_capacity as u64
    }
}

/// The single-kernel traversal of one active queue.
pub(crate) struct SingleQueueTraverse {
    pub(crate) program: Program,
    pub(crate) grid: [u32; 3],
    pub(crate) row_strided: bool,
}

/// Traverse `queue_capacity` queued sources with one kernel.
pub(crate) fn single_queue_traverse(
    max_degree: u32,
    node_count: u32,
    edge_count: u32,
    queue_capacity: u32,
    allow_mask: u32,
) -> SingleQueueTraverse {
    let row_strided = should_use_row_strided(max_degree);
    let program = if row_strided {
        csr_queue_strided_forward_traverse(
            "active_queue",
            "queue_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            node_count,
            edge_count,
            queue_capacity,
            allow_mask,
        )
    } else {
        csr_queue_forward_traverse(
            "active_queue",
            "queue_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            node_count,
            edge_count,
            queue_capacity,
            allow_mask,
        )
    };
    let grid = if row_strided {
        csr_queue_strided_forward_dispatch_grid(queue_capacity)
    } else {
        [
            queue_capacity.div_ceil(QUEUE_TRAVERSE_WORKGROUP_X).max(1),
            1,
            1,
        ]
    };

    SingleQueueTraverse {
        program,
        grid,
        row_strided,
    }
}

/// The traversal a materialized queue runs, in one or two kernels.
pub(crate) struct QueueTraversePlan {
    pub(crate) program: Program,
    pub(crate) grid: [u32; 3],
    pub(crate) row_strided: bool,
    pub(crate) split_high_degree: bool,
    pub(crate) high_program: Option<Program>,
    pub(crate) high_grid: [u32; 3],
    pub(crate) logical_lanes: u64,
}

/// Choose the traversal for a queue of `queue_capacity` slots.
///
/// `high_degree_threshold` is the degree at which the family sends a row to the
/// high-degree pass, and `high_degree_queue_capacity` is how many active sources
/// met it. The two families set the threshold differently, so it is an argument
/// rather than a constant here.
pub(crate) fn queue_traverse_plan(
    max_degree: u32,
    node_count: u32,
    edge_count: u32,
    queue_capacity: u32,
    high_degree_queue_capacity: u32,
    allow_mask: u32,
    high_degree_threshold: u32,
) -> QueueTraversePlan {
    if should_use_split_high_degree(queue_capacity, high_degree_queue_capacity) {
        let program = csr_queue_split_low_forward_traverse(
            "active_queue",
            "queue_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            "high_queue",
            "high_len",
            node_count,
            edge_count,
            queue_capacity,
            high_degree_queue_capacity,
            high_degree_threshold,
            allow_mask,
        );
        let high_program = csr_queue_strided_forward_traverse(
            "high_queue",
            "high_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            node_count,
            edge_count,
            high_degree_queue_capacity,
            allow_mask,
        );
        return QueueTraversePlan {
            program,
            grid: csr_queue_split_low_dispatch_grid(queue_capacity),
            row_strided: true,
            split_high_degree: true,
            high_program: Some(high_program),
            high_grid: csr_queue_strided_forward_dispatch_grid(high_degree_queue_capacity),
            logical_lanes: csr_queue_split_mixed_logical_lanes(
                queue_capacity,
                high_degree_queue_capacity,
            ),
        };
    }

    let single = single_queue_traverse(
        max_degree,
        node_count,
        edge_count,
        queue_capacity,
        allow_mask,
    );
    QueueTraversePlan {
        logical_lanes: traverse_logical_lanes(queue_capacity, single.row_strided),
        program: single.program,
        grid: single.grid,
        row_strided: single.row_strided,
        split_high_degree: false,
        high_program: None,
        high_grid: [1, 1, 1],
    }
}
