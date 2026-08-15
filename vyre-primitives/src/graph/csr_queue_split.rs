//! Mixed sparse CSR queue traversal for active sets with a small number of hubs.
//!
//! A global row-strided pass is excellent for true hub rows, but wastes lanes on
//! the many one-edge and three-edge rows that usually travel in the same active
//! queue. This primitive keeps low-degree rows in a scalar queue pass and
//! compacts only high-degree sources into a second queue for row-strided
//! traversal.

use vyre_foundation::algebra::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use crate::graph::csr_frontier_step::{
    csr_queue_step_program, CsrQueueEmit, CsrQueueInputs, CsrQueueLanes, CsrQueueRowPlan,
    CsrQueueStepSpec,
};
use crate::graph::csr_queue_strided::CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE;

/// Canonical op id for mixed low-row traversal and high-row compaction.
pub const CSR_QUEUE_SPLIT_LOW_FORWARD_OP_ID: &str =
    "vyre-primitives::graph::csr_queue_split_low_forward_traverse";

/// Workgroup shape for the low-row split pass.
pub const CSR_QUEUE_SPLIT_LOW_FORWARD_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Degree at which a queued row has enough work to amortize a 32-lane team.
pub const CSR_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD: u32 =
    CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE * CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE;

/// Dispatch grid for the one-lane-per-active-source low split pass.
#[must_use]
pub const fn csr_queue_split_low_dispatch_grid(queue_capacity: u32) -> [u32; 3] {
    crate::graph::lane_grid(
        queue_capacity,
        CSR_QUEUE_SPLIT_LOW_FORWARD_WORKGROUP_SIZE[0],
    )
}

/// Logical lanes consumed by low split plus a high row-strided follow-up pass.
#[must_use]
pub const fn csr_queue_split_mixed_logical_lanes(
    queue_capacity: u32,
    high_queue_capacity: u32,
) -> u64 {
    (queue_capacity as u64).saturating_add(
        (high_queue_capacity as u64)
            .saturating_mul(CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE as u64),
    )
}

/// Positional inputs for [`csr_queue_split_low_forward_traverse`].
#[derive(Clone, Copy, Debug)]
pub struct CsrQueueSplitLowForwardParams<'a> {
    /// Compacted queue of active source nodes.
    pub active_queue: &'a str,
    /// Single-element resident length of `active_queue`.
    pub queue_len: &'a str,
    /// CSR row pointers, `node_count + 1` entries.
    pub edge_offsets: &'a str,
    /// CSR edge destinations.
    pub edge_targets: &'a str,
    /// Per-edge kind bits tested against `allow_mask`.
    pub edge_kind_mask: &'a str,
    /// Packed bitset the reached destinations are ORed into.
    pub frontier_out: &'a str,
    /// Compact queue collecting hub sources for a later row-strided pass.
    pub high_queue: &'a str,
    /// Single-element observed hub count, which may exceed the capacity.
    pub high_len: &'a str,
    /// Node count the CSR row pointers and destination bounds are sized by.
    pub node_count: u32,
    /// Logical edge count the edge-slot bound check uses.
    pub edge_count: u32,
    /// Static capacity of `active_queue`.
    pub queue_capacity: u32,
    /// Static capacity of `high_queue`.
    pub high_queue_capacity: u32,
    /// Row degree at which a source is worth a 32-lane team.
    pub high_degree_threshold: u32,
    /// Edge kinds this traversal is allowed to follow.
    pub allow_mask: u32,
}

/// Build the low-row half of a mixed queue traversal.
///
/// Low-degree rows are expanded directly into `frontier_out`. High-degree rows
/// are appended to `high_queue` and counted in `high_len`; callers then run
/// `csr_queue_strided_forward_traverse` over that compact high queue. If
/// `high_queue` is undersized, overflow high rows are expanded by the scalar
/// lane in this pass so correctness does not depend on perfect sizing.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn csr_queue_split_low_forward_traverse(
    active_queue: &str,
    queue_len: &str,
    edge_offsets: &str,
    edge_targets: &str,
    edge_kind_mask: &str,
    frontier_out: &str,
    high_queue: &str,
    high_len: &str,
    node_count: u32,
    edge_count: u32,
    queue_capacity: u32,
    high_queue_capacity: u32,
    high_degree_threshold: u32,
    allow_mask: u32,
) -> Program {
    csr_queue_split_low_forward_traverse_with(CsrQueueSplitLowForwardParams {
        active_queue,
        queue_len,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_out,
        high_queue,
        high_len,
        node_count,
        edge_count,
        queue_capacity,
        high_queue_capacity,
        high_degree_threshold,
        allow_mask,
    })
}

/// Build the low-row half of a mixed queue traversal.
#[must_use]
pub fn csr_queue_split_low_forward_traverse_with(
    params: CsrQueueSplitLowForwardParams<'_>,
) -> Program {
    let CsrQueueSplitLowForwardParams {
        active_queue,
        queue_len,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_out,
        high_queue,
        high_len,
        node_count,
        edge_count,
        queue_capacity,
        high_queue_capacity,
        high_degree_threshold,
        allow_mask,
    } = params;
    if node_count == 0
        || queue_capacity == 0
        || high_queue_capacity == 0
        || high_degree_threshold == 0
    {
        return trap_program(CSR_QUEUE_SPLIT_LOW_FORWARD_OP_ID, Some((frontier_out, DataType::U32)), format!(
            "Fix: csr_queue_split_low_forward_traverse requires node_count > 0, non-zero queue capacities, and high_degree_threshold > 0; got node_count={node_count} queue_capacity={queue_capacity} high_queue_capacity={high_queue_capacity} high_degree_threshold={high_degree_threshold}."
        ));
    }
    csr_queue_step_program(&CsrQueueStepSpec {
        op_id: CSR_QUEUE_SPLIT_LOW_FORWARD_OP_ID,
        builder_name: "csr_queue_split_low_forward_traverse",
        prefix: "qsl",
        workgroup_size: CSR_QUEUE_SPLIT_LOW_FORWARD_WORKGROUP_SIZE,
        inputs: CsrQueueInputs {
            active_queue,
            queue_len,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
        },
        lanes: CsrQueueLanes::Scalar,
        row_plan: CsrQueueRowPlan::CompactHighDegree {
            high_queue,
            high_len,
            high_queue_capacity,
            high_degree_threshold,
        },
        emit: CsrQueueEmit::Frontier { frontier_out },
        node_count,
        edge_count,
        queue_capacity,
        allow_mask,
    })
}

/// CPU result for the low split pass.
#[cfg(any(test, feature = "cpu-parity"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CsrQueueSplitLowForwardCpuResult {
    /// Frontier bitset after low-degree rows and overflow high rows were emitted.
    pub frontier_out: Vec<u32>,
    /// Compact queue of high-degree sources that fit in the high queue capacity.
    pub high_queue: Vec<u32>,
    /// Total high-degree source count observed, including entries beyond capacity.
    pub high_len: u32,
}

/// Fallible CPU reference for the low split pass.
#[cfg(any(test, feature = "cpu-parity"))]
#[allow(clippy::too_many_arguments)]
pub fn try_csr_queue_split_low_forward_traverse_cpu(
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_out_seed: &[u32],
    node_count: u32,
    high_queue_capacity: usize,
    high_degree_threshold: u32,
    allow_mask: u32,
) -> Result<CsrQueueSplitLowForwardCpuResult, String> {
    let layout = super::csr_frontier_queue::validate_csr_queue_graph(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
    )?;
    if frontier_out_seed.len() != layout.words {
        return Err(format!(
            "Fix: csr_queue_split_low_forward_traverse requires frontier_out_seed.len() == bitset_words(node_count), got len={} but expected {} for node_count={node_count}.",
            frontier_out_seed.len(),
            layout.words
        ));
    }
    let mut high_queue_probe: Vec<u32> = Vec::new();
    crate::graph::scratch::reserve_graph_items(
        &mut high_queue_probe,
        high_queue_capacity,
        "CSR queue split CPU oracle",
        "high-degree active queue",
    )?;

    let mut frontier_out = frontier_out_seed.to_vec();
    let mut high_queue = Vec::with_capacity(high_queue_capacity);
    let mut high_len = 0_u32;
    let take = (queue_len as usize).min(active_queue.len());

    for &src in &active_queue[..take] {
        if src >= node_count {
            continue;
        }
        let start = edge_offsets[src as usize] as usize;
        let end = edge_offsets[src as usize + 1] as usize;
        if end.saturating_sub(start) as u32 >= high_degree_threshold {
            high_len = high_len.saturating_add(1);
            if high_queue.len() < high_queue_capacity {
                high_queue.push(src);
                continue;
            }
        }
        emit_scalar_row_cpu(
            start,
            end,
            edge_targets,
            edge_kind_mask,
            node_count,
            allow_mask,
            &mut frontier_out,
        );
    }

    Ok(CsrQueueSplitLowForwardCpuResult {
        frontier_out,
        high_queue,
        high_len,
    })
}

#[cfg(any(test, feature = "cpu-parity"))]
fn emit_scalar_row_cpu(
    start: usize,
    end: usize,
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    node_count: u32,
    allow_mask: u32,
    frontier_out: &mut [u32],
) {
    for edge in start..end {
        if edge_kind_mask[edge] & allow_mask == 0 {
            continue;
        }
        let dst = edge_targets[edge];
        if dst >= node_count {
            continue;
        }
        frontier_out[dst as usize / 32] |= 1_u32 << (dst % 32);
    }
}

#[cfg(test)]
mod tests;
