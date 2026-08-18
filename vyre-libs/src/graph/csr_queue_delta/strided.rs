//! Row-strided queue-to-queue sparse CSR expansion.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use crate::graph::csr_frontier_step::{csr_queue_step_program, CsrQueueLanes};

#[cfg(test)]
use super::CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE;
use super::{define_csr_queue_delta_entry_point, CsrQueueDeltaEnqueueParams};

/// Canonical op id for row-strided queue-to-queue delta CSR expansion.
pub const CSR_QUEUE_DELTA_STRIDED_ENQUEUE_OP_ID: &str =
    "vyre-libs::graph::csr_queue_delta_strided_enqueue";

/// Fixed lane team assigned to each queued source row in the strided delta path.
pub const CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE: u32 =
    crate::graph::csr_queue_strided::CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE;

/// Maximum queued source rows assigned one logical lane team in a delta launch.
///
/// Larger queue waves are covered by a grid-stride loop inside the kernel. This
/// keeps resident queue closure on the fused repeated-sequence path without
/// launching worst-wave-sized grids for every half-wave.
pub const CSR_QUEUE_DELTA_STRIDED_MAX_SOURCE_SLOTS_PER_LAUNCH: u32 = 65_536;

/// Queue capacity above which launch compaction is worth grid-striding.
///
/// Medium queues keep one source row per logical lane. That avoids trading
/// empty-lane elision for extra loop work on graph waves that still have enough
/// active rows to occupy the device.
pub const CSR_QUEUE_DELTA_STRIDED_CAPPED_LAUNCH_MIN_CAPACITY: u32 = 65_536;

/// Queued source rows covered directly by one row-strided delta launch.
#[must_use]
pub const fn csr_queue_delta_strided_source_slots_per_launch(active_queue_capacity: u32) -> u32 {
    if active_queue_capacity == 0 {
        1
    } else if active_queue_capacity > CSR_QUEUE_DELTA_STRIDED_CAPPED_LAUNCH_MIN_CAPACITY {
        CSR_QUEUE_DELTA_STRIDED_MAX_SOURCE_SLOTS_PER_LAUNCH
    } else {
        active_queue_capacity
    }
}

/// Logical source-row lanes covered directly by one row-strided delta launch.
#[must_use]
pub const fn csr_queue_delta_strided_logical_lanes_per_launch(active_queue_capacity: u32) -> u32 {
    csr_queue_delta_strided_source_slots_per_launch(active_queue_capacity)
        .saturating_mul(CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE)
}

/// Dispatch grid for row-strided queue-to-queue delta expansion.
#[cfg(test)]
#[must_use]
pub const fn csr_queue_delta_strided_dispatch_grid(active_queue_capacity: u32) -> [u32; 3] {
    let total_lanes = csr_queue_delta_strided_logical_lanes_per_launch(active_queue_capacity);
    vyre_primitives::lane_grid(total_lanes, CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE[0])
}

define_csr_queue_delta_entry_point! {
    /// Build a row-strided delta enqueue program for skewed CSR source rows.
    ///
    /// This uses the same resident buffer ABI as
    /// [`super::csr_queue_delta_enqueue`], but assigns a fixed lane team to each
    /// queued source and stripes that source row across the team. It keeps
    /// high-degree IFDS hubs from serializing all edge work behind a single
    /// invocation.
    csr_queue_delta_strided_enqueue -> csr_queue_delta_strided_enqueue_with
}

/// Build a row-strided delta enqueue program for skewed CSR source rows.
#[must_use]
pub fn csr_queue_delta_strided_enqueue_with(params: CsrQueueDeltaEnqueueParams<'_>) -> Program {
    let node_count = params.node_count;
    let active_queue_capacity = params.active_queue_capacity;
    let next_queue_capacity = params.next_queue_capacity;
    if node_count == 0 || active_queue_capacity == 0 || next_queue_capacity == 0 {
        return trap_program(CSR_QUEUE_DELTA_STRIDED_ENQUEUE_OP_ID, Some((params.next_len, DataType::U32)), format!(
            "Fix: csr_queue_delta_strided_enqueue requires node_count > 0 and non-zero queue capacities, got node_count={node_count} active_queue_capacity={active_queue_capacity} next_queue_capacity={next_queue_capacity}."
        ));
    }
    // A wave wider than one launch covers its tail with a grid-stride loop
    // instead of a worst-wave-sized grid.
    let launch_lanes = (csr_queue_delta_strided_source_slots_per_launch(active_queue_capacity)
        < active_queue_capacity)
        .then(|| csr_queue_delta_strided_logical_lanes_per_launch(active_queue_capacity));
    csr_queue_step_program(&params.spec(
        CSR_QUEUE_DELTA_STRIDED_ENQUEUE_OP_ID,
        "csr_queue_delta_strided_enqueue",
        "qds",
        CsrQueueLanes::ActiveTeam {
            lanes: CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE,
            launch_lanes,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{assert_offset_overflow_traps, delta_program};
    use super::*;
    use crate::graph::mix32;

    #[test]
    fn emitted_strided_program_keeps_delta_queue_abi_and_expands_grid() {
        let program = delta_program(csr_queue_delta_strided_enqueue, 64, 7, 8);

        assert_eq!(
            program.workgroup_size,
            CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE
        );
        assert_eq!(program.buffers.len(), 8);
        assert_eq!(program.buffers[0].name.as_ref(), "active_queue");
        assert_eq!(program.buffers[0].count, 8);
        assert_eq!(program.buffers[6].name.as_ref(), "next_queue");
        assert_eq!(program.buffers[6].count, 16);
        assert_eq!(
            csr_queue_delta_strided_dispatch_grid(8),
            [
                (8 * CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE)
                    .div_ceil(CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE[0]),
                1,
                1,
            ]
        );
        let program_debug = format!("{:?}", program.entry);
        assert!(!program_debug.contains("qds_lane_iter"));
        assert!(program_debug.contains("qds_logical_lane"));

        let capped_program = delta_program(
            csr_queue_delta_strided_enqueue,
            64,
            7,
            CSR_QUEUE_DELTA_STRIDED_CAPPED_LAUNCH_MIN_CAPACITY + 1,
        );
        let capped_debug = format!("{:?}", capped_program.entry);
        assert!(capped_debug.contains("qds_lane_iter"));
        assert!(capped_debug.contains("qds_logical_lane"));
    }

    #[test]
    fn strided_delta_rejects_offset_count_overflow_without_panic() {
        assert_offset_overflow_traps(
            csr_queue_delta_strided_enqueue,
            "CSR queue delta strided builder",
        );
    }

    #[test]
    fn generated_strided_delta_launch_grid_caps_capacity_and_preserves_coverage() {
        const CASES: u32 = 20_000;
        let mut capped_cases = 0_u32;

        for case in 0..CASES {
            let capacity = mix32(case ^ 0x5D17_1D3A);
            let source_slots = csr_queue_delta_strided_source_slots_per_launch(capacity);
            let logical_lanes = csr_queue_delta_strided_logical_lanes_per_launch(capacity);
            let grid = csr_queue_delta_strided_dispatch_grid(capacity);
            let launched_lanes = grid[0].saturating_mul(CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE[0]);

            assert!(source_slots > 0, "source slots case {case}");
            if capacity == 0 {
                assert_eq!(source_slots, 1, "zero capacity source slots case {case}");
            } else if capacity > CSR_QUEUE_DELTA_STRIDED_CAPPED_LAUNCH_MIN_CAPACITY {
                assert_eq!(
                    source_slots, CSR_QUEUE_DELTA_STRIDED_MAX_SOURCE_SLOTS_PER_LAUNCH,
                    "source slot cap case {case}"
                );
            } else {
                assert_eq!(
                    source_slots, capacity,
                    "medium queue source slots case {case}"
                );
            }
            assert_eq!(
                logical_lanes,
                source_slots.saturating_mul(CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE),
                "logical lanes case {case}"
            );
            assert!(
                launched_lanes >= logical_lanes,
                "grid underlaunch case {case}"
            );
            assert!(
                launched_lanes < logical_lanes + CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE[0],
                "grid overlaunch case {case}"
            );
            if capacity > CSR_QUEUE_DELTA_STRIDED_CAPPED_LAUNCH_MIN_CAPACITY {
                capped_cases += 1;
                let active_lanes =
                    capacity.saturating_mul(CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE);
                let iterations = 1 + active_lanes.saturating_sub(1) / logical_lanes;
                assert!(iterations > 1, "grid-stride iterations case {case}");
            }
        }

        assert!(capped_cases > CASES * 9 / 10);
    }
}
