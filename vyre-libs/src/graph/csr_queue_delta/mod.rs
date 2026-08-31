//! Queue-to-queue sparse CSR expansion for delta fixpoint waves.
//!
//! A full frontier bitset scan is the wrong shape once a dataflow pipeline has
//! already compacted the active wave. This primitive consumes only queued
//! sources, updates a resident accumulator bitset, and appends first-time
//! discoveries directly into the next active queue.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use crate::graph::csr_frontier_step::{
    csr_queue_step_program, CsrQueueEmit, CsrQueueInputs, CsrQueueLanes, CsrQueueRowPlan,
    CsrQueueStepSpec,
};

mod strided;

pub use strided::{
    csr_queue_delta_strided_enqueue, csr_queue_delta_strided_enqueue_with,
    csr_queue_delta_strided_logical_lanes_per_launch,
    csr_queue_delta_strided_source_slots_per_launch,
    CSR_QUEUE_DELTA_STRIDED_CAPPED_LAUNCH_MIN_CAPACITY, CSR_QUEUE_DELTA_STRIDED_ENQUEUE_OP_ID,
    CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE, CSR_QUEUE_DELTA_STRIDED_MAX_SOURCE_SLOTS_PER_LAUNCH,
};

/// Canonical op id for queue-to-queue delta CSR expansion.
pub const CSR_QUEUE_DELTA_ENQUEUE_OP_ID: &str = "vyre-libs::graph::csr_queue_delta_enqueue";

/// Default workgroup size for queue-to-queue delta expansion.
pub const CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Positional inputs shared by [`csr_queue_delta_enqueue`] and
/// [`csr_queue_delta_strided_enqueue`].
#[derive(Clone, Copy, Debug)]
pub struct CsrQueueDeltaEnqueueParams<'a> {
    /// Compacted queue of active source nodes.
    pub active_queue: &'a str,
    /// Single-element resident length of `active_queue`.
    pub active_len: &'a str,
    /// CSR row pointers, `node_count + 1` entries.
    pub edge_offsets: &'a str,
    /// CSR edge destinations.
    pub edge_targets: &'a str,
    /// Per-edge kind bits tested against `allow_mask`.
    pub edge_kind_mask: &'a str,
    /// Monotone reachability bitset each reached destination is ORed into.
    pub accumulator: &'a str,
    /// Queue first-time discoveries are appended to.
    pub next_queue: &'a str,
    /// Single-element observed next length, which may exceed the capacity.
    pub next_len: &'a str,
    /// Node count the CSR row pointers and destination bounds are sized by.
    pub node_count: u32,
    /// Logical edge count the edge-slot bound check uses.
    pub edge_count: u32,
    /// Static capacity of `active_queue`.
    pub active_queue_capacity: u32,
    /// Static capacity of `next_queue`.
    pub next_queue_capacity: u32,
    /// Edge kinds this traversal is allowed to follow.
    pub allow_mask: u32,
}

/// Publish one queue-to-queue delta expansion entry point.
///
/// Both delta variants take the same resident buffer ABI and forward to their
/// `_with` form, so the positional argument list is stated once here instead of
/// once per lane strategy.
macro_rules! define_csr_queue_delta_entry_point {
    (
        $(#[$attr:meta])*
        $name:ident -> $with:ident
    ) => {
        $(#[$attr])*
        #[must_use]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
            active_queue: &str,
            active_len: &str,
            edge_offsets: &str,
            edge_targets: &str,
            edge_kind_mask: &str,
            accumulator: &str,
            next_queue: &str,
            next_len: &str,
            node_count: u32,
            edge_count: u32,
            active_queue_capacity: u32,
            next_queue_capacity: u32,
            allow_mask: u32,
        ) -> vyre_foundation::ir::Program {
            $with($crate::graph::csr_queue_delta::CsrQueueDeltaEnqueueParams {
                active_queue,
                active_len,
                edge_offsets,
                edge_targets,
                edge_kind_mask,
                accumulator,
                next_queue,
                next_len,
                node_count,
                edge_count,
                active_queue_capacity,
                next_queue_capacity,
                allow_mask,
            })
        }
    };
}

pub(crate) use define_csr_queue_delta_entry_point;

define_csr_queue_delta_entry_point! {
    /// Build a GPU program that expands queued CSR rows and enqueues only new nodes.
    ///
    /// `accumulator` is the monotone reachability bitset. When an allowed edge
    /// reaches a destination whose bit was absent, the destination is appended to
    /// `next_queue` and `next_len` is incremented. The observed next length can
    /// exceed `next_queue_capacity`; stores are clamped so callers can detect
    /// overflow pressure without corrupting resident memory.
    csr_queue_delta_enqueue -> csr_queue_delta_enqueue_with
}

/// Build a GPU program that expands queued CSR rows and enqueues only new nodes.
#[must_use]
pub fn csr_queue_delta_enqueue_with(params: CsrQueueDeltaEnqueueParams<'_>) -> Program {
    let node_count = params.node_count;
    let active_queue_capacity = params.active_queue_capacity;
    let next_queue_capacity = params.next_queue_capacity;
    if node_count == 0 || active_queue_capacity == 0 || next_queue_capacity == 0 {
        return trap_program(CSR_QUEUE_DELTA_ENQUEUE_OP_ID, Some((params.next_len, DataType::U32)), format!(
            "Fix: csr_queue_delta_enqueue requires node_count > 0 and non-zero queue capacities, got node_count={node_count} active_queue_capacity={active_queue_capacity} next_queue_capacity={next_queue_capacity}."
        ));
    }
    csr_queue_step_program(&params.spec(
        CSR_QUEUE_DELTA_ENQUEUE_OP_ID,
        "csr_queue_delta_enqueue",
        "qd",
        CsrQueueLanes::Scalar,
    ))
}

impl<'a> CsrQueueDeltaEnqueueParams<'a> {
    /// Point these inputs at the shared queue-step builder. Both delta entry
    /// points differ only in op id, variable prefix, and lane assignment.
    fn spec(
        &self,
        op_id: &'static str,
        builder_name: &'static str,
        prefix: &'a str,
        lanes: CsrQueueLanes,
    ) -> CsrQueueStepSpec<'a> {
        CsrQueueStepSpec {
            op_id,
            builder_name,
            prefix,
            workgroup_size: CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE,
            inputs: CsrQueueInputs {
                active_queue: self.active_queue,
                queue_len: self.active_len,
                edge_offsets: self.edge_offsets,
                edge_targets: self.edge_targets,
                edge_kind_mask: self.edge_kind_mask,
            },
            lanes,
            row_plan: CsrQueueRowPlan::ExpandAll,
            emit: CsrQueueEmit::Delta {
                accumulator: self.accumulator,
                next_queue: self.next_queue,
                next_len: self.next_len,
                next_queue_capacity: self.next_queue_capacity,
            },
            node_count: self.node_count,
            edge_count: self.edge_count,
            queue_capacity: self.active_queue_capacity,
            allow_mask: self.allow_mask,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_csr_queue_delta_enqueue_cpu_into(
        active_queue: &[u32],
        active_len: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        accumulator: &mut Vec<u32>,
        node_count: u32,
        next_queue_capacity: u32,
        allow_mask: u32,
        next_queue: &mut Vec<u32>,
    ) -> Result<u32, String> {
        let expected_words = crate::bitset::bitset_words(node_count) as usize;
        if accumulator.len() != expected_words {
            return Err(format!(
                "Fix: delta enqueue requires accumulator.len() == bitset_words(node_count), expected {}, got {}.",
                expected_words,
                accumulator.len()
            ));
        }
        if edge_offsets.len() != (node_count as usize) + 1 {
            return Err(format!(
                "Fix: delta enqueue requires edge_offsets.len() == node_count + 1, expected {}, got {}.",
                (node_count as usize) + 1,
                edge_offsets.len()
            ));
        }
        if edge_targets.len() != edge_kind_mask.len() {
            return Err(format!(
                "Fix: delta enqueue requires edge_targets.len() == edge_kind_mask.len(), got {} vs {}.",
                edge_targets.len(),
                edge_kind_mask.len()
            ));
        }
        let take = (active_len as usize).min(active_queue.len());
        let mut acc = accumulator.clone();
        let mut discoveries = Vec::new();
        for &src in &active_queue[..take] {
            if src >= node_count {
                continue;
            }
            let start = edge_offsets[src as usize] as usize;
            let end = edge_offsets[src as usize + 1] as usize;
            for edge in start..end {
                if edge_kind_mask[edge] & allow_mask == 0 {
                    continue;
                }
                let dst = edge_targets[edge];
                if dst < node_count {
                    let word = (dst / 32) as usize;
                    let bit = 1_u32 << (dst % 32);
                    if (acc[word] & bit) == 0 {
                        acc[word] |= bit;
                        discoveries.push(dst);
                    }
                }
            }
        }
        let total_discoveries = discoveries.len() as u32;
        let cap = next_queue_capacity as usize;
        next_queue.clear();
        next_queue.extend(discoveries.into_iter().take(cap));
        *accumulator = acc;
        Ok(total_discoveries)
    }

    #[allow(clippy::too_many_arguments)]
    fn csr_queue_delta_enqueue_cpu(
        active_queue: &[u32],
        active_len: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
        edge_kind_mask: &[u32],
        accumulator: &[u32],
        node_count: u32,
        next_queue_capacity: u32,
        allow_mask: u32,
    ) -> (Vec<u32>, Vec<u32>, u32) {
        let mut acc = accumulator.to_vec();
        let mut next_queue = Vec::new();
        let next_len = try_csr_queue_delta_enqueue_cpu_into(
            active_queue,
            active_len,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
            &mut acc,
            node_count,
            next_queue_capacity,
            allow_mask,
            &mut next_queue,
        )
        .expect("csr_queue_delta_enqueue_cpu failed");
        (acc, next_queue, next_len)
    }

    /// One positional delta entry point under test.
    pub(super) type DeltaEnqueueBuilder =
        fn(&str, &str, &str, &str, &str, &str, &str, &str, u32, u32, u32, u32, u32) -> Program;

    /// Build a delta program over the canonical buffer names. Both entry points
    /// carry one ABI, so the argument list is stated here and not once per test.
    pub(super) fn delta_program(
        build: DeltaEnqueueBuilder,
        node_count: u32,
        edge_count: u32,
        active_queue_capacity: u32,
    ) -> Program {
        build(
            "active_queue",
            "active_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "accumulator",
            "next_queue",
            "next_len",
            node_count,
            edge_count,
            active_queue_capacity,
            16,
            1,
        )
    }

    /// Every delta entry point owes the caller a trap program, not a panic, when
    /// `node_count + 1` overflows the CSR offset count.
    pub(super) fn assert_offset_overflow_traps(build: DeltaEnqueueBuilder, label: &str) {
        let result = std::panic::catch_unwind(|| delta_program(build, u32::MAX, 0, 1));

        assert!(
            result.is_ok(),
            "{label} must reject offset-count overflow without panicking"
        );
        let program = result.unwrap();
        assert!(program.stats().trap());
        let entry = format!("{:?}", program.entry());
        assert!(
            entry.contains("node_count + 1 overflows u32"),
            "Fix: trap must retain the CSR offset-count overflow diagnostic, got: {entry}"
        );
    }

    #[test]
    fn emitted_program_has_stable_delta_queue_shape() {
        let program = delta_program(csr_queue_delta_enqueue, 64, 7, 8);

        assert_eq!(
            program.workgroup_size,
            CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE
        );
        assert_eq!(program.buffers.len(), 8);
    }

    #[test]
    fn delta_enqueue_rejects_offset_count_overflow_without_panic() {
        assert_offset_overflow_traps(csr_queue_delta_enqueue, "CSR queue delta builder");
    }

    #[test]
    fn cpu_delta_enqueue_only_emits_first_time_discoveries() {
        let edge_offsets = [0, 3, 4, 4, 4, 4];
        let edge_targets = [1, 2, 3, 4];
        let edge_kind_mask = [1, 1, 2, 1];
        let accumulator = vec![0b00001];

        let (accumulator, next_queue, next_len) = csr_queue_delta_enqueue_cpu(
            &[0, 1],
            2,
            &edge_offsets,
            &edge_targets,
            &edge_kind_mask,
            &accumulator,
            5,
            8,
            1,
        );

        assert_eq!(accumulator, vec![0b10111]);
        assert_eq!(next_queue, vec![1, 2, 4]);
        assert_eq!(next_len, 3);
    }

    #[test]
    fn cpu_delta_enqueue_reports_queue_pressure_without_clobbering_accumulator() {
        let edge_offsets = [0, 3, 3, 3, 3];
        let edge_targets = [1, 2, 3];
        let edge_kind_mask = [1, 1, 1];
        let mut accumulator = vec![0b0001];
        let mut next_queue = Vec::new();

        let next_len = try_csr_queue_delta_enqueue_cpu_into(
            &[0],
            1,
            &edge_offsets,
            &edge_targets,
            &edge_kind_mask,
            &mut accumulator,
            4,
            2,
            1,
            &mut next_queue,
        )
        .expect("Fix: canonical queue delta graph should enqueue bounded discoveries");

        assert_eq!(accumulator, vec![0b1111]);
        assert_eq!(next_queue, vec![1, 2]);
        assert_eq!(next_len, 3);
    }

    #[test]
    fn cpu_delta_enqueue_rejects_bad_accumulator_without_clobbering_outputs() {
        let mut accumulator = vec![0xCAFE_BABE, 0xDEAD_BEEF];
        let mut next_queue = vec![9, 8, 7];

        let err = try_csr_queue_delta_enqueue_cpu_into(
            &[0],
            1,
            &[0, 1],
            &[0],
            &[1],
            &mut accumulator,
            1,
            4,
            1,
            &mut next_queue,
        )
        .expect_err("wrong accumulator width must fail before mutation");

        assert!(err.contains("accumulator.len() == bitset_words(node_count)"));
        assert_eq!(accumulator, vec![0xCAFE_BABE, 0xDEAD_BEEF]);
        assert_eq!(next_queue, vec![9, 8, 7]);
    }
}
