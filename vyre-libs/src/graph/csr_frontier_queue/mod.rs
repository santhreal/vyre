//! Device-side active-frontier queues for sparse CSR expansion.
//!
//! Low-density dataflow frontiers should not launch one useful lane and
//! thousands of empty source-node lanes. This module splits sparse expansion
//! into two GPU-resident primitives:
//!
//! 1. `frontier_to_queue` compacts active source-node ids from a packed bitset
//!    into an active queue with an atomic device-side length. The legacy variant
//!    uses one cooperative workgroup and a strided scan so the queue length can
//!    be initialized inside the same dispatch without an unsupported grid barrier.
//!    The packed-word variants let resident traversal pipelines initialize
//!    `queue_len` once, then reserve one queue slice per nonzero frontier word.
//! 2. `csr_queue_forward_traverse` consumes only queued sources and expands
//!    their CSR rows into `frontier_out`.
//!
//! The queue length can exceed queue capacity to expose overflow pressure; the
//! traversal consumes only the first `queue_capacity` entries.
use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use crate::graph::csr_frontier_step::{
    CsrQueueEmit, CsrQueueInputs, CsrQueueLanes, CsrQueueRowPlan, CsrQueueStepSpec,
};

pub mod batch_memory;
mod cpu_reference;
mod graph_validation;
mod packed_word_compaction;
mod queue_compaction;
mod queue_traverse;
pub(crate) mod resident_programs;
pub(crate) mod scratch;
mod sizing_diagnostics;
mod word_block_scan;
mod word_block_scatter;

#[cfg(test)]
mod emitted_program_shape;

#[cfg(any(test, feature = "cpu-parity"))]
pub use self::cpu_reference::{
    csr_queue_forward_traverse_cpu, frontier_to_queue_cpu, try_csr_queue_forward_traverse_cpu,
    try_csr_queue_forward_traverse_cpu_into, try_frontier_to_queue_cpu,
    try_frontier_to_queue_cpu_into,
};
pub use self::graph_validation::{
    validate_csr_queue_graph, validate_frontier_queue_batch, validate_frontier_queue_query,
};
pub use self::packed_word_compaction::{
    frontier_words_to_queue_clear_out_parallel, frontier_words_to_queue_parallel,
};
pub use self::queue_compaction::{
    frontier_queue_len_init, frontier_to_queue, frontier_to_queue_parallel,
};
pub use self::queue_traverse::{csr_queue_forward_traverse, csr_queue_forward_traverse_with};
pub use self::word_block_scan::{
    frontier_word_block_offsets_in_place, frontier_word_counts_scan_pass_a,
};
pub use self::word_block_scatter::{
    frontier_word_block_offsets_to_queue_parallel, frontier_word_block_prefix_to_queue_parallel,
};

/// Canonical op id for bitset-to-queue compaction.
pub const FRONTIER_TO_QUEUE_OP_ID: &str = "vyre-libs::graph::frontier_to_queue";
/// Canonical op id for multi-workgroup bitset-to-queue compaction.
pub const FRONTIER_TO_QUEUE_PARALLEL_OP_ID: &str = "vyre-libs::graph::frontier_to_queue_parallel";
/// Canonical op id for word-level multi-workgroup bitset-to-queue compaction.
pub const FRONTIER_WORDS_TO_QUEUE_PARALLEL_OP_ID: &str =
    "vyre-libs::graph::frontier_words_to_queue_parallel";
/// Canonical op id for word-level compaction that also clears an output bitset.
pub const FRONTIER_WORDS_TO_QUEUE_CLEAR_OUT_PARALLEL_OP_ID: &str =
    "vyre-libs::graph::frontier_words_to_queue_clear_out_parallel";
/// Canonical op id for packed-frontier word popcount prefix-scan pass A.
pub const FRONTIER_WORD_COUNTS_SCAN_PASS_A_OP_ID: &str =
    "vyre-libs::graph::frontier_word_counts_scan_pass_a";
/// Canonical op id for deterministic packed-frontier block-prefix scatter.
pub const FRONTIER_WORD_BLOCK_PREFIX_TO_QUEUE_PARALLEL_OP_ID: &str =
    "vyre-libs::graph::frontier_word_block_prefix_to_queue_parallel";
/// Canonical op id for in-place packed-frontier block-offset scan.
pub const FRONTIER_WORD_BLOCK_OFFSETS_IN_PLACE_OP_ID: &str =
    "vyre-libs::graph::frontier_word_block_offsets_in_place";
/// Canonical op id for packed-frontier scatter with precomputed block offsets.
pub const FRONTIER_WORD_BLOCK_OFFSETS_TO_QUEUE_PARALLEL_OP_ID: &str =
    "vyre-libs::graph::frontier_word_block_offsets_to_queue_parallel";
/// Canonical op id for device-side queue length initialization.
pub const FRONTIER_QUEUE_LEN_INIT_OP_ID: &str = "vyre-libs::graph::frontier_queue_len_init";
/// Canonical op id for queue-driven CSR expansion.
pub const CSR_QUEUE_FORWARD_OP_ID: &str = "vyre-libs::graph::csr_queue_forward_traverse";

/// Positional inputs for [`csr_queue_forward_traverse`].
#[derive(Clone, Copy, Debug)]
pub struct CsrQueueForwardTraverseParams<'a> {
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
    /// Node count the CSR row pointers and destination bounds are sized by.
    pub node_count: u32,
    /// Logical edge count the edge-slot bound check uses.
    pub edge_count: u32,
    /// Static capacity of `active_queue`.
    pub queue_capacity: u32,
    /// Edge kinds this traversal is allowed to follow.
    pub allow_mask: u32,
}

impl<'a> CsrQueueForwardTraverseParams<'a> {
    /// Refuse a shape that cannot address one queue slot. `builder_name` is the
    /// entry point named in the diagnostic the caller reads.
    pub(crate) fn empty_shape_program(
        &self,
        op_id: &'static str,
        builder_name: &str,
    ) -> Option<Program> {
        let node_count = self.node_count;
        let queue_capacity = self.queue_capacity;
        if node_count != 0 && queue_capacity != 0 {
            return None;
        }
        Some(trap_program(op_id, Some((self.frontier_out, DataType::U32)), format!(
            "Fix: {builder_name} requires node_count > 0 and queue_capacity > 0, got node_count={node_count} queue_capacity={queue_capacity}."
        )))
    }

    /// Point these inputs at the shared queue-step builder. The queued-row
    /// forward entry points differ only in op id, variable prefix, workgroup
    /// shape, and how lanes are assigned to a queued row.
    pub(crate) fn spec(
        &self,
        op_id: &'static str,
        builder_name: &'static str,
        prefix: &'a str,
        workgroup_size: [u32; 3],
        lanes: CsrQueueLanes,
    ) -> CsrQueueStepSpec<'a> {
        CsrQueueStepSpec {
            op_id,
            builder_name,
            prefix,
            workgroup_size,
            inputs: CsrQueueInputs {
                active_queue: self.active_queue,
                queue_len: self.queue_len,
                edge_offsets: self.edge_offsets,
                edge_targets: self.edge_targets,
                edge_kind_mask: self.edge_kind_mask,
            },
            lanes,
            row_plan: CsrQueueRowPlan::ExpandAll,
            emit: CsrQueueEmit::Frontier {
                frontier_out: self.frontier_out,
            },
            node_count: self.node_count,
            edge_count: self.edge_count,
            queue_capacity: self.queue_capacity,
            allow_mask: self.allow_mask,
        }
    }
}

/// Publish one queued-row forward expansion entry point.
///
/// Every variant of this family takes the same resident buffer ABI and forwards
/// to its `_with` form, so the positional argument list is stated once here
/// instead of once per lane strategy.
macro_rules! define_csr_queue_forward_entry_point {
    (
        $(#[$attr:meta])*
        $name:ident -> $with:ident
    ) => {
        $(#[$attr])*
        #[must_use]
        #[allow(clippy::too_many_arguments)]
        pub fn $name(
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
        ) -> vyre_foundation::ir::Program {
            $with(
                $crate::graph::csr_frontier_queue::CsrQueueForwardTraverseParams {
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
                },
            )
        }
    };
}

pub(crate) use define_csr_queue_forward_entry_point;

/// One positional queued-row forward entry point under test.
#[cfg(test)]
pub(crate) type CsrQueueForwardBuilder =
    fn(&str, &str, &str, &str, &str, &str, u32, u32, u32, u32) -> Program;

/// Every queued-row forward entry point owes the caller an invalid-output
/// program, not a panic, when `node_count + 1` overflows the CSR offset count.
#[cfg(test)]
pub(crate) fn assert_offset_overflow_traps(build: CsrQueueForwardBuilder, builder_name: &str) {
    let result = std::panic::catch_unwind(|| {
        build(
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
        "{builder_name} must emit an invalid program instead of panicking"
    );
    let program = result.unwrap();
    assert_eq!(program.workgroup_size, [1, 1, 1]);
    assert_eq!(program.buffers.len(), 1);
    assert_eq!(program.buffers[0].name.as_ref(), "out");
    assert!(program.stats().trap());
    let entry = format!("{:?}", program.entry());
    assert!(
        entry.contains("node_count + 1 overflows u32"),
        "Fix: {builder_name} must preserve the CSR offset overflow diagnostic, got: {entry}"
    );
}

/// Validated resident graph layout for queue-driven sparse traversal.
///
/// The primitive owns these derived counts so resident dispatch wrappers do not
/// fork CSR edge-count, edge-padding, or frontier bitset sizing rules.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CsrQueueGraphLayout {
    /// Number of graph nodes accepted by the primitive.
    pub node_count: u32,
    /// Exact physical edge count declared by `edge_offsets[node_count]`.
    pub edge_count: u32,
    /// Largest CSR row degree in the graph.
    pub max_row_degree: u32,
    /// Number of u32 words in each packed frontier bitset.
    pub words: usize,
    /// Number of u32 words to allocate/upload for edge target and kind arrays.
    pub edge_storage_words: usize,
}
