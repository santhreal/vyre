//! Resident CSR frontier-queue Program construction.
//!
//! Three resident entry points run the same queue-driven CSR traversal against
//! different resident-buffer protocols: the single-query resident path
//! (`csr_frontier_queue_resident`), the batched resident path
//! (`csr_frontier_queue_batch_resident`), and adaptive traversal's sparse-queue
//! mode (`adaptive_traverse`). They differ only in which resident handles they
//! bind, which frontier upload buffer they read, and how they cache the
//! resulting Programs.
//!
//! The Programs themselves are built here, once, out of the `vyre-primitives`
//! queue-step builders. Nothing in this crate re-implements the queue bound
//! check, the CSR row lookup, the edge walk, or the edge guard.

use vyre_foundation::ir::Program;
use vyre_primitives::bitset::zero::bitset_zero;
use vyre_primitives::graph::csr_frontier_queue::{
    csr_queue_forward_traverse, frontier_queue_len_init, frontier_word_block_offsets_in_place,
    frontier_word_block_offsets_to_queue_parallel, frontier_word_block_prefix_to_queue_parallel,
    frontier_word_counts_scan_pass_a, frontier_words_to_queue_clear_out_parallel,
};
use vyre_primitives::graph::csr_queue_split::csr_queue_split_low_forward_traverse;
use vyre_primitives::graph::csr_queue_strided::csr_queue_strided_forward_traverse;

use crate::graph::csr_frontier_queue_scratch::{
    ResidentCsrQueueMaterializer, ResidentCsrQueueTraverseKind, STRIDED_FORWARD_MIN_ROW_DEGREE,
};

#[cfg(test)]
mod tests;

// Resident buffer names. Every site binds these handles in this order, so the
// names live here once instead of as string literals at three call sites.
const ACTIVE_QUEUE: &str = "active_queue";
const QUEUE_LEN: &str = "queue_len";
const HIGH_QUEUE: &str = "high_queue";
const HIGH_LEN: &str = "high_len";
const EDGE_OFFSETS: &str = "edge_offsets";
const EDGE_TARGETS: &str = "edge_targets";
const EDGE_KIND_MASK: &str = "edge_kind_mask";
const FRONTIER_OUT: &str = "frontier_out";
const WORD_PARTIALS: &str = "word_partials";
const BLOCK_TOTALS: &str = "block_totals";

/// Build the queue consumer for one resident traverse kind.
///
/// `queue_capacity` bounds the primary queue; a mixed split consumes the high
/// queue it filled, so its own capacity is the one that bounds the launch.
pub(crate) fn resident_csr_queue_traverse_program(
    node_count: u32,
    edge_count: u32,
    queue_capacity: u32,
    allow_mask: u32,
    kind: ResidentCsrQueueTraverseKind,
) -> Program {
    match kind {
        ResidentCsrQueueTraverseKind::RowSerial => csr_queue_forward_traverse(
            ACTIVE_QUEUE,
            QUEUE_LEN,
            EDGE_OFFSETS,
            EDGE_TARGETS,
            EDGE_KIND_MASK,
            FRONTIER_OUT,
            node_count,
            edge_count,
            queue_capacity,
            allow_mask,
        ),
        ResidentCsrQueueTraverseKind::RowStrided => csr_queue_strided_forward_traverse(
            ACTIVE_QUEUE,
            QUEUE_LEN,
            EDGE_OFFSETS,
            EDGE_TARGETS,
            EDGE_KIND_MASK,
            FRONTIER_OUT,
            node_count,
            edge_count,
            queue_capacity,
            allow_mask,
        ),
        ResidentCsrQueueTraverseKind::MixedSplit {
            high_queue_capacity,
        } => csr_queue_strided_forward_traverse(
            HIGH_QUEUE,
            HIGH_LEN,
            EDGE_OFFSETS,
            EDGE_TARGETS,
            EDGE_KIND_MASK,
            FRONTIER_OUT,
            node_count,
            edge_count,
            high_queue_capacity,
            allow_mask,
        ),
    }
}

/// Build the mixed-split low-degree pass: walk short rows in place and compact
/// hubs into the bounded high queue for the row-strided consumer.
pub(crate) fn resident_csr_queue_split_low_program(
    node_count: u32,
    edge_count: u32,
    queue_capacity: u32,
    high_queue_capacity: u32,
    allow_mask: u32,
) -> Program {
    csr_queue_split_low_forward_traverse(
        ACTIVE_QUEUE,
        QUEUE_LEN,
        EDGE_OFFSETS,
        EDGE_TARGETS,
        EDGE_KIND_MASK,
        FRONTIER_OUT,
        HIGH_QUEUE,
        HIGH_LEN,
        node_count,
        edge_count,
        queue_capacity,
        high_queue_capacity,
        STRIDED_FORWARD_MIN_ROW_DEGREE,
        allow_mask,
    )
}

/// Reset a resident queue length counter to zero.
pub(crate) fn resident_csr_queue_len_init_program(queue_len: &str) -> Program {
    frontier_queue_len_init(queue_len)
}

/// Clear the packed output frontier before a word-prefix materialization,
/// which does not clear it as a side effect of scanning.
pub(crate) fn resident_csr_queue_clear_frontier_out_program(words: u32) -> Program {
    bitset_zero(FRONTIER_OUT, words)
}

/// Scan packed frontier words and append active sources atomically, clearing
/// the output frontier in the same pass.
pub(crate) fn resident_csr_queue_atomic_word_scan_program(
    frontier_in: &str,
    node_count: u32,
    queue_capacity: u32,
) -> Program {
    frontier_words_to_queue_clear_out_parallel(
        frontier_in,
        ACTIVE_QUEUE,
        QUEUE_LEN,
        FRONTIER_OUT,
        node_count,
        queue_capacity,
    )
}

/// Popcount every frontier word into per-word partials and per-block totals.
pub(crate) fn resident_csr_queue_word_counts_program(
    frontier_in: &str,
    node_count: u32,
) -> Program {
    frontier_word_counts_scan_pass_a(frontier_in, WORD_PARTIALS, BLOCK_TOTALS, node_count)
}

/// Turn per-block totals into per-block offsets in place, for block counts
/// where a separate scan launch beats summing inside the scatter pass.
pub(crate) fn resident_csr_queue_block_offsets_program(node_count: u32) -> Program {
    frontier_word_block_offsets_in_place(BLOCK_TOTALS, node_count)
}

/// Scatter active sources into queue order from the word-prefix scan.
///
/// `precomputed_block_offsets` selects the variant that reads offsets produced
/// by [`resident_csr_queue_block_offsets_program`] instead of summing block
/// totals inline.
pub(crate) fn resident_csr_queue_word_prefix_queue_program(
    frontier_in: &str,
    node_count: u32,
    queue_capacity: u32,
    precomputed_block_offsets: bool,
) -> Program {
    if precomputed_block_offsets {
        frontier_word_block_offsets_to_queue_parallel(
            frontier_in,
            WORD_PARTIALS,
            BLOCK_TOTALS,
            ACTIVE_QUEUE,
            QUEUE_LEN,
            node_count,
            queue_capacity,
        )
    } else {
        frontier_word_block_prefix_to_queue_parallel(
            frontier_in,
            WORD_PARTIALS,
            BLOCK_TOTALS,
            ACTIVE_QUEUE,
            QUEUE_LEN,
            node_count,
            queue_capacity,
        )
    }
}

/// Programs that turn a packed input frontier into an active-source queue.
///
/// Which fields are populated is a function of the materializer alone, so the
/// resident sites clear and refill their whole cached set from one value.
pub(crate) struct ResidentCsrQueueMaterializerPrograms {
    pub(crate) clear_frontier_out: Option<Program>,
    pub(crate) queue_len_init: Option<Program>,
    pub(crate) word_counts: Option<Program>,
    pub(crate) word_block_offsets: Option<Program>,
    pub(crate) queue: Program,
}

/// Build the whole queue-materialization Program set for one resident query
/// shape. `precomputed_block_offsets` is ignored by the atomic scan.
pub(crate) fn resident_csr_queue_materializer_programs(
    frontier_in: &str,
    node_count: u32,
    words: u32,
    queue_capacity: u32,
    materializer: ResidentCsrQueueMaterializer,
    precomputed_block_offsets: bool,
) -> ResidentCsrQueueMaterializerPrograms {
    match materializer {
        ResidentCsrQueueMaterializer::AtomicWordScan => ResidentCsrQueueMaterializerPrograms {
            clear_frontier_out: None,
            queue_len_init: Some(resident_csr_queue_len_init_program(QUEUE_LEN)),
            word_counts: None,
            word_block_offsets: None,
            queue: resident_csr_queue_atomic_word_scan_program(
                frontier_in,
                node_count,
                queue_capacity,
            ),
        },
        ResidentCsrQueueMaterializer::DeterministicWordPrefix => {
            ResidentCsrQueueMaterializerPrograms {
                clear_frontier_out: Some(resident_csr_queue_clear_frontier_out_program(words)),
                queue_len_init: None,
                word_counts: Some(resident_csr_queue_word_counts_program(
                    frontier_in,
                    node_count,
                )),
                word_block_offsets: precomputed_block_offsets
                    .then(|| resident_csr_queue_block_offsets_program(node_count)),
                queue: resident_csr_queue_word_prefix_queue_program(
                    frontier_in,
                    node_count,
                    queue_capacity,
                    precomputed_block_offsets,
                ),
            }
        }
    }
}
