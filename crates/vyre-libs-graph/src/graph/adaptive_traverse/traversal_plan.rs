//! Traversal selection that returns the composition it selects: the traversal
//! step for measured graph statistics, and the dense kernel with the constant
//! tile table that kernel binds.

use vyre_foundation::ir::Program;

use super::dense_step::adaptive_dense_step;
use super::four_russians::{
    adaptive_four_russians_dense_step, four_russians_dense_lut_from_adj_rows,
};
use super::mode_selection::{
    select_adaptive_traversal_mode, select_dense_traversal_kernel, AdaptiveTraversalMode,
    DenseTraversalKernel,
};
use super::sparse_dense_step::adaptive_sparse_dense_step;
use super::{
    DENSE_THRESHOLD_PCT, NAME_ACTIVE_QUEUE, NAME_ADJ_ROWS_DENSE, NAME_EDGE_KIND_MASK,
    NAME_EDGE_OFFSETS, NAME_EDGE_TARGETS, NAME_FRONTIER_IN, NAME_FRONTIER_OUT,
    NAME_FRONTIER_POPCOUNT, NAME_QUEUE_LEN,
};
use crate::graph::csr_frontier_queue::{
    csr_queue_forward_traverse_with, CsrQueueForwardTraverseParams,
};

/// Buffer names one adaptive traversal step binds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdaptiveTraversalBuffers<'a> {
    /// Packed input frontier bitset.
    pub frontier_in: &'a str,
    /// Packed output frontier bitset.
    pub frontier_out: &'a str,
    /// Single-element device-resident frontier popcount.
    pub frontier_popcount: &'a str,
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
    /// Dense reverse-adjacency rows.
    pub adj_rows_dense: &'a str,
}

impl AdaptiveTraversalBuffers<'static> {
    /// The names an adaptive traversal step binds, in this crate's spelling.
    ///
    /// A caller that names the set here rather than restating nine literals
    /// produces a program comparable byte for byte with every other caller's.
    pub const CANONICAL: Self = Self {
        frontier_in: NAME_FRONTIER_IN,
        frontier_out: NAME_FRONTIER_OUT,
        frontier_popcount: NAME_FRONTIER_POPCOUNT,
        active_queue: NAME_ACTIVE_QUEUE,
        queue_len: NAME_QUEUE_LEN,
        edge_offsets: NAME_EDGE_OFFSETS,
        edge_targets: NAME_EDGE_TARGETS,
        edge_kind_mask: NAME_EDGE_KIND_MASK,
        adj_rows_dense: NAME_ADJ_ROWS_DENSE,
    };
}

/// One adaptive traversal step and the mode it was selected for.
#[derive(Clone, Debug)]
pub struct AdaptiveTraversalStepPlan {
    /// Selected traversal mode.
    pub mode: AdaptiveTraversalMode,
    /// Traversal step Program for the selected mode.
    pub program: Program,
}

/// One dense traversal step and the constant table it binds.
#[derive(Clone, Debug)]
pub struct AdaptiveDenseStepPlan {
    /// Selected dense kernel.
    pub kernel: DenseTraversalKernel,
    /// Dense traversal step Program for the selected kernel.
    pub program: Program,
    /// Byte-tile table words the Four-Russians kernel binds. Empty for the row
    /// scan, which reads the dense rows directly.
    pub tile_lut: Vec<u32>,
}

/// Select and build one adaptive traversal step for measured statistics.
///
/// Selection picks which composition to build, so it returns that composition
/// rather than a mode a caller then has to act on. Nothing here states launch
/// geometry: each step declares its own logical span and the compiler ranks the
/// schedule.
#[must_use]
pub fn plan_adaptive_traversal_step(
    buffers: &AdaptiveTraversalBuffers<'_>,
    node_count: u32,
    edge_count: u32,
    frontier_popcount: u32,
    queue_capacity: u32,
    allow_mask: u32,
) -> AdaptiveTraversalStepPlan {
    let mode = select_adaptive_traversal_mode(
        node_count,
        edge_count,
        frontier_popcount,
        DENSE_THRESHOLD_PCT,
    );
    let program = match mode {
        AdaptiveTraversalMode::SparseQueue => {
            csr_queue_forward_traverse_with(CsrQueueForwardTraverseParams {
                active_queue: buffers.active_queue,
                queue_len: buffers.queue_len,
                edge_offsets: buffers.edge_offsets,
                edge_targets: buffers.edge_targets,
                edge_kind_mask: buffers.edge_kind_mask,
                frontier_out: buffers.frontier_out,
                node_count,
                edge_count,
                queue_capacity,
                allow_mask,
            })
        }
        AdaptiveTraversalMode::SparseDense => adaptive_sparse_dense_step(
            buffers.frontier_in,
            buffers.frontier_out,
            buffers.frontier_popcount,
            buffers.edge_offsets,
            buffers.edge_targets,
            buffers.edge_kind_mask,
            buffers.adj_rows_dense,
            node_count,
            edge_count,
            allow_mask,
            DENSE_THRESHOLD_PCT,
        ),
    };
    AdaptiveTraversalStepPlan { mode, program }
}

/// Select and build the dense traversal step for measured statistics.
///
/// The Four-Russians byte-tile kernel amortizes a larger constant table over
/// repeated waves, so it is selected only once the frontier is dense, the graph
/// is large enough for row-scan waste to matter, and the table is reused across
/// at least two steps. The table is a compile-time constant derived from the
/// dense rows and is uploaded as an input, never computed on the host per step.
///
/// # Errors
///
/// Propagates dense reverse-row validation failures from the table build.
pub fn plan_adaptive_dense_step(
    frontier_in: &str,
    tile_lut: &str,
    frontier_out: &str,
    adj_rows_dense_name: &str,
    node_count: u32,
    frontier_popcount: u32,
    expected_lut_reuse_steps: u32,
    adj_rows_dense: &[u32],
) -> Result<AdaptiveDenseStepPlan, String> {
    let kernel =
        select_dense_traversal_kernel(node_count, frontier_popcount, expected_lut_reuse_steps);
    match kernel {
        DenseTraversalKernel::FourRussiansByteTile => Ok(AdaptiveDenseStepPlan {
            kernel,
            program: adaptive_four_russians_dense_step(
                frontier_in,
                tile_lut,
                frontier_out,
                node_count,
            ),
            tile_lut: four_russians_dense_lut_from_adj_rows(node_count, adj_rows_dense)?,
        }),
        DenseTraversalKernel::RowScanBitmatrix => Ok(AdaptiveDenseStepPlan {
            kernel,
            program: adaptive_dense_step(
                frontier_in,
                frontier_out,
                adj_rows_dense_name,
                node_count,
            ),
            tile_lut: Vec::new(),
        }),
    }
}
