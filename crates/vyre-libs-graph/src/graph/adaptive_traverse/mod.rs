//! Adaptive CSR / dense-bitmatrix traversal (G4).
//!
//! # What this is
//!
//! `csr_forward_traverse` is ideal when the BFS frontier is sparse
//! (<~5% of nodes). When the frontier saturates, a dense-bitmatrix
//! step (adjacency × frontier) wins  -  each tile's adjacency bitrow
//! × its frontier bitset is one vectorised OR over a pair of 32-bit
//! words, with contiguous DRAM access patterns that outrun CSR.
//!
//! This module exposes both a dense step and a hybrid sparse/dense
//! step. The hybrid step consumes a device-resident frontier popcount
//! buffer, so a prior GPU reduction can select CSR or dense execution
//! without reading the frontier back to the CPU:
//!
//! ```text
//!   density_pct = 100 * popcount(frontier_in) / node_count
//!   if density_pct >= DENSE_THRESHOLD_PCT: dense step
//!   else: CSR step
//! ```
//!
//! The dense step is a bitmatrix multiply:
//!
//! ```text
//!   for dst in 0..node_count:
//!     if (adj_row[dst] & frontier_in) != 0:
//!       frontier_out[dst] = 1
//! ```
//!
//! where `adj_row[dst]` is a bitset over source-node predecessors
//! (reverse adjacency, encoded as one row of `bitset_words(node_count)`
//! u32s per destination node).
//!
//! # Buffers
//!
//! - `frontier_in`   -  ReadOnly, packed bitset, `bitset_words(n)` u32.
//! - `frontier_out`  -  ReadWrite, same shape.
//! - `frontier_popcount`  -  ReadOnly, one u32 set-bit count for
//!   device-side sparse/dense selection in the hybrid step.
//! - `edge_offsets`, `edge_targets`, `edge_kind_mask`  -  CSR graph
//!   buffers for sparse expansion in the hybrid step.
//! - `adj_rows_dense`  -  ReadOnly, `node_count × bitset_words(n)` u32.
//!   Row `d` is the bitset of predecessors of node `d`.

mod dense_step;
mod four_russians;
mod frontier_plan;
mod mode_selection;
mod plan_cache_key;
mod sparse_dense_step;
#[cfg(test)]
mod test_graphs;
mod traversal_plan;

pub use dense_step::adaptive_dense_step;
pub use four_russians::{
    adaptive_four_russians_dense_step, four_russians_dense_columns_from_adj_rows,
    four_russians_dense_lut_from_adj_rows, four_russians_dense_lut_words,
    four_russians_frontier_words, four_russians_source_tile_count,
};
#[cfg(test)]
pub use frontier_plan::{adaptive_frontier_popcount, adaptive_frontier_popcount_in_domain};
pub use frontier_plan::{
    adaptive_frontier_stats, plan_adaptive_frontier_work, validate_adaptive_frontier,
    validate_adaptive_traversal_layout, AdaptiveFrontierLayout, AdaptiveFrontierStats,
    AdaptiveFrontierWorkPlan, AdaptiveTraversalLayout,
};
#[cfg(test)]
pub use mode_selection::should_use_dense;
pub use mode_selection::{
    select_adaptive_traversal_mode, select_dense_traversal_kernel, AdaptiveTraversalMode,
    DenseTraversalKernel,
};
pub use plan_cache_key::{
    adaptive_four_russians_graph_content_hash, adaptive_sparse_queue_graph_content_hash,
    adaptive_traversal_graph_content_hash, adaptive_traversal_program_layout_hash,
    adaptive_traversal_split_program_layout_hash, AdaptiveTraversalPlanCacheKey,
    AdaptiveTraversalProgramKind,
};
pub use sparse_dense_step::adaptive_sparse_dense_step;
pub use traversal_plan::{
    plan_adaptive_dense_step, plan_adaptive_traversal_step, AdaptiveDenseStepPlan,
    AdaptiveTraversalBuffers, AdaptiveTraversalStepPlan,
};

const ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Density threshold (percent). Tiles with ≥ this fraction of
/// frontier bits set use the dense-bitmatrix step; below it, CSR.
/// 25% is the empirical crossover on current desktop GPU architectures.
pub const DENSE_THRESHOLD_PCT: u32 = 25;

/// Canonical op id for the dense step.
pub const OP_ID: &str = "vyre-libs::graph::adaptive_traverse_dense";
/// Canonical op id for the device-selected sparse/dense step.
pub const HYBRID_OP_ID: &str = "vyre-libs::graph::adaptive_traverse_sparse_dense";
/// Canonical op id for graph-level dense Four-Russians traversal planning.
pub const FOUR_RUSSIANS_DENSE_OP_ID: &str =
    "vyre-libs::graph::adaptive_traverse_four_russians_dense";

/// Canonical input-frontier buffer name.
pub const NAME_FRONTIER_IN: &str = "adap_frontier_in";
/// Canonical output-frontier buffer name.
pub const NAME_FRONTIER_OUT: &str = "adap_frontier_out";
/// Canonical frontier-popcount buffer name.
pub const NAME_FRONTIER_POPCOUNT: &str = "adap_frontier_popcount";
/// Canonical CSR row-offset buffer name.
pub const NAME_EDGE_OFFSETS: &str = "adap_edge_offsets";
/// Canonical CSR edge-target buffer name.
pub const NAME_EDGE_TARGETS: &str = "adap_edge_targets";
/// Canonical CSR edge-kind mask buffer name.
pub const NAME_EDGE_KIND_MASK: &str = "adap_edge_kind_mask";
/// Canonical dense adjacency-row buffer name.
pub const NAME_ADJ_ROWS_DENSE: &str = "adap_adj_rows_dense";
/// Canonical compacted active-source queue buffer name.
pub const NAME_ACTIVE_QUEUE: &str = "adap_active_queue";
/// Canonical resident active-queue length buffer name.
pub const NAME_QUEUE_LEN: &str = "adap_queue_len";
