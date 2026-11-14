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
mod mode_selection;
mod sparse_dense_step;

pub use dense_step::adaptive_dense_step;
pub use four_russians::{
    adaptive_four_russians_dense_step, four_russians_dense_columns_from_adj_rows,
    four_russians_dense_lut_from_adj_rows, four_russians_dense_lut_words,
    four_russians_frontier_words, four_russians_source_tile_count,
};
#[cfg(test)]
pub use mode_selection::should_use_dense;
pub use mode_selection::{
    select_adaptive_traversal_mode, select_dense_traversal_kernel, AdaptiveTraversalMode,
    DenseTraversalKernel,
};
pub use sparse_dense_step::adaptive_sparse_dense_step;

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
