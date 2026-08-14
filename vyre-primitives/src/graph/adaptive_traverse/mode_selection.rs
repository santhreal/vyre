//! Traversal mode selection: the frontier-density cutover between the CSR,
//! sparse-queue, and dense-bitmatrix paths, and the dense kernel choice that
//! follows it.

use super::DENSE_THRESHOLD_PCT;
use crate::bitset::{bitset_words, frontier::frontier_tail_mask};

/// Runtime traversal strategy selected from frontier and graph statistics.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveTraversalMode {
    /// Materialize active source nodes into a device queue, then consume only
    /// queued CSR rows. Best for low-density frontiers.
    SparseQueue,
    /// Let the GPU selector choose sparse CSR vs dense reverse-bitmatrix from
    /// a device-resident frontier popcount.
    SparseDense,
}

/// Dense-frontier kernel selected after the sparse/dense branch chooses dense.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DenseTraversalKernel {
    /// Scan one dense reverse-adjacency row per destination node.
    RowScanBitmatrix,
    /// Use byte-tile Four-Russians source-column LUTs.
    FourRussiansByteTile,
}

#[must_use]
pub(super) fn dense_cutover_nodes(node_count: u32, threshold_pct: u32) -> u32 {
    if node_count == 0 {
        return u32::MAX;
    }
    let numerator = u64::from(node_count).saturating_mul(u64::from(threshold_pct));
    let cutover = numerator.div_ceil(100);
    cutover.min(u64::from(u32::MAX)) as u32
}

#[must_use]
pub(super) fn should_use_dense_with_popcount(popcount: u32, node_count: u32, threshold_pct: u32) -> bool {
    if node_count == 0 {
        return false;
    }
    popcount >= dense_cutover_nodes(node_count, threshold_pct)
}

/// Host-side density probe. Returns `true` iff
/// `popcount(frontier_in) / node_count ≥ DENSE_THRESHOLD_PCT / 100`.
///
/// `frontier_in` is the packed bitset; `node_count` is the total
/// number of nodes (not necessarily a multiple of 32). Integer-only
/// comparison  -  no floating-point rounding surprises.
#[must_use]
pub fn should_use_dense(frontier_in: &[u32], node_count: u32) -> bool {
    if node_count == 0 {
        return false;
    }
    let expected_words = bitset_words(node_count) as usize;
    let final_word_mask = frontier_tail_mask(node_count);
    let popcount: u32 = frontier_in
        .iter()
        .take(expected_words)
        .enumerate()
        .map(|(index, &word)| {
            if index + 1 == expected_words {
                word & final_word_mask
            } else {
                word
            }
            .count_ones()
        })
        .sum();
    should_use_dense_with_popcount(popcount, node_count, DENSE_THRESHOLD_PCT)
}

/// Select an adaptive traversal mode from measured frontier/graph statistics.
///
/// The sparse queue path removes whole-graph lane waste, but pays an extra
/// queue zero/upload and one atomic append per active source. The sparse/dense
/// path is better once the frontier is broad enough that scanning node lanes is
/// not mostly empty or when graph average degree makes queue materialization
/// less decisive than dense row coalescing.
#[must_use]
pub fn select_adaptive_traversal_mode(
    node_count: u32,
    edge_count: u32,
    frontier_popcount: u32,
    dense_threshold_pct: u32,
) -> AdaptiveTraversalMode {
    if node_count == 0 || frontier_popcount == 0 {
        return AdaptiveTraversalMode::SparseQueue;
    }
    let frontier_bps = (u64::from(frontier_popcount) * 10_000) / u64::from(node_count);
    let dense_cutover_bps = u64::from(dense_threshold_pct).saturating_mul(100);
    if frontier_bps >= dense_cutover_bps {
        return AdaptiveTraversalMode::SparseDense;
    }
    let avg_degree_x100 = (u64::from(edge_count) * 100) / u64::from(node_count);
    if frontier_bps <= 625 || (frontier_bps <= 1_250 && avg_degree_x100 >= 400) {
        AdaptiveTraversalMode::SparseQueue
    } else {
        AdaptiveTraversalMode::SparseDense
    }
}

/// Select the dense traversal kernel after the sparse/dense cutover fires.
///
/// Four-Russians byte tiles amortize a larger LUT over repeated graph waves.
/// They are selected only when the frontier is dense, the graph is large
/// enough for row-scan waste to matter, and the caller expects to reuse the
/// precomputed tile LUT across at least two traversal steps.
#[must_use]
pub fn select_dense_traversal_kernel(
    node_count: u32,
    frontier_popcount: u32,
    expected_lut_reuse_steps: u32,
) -> DenseTraversalKernel {
    if node_count < 64 || frontier_popcount == 0 || expected_lut_reuse_steps < 2 {
        return DenseTraversalKernel::RowScanBitmatrix;
    }
    if should_use_dense_with_popcount(frontier_popcount, node_count, DENSE_THRESHOLD_PCT) {
        DenseTraversalKernel::FourRussiansByteTile
    } else {
        DenseTraversalKernel::RowScanBitmatrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::adaptive_traverse::test_graphs::pack_nodes;

    #[test]
    fn should_use_dense_empty_frontier_is_false() {
        assert!(!should_use_dense(&[0_u32], 32));
    }

    #[test]
    fn should_use_dense_zero_nodes_returns_false() {
        assert!(!should_use_dense(&[], 0));
    }

    #[test]
    fn should_use_dense_full_frontier_is_true() {
        let f = vec![0xFFFF_FFFF_u32; 4];
        assert!(should_use_dense(&f, 128));
    }

    #[test]
    fn should_use_dense_quarter_frontier_at_threshold() {
        // 32 nodes, 8 bits set = 25% (exactly threshold).
        assert!(should_use_dense(&[0xFF_u32], 32));
    }

    #[test]
    fn should_use_dense_just_under_threshold_is_false() {
        // 32 nodes, 7 bits set = ~21%, below 25%.
        assert!(!should_use_dense(&[0x7F_u32], 32));
    }

    #[test]
    fn dense_cutover_rounds_up_without_u32_multiply_overflow() {
        assert_eq!(dense_cutover_nodes(32, 25), 8);
        assert_eq!(dense_cutover_nodes(33, 25), 9);
        assert_eq!(dense_cutover_nodes(u32::MAX, 100), u32::MAX);
    }

    #[test]
    fn selector_roundtrip_common_density_profiles() {
        // Sparse (1% density) → CSR.
        assert!(!should_use_dense(&pack_nodes(&[5], 512), 512));

        // Dense (50% density) → dense.
        let mut f = vec![0_u32; bitset_words(512) as usize];
        for b in 0..256_u32 {
            f[b as usize / 32] |= 1 << (b % 32);
        }
        assert!(should_use_dense(&f, 512));
    }

    #[test]
    fn mode_selector_keeps_ultra_sparse_frontiers_on_queue_path() {
        assert_eq!(
            select_adaptive_traversal_mode(1_000, 10_000, 3, 25),
            AdaptiveTraversalMode::SparseQueue
        );
        assert_eq!(
            select_adaptive_traversal_mode(1_000, 10_000, 250, 25),
            AdaptiveTraversalMode::SparseDense
        );
        assert_eq!(
            select_adaptive_traversal_mode(1_000, 1_000, 100, 25),
            AdaptiveTraversalMode::SparseDense
        );
    }
}
