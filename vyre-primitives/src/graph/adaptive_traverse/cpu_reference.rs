//! CPU parity references for the adaptive traversal steps.

use super::four_russians::{four_russians_dense_lut_from_adj_rows, four_russians_source_tile_count};
use super::mode_selection::should_use_dense_with_popcount;
use crate::bitset::bitset_words;

/// CPU reference for the dense step. `frontier_in` is a packed
/// bitset over `node_count` nodes; `adj_rows_dense` is the reverse
/// adjacency laid out as `node_count × bitset_words(node_count)`.
#[must_use]
pub fn cpu_dense_step(frontier_in: &[u32], adj_rows_dense: &[u32], node_count: u32) -> Vec<u32> {
    let words = bitset_words(node_count) as usize;

    let mut out = vec![0_u32; words];
    for d in 0..node_count as usize {
        let row_start = d * words;
        let mut hit: u32 = 0;
        for w in 0..words {
            let adj = adj_rows_dense.get(row_start + w).copied().unwrap_or(0);
            let frontier = frontier_in.get(w).copied().unwrap_or(0);
            hit |= adj & frontier;
        }
        if hit != 0 {
            out[d / 32] |= 1 << (d % 32);
        }
    }
    out
}

/// CPU reference for graph-level Four-Russians dense traversal.
#[must_use]
pub fn cpu_four_russians_dense_step(
    frontier_in: &[u32],
    adj_rows_dense: &[u32],
    node_count: u32,
) -> Result<Vec<u32>, String> {
    let lut = four_russians_dense_lut_from_adj_rows(node_count, adj_rows_dense)?;
    Ok(crate::bitset::four_russians::dense_matvec_cpu_ref(
        frontier_in,
        &lut,
        four_russians_source_tile_count(node_count),
        bitset_words(node_count),
    ))
}

/// CPU reference for the adaptive sparse/dense step.
#[must_use]
pub fn cpu_sparse_dense_step(
    frontier_in: &[u32],
    frontier_popcount: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    adj_rows_dense: &[u32],
    node_count: u32,
    allow_mask: u32,
    dense_threshold_pct: u32,
) -> Vec<u32> {
    if should_use_dense_with_popcount(frontier_popcount, node_count, dense_threshold_pct) {
        return cpu_dense_step(frontier_in, adj_rows_dense, node_count);
    }

    let words = bitset_words(node_count) as usize;
    let mut out = vec![0_u32; words];
    for src in 0..node_count as usize {
        let word_idx = src / 32;
        let bit_mask = 1_u32 << (src % 32);
        if frontier_in.get(word_idx).copied().unwrap_or(0) & bit_mask == 0 {
            continue;
        }
        let edge_start = edge_offsets.get(src).copied().unwrap_or(0) as usize;
        let edge_end = edge_offsets
            .get(src + 1)
            .copied()
            .unwrap_or(edge_start as u32) as usize;
        for e in edge_start..edge_end {
            if edge_kind_mask.get(e).copied().unwrap_or(0) & allow_mask == 0 {
                continue;
            }
            let Some(dst) = edge_targets.get(e).copied() else {
                continue;
            };
            if dst < node_count {
                out[dst as usize / 32] |= 1_u32 << (dst % 32);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::adaptive_traverse::test_graphs::{build_dense_adj, pack_nodes};

    #[test]
    fn cpu_dense_step_empty_frontier_produces_empty() {
        let frontier_in = pack_nodes(&[], 16);
        let adj = build_dense_adj(&[(0, 1), (1, 2)], 16);
        let out = cpu_dense_step(&frontier_in, &adj, 16);
        assert_eq!(out, vec![0; bitset_words(16) as usize]);
    }

    #[test]
    fn cpu_dense_step_single_edge() {
        let out = cpu_dense_step(&pack_nodes(&[0], 16), &build_dense_adj(&[(0, 1)], 16), 16);
        assert_eq!(out, pack_nodes(&[1], 16));
    }

    #[test]
    fn cpu_dense_step_fanout() {
        let out = cpu_dense_step(
            &pack_nodes(&[0], 16),
            &build_dense_adj(&[(0, 1), (0, 2), (0, 5)], 16),
            16,
        );
        assert_eq!(out, pack_nodes(&[1, 2, 5], 16));
    }

    #[test]
    fn cpu_dense_step_fanin() {
        let out = cpu_dense_step(
            &pack_nodes(&[1, 2], 16),
            &build_dense_adj(&[(1, 3), (2, 3), (4, 3)], 16),
            16,
        );
        assert_eq!(out, pack_nodes(&[3], 16));
    }

    #[test]
    fn cpu_dense_step_cross_word_boundary() {
        // 70 nodes → 3 words. Edge src=5 (word 0) → dst=65 (word 2).
        let out = cpu_dense_step(&pack_nodes(&[5], 70), &build_dense_adj(&[(5, 65)], 70), 70);
        assert_eq!(out, pack_nodes(&[65], 70));
    }

    #[test]
    fn cpu_dense_step_short_buffers_treat_missing_words_as_zero() {
        let out = cpu_dense_step(&[1], &[], 16);
        assert!(out.iter().all(|&word| word == 0));
    }

    #[test]
    fn cpu_dense_step_is_one_hop_only() {
        // Single invocation is one hop. 0 → 1 → 2 → 3; seeded with
        // {0} yields {1}, not the full closure.
        let out = cpu_dense_step(
            &pack_nodes(&[0], 16),
            &build_dense_adj(&[(0, 1), (1, 2), (2, 3)], 16),
            16,
        );
        assert_eq!(out, pack_nodes(&[1], 16));
    }

    #[test]
    fn cpu_hybrid_sparse_branch_uses_csr_not_dense_rows() {
        let frontier = pack_nodes(&[0], 8);
        let offsets = vec![0, 1, 1, 1, 1, 1, 1, 1, 1];
        let targets = vec![1];
        let kinds = vec![1];
        let dense = build_dense_adj(&[(0, 2)], 8);
        let out = cpu_sparse_dense_step(&frontier, 1, &offsets, &targets, &kinds, &dense, 8, 1, 50);
        assert_eq!(out, pack_nodes(&[1], 8));
    }

    #[test]
    fn cpu_hybrid_dense_branch_uses_dense_rows_not_csr() {
        let frontier = pack_nodes(&[0, 1, 2, 3], 8);
        let offsets = vec![0, 1, 1, 1, 1, 1, 1, 1, 1];
        let targets = vec![1];
        let kinds = vec![1];
        let dense = build_dense_adj(&[(0, 5)], 8);
        let out = cpu_sparse_dense_step(&frontier, 4, &offsets, &targets, &kinds, &dense, 8, 1, 50);
        assert_eq!(out, pack_nodes(&[5], 8));
    }
}
