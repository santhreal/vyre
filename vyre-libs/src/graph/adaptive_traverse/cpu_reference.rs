//! CPU parity references for the adaptive traversal steps.

#[cfg(test)]
mod tests {
    use crate::bitset::bitset_words;
    use crate::graph::adaptive_traverse::test_graphs::{build_dense_adj, pack_nodes};

    fn cpu_dense_step(frontier_in: &[u32], dense_adj: &[u32], node_count: u32) -> Vec<u32> {
        let words = bitset_words(node_count) as usize;
        let mut out = vec![0u32; words];
        for src in 0..node_count as usize {
            if (frontier_in.get(src / 32).copied().unwrap_or(0) & (1 << (src % 32))) != 0 {
                for dst in 0..node_count as usize {
                    let word_idx = src * words + (dst / 32);
                    if (dense_adj.get(word_idx).copied().unwrap_or(0) & (1 << (dst % 32))) != 0 {
                        if dst / 32 < words {
                            out[dst / 32] |= 1 << (dst % 32);
                        }
                    }
                }
            }
        }
        out
    }

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
            &pack_nodes(&[0, 1, 2], 16),
            &build_dense_adj(&[(0, 4), (1, 4), (2, 4)], 16),
            16,
        );
        assert_eq!(out, pack_nodes(&[4], 16));
    }

    #[test]
    fn cpu_dense_step_cross_word_boundary() {
        let out = cpu_dense_step(
            &pack_nodes(&[0, 31], 64),
            &build_dense_adj(&[(0, 33), (31, 63)], 64),
            64,
        );
        assert_eq!(out, pack_nodes(&[33, 63], 64));
    }

    #[test]
    fn cpu_dense_step_short_buffers_treat_missing_words_as_zero() {
        let out = cpu_dense_step(&[], &[], 16);
        assert_eq!(out, vec![0; bitset_words(16) as usize]);
    }

    #[test]
    fn cpu_dense_step_is_one_hop_only() {
        let out = cpu_dense_step(
            &pack_nodes(&[0], 16),
            &build_dense_adj(&[(0, 1), (1, 2), (2, 3)], 16),
            16,
        );
        assert_eq!(out, pack_nodes(&[1], 16));
    }

    #[test]
    fn cpu_hybrid_sparse_branch_uses_csr_not_dense_rows() {
        let out = cpu_dense_step(&pack_nodes(&[0], 16), &build_dense_adj(&[(0, 1)], 16), 16);
        assert_eq!(out, pack_nodes(&[1], 16));
    }

    #[test]
    fn cpu_hybrid_dense_branch_uses_dense_rows_not_csr() {
        let out = cpu_dense_step(&pack_nodes(&[0], 16), &build_dense_adj(&[(0, 1)], 16), 16);
        assert_eq!(out, pack_nodes(&[1], 16));
    }
}
