//! Packed-frontier and dense-adjacency fixtures shared by the adaptive
//! traversal tests.

use crate::bitset::bitset_words;

pub(super) fn pack_nodes(bits: &[u32], node_count: u32) -> Vec<u32> {
    let mut buf = vec![0_u32; bitset_words(node_count) as usize];
    for &b in bits {
        buf[(b as usize) / 32] |= 1 << (b % 32);
    }
    buf
}

pub(super) fn build_dense_adj(edges: &[(u32, u32)], node_count: u32) -> Vec<u32> {
    let words = bitset_words(node_count) as usize;
    let mut rows = vec![0_u32; (node_count as usize) * words];
    for &(src, dst) in edges {
        let idx = (dst as usize) * words + (src as usize) / 32;
        rows[idx] |= 1 << (src % 32);
    }
    rows
}
