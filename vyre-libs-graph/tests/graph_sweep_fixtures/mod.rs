//! Deterministic graph fixtures shared by this crate's oracle suites.
//!
//! The frontier word count and the generated-CSR family have one home here, so
//! a case that fails in one suite is the same graph another suite passed.

use crate::wire_words::mix64;

/// Return the number of words required by a node frontier.
pub(crate) fn bitset_words(node_count: u32) -> usize {
    vyre_libs_bitset::bitset::bitset_words(node_count) as usize
}

#[derive(Clone, Debug)]
pub(crate) struct GeneratedCsr {
    pub(crate) edge_offsets: Vec<u32>,
    pub(crate) edge_targets: Vec<u32>,
    pub(crate) edge_kind_mask: Vec<u32>,
}

pub(crate) fn generated_frontier_words(node_count: u32, seed: u64) -> Vec<u32> {
    let words = bitset_words(node_count);
    let mut frontier = Vec::with_capacity(words);
    for word in 0..words {
        frontier.push(mix64(seed ^ (word as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)) as u32);
    }
    if seed & 1 == 0 {
        set_node(&mut frontier, 0);
    }
    if seed & 2 == 0 {
        set_node(&mut frontier, node_count - 1);
    }
    if node_count > 32 && seed & 4 == 0 {
        set_node(&mut frontier, 32);
    }
    frontier
}

pub(crate) fn generated_csr(node_count: u32, seed: u64) -> GeneratedCsr {
    let mut edge_offsets = Vec::with_capacity(node_count as usize + 1);
    let mut edge_targets = Vec::new();
    let mut edge_kind_mask = Vec::new();
    edge_offsets.push(0);
    for src in 0..node_count {
        let row_seed = mix64(seed ^ (src as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
        let degree = (row_seed % 5) as u32;
        for edge_ordinal in 0..degree {
            let edge_seed =
                mix64(row_seed ^ (edge_ordinal as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
            edge_targets.push((edge_seed % u64::from(node_count)) as u32);
            let mask_bit = ((edge_seed >> 17) % 9) as u32;
            edge_kind_mask.push(if mask_bit == 8 { 0 } else { 1u32 << mask_bit });
        }
        edge_offsets.push(edge_targets.len() as u32);
    }
    GeneratedCsr {
        edge_offsets,
        edge_targets,
        edge_kind_mask,
    }
}

pub(crate) fn active_nodes(frontier: &[u32], node_count: u32) -> Vec<u32> {
    (0..node_count)
        .filter(|&node| frontier_has_node(frontier, node))
        .collect()
}

pub(crate) fn max_row_degree(edge_offsets: &[u32]) -> u32 {
    edge_offsets
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .max()
        .unwrap_or(0)
}

pub(crate) fn frontier_has_node(frontier: &[u32], node: u32) -> bool {
    frontier[node as usize / 32] & (1u32 << (node & 31)) != 0
}

pub(crate) fn set_node(frontier: &mut [u32], node: u32) {
    frontier[node as usize / 32] |= 1u32 << (node & 31);
}
