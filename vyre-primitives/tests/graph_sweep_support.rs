//! Deterministic graph fixtures shared by volume-oracle integration suites.

/// Return the number of words required by a node frontier.
pub(crate) fn bitset_words(node_count: u32) -> usize {
    vyre_primitives::bitset::bitset_words(node_count) as usize
}

/// Advance the deterministic graph fixture generator.
pub(crate) fn next_u32(rng: &mut u64) -> u32 {
    *rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (*rng >> 32) as u32
}

/// Generate a bounded CSR graph, one frontier seed, and an all-kinds mask.
pub(crate) fn generated_csr_frontier(
    seed: u64,
) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let mut rng = seed;
    let node_count = 1 + next_u32(&mut rng) % 96;
    let words = bitset_words(node_count);
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    let mut masks = Vec::new();
    offsets.push(0);
    for _ in 0..node_count {
        let degree = next_u32(&mut rng) % 6;
        for _ in 0..degree {
            targets.push(next_u32(&mut rng) % node_count);
            masks.push(1u32 << (next_u32(&mut rng) % 5));
        }
        offsets.push(targets.len() as u32);
    }
    let mut frontier = vec![0u32; words];
    let start = next_u32(&mut rng) % node_count;
    frontier[(start / 32) as usize] |= 1u32 << (start % 32);
    (node_count, offsets, targets, masks, frontier, u32::MAX)
}
