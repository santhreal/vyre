//! Property gates for device-sharded forward frontier expansion (W3-5
//! graph-frontier-device-shards). The load-bearing invariant: sharding the active
//! frontier across ANY number of device shards and OR-merging the per-shard outputs
//! must reproduce the single-device expansion EXACTLY, for ANY graph and frontier.

#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::bitset::bitset_words;
use vyre_primitives::graph::csr_frontier_shard::{
    frontier_step_sharded, merge_frontier_out, partition_frontier_by_vertex,
};

/// splitmix64, a deterministic, seedable generator so each proptest case builds a
/// reproducible random graph + frontier without pulling in an RNG crate.
fn mix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Build a random directed CSR graph (edge_offsets, edge_targets) over `node_count`
/// vertices from `seed`: each vertex gets 0..=4 out-edges to random targets.
fn generated_csr(node_count: u32, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
    let mut edge_offsets = Vec::with_capacity(node_count as usize + 1);
    let mut edge_targets = Vec::new();
    edge_offsets.push(0u32);
    for _ in 0..node_count {
        let degree = (mix(&mut state) % 5) as u32; // 0..=4
        for _ in 0..degree {
            let dst = (mix(&mut state) % u64::from(node_count)) as u32;
            edge_targets.push(dst);
        }
        edge_offsets.push(edge_targets.len() as u32);
    }
    (edge_offsets, edge_targets)
}

/// A random frontier bitset over `node_count` vertices from `seed` (~1/3 active).
fn generated_frontier(node_count: u32, seed: u64) -> Vec<u32> {
    let mut state = seed ^ 0x7F4A_7C15_9E37_79B9;
    let mut frontier = vec![0u32; bitset_words(node_count) as usize];
    for v in 0..node_count {
        if mix(&mut state) % 3 == 0 {
            frontier[(v >> 5) as usize] |= 1u32 << (v & 31);
        }
    }
    frontier
}

fn bit_set(bitset: &[u32], v: u32) -> bool {
    bitset[(v >> 5) as usize] & (1u32 << (v & 31)) != 0
}

/// Independent single-device forward expansion oracle: mark every out-neighbour of
/// every active vertex.
fn cpu_expand(
    frontier_in: &[u32],
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
) -> Vec<u32> {
    let mut out = vec![0u32; bitset_words(node_count) as usize];
    for v in 0..node_count {
        if bit_set(frontier_in, v) {
            let lo = edge_offsets[v as usize] as usize;
            let hi = edge_offsets[v as usize + 1] as usize;
            for &dst in &edge_targets[lo..hi] {
                if dst < node_count {
                    out[(dst >> 5) as usize] |= 1u32 << (dst & 31);
                }
            }
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    /// For ANY graph, ANY frontier, and ANY shard count, the sharded expansion equals
    /// the single-device expansion (device sharding never changes the reachable set).
    #[test]
    fn sharded_expansion_always_equals_single_device(
        node_count in 1u32..=2048,
        graph_seed in any::<u64>(),
        frontier_seed in any::<u64>(),
        shard_count in 1usize..=8,
    ) {
        let (edge_offsets, edge_targets) = generated_csr(node_count, graph_seed);
        let frontier = generated_frontier(node_count, frontier_seed);

        let single = cpu_expand(&frontier, node_count, &edge_offsets, &edge_targets);
        let sharded = frontier_step_sharded(&frontier, node_count, shard_count, |_, masked| {
            Ok(cpu_expand(masked, node_count, &edge_offsets, &edge_targets))
        })
        .expect("sharded expansion must succeed on a well-formed frontier");

        prop_assert_eq!(
            sharded,
            single,
            "sharded expansion over {} shard(s) diverged from single-device on a {}-vertex graph",
            shard_count,
            node_count
        );
    }

    /// The vertex partition is always disjoint and complete over the active set: every
    /// active vertex is owned by exactly one shard, every inactive vertex by none.
    #[test]
    fn partition_is_always_disjoint_and_complete(
        node_count in 1u32..=2048,
        frontier_seed in any::<u64>(),
        shard_count in 1usize..=8,
    ) {
        let frontier = generated_frontier(node_count, frontier_seed);
        let parts = partition_frontier_by_vertex(&frontier, node_count, shard_count)
            .expect("partition must succeed on a well-formed frontier");
        prop_assert_eq!(parts.len(), shard_count);
        for v in 0..node_count {
            let owners = parts.iter().filter(|p| bit_set(p, v)).count();
            if bit_set(&frontier, v) {
                prop_assert_eq!(owners, 1, "active vertex {} must have exactly one owner", v);
            } else {
                prop_assert_eq!(owners, 0, "inactive vertex {} must have no owner", v);
            }
        }
    }

    /// The cross-shard OR-merge is order-independent: reversing the shard order yields
    /// the identical merged frontier (OR is associative and commutative).
    #[test]
    fn merge_is_order_independent(
        node_count in 1u32..=2048,
        frontier_seed in any::<u64>(),
        shard_count in 1usize..=8,
    ) {
        let words = bitset_words(node_count) as usize;
        let frontier = generated_frontier(node_count, frontier_seed);
        let parts = partition_frontier_by_vertex(&frontier, node_count, shard_count)
            .expect("partition");
        let mut reversed = parts.clone();
        reversed.reverse();
        let forward = merge_frontier_out(&parts, words).expect("merge forward");
        let backward = merge_frontier_out(&reversed, words).expect("merge reversed");
        prop_assert_eq!(&forward, &backward, "OR-merge must be shard-order-independent");
        // And the merge of the partition reconstructs exactly the original frontier
        // (the partition loses nothing (every active bit survives)).
        prop_assert_eq!(forward, frontier, "partition + merge must round-trip the frontier");
    }
}
