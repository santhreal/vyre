//! Device-sharded forward frontier expansion. W3-5 `graph-frontier-device-shards`.
//!
//! A forward frontier step (`csr_frontier_step`) expands ONLY the vertices set in
//! `frontier_in`: for each active vertex it marks that vertex's out-neighbours in
//! `frontier_out`. Because each active vertex is expanded independently, the active
//! frontier can be PARTITIONED across device shards by vertex ownership, shard `s`
//! expands only the frontier vertices it owns, and OR-ing the shards' `frontier_out`
//! bitsets back together reproduces the single-device expansion EXACTLY (the partition
//! is disjoint and complete, so every active vertex is expanded by exactly one shard
//! and no out-edge is dropped or double-counted).
//!
//! This module owns that partition + merge, the graph-specific decomposition that
//! makes frontier expansion shardable across devices. On a real multi-GPU host each
//! shard's expansion runs on a distinct peer device (dispatched concurrently with the
//! same per-device-thread pattern proven for byte-range scan sharding in
//! `vyre_libs::scan::paged_corpus::scan_sharded_core`), and the cross-shard merge is a
//! peer-transfer bitwise-OR reduce; on a single device the same program runs per shard
//! and the merge is a host OR. Only the wall-clock parallel SPEEDUP and the on-device
//! peer-transfer merge need a second physical GPU, the decomposition and its
//! exactness are backend-agnostic and proven here on any single backend, so device
//! sharding changes no reachability bit.

use crate::bitset::bitset_words;

/// Partition `frontier_in`'s active vertices across `shard_count` contiguous vertex
/// ranges. Returns one masked frontier bitset per shard: shard `s` keeps only the bits
/// of `frontier_in` for vertices it owns, zeroed elsewhere. Vertex `v` is owned by
/// shard `v * shard_count / node_count`, so the shards partition the vertex id space
/// into disjoint contiguous ranges.
///
/// # Errors
/// Fails closed if `shard_count` is zero, or if `frontier_in` is not exactly
/// `bitset_words(node_count)` words (a mis-sized frontier is an ABI break, never a
/// silent truncation).
pub fn partition_frontier_by_vertex(
    frontier_in: &[u32],
    node_count: u32,
    shard_count: usize,
) -> Result<Vec<Vec<u32>>, String> {
    if shard_count == 0 {
        return Err(
            "csr_frontier_shard: shard_count must be >= 1. Fix: pass at least one device shard to partition the frontier across.".to_string(),
        );
    }
    let words = bitset_words(node_count) as usize;
    if frontier_in.len() != words {
        return Err(format!(
            "csr_frontier_shard: frontier_in has {} word(s) but node_count {node_count} needs {words} (bitset_words). Fix: size the frontier bitset to the graph before sharding.",
            frontier_in.len()
        ));
    }
    let mut shards = vec![vec![0u32; words]; shard_count];
    for v in 0..node_count {
        let word = (v >> 5) as usize;
        let bit = 1u32 << (v & 31);
        if frontier_in[word] & bit != 0 {
            // Contiguous vertex ranges: shard = floor(v * shard_count / node_count),
            // clamped so the last vertex can never index past the final shard.
            let shard = ((u64::from(v) * shard_count as u64) / u64::from(node_count)) as usize;
            let shard = shard.min(shard_count - 1);
            shards[shard][word] |= bit;
        }
    }
    Ok(shards)
}

/// Merge per-shard `frontier_out` bitsets into one by bitwise OR, the cross-shard
/// frontier/visited merge each expansion level performs (a peer-transfer reduce on real
/// multi-GPU, a host OR here). OR is associative and commutative, so the merged result
/// is independent of shard order.
///
/// # Errors
/// Fails closed if any shard's `frontier_out` is not exactly `words` long.
pub fn merge_frontier_out(shards: &[Vec<u32>], words: usize) -> Result<Vec<u32>, String> {
    let mut merged = vec![0u32; words];
    for (index, shard) in shards.iter().enumerate() {
        if shard.len() != words {
            return Err(format!(
                "csr_frontier_shard: shard {index} frontier_out has {} word(s), expected {words}. Fix: every shard must expand into a frontier bitset sized to the whole graph.",
                shard.len()
            ));
        }
        for (slot, value) in merged.iter_mut().zip(shard.iter()) {
            *slot |= *value;
        }
    }
    Ok(merged)
}

/// Run one forward frontier-expansion level SHARDED across `shard_count` device shards:
/// partition the active frontier by vertex ownership, expand each shard's owned subset
/// via `expand` (dispatched on that shard's device, the same GPU frontier-step
/// program), and OR the per-shard `frontier_out` bitsets into the next frontier.
///
/// `expand(shard_index, masked_frontier_in) -> frontier_out` runs one device's
/// expansion; a caller with N real devices dispatches these concurrently (one thread
/// per device, the pattern proven in `scan_sharded_core`). The result is identical to a
/// single-device expansion of the whole `frontier_in`: proven by the parity tests 
/// so device sharding never changes the reachable set.
///
/// # Errors
/// Fails closed on a zero shard count, a mis-sized `frontier_in`, an `expand` error, or
/// an `expand` result that is not sized to the whole graph.
pub fn frontier_step_sharded(
    frontier_in: &[u32],
    node_count: u32,
    shard_count: usize,
    mut expand: impl FnMut(usize, &[u32]) -> Result<Vec<u32>, String>,
) -> Result<Vec<u32>, String> {
    let words = bitset_words(node_count) as usize;
    let partitions = partition_frontier_by_vertex(frontier_in, node_count, shard_count)?;
    let mut outputs = Vec::with_capacity(shard_count);
    for (shard_index, masked_frontier_in) in partitions.iter().enumerate() {
        let out = expand(shard_index, masked_frontier_in)?;
        if out.len() != words {
            return Err(format!(
                "csr_frontier_shard: shard {shard_index} expand returned a {}-word frontier_out, expected {words} for node_count {node_count}. Fix: the per-shard expansion must write a full-graph frontier bitset.",
                out.len()
            ));
        }
        outputs.push(out);
    }
    merge_frontier_out(&outputs, words)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::csr_frontier_step::{csr_frontier_step_program, CsrFrontierStepKind};
    use crate::graph::program_graph::ProgramGraphShape;
    use vyre_reference::value::Value;

    /// A tiny CSR graph for the tests: 8 vertices, directed out-edges. Chosen so edges
    /// cross the shard boundaries the partition creates (e.g. 4→5, 1→6), which is the
    /// case device sharding must get right (a vertex on one shard reaching a vertex on
    /// another).
    fn sample_csr() -> (u32, Vec<u32>, Vec<u32>) {
        // out-adjacency: 0->{1,2}, 1->{6}, 2->{3}, 3->{4}, 4->{5}, 5->{}, 6->{7}, 7->{}
        let adjacency: [&[u32]; 8] = [&[1, 2], &[6], &[3], &[4], &[5], &[], &[7], &[]];
        let node_count = adjacency.len() as u32;
        let mut edge_offsets = Vec::with_capacity(adjacency.len() + 1);
        let mut edge_targets = Vec::new();
        edge_offsets.push(0u32);
        for outs in adjacency {
            edge_targets.extend_from_slice(outs);
            edge_offsets.push(edge_targets.len() as u32);
        }
        (node_count, edge_offsets, edge_targets)
    }

    fn empty_bitset(node_count: u32) -> Vec<u32> {
        vec![0u32; bitset_words(node_count) as usize]
    }

    fn set_bit(bitset: &mut [u32], v: u32) {
        bitset[(v >> 5) as usize] |= 1u32 << (v & 31);
    }

    fn bit_set(bitset: &[u32], v: u32) -> bool {
        bitset[(v >> 5) as usize] & (1u32 << (v & 31)) != 0
    }

    /// Pure-Rust single-hop forward expansion oracle (independent of the GPU program):
    /// mark every out-neighbour of every active vertex.
    fn cpu_expand(
        frontier_in: &[u32],
        node_count: u32,
        edge_offsets: &[u32],
        edge_targets: &[u32],
    ) -> Vec<u32> {
        let mut out = empty_bitset(node_count);
        for v in 0..node_count {
            if bit_set(frontier_in, v) {
                let lo = edge_offsets[v as usize] as usize;
                let hi = edge_offsets[v as usize + 1] as usize;
                for &dst in &edge_targets[lo..hi] {
                    if dst < node_count {
                        set_bit(&mut out, dst);
                    }
                }
            }
        }
        out
    }

    #[test]
    fn partition_is_disjoint_and_complete_over_active_vertices() {
        let (node_count, _, _) = sample_csr();
        let mut frontier = empty_bitset(node_count);
        for v in [0u32, 1, 3, 4, 6] {
            set_bit(&mut frontier, v);
        }
        let parts = partition_frontier_by_vertex(&frontier, node_count, 3).expect("partition");
        assert_eq!(parts.len(), 3);
        // Every active vertex appears in EXACTLY one shard, and no inactive vertex
        // appears anywhere (a disjoint, complete partition).
        for v in 0..node_count {
            let count = parts.iter().filter(|p| bit_set(p, v)).count();
            if bit_set(&frontier, v) {
                assert_eq!(
                    count, 1,
                    "active vertex {v} must be owned by exactly one shard"
                );
            } else {
                assert_eq!(count, 0, "inactive vertex {v} must not appear in any shard");
            }
        }
        // Contiguous ownership: shard 0 owns the low vertex ids, shard 2 the high ones.
        assert!(bit_set(&parts[0], 0), "shard 0 owns vertex 0");
        assert!(bit_set(&parts[2], 6), "shard 2 owns vertex 6");
    }

    #[test]
    fn sharded_expansion_equals_single_device_across_shard_counts() {
        let (node_count, edge_offsets, edge_targets) = sample_csr();
        let mut frontier = empty_bitset(node_count);
        // Active set spans shard boundaries with cross-shard edges (1->6, 4->5).
        for v in [0u32, 1, 4] {
            set_bit(&mut frontier, v);
        }
        let single = cpu_expand(&frontier, node_count, &edge_offsets, &edge_targets);

        // Hand oracle: 0->{1,2}, 1->{6}, 4->{5} => {1,2,5,6}.
        let mut expected = empty_bitset(node_count);
        for v in [1u32, 2, 5, 6] {
            set_bit(&mut expected, v);
        }
        assert_eq!(
            single, expected,
            "single-device expansion must match the hand oracle"
        );

        for shard_count in 1..=5usize {
            let sharded = frontier_step_sharded(&frontier, node_count, shard_count, |_, masked| {
                Ok(cpu_expand(masked, node_count, &edge_offsets, &edge_targets))
            })
            .expect("sharded expansion");
            assert_eq!(
                sharded, single,
                "sharded expansion over {shard_count} shard(s) must equal the single-device expansion"
            );
        }
    }

    #[test]
    fn sharding_fails_closed_on_bad_inputs() {
        let (node_count, _, _) = sample_csr();
        let frontier = empty_bitset(node_count);
        assert!(
            partition_frontier_by_vertex(&frontier, node_count, 0).is_err(),
            "zero shard count must fail closed"
        );
        // node_count=8 needs bitset_words(8)=1 word; a 5-word frontier is mis-sized.
        assert_eq!(
            bitset_words(node_count),
            1,
            "sample graph is one bitset word"
        );
        assert!(
            partition_frontier_by_vertex(&[0u32; 5], node_count, 2).is_err(),
            "a mis-sized frontier must fail closed, not silently truncate"
        );
        // An expand that returns a wrong-sized frontier_out is rejected.
        let bad = frontier_step_sharded(&frontier, node_count, 2, |_, _| Ok(vec![0u32; 999]));
        assert!(bad.is_err(), "a wrong-sized shard output must fail closed");
    }

    /// The decomposition composes with the REAL GPU `csr_frontier_step` program (run
    /// through the reference interpreter): sharding the frontier and OR-merging the
    /// per-shard program outputs equals a single-device run of the program over the
    /// whole frontier. This proves the sharding is exact against the actual expansion
    /// semantics, not just a Rust re-implementation of them.
    #[test]
    fn sharded_expansion_equals_single_device_through_real_frontier_step_program() {
        let (node_count, edge_offsets, edge_targets) = sample_csr();
        let edge_count = edge_targets.len() as u32;
        let allow_mask = 1u32;

        // Drive one forward frontier-step through the reference interpreter for a given
        // frontier_in bitset, returning frontier_out.
        let run_program = |frontier_in: &[u32]| -> Vec<u32> {
            let shape = ProgramGraphShape::new(node_count, edge_count);
            let program = csr_frontier_step_program(
                "csr_frontier_shard_test",
                CsrFrontierStepKind::Forward,
                shape,
                "frontier_in",
                "frontier_out",
                allow_mask,
            );
            let words = bitset_words(node_count) as usize;
            let to_bytes = |w: &[u32]| w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
            // Buffer order: pg_nodes, pg_edge_offsets, pg_edge_targets, pg_edge_kind_mask,
            // pg_node_tags, frontier_in, frontier_out (the sole ReadWrite output).
            let values: Vec<Value> = vec![
                Value::from(to_bytes(&vec![0u32; node_count as usize])), // pg_nodes (unused)
                Value::from(to_bytes(&edge_offsets)),
                Value::from(to_bytes(&edge_targets)),
                Value::from(to_bytes(&vec![allow_mask; edge_count.max(1) as usize])), // kind mask: all edges allowed
                Value::from(to_bytes(&vec![0u32; node_count as usize])), // pg_node_tags (unused)
                Value::from(to_bytes(frontier_in)),
                Value::from(to_bytes(&vec![0u32; words])), // frontier_out, zero-init
            ];
            let outputs = vyre_reference::reference_eval(&program, &values)
                .expect("csr_frontier_step reference program must evaluate");
            let bytes = outputs[0].to_bytes();
            assert_eq!(
                bytes.len(),
                words * 4,
                "frontier_out must be the bitset-word output buffer"
            );
            bytes
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        let mut frontier = empty_bitset(node_count);
        for v in [0u32, 1, 4] {
            set_bit(&mut frontier, v);
        }

        let single = run_program(&frontier);
        // Sanity: the real program reproduces the hand oracle {1,2,5,6}.
        let mut expected = empty_bitset(node_count);
        for v in [1u32, 2, 5, 6] {
            set_bit(&mut expected, v);
        }
        assert_eq!(
            single, expected,
            "the real frontier-step program must expand {{0,1,4}} to {{1,2,5,6}}"
        );

        for shard_count in 1..=4usize {
            let sharded = frontier_step_sharded(&frontier, node_count, shard_count, |_, masked| {
                Ok(run_program(masked))
            })
            .expect("sharded expansion via the real program");
            assert_eq!(
                sharded, single,
                "sharding across {shard_count} shard(s) through the real frontier-step program must equal the single-device run"
            );
        }
    }
}
