//! Generated adversarial graph-fixpoint tests for CSR traversal and persistent BFS.

#![cfg(all(feature = "graph", feature = "bitset"))]

use vyre_libs::bitset::bitset_words;
use vyre_libs::graph::csr_closure_inputs::{CsrClosureInputs, CsrGraphView};
use vyre_libs::graph::persistent_bfs;
use vyre_reference::composition_witness::csr_closure_with_step_hook_witness;

fn next_u32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn generated_graph(seed: u32) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>) {
    let node_count = 1 + (seed % 97);
    let mut state = seed ^ 0xC0FF_EE11;
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    let mut masks = Vec::new();
    offsets.push(0);
    for src in 0..node_count {
        let degree = (next_u32(&mut state) % 4) as usize;
        for edge_index in 0..degree {
            let raw = next_u32(&mut state);
            let target = raw.wrapping_add(src.rotate_left((edge_index % 31) as u32)) % node_count;
            let kind = 1u32 << ((raw >> 11) % 4);
            targets.push(target);
            masks.push(kind);
        }
        offsets.push(targets.len() as u32);
    }
    (node_count, offsets, targets, masks)
}

fn generated_frontier(seed: u32, node_count: u32) -> Vec<u32> {
    let words = bitset_words(node_count) as usize;
    let mut state = seed ^ 0x5151_9EED;
    let mut frontier = vec![0u32; words.max(1)];
    for node in 0..node_count {
        if next_u32(&mut state) & 0b111 == 0 {
            frontier[(node / 32) as usize] |= 1u32 << (node % 32);
        }
    }
    if frontier.iter().all(|&word| word == 0) {
        let node = seed % node_count;
        frontier[(node / 32) as usize] |= 1u32 << (node % 32);
    }
    if node_count % 32 != 0 {
        let valid_bits = node_count % 32;
        let mask = (1u32 << valid_bits) - 1;
        let last = frontier.len() - 1;
        frontier[last] &= mask;
    }
    frontier
}

fn allow_mask(seed: u32) -> u32 {
    match seed % 7 {
        0 => 0,
        1 => 0b0001,
        2 => 0b0010,
        3 => 0b0100,
        4 => 0b1000,
        5 => 0b0101,
        _ => 0xFFFF_FFFF,
    }
}

fn reference_closure(inputs: CsrClosureInputs<'_>, frontier: &[u32]) -> Vec<u32> {
    csr_closure_with_step_hook_witness(
        inputs.graph.node_count,
        inputs.graph.edge_offsets,
        inputs.graph.edge_targets,
        inputs.graph.edge_kind_mask,
        inputs.allow_mask,
        inputs.max_iters,
        frontier,
        |_| {},
    )
}

fn independent_queue_closure(inputs: CsrClosureInputs<'_>, frontier: &[u32]) -> Vec<u32> {
    let node_count = inputs.graph.node_count;
    let mut closure = frontier.to_vec();
    closure.resize(bitset_words(node_count) as usize, 0);
    let mut queue = std::collections::VecDeque::new();
    for node in 0..node_count {
        if closure[(node / 32) as usize] & (1 << (node % 32)) != 0 {
            queue.push_back(node);
        }
    }
    while let Some(source) = queue.pop_front() {
        let start = inputs.graph.edge_offsets[source as usize] as usize;
        let end = inputs.graph.edge_offsets[source as usize + 1] as usize;
        for edge in start..end {
            if inputs.graph.edge_kind_mask[edge] & inputs.allow_mask == 0 {
                continue;
            }
            let destination = inputs.graph.edge_targets[edge];
            let word = (destination / 32) as usize;
            let bit = 1 << (destination % 32);
            if closure[word] & bit == 0 {
                closure[word] |= bit;
                queue.push_back(destination);
            }
        }
    }
    closure
}

#[test]
fn persistent_bfs_matches_csr_forward_closure_for_generated_graphs() {
    for seed in 0..8192u32 {
        let (node_count, offsets, targets, masks) = generated_graph(seed);
        let frontier = generated_frontier(seed, node_count);
        let allow = allow_mask(seed);
        let max_iters = node_count.saturating_add(2);

        let inputs = CsrClosureInputs {
            graph: CsrGraphView {
                node_count,
                edge_offsets: &offsets,
                edge_targets: &targets,
                edge_kind_mask: &masks,
            },
            allow_mask: allow,
            max_iters,
        };
        let via_csr = reference_closure(inputs, &frontier);
        let via_bfs = independent_queue_closure(inputs, &frontier);

        assert_eq!(via_bfs, via_csr, "seed {seed}");
    }
}

#[test]
fn persistent_bfs_generated_fixpoints_are_idempotent() {
    for seed in 8192..12_288u32 {
        let (node_count, offsets, targets, masks) = generated_graph(seed);
        let frontier = generated_frontier(seed, node_count);
        let allow = allow_mask(seed);
        let max_iters = node_count.saturating_add(2);

        let inputs = CsrClosureInputs {
            graph: CsrGraphView {
                node_count,
                edge_offsets: &offsets,
                edge_targets: &targets,
                edge_kind_mask: &masks,
            },
            allow_mask: allow,
            max_iters,
        };
        let closure = reference_closure(inputs, &frontier);
        let closure_again = reference_closure(inputs, &closure);
        let second_changed = u32::from(closure_again != closure);

        assert_eq!(closure_again, closure, "idempotent closure seed {seed}");
        assert_eq!(
            second_changed, 0,
            "fixpoint must report no new bits at seed {seed}"
        );
    }
}

#[test]
fn generated_validation_rejects_corrupted_csr_shapes() {
    for seed in 12_288..14_336u32 {
        let (node_count, mut offsets, mut targets, mut masks) = generated_graph(seed);
        match seed % 4 {
            0 => {
                offsets.pop();
            }
            1 if offsets.len() > 2 => {
                offsets[1] = u32::MAX;
                offsets[2] = 0;
            }
            2 => {
                targets.push(node_count);
                if let Some(last) = offsets.last_mut() {
                    *last = targets.len() as u32;
                }
                masks.push(1);
            }
            _ => {
                masks.push(1);
            }
        }

        let err = persistent_bfs::validate_persistent_bfs_graph_layout(
            node_count, &offsets, &targets, &masks,
        )
        .expect_err("corrupted generated graph should fail validation at seed {seed}");
        assert!(
            err.contains("Fix:"),
            "validation error must be actionable at seed {seed}: {err}"
        );
    }
}
