//! Property gates for row-strided queue-driven CSR traversal.

#![cfg(feature = "graph")]

mod wire_words;
use wire_words::{mix64, queue_forward_oracle};

use proptest::prelude::*;
use vyre_libs::graph::csr_frontier_queue::validate_csr_queue_graph;
use vyre_libs::graph::csr_queue_strided::CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE;
use vyre_reference::composition_witness::csr_queue_strided_forward_witness;

#[derive(Clone, Debug)]
struct GeneratedSkewedCsr {
    edge_offsets: Vec<u32>,
    edge_targets: Vec<u32>,
    edge_kind_mask: Vec<u32>,
    hub: u32,
}
fn forward_witness(
    graph: &GeneratedSkewedCsr,
    queue: &[u32],
    queue_len: u32,
    node_count: u32,
    allow_mask: u32,
) -> Vec<u32> {
    csr_queue_strided_forward_witness(
        queue,
        queue_len,
        &graph.edge_offsets,
        &graph.edge_targets,
        &graph.edge_kind_mask,
        node_count,
        allow_mask,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn strided_queue_cpu_matches_independent_oracle_on_skewed_graphs(
        node_count in 1u32..=512,
        graph_seed in any::<u64>(),
        queue_seed in any::<u64>(),
        queue_slots in 0usize..=128,
        queue_len_extra in 0u32..=96,
        allow_mask in any::<u32>(),
    ) {
        let graph = generated_skewed_csr(node_count, graph_seed);
        let queue = generated_active_queue(node_count, graph.hub, queue_slots, queue_seed);
        let queue_len = (queue.len() as u32).saturating_add(queue_len_extra);
        let allow_mask = allow_mask | 1;
        let expected = queue_forward_oracle(
            &queue,
            queue_len,
            &graph.edge_offsets,
            &graph.edge_targets,
            &graph.edge_kind_mask,
            node_count,
            allow_mask,
        );

        let actual = forward_witness(&graph, &queue, queue_len, node_count, allow_mask);

        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn strided_queue_witness_repeated_calls_do_not_retain_prior_state(
        node_count in 1u32..=384,
        graph_seed in any::<u64>(),
        queue_seed in any::<u64>(),
        queue_slots in 1usize..=96,
        queue_len_extra in 0u32..=32,
    ) {
        let graph = generated_skewed_csr(node_count, graph_seed);
        let queue = generated_active_queue(node_count, graph.hub, queue_slots, queue_seed);
        let queue_len = (queue.len() as u32).saturating_add(queue_len_extra);
        let allow_mask = 0b1011;
        let first = forward_witness(&graph, &queue, queue_len, node_count, allow_mask);
        let unrelated =
            csr_queue_strided_forward_witness(&[], 0, &[0, 0], &[], &[], 1, allow_mask);
        let second = forward_witness(&graph, &queue, queue_len, node_count, allow_mask);

        prop_assert_eq!(unrelated, vec![0]);
        prop_assert_eq!(second, first);
    }
}

#[test]
fn strided_queue_validator_rejects_malformed_csr() {
    let err = validate_csr_queue_graph(1, &[0, 2], &[0], &[1])
        .expect_err("Fix: malformed CSR final offset must be rejected");

    assert!(
        err.contains("final offset declares edge_count"),
        "Fix: malformed CSR diagnostic must identify the final offset mismatch, got: {err}"
    );
}

fn generated_skewed_csr(node_count: u32, seed: u64) -> GeneratedSkewedCsr {
    let hub = (mix64(seed ^ 0x8d12_f4b7_0c55_9d33) % u64::from(node_count)) as u32;
    let hub_degree = CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE
        .saturating_mul(CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE)
        .saturating_add((mix64(seed ^ 0x4471_4f03_abcd_02d1) % 2049) as u32);
    let mut edge_offsets = Vec::with_capacity(node_count as usize + 1);
    let mut edge_targets = Vec::new();
    let mut edge_kind_mask = Vec::new();
    edge_offsets.push(0);
    for src in 0..node_count {
        let row_seed = mix64(seed ^ (src as u64).wrapping_mul(0xd1b5_4a32_d192_ed03));
        let degree = if src == hub {
            hub_degree
        } else if src % 31 == 0 {
            32 + (row_seed % 17) as u32
        } else {
            (row_seed % 6) as u32
        };
        for edge_ordinal in 0..degree {
            let edge_seed =
                mix64(row_seed ^ (edge_ordinal as u64).wrapping_mul(0x94d0_49bb_1331_11eb));
            edge_targets.push((edge_seed % u64::from(node_count)) as u32);
            let selector = ((edge_seed >> 19) % 11) as u32;
            edge_kind_mask.push(match selector {
                0 => 0,
                1..=8 => 1u32 << (selector - 1),
                9 => 0b1011,
                _ => 0x8000_0001,
            });
        }
        edge_offsets.push(edge_targets.len() as u32);
    }
    GeneratedSkewedCsr {
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        hub,
    }
}

fn generated_active_queue(node_count: u32, hub: u32, slots: usize, seed: u64) -> Vec<u32> {
    let mut queue = Vec::with_capacity(slots);
    for slot in 0..slots {
        let slot_seed = mix64(seed ^ (slot as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let src = match slot % 13 {
            0 | 5 => hub,
            1 => node_count.saturating_add((slot_seed % 257) as u32),
            2 => node_count - 1,
            _ => (slot_seed % u64::from(node_count)) as u32,
        };
        queue.push(src);
    }
    queue
}
