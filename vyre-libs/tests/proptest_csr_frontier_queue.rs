//! Property gates for queue-driven sparse CSR traversal.

#![cfg(feature = "graph")]

mod graph_sweep_fixtures;
use graph_sweep_fixtures::{
    active_nodes, generated_csr, generated_frontier_words, max_row_degree, GeneratedCsr,
};
mod wire_words;
use wire_words::queue_forward_oracle;

use proptest::prelude::*;
use vyre_libs::bitset::bitset_words;
use vyre_libs::graph::csr_frontier_queue::{validate_csr_queue_graph, CsrQueueGraphLayout};
use vyre_reference::composition_witness::{
    csr_queue_strided_forward_witness, frontier_to_queue_witness,
};


proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn queue_materialization_matches_independent_sparse_frontier_oracle(
        node_count in 1u32..=4096,
        seed in any::<u64>(),
        capacity_salt in any::<u32>(),
    ) {
        let frontier = generated_frontier_words(node_count, seed);
        let queue_capacity = (capacity_salt as usize) % (node_count as usize + 33);
        let expected_nodes = active_nodes(&frontier, node_count);
        let expected_queue = expected_nodes
            .iter()
            .copied()
            .take(queue_capacity)
            .collect::<Vec<_>>();

        let (queue, seen) = frontier_to_queue_witness(&frontier, node_count, queue_capacity);

        prop_assert_eq!(seen, expected_nodes.len() as u32);
        prop_assert_eq!(queue, expected_queue);
    }

    #[test]
    fn queue_forward_traverse_matches_independent_csr_oracle(
        node_count in 1u32..=384,
        graph_seed in any::<u64>(),
        frontier_seed in any::<u64>(),
        capacity_salt in any::<u32>(),
        allow_mask in any::<u32>(),
    ) {
        let graph = generated_csr(node_count, graph_seed);
        let frontier = generated_frontier_words(node_count, frontier_seed);
        let queue_capacity = (capacity_salt as usize) % (node_count as usize + 1);
        let (queue, queue_len) = frontier_to_queue_witness(&frontier, node_count, queue_capacity);
        let expected = queue_forward_oracle(
            &queue,
            queue_len,
            &graph.edge_offsets,
            &graph.edge_targets,
            &graph.edge_kind_mask,
            node_count,
            allow_mask,
        );

        let actual = csr_queue_strided_forward_witness(
            &queue,
            queue_len,
            &graph.edge_offsets,
            &graph.edge_targets,
            &graph.edge_kind_mask,
            node_count,
            allow_mask,
        );

        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn queue_forward_ignores_invalid_sources_and_clamps_queue_len_to_active_storage(
        node_count in 1u32..=256,
        graph_seed in any::<u64>(),
        active_queue in prop::collection::vec(any::<u32>(), 0..96),
        queue_len in any::<u32>(),
        allow_mask in any::<u32>(),
    ) {
        let graph = generated_csr(node_count, graph_seed);
        let expected = queue_forward_oracle(
            &active_queue,
            queue_len,
            &graph.edge_offsets,
            &graph.edge_targets,
            &graph.edge_kind_mask,
            node_count,
            allow_mask,
        );

        let actual = csr_queue_strided_forward_witness(
            &active_queue,
            queue_len,
            &graph.edge_offsets,
            &graph.edge_targets,
            &graph.edge_kind_mask,
            node_count,
            allow_mask,
        );

        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn csr_queue_validation_accepts_canonical_and_rejects_single_field_mutations(
        node_count in 1u32..=512,
        graph_seed in any::<u64>(),
    ) {
        let graph = generated_csr(node_count, graph_seed);
        let layout = validate_csr_queue_graph(
            node_count,
            &graph.edge_offsets,
            &graph.edge_targets,
            &graph.edge_kind_mask,
        )
        .expect("Fix: generated canonical CSR queue graph should validate");
        prop_assert_eq!(
            layout,
            CsrQueueGraphLayout {
                node_count,
                edge_count: graph.edge_targets.len() as u32,
                max_row_degree: max_row_degree(&graph.edge_offsets),
                words: bitset_words(node_count) as usize,
                edge_storage_words: graph.edge_targets.len().max(1),
            }
        );

        let mut non_zero_start = graph.edge_offsets.clone();
        non_zero_start[0] = 1;
        prop_assert!(
            validate_csr_queue_graph(
                node_count,
                &non_zero_start,
                &graph.edge_targets,
                &graph.edge_kind_mask,
            )
            .expect_err("Fix: non-zero CSR start offset must be rejected")
            .contains("edge_offsets[0] == 0")
        );

        let mut bad_final = graph.edge_offsets.clone();
        let last = bad_final
            .last_mut()
            .expect("Fix: generated CSR offsets include node_count + 1 entries");
        *last = last.saturating_add(1);
        prop_assert!(
            validate_csr_queue_graph(
                node_count,
                &bad_final,
                &graph.edge_targets,
                &graph.edge_kind_mask,
            )
            .expect_err("Fix: final CSR offset mismatch must be rejected")
            .contains("final offset declares edge_count")
        );

        let mut mismatched_masks = graph.edge_kind_mask.clone();
        mismatched_masks.push(1);
        prop_assert!(
            validate_csr_queue_graph(
                node_count,
                &graph.edge_offsets,
                &graph.edge_targets,
                &mismatched_masks,
            )
            .expect_err("Fix: CSR edge target/mask length mismatch must be rejected")
            .contains("edge_targets.len() == edge_kind_mask.len()")
        );

        if !graph.edge_targets.is_empty() {
            let mut out_of_range_targets = graph.edge_targets.clone();
            out_of_range_targets[0] = node_count;
            prop_assert!(
                validate_csr_queue_graph(
                    node_count,
                    &graph.edge_offsets,
                    &out_of_range_targets,
                    &graph.edge_kind_mask,
                )
                .expect_err("Fix: out-of-range CSR target must be rejected")
                .contains("outside node_count")
            );
        }
    }
}
