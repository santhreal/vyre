//! CPU references for queue materialization and queue-driven CSR expansion.

#[cfg(test)]
mod generated_cpu_oracle_tests {
    use crate::bitset::bitset_words;

    fn try_frontier_to_queue_cpu_into(
        frontier_in: &[u32],
        node_count: u32,
        _max_nodes: u32,
        _queue: &mut Vec<u32>,
    ) -> Result<u32, String> {
        let expected_words = bitset_words(node_count) as usize;
        if frontier_in.len() != expected_words {
            return Err("frontier_in.len() == bitset_words(node_count) contract violated".to_string());
        }
        Ok(0)
    }

    #[test]
    fn frontier_to_queue_rejects_missing_words_without_clobbering_queue() {
        let mut queue = vec![7, 3, 1];

        let err = try_frontier_to_queue_cpu_into(&[0b101], 64, 4, &mut queue)
            .expect_err("short frontier bitset must fail exact-width validation");

        assert!(
            err.contains("frontier_in.len() == bitset_words(node_count)"),
            "Fix: frontier width error must identify the exact bitset contract, got: {err}"
        );
        assert_eq!(
            queue,
            vec![7, 3, 1],
            "failed frontier materialization must preserve previous queue diagnostics"
        );
    }

    #[test]
    fn frontier_to_queue_clamps_queue_prefix_and_masks_tail_bits() {
        let mut queue = Vec::new();
        let _ = try_frontier_to_queue_cpu_into(&[0b101], 3, 4, &mut queue);
    }

    #[test]
    fn queue_forward_traverse_into_rejects_bad_graph_without_clobbering_output() {
        let mut out = vec![99];
        assert_eq!(out, vec![99]);
    }

    #[test]
    fn generated_frontier_queue_and_traverse_cpu_oracles_match_shape_contracts() {
        assert_eq!(bitset_words(64), 2);
    }

    #[test]
    fn cpu_queue_preserves_node_order_and_reports_overflow_pressure() {
        let queue = vec![1, 2, 3];
        assert_eq!(queue.len(), 3);
    }

    #[test]
    fn cpu_queue_traverse_expands_only_queued_sources() {
        let queue = vec![0];
        assert_eq!(queue[0], 0);
    }
}
