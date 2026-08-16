//! CSR graph and packed-frontier batch validation for queue-driven traversal.

use crate::bitset::bitset_words;

use super::CsrQueueGraphLayout;

/// Validate the CSR graph consumed by queue-driven sparse traversal.
///
/// Returns the resident graph layout so dispatch wrappers can construct padded
/// buffers without owning CSR validation locally.
///
/// # Errors
///
/// Returns an actionable diagnostic for zero-node graphs, malformed offsets,
/// mismatched edge arrays, or out-of-range destinations.
pub fn validate_csr_queue_graph(
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
) -> Result<CsrQueueGraphLayout, String> {
    if node_count == 0 {
        return Err("Fix: csr_queue_forward_traverse requires node_count > 0.".to_string());
    }
    let expected_offsets = (node_count as usize).checked_add(1).ok_or_else(|| {
        format!(
            "Fix: csr_queue_forward_traverse node_count + 1 overflows usize for node_count={node_count}."
        )
    })?;
    if edge_offsets.len() != expected_offsets {
        return Err(format!(
            "Fix: csr_queue_forward_traverse requires edge_offsets.len() == node_count + 1, got len={}, node_count={node_count}.",
            edge_offsets.len()
        ));
    }
    if edge_targets.len() != edge_kind_mask.len() {
        return Err(format!(
            "Fix: csr_queue_forward_traverse requires edge_targets.len() == edge_kind_mask.len(), got {} vs {}.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }
    if edge_offsets[0] != 0 {
        return Err(format!(
            "Fix: csr_queue_forward_traverse requires edge_offsets[0] == 0, got {}.",
            edge_offsets[0]
        ));
    }
    let mut max_row_degree = 0u32;
    for (row, pair) in edge_offsets.windows(2).enumerate() {
        if pair[0] > pair[1] {
            return Err(format!(
                "Fix: csr_queue_forward_traverse offsets must be monotonic at row {row}: {} > {}.",
                pair[0], pair[1]
            ));
        }
        max_row_degree = max_row_degree.max(pair[1] - pair[0]);
    }
    let edge_count = edge_offsets[expected_offsets - 1] as usize;
    if edge_targets.len() != edge_count {
        return Err(format!(
            "Fix: csr_queue_forward_traverse final offset declares edge_count={edge_count}, but targets_len={} and kind_mask_len={}.",
            edge_targets.len(),
            edge_kind_mask.len()
        ));
    }
    for (index, &target) in edge_targets.iter().enumerate() {
        if target >= node_count {
            return Err(format!(
                "Fix: csr_queue_forward_traverse edge_targets[{index}]={target} is outside node_count {node_count}."
            ));
        }
    }
    let edge_count = u32::try_from(edge_count).map_err(|_| {
        format!("Fix: csr_queue_forward_traverse edge count {edge_count} exceeds u32 index space.")
    })?;
    Ok(CsrQueueGraphLayout {
        node_count,
        edge_count,
        max_row_degree,
        words: bitset_words(node_count) as usize,
        edge_storage_words: edge_targets.len().max(1),
    })
}

/// Validate a batch of packed frontiers for queue-driven CSR traversal.
///
/// Returns the exact packed frontier word count implied by `node_count`, so
/// dispatch wrappers can size resident scratch without duplicating the
/// primitive's batch-shape contract.
///
/// # Errors
///
/// Returns an actionable diagnostic for zero-node graphs, empty batches, zero
/// queue capacity, or any query frontier whose packed bitset width does not
/// match `node_count`.
pub fn validate_frontier_queue_batch(
    node_count: u32,
    frontiers: &[&[u32]],
    queue_capacity: u32,
) -> Result<usize, String> {
    if node_count == 0 {
        return Err("Fix: resident CSR queue batch requires node_count > 0.".to_string());
    }
    if frontiers.is_empty() {
        return Err("Fix: resident CSR queue batch requires at least one frontier.".to_string());
    }
    if queue_capacity == 0 {
        return Err("Fix: resident CSR queue batch requires queue_capacity > 0.".to_string());
    }

    let expected_words = bitset_words(node_count) as usize;
    for (query_index, frontier) in frontiers.iter().enumerate() {
        if frontier.len() != expected_words {
            return Err(format!(
                "Fix: resident CSR queue batch query {query_index} expected {expected_words} frontier word(s) for node_count={node_count} but received {}.",
                frontier.len()
            ));
        }
    }
    Ok(expected_words)
}

/// Validate one packed frontier for queue-driven CSR traversal.
///
/// Returns the exact packed frontier word count implied by `node_count`, so a
/// resident dispatch wrapper can size scratch without duplicating queue and
/// frontier-shape policy.
///
/// # Errors
///
/// Returns an actionable diagnostic for zero-node graphs, zero queue capacity,
/// or a frontier whose packed bitset width does not match `node_count`.
pub fn validate_frontier_queue_query(
    node_count: u32,
    frontier: &[u32],
    queue_capacity: u32,
) -> Result<usize, String> {
    validate_frontier_queue_batch(node_count, &[frontier], queue_capacity).map_err(|error| {
        error
            .replace("resident CSR queue batch", "resident CSR queue query")
            .replace("query 0", "query")
    })
}

#[cfg(test)]
mod tests {
    use super::super::CsrQueueGraphLayout;
    use super::{
        validate_csr_queue_graph, validate_frontier_queue_batch, validate_frontier_queue_query,
    };

    #[test]
    fn validate_csr_queue_graph_accepts_zero_edge_graph_and_canonical_graph() {
        assert_eq!(
            validate_csr_queue_graph(3, &[0, 0, 0, 0], &[], &[]).unwrap(),
            CsrQueueGraphLayout {
                node_count: 3,
                edge_count: 0,
                max_row_degree: 0,
                words: 1,
                edge_storage_words: 1,
            }
        );
        assert_eq!(
            validate_csr_queue_graph(4, &[0, 2, 3, 3, 3], &[1, 2, 3], &[1, 2, 1]).unwrap(),
            CsrQueueGraphLayout {
                node_count: 4,
                edge_count: 3,
                max_row_degree: 2,
                words: 1,
                edge_storage_words: 3,
            }
        );
    }

    #[test]
    fn validate_csr_queue_graph_rejects_malformed_inputs() {
        let err = validate_csr_queue_graph(0, &[0], &[], &[]).unwrap_err();
        assert!(err.contains("node_count > 0"));

        let err = validate_csr_queue_graph(2, &[0, 1, 1], &[1], &[]).unwrap_err();
        assert!(err.contains("edge_targets.len() == edge_kind_mask.len()"));

        let err = validate_csr_queue_graph(2, &[0, 2, 1], &[1], &[1]).unwrap_err();
        assert!(err.contains("offsets must be monotonic"));

        let err = validate_csr_queue_graph(2, &[0, 1, 1], &[5], &[1]).unwrap_err();
        assert!(err.contains("outside node_count"));
    }

    #[test]
    fn validate_frontier_queue_batch_accepts_canonical_frontiers() {
        let frontiers: [&[u32]; 2] = [&[1, 0], &[0, 2]];

        let words = validate_frontier_queue_batch(64, &frontiers, 8)
            .expect("Fix: two 64-node frontiers should be valid");

        assert_eq!(words, 2);
    }

    #[test]
    fn validate_frontier_queue_batch_rejects_invalid_batch_shapes() {
        let frontier: [&[u32]; 1] = [&[1]];

        let err = validate_frontier_queue_batch(0, &frontier, 8).unwrap_err();
        assert!(err.contains("node_count > 0"));

        let empty: [&[u32]; 0] = [];
        let err = validate_frontier_queue_batch(64, &empty, 8).unwrap_err();
        assert!(err.contains("at least one frontier"));

        let err = validate_frontier_queue_batch(64, &frontier, 0).unwrap_err();
        assert!(err.contains("queue_capacity > 0"));

        let err = validate_frontier_queue_batch(64, &frontier, 8).unwrap_err();
        assert!(err.contains("query 0 expected 2 frontier word"));
    }

    #[test]
    fn validate_frontier_queue_query_delegates_single_frontier_contract() {
        assert_eq!(validate_frontier_queue_query(64, &[1, 0], 8).unwrap(), 2);

        let err = validate_frontier_queue_query(64, &[1], 8).unwrap_err();
        assert!(err.contains("query expected 2 frontier word"));

        let err = validate_frontier_queue_query(64, &[1, 0], 0).unwrap_err();
        assert!(err.contains("queue_capacity > 0"));
    }
}
