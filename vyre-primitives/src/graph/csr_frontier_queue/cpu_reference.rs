//! CPU references for queue materialization and queue-driven CSR expansion.

use super::validate_csr_queue_graph;

/// CPU reference for queue materialization.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn frontier_to_queue_cpu(
    frontier_in: &[u32],
    node_count: u32,
    queue_capacity: usize,
) -> (Vec<u32>, u32) {
    try_frontier_to_queue_cpu(frontier_in, node_count, queue_capacity).unwrap_or_else(|err| {
        panic!("frontier_to_queue CPU oracle received malformed input. {err}")
    })
}

/// Fallible CPU reference for queue materialization.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_frontier_to_queue_cpu(
    frontier_in: &[u32],
    node_count: u32,
    queue_capacity: usize,
) -> Result<(Vec<u32>, u32), String> {
    let mut queue: Vec<u32> = Vec::new();
    let seen = try_frontier_to_queue_cpu_into(frontier_in, node_count, queue_capacity, &mut queue)?;
    Ok((queue, seen))
}

/// Fallible CPU reference for queue materialization into caller-owned storage.
///
/// On error, `queue` is left unchanged. This keeps parity harnesses and
/// resident dispatch diagnostics from losing the last queue snapshot when a
/// malformed frontier arrives.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_frontier_to_queue_cpu_into(
    frontier_in: &[u32],
    node_count: u32,
    queue_capacity: usize,
    queue: &mut Vec<u32>,
) -> Result<u32, String> {
    crate::bitset::frontier::materialize_frontier_queue_prefix_into(
        node_count,
        frontier_in,
        queue_capacity,
        queue,
    )
    .map_err(|error| match error {
        crate::bitset::frontier::FrontierError::BadShape {
            expected_words,
            actual_words,
            ..
        } => format!(
            "Fix: frontier_to_queue requires frontier_in.len() == bitset_words(node_count), got len={actual_words} but expected {expected_words} for node_count={node_count}."
        ),
        other => format!(
            "Fix: frontier_to_queue CPU oracle could not materialize the active frontier queue: {other}"
        ),
    })
}

/// CPU reference for queue-driven CSR expansion.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn csr_queue_forward_traverse_cpu(
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    node_count: u32,
    allow_mask: u32,
) -> Vec<u32> {
    try_csr_queue_forward_traverse_cpu(
        active_queue,
        queue_len,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        node_count,
        allow_mask,
    )
    .unwrap_or_else(|err| {
        panic!("csr_queue_forward_traverse CPU oracle received malformed input. {err}")
    })
}

/// Fallible CPU reference for queue-driven CSR expansion.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_csr_queue_forward_traverse_cpu(
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    node_count: u32,
    allow_mask: u32,
) -> Result<Vec<u32>, String> {
    let mut out: Vec<u32> = Vec::new();
    try_csr_queue_forward_traverse_cpu_into(
        active_queue,
        queue_len,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        node_count,
        allow_mask,
        &mut out,
    )?;
    Ok(out)
}

/// Fallible CPU reference for queue-driven CSR expansion into caller-owned storage.
#[cfg(any(test, feature = "cpu-parity"))]
#[allow(clippy::too_many_arguments)]
pub fn try_csr_queue_forward_traverse_cpu_into(
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    node_count: u32,
    allow_mask: u32,
    out: &mut Vec<u32>,
) -> Result<(), String> {
    let layout = validate_csr_queue_graph(node_count, edge_offsets, edge_targets, edge_kind_mask)?;
    crate::graph::scratch::reserve_graph_items(
        out,
        layout.words,
        "CSR frontier queue CPU oracle",
        "frontier output bitset",
    )?;
    out.clear();
    out.resize(layout.words, 0);
    let take = (queue_len as usize).min(active_queue.len());
    for &src in &active_queue[..take] {
        if src >= node_count {
            continue;
        }
        let start = edge_offsets[src as usize] as usize;
        let end = edge_offsets[src as usize + 1] as usize;
        for edge in start..end.min(edge_targets.len()).min(edge_kind_mask.len()) {
            if edge_kind_mask[edge] & allow_mask == 0 {
                continue;
            }
            let dst = edge_targets[edge];
            if dst < node_count {
                out[dst as usize / 32] |= 1u32 << (dst % 32);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod generated_cpu_oracle_tests {
    use super::*;
    use crate::bitset::bitset_words;

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
        let frontier = [0b1010_u32, u32::MAX];
        let mut queue = Vec::new();

        let seen = try_frontier_to_queue_cpu_into(&frontier, 33, 2, &mut queue)
            .expect("Fix: canonical frontier should materialize through the CPU oracle");

        assert_eq!(seen, 3);
        assert_eq!(queue, vec![1, 3]);
        assert!(
            queue.iter().all(|node| *node < 33),
            "out-of-domain tail bits must not enter the compact queue prefix"
        );
    }

    #[test]
    fn queue_forward_traverse_into_rejects_bad_graph_without_clobbering_output() {
        let mut out = vec![0xDEAD_BEEF];

        let err = try_csr_queue_forward_traverse_cpu_into(
            &[0],
            1,
            &[0, 1, 1],
            &[2],
            &[1],
            2,
            1,
            &mut out,
        )
        .expect_err("out-of-range target must fail CSR queue graph validation");

        assert!(
            err.contains("outside node_count"),
            "Fix: queue traversal graph errors must identify invalid targets, got: {err}"
        );
        assert_eq!(
            out,
            vec![0xDEAD_BEEF],
            "failed queue traversal preflight must preserve previous output diagnostics"
        );
    }

    #[test]
    fn generated_frontier_queue_and_traverse_cpu_oracles_match_shape_contracts() {
        for node_count in 1u32..=128 {
            let edge_offsets: Vec<u32> = (0..=node_count).collect();
            let edge_targets: Vec<u32> = (0..node_count)
                .map(|node| (node + 1) % node_count)
                .collect();
            let edge_kind_mask = vec![1u32; node_count as usize];
            for queue_capacity in 0usize..32 {
                let mut frontier = vec![0u32; bitset_words(node_count) as usize];
                let period = (queue_capacity as u32 % 7) + 1;
                let mut expected_seen = 0u32;
                for node in 0..node_count {
                    if node % period == 0 {
                        frontier[node as usize / 32] |= 1u32 << (node % 32);
                        expected_seen = expected_seen.saturating_add(1);
                    }
                }
                let (queue, seen) =
                    try_frontier_to_queue_cpu(&frontier, node_count, queue_capacity).unwrap();
                assert_eq!(seen, expected_seen);
                assert_eq!(queue.len(), queue_capacity.min(expected_seen as usize));
                let out = try_csr_queue_forward_traverse_cpu(
                    &queue,
                    seen,
                    &edge_offsets,
                    &edge_targets,
                    &edge_kind_mask,
                    node_count,
                    1,
                )
                .unwrap();
                assert_eq!(out.len(), bitset_words(node_count) as usize);
                for &src in &queue {
                    let dst = (src + 1) % node_count;
                    assert_ne!(out[dst as usize / 32] & (1u32 << (dst % 32)), 0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{csr_queue_forward_traverse_cpu, frontier_to_queue_cpu};

    #[test]
    fn cpu_queue_preserves_node_order_and_reports_overflow_pressure() {
        let (queue, len) = frontier_to_queue_cpu(&[0b10111], 5, 3);
        assert_eq!(queue, vec![0, 1, 2]);
        assert_eq!(len, 4);
    }

    #[test]
    fn cpu_queue_traverse_expands_only_queued_sources() {
        let edge_offsets = vec![0, 2, 3, 3, 3];
        let edge_targets = vec![1, 2, 3];
        let edge_kind_mask = vec![1, 2, 1];
        let out = csr_queue_forward_traverse_cpu(
            &[0, 1],
            2,
            &edge_offsets,
            &edge_targets,
            &edge_kind_mask,
            4,
            1,
        );
        assert_eq!(out, vec![0b1010]);
    }
}
