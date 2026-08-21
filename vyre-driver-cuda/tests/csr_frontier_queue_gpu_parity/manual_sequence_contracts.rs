use super::*;

#[test]
fn cuda_resident_frontier_queue_drives_sparse_csr_without_selector_readback() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher::new(&backend);
    let node_count = 8u32;
    let queue_capacity = 8u32;
    let edge_offsets = vec![0, 2, 2, 2, 5, 5, 5, 5, 5];
    let edge_targets = vec![1, 2, 4, 5, 6];
    let edge_kind_mask = vec![1, 2, 1, 1, 2];
    let graph = QueueGraph {
        node_count,
        edge_offsets: &edge_offsets,
        edge_targets: &edge_targets,
        edge_kind_mask: &edge_kind_mask,
    };
    let frontier = pack_nodes(&[0, 3], node_count);
    let (expected_out, expected_len) = graph.expected_traverse(&frontier, queue_capacity);

    let outcome = run_manual_queue_sequence(
        &backend,
        &dispatcher,
        &graph,
        &frontier,
        queue_capacity,
        QueueBuild::Serial,
    );

    assert_eq!(outcome.frontier_out, expected_out);
    assert_eq!(outcome.queue_len, Some(expected_len));
    assert_eq!(
        outcome.telemetry.kernel_launches, 2,
        "Fix: queue sparse traversal must be exactly queue-build + queue-consume kernels."
    );
    assert_eq!(
        outcome.telemetry.sync_points, 1,
        "Fix: resident queue sparse traversal must fence once for uploads, kernels, and compact readbacks."
    );
    assert_eq!(
        outcome.telemetry.readback_bytes,
        (frontier.len() * std::mem::size_of::<u32>() + std::mem::size_of::<u32>()) as u64,
        "Fix: queue sparse traversal readback must be compact and avoid queue payload D2H."
    );
}

#[test]
fn cuda_resident_parallel_frontier_queue_scans_large_sparse_bitset() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher::new(&backend);
    let node_count = 1024u32;
    let queue_capacity = 16u32;
    // One edge per node, so the frontier is sparse against a large bitset: that
    // density is what makes the word-parallel build differ from the serial one.
    let mut edge_offsets = Vec::with_capacity(node_count as usize + 1);
    let mut edge_targets = Vec::with_capacity(node_count as usize);
    let mut edge_kind_mask = Vec::with_capacity(node_count as usize);
    edge_offsets.push(0);
    for src in 0..node_count {
        edge_targets.push((src.wrapping_mul(17).wrapping_add(9)) % node_count);
        edge_kind_mask.push(if src % 5 == 0 { 2 } else { 1 });
        edge_offsets.push(edge_targets.len() as u32);
    }
    let graph = QueueGraph {
        node_count,
        edge_offsets: &edge_offsets,
        edge_targets: &edge_targets,
        edge_kind_mask: &edge_kind_mask,
    };
    let frontier = pack_nodes(&[0, 3, 511, 700], node_count);
    let (expected_out, expected_len) = graph.expected_traverse(&frontier, queue_capacity);

    let outcome = run_manual_queue_sequence(
        &backend,
        &dispatcher,
        &graph,
        &frontier,
        queue_capacity,
        QueueBuild::Parallel,
    );

    assert_eq!(outcome.frontier_out, expected_out);
    assert_eq!(outcome.queue_len, Some(expected_len));
    assert_eq!(
        outcome.telemetry.kernel_launches, 2,
        "Fix: parallel queue traversal should be queue-build + queue-consume kernels after setup uploads."
    );
    assert_eq!(
        outcome.telemetry.readback_bytes,
        (frontier.len() * std::mem::size_of::<u32>() + std::mem::size_of::<u32>()) as u64
    );
}
