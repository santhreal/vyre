use super::*;

/// The small fixed graph the residency cases reuse: node 0 fans out to two
/// neighbours, node 3 to three, and the rest are isolated.
fn small_graph() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    (
        vec![0, 2, 2, 2, 5, 5, 5, 5, 5],
        vec![1, 2, 4, 5, 6],
        vec![1, 2, 1, 1, 2],
    )
}

#[test]
fn cuda_resident_frontier_queue_reuses_static_graph_across_queries() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher::new(&backend);
    let node_count = 8u32;
    let queue_capacity = 8u32;
    let (edge_offsets, edge_targets, edge_kind_mask) = small_graph();
    let graph = QueueGraph {
        node_count,
        edge_offsets: &edge_offsets,
        edge_targets: &edge_targets,
        edge_kind_mask: &edge_kind_mask,
    };
    let frontier_bytes_len = bitset_words(node_count) as usize * std::mem::size_of::<u32>();

    let handles = ManualQueueHandles::alloc(&dispatcher, &graph, queue_capacity);
    handles.upload_graph(&dispatcher, &graph);

    for active_nodes in [&[0, 3][..], &[3][..]] {
        let frontier = pack_nodes(active_nodes, node_count);
        let (expected_out, _) = graph.expected_traverse(&frontier, queue_capacity);

        // The graph is already resident, so this query refreshes only the
        // frontier, the queue length and the output, and reads back the frontier
        // alone.
        let outcome = handles.run(
            &backend,
            &dispatcher,
            &graph,
            &frontier,
            &QueueBuild::Serial,
            &GraphUpload::AlreadyResident,
            &QueueReadback::FrontierOut,
        );

        assert_eq!(outcome.frontier_out, expected_out);
        assert_eq!(outcome.telemetry.kernel_launches, 2);
        assert_eq!(outcome.telemetry.sync_points, 1);
        assert_eq!(
            outcome.telemetry.readback_bytes,
            frontier_bytes_len as u64,
            "Fix: repeated resident queue query must read back only frontier_out, not queue payload or selector count."
        );
        assert_eq!(
            outcome
                .telemetry
                .host_to_device_bytes
                .saturating_sub(outcome.telemetry.param_upload_bytes),
            outcome.uploaded_bytes,
            "Fix: repeated resident queue query must refresh only frontier/scratch/output buffers and keep CSR graph state resident."
        );
        assert!(
            outcome.telemetry.host_upload_operations <= 5,
            "Fix: repeated resident queue query must issue only frontier/scratch/output data uploads plus cached parameter uploads, not CSR graph uploads; observed {} upload operations.",
            outcome.telemetry.host_upload_operations
        );
    }

    handles.free(&dispatcher);
}

#[test]
fn cuda_resident_csr_queue_api_reuses_graph_and_scratch() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher::new(&backend);
    let node_count = 8u32;
    let queue_capacity = 8u32;
    let (edge_offsets, edge_targets, edge_kind_mask) = small_graph();
    let graph = QueueGraph {
        node_count,
        edge_offsets: &edge_offsets,
        edge_targets: &edge_targets,
        edge_kind_mask: &edge_kind_mask,
    };
    let mut session = ResidentQueueSession::open(&dispatcher, &graph, "reusable");
    let output_ptr = session.output_ptr();

    for active_nodes in [&[0, 3][..], &[3][..]] {
        let frontier = pack_nodes(active_nodes, node_count);
        let telemetry = session.query(&backend, &dispatcher, &graph, &frontier, queue_capacity);

        assert_eq!(
            session.output_ptr(),
            output_ptr,
            "Fix: resident CSR queue API must preserve caller-owned output capacity."
        );
        assert_eq!(telemetry.kernel_launches, 3);
        assert_eq!(telemetry.sync_points, 1);
        assert_eq!(
            telemetry.readback_bytes,
            session.output_len() as u64,
            "Fix: resident CSR queue API must compact readback to frontier_out only."
        );
        assert_eq!(
            telemetry
                .host_to_device_bytes
                .saturating_sub(telemetry.param_upload_bytes),
            (frontier.len() * std::mem::size_of::<u32>()) as u64,
            "Fix: resident CSR queue API must upload only the frontier seed; queue length and frontier output are initialized on device."
        );
    }

    session.close(&dispatcher);
}

#[test]
fn cuda_resident_csr_queue_uses_atomic_word_scan_for_large_sparse_frontier() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher::new(&backend);
    let node_count = 9_000u32;
    let queue_capacity = 16u32;
    let mut edge_offsets = Vec::with_capacity(node_count as usize + 1);
    let mut edge_targets = Vec::with_capacity(node_count as usize);
    let mut edge_kind_mask = Vec::with_capacity(node_count as usize);
    edge_offsets.push(0);
    for src in 0..node_count {
        edge_targets.push(src.wrapping_mul(17).wrapping_add(13) % node_count);
        edge_kind_mask.push(if src % 11 == 0 { 2 } else { 1 });
        edge_offsets.push(edge_targets.len() as u32);
    }
    let graph = QueueGraph {
        node_count,
        edge_offsets: &edge_offsets,
        edge_targets: &edge_targets,
        edge_kind_mask: &edge_kind_mask,
    };
    let frontier = pack_nodes(&[0, 3, 511, 7_000, 8_999], node_count);

    let mut session = ResidentQueueSession::open(&dispatcher, &graph, "large sparse");
    let telemetry = session.query(&backend, &dispatcher, &graph, &frontier, queue_capacity);

    assert_eq!(
        telemetry.kernel_launches, 3,
        "Fix: sparse resident CSR queue should run clear, atomic word queue-build, and traverse kernels; deterministic word-prefix is reserved for dense high-capacity frontiers."
    );
    assert_eq!(
        telemetry
            .host_to_device_bytes
            .saturating_sub(telemetry.param_upload_bytes),
        (frontier.len() * std::mem::size_of::<u32>()) as u64,
        "Fix: large sparse resident CSR queue must upload only the packed frontier; queue scratch stays device-side."
    );

    session.close(&dispatcher);
}

#[test]
fn cuda_resident_csr_queue_api_splits_skewed_high_degree_rows() {
    let backend = live_backend();
    let dispatcher = CudaProgramDispatcher::new(&backend);
    let node_count = 64u32;
    let queue_capacity = 1024u32;
    let (edge_offsets, edge_targets, edge_kind_mask) = skewed_high_degree_graph(node_count);
    let graph = QueueGraph {
        node_count,
        edge_offsets: &edge_offsets,
        edge_targets: &edge_targets,
        edge_kind_mask: &edge_kind_mask,
    };
    let frontier = pack_nodes(&[0, 1, 2, 3, 4, 5, 6, 7, 8], node_count);

    let mut session = ResidentQueueSession::open(&dispatcher, &graph, "skewed high-degree");
    let telemetry = session.query(&backend, &dispatcher, &graph, &frontier, queue_capacity);

    assert_eq!(
        telemetry.kernel_launches, 5,
        "Fix: skewed resident CSR queue query must run queue_len init, queue build, high_len init, split-low, and bounded high-row traverse."
    );
    assert_eq!(telemetry.sync_points, 1);
    assert_eq!(
        telemetry.readback_bytes,
        session.output_len() as u64,
        "Fix: skewed resident CSR queue query must read back only frontier_out."
    );
    assert_eq!(
        telemetry
            .host_to_device_bytes
            .saturating_sub(telemetry.param_upload_bytes),
        (frontier.len() * std::mem::size_of::<u32>()) as u64,
        "Fix: skewed resident CSR queue query must upload only the packed frontier seed."
    );

    session.close(&dispatcher);
}
