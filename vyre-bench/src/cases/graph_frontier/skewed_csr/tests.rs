use super::*;

#[test]
fn skewed_csr_fixture_has_variable_degree_and_bitset_frontier() {
    let fixture = build_skewed_csr_fixture(4096).unwrap();

    assert_eq!(fixture.edge_offsets.len(), 4097);
    assert!(fixture.edge_targets.len() > 4096);
    assert_eq!(fixture.stats.max_degree, 96);
    assert!(fixture.stats.high_degree_sources > 0);
    assert!(fixture.stats.active_sources > 0);
    assert_eq!(fixture.frontier_in.len(), 128);
    assert_eq!(fixture.frontier_out_seed, vec![0_u32; 128]);
}

#[test]
fn skewed_csr_cpu_oracle_sets_packed_output_words() {
    let fixture = build_skewed_csr_fixture(4096).unwrap();
    let oracle = skewed_csr_cpu_oracle(&fixture);

    assert_eq!(oracle.output.len(), fixture.frontier_out_seed.len());
    assert!(oracle.allowed_edges_from_active > fixture.stats.active_sources);
    assert!(oracle.output_words_set > 0);
    assert!(oracle.output.iter().any(|word| *word != 0));
}

#[test]
fn skewed_csr_prepare_builds_primitive_program_and_oracle() {
    let prepared = prepare_skewed_csr_case(None).unwrap();

    assert_eq!(prepared.program.workgroup_size(), [256, 1, 1]);
    assert_eq!(prepared.stats.node_count, CSR_NODE_COUNT);
    assert_eq!(
        prepared.baseline_output.len(),
        prepared.stats.frontier_words as usize * 4
    );
    assert_eq!(prepared.inputs.len(), 7);
    assert!(prepared.input_bytes_total > u64::from(CSR_NODE_COUNT) * 20);
}

#[test]
fn skewed_csr_queue_inputs_preserve_frontier_and_device_scratch() {
    let fixture = build_skewed_csr_fixture(4096).unwrap();
    let capacity = support::skewed_csr_queue_capacity(fixture.stats.active_sources).unwrap();
    let inputs = support::skewed_csr_queue_inputs(&fixture, capacity).unwrap();

    assert_eq!(inputs.len(), 7);
    assert_eq!(
        inputs[queue_materialize::QUEUE_FRONTIER_IN_INDEX],
        vyre_primitives::wire::pack_u32_slice(&fixture.frontier_in)
    );
    assert_eq!(
        inputs[queue_materialize::QUEUE_ACTIVE_QUEUE_INDEX].len(),
        capacity as usize * std::mem::size_of::<u32>()
    );
    assert_eq!(
        vyre_primitives::wire::decode_u32_le_bytes_all(&inputs[queue_materialize::QUEUE_LEN_INDEX]),
        vec![0]
    );

    let undersized = capacity.saturating_sub(1);
    let err = support::skewed_csr_queue_inputs(&fixture, undersized).unwrap_err();
    assert!(
        err.to_string().contains("queue_capacity >= active_sources"),
        "queue capacity errors must name the invariant, got: {err}"
    );
}

#[test]
fn skewed_csr_queue_prepare_builds_sparse_resident_sequence() {
    let prepared = queue_materialize::prepare_skewed_csr_queue_materialize_step(None).unwrap();

    assert_eq!(prepared.reset_program.workgroup_size(), [256, 1, 1]);
    assert_eq!(prepared.queue_program.workgroup_size(), [256, 1, 1]);
    assert!(prepared.row_strided_traverse);
    assert_eq!(
        prepared.traverse_grid,
        vyre_primitives::graph::csr_queue_strided::csr_queue_strided_forward_dispatch_grid(
            prepared.queue_capacity
        )
    );
    assert_eq!(prepared.inputs.len(), 7);
    assert_eq!(prepared.stats.node_count, CSR_NODE_COUNT);
    assert_eq!(
        u64::from(prepared.queue_capacity),
        prepared.stats.active_sources
    );
    assert!(
        prepared.queue_capacity < prepared.stats.node_count / 32,
        "queue capacity should stay sparse relative to the full node-grid launch"
    );
}

#[test]
fn generated_skewed_csr_queue_capacity_covers_active_sources_without_node_grid() {
    const CASES: u32 = 10_000;

    let mut total_nodes = 0_u64;
    let mut total_queue_capacity = 0_u64;
    let mut row_strided_cases = 0_u32;

    for case in 0..CASES {
        let node_count = 32_u32 << (case % 8);
        let fixture = build_skewed_csr_fixture(node_count).unwrap_or_else(|error| {
            panic!("generated skewed CSR fixture case {case} failed: {error}")
        });
        let capacity = support::skewed_csr_queue_capacity(fixture.stats.active_sources)
            .unwrap_or_else(|error| panic!("generated queue capacity case {case} failed: {error}"));
        let inputs = support::skewed_csr_queue_inputs(&fixture, capacity)
            .unwrap_or_else(|error| panic!("generated queue inputs case {case} failed: {error}"));

        assert_eq!(
            u64::from(capacity),
            fixture.stats.active_sources,
            "queue capacity should exactly cover active sources case {case}"
        );
        assert_eq!(
            inputs[queue_materialize::QUEUE_ACTIVE_QUEUE_INDEX].len(),
            capacity as usize * std::mem::size_of::<u32>(),
            "active queue byte length case {case}"
        );
        assert!(
            support::skewed_csr_queue_inputs(&fixture, capacity.saturating_sub(1)).is_err(),
            "undersized queue should fail case {case}"
        );
        row_strided_cases += u32::from(queue_materialize::graph_queue_should_use_row_strided(
            fixture.stats.max_degree,
        ));
        total_nodes += u64::from(node_count);
        total_queue_capacity += u64::from(capacity);
    }

    assert_eq!(row_strided_cases, CASES);
    assert!(
        total_queue_capacity * 8 < total_nodes,
        "generated sparse frontiers should avoid graph-sized queue traversal"
    );
}
