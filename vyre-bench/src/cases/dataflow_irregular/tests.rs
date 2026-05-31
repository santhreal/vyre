use super::fixture::IfdsSkewedFixture;
use super::fixture::{ifds_active_queue_inputs, ifds_queue_inputs};
use super::queue::{
    ifds_queue_closure_inputs, ifds_queue_closure_reset_program, ifds_queue_reset_program,
    ifds_queue_should_use_row_strided, ifds_skewed_queue_closure_oracle,
    ifds_sparse_queue_capacity, prepare_ifds_skewed_active_queue_step,
    prepare_ifds_skewed_queue_closure, prepare_ifds_skewed_queue_materialize_step,
    ACTIVE_QUEUE_ACTIVE_QUEUE_INDEX, ACTIVE_QUEUE_EDGE_KIND_INDEX, ACTIVE_QUEUE_EDGE_OFFSETS_INDEX,
    ACTIVE_QUEUE_EDGE_TARGETS_INDEX, ACTIVE_QUEUE_FRONTIER_OUT_INDEX, ACTIVE_QUEUE_LEN_INDEX,
    QUEUE_ACTIVE_QUEUE_INDEX, QUEUE_CLOSURE_ACCUMULATOR_INDEX, QUEUE_CLOSURE_EDGE_KIND_INDEX,
    QUEUE_CLOSURE_EDGE_OFFSETS_INDEX, QUEUE_CLOSURE_EDGE_TARGETS_INDEX, QUEUE_CLOSURE_LEN_A_INDEX,
    QUEUE_CLOSURE_LEN_B_INDEX, QUEUE_CLOSURE_QUEUE_A_INDEX, QUEUE_CLOSURE_QUEUE_B_INDEX,
    QUEUE_CLOSURE_SEED_FRONTIER_INDEX, QUEUE_CLOSURE_SEED_LEN_INDEX,
    QUEUE_CLOSURE_SEED_QUEUE_INDEX, QUEUE_FRONTIER_IN_INDEX, QUEUE_FRONTIER_OUT_INDEX,
    QUEUE_LEN_INDEX,
};
use super::*;
use proptest::prelude::*;

#[test]
fn ifds_skewed_fixture_has_filtered_edges_and_bitset_frontier() {
    let fixture = build_ifds_skewed_fixture(4096).unwrap();
    let oracle = ifds_skewed_cpu_oracle(&fixture);

    assert_eq!(fixture.edge_offsets.len(), 4097);
    assert!(fixture.edge_targets.len() > 4096);
    assert_eq!(fixture.stats.max_degree, UGLY_HUB_DEGREE);
    assert!(fixture.stats.high_degree_sources > 0);
    assert!(ifds_queue_should_use_row_strided(fixture.stats.max_degree));
    assert!(fixture.stats.active_sources > 0);
    assert!(oracle.allowed_edges_from_active > 0);
    assert!(oracle.filtered_edges_from_active > 0);
    assert_eq!(fixture.frontier_in.len(), 128);
}

#[test]
fn ifds_skewed_cpu_oracle_sets_packed_output_words() {
    let fixture = build_ifds_skewed_fixture(4096).unwrap();
    let oracle = ifds_skewed_cpu_oracle(&fixture);

    assert_eq!(oracle.output.len(), fixture.frontier_out_seed.len());
    assert!(oracle.output_words_set > 0);
    assert!(oracle.output.iter().any(|word| *word != 0));
}

#[test]
fn ifds_skewed_prepare_builds_vyre_program_and_oracle() {
    let prepared = prepare_ifds_skewed_step(None).unwrap();

    assert_eq!(prepared.program.workgroup_size(), [256, 1, 1]);
    assert_eq!(prepared.stats.nodes, NODE_COUNT);
    assert_eq!(prepared.baseline_output.len(), FRONTIER_WORDS * 4);
    assert_eq!(prepared.inputs.len(), 7);
    assert!(prepared.stats.filtered_edges_from_active > 0);
    assert!(prepared.input_bytes_total > u64::from(NODE_COUNT) * 20);
}

#[test]
fn ifds_queue_inputs_preserve_sparse_frontier_and_device_scratch() {
    let fixture = build_ifds_skewed_fixture(4096).unwrap();
    let capacity = ifds_sparse_queue_capacity(fixture.stats.active_sources).unwrap();
    let inputs = ifds_queue_inputs(&fixture, capacity).unwrap();

    assert_eq!(inputs.len(), 7);
    assert_eq!(
        inputs[QUEUE_FRONTIER_IN_INDEX],
        vyre_primitives::wire::pack_u32_slice(&fixture.frontier_in)
    );
    assert_eq!(
        inputs[QUEUE_ACTIVE_QUEUE_INDEX].len(),
        capacity as usize * std::mem::size_of::<u32>()
    );
    assert!(inputs[QUEUE_ACTIVE_QUEUE_INDEX]
        .iter()
        .all(|byte| *byte == 0));
    assert_eq!(
        inputs[QUEUE_LEN_INDEX],
        vyre_primitives::wire::pack_u32_slice(&[0])
    );
    assert_eq!(
        inputs[QUEUE_FRONTIER_OUT_INDEX],
        vyre_primitives::wire::pack_u32_slice(&fixture.frontier_out_seed)
    );
}

#[test]
fn ifds_queue_inputs_reject_capacity_below_active_sources() {
    let fixture = build_ifds_skewed_fixture(4096).unwrap();
    let undersized = fixture.stats.active_sources.saturating_sub(1) as u32;

    let err = ifds_queue_inputs(&fixture, undersized).unwrap_err();

    assert!(
        err.to_string().contains("queue_capacity >= active_sources"),
        "queue fixture errors must name the capacity invariant, got: {err}"
    );
}

#[test]
fn ifds_active_queue_inputs_materialize_frontier_queue_once() {
    let fixture = build_ifds_skewed_fixture(4096).unwrap();
    let capacity = ifds_sparse_queue_capacity(fixture.stats.active_sources).unwrap();
    let inputs = ifds_active_queue_inputs(&fixture, capacity).unwrap();
    let mut active_queue = Vec::new();
    vyre_primitives::wire::unpack_u32_slice_into(
        &inputs[ACTIVE_QUEUE_ACTIVE_QUEUE_INDEX],
        capacity as usize,
        "active queue test",
        &mut active_queue,
    )
    .unwrap();
    let mut queue_len = Vec::new();
    vyre_primitives::wire::unpack_u32_slice_into(
        &inputs[ACTIVE_QUEUE_LEN_INDEX],
        1,
        "active queue len test",
        &mut queue_len,
    )
    .unwrap();

    assert_eq!(inputs.len(), 6);
    assert_eq!(queue_len, vec![fixture.stats.active_sources as u32]);
    assert_eq!(
        active_queue.len(),
        capacity as usize,
        "active queue buffer should be capacity-padded for stable resident dispatch"
    );
    assert_eq!(active_queue[0], 0);
    assert!(
        active_queue[..fixture.stats.active_sources as usize]
            .windows(2)
            .all(|pair| pair[0] < pair[1]),
        "pre-materialized active queue should preserve source order"
    );
    assert_eq!(
        inputs[ACTIVE_QUEUE_FRONTIER_OUT_INDEX],
        vyre_primitives::wire::pack_u32_slice(&fixture.frontier_out_seed)
    );
    assert_eq!(
        inputs[ACTIVE_QUEUE_EDGE_OFFSETS_INDEX],
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_offsets)
    );
    assert_eq!(
        inputs[ACTIVE_QUEUE_EDGE_TARGETS_INDEX],
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_targets)
    );
    assert_eq!(
        inputs[ACTIVE_QUEUE_EDGE_KIND_INDEX],
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_kind_mask)
    );
}

#[test]
fn ifds_queue_materialize_prepare_builds_parallel_sparse_sequence() {
    let prepared = prepare_ifds_skewed_queue_materialize_step(None).unwrap();

    assert_eq!(prepared.reset_program.workgroup_size(), [256, 1, 1]);
    assert_eq!(prepared.queue_program.workgroup_size(), [256, 1, 1]);
    assert_eq!(prepared.traverse_program.workgroup_size(), [256, 1, 1]);
    assert!(prepared.row_strided_traverse);
    assert_eq!(
        prepared.traverse_grid,
        vyre_primitives::graph::csr_queue_strided::csr_queue_strided_forward_dispatch_grid(
            prepared.queue_capacity
        )
    );
    assert_eq!(
        prepared.queue_program.buffers()[0].name.as_ref(),
        "frontier_in"
    );
    assert_eq!(
        prepared.queue_program.buffers()[0].count as usize,
        FRONTIER_WORDS
    );
    assert_eq!(prepared.stats.nodes, NODE_COUNT);
    assert_eq!(prepared.inputs.len(), 7);
    assert_eq!(
        prepared.inputs[QUEUE_FRONTIER_IN_INDEX].len(),
        FRONTIER_WORDS * 4
    );
    assert_eq!(
        prepared.inputs[QUEUE_ACTIVE_QUEUE_INDEX].len(),
        prepared.queue_capacity as usize * std::mem::size_of::<u32>()
    );
    assert_eq!(prepared.baseline_output.len(), FRONTIER_WORDS * 4);
    assert!(u64::from(prepared.queue_capacity) >= prepared.stats.active_sources);
    assert!(
        prepared.queue_capacity < prepared.stats.nodes / 32,
        "queue capacity should stay sparse relative to the full node-grid launch"
    );
    assert!(prepared.stats.allowed_edges_from_active > 0);
    assert!(prepared.input_bytes_total > u64::from(NODE_COUNT) * 12);
}

#[test]
fn ifds_active_queue_prepare_builds_sparse_traversal_program() {
    let prepared = prepare_ifds_skewed_active_queue_step(None).unwrap();

    assert_eq!(prepared.traverse_program.workgroup_size(), [256, 1, 1]);
    assert!(prepared.row_strided_traverse);
    assert_eq!(
        prepared.traverse_grid,
        vyre_primitives::graph::csr_queue_strided::csr_queue_strided_forward_dispatch_grid(
            prepared.queue_capacity
        )
    );
    assert_eq!(prepared.stats.nodes, NODE_COUNT);
    assert_eq!(prepared.inputs.len(), 6);
    assert_eq!(
        prepared.inputs[ACTIVE_QUEUE_ACTIVE_QUEUE_INDEX].len(),
        prepared.queue_capacity as usize * std::mem::size_of::<u32>()
    );
    assert_eq!(
        prepared.inputs[ACTIVE_QUEUE_LEN_INDEX],
        vyre_primitives::wire::pack_u32_slice(&[prepared.stats.active_sources as u32])
    );
    assert_eq!(prepared.baseline_output.len(), FRONTIER_WORDS * 4);
    assert!(u64::from(prepared.queue_capacity) >= prepared.stats.active_sources);
    assert!(prepared.queue_capacity < prepared.stats.nodes / 32);
    assert!(prepared.stats.allowed_edges_from_active > 0);
}

#[test]
fn ifds_queue_reset_program_clears_len_and_frontier_out() {
    let program = ifds_queue_reset_program(128);

    assert_eq!(program.workgroup_size(), [256, 1, 1]);
    assert_eq!(program.buffers().len(), 2);
    assert_eq!(program.buffers()[0].name.as_ref(), "queue_len");
    assert_eq!(program.buffers()[0].binding, 0);
    assert_eq!(program.buffers()[1].name.as_ref(), "frontier_out");
    assert_eq!(program.buffers()[1].binding, 1);
    assert_eq!(program.buffers()[1].count, 128);
}

#[test]
fn ifds_queue_closure_inputs_allocate_ping_pong_queues_and_seed_accumulator() {
    let fixture = build_ifds_skewed_fixture(4096).unwrap();
    let inputs = ifds_queue_closure_inputs(&fixture, fixture.stats.nodes).unwrap();

    assert_eq!(inputs.len(), 11);
    assert_eq!(
        inputs[QUEUE_CLOSURE_SEED_FRONTIER_INDEX],
        vyre_primitives::wire::pack_u32_slice(&fixture.frontier_in)
    );
    assert_eq!(
        inputs[QUEUE_CLOSURE_SEED_LEN_INDEX],
        vyre_primitives::wire::pack_u32_slice(&[fixture.stats.active_sources as u32])
    );
    let mut seed_queue = Vec::new();
    vyre_primitives::wire::unpack_u32_slice_into(
        &inputs[QUEUE_CLOSURE_SEED_QUEUE_INDEX],
        fixture.stats.active_sources as usize,
        "queue closure seed queue test",
        &mut seed_queue,
    )
    .unwrap();
    assert_eq!(seed_queue.len(), fixture.stats.active_sources as usize);
    assert_eq!(seed_queue[0], 0);
    assert!(
        seed_queue.windows(2).all(|pair| pair[0] < pair[1]),
        "pre-materialized queue closure seed queue should preserve source order"
    );
    assert_eq!(
        inputs[QUEUE_CLOSURE_ACCUMULATOR_INDEX],
        inputs[QUEUE_CLOSURE_SEED_FRONTIER_INDEX]
    );
    assert_eq!(
        inputs[QUEUE_CLOSURE_QUEUE_A_INDEX].len(),
        fixture.stats.nodes as usize * std::mem::size_of::<u32>()
    );
    assert_eq!(
        inputs[QUEUE_CLOSURE_QUEUE_B_INDEX].len(),
        fixture.stats.nodes as usize * std::mem::size_of::<u32>()
    );
    assert!(inputs[QUEUE_CLOSURE_QUEUE_A_INDEX]
        .iter()
        .all(|byte| *byte == 0));
    assert!(inputs[QUEUE_CLOSURE_QUEUE_B_INDEX]
        .iter()
        .all(|byte| *byte == 0));
    assert_eq!(
        inputs[QUEUE_CLOSURE_LEN_A_INDEX],
        vyre_primitives::wire::pack_u32_slice(&[0])
    );
    assert_eq!(
        inputs[QUEUE_CLOSURE_LEN_B_INDEX],
        vyre_primitives::wire::pack_u32_slice(&[0])
    );
    assert_eq!(
        inputs[QUEUE_CLOSURE_EDGE_OFFSETS_INDEX],
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_offsets)
    );
    assert_eq!(
        inputs[QUEUE_CLOSURE_EDGE_TARGETS_INDEX],
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_targets)
    );
    assert_eq!(
        inputs[QUEUE_CLOSURE_EDGE_KIND_INDEX],
        vyre_primitives::wire::pack_u32_slice(&fixture.edge_kind_mask)
    );
}

#[test]
fn ifds_queue_closure_reset_program_restores_accumulator_and_clears_lengths() {
    let program = ifds_queue_closure_reset_program(128, 7, 256);

    assert_eq!(program.workgroup_size(), [256, 1, 1]);
    assert_eq!(program.buffers().len(), 7);
    assert_eq!(program.buffers()[0].name.as_ref(), "frontier_seed");
    assert_eq!(program.buffers()[1].name.as_ref(), "seed_queue");
    assert_eq!(program.buffers()[2].name.as_ref(), "seed_len");
    assert_eq!(program.buffers()[3].name.as_ref(), "active_queue");
    assert_eq!(program.buffers()[4].name.as_ref(), "accumulator");
    assert_eq!(program.buffers()[5].name.as_ref(), "queue_a_len");
    assert_eq!(program.buffers()[6].name.as_ref(), "queue_b_len");
    assert_eq!(program.buffers()[1].count, 7);
    assert_eq!(program.buffers()[3].count, 256);
    assert_eq!(program.buffers()[4].count, 128);
}

#[test]
fn ifds_queue_closure_prepare_builds_delta_fixpoint_sequence() {
    let prepared = prepare_ifds_skewed_queue_closure(None).unwrap();

    assert_eq!(prepared.reset_program.workgroup_size(), [256, 1, 1]);
    assert_eq!(prepared.clear_len_program.workgroup_size(), [1, 1, 1]);
    assert_eq!(prepared.delta_program.workgroup_size(), [256, 1, 1]);
    assert_eq!(
        prepared.reset_program.buffers()[0].name.as_ref(),
        "frontier_seed"
    );
    assert_eq!(
        prepared.reset_program.buffers()[0].count as usize,
        FRONTIER_WORDS
    );
    assert_eq!(prepared.stats.nodes, NODE_COUNT);
    assert_eq!(
        prepared.seed_queue_len,
        prepared.stats.active_sources as u32
    );
    assert_eq!(prepared.queue_capacity, prepared.max_wave_queue_len);
    assert!(prepared.queue_capacity < NODE_COUNT);
    assert_eq!(prepared.inputs.len(), 11);
    assert_eq!(
        prepared.inputs[QUEUE_CLOSURE_SEED_QUEUE_INDEX].len(),
        prepared.seed_queue_len as usize * std::mem::size_of::<u32>()
    );
    assert_eq!(
        prepared.inputs[QUEUE_CLOSURE_QUEUE_A_INDEX].len(),
        prepared.queue_capacity as usize * std::mem::size_of::<u32>()
    );
    assert_eq!(prepared.baseline_output.len(), FRONTIER_WORDS * 4);
    assert_eq!(prepared.closure_changed, 1);
    assert!(prepared.closure_iterations > 0);
    assert!(prepared.closure_iterations <= closure::CLOSURE_MAX_ITERS);
    assert!(prepared.total_queue_pops >= prepared.stats.active_sources);
    assert!(prepared.max_wave_queue_len >= prepared.stats.active_sources as u32);
}

fn generated_ifds_fixture(
    node_count: u32,
    seeds: &[u32],
    edges: &[(u32, u32, bool)],
) -> IfdsSkewedFixture {
    let frontier_words = node_count.div_ceil(32);
    let mut frontier_in = vec![0_u32; frontier_words as usize];
    for &seed in seeds {
        let node = seed % node_count;
        frontier_in[(node / 32) as usize] |= 1_u32 << (node % 32);
    }
    let active_sources = frontier_in
        .iter()
        .map(|word| u64::from(word.count_ones()))
        .sum::<u64>();
    let mut by_source = vec![Vec::<(u32, u32)>::new(); node_count as usize];
    for &(src, dst, allowed) in edges {
        let kind_mask = if allowed { IFDS_REACH_MASK } else { 0 };
        by_source[(src % node_count) as usize].push((dst % node_count, kind_mask));
    }

    let mut edge_offsets = Vec::with_capacity(node_count as usize + 1);
    let mut edge_targets = Vec::new();
    let mut edge_kind_mask = Vec::new();
    let mut allowed_edges_from_active = 0_u64;
    let mut filtered_edges_from_active = 0_u64;
    let mut max_degree = 0_u32;
    let mut high_degree_sources = 0_u64;
    edge_offsets.push(0);
    for src in 0..node_count {
        let src_edges = &by_source[src as usize];
        let degree = src_edges.len() as u32;
        max_degree = max_degree.max(degree);
        high_degree_sources += u64::from(degree >= 24);
        let src_word = (src / 32) as usize;
        let src_bit = 1_u32 << (src % 32);
        let src_active = frontier_in[src_word] & src_bit != 0;
        for &(dst, kind_mask) in src_edges {
            if src_active && kind_mask & IFDS_REACH_MASK != 0 {
                allowed_edges_from_active += 1;
            } else if src_active {
                filtered_edges_from_active += 1;
            }
            edge_targets.push(dst);
            edge_kind_mask.push(kind_mask);
        }
        edge_offsets.push(edge_targets.len() as u32);
    }

    IfdsSkewedFixture {
        nodes: vec![0; node_count as usize],
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        node_tags: vec![0; node_count as usize],
        frontier_in,
        frontier_out_seed: vec![0; frontier_words as usize],
        stats: IfdsSkewedStats {
            nodes: node_count,
            edges: edges.len() as u32,
            frontier_words,
            active_sources,
            allowed_edges_from_active,
            filtered_edges_from_active,
            output_words_set: 0,
            max_degree,
            high_degree_sources,
        },
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 4_096,
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::Off
        )),
        ..ProptestConfig::default()
    })]

    #[test]
    fn ifds_queue_closure_compact_capacity_matches_full_capacity_for_generated_graphs(
        node_count in 1_u32..=128,
        seeds in proptest::collection::vec(any::<u32>(), 0..=24),
        edges in proptest::collection::vec((any::<u32>(), any::<u32>(), any::<bool>()), 0..=256),
    ) {
        let fixture = generated_ifds_fixture(node_count, &seeds, &edges);
        let max_iters = node_count.saturating_add(1);
        let full = ifds_skewed_queue_closure_oracle(&fixture, max_iters, fixture.stats.nodes)?;
        let bitset = ifds_skewed_closure_oracle(&fixture, max_iters);
        let compact_capacity = full
            .max_wave_queue_len
            .max(fixture.stats.active_sources as u32)
            .max(1);
        let compact = ifds_skewed_queue_closure_oracle(&fixture, max_iters, compact_capacity)?;
        let compact_inputs = ifds_queue_closure_inputs(&fixture, compact_capacity)?;

        prop_assert_eq!(&full.output, &bitset.output);
        prop_assert_eq!(&full.output, &compact.output);
        prop_assert_eq!(full.iterations, compact.iterations);
        prop_assert_eq!(full.changed, compact.changed);
        prop_assert_eq!(full.total_queue_pops, compact.total_queue_pops);
        prop_assert_eq!(full.max_wave_queue_len, compact.max_wave_queue_len);
        prop_assert_eq!(
            compact_inputs[QUEUE_CLOSURE_QUEUE_A_INDEX].len(),
            compact_capacity as usize * std::mem::size_of::<u32>()
        );
        prop_assert_eq!(
            compact_inputs[QUEUE_CLOSURE_QUEUE_B_INDEX].len(),
            compact_capacity as usize * std::mem::size_of::<u32>()
        );

        if fixture.stats.active_sources > 0 || full.max_wave_queue_len > 0 {
            prop_assert!(
                ifds_skewed_queue_closure_oracle(
                    &fixture,
                    max_iters,
                    compact_capacity.saturating_sub(1)
                )
                .is_err()
            );
        }
    }
}

#[test]
fn ifds_skewed_closure_oracle_expands_seed_frontier() {
    let fixture = build_ifds_skewed_fixture(4096).unwrap();
    let oracle = ifds_skewed_closure_oracle(&fixture, closure::CLOSURE_MAX_ITERS);

    assert_eq!(oracle.output.len(), fixture.frontier_in.len());
    assert_eq!(oracle.changed, 1);
    assert!(oracle.iterations > 0);
    assert!(oracle.iterations <= closure::CLOSURE_MAX_ITERS);
    assert!(
        oracle.output_words_set
            >= fixture
                .frontier_in
                .iter()
                .filter(|word| **word != 0)
                .count() as u64
    );
    let launch_waves = ifds_skewed_launch_wave_iterations(&fixture, closure::CLOSURE_MAX_ITERS);
    assert!(launch_waves >= oracle.iterations);
    assert!(launch_waves <= closure::CLOSURE_MAX_ITERS);
}

#[test]
fn ifds_skewed_closure_resident_inputs_keep_immutable_seed() {
    let fixture = build_ifds_skewed_fixture(4096).unwrap();
    let inputs = super::fixture::ifds_closure_resident_inputs(&fixture);

    assert_eq!(inputs.len(), 8);
    assert_eq!(inputs[5].len(), fixture.frontier_in.len() * 4);
    assert_eq!(inputs[5], inputs[6]);
    assert_eq!(inputs[7], vyre_primitives::wire::pack_u32_slice(&[0]));
}

#[test]
fn ifds_skewed_closure_prepare_builds_resident_fixpoint_program() {
    let prepared = closure::prepare_ifds_skewed_closure(None).unwrap();

    assert_eq!(prepared.program.workgroup_size(), [256, 1, 1]);
    assert_eq!(prepared.reset_program.workgroup_size(), [256, 1, 1]);
    assert_eq!(prepared.stats.nodes, NODE_COUNT);
    assert_eq!(prepared.inputs.len(), 7);
    assert_eq!(prepared.inputs[5].len(), FRONTIER_WORDS * 4);
    assert_eq!(prepared.baseline_outputs.len(), 2);
    assert_eq!(prepared.baseline_outputs[0].len(), FRONTIER_WORDS * 4);
    assert_eq!(prepared.baseline_outputs[1].len(), 4);
    assert_eq!(prepared.closure_changed, 1);
    assert!(prepared.closure_iterations > 0);
    assert!(prepared.dispatch_iterations >= prepared.closure_iterations);
    assert!(prepared.dispatch_iterations < closure::CLOSURE_MAX_ITERS);
    assert!(prepared.input_bytes_total > u64::from(NODE_COUNT) * 20);
}
