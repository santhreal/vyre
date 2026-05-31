use super::*;

#[test]
fn ifds_skewed_fixture_has_filtered_edges_and_bitset_frontier() {
    let fixture = build_ifds_skewed_fixture(4096).unwrap();
    let oracle = ifds_skewed_cpu_oracle(&fixture);

    assert_eq!(fixture.edge_offsets.len(), 4097);
    assert!(fixture.edge_targets.len() > 4096);
    assert_eq!(fixture.stats.max_degree, 96);
    assert!(fixture.stats.high_degree_sources > 0);
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
fn ifds_skewed_closure_prepare_builds_resident_fixpoint_program() {
    let prepared = closure::prepare_ifds_skewed_closure(None).unwrap();

    assert_eq!(prepared.program.workgroup_size(), [1, 1, 1]);
    assert_eq!(prepared.stats.nodes, NODE_COUNT);
    assert_eq!(prepared.inputs.len(), 7);
    assert_eq!(prepared.seed_frontier_bytes.len(), FRONTIER_WORDS * 4);
    assert_eq!(prepared.baseline_outputs.len(), 2);
    assert_eq!(prepared.baseline_outputs[0].len(), FRONTIER_WORDS * 4);
    assert_eq!(prepared.baseline_outputs[1].len(), 4);
    assert_eq!(prepared.closure_changed, 1);
    assert!(prepared.closure_iterations > 0);
    assert!(prepared.dispatch_iterations >= prepared.closure_iterations);
    assert!(prepared.dispatch_iterations < closure::CLOSURE_MAX_ITERS);
    assert!(prepared.input_bytes_total > u64::from(NODE_COUNT) * 20);
}
