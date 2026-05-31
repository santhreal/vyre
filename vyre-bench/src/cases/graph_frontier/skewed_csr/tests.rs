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
