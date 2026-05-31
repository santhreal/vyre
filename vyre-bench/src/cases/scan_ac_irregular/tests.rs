use super::*;

#[test]
fn irregular_haystack_plants_unaligned_varied_literals() {
    let (haystack, planted) = build_irregular_haystack(128 * 1024);
    let ac = classic_ac_compile(PATTERNS);
    let pattern_lengths = pattern_lengths().unwrap();
    let matches = cpu_bounded_range_matches(&ac, &pattern_lengths, &haystack);

    assert!(planted > PATTERNS.len() as u32);
    assert!(matches.len() >= planted as usize);
    assert!(matches.iter().any(|hit| hit.start % 4 != 0));
    assert!(
        pattern_lengths.iter().max().copied().unwrap() > 16,
        "fixture must include long parser/security literals"
    );
}

#[test]
fn prepare_builds_cuda_compatible_bounded_ranges_program() {
    let prepared = prepare_scan_ac_irregular(None).unwrap();

    assert_eq!(prepared.program.workgroup_size(), [128, 1, 1]);
    assert_eq!(prepared.inputs.len(), 7);
    assert_eq!(MATCH_COUNT_INPUT_INDEX, prepared.inputs.len() - 1);
    assert_eq!(
        match_triples_output_bytes(MAX_MATCHES).unwrap(),
        MAX_MATCHES as usize * 12
    );
    assert_eq!(prepared.stats.haystack_bytes, HAYSTACK_BYTES as u32);
    assert_eq!(prepared.stats.patterns, PATTERNS.len() as u32);
    assert!(prepared.stats.expected_matches > 0);
    assert!(prepared.stats.expected_matches <= prepared.stats.max_matches);
}

#[test]
fn aho_cpu_baseline_matches_classic_bounded_ranges_oracle() {
    let (haystack, _) = build_irregular_haystack(32 * 1024);
    let ac = classic_ac_compile(PATTERNS);
    let pattern_lengths = pattern_lengths().unwrap();
    let mut classic = cpu_bounded_range_matches(&ac, &pattern_lengths, &haystack);
    classic.sort_unstable();

    let aho = cpu_aho_overlapping_matches(PATTERNS, &haystack).unwrap();

    assert_eq!(aho, classic);
}

#[test]
fn bounded_ranges_program_reference_eval_matches_cpu_oracle() {
    let (haystack, _) = build_irregular_haystack(4 * 1024);
    let ac = classic_ac_compile(PATTERNS);
    let pattern_lengths = pattern_lengths().unwrap();
    let mut expected = cpu_bounded_range_matches(&ac, &pattern_lengths, &haystack);
    expected.sort_unstable();
    let program =
        try_build_ac_bounded_ranges_program_ext(&ac.dfa, pattern_lengths.len() as u32, 4096, false)
            .unwrap();
    let inputs = scan_ac_inputs(&ac, &pattern_lengths, &haystack);
    let values = inputs
        .into_iter()
        .map(vyre_reference::value::Value::from)
        .collect::<Vec<_>>();
    let outputs = vyre_reference::reference_eval(&program, &values)
        .unwrap()
        .into_iter()
        .map(|value| value.to_bytes())
        .collect::<Vec<_>>();
    let actual = decode_scan_outputs(&outputs, "reference").unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn decoded_outputs_canonicalize_atomic_order_variation() {
    let first = Match::new(0, 3, 7);
    let second = Match::new(1, 11, 15);
    let actual = vec![pack_u32_slice(&[2]), encode_match_triples(&[second, first])];

    let decoded = decode_scan_outputs(&actual, "actual").unwrap();
    assert_eq!(decoded, vec![first, second]);
}
