//! Generated NFA plan and table layout contracts.

#[cfg(feature = "matching-nfa")]
#[test]
fn nfa_plan_and_state_major_table_encode_exactly_the_declared_edges() {
    use vyre_libs::scan::nfa::{build_transition_table, plan_shards, try_compile};
    use vyre_libs::nfa::subgroup_nfa::{LANES_PER_SUBGROUP, MAX_STATES_PER_SUBGROUP};

    let pattern_sets: &[&[&str]] = &[
        &[],
        &[""],
        &["a"],
        &["abc", "de", "f"],
        &["alpha", "beta", "gamma", "delta"],
        &["\0", "\u{7f}", "\u{80}", "\u{ff}"],
    ];
    let mut checked_sets = 0usize;

    for patterns in pattern_sets {
        let plan = try_compile(patterns).expect("Fix: generated NFA pattern set should compile.");
        let expected_states = 1 + patterns
            .iter()
            .map(|pattern| pattern.len() as u32)
            .sum::<u32>();
        assert_eq!(plan.num_states, expected_states);
        assert_eq!(plan.accept_states.len(), patterns.len());
        assert_eq!(plan.accept_state_ids.len(), patterns.len());
        assert_eq!(plan.accept_start_anchored, vec![false; patterns.len()]);
        assert_eq!(plan.accept_end_anchored, vec![false; patterns.len()]);

        let table = build_transition_table(patterns);
        assert_eq!(
            table.len(),
            plan.num_states as usize * 256 * LANES_PER_SUBGROUP
        );

        // One destination edge per pattern byte, and no edge anywhere else.
        let mut expected_edges = 0usize;
        let mut state_cursor = 1usize;
        for pattern in *patterns {
            let mut src = 0usize;
            for byte in pattern.bytes() {
                let dst = state_cursor;
                let idx =
                    src * 256 * LANES_PER_SUBGROUP + byte as usize * LANES_PER_SUBGROUP + dst / 32;
                assert_ne!(
                    table[idx] & (1 << (dst % 32)),
                    0,
                    "Fix: state-major table lost edge {src} -{byte}-> {dst}."
                );
                expected_edges += 1;
                src = dst;
                state_cursor += 1;
            }
        }
        assert_eq!(
            table
                .iter()
                .map(|word| word.count_ones() as usize)
                .sum::<usize>(),
            expected_edges,
            "Fix: state-major table holds edges the pattern set does not declare."
        );
        checked_sets += 1;
    }

    let huge = vec!["a".repeat(100); 20];
    let refs = huge.iter().map(String::as_str).collect::<Vec<_>>();
    for shard in plan_shards(&refs) {
        let states = 1 + shard.iter().map(|pattern| pattern.len()).sum::<usize>();
        assert!(
            states <= MAX_STATES_PER_SUBGROUP,
            "Fix: generated NFA shard has {states} states, above subgroup limit."
        );
    }
    assert_eq!(checked_sets, pattern_sets.len());
}
