//! The AC walk reads inside its buffers when its tables hold garbage.
//!
//! Every index the walk derives comes out of a read-only buffer: the next state
//! is a transition-table entry, the output span is a pair of `output_offsets`
//! entries selected by that state, and the pattern length is read at an
//! `output_records` entry. Nothing constrains what a caller puts in any of
//! them, so a table built by something other than this crate's compiler makes
//! the walk index past a buffer end on a device that bounds-checks nothing.
//!
//! Three cases, because no single fixture reaches all three reads. Whole-buffer
//! garbage drives the state and the output span out of range and leaves the
//! record span empty. An `output_records` entry naming no compiled pattern is
//! what reaches the length table, and that read is on the match-emitting
//! bounded-ranges walk rather than the regex DFA's exact-starts walk, so it
//! takes its own program.
//!
//! Together they close the class for the dense
//! `state = transitions[state * 256 + byte]` walk that
//! `pattern::classic_ac::bounded_ranges` owns, since both the regex DFA scan
//! and the bounded-ranges scan are built out of exactly those helpers.
//!
//! No case here catches an index derived from the grid rather than from data;
//! the three grid-varying registry nets cover that.

use vyre_foundation::ir::{BufferDecl, Program};
use vyre_libs_pattern::pattern::classic_ac::build_ac_bounded_ranges_program_with_subgroup_coalesce;
use vyre_reference::value::Value;
// The out-of-range index and the hostile word rewrite are the same two
// derivations the registry out-of-bounds sweep applies, so they have one owner
// rather than one copy per suite.
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
use vyre_test_support::registry_nets::{first_out_of_range_index, hostile_contents};

/// The compiled regex fixture program and the buffer list its registration
/// supplies.
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
fn fixture(haystack: Vec<u8>) -> (Program, Vec<Vec<u8>>) {
    let pipeline = vyre_libs_pattern::pattern::build_regex_dfa_pipeline(&["[a-z]+"], 64, 256)
        .expect("Fix: the regex DFA fixture pattern must compile");
    let buffers = vec![
        haystack,
        vyre_primitives::wire::pack_u32_slice(&pipeline.dfa.transitions),
        vyre_primitives::wire::pack_u32_slice(&pipeline.dfa.output_offsets),
        vyre_primitives::wire::pack_u32_slice(&pipeline.dfa.output_records),
        vyre_primitives::wire::pack_u32_slice(&pipeline.pattern_lengths),
        vec![0u8; 4],
        vec![0u8; 4],
    ];
    (pipeline.program, buffers)
}

/// Run the walk and assert the interpreter saw no access outside a buffer.
fn assert_oob_clean(program: &Program, inputs: &[Value], case: &str) {
    vyre_reference::ReferenceRequest::standard(program, inputs)
        .outputs()
        .unwrap_or_else(|error| {
            panic!(
                "Fix: the AC walk must execute in the reference interpreter with {case}, and an \
                 out-of-bounds access is refused rather than absorbed. Fold each data-derived \
                 index with `vyre_foundation::composition::bounded_index`, or gate the access \
                 with control flow: {error:?}"
            )
        });
}

#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
#[test]
fn a_regex_dfa_scan_over_a_garbage_table_reads_inside_its_buffers() {
    let (program, buffers) = fixture(vec![0u8; 64]);
    let inputs = vyre_reference::reference_inputs(&program, buffers);
    let index = first_out_of_range_index(&program, &inputs);
    assert!(
        index > 0,
        "Fix: the fixture must declare an extent, otherwise the hostile word is zero and indexes \
         the first element of every buffer"
    );
    assert_oob_clean(
        &program,
        &hostile_contents(&inputs, index),
        &format!("every input word holding {index}, which no buffer in the program accepts"),
    );
}

#[test]
fn an_output_record_naming_no_pattern_reads_inside_the_length_table() {
    let patterns: [&[u8]; 1] = [b"abra"];
    let dfa = vyre_libs_pattern::pattern::dfa_compile(&patterns);
    let pattern_count = u32::try_from(patterns.len()).expect("Fix: one pattern must count as u32");
    let max_matches = 8u32;
    // Both appends, because both are shipped. The coalesced one puts a
    // subgroup collective in the record loop, and a bound folded on one path
    // and not the other is the defect this case exists to catch.
    for use_subgroup_coalesce in [false, true] {
        let program = build_ac_bounded_ranges_program_with_subgroup_coalesce(
            &dfa,
            pattern_count,
            max_matches,
            use_subgroup_coalesce,
        );
        assert_eq!(
            program
                .buffers()
                .iter()
                .map(BufferDecl::name)
                .collect::<Vec<_>>(),
            vec![
                "haystack",
                "transitions",
                "output_offsets",
                "output_records",
                "pattern_lengths",
                "haystack_len",
                "match_count",
                "matches",
            ],
            "Fix: the bounded-ranges buffer ABI moved, so the buffers this case marshals no \
             longer reach the reads it exercises"
        );

        let haystack = b"abracadabra\0";
        let past_every_pattern = pattern_count;
        let corrupt_records = vec![past_every_pattern; dfa.output_records.len().max(1)];
        let buffers = vec![
            haystack.to_vec(),
            vyre_primitives::wire::pack_u32_slice(&dfa.transitions),
            vyre_primitives::wire::pack_u32_slice(&dfa.output_offsets),
            vyre_primitives::wire::pack_u32_slice(&corrupt_records),
            vyre_primitives::wire::pack_u32_slice(&[dfa.max_pattern_len]),
            vyre_primitives::wire::pack_u32_slice(&[11]),
            vec![0u8; 4],
            vec![0u8; max_matches as usize * 3 * 4],
        ];

        let inputs = vyre_reference::reference_inputs(&program, buffers);
        assert_oob_clean(
            &program,
            &inputs,
            &format!(
                "an output record naming pattern {past_every_pattern}, which the length table \
                 does not hold, with subgroup coalescing {use_subgroup_coalesce}"
            ),
        );
    }
}
