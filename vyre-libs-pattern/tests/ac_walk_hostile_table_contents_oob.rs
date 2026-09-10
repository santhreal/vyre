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

/// The first element index neither the program's declared extents nor the
/// buffers it was handed accept.
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
fn first_out_of_range_index(program: &Program, inputs: &[Value]) -> u32 {
    let declared = program
        .buffers()
        .iter()
        .map(BufferDecl::count)
        .max()
        .unwrap_or(0);
    let supplied = inputs
        .iter()
        .map(|input| match input {
            Value::Bytes(bytes) => u32::try_from(bytes.len() / 4).unwrap_or(u32::MAX),
            _ => 1,
        })
        .max()
        .unwrap_or(0);
    declared.max(supplied)
}

/// Every four-byte word of every input buffer replaced with `index`.
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
fn hostile_contents(inputs: &[Value], index: u32) -> Vec<Value> {
    let word = index.to_le_bytes();
    inputs
        .iter()
        .map(|input| match input {
            Value::Bytes(bytes) => {
                let mut hostile = bytes.to_vec();
                for chunk in hostile.chunks_exact_mut(4) {
                    chunk.copy_from_slice(&word);
                }
                Value::Bytes(hostile.into())
            }
            Value::U32(_) => Value::U32(index),
            other => other.clone(),
        })
        .collect()
}

/// Run the walk and assert the interpreter saw no access outside a buffer.
fn assert_oob_clean(program: &Program, inputs: &[Value], case: &str) {
    let (_outputs, report) = vyre_reference::reference_eval_oob_report(program, inputs)
        .unwrap_or_else(|error| {
            panic!("Fix: the AC walk must execute in the reference interpreter: {error:?}")
        });
    assert_eq!(
        (report.oob_loads, report.oob_stores, report.oob_atomics),
        (0, 0, 0),
        "Fix: the AC walk accessed a buffer out of bounds with {case}. Fold each data-derived \
         index with `vyre_foundation::composition::bounded_index`, or gate the access with \
         control flow."
    );
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
