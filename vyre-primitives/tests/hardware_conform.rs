//! Cat-C hardware intrinsic differential harness.
//!
//! Iterates every canonical `SemanticOperation` whose id begins with
//! `vyre-primitives::hardware::` and asserts the CPU reference matches
//! the declared `expected_output` bit-for-bit. This is the lightweight
//! gate; GPU conform tests run separately through the backend lowering
//! and dispatch suites.

mod gate_fixtures;

use gate_fixtures::run_cpu;
use vyre_primitives::hardware::all_entries;

#[test]
fn hardware_intrinsics_match_expected_output() {
    let entries: Vec<_> = all_entries()
        .filter(|e| e.id.starts_with("vyre-primitives::hardware::"))
        .collect();
    assert!(
        !entries.is_empty(),
        "no intrinsic entries registered  -  feature gates or registration broken"
    );
    for entry in entries {
        let inputs = (entry.test_inputs.expect("test_inputs required"))();
        let expected = (entry.expected_output.expect("expected_output required"))();
        assert_eq!(
            inputs.len(),
            expected.len(),
            "{}: fixture count mismatch",
            entry.id
        );
        for (case, (case_inputs, case_expected)) in inputs.iter().zip(expected.iter()).enumerate() {
            let got = run_cpu(&entry, case_inputs);
            assert_eq!(
                &got, case_expected,
                "{} case {}: CPU ref drifted from expected_output",
                entry.id, case
            );
        }
    }
}
