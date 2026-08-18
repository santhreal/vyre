//! Regression and contract suite for CUDA release benchmark workload correctness.
//!
//! # WHY
//! Closes the defect class where:
//! 1. `triple_mask_threshold_count_program` used unreduced global atomic additions,
//!    causing massive GPU memory pipeline serialization on counting workloads such as
//!    `release.egraph_saturation.1m`.
//! 2. Release macro workload cases generated for testing or measurement must produce
//!    exact matching output buffers against `vyre_reference::reference_eval`.

use vyre_bench::cases::release_workloads::{
    build_release_macro_case_for_records, release_macro_program_specs_for_records,
};
use vyre_reference::{reference_eval, value::Value};

#[test]
fn quantified_condition_loops_matches_reference_evaluation() {
    // Verify across varied record counts that quantified condition loops case builds
    // and matches the reference interpreter output bytes exactly.
    for records in [1, 2, 7, 16, 32, 64, 128, 256] {
        let case =
            build_release_macro_case_for_records("release.quantified_condition_loops.1m", records)
                .expect("Fix: quantified condition loops case must build for test record count.");

        let values: Vec<Value> = case.inputs.iter().cloned().map(Value::from).collect();
        let ref_outputs: Vec<Vec<u8>> = reference_eval(&case.program, &values)
            .expect("Fix: reference eval must succeed for quantified condition loops")
            .into_iter()
            .map(|v| v.to_bytes())
            .collect();

        assert_eq!(
            ref_outputs, case.expected_outputs,
            "Fix: quantified condition loops output must match oracle for records={records}."
        );
    }
}
#[test]
fn egraph_saturation_and_triple_mask_programs_match_cpu_oracles() {
    let specs = release_macro_program_specs_for_records(64);
    for spec in specs {
        let case = build_release_macro_case_for_records(spec.id, 64)
            .expect("Fix: release macro case must build for test record count.");

        let values: Vec<Value> = case.inputs.iter().cloned().map(Value::from).collect();
        let ref_outputs: Vec<Vec<u8>> = reference_eval(&case.program, &values)
            .unwrap_or_else(|err| {
                panic!("Fix: reference eval failed for {}: {err}", spec.id);
            })
            .into_iter()
            .map(|v| v.to_bytes())
            .collect();

        assert_eq!(
            ref_outputs, case.expected_outputs,
            "Fix: release workload {} output must match CPU oracle.",
            spec.id
        );
    }
}
