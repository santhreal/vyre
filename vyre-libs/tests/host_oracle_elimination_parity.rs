//! Parity tests for operation registration expected-output byte constants.
//!
//! Operation registrations must carry exact byte constants for expected outputs
//! rather than evaluating dynamic host oracles at registration time.
//! This test executes registered operations through the pure Rust reference interpreter
//! (`vyre_reference::reference_eval`) and asserts that returned buffers match the
//! registered `expected_output` byte constants bit-for-bit (or within floating-point
//! tolerance for float buffers).
//!
//! The test enumerates the catalog dynamically at run time so newly registered
//! operations with fixtures are automatically verified.

use vyre::ir::DataType;
use vyre_foundation::operation::OperationRegistry;
use vyre_reference::value::Value;

#[test]
fn registered_operations_match_reference_interpreter_outputs() {
    let registry = OperationRegistry::global();
    let mut verified_count = 0usize;

    for op in registry.iter() {
        let (Some(build), Some(test_inputs), Some(expected_output)) =
            (op.build, op.test_inputs, op.expected_output)
        else {
            continue;
        };

        let program = build();
        let input_cases = test_inputs();
        let expected_cases = expected_output();

        assert_eq!(
            input_cases.len(),
            expected_cases.len(),
            "Operation {}: test_inputs count ({}) must match expected_output count ({})",
            op.id,
            input_cases.len(),
            expected_cases.len()
        );

        let output_indices: Vec<usize> = program
            .output_buffer_indices()
            .iter()
            .map(|&index| index as usize)
            .collect();

        let tolerance = vyre_foundation::fp_parity::effective_tolerance(op.id, &program);

        for (case_idx, (inputs, expected_buffers)) in input_cases
            .into_iter()
            .zip(expected_cases.into_iter())
            .enumerate()
        {
            let ref_inputs: Vec<Value> = inputs.into_iter().map(Value::from).collect();
            let actual_buffers = vyre_reference::reference_eval(&program, &ref_inputs)
                .unwrap_or_else(|err| {
                    panic!(
                        "Reference execution failed for operation {} (case {case_idx}): {err}",
                        op.id
                    )
                })
                .into_iter()
                .map(|val| val.to_bytes())
                .collect::<Vec<Vec<u8>>>();

            assert_eq!(
                actual_buffers.len(),
                expected_buffers.len(),
                "Operation {} (case {case_idx}): reference output buffer count ({}) does not match expected ({})",
                op.id,
                actual_buffers.len(),
                expected_buffers.len()
            );

            for (decl_index, &output_pos) in output_indices.iter().enumerate() {
                let actual = &actual_buffers[decl_index];
                let expected = &expected_buffers[decl_index];
                let is_f32 = program.buffers()[output_pos].element() == DataType::F32;

                if is_f32 && tolerance > 0 {
                    assert!(
                        vyre_foundation::fp_parity::f32_buffer_matches(
                            expected, actual, tolerance,
                        ),
                        "Operation {} (case {case_idx}, buffer {decl_index}): float output mismatch within tolerance {tolerance} ULP",
                        op.id
                    );
                } else {
                    assert_eq!(
                        actual, expected,
                        "Operation {} (case {case_idx}, buffer {decl_index}): reference output does not match expected byte constants",
                        op.id
                    );
                }
            }
            verified_count += 1;
        }
    }

    assert!(
        verified_count > 0,
        "Expected to verify at least one operation from the catalog, verified {verified_count}"
    );
}

#[test]
fn affected_operations_are_explicitly_verified() {
    let registry = OperationRegistry::global();
    let mut verified_affected = 0usize;

    for op in registry.iter() {
        let is_target = op.id == "vyre-libs::llm::logit_adjust"
            || op.id == "vyre-libs::llm::nucleus_select"
            || op.id == "vyre-libs::llm::sample_token"
            || op.id == "vyre-libs::security::flows_to_to_sink"
            || op.id == "vyre-libs::security::taint_pollution"
            || op.id == "vyre-libs::security::flows_to_with_sanitizer";

        if !is_target {
            continue;
        }

        let (Some(build), Some(test_inputs), Some(expected_output)) =
            (op.build, op.test_inputs, op.expected_output)
        else {
            panic!(
                "Target operation {} must have build, test_inputs, and expected_output",
                op.id
            );
        };

        let program = build();
        let input_cases = test_inputs();
        let expected_cases = expected_output();

        for (case_idx, (inputs, expected_buffers)) in input_cases
            .into_iter()
            .zip(expected_cases.into_iter())
            .enumerate()
        {
            let ref_inputs: Vec<Value> = inputs.into_iter().map(Value::from).collect();
            let actual_buffers = vyre_reference::reference_eval(&program, &ref_inputs)
                .unwrap_or_else(|err| {
                    panic!(
                        "Reference execution failed for operation {} (case {case_idx}): {err}",
                        op.id
                    )
                })
                .into_iter()
                .map(|val| val.to_bytes())
                .collect::<Vec<Vec<u8>>>();

            assert_eq!(
                actual_buffers, expected_buffers,
                "Target operation {} (case {case_idx}): actual reference output must equal expected byte constants exactly",
                op.id
            );
        }
        verified_affected += 1;
    }

    let _ = verified_affected;
}
