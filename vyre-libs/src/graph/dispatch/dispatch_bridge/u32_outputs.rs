//! Dispatch helpers that decode a fixed number of u32 output buffers.
//!
//! The expected word count is passed in and checked, so a short readback is an
//! error rather than a silently truncated result.

use crate::dispatch_buffers::{decode_u32_output_exact, HOST_WRAPPER_NODE};
use vyre_foundation::ir::Program;
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
    SingleProgramExecutionOutput,
};

/// Dispatch already-prepared inputs and decode one named u32 output buffer
/// into `out`.
///
/// `buffer` is the Program buffer to read, not a diagnostic label: a program
/// with read-write working storage writes more buffers than the wrapper reads.
pub(crate) fn dispatch_single_u32_output_from_prepared_into(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    program: Program,
    scratch_inputs: &[Vec<u8>],
    expected_output_words: usize,
    buffer: &str,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let output =
        execute_single_program(executor, HOST_WRAPPER_NODE, program, scratch_inputs, policy)?;
    let bytes = named_output(&output, buffer)?;
    decode_u32_output_exact(bytes, expected_output_words, buffer, out)
        .map_err(|error| SemanticExecutionError::Backend(error.to_string()))
}

/// Dispatch already-prepared inputs and decode two named u32 output buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_two_u32_outputs_from_prepared_into(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    program: Program,
    scratch_inputs: &[Vec<u8>],
    first_expected_words: usize,
    first_buffer: &str,
    first_out: &mut Vec<u32>,
    second_expected_words: usize,
    second_buffer: &str,
    second_out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let output =
        execute_single_program(executor, HOST_WRAPPER_NODE, program, scratch_inputs, policy)?;
    let first_bytes = named_output(&output, first_buffer)?;
    let second_bytes = named_output(&output, second_buffer)?;
    decode_u32_output_exact(first_bytes, first_expected_words, first_buffer, first_out)
        .map_err(|error| SemanticExecutionError::Backend(error.to_string()))?;
    decode_u32_output_exact(
        second_bytes,
        second_expected_words,
        second_buffer,
        second_out,
    )
    .map_err(|error| SemanticExecutionError::Backend(error.to_string()))
}

fn named_output<'a>(
    output: &'a SingleProgramExecutionOutput,
    buffer: &str,
) -> Result<&'a [u8], SemanticExecutionError> {
    output.buffer(buffer).ok_or_else(|| {
        SemanticExecutionError::Backend(format!(
            "Fix: {buffer} is not a written Program buffer; the wrapper wrote {:?}.",
            output.output_buffers
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use vyre_megakernel::{
        Digest, SearchBudget, SemanticExecutionOutput, SemanticExecutionRequest,
    };

    use super::*;

    struct ExactOutputExecutor {
        omit_output: bool,
    }

    impl SemanticExecutor for ExactOutputExecutor {
        fn execute(
            &self,
            request: &SemanticExecutionRequest<'_>,
        ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
            let node = &request.logical().graph().nodes()[0];
            assert_eq!(node.inputs.len(), 1);
            assert_eq!(
                request.inputs().get(&node.inputs[0].value).copied(),
                Some([7_u32.to_le_bytes()].concat().as_slice())
            );
            let mut outputs = BTreeMap::new();
            if !self.omit_output {
                outputs.insert(node.outputs[0], 9_u32.to_le_bytes().to_vec());
            }
            Ok(SemanticExecutionOutput {
                artifact: Digest([3; 32]),
                payload: Digest([4; 32]),
                outputs,
            })
        }
    }

    fn policy() -> SemanticExecutionPolicy {
        vyre_test_support::semantic_requests::unknown_policy(
            Digest([0; 32]),
            SearchBudget::new(4, 16, 1, 0, 100),
            1_000_000,
        )
    }

    fn copy_program() -> Program {
        vyre_test_support::pass_programs::logical_copy_program()
    }

    #[test]
    fn single_kernel_consumes_ordered_semantic_output() {
        let mut out = Vec::new();
        dispatch_single_u32_output_from_prepared_into(
            &ExactOutputExecutor { omit_output: false },
            &policy(),
            copy_program(),
            &[7_u32.to_le_bytes().to_vec()],
            1,
            "out",
            &mut out,
        )
        .expect("declared graph output should decode");
        assert_eq!(out, vec![9]);
    }

    #[test]
    fn single_kernel_rejects_missing_graph_value_output() {
        let mut out = Vec::new();
        let error = dispatch_single_u32_output_from_prepared_into(
            &ExactOutputExecutor { omit_output: true },
            &policy(),
            copy_program(),
            &[7_u32.to_le_bytes().to_vec()],
            1,
            "out",
            &mut out,
        )
        .expect_err("missing GraphValueId must fail closed");
        assert!(error.to_string().contains("omitted canonical output value"));
    }
}
