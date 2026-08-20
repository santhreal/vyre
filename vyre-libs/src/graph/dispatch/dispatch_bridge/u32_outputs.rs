//! Dispatch helpers that decode a fixed number of u32 output buffers.
//!
//! The expected word count is passed in and checked, so a short readback is an
//! error rather than a silently truncated result.

use crate::dispatch_buffers::decode_u32_output_exact;
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Dispatch already-prepared inputs and decode exactly one u32 output buffer
/// into `out`.
pub(crate) fn dispatch_single_u32_output_from_prepared_into<D: ProgramDispatcher + ?Sized>(
    dispatcher: &D,
    program: &Program,
    scratch_inputs: &[Vec<u8>],
    expected_output_words: usize,
    context: &str,
    grid_override: Option<[u32; 3]>,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let outputs = dispatcher.dispatch(program, scratch_inputs, grid_override)?;
    let [buf] = match outputs.as_slice() {
        [buf] => [buf],
        _ => {
            return Err(DispatchError::BackendError(format!(
                "Fix: {context} expected exactly one u32 output buffer, got {}.",
                outputs.len()
            )));
        }
    };
    decode_u32_output_exact(buf, expected_output_words, context, out)
}

/// Dispatch already-prepared inputs and decode exactly two u32 output buffers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch_two_u32_outputs_from_prepared_into<D: ProgramDispatcher + ?Sized>(
    dispatcher: &D,
    program: &Program,
    scratch_inputs: &[Vec<u8>],
    first_expected_words: usize,
    first_context: &str,
    first_out: &mut Vec<u32>,
    second_expected_words: usize,
    second_context: &str,
    second_out: &mut Vec<u32>,
    grid_override: Option<[u32; 3]>,
) -> Result<(), DispatchError> {
    let outputs = dispatcher.dispatch(program, scratch_inputs, grid_override)?;
    let [first_buf, second_buf] = match outputs.as_slice() {
        [first_buf, second_buf] => [first_buf, second_buf],
        _ => {
            return Err(DispatchError::BackendError(format!(
                "Fix: {first_context} expected exactly two u32 output buffers, got {}.",
                outputs.len()
            )));
        }
    };
    decode_u32_output_exact(first_buf, first_expected_words, first_context, first_out)?;
    decode_u32_output_exact(
        second_buf,
        second_expected_words,
        second_context,
        second_out,
    )
}
