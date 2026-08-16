//! Running the kernel and turning its answer back into a program.
//!
//! The arena encoding goes in, one dispatch produces an action per Expr, and
//! the decoder applies it. The dispatcher is injected, so the same kernel runs
//! on every backend and nothing here escapes to a host reference path.

use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

use super::decode::rewrite_program_with_actions;
use super::program::build_pattern_match_program;
use crate::optimizer::encode::EncodeError;
use crate::optimizer::expr_arena::{encode_expr_arena, ExprArenaEncoding};

#[derive(Debug, Default)]
struct PatternKernelScratch {
    inputs: Vec<Vec<u8>>,
}

/// Errors surfaced by `gpu_algebraic_identities`.
#[derive(Debug)]
pub enum PatternMatchError {
    /// Expression-arena encoding failed.
    Encode(EncodeError),
    /// Backend dispatch or output decoding failed.
    Dispatch(DispatchError),
}

impl std::fmt::Display for PatternMatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(err) => write!(f, "gpu_algebraic_identities encode error: {err:?}"),
            Self::Dispatch(err) => write!(f, "gpu_algebraic_identities dispatch error: {err}"),
        }
    }
}

impl std::error::Error for PatternMatchError {}

/// Run V1 algebraic-identity pattern-match against `program`. Returns
/// the rewritten Program with simplified BinOps.
pub fn gpu_algebraic_identities(
    program: Program,
    dispatcher: &dyn ProgramDispatcher,
) -> Result<Program, PatternMatchError> {
    let arena = encode_expr_arena(&program).map_err(PatternMatchError::Encode)?;
    if arena.expr_count == 0 {
        return Ok(program);
    }
    let mut scratch = PatternKernelScratch::default();
    let mut actions = Vec::with_capacity(arena.expr_count as usize);
    run_pattern_kernel_with_scratch_into(&arena, dispatcher, &mut scratch, &mut actions)
        .map_err(PatternMatchError::Dispatch)?;
    Ok(rewrite_program_with_actions(program, &actions))
}

#[cfg(test)]
fn run_pattern_kernel_into(
    arena: &ExprArenaEncoding,
    dispatcher: &dyn ProgramDispatcher,
    actions: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut scratch = PatternKernelScratch::default();
    run_pattern_kernel_with_scratch_into(arena, dispatcher, &mut scratch, actions)
}

fn run_pattern_kernel_with_scratch_into(
    arena: &ExprArenaEncoding,
    dispatcher: &dyn ProgramDispatcher,
    scratch: &mut PatternKernelScratch,
    actions: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    crate::optimizer::run_encoded_analysis_kernel(
        arena,
        dispatcher,
        &mut scratch.inputs,
        actions,
        build_pattern_match_program,
        "pattern-match",
        "rewrite_action",
    )
}

#[cfg(test)]
/// WHY: `run_pattern_kernel_into`, `run_pattern_kernel_with_scratch_into` and
/// `PatternKernelScratch` are private to this module and no integration test
/// can reach them. They are where a dispatch that returns the wrong number of
/// outputs, or trailing bytes past the action column, has to be refused rather
/// than decoded, and where the reused buffers must stay reused.
mod tests {
    use super::*;
    use vyre_libs::dispatch_buffers::u32_slice_to_le_bytes;

    use crate::optimizer::arena_kernel::{
        single_lit_u32_arena as one_expr_arena, FixedOutputDispatcher, GridExpectation,
    };
    use crate::optimizer::pattern_match_via_encoded::rewrite_action;

    fn dispatcher(outputs: Vec<Vec<u8>>) -> FixedOutputDispatcher {
        FixedOutputDispatcher {
            pass: "pattern",
            expected_inputs: 5,
            grid: GridExpectation::SingleWorkgroup,
            outputs,
        }
    }

    #[test]
    fn kernel_into_decodes_exact_actions_into_reused_buffer() {
        let dispatcher = dispatcher(vec![u32_slice_to_le_bytes(&[rewrite_action::NONE])]);
        let mut actions = Vec::with_capacity(4);
        let ptr = actions.as_ptr();
        run_pattern_kernel_into(&one_expr_arena(), &dispatcher, &mut actions)
            .expect("Fix: dispatch succeeds");
        assert_eq!(actions, vec![rewrite_action::NONE]);
        assert_eq!(actions.as_ptr(), ptr);
    }

    #[test]
    fn kernel_with_scratch_reuses_dispatch_and_output_storage() {
        let dispatcher = dispatcher(vec![u32_slice_to_le_bytes(&[rewrite_action::NONE])]);
        let arena = one_expr_arena();
        let mut scratch = PatternKernelScratch::default();
        let mut actions = Vec::with_capacity(1);

        run_pattern_kernel_with_scratch_into(&arena, &dispatcher, &mut scratch, &mut actions)
            .expect("Fix: dispatch succeeds");

        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let actions_capacity = actions.capacity();

        run_pattern_kernel_with_scratch_into(&arena, &dispatcher, &mut scratch, &mut actions)
            .expect("Fix: dispatch succeeds");

        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_capacities
        );
        assert_eq!(actions.capacity(), actions_capacity);
        assert_eq!(actions, vec![rewrite_action::NONE]);
    }

    #[test]
    fn kernel_rejects_extra_outputs() {
        let dispatcher = dispatcher(vec![
            u32_slice_to_le_bytes(&[0]),
            u32_slice_to_le_bytes(&[0]),
        ]);
        let mut actions = Vec::new();
        let err = run_pattern_kernel_into(&one_expr_arena(), &dispatcher, &mut actions)
            .expect_err("extra outputs must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn kernel_rejects_trailing_action_bytes() {
        let dispatcher = dispatcher(vec![vec![0, 0, 0, 0, 1]]);
        let mut actions = Vec::new();
        let err = run_pattern_kernel_into(&one_expr_arena(), &dispatcher, &mut actions)
            .expect_err("trailing bytes must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }
}
