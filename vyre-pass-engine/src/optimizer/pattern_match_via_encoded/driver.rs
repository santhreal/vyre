//! Semantic execution and decoding for the algebraic identity kernel.

use vyre_foundation::ir::Program;
use vyre_megakernel::{SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor};

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
    /// Semantic execution or output decoding failed.
    Semantic(SemanticExecutionError),
}

impl std::fmt::Display for PatternMatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encode(err) => write!(f, "gpu_algebraic_identities encode error: {err:?}"),
            Self::Semantic(err) => write!(
                f,
                "gpu_algebraic_identities semantic execution error: {err}"
            ),
        }
    }
}

impl std::error::Error for PatternMatchError {}

/// Run V1 algebraic-identity pattern-match against `program`. Returns
/// the rewritten Program with simplified BinOps.
pub fn gpu_algebraic_identities(
    program: Program,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
) -> Result<Program, PatternMatchError> {
    let arena = encode_expr_arena(&program).map_err(PatternMatchError::Encode)?;
    if arena.expr_count == 0 {
        return Ok(program);
    }
    let mut scratch = PatternKernelScratch::default();
    let mut actions = Vec::with_capacity(arena.expr_count as usize);
    run_pattern_kernel_with_scratch_into(&arena, executor, policy, &mut scratch, &mut actions)
        .map_err(PatternMatchError::Semantic)?;
    Ok(rewrite_program_with_actions(program, &actions))
}

#[cfg(test)]
fn run_pattern_kernel_into(
    arena: &ExprArenaEncoding,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    actions: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = PatternKernelScratch::default();
    run_pattern_kernel_with_scratch_into(arena, executor, policy, &mut scratch, actions)
}

fn run_pattern_kernel_with_scratch_into(
    arena: &ExprArenaEncoding,
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    scratch: &mut PatternKernelScratch,
    actions: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    crate::optimizer::run_encoded_analysis_kernel(
        arena,
        executor,
        policy,
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
        semantic_test_policy, single_lit_u32_arena as one_expr_arena, FixedOutputExecutor,
    };
    use crate::optimizer::pattern_match_via_encoded::rewrite_action;

    #[test]
    fn kernel_decodes_canonical_action_output() {
        let executor = FixedOutputExecutor {
            pass: "pattern-match",
            expected_inputs: 4,
            outputs: vec![u32_slice_to_le_bytes(&[rewrite_action::NONE])],
        };
        let mut actions = Vec::with_capacity(4);
        run_pattern_kernel_into(
            &one_expr_arena(),
            &executor,
            &semantic_test_policy(),
            &mut actions,
        )
        .expect("semantic execution succeeds");
        assert_eq!(actions, vec![rewrite_action::NONE]);
    }

    #[test]
    fn kernel_rejects_trailing_action_bytes() {
        let executor = FixedOutputExecutor {
            pass: "pattern-match",
            expected_inputs: 4,
            outputs: vec![vec![0, 0, 0, 0, 1]],
        };
        let mut actions = Vec::new();
        let err = run_pattern_kernel_into(
            &one_expr_arena(),
            &executor,
            &semantic_test_policy(),
            &mut actions,
        )
        .expect_err("trailing bytes must be rejected");
        assert!(matches!(err, SemanticExecutionError::Backend(_)));
    }
}
