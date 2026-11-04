//! Semantic execution and decoding for the algebraic identity kernel.

use vyre_foundation::ir::Program;
use vyre_libs::dispatch_buffers::{
    decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes,
};
use vyre_megakernel::{SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor};

use super::decode::rewrite_program_with_actions;
use super::program::{build_pattern_match_program, build_pattern_match_program_with_cse};
use super::rewrite_action;
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

/// Run the identity bank that also reads structural equality, against the
/// `arena` a canonical-id table was computed from.
///
/// The rules a canonical column unlocks are the ones whose premise is that two
/// operands are one value: `x - x`, `x ^ x`, `Min x x`, the reflexive
/// comparisons, and the cancellations that reach inside the left child. They
/// are unreachable without it, so this is the entry point the pipeline runs and
/// [`gpu_algebraic_identities`] is the literal-operand subset a caller with no
/// canonical table can still get.
///
/// `arena` and `canonical` must be the pair one [`crate::optimizer::cse_via_encoded::gpu_cse_canonicals`]
/// call returned: the action column is indexed by that arena's Expr ids, and
/// `program` must still have the node structure it was encoded from.
pub fn gpu_algebraic_identities_with_canonicals(
    program: Program,
    arena: &ExprArenaEncoding,
    canonical: &[u32],
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
) -> Result<Program, PatternMatchError> {
    if arena.expr_count == 0 {
        return Ok(program);
    }
    if canonical.len() != arena.expr_count as usize {
        return Err(PatternMatchError::Semantic(
            SemanticExecutionError::InvalidRequest(format!(
                "gpu_algebraic_identities_with_canonicals received {} canonical ids for an arena of \
                 {} Exprs. Fix: pass the arena and canonical table from one gpu_cse_canonicals call",
                canonical.len(),
                arena.expr_count
            )),
        ));
    }
    let mut scratch = PatternKernelScratch::default();
    let mut actions = Vec::with_capacity(arena.expr_count as usize);
    run_cse_pattern_kernel_with_scratch_into(
        arena,
        canonical,
        executor,
        policy,
        &mut scratch,
        &mut actions,
    )
    .map_err(PatternMatchError::Semantic)?;
    Ok(rewrite_program_with_actions(program, &actions))
}

/// Dispatch the canonical-aware kernel and decode its action column.
///
/// `rewrite_action` is a read-write buffer here rather than the single output
/// the literal-only program declares, because that program already spends its
/// one output slot and this one carries the `canonical` column as well. A
/// read-write result is seeded by the request, so the seed is the `NONE` action
/// for every Expr and any row the kernel leaves alone decodes as "no rewrite".
fn run_cse_pattern_kernel_with_scratch_into(
    arena: &ExprArenaEncoding,
    canonical: &[u32],
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    scratch: &mut PatternKernelScratch,
    actions: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let n = arena.expr_count;
    ensure_input_slots(&mut scratch.inputs, 6);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], &arena.kinds);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], &arena.arg0);
    write_u32_slice_le_bytes(&mut scratch.inputs[2], &arena.arg1);
    write_u32_slice_le_bytes(&mut scratch.inputs[3], &arena.arg2);
    let seed = vec![rewrite_action::NONE; n.max(1) as usize];
    write_u32_slice_le_bytes(&mut scratch.inputs[4], &seed);
    write_u32_slice_le_bytes(&mut scratch.inputs[5], canonical);

    let mut execution = vyre_megakernel::execute_single_program(
        executor,
        "pattern-match-cse",
        build_pattern_match_program_with_cse(n),
        &scratch.inputs,
        policy,
    )?;
    if execution.outputs.len() != 1 {
        return Err(SemanticExecutionError::Backend(format!(
            "pattern-match-cse semantic execution expected exactly one rewrite_action output, got \
             {}. Fix: return the canonical graph output",
            execution.outputs.len()
        )));
    }
    decode_u32_output_exact(
        &execution.outputs.remove(0),
        n as usize,
        "pattern-match-cse rewrite_action",
        actions,
    )
    .map_err(|error| {
        SemanticExecutionError::Backend(format!(
            "pattern-match-cse semantic output decoding failed: {error}"
        ))
    })
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
