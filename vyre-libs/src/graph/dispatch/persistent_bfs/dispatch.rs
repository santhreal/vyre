//! Semantic execution of the complete persistent-BFS algorithm.
//!
//! The semantic executor compiles the complete program and selects its
//! artifact schedule.

use super::scratch::{copy_frontier_seed_into, PersistentBfsGpuScratch};

use crate::dispatch_buffers::decode_u32_output_exact;
use crate::graph::csr_closure_inputs::CsrClosureInputs;
use crate::graph::dispatch::dispatch_bridge::{refresh_keyed_dispatch_inputs, DispatchInput};
use crate::graph::persistent_bfs::{
    plan_persistent_bfs_dispatch, validate_persistent_bfs_changed_flag,
    validate_persistent_bfs_converged_flag,
};
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Dispatcher-backed persistent BFS expansion. Returns the saturated frontier,
/// the sticky changed-flag, and the device converged word.
///
/// The converged word is `1` when the fixpoint was reached within `max_iters`
/// and `0` when the budget was exhausted while the frontier was still growing.
/// A caller that requires an exact closure rejects a `0` converged result
/// instead of silently trusting an under-approximated frontier.
///
/// # Errors
///
/// Propagates dispatch failures and rejects malformed CSR/frontier
/// shapes or truncated readback.
pub fn bfs_expand_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
) -> Result<(Vec<u32>, u32, u32), SemanticExecutionError> {
    let mut frontier = Vec::new();
    let (changed, converged) =
        bfs_expand_via_into(dispatcher, policy, inputs, frontier_in, &mut frontier)?;
    Ok((frontier, changed, converged))
}

/// Dispatcher-backed persistent BFS expansion into caller-owned frontier storage.
///
/// # Errors
///
/// Propagates dispatch failures and rejects malformed CSR/frontier
/// shapes or truncated readback.
pub fn bfs_expand_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
    frontier_out: &mut Vec<u32>,
) -> Result<(u32, u32), SemanticExecutionError> {
    let mut scratch = PersistentBfsGpuScratch::default();
    bfs_expand_via_with_scratch_into(
        dispatcher,
        policy,
        inputs,
        frontier_in,
        &mut scratch,
        frontier_out,
    )
}

/// Dispatcher-backed persistent BFS expansion into caller-owned frontier and dispatch scratch.
///
/// # Errors
///
/// Propagates dispatch failures and rejects malformed CSR/frontier
/// shapes or truncated readback.
pub fn bfs_expand_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    inputs: CsrClosureInputs<'_>,
    frontier_in: &[u32],
    scratch: &mut PersistentBfsGpuScratch,
    frontier_out: &mut Vec<u32>,
) -> Result<(u32, u32), SemanticExecutionError> {
    let max_iters = inputs.max_iters;
    let graph = inputs.graph;
    let plan = plan_persistent_bfs_dispatch(inputs, frontier_in)
        .map_err(SemanticExecutionError::InvalidRequest)?;
    let layout = plan.layout();
    let words = plan.frontier_words();
    if layout.node_count == 0 {
        frontier_out.clear();
        // An empty graph is a trivial fixpoint: there is nothing to expand.
        return Ok((0, 1));
    }
    if max_iters == 0 {
        copy_frontier_seed_into(
            frontier_out,
            frontier_in,
            "bfs_expand_via zero-iteration frontier_out",
        )?;
        // No confirming step ran, so the seed is not proven converged.
        return Ok((0, 0));
    }
    let program = plan.program("frontier_in", "frontier_out");
    let changed_words = program
        .buffers()
        .iter()
        .find(|buffer| buffer.name() == "changed")
        .map(|buffer| buffer.count().max(1) as usize)
        .unwrap_or(1);
    refresh_keyed_dispatch_inputs(
        &mut scratch.inputs,
        &mut scratch.static_input_key,
        plan.static_input_key(),
        &[
            DispatchInput::zero_u32_words(plan.node_words(), "bfs_expand_via graph nodes"),
            DispatchInput::u32_slice(graph.edge_offsets),
            DispatchInput::u32_slice_or_zero_words(
                graph.edge_targets,
                plan.edge_storage_words(),
                "bfs_expand_via edge_targets",
            ),
            DispatchInput::u32_slice_or_zero_words(
                graph.edge_kind_mask,
                plan.edge_storage_words(),
                "bfs_expand_via edge_kind_mask",
            ),
            DispatchInput::zero_u32_words(plan.node_words(), "bfs_expand_via node_tags"),
            DispatchInput::u32_slice(frontier_in),
            DispatchInput::zero_u32_words(words, "bfs_expand_via frontier_out"),
            DispatchInput::zero_u32_words(changed_words, "bfs_expand_via changed"),
            DispatchInput::zero_u32_words(1, "bfs_expand_via converged"),
        ],
        &[
            (5, DispatchInput::u32_slice(frontier_in)),
            (
                6,
                DispatchInput::zero_u32_words(words, "bfs_expand_via frontier_out"),
            ),
            (
                7,
                DispatchInput::zero_u32_words(changed_words, "bfs_expand_via changed"),
            ),
            (
                8,
                DispatchInput::zero_u32_words(1, "bfs_expand_via converged"),
            ),
        ],
    )?;
    let outputs = execute_single_program(
        dispatcher,
        "persistent_bfs_expand",
        program,
        &scratch.inputs,
        policy,
    )?
    .outputs;
    let [frontier_buf, changed_buf, converged_buf] = match outputs.as_slice() {
        [frontier_buf, changed_buf, converged_buf] => [frontier_buf, changed_buf, converged_buf],
        _ => {
            return Err(SemanticExecutionError::Backend(format!(
                "Fix: bfs_expand_via expected exactly three u32 output buffers (frontier_out, changed, converged), got {}.",
                outputs.len()
            )));
        }
    };
    decode_u32_output_exact(
        frontier_buf,
        words,
        "bfs_expand_via frontier_out",
        frontier_out,
    )?;
    decode_u32_output_exact(
        changed_buf,
        changed_words,
        "bfs_expand_via changed",
        &mut scratch.changed,
    )?;
    decode_u32_output_exact(
        converged_buf,
        1,
        "bfs_expand_via converged",
        &mut scratch.converged,
    )?;
    let changed = scratch.changed[0];
    validate_persistent_bfs_changed_flag(changed).map_err(SemanticExecutionError::Backend)?;
    let converged = scratch.converged[0];
    validate_persistent_bfs_converged_flag(converged).map_err(SemanticExecutionError::Backend)?;
    Ok((changed, converged))
}
