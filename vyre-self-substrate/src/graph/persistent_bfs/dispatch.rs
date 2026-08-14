use super::state::{copy_frontier_seed_into, PersistentBfsGpuScratch};

use vyre_libs::dispatch_buffers::decode_u32_output_exact;
use crate::graph::dispatch_bridge::{refresh_keyed_dispatch_inputs, DispatchInput};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};
use vyre_primitives::graph::persistent_bfs::{
    plan_persistent_bfs_dispatch, validate_persistent_bfs_changed_flag,
    validate_persistent_bfs_converged_flag,
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
#[allow(clippy::too_many_arguments)]
pub fn bfs_expand_via(
    dispatcher: &dyn ProgramDispatcher,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
) -> Result<(Vec<u32>, u32, u32), DispatchError> {
    let mut frontier = Vec::new();
    let (changed, converged) = bfs_expand_via_into(
        dispatcher,
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
        max_iters,
        &mut frontier,
    )?;
    Ok((frontier, changed, converged))
}

/// Dispatcher-backed persistent BFS expansion into caller-owned frontier storage.
#[allow(clippy::too_many_arguments)]
pub fn bfs_expand_via_into(
    dispatcher: &dyn ProgramDispatcher,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
    frontier_out: &mut Vec<u32>,
) -> Result<(u32, u32), DispatchError> {
    let mut scratch = PersistentBfsGpuScratch::default();
    bfs_expand_via_with_scratch_into(
        dispatcher,
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
        max_iters,
        &mut scratch,
        frontier_out,
    )
}

/// Dispatcher-backed persistent BFS expansion into caller-owned frontier and dispatch scratch.
#[allow(clippy::too_many_arguments)]
pub fn bfs_expand_via_with_scratch_into(
    dispatcher: &dyn ProgramDispatcher,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
    scratch: &mut PersistentBfsGpuScratch,
    frontier_out: &mut Vec<u32>,
) -> Result<(u32, u32), DispatchError> {
    let plan = plan_persistent_bfs_dispatch(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        frontier_in,
        allow_mask,
        max_iters,
    )
    .map_err(DispatchError::BadInputs)?;
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
    let key = plan.program_cache_key(dispatcher.device_feature_cache_key());
    let program = scratch
        .plan_cache
        .get_or_build(key, || plan.program("frontier_in", "frontier_out"));
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
            DispatchInput::u32_slice(edge_offsets),
            DispatchInput::u32_slice_or_zero_words(
                edge_targets,
                plan.edge_storage_words(),
                "bfs_expand_via edge_targets",
            ),
            DispatchInput::u32_slice_or_zero_words(
                edge_kind_mask,
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
    let outputs = dispatcher.dispatch(&program, &scratch.inputs, Some(plan.dispatch_grid()))?;
    if outputs.len() != 3 {
        return Err(DispatchError::BackendError(format!(
            "Fix: bfs_expand_via expected exactly three u32 output buffers (frontier_out, changed, converged), got {}.",
            outputs.len()
        )));
    }
    decode_u32_output_exact(
        &outputs[0],
        words,
        "bfs_expand_via frontier_out",
        frontier_out,
    )?;
    decode_u32_output_exact(
        &outputs[1],
        changed_words,
        "bfs_expand_via changed",
        &mut scratch.changed,
    )?;
    decode_u32_output_exact(
        &outputs[2],
        1,
        "bfs_expand_via converged",
        &mut scratch.converged,
    )?;
    let changed = scratch.changed[0];
    validate_persistent_bfs_changed_flag(changed).map_err(DispatchError::BackendError)?;
    let converged = scratch.converged[0];
    validate_persistent_bfs_converged_flag(converged).map_err(DispatchError::BackendError)?;
    Ok((changed, converged))
}
