//! Host-side dispatch of motif matching, with the reductions that turn per-node
//! match flags into a count and a witness.

use super::{CachedMotifProgram, MotifGpuScratch};
use crate::graph::motif::{
    plan_motif_launch, validate_motif_witness, MotifEdge, MotifStaticInputKey,
};
use crate::reduce::reduction_metrics::{reduce_any_via, reduce_count_non_zero_via};

use crate::graph::dispatch::dispatch_bridge::{
    dispatch_two_u32_outputs_from_prepared_into, refresh_keyed_dispatch_inputs, DispatchInput,
};
use vyre_megakernel::{SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor};

/// Dispatcher-backed motif match.
///
/// # Errors
///
/// Propagates dispatch failures and rejects malformed CSR or
/// truncated readback.
pub fn match_motif_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut out = Vec::new();
    match_motif_via_into(
        dispatcher,
        policy,
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
        &mut out,
    )?;
    Ok(out)
}

/// Dispatcher-backed motif match into caller-owned storage.
pub fn match_motif_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
    witness_out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = MotifGpuScratch::default();
    match_motif_via_with_scratch_into(
        dispatcher,
        policy,
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
        &mut scratch,
        witness_out,
    )
}

/// Dispatcher-backed motif match into caller-owned dispatch and output storage.
pub fn match_motif_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
    scratch: &mut MotifGpuScratch,
    witness_out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let plan = plan_motif_launch(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
        "witness_out",
    )
    .map_err(SemanticExecutionError::InvalidRequest)?;
    if plan.output_words() == 0 {
        witness_out.clear();
        return Ok(());
    }
    let MotifGpuScratch {
        inputs,
        motif_hits,
        static_input_key,
        program_cache,
    } = scratch;
    let cached =
        program_cache.get_or_insert_with(plan.cache_key().clone(), || CachedMotifProgram {
            layout: plan.layout(),
            program: plan.program(),
        });
    let next_static_input_key = plan
        .static_input_key(edge_offsets, edge_targets, edge_kind_mask)
        .map_err(SemanticExecutionError::InvalidRequest)?;
    refresh_motif_inputs(
        inputs,
        static_input_key,
        next_static_input_key,
        cached.layout,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
    )?;
    dispatch_two_u32_outputs_from_prepared_into(
        dispatcher,
        policy,
        cached.program.clone(),
        inputs,
        cached.layout.output_words,
        "motif_hits",
        motif_hits,
        cached.layout.output_words,
        "witness_out",
        witness_out,
    )?;
    validate_motif_witness(cached.layout, witness_out).map_err(SemanticExecutionError::Backend)
}

fn refresh_motif_inputs(
    inputs: &mut Vec<Vec<u8>>,
    static_input_key: &mut Option<MotifStaticInputKey>,
    next_static_input_key: MotifStaticInputKey,
    layout: crate::graph::motif::MotifLayout,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
) -> Result<(), SemanticExecutionError> {
    refresh_keyed_dispatch_inputs(
        inputs,
        static_input_key,
        next_static_input_key,
        &[
            DispatchInput::zero_u32_words(layout.output_words, "match_motif_via nodes"),
            DispatchInput::u32_slice(edge_offsets),
            DispatchInput::u32_slice_or_zero_words(
                edge_targets,
                layout.edge_storage_words,
                "match_motif_via edge_targets",
            ),
            DispatchInput::u32_slice_or_zero_words(
                edge_kind_mask,
                layout.edge_storage_words,
                "match_motif_via edge_kind_mask",
            ),
            DispatchInput::zero_u32_words(layout.output_words, "match_motif_via node_tags"),
        ],
        &[
            (
                0,
                DispatchInput::zero_u32_words(layout.output_words, "match_motif_via nodes"),
            ),
            (
                4,
                DispatchInput::zero_u32_words(layout.output_words, "match_motif_via node_tags"),
            ),
        ],
    )?;
    Ok(())
}

/// Dispatcher-backed motif existence predicate.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when graph validation or backend execution fails.
pub fn motif_matches_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Result<bool, SemanticExecutionError> {
    let witness = match_motif_via(
        dispatcher,
        policy,
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )?;
    reduce_any_via(dispatcher, policy, &witness)
}

/// Dispatcher-backed motif participation count.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when graph validation or backend execution fails.
pub fn motif_participation_count_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    motif_edges: &[MotifEdge],
) -> Result<u32, SemanticExecutionError> {
    let witness = match_motif_via(
        dispatcher,
        policy,
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
        motif_edges,
    )?;
    reduce_count_non_zero_via(dispatcher, policy, &witness)
}
