use super::resident_scratch::{
    copy_frontier_batch_seed_and_clear_changed, copy_frontier_seed_into,
    PersistentBfsResidentScratch, ResidentBfsGraph,
};

use crate::dispatch_buffers::{u32_word_bytes, write_u32_slice_le_bytes};
use crate::graph::dispatch::dispatch_bridge::{
    resident_dispatch_three_u32_outputs_into, upload_resident_dispatch_inputs, DispatchInput,
};
use crate::graph::persistent_bfs::{
    persistent_bfs_layout_hash as primitive_persistent_bfs_layout_hash,
    plan_persistent_bfs_resident_batch_dispatch, plan_persistent_bfs_resident_dispatch,
    validate_persistent_bfs_changed_flag, validate_persistent_bfs_converged_flag,
    validate_persistent_bfs_graph_layout,
};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher, ResidentReadRange};

/// Upload CSR graph topology once into resident device buffers.
///
/// Use the returned [`ResidentBfsGraph`] with
/// [`bfs_expand_resident_graph_with_scratch_into`] for repeated dataflow
/// queries that share the same graph topology.
///
/// # Errors
///
/// Rejects malformed CSR shapes or dispatchers without resident-buffer support.
pub fn upload_resident_bfs_graph(
    dispatcher: &dyn ProgramDispatcher,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
) -> Result<ResidentBfsGraph, DispatchError> {
    let layout = validate_persistent_bfs_graph_layout(
        node_count,
        edge_offsets,
        edge_targets,
        edge_kind_mask,
    )
    .map_err(DispatchError::BadInputs)?;

    let mut payload_storage = Vec::new();
    let unique_handles = upload_resident_dispatch_inputs(
        dispatcher,
        &mut payload_storage,
        [
            DispatchInput::zero_u32_words(layout.node_words, "resident BFS graph nodes"),
            DispatchInput::u32_slice(edge_offsets),
            DispatchInput::u32_slice_or_zero_words(
                edge_targets,
                layout.edge_storage_words,
                "resident BFS graph edge_targets",
            ),
            DispatchInput::u32_slice_or_zero_words(
                edge_kind_mask,
                layout.edge_storage_words,
                "resident BFS graph edge_kind_mask",
            ),
        ],
    )?;
    let handles = [
        unique_handles[0],
        unique_handles[1],
        unique_handles[2],
        unique_handles[3],
        unique_handles[0],
    ];

    Ok(ResidentBfsGraph {
        node_count: layout.node_count,
        edge_count: layout.edge_count,
        words: layout.words,
        words_u32: layout.words_u32,
        layout_hash: primitive_persistent_bfs_layout_hash(
            layout.node_count,
            edge_offsets,
            edge_targets,
            edge_kind_mask,
        ),
        handles,
    })
}

/// Run persistent BFS over an already-resident graph.
///
/// The graph buffers are not re-uploaded. The scratch object owns resident
/// frontier/change buffers and reuses them across calls when the frontier byte
/// width is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn bfs_expand_resident_graph_with_scratch_into(
    dispatcher: &dyn ProgramDispatcher,
    graph: &ResidentBfsGraph,
    frontier_in: &[u32],
    allow_mask: u32,
    max_iters: u32,
    scratch: &mut PersistentBfsResidentScratch,
    frontier_out: &mut Vec<u32>,
) -> Result<(u32, u32), DispatchError> {
    let plan = plan_persistent_bfs_resident_dispatch(
        graph.node_count,
        graph.edge_count,
        graph.words,
        frontier_in,
        allow_mask,
        max_iters,
    )
    .map_err(DispatchError::BadInputs)?;
    if graph.node_count == 0 || max_iters == 0 {
        copy_frontier_seed_into(
            frontier_out,
            frontier_in,
            "bfs_expand_resident_graph zero-iteration frontier_out",
        )?;
        // An empty graph is a trivial fixpoint; a zero-iteration run confirms nothing.
        let converged = u32::from(graph.node_count == 0);
        return Ok((0, converged));
    }
    let frontier_bytes =
        u32_word_bytes(plan.frontier_words(), "bfs_expand_resident_graph frontier")?;
    let frontier_handles = ensure_resident_query_handles(
        dispatcher,
        scratch,
        &[frontier_bytes, frontier_bytes, 4, 4],
    )?;
    write_u32_slice_le_bytes(&mut scratch.frontier_in_bytes, frontier_in);

    let uploads = [(frontier_handles[0], scratch.frontier_in_bytes.as_slice())];
    let key = plan.program_cache_key(dispatcher.device_feature_cache_key());
    let program = scratch
        .plan_cache
        .get_or_build(key, || plan.program("frontier_in", "frontier_out"));
    let graph_handles = graph.handles;
    let handles = [
        graph_handles[0],
        graph_handles[1],
        graph_handles[2],
        graph_handles[3],
        graph_handles[4],
        frontier_handles[0],
        frontier_handles[1],
        frontier_handles[2],
        frontier_handles[3],
    ];
    resident_dispatch_three_u32_outputs_into(
        dispatcher,
        &uploads,
        &program,
        &handles,
        Some(plan.dispatch_grid()),
        [
            ResidentReadRange {
                handle_id: frontier_handles[1],
                byte_offset: 0,
                byte_len: frontier_bytes,
            },
            ResidentReadRange {
                handle_id: frontier_handles[2],
                byte_offset: 0,
                byte_len: 4,
            },
            ResidentReadRange {
                handle_id: frontier_handles[3],
                byte_offset: 0,
                byte_len: 4,
            },
        ],
        &mut scratch.readbacks,
        [
            (
                plan.frontier_words(),
                "bfs_expand_resident_graph frontier_out",
                frontier_out,
            ),
            (1, "bfs_expand_resident_graph changed", &mut scratch.changed),
            (
                1,
                "bfs_expand_resident_graph converged",
                &mut scratch.converged,
            ),
        ],
    )?;
    let changed = scratch.changed[0];
    validate_persistent_bfs_changed_flag(changed).map_err(DispatchError::BackendError)?;
    let converged = scratch.converged[0];
    validate_persistent_bfs_converged_flag(converged).map_err(DispatchError::BackendError)?;
    Ok((changed, converged))
}

/// Run many persistent-BFS queries over one resident graph.
///
/// `frontier_inputs` is a flat array of `query_count * graph.words()` u32
/// words. Outputs are written flat in the same order. This keeps graph topology
/// resident and reuses the scratch-owned frontier/change handles across all
/// queries, so the only per-query H2D payload is the seed frontier plus zeroed
/// output/change state.
#[allow(clippy::too_many_arguments)]
pub fn bfs_expand_resident_graph_batch_with_scratch_into(
    dispatcher: &dyn ProgramDispatcher,
    graph: &ResidentBfsGraph,
    frontier_inputs: &[u32],
    query_count: usize,
    allow_mask: u32,
    max_iters: u32,
    scratch: &mut PersistentBfsResidentScratch,
    frontier_outputs: &mut Vec<u32>,
    changed_outputs: &mut Vec<u32>,
    converged_outputs: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let plan = plan_persistent_bfs_resident_batch_dispatch(
        graph.node_count,
        graph.edge_count,
        graph.words,
        frontier_inputs,
        query_count,
        allow_mask,
        max_iters,
    )
    .map_err(DispatchError::BadInputs)?;
    if plan.words_per_query() != graph.words_u32 {
        return Err(DispatchError::BadInputs(format!(
            "Fix: resident BFS graph word metadata diverged from primitive batch plan: graph={} plan={}.",
            graph.words_u32,
            plan.words_per_query()
        )));
    }
    if plan.query_count() == 0 {
        frontier_outputs.clear();
        changed_outputs.clear();
        converged_outputs.clear();
        return Ok(());
    }
    if graph.node_count == 0 || max_iters == 0 {
        copy_frontier_batch_seed_and_clear_changed(
            frontier_outputs,
            frontier_inputs,
            changed_outputs,
            plan.query_count(),
            "bfs_expand_resident_graph_batch zero-iteration",
        )?;
        // An empty graph is a trivial per-query fixpoint (converged = 1); a
        // zero-iteration run over a non-empty graph confirms nothing (converged
        // = 0). Mirrors the single-query zero path.
        let converged = u32::from(graph.node_count == 0);
        converged_outputs.clear();
        converged_outputs.resize(plan.query_count(), converged);
        return Ok(());
    }

    let frontier_bytes = u32_word_bytes(
        plan.total_words(),
        "bfs_expand_resident_graph_batch frontier",
    )?;
    let changed_bytes = u32_word_bytes(
        plan.query_count(),
        "bfs_expand_resident_graph_batch changed",
    )?;
    // converged is a per-query u32 array with the same width as changed.
    let converged_bytes = changed_bytes;
    let frontier_handles = ensure_resident_query_handles(
        dispatcher,
        scratch,
        &[
            frontier_bytes,
            frontier_bytes,
            changed_bytes,
            converged_bytes,
        ],
    )?;
    write_u32_slice_le_bytes(&mut scratch.frontier_in_bytes, frontier_inputs);

    let uploads = [(frontier_handles[0], scratch.frontier_in_bytes.as_slice())];
    let key = plan.program_cache_key(dispatcher.device_feature_cache_key());
    let program = scratch.plan_cache.get_or_build(key, || {
        plan.program("frontier_in", "frontier_out", "changed", "converged")
    });
    let graph_handles = graph.handles;
    let handles = [
        graph_handles[0],
        graph_handles[1],
        graph_handles[2],
        graph_handles[3],
        graph_handles[4],
        frontier_handles[0],
        frontier_handles[1],
        frontier_handles[2],
        frontier_handles[3],
    ];
    resident_dispatch_three_u32_outputs_into(
        dispatcher,
        &uploads,
        &program,
        &handles,
        Some(plan.dispatch_grid()),
        [
            ResidentReadRange {
                handle_id: frontier_handles[1],
                byte_offset: 0,
                byte_len: frontier_bytes,
            },
            ResidentReadRange {
                handle_id: frontier_handles[2],
                byte_offset: 0,
                byte_len: changed_bytes,
            },
            ResidentReadRange {
                handle_id: frontier_handles[3],
                byte_offset: 0,
                byte_len: converged_bytes,
            },
        ],
        &mut scratch.readbacks,
        [
            (
                plan.total_words(),
                "bfs_expand_resident_graph_batch frontier_out",
                frontier_outputs,
            ),
            (
                plan.query_count(),
                "bfs_expand_resident_graph_batch changed",
                changed_outputs,
            ),
            (
                plan.query_count(),
                "bfs_expand_resident_graph_batch converged",
                converged_outputs,
            ),
        ],
    )?;
    // Validate every per-query converged flag is a strict 0/1 device word before
    // the caller enforces its non-convergence policy.
    for &converged in converged_outputs.iter() {
        validate_persistent_bfs_converged_flag(converged).map_err(DispatchError::BackendError)?;
    }
    Ok(())
}

/// Acquire the scratch-owned resident query buffers for one dispatch, reusing
/// the cached handle group when the requested byte-length signature is
/// unchanged. The signature length varies by query kind: single queries use
/// four buffers (frontier_in, frontier_out, changed, converged), batch queries
/// use three (frontier_in, frontier_out, changed).
pub(super) fn ensure_resident_query_handles(
    dispatcher: &dyn ProgramDispatcher,
    scratch: &mut PersistentBfsResidentScratch,
    byte_lengths: &[usize],
) -> Result<Vec<u64>, DispatchError> {
    if let Some(handles) = &scratch.frontier_handles {
        if scratch.handle_byte_lengths == byte_lengths {
            return Ok(handles.clone());
        }
        scratch.free(dispatcher)?;
    }
    // `alloc_resident_many` rolls back every earlier handle if a later buffer
    // in the group fails to allocate, so a failed acquisition never publishes
    // partial handles onto the scratch.
    let handles = dispatcher.alloc_resident_many(byte_lengths)?;
    scratch.frontier_handles = Some(handles.clone());
    scratch.handle_byte_lengths = byte_lengths.to_vec();
    Ok(handles)
}
