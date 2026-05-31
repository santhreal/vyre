use super::{
    ResidentCsrQueueBatchQueryHandles, ResidentCsrQueueBatchScratch, ResidentCsrQueueBatchShape,
};
use vyre_primitives::bitset::zero::bitset_zero;
use vyre_primitives::graph::csr_frontier_queue::{
    csr_queue_forward_traverse, frontier_to_queue, frontier_word_block_offsets_in_place,
    frontier_word_block_offsets_to_queue_parallel, frontier_word_block_prefix_to_queue_parallel,
    frontier_word_counts_scan_pass_a, validate_frontier_queue_batch,
};
use vyre_primitives::graph::csr_queue_strided::csr_queue_strided_forward_traverse;

use crate::csr_frontier_queue_batch_memory::{
    plan_resident_csr_queue_batch_memory, ResidentCsrQueueBatchMemoryPlan,
};
use crate::csr_frontier_queue_resident::ResidentCsrQueueGraph;
use crate::dispatch_buffers::u32_word_bytes;
use crate::graph::csr_frontier_queue_scratch::{
    frontier_word_dispatch_grid, frontier_word_prefix_scratch,
    frontier_word_prefix_uses_precomputed_offsets, resident_csr_queue_effective_capacity,
    resident_csr_queue_materializer, resident_csr_queue_scratch_bytes_per_query,
    resident_csr_queue_traverse_grid, resident_csr_queue_traverse_kind, FrontierWordPrefixScratch,
    ResidentCsrQueueMaterializer, ResidentCsrQueueTraverseKind,
};
use crate::graph::dispatch_bridge::alloc_resident_buffers;
use crate::hardware::scratch::reserve_vec as reserve_graph_vec;
use crate::optimizer::dispatcher::{
    DispatchError, OptimizerDispatcher, ResidentDispatchStep, ResidentReadRange,
};

/// Run many sparse frontier queries over one resident CSR graph.
pub fn run_resident_csr_queue_batch_into(
    dispatcher: &dyn OptimizerDispatcher,
    graph: &ResidentCsrQueueGraph,
    scratch: &mut ResidentCsrQueueBatchScratch,
    frontiers: &[&[u32]],
    queue_capacity: u32,
    allow_mask: u32,
    outputs: &mut Vec<Vec<u8>>,
) -> Result<(), DispatchError> {
    validate_frontier_queue_batch(graph.node_count(), frontiers, queue_capacity)
        .map_err(DispatchError::BadInputs)?;
    let effective_queue_capacity =
        resident_csr_queue_effective_capacity(graph.node_count(), frontiers, queue_capacity)
            .map_err(DispatchError::BadInputs)?;
    ensure_batch_scratch(
        dispatcher,
        graph,
        scratch,
        frontiers.len(),
        effective_queue_capacity,
        allow_mask,
    )?;

    let frontier_bytes = u32_word_bytes(graph.words(), "resident CSR queue batch frontier")?;
    if scratch.frontier_payloads.len() < frontiers.len() {
        scratch
            .frontier_payloads
            .resize_with(frontiers.len(), Vec::new);
    }
    scratch.frontier_payloads.truncate(frontiers.len());
    for (payload, frontier) in scratch.frontier_payloads.iter_mut().zip(frontiers) {
        payload.clear();
        vyre_primitives::wire::append_u32_slice_le_bytes(frontier, payload);
    }
    prepare_batch_sequence_tables(graph, scratch, frontiers.len(), frontier_bytes)?;

    let mut upload_refs = Vec::new();
    reserve_graph_vec(
        &mut upload_refs,
        frontiers.len(),
        "resident CSR queue batch uploads",
    )?;
    for query_index in 0..frontiers.len() {
        let handles = scratch.handles[query_index];
        upload_refs.push((
            handles.frontier,
            scratch.frontier_payloads[query_index].as_slice(),
        ));
    }

    let clear_frontier_out_program = scratch.clear_frontier_out_program.as_ref().ok_or_else(|| {
        DispatchError::BackendError(
            "batch CSR queue output clear program is missing after ensure_batch_scratch. Fix: rebuild batch scratch before resident CSR queue dispatch.".to_string(),
        )
    })?;
    let queue_program = scratch.queue_program.as_ref().ok_or_else(|| {
        DispatchError::BackendError(
            "batch CSR queue program is missing after ensure_batch_scratch. Fix: rebuild batch scratch before resident CSR queue dispatch.".to_string(),
        )
    })?;
    let traverse_program = scratch.traverse_program.as_ref().ok_or_else(|| {
        DispatchError::BackendError(
            "batch CSR traverse program is missing after ensure_batch_scratch. Fix: rebuild batch scratch before resident CSR traverse dispatch.".to_string(),
        )
    })?;

    let mut steps = Vec::new();
    let traverse_kind = resident_csr_queue_traverse_kind(graph.max_row_degree());
    let traverse_grid = resident_csr_queue_traverse_grid(effective_queue_capacity, traverse_kind);
    reserve_graph_vec(
        &mut steps,
        frontiers
            .len()
            .checked_mul(5)
            .ok_or_else(|| DispatchError::BackendError(
                "Fix: resident CSR queue batch step count overflowed while reserving dispatch sequence slots."
                    .to_string(),
            ))?,
        "resident CSR queue batch steps",
    )?;
    match resident_csr_queue_materializer(graph.words()) {
        ResidentCsrQueueMaterializer::AtomicNodeScan => {
            for query_index in 0..frontiers.len() {
                steps.push(ResidentDispatchStep {
                    program: clear_frontier_out_program,
                    handle_ids: &scratch.clear_handle_sets[query_index],
                    grid_override: Some(frontier_word_grid(graph.words())?),
                });
                steps.push(ResidentDispatchStep {
                    program: queue_program,
                    handle_ids: &scratch.queue_handle_sets[query_index],
                    grid_override: Some([1, 1, 1]),
                });
                steps.push(ResidentDispatchStep {
                    program: traverse_program,
                    handle_ids: &scratch.traverse_handle_sets[query_index],
                    grid_override: Some(traverse_grid),
                });
            }
        }
        ResidentCsrQueueMaterializer::DeterministicWordPrefix => {
            let word_prefix = word_prefix_scratch(graph.words())?;
            let precompute_block_offsets =
                frontier_word_prefix_uses_precomputed_offsets(word_prefix.block_count);
            let word_counts_program = scratch.word_counts_program.as_ref().ok_or_else(|| {
                DispatchError::BackendError(
                    "batch CSR queue word-count scan program is missing after ensure_batch_scratch. Fix: rebuild batch scratch before resident CSR queue dispatch.".to_string(),
                )
            })?;
            let block_offsets_program = scratch.word_block_offsets_program.as_ref();
            for query_index in 0..frontiers.len() {
                steps.push(ResidentDispatchStep {
                    program: clear_frontier_out_program,
                    handle_ids: &scratch.clear_handle_sets[query_index],
                    grid_override: Some(frontier_word_grid(graph.words())?),
                });
                steps.push(ResidentDispatchStep {
                    program: word_counts_program,
                    handle_ids: &scratch.word_count_handle_sets[query_index],
                    grid_override: Some([word_prefix.block_count, 1, 1]),
                });
                if precompute_block_offsets {
                    let block_offsets_program = block_offsets_program.ok_or_else(|| {
                        DispatchError::BackendError(
                            "batch CSR queue block-offset scan program is missing after ensure_batch_scratch. Fix: rebuild batch scratch before resident CSR queue dispatch.".to_string(),
                        )
                    })?;
                    steps.push(ResidentDispatchStep {
                        program: block_offsets_program,
                        handle_ids: &scratch.word_block_offsets_handle_sets[query_index],
                        grid_override: Some([1, 1, 1]),
                    });
                }
                steps.push(ResidentDispatchStep {
                    program: queue_program,
                    handle_ids: &scratch.word_prefix_queue_handle_sets[query_index],
                    grid_override: Some(frontier_word_grid(graph.words())?),
                });
                steps.push(ResidentDispatchStep {
                    program: traverse_program,
                    handle_ids: &scratch.traverse_handle_sets[query_index],
                    grid_override: Some(traverse_grid),
                });
            }
        }
    }

    dispatcher.upload_resident_many_sequence_read_ranges_into(
        &upload_refs,
        &steps,
        &scratch.read_ranges,
        &mut scratch.readbacks,
    )?;

    if outputs.len() < frontiers.len() {
        outputs.resize_with(frontiers.len(), Vec::new);
    }
    outputs.truncate(frontiers.len());
    for (output, readback) in outputs.iter_mut().zip(&scratch.readbacks) {
        output.clear();
        output.extend_from_slice(readback);
    }
    Ok(())
}

/// Run many sparse frontier queries, sharded by resident scratch budget.
pub fn run_resident_csr_queue_batch_budgeted_into(
    dispatcher: &dyn OptimizerDispatcher,
    graph: &ResidentCsrQueueGraph,
    scratch: &mut ResidentCsrQueueBatchScratch,
    frontiers: &[&[u32]],
    queue_capacity: u32,
    allow_mask: u32,
    max_scratch_bytes: usize,
    outputs: &mut Vec<Vec<u8>>,
) -> Result<ResidentCsrQueueBatchMemoryPlan, DispatchError> {
    validate_frontier_queue_batch(graph.node_count(), frontiers, queue_capacity)
        .map_err(DispatchError::BadInputs)?;
    let (plan, chunks) = plan_budgeted_effective_chunks(
        graph.node_count(),
        graph.words(),
        frontiers,
        queue_capacity,
        max_scratch_bytes,
    )?;
    if outputs.len() < frontiers.len() {
        outputs.resize_with(frontiers.len(), Vec::new);
    }
    outputs.truncate(frontiers.len());

    let mut chunk_outputs = Vec::new();
    for chunk in chunks {
        let frontier_chunk = &frontiers[chunk.start..chunk.end];
        run_resident_csr_queue_batch_into(
            dispatcher,
            graph,
            scratch,
            frontier_chunk,
            chunk.queue_capacity,
            allow_mask,
            &mut chunk_outputs,
        )?;
        let offset = chunk.start;
        for (target, source) in outputs[offset..offset + frontier_chunk.len()]
            .iter_mut()
            .zip(&chunk_outputs)
        {
            target.clear();
            target.extend_from_slice(source);
        }
    }

    Ok(plan)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ResidentCsrQueueBudgetedChunk {
    start: usize,
    end: usize,
    queue_capacity: u32,
}

fn plan_budgeted_effective_chunks(
    node_count: u32,
    frontier_words: usize,
    frontiers: &[&[u32]],
    requested_queue_capacity: u32,
    max_scratch_bytes: usize,
) -> Result<
    (
        ResidentCsrQueueBatchMemoryPlan,
        Vec<ResidentCsrQueueBudgetedChunk>,
    ),
    DispatchError,
> {
    let mut query_capacities = Vec::new();
    reserve_graph_vec(
        &mut query_capacities,
        frontiers.len(),
        "resident CSR queue budgeted query capacities",
    )?;
    for frontier in frontiers {
        query_capacities.push(
            resident_csr_queue_effective_capacity(
                node_count,
                &[*frontier],
                requested_queue_capacity,
            )
            .map_err(DispatchError::BadInputs)?,
        );
    }

    let mut chunks = Vec::new();
    reserve_graph_vec(
        &mut chunks,
        frontiers.len(),
        "resident CSR queue budgeted dispatch chunks",
    )?;
    let mut start = 0usize;
    let mut max_queries_per_dispatch = 0usize;
    let mut peak_bytes_per_query = 0usize;
    let mut peak_batch_scratch_bytes = 0usize;

    while start < frontiers.len() {
        let mut end = start + 1;
        let mut chunk_capacity = query_capacities[start];
        let mut bytes_per_query =
            budgeted_chunk_bytes_per_query(frontier_words, chunk_capacity, max_scratch_bytes)?;

        while end < frontiers.len() {
            let candidate_capacity = chunk_capacity.max(query_capacities[end]);
            let candidate_bytes =
                resident_csr_queue_scratch_bytes_per_query(frontier_words, candidate_capacity)
                    .map_err(DispatchError::BadInputs)?;
            let candidate_queries = end - start + 1;
            let candidate_peak = checked_batch_scratch_bytes(candidate_bytes, candidate_queries)?;
            if candidate_peak > max_scratch_bytes {
                break;
            }
            chunk_capacity = candidate_capacity;
            bytes_per_query = candidate_bytes;
            end += 1;
        }

        let query_count = end - start;
        let chunk_peak = checked_batch_scratch_bytes(bytes_per_query, query_count)?;
        max_queries_per_dispatch = max_queries_per_dispatch.max(query_count);
        peak_bytes_per_query = peak_bytes_per_query.max(bytes_per_query);
        peak_batch_scratch_bytes = peak_batch_scratch_bytes.max(chunk_peak);
        chunks.push(ResidentCsrQueueBudgetedChunk {
            start,
            end,
            queue_capacity: chunk_capacity,
        });
        start = end;
    }

    Ok((
        ResidentCsrQueueBatchMemoryPlan {
            query_count: frontiers.len(),
            max_queries_per_dispatch,
            dispatch_batches: chunks.len(),
            bytes_per_query: peak_bytes_per_query,
            peak_batch_scratch_bytes,
        },
        chunks,
    ))
}

fn budgeted_chunk_bytes_per_query(
    frontier_words: usize,
    queue_capacity: u32,
    max_scratch_bytes: usize,
) -> Result<usize, DispatchError> {
    plan_resident_csr_queue_batch_memory(1, frontier_words, queue_capacity, max_scratch_bytes)
        .map(|plan| plan.bytes_per_query)
        .map_err(|error| DispatchError::BadInputs(error.to_string()))
}

fn checked_batch_scratch_bytes(
    bytes_per_query: usize,
    query_count: usize,
) -> Result<usize, DispatchError> {
    bytes_per_query.checked_mul(query_count).ok_or_else(|| {
        DispatchError::BadInputs(
            "resident CSR queue budgeted batch scratch byte calculation overflowed. Fix: shard the query batch before planning."
                .to_string(),
        )
    })
}

fn prepare_batch_sequence_tables(
    graph: &ResidentCsrQueueGraph,
    scratch: &mut ResidentCsrQueueBatchScratch,
    batch_len: usize,
    frontier_bytes: usize,
) -> Result<(), DispatchError> {
    scratch.clear_handle_sets.clear();
    scratch.word_count_handle_sets.clear();
    scratch.word_block_offsets_handle_sets.clear();
    scratch.queue_handle_sets.clear();
    scratch.word_prefix_queue_handle_sets.clear();
    scratch.traverse_handle_sets.clear();
    scratch.read_ranges.clear();

    reserve_graph_vec(
        &mut scratch.clear_handle_sets,
        batch_len,
        "resident CSR queue batch output clear handles",
    )?;
    reserve_graph_vec(
        &mut scratch.word_count_handle_sets,
        batch_len,
        "resident CSR queue batch word-count handles",
    )?;
    reserve_graph_vec(
        &mut scratch.word_block_offsets_handle_sets,
        batch_len,
        "resident CSR queue batch block-offset handles",
    )?;
    reserve_graph_vec(
        &mut scratch.queue_handle_sets,
        batch_len,
        "resident CSR queue batch queue handles",
    )?;
    reserve_graph_vec(
        &mut scratch.word_prefix_queue_handle_sets,
        batch_len,
        "resident CSR queue batch word-prefix queue handles",
    )?;
    reserve_graph_vec(
        &mut scratch.traverse_handle_sets,
        batch_len,
        "resident CSR queue batch traverse handles",
    )?;
    reserve_graph_vec(
        &mut scratch.read_ranges,
        batch_len,
        "resident CSR queue batch read ranges",
    )?;

    let materializer = resident_csr_queue_materializer(graph.words());
    let precompute_block_offsets =
        if materializer == ResidentCsrQueueMaterializer::DeterministicWordPrefix {
            let word_prefix = word_prefix_scratch(graph.words())?;
            frontier_word_prefix_uses_precomputed_offsets(word_prefix.block_count)
        } else {
            false
        };
    for (query_index, handles) in scratch.handles.iter().take(batch_len).enumerate() {
        scratch.clear_handle_sets.push([handles.frontier_out]);
        scratch
            .queue_handle_sets
            .push([handles.frontier, handles.active_queue, handles.queue_len]);
        match (handles.word_partials, handles.block_totals) {
            (Some(word_partials), Some(block_totals)) => {
                scratch.word_count_handle_sets.push([
                    handles.frontier,
                    word_partials,
                    block_totals,
                ]);
                if precompute_block_offsets {
                    scratch.word_block_offsets_handle_sets.push([block_totals]);
                }
                scratch.word_prefix_queue_handle_sets.push([
                    handles.frontier,
                    word_partials,
                    block_totals,
                    handles.active_queue,
                    handles.queue_len,
                ]);
            }
            (None, None) if materializer == ResidentCsrQueueMaterializer::AtomicNodeScan => {}
            (None, None) => {
                return Err(DispatchError::BackendError(format!(
                    "Fix: resident CSR queue batch query {query_index} is missing word-prefix scratch handles."
                )));
            }
            _ => {
                return Err(DispatchError::BackendError(format!(
                    "Fix: resident CSR queue batch query {query_index} has incomplete word-prefix scratch handles."
                )));
            }
        }
        scratch.traverse_handle_sets.push([
            handles.active_queue,
            handles.queue_len,
            graph.edge_offsets_handle(),
            graph.edge_targets_handle(),
            graph.edge_kind_mask_handle(),
            handles.frontier_out,
        ]);
        scratch.read_ranges.push(ResidentReadRange {
            handle_id: handles.frontier_out,
            byte_offset: 0,
            byte_len: frontier_bytes,
        });
    }

    Ok(())
}

fn ensure_batch_scratch(
    dispatcher: &dyn OptimizerDispatcher,
    graph: &ResidentCsrQueueGraph,
    scratch: &mut ResidentCsrQueueBatchScratch,
    batch_len: usize,
    queue_capacity: u32,
    allow_mask: u32,
) -> Result<(), DispatchError> {
    let frontier_bytes =
        u32_word_bytes(graph.words(), "resident CSR queue batch scratch frontier")?;
    let queue_bytes = u32_word_bytes(
        queue_capacity as usize,
        "resident CSR queue batch scratch active_queue",
    )?;
    let queue_len_bytes = u32_word_bytes(1, "resident CSR queue batch scratch queue_len")?;
    let materializer = resident_csr_queue_materializer(graph.words());
    let traverse_kind = resident_csr_queue_traverse_kind(graph.max_row_degree());
    let shape = ResidentCsrQueueBatchShape {
        batch_len,
        frontier_bytes,
        queue_capacity,
        allow_mask,
        node_count: graph.node_count(),
        edge_count: graph.edge_count(),
        materializer,
        traverse_kind,
    };
    if matches!(
        scratch.shape,
        Some(existing)
            if existing.batch_len >= batch_len
                && existing.frontier_bytes == frontier_bytes
                && existing.queue_capacity == queue_capacity
                && existing.allow_mask == allow_mask
                && existing.node_count == graph.node_count()
                && existing.edge_count == graph.edge_count()
                && existing.materializer == materializer
                && existing.traverse_kind == traverse_kind
    ) {
        return Ok(());
    }
    if scratch.shape == Some(shape) {
        return Ok(());
    }

    scratch.free(dispatcher)?;
    reserve_graph_vec(
        &mut scratch.handles,
        batch_len,
        "resident CSR queue batch scratch handles",
    )?;
    for _ in 0..batch_len {
        match materializer {
            ResidentCsrQueueMaterializer::AtomicNodeScan => {
                let [frontier, active_queue, queue_len, frontier_out] = match alloc_resident_buffers(
                    dispatcher,
                    [frontier_bytes, queue_bytes, queue_len_bytes, frontier_bytes],
                    "resident CSR queue batch scratch query",
                ) {
                    Ok(handles) => handles,
                    Err(error) => {
                        if let Err(free_error) = scratch.free(dispatcher) {
                            return Err(batch_scratch_allocation_cleanup_error(error, free_error));
                        }
                        return Err(error);
                    }
                };
                scratch.handles.push(ResidentCsrQueueBatchQueryHandles {
                    frontier,
                    active_queue,
                    queue_len,
                    frontier_out,
                    word_partials: None,
                    block_totals: None,
                });
            }
            ResidentCsrQueueMaterializer::DeterministicWordPrefix => {
                let word_prefix = word_prefix_scratch(graph.words())?;
                let word_partials_bytes = u32_word_bytes(
                    word_prefix.partial_words,
                    "resident CSR queue batch scratch word_partials",
                )?;
                let block_totals_bytes = u32_word_bytes(
                    word_prefix.block_total_words,
                    "resident CSR queue batch scratch block_totals",
                )?;
                let [frontier, active_queue, queue_len, frontier_out, word_partials, block_totals] =
                    match alloc_resident_buffers(
                        dispatcher,
                        [
                            frontier_bytes,
                            queue_bytes,
                            queue_len_bytes,
                            frontier_bytes,
                            word_partials_bytes,
                            block_totals_bytes,
                        ],
                        "resident CSR queue batch scratch query",
                    ) {
                        Ok(handles) => handles,
                        Err(error) => {
                            if let Err(free_error) = scratch.free(dispatcher) {
                                return Err(batch_scratch_allocation_cleanup_error(
                                    error, free_error,
                                ));
                            }
                            return Err(error);
                        }
                    };
                scratch.handles.push(ResidentCsrQueueBatchQueryHandles {
                    frontier,
                    active_queue,
                    queue_len,
                    frontier_out,
                    word_partials: Some(word_partials),
                    block_totals: Some(block_totals),
                });
            }
        }
    }
    scratch.word_counts_program = None;
    scratch.word_block_offsets_program = None;
    scratch.clear_frontier_out_program = Some(bitset_zero("frontier_out", graph.words() as u32));
    match materializer {
        ResidentCsrQueueMaterializer::AtomicNodeScan => {
            scratch.queue_program = Some(frontier_to_queue(
                "frontier",
                "active_queue",
                "queue_len",
                graph.node_count(),
                queue_capacity,
            ));
        }
        ResidentCsrQueueMaterializer::DeterministicWordPrefix => {
            let word_prefix = word_prefix_scratch(graph.words())?;
            scratch.word_counts_program = Some(frontier_word_counts_scan_pass_a(
                "frontier",
                "word_partials",
                "block_totals",
                graph.node_count(),
            ));
            if frontier_word_prefix_uses_precomputed_offsets(word_prefix.block_count) {
                scratch.word_block_offsets_program = Some(frontier_word_block_offsets_in_place(
                    "block_totals",
                    graph.node_count(),
                ));
                scratch.queue_program = Some(frontier_word_block_offsets_to_queue_parallel(
                    "frontier",
                    "word_partials",
                    "block_totals",
                    "active_queue",
                    "queue_len",
                    graph.node_count(),
                    queue_capacity,
                ));
            } else {
                scratch.queue_program = Some(frontier_word_block_prefix_to_queue_parallel(
                    "frontier",
                    "word_partials",
                    "block_totals",
                    "active_queue",
                    "queue_len",
                    graph.node_count(),
                    queue_capacity,
                ));
            }
        }
    }
    scratch.traverse_program = Some(match traverse_kind {
        ResidentCsrQueueTraverseKind::RowSerial => csr_queue_forward_traverse(
            "active_queue",
            "queue_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            graph.node_count(),
            graph.edge_count(),
            queue_capacity,
            allow_mask,
        ),
        ResidentCsrQueueTraverseKind::RowStrided => csr_queue_strided_forward_traverse(
            "active_queue",
            "queue_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            graph.node_count(),
            graph.edge_count(),
            queue_capacity,
            allow_mask,
        ),
    });
    scratch.shape = Some(shape);
    Ok(())
}

fn word_prefix_scratch(words: usize) -> Result<FrontierWordPrefixScratch, DispatchError> {
    frontier_word_prefix_scratch(words).map_err(DispatchError::BackendError)
}

fn frontier_word_grid(words: usize) -> Result<[u32; 3], DispatchError> {
    frontier_word_dispatch_grid(words).map_err(DispatchError::BackendError)
}

fn batch_scratch_allocation_cleanup_error(
    allocation: DispatchError,
    cleanup: DispatchError,
) -> DispatchError {
    DispatchError::BackendError(format!(
        "Fix: resident CSR queue batch scratch allocation failed and cleanup also failed: allocation={allocation}; cleanup={cleanup}."
    ))
}
