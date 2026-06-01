use super::{
    ResidentCsrQueueGraph, ResidentCsrQueueProgramShape, ResidentCsrQueueScratch,
    ResidentCsrQueueScratchHandles,
};
use vyre_primitives::bitset::zero::bitset_zero;
use vyre_primitives::graph::csr_frontier_queue::{
    csr_queue_forward_traverse, frontier_queue_len_init, frontier_word_block_offsets_in_place,
    frontier_word_block_offsets_to_queue_parallel, frontier_word_block_prefix_to_queue_parallel,
    frontier_word_counts_scan_pass_a, frontier_words_to_queue_clear_out_parallel,
    validate_frontier_queue_query,
};
use vyre_primitives::graph::csr_queue_strided::csr_queue_strided_forward_traverse;

use crate::dispatch_buffers::u32_word_bytes;
use crate::graph::csr_frontier_queue_scratch::{
    frontier_word_dispatch_grid, frontier_word_prefix_scratch,
    frontier_word_prefix_uses_precomputed_offsets, resident_csr_queue_frontier_stats,
    resident_csr_queue_materializer_for_stats, resident_csr_queue_traverse_grid,
    resident_csr_queue_traverse_kind, FrontierWordPrefixScratch, ResidentCsrQueueMaterializer,
    ResidentCsrQueueTraverseKind,
};
use crate::graph::dispatch_bridge::alloc_resident_buffers;
use crate::optimizer::dispatcher::{
    DispatchError, OptimizerDispatcher, ResidentDispatchStep, ResidentReadRange,
};

/// Run one sparse frontier query over a resident CSR graph.
pub fn run_resident_csr_queue_query_into(
    dispatcher: &dyn OptimizerDispatcher,
    graph: &ResidentCsrQueueGraph,
    scratch: &mut ResidentCsrQueueScratch,
    frontier_words: &[u32],
    queue_capacity: u32,
    allow_mask: u32,
    output: &mut Vec<u8>,
) -> Result<(), DispatchError> {
    validate_frontier_queue_query(graph.node_count, frontier_words, queue_capacity)
        .map_err(DispatchError::BadInputs)?;
    let frontier_stats =
        resident_csr_queue_frontier_stats(graph.node_count, &[frontier_words], queue_capacity)
            .map_err(DispatchError::BadInputs)?;
    let effective_queue_capacity = frontier_stats.effective_queue_capacity;
    let materializer = resident_csr_queue_materializer_for_stats(
        graph.words,
        effective_queue_capacity,
        frontier_stats.max_nonzero_words,
    );
    ensure_scratch(
        dispatcher,
        scratch,
        graph.words,
        effective_queue_capacity,
        materializer,
    )?;
    let handles = scratch.handles.ok_or_else(|| {
        DispatchError::BackendError(
            "resident CSR queue scratch handles are missing after ensure_scratch. Fix: rebuild scratch before resident CSR queue dispatch.".to_string(),
        )
    })?;
    ensure_programs(
        scratch,
        graph,
        effective_queue_capacity,
        allow_mask,
        materializer,
    )?;
    scratch.frontier_bytes.clear();
    vyre_primitives::wire::append_u32_slice_le_bytes(frontier_words, &mut scratch.frontier_bytes);
    let frontier_bytes = u32_word_bytes(graph.words, "resident CSR queue query frontier")?;
    let traverse_handles = [
        handles.active_queue,
        handles.queue_len,
        graph.edge_offsets_handle,
        graph.edge_targets_handle,
        graph.edge_kind_mask_handle,
        handles.frontier_out,
    ];
    let traverse_program = scratch.traverse_program.as_ref().ok_or_else(|| {
        DispatchError::BackendError(
            "resident CSR queue traverse program is missing after ensure_programs. Fix: rebuild programs before resident CSR traverse dispatch.".to_string(),
        )
    })?;
    let traverse_grid = resident_csr_queue_traverse_grid(
        effective_queue_capacity,
        resident_csr_queue_traverse_kind(graph.max_row_degree),
    );
    let read_ranges = [ResidentReadRange {
        handle_id: handles.frontier_out,
        byte_offset: 0,
        byte_len: frontier_bytes,
    }];
    match handles.materializer {
        ResidentCsrQueueMaterializer::AtomicWordScan => {
            let queue_len_handles = [handles.queue_len];
            let queue_handles = [
                handles.frontier,
                handles.active_queue,
                handles.queue_len,
                handles.frontier_out,
            ];
            let queue_len_init_program = scratch.queue_len_init_program.as_ref().ok_or_else(|| {
                DispatchError::BackendError(
                    "resident CSR queue length init program is missing after ensure_programs. Fix: rebuild programs before resident CSR queue dispatch.".to_string(),
                )
            })?;
            let queue_program = scratch.queue_program.as_ref().ok_or_else(|| {
                DispatchError::BackendError(
                    "resident CSR queue program is missing after ensure_programs. Fix: rebuild programs before resident CSR queue dispatch.".to_string(),
                )
            })?;
            let steps = [
                ResidentDispatchStep {
                    program: queue_len_init_program,
                    handle_ids: &queue_len_handles,
                    grid_override: Some([1, 1, 1]),
                },
                ResidentDispatchStep {
                    program: queue_program,
                    handle_ids: &queue_handles,
                    grid_override: Some(frontier_word_grid(graph.words)?),
                },
                ResidentDispatchStep {
                    program: traverse_program,
                    handle_ids: &traverse_handles,
                    grid_override: Some(traverse_grid),
                },
            ];
            dispatcher.upload_resident_many_sequence_read_ranges_into(
                &[(handles.frontier, scratch.frontier_bytes.as_slice())],
                &steps,
                &read_ranges,
                &mut scratch.readbacks,
            )?;
        }
        ResidentCsrQueueMaterializer::DeterministicWordPrefix => {
            let clear_handles = [handles.frontier_out];
            let word_prefix = word_prefix_scratch(graph.words)?;
            let (word_partials, block_totals) = word_prefix_handles(handles)?;
            let word_count_handles = [handles.frontier, word_partials, block_totals];
            let queue_handles = [
                handles.frontier,
                word_partials,
                block_totals,
                handles.active_queue,
                handles.queue_len,
            ];
            let word_counts_program = scratch.word_counts_program.as_ref().ok_or_else(|| {
                DispatchError::BackendError(
                    "resident CSR queue word-count scan program is missing after ensure_programs. Fix: rebuild programs before resident CSR queue dispatch.".to_string(),
                )
            })?;
            let clear_frontier_out_program =
                scratch.clear_frontier_out_program.as_ref().ok_or_else(|| {
                    DispatchError::BackendError(
                        "resident CSR queue output clear program is missing after ensure_programs. Fix: rebuild programs before resident CSR queue dispatch.".to_string(),
                    )
                })?;
            let queue_program = scratch.queue_program.as_ref().ok_or_else(|| {
                DispatchError::BackendError(
                    "resident CSR queue word-prefix scatter program is missing after ensure_programs. Fix: rebuild programs before resident CSR queue dispatch.".to_string(),
                )
            })?;
            let block_offsets_handles = [block_totals];
            if frontier_word_prefix_uses_precomputed_offsets(word_prefix.block_count) {
                let block_offsets_program =
                    scratch.word_block_offsets_program.as_ref().ok_or_else(|| {
                    DispatchError::BackendError(
                        "resident CSR queue block-offset scan program is missing after ensure_programs. Fix: rebuild programs before resident CSR queue dispatch.".to_string(),
                    )
                })?;
                let steps = [
                    ResidentDispatchStep {
                        program: clear_frontier_out_program,
                        handle_ids: &clear_handles,
                        grid_override: Some(frontier_word_grid(graph.words)?),
                    },
                    ResidentDispatchStep {
                        program: word_counts_program,
                        handle_ids: &word_count_handles,
                        grid_override: Some([word_prefix.block_count, 1, 1]),
                    },
                    ResidentDispatchStep {
                        program: block_offsets_program,
                        handle_ids: &block_offsets_handles,
                        grid_override: Some([1, 1, 1]),
                    },
                    ResidentDispatchStep {
                        program: queue_program,
                        handle_ids: &queue_handles,
                        grid_override: Some(frontier_word_grid(graph.words)?),
                    },
                    ResidentDispatchStep {
                        program: traverse_program,
                        handle_ids: &traverse_handles,
                        grid_override: Some(traverse_grid),
                    },
                ];
                dispatcher.upload_resident_many_sequence_read_ranges_into(
                    &[(handles.frontier, scratch.frontier_bytes.as_slice())],
                    &steps,
                    &read_ranges,
                    &mut scratch.readbacks,
                )?;
            } else {
                let steps = [
                    ResidentDispatchStep {
                        program: clear_frontier_out_program,
                        handle_ids: &clear_handles,
                        grid_override: Some(frontier_word_grid(graph.words)?),
                    },
                    ResidentDispatchStep {
                        program: word_counts_program,
                        handle_ids: &word_count_handles,
                        grid_override: Some([word_prefix.block_count, 1, 1]),
                    },
                    ResidentDispatchStep {
                        program: queue_program,
                        handle_ids: &queue_handles,
                        grid_override: Some(frontier_word_grid(graph.words)?),
                    },
                    ResidentDispatchStep {
                        program: traverse_program,
                        handle_ids: &traverse_handles,
                        grid_override: Some(traverse_grid),
                    },
                ];
                dispatcher.upload_resident_many_sequence_read_ranges_into(
                    &[(handles.frontier, scratch.frontier_bytes.as_slice())],
                    &steps,
                    &read_ranges,
                    &mut scratch.readbacks,
                )?;
            }
        }
    }
    output.clear();
    output.extend_from_slice(&scratch.readbacks[0]);
    Ok(())
}

fn ensure_scratch(
    dispatcher: &dyn OptimizerDispatcher,
    scratch: &mut ResidentCsrQueueScratch,
    words: usize,
    queue_capacity: u32,
    materializer: ResidentCsrQueueMaterializer,
) -> Result<(), DispatchError> {
    let frontier_bytes = u32_word_bytes(words, "resident CSR queue scratch frontier")?;
    if matches!(
        scratch.handles,
        Some(handles)
            if handles.frontier_bytes == frontier_bytes
                && handles.queue_capacity >= queue_capacity
                && handles.materializer == materializer
    ) {
        return Ok(());
    }
    scratch.free(dispatcher)?;
    match materializer {
        ResidentCsrQueueMaterializer::AtomicWordScan => {
            let [frontier, active_queue, queue_len, frontier_out] = alloc_resident_buffers(
                dispatcher,
                [
                    frontier_bytes,
                    u32_word_bytes(
                        queue_capacity as usize,
                        "resident CSR queue scratch active_queue",
                    )?,
                    u32_word_bytes(1, "resident CSR queue scratch queue_len")?,
                    frontier_bytes,
                ],
                "resident CSR queue scratch",
            )?;
            scratch.handles = Some(ResidentCsrQueueScratchHandles {
                frontier,
                active_queue,
                queue_len,
                frontier_out,
                word_partials: None,
                block_totals: None,
                queue_capacity,
                frontier_bytes,
                materializer,
            });
        }
        ResidentCsrQueueMaterializer::DeterministicWordPrefix => {
            let word_prefix = word_prefix_scratch(words)?;
            let [frontier, active_queue, queue_len, frontier_out, word_partials, block_totals] =
                alloc_resident_buffers(
                    dispatcher,
                    [
                        frontier_bytes,
                        u32_word_bytes(
                            queue_capacity as usize,
                            "resident CSR queue scratch active_queue",
                        )?,
                        u32_word_bytes(1, "resident CSR queue scratch queue_len")?,
                        frontier_bytes,
                        u32_word_bytes(
                            word_prefix.partial_words,
                            "resident CSR queue scratch word_partials",
                        )?,
                        u32_word_bytes(
                            word_prefix.block_total_words,
                            "resident CSR queue scratch block_totals",
                        )?,
                    ],
                    "resident CSR queue scratch",
                )?;
            scratch.handles = Some(ResidentCsrQueueScratchHandles {
                frontier,
                active_queue,
                queue_len,
                frontier_out,
                word_partials: Some(word_partials),
                block_totals: Some(block_totals),
                queue_capacity,
                frontier_bytes,
                materializer,
            });
        }
    }
    Ok(())
}

fn ensure_programs(
    scratch: &mut ResidentCsrQueueScratch,
    graph: &ResidentCsrQueueGraph,
    queue_capacity: u32,
    allow_mask: u32,
    materializer: ResidentCsrQueueMaterializer,
) -> Result<(), DispatchError> {
    let shape = ResidentCsrQueueProgramShape {
        node_count: graph.node_count,
        edge_count: graph.edge_count,
        queue_capacity,
        allow_mask,
        materializer,
        traverse_kind: resident_csr_queue_traverse_kind(graph.max_row_degree),
    };
    if scratch.cached_shape == Some(shape) {
        return Ok(());
    }
    scratch.word_counts_program = None;
    scratch.word_block_offsets_program = None;
    scratch.queue_len_init_program = None;
    scratch.clear_frontier_out_program = None;
    match shape.materializer {
        ResidentCsrQueueMaterializer::AtomicWordScan => {
            scratch.queue_len_init_program = Some(frontier_queue_len_init("queue_len"));
            scratch.queue_program = Some(frontier_words_to_queue_clear_out_parallel(
                "frontier",
                "active_queue",
                "queue_len",
                "frontier_out",
                graph.node_count,
                queue_capacity,
            ));
        }
        ResidentCsrQueueMaterializer::DeterministicWordPrefix => {
            scratch.clear_frontier_out_program =
                Some(bitset_zero("frontier_out", graph.words as u32));
            let word_prefix = word_prefix_scratch(graph.words)?;
            scratch.word_counts_program = Some(frontier_word_counts_scan_pass_a(
                "frontier",
                "word_partials",
                "block_totals",
                graph.node_count,
            ));
            if frontier_word_prefix_uses_precomputed_offsets(word_prefix.block_count) {
                scratch.word_block_offsets_program = Some(frontier_word_block_offsets_in_place(
                    "block_totals",
                    graph.node_count,
                ));
                scratch.queue_program = Some(frontier_word_block_offsets_to_queue_parallel(
                    "frontier",
                    "word_partials",
                    "block_totals",
                    "active_queue",
                    "queue_len",
                    graph.node_count,
                    queue_capacity,
                ));
            } else {
                scratch.queue_program = Some(frontier_word_block_prefix_to_queue_parallel(
                    "frontier",
                    "word_partials",
                    "block_totals",
                    "active_queue",
                    "queue_len",
                    graph.node_count,
                    queue_capacity,
                ));
            }
        }
    }
    scratch.traverse_program = Some(match shape.traverse_kind {
        ResidentCsrQueueTraverseKind::RowSerial => csr_queue_forward_traverse(
            "active_queue",
            "queue_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            graph.node_count,
            graph.edge_count,
            queue_capacity,
            allow_mask,
        ),
        ResidentCsrQueueTraverseKind::RowStrided
        | ResidentCsrQueueTraverseKind::MixedSplit { .. } => csr_queue_strided_forward_traverse(
            "active_queue",
            "queue_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            graph.node_count,
            graph.edge_count,
            queue_capacity,
            allow_mask,
        ),
    });
    scratch.cached_shape = Some(shape);
    Ok(())
}

fn word_prefix_scratch(words: usize) -> Result<FrontierWordPrefixScratch, DispatchError> {
    frontier_word_prefix_scratch(words).map_err(DispatchError::BackendError)
}

fn word_prefix_handles(
    handles: ResidentCsrQueueScratchHandles,
) -> Result<(u64, u64), DispatchError> {
    let word_partials = handles.word_partials.ok_or_else(|| {
        DispatchError::BackendError(
            "resident CSR queue word-prefix scratch is missing word_partials. Fix: rebuild scratch before resident CSR queue dispatch.".to_string(),
        )
    })?;
    let block_totals = handles.block_totals.ok_or_else(|| {
        DispatchError::BackendError(
            "resident CSR queue word-prefix scratch is missing block_totals. Fix: rebuild scratch before resident CSR queue dispatch.".to_string(),
        )
    })?;
    Ok((word_partials, block_totals))
}

fn frontier_word_grid(words: usize) -> Result<[u32; 3], DispatchError> {
    frontier_word_dispatch_grid(words).map_err(DispatchError::BackendError)
}
