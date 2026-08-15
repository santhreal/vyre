use super::{ResidentCsrQueueGraph, ResidentCsrQueueScratch, ResidentCsrQueueScratchShape};
use vyre_primitives::graph::csr_frontier_queue::validate_frontier_queue_query;

use crate::dispatch_buffers::u32_word_bytes;
use crate::graph::dispatch::csr_frontier_queue_programs::ResidentCsrQueueProgramShape;
use crate::graph::dispatch::csr_frontier_queue_scratch::{
    frontier_word_dispatch_grid, frontier_word_prefix_scratch,
    frontier_word_prefix_uses_precomputed_offsets, resident_csr_queue_frontier_stats,
    resident_csr_queue_materializer_for_stats, resident_csr_queue_split_low_grid,
    resident_csr_queue_traverse_grid, resident_csr_queue_traverse_kind_for_graph_stats,
    FrontierWordPrefixScratch, ResidentCsrQueueMaterializer, ResidentCsrQueueSlotPlan,
    ResidentCsrQueueTraverseKind,
};
use vyre_foundation::program_dispatch::{
    DispatchError, ProgramDispatcher, ResidentDispatchStep, ResidentReadRange,
};

/// Run one sparse frontier query over a resident CSR graph.
pub fn run_resident_csr_queue_query_into(
    dispatcher: &dyn ProgramDispatcher,
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
    let traverse_kind = resident_csr_queue_traverse_kind_for_graph_stats(
        graph.node_count,
        graph.max_row_degree,
        graph.high_degree_source_count,
        effective_queue_capacity,
    );
    ensure_scratch(
        dispatcher,
        scratch,
        graph.words,
        effective_queue_capacity,
        materializer,
        traverse_kind,
    )?;
    let slots = scratch.slots.ok_or_else(|| {
        DispatchError::BackendError(
            "resident CSR queue scratch handles are missing after ensure_scratch. Fix: rebuild scratch before resident CSR queue dispatch.".to_string(),
        )
    })?;
    let word_prefix_blocks = match materializer {
        ResidentCsrQueueMaterializer::DeterministicWordPrefix => {
            word_prefix_scratch(graph.words)?.block_count
        }
        ResidentCsrQueueMaterializer::AtomicWordScan => 0,
    };
    let precomputed_block_offsets = matches!(
        materializer,
        ResidentCsrQueueMaterializer::DeterministicWordPrefix
    ) && frontier_word_prefix_uses_precomputed_offsets(word_prefix_blocks);
    scratch.programs.ensure(
        "frontier",
        ResidentCsrQueueProgramShape {
            node_count: graph.node_count,
            edge_count: graph.edge_count,
            words: graph.words,
            queue_capacity: effective_queue_capacity,
            allow_mask,
            materializer,
            traverse_kind,
        },
        precomputed_block_offsets,
    );

    scratch.frontier_bytes.clear();
    vyre_primitives::wire::append_u32_slice_le_bytes(frontier_words, &mut scratch.frontier_bytes);

    // Handle-id arrays outlive `steps`, which borrows them.
    let word_grid = frontier_word_grid(graph.words)?;
    let (word_partials, block_totals) = match materializer {
        ResidentCsrQueueMaterializer::DeterministicWordPrefix => {
            slots.word_prefix().map_err(DispatchError::BackendError)?
        }
        ResidentCsrQueueMaterializer::AtomicWordScan => (0, 0),
    };
    let (high_queue, high_len) = match traverse_kind {
        ResidentCsrQueueTraverseKind::MixedSplit { .. } => {
            slots.high_split().map_err(DispatchError::BackendError)?
        }
        ResidentCsrQueueTraverseKind::RowSerial | ResidentCsrQueueTraverseKind::RowStrided => (0, 0),
    };
    let clear_handles = [slots.frontier_out];
    let queue_len_handles = [slots.queue_len];
    let word_count_handles = [slots.frontier, word_partials, block_totals];
    let block_offsets_handles = [block_totals];
    let atomic_queue_handles = [
        slots.frontier,
        slots.active_queue,
        slots.queue_len,
        slots.frontier_out,
    ];
    let prefix_queue_handles = [
        slots.frontier,
        word_partials,
        block_totals,
        slots.active_queue,
        slots.queue_len,
    ];
    let base_traverse_handles = [
        slots.active_queue,
        slots.queue_len,
        graph.edge_offsets_handle,
        graph.edge_targets_handle,
        graph.edge_kind_mask_handle,
        slots.frontier_out,
    ];
    let high_len_handles = [high_len];
    let split_handles = [
        slots.active_queue,
        slots.queue_len,
        graph.edge_offsets_handle,
        graph.edge_targets_handle,
        graph.edge_kind_mask_handle,
        slots.frontier_out,
        high_queue,
        high_len,
    ];
    let high_traverse_handles = [
        high_queue,
        high_len,
        graph.edge_offsets_handle,
        graph.edge_targets_handle,
        graph.edge_kind_mask_handle,
        slots.frontier_out,
    ];

    let programs = &scratch.programs;
    let mut steps = Vec::new();
    match materializer {
        ResidentCsrQueueMaterializer::AtomicWordScan => {
            steps.push(ResidentDispatchStep {
                program: programs.queue_len_init()?,
                handle_ids: &queue_len_handles,
                grid_override: Some([1, 1, 1]),
            });
            steps.push(ResidentDispatchStep {
                program: programs.queue()?,
                handle_ids: &atomic_queue_handles,
                grid_override: Some(word_grid),
            });
        }
        ResidentCsrQueueMaterializer::DeterministicWordPrefix => {
            steps.push(ResidentDispatchStep {
                program: programs.clear_frontier_out()?,
                handle_ids: &clear_handles,
                grid_override: Some(word_grid),
            });
            steps.push(ResidentDispatchStep {
                program: programs.word_counts()?,
                handle_ids: &word_count_handles,
                grid_override: Some([word_prefix_blocks, 1, 1]),
            });
            if precomputed_block_offsets {
                steps.push(ResidentDispatchStep {
                    program: programs.word_block_offsets()?,
                    handle_ids: &block_offsets_handles,
                    grid_override: Some([1, 1, 1]),
                });
            }
            steps.push(ResidentDispatchStep {
                program: programs.queue()?,
                handle_ids: &prefix_queue_handles,
                grid_override: Some(word_grid),
            });
        }
    }
    let traverse_program = programs.traverse()?;
    if let ResidentCsrQueueTraverseKind::MixedSplit {
        high_queue_capacity,
    } = traverse_kind
    {
        steps.push(ResidentDispatchStep {
            program: programs.high_len_init()?,
            handle_ids: &high_len_handles,
            grid_override: Some([1, 1, 1]),
        });
        steps.push(ResidentDispatchStep {
            program: programs.split_low()?,
            handle_ids: &split_handles,
            grid_override: Some(resident_csr_queue_split_low_grid(effective_queue_capacity)),
        });
        steps.push(ResidentDispatchStep {
            program: traverse_program,
            handle_ids: &high_traverse_handles,
            grid_override: Some(resident_csr_queue_traverse_grid(
                high_queue_capacity,
                ResidentCsrQueueTraverseKind::RowStrided,
            )),
        });
    } else {
        steps.push(ResidentDispatchStep {
            program: traverse_program,
            handle_ids: &base_traverse_handles,
            grid_override: Some(resident_csr_queue_traverse_grid(
                effective_queue_capacity,
                traverse_kind,
            )),
        });
    }

    let frontier_bytes = u32_word_bytes(graph.words, "resident CSR queue query frontier")?;
    dispatcher.upload_resident_many_sequence_read_ranges_into(
        &[(slots.frontier, scratch.frontier_bytes.as_slice())],
        steps.as_slice(),
        &[ResidentReadRange {
            handle_id: slots.frontier_out,
            byte_offset: 0,
            byte_len: frontier_bytes,
        }],
        &mut scratch.readbacks,
    )?;
    output.clear();
    output.extend_from_slice(&scratch.readbacks[0]);
    Ok(())
}

fn ensure_scratch(
    dispatcher: &dyn ProgramDispatcher,
    scratch: &mut ResidentCsrQueueScratch,
    words: usize,
    queue_capacity: u32,
    materializer: ResidentCsrQueueMaterializer,
    traverse_kind: ResidentCsrQueueTraverseKind,
) -> Result<(), DispatchError> {
    let frontier_bytes = u32_word_bytes(words, "resident CSR queue scratch frontier")?;
    let high_queue_capacity = match traverse_kind {
        ResidentCsrQueueTraverseKind::MixedSplit {
            high_queue_capacity,
        } => high_queue_capacity,
        ResidentCsrQueueTraverseKind::RowSerial | ResidentCsrQueueTraverseKind::RowStrided => 0,
    };
    if matches!(
        scratch.shape,
        Some(shape)
            if shape.frontier_bytes == frontier_bytes
                && shape.queue_capacity >= queue_capacity
                && shape.high_queue_capacity >= high_queue_capacity
                && shape.materializer == materializer
    ) {
        return Ok(());
    }
    scratch.free(dispatcher)?;
    let plan = ResidentCsrQueueSlotPlan::new(words, queue_capacity, materializer, traverse_kind)
        .map_err(DispatchError::BadInputs)?;
    let handles = dispatcher.alloc_resident_many(plan.byte_lengths())?;
    scratch.slots = Some(plan.slots(&handles).map_err(DispatchError::BackendError)?);
    scratch.shape = Some(ResidentCsrQueueScratchShape {
        queue_capacity,
        high_queue_capacity,
        frontier_bytes,
        materializer,
    });
    Ok(())
}

fn word_prefix_scratch(words: usize) -> Result<FrontierWordPrefixScratch, DispatchError> {
    frontier_word_prefix_scratch(words).map_err(DispatchError::BackendError)
}

fn frontier_word_grid(words: usize) -> Result<[u32; 3], DispatchError> {
    frontier_word_dispatch_grid(words).map_err(DispatchError::BackendError)
}
