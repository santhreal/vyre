use std::time::Instant;

use crate::api::case::{BenchContext, BenchError};
use crate::api::resident::ResidentInputSet;
use vyre_driver::{ResidentDispatchStep, ResidentReadRange, TimedDispatchResult};
use vyre_foundation::ir::Program;

use super::{
    GraphCsrSkewedQueuePrepared, QUEUE_ACTIVE_QUEUE_INDEX, QUEUE_EDGE_KIND_INDEX,
    QUEUE_EDGE_OFFSETS_INDEX, QUEUE_EDGE_TARGETS_INDEX, QUEUE_FRONTIER_IN_INDEX,
    QUEUE_FRONTIER_OUT_INDEX, QUEUE_LEN_INDEX, QUEUE_RESET_GRID,
};

const QUEUE_RESET_RESOURCE_INDICES: [usize; 1] = [QUEUE_LEN_INDEX];
const QUEUE_BUILD_RESOURCE_INDICES: [usize; 4] = [
    QUEUE_FRONTIER_IN_INDEX,
    QUEUE_ACTIVE_QUEUE_INDEX,
    QUEUE_LEN_INDEX,
    QUEUE_FRONTIER_OUT_INDEX,
];
const QUEUE_TRAVERSE_RESOURCE_INDICES: [usize; 6] = [
    QUEUE_ACTIVE_QUEUE_INDEX,
    QUEUE_LEN_INDEX,
    QUEUE_EDGE_OFFSETS_INDEX,
    QUEUE_EDGE_TARGETS_INDEX,
    QUEUE_EDGE_KIND_INDEX,
    QUEUE_FRONTIER_OUT_INDEX,
];

pub(super) struct QueueSequenceRun {
    pub(super) outputs: Vec<Vec<u8>>,
    pub(super) wall_ns: u64,
    pub(super) dispatch_ns: Option<u64>,
    pub(super) resident_used: bool,
    pub(super) bytes_read: u64,
    pub(super) bytes_written: u64,
}

pub(super) fn dispatch_resident_queue_sequence(
    ctx: &BenchContext,
    prepared: &GraphCsrSkewedQueuePrepared,
    resident: &ResidentInputSet,
    workgroup: [u32; 3],
) -> Result<QueueSequenceRun, BenchError> {
    let reset_resources =
        resident.resources_for_indices(&QUEUE_RESET_RESOURCE_INDICES, "skewed CSR queue reset")?;
    let queue_resources =
        resident.resources_for_indices(&QUEUE_BUILD_RESOURCE_INDICES, "skewed CSR queue build")?;
    let traverse_resources = resident.resources_for_indices(
        &QUEUE_TRAVERSE_RESOURCE_INDICES,
        "skewed CSR queue traverse",
    )?;
    let reset_step = ResidentDispatchStep {
        program: &prepared.reset_program,
        resources: &reset_resources,
        grid_override: Some(QUEUE_RESET_GRID),
    };
    let queue_step = ResidentDispatchStep {
        program: &prepared.queue_program,
        resources: &queue_resources,
        grid_override: Some(frontier_word_grid(prepared.stats.frontier_words, workgroup)),
    };
    let traverse_step = ResidentDispatchStep {
        program: &prepared.traverse_program,
        resources: &traverse_resources,
        grid_override: Some(prepared.traverse_grid),
    };
    let read_ranges = [ResidentReadRange {
        resource: &traverse_resources[5],
        byte_offset: 0,
        byte_len: prepared.baseline_output.len(),
    }];

    let mut frontier_output = Vec::with_capacity(prepared.baseline_output.len());
    let started = Instant::now();
    ctx.preferred_backend
        .dispatch_resident_sequence_read_ranges_into(
            &[reset_step, queue_step, traverse_step],
            &read_ranges,
            &mut [&mut frontier_output],
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    let wall_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let bytes_written = frontier_output.len() as u64;

    Ok(QueueSequenceRun {
        outputs: vec![frontier_output],
        wall_ns,
        dispatch_ns: None,
        resident_used: true,
        bytes_read: 0,
        bytes_written,
    })
}

pub(super) fn dispatch_host_queue_sequence(
    ctx: &BenchContext,
    prepared: &GraphCsrSkewedQueuePrepared,
    workgroup: [u32; 3],
) -> Result<QueueSequenceRun, BenchError> {
    let started = Instant::now();
    let reset_inputs = vec![prepared.inputs[QUEUE_LEN_INDEX].clone()];
    let reset = dispatch_queue_stage(
        ctx,
        &prepared.reset_program,
        reset_inputs,
        QUEUE_RESET_GRID,
        prepared.reset_program.workgroup_size(),
    )?;
    let reset_queue_len = stage_output(&reset, 0, "skewed CSR queue reset queue_len")?.clone();

    let queue_inputs = vec![
        prepared.inputs[QUEUE_FRONTIER_IN_INDEX].clone(),
        prepared.inputs[QUEUE_ACTIVE_QUEUE_INDEX].clone(),
        reset_queue_len,
        prepared.inputs[QUEUE_FRONTIER_OUT_INDEX].clone(),
    ];
    let queue = dispatch_queue_stage(
        ctx,
        &prepared.queue_program,
        queue_inputs,
        frontier_word_grid(prepared.stats.frontier_words, workgroup),
        workgroup,
    )?;
    let active_queue = stage_output(&queue, 0, "skewed CSR queue build active_queue")?.clone();
    let queue_len = stage_output(&queue, 1, "skewed CSR queue build queue_len")?.clone();
    let cleared_frontier_out =
        stage_output(&queue, 2, "skewed CSR queue build frontier_out")?.clone();

    let traverse_inputs = vec![
        active_queue,
        queue_len,
        prepared.inputs[QUEUE_EDGE_OFFSETS_INDEX].clone(),
        prepared.inputs[QUEUE_EDGE_TARGETS_INDEX].clone(),
        prepared.inputs[QUEUE_EDGE_KIND_INDEX].clone(),
        cleared_frontier_out,
    ];
    let traverse = dispatch_queue_stage(
        ctx,
        &prepared.traverse_program,
        traverse_inputs,
        prepared.traverse_grid,
        workgroup,
    )?;
    let wall_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    let bytes_read = queue_stage_input_bytes(&reset.inputs)
        .saturating_add(queue_stage_input_bytes(&queue.inputs))
        .saturating_add(queue_stage_input_bytes(&traverse.inputs));
    let bytes_written = queue_stage_output_bytes(&reset.outputs)
        .saturating_add(queue_stage_output_bytes(&queue.outputs))
        .saturating_add(queue_stage_output_bytes(&traverse.outputs));
    let dispatch_ns = sum_dispatch_ns([&reset.timed, &queue.timed, &traverse.timed]);

    Ok(QueueSequenceRun {
        outputs: traverse.outputs,
        wall_ns,
        dispatch_ns,
        resident_used: false,
        bytes_read,
        bytes_written,
    })
}

fn frontier_word_grid(frontier_words: u32, workgroup: [u32; 3]) -> [u32; 3] {
    [frontier_words.div_ceil(workgroup[0]).max(1), 1, 1]
}

struct QueueStageRun {
    inputs: Vec<Vec<u8>>,
    outputs: Vec<Vec<u8>>,
    timed: TimedDispatchResult,
}

fn dispatch_queue_stage(
    ctx: &BenchContext,
    program: &Program,
    inputs: Vec<Vec<u8>>,
    grid_override: [u32; 3],
    workgroup: [u32; 3],
) -> Result<QueueStageRun, BenchError> {
    let mut config = ctx.dispatch_config.clone();
    config.workgroup_override = Some(workgroup);
    config.grid_override = Some(grid_override);
    let timed = ctx
        .dispatch_timed(program, &inputs, &config)
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    let outputs = timed.outputs.clone();
    Ok(QueueStageRun {
        inputs,
        outputs,
        timed,
    })
}

fn stage_output<'a>(
    stage: &'a QueueStageRun,
    output_index: usize,
    context: &str,
) -> Result<&'a Vec<u8>, BenchError> {
    stage.outputs.get(output_index).ok_or_else(|| {
        BenchError::ExecutionFailed(format!(
            "{context} did not produce output index {output_index}. Fix: preserve the skewed CSR queue sequence buffer layout."
        ))
    })
}

fn queue_stage_input_bytes(inputs: &[Vec<u8>]) -> u64 {
    inputs.iter().map(Vec::len).sum::<usize>() as u64
}

fn queue_stage_output_bytes(outputs: &[Vec<u8>]) -> u64 {
    outputs.iter().map(Vec::len).sum::<usize>() as u64
}

fn sum_dispatch_ns<const N: usize>(stages: [&TimedDispatchResult; N]) -> Option<u64> {
    let mut total = 0_u64;
    for stage in stages {
        total = total.saturating_add(stage.device_ns?);
    }
    Some(total)
}
