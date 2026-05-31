use std::time::Instant;

use crate::api::case::{BenchContext, BenchError};
use crate::api::resident::ResidentInputSet;
use vyre_driver::{ResidentDispatchStep, ResidentReadRange};
use vyre_foundation::ir::Program;

use super::{
    DataflowIfdsSkewedQueueClosurePrepared, QUEUE_CLOSURE_ACCUMULATOR_INDEX,
    QUEUE_CLOSURE_EDGE_KIND_INDEX, QUEUE_CLOSURE_EDGE_OFFSETS_INDEX,
    QUEUE_CLOSURE_EDGE_TARGETS_INDEX, QUEUE_CLOSURE_LEN_A_INDEX, QUEUE_CLOSURE_LEN_B_INDEX,
    QUEUE_CLOSURE_QUEUE_A_INDEX, QUEUE_CLOSURE_QUEUE_B_INDEX, QUEUE_CLOSURE_SEED_FRONTIER_INDEX,
    QUEUE_CLOSURE_SEED_LEN_INDEX, QUEUE_CLOSURE_SEED_QUEUE_INDEX, QUEUE_CLOSURE_WORKGROUP_SIZE,
};

const QUEUE_CLOSURE_RESET_ACCUMULATOR_RESOURCE: usize = 4;
const QUEUE_CLOSURE_RESET_RESOURCE_INDICES: [usize; 7] = [
    QUEUE_CLOSURE_SEED_FRONTIER_INDEX,
    QUEUE_CLOSURE_SEED_QUEUE_INDEX,
    QUEUE_CLOSURE_SEED_LEN_INDEX,
    QUEUE_CLOSURE_QUEUE_A_INDEX,
    QUEUE_CLOSURE_ACCUMULATOR_INDEX,
    QUEUE_CLOSURE_LEN_A_INDEX,
    QUEUE_CLOSURE_LEN_B_INDEX,
];
const QUEUE_CLOSURE_CLEAR_A_RESOURCE_INDICES: [usize; 1] = [QUEUE_CLOSURE_LEN_A_INDEX];
const QUEUE_CLOSURE_CLEAR_B_RESOURCE_INDICES: [usize; 1] = [QUEUE_CLOSURE_LEN_B_INDEX];
const QUEUE_CLOSURE_DELTA_A_TO_B_RESOURCE_INDICES: [usize; 8] = [
    QUEUE_CLOSURE_QUEUE_A_INDEX,
    QUEUE_CLOSURE_LEN_A_INDEX,
    QUEUE_CLOSURE_EDGE_OFFSETS_INDEX,
    QUEUE_CLOSURE_EDGE_TARGETS_INDEX,
    QUEUE_CLOSURE_EDGE_KIND_INDEX,
    QUEUE_CLOSURE_ACCUMULATOR_INDEX,
    QUEUE_CLOSURE_QUEUE_B_INDEX,
    QUEUE_CLOSURE_LEN_B_INDEX,
];
const QUEUE_CLOSURE_DELTA_B_TO_A_RESOURCE_INDICES: [usize; 8] = [
    QUEUE_CLOSURE_QUEUE_B_INDEX,
    QUEUE_CLOSURE_LEN_B_INDEX,
    QUEUE_CLOSURE_EDGE_OFFSETS_INDEX,
    QUEUE_CLOSURE_EDGE_TARGETS_INDEX,
    QUEUE_CLOSURE_EDGE_KIND_INDEX,
    QUEUE_CLOSURE_ACCUMULATOR_INDEX,
    QUEUE_CLOSURE_QUEUE_A_INDEX,
    QUEUE_CLOSURE_LEN_A_INDEX,
];

pub(super) struct QueueClosureSequenceRun {
    pub(super) outputs: Vec<Vec<u8>>,
    pub(super) wall_ns: u64,
}

pub(super) fn dispatch_resident_queue_closure_sequence(
    ctx: &BenchContext,
    prepared: &DataflowIfdsSkewedQueueClosurePrepared,
    resident: &ResidentInputSet,
) -> Result<QueueClosureSequenceRun, BenchError> {
    let mut resource_sets = Vec::new();
    resource_sets.push(resident.resources_for_indices(
        &QUEUE_CLOSURE_RESET_RESOURCE_INDICES,
        "IFDS queue closure reset",
    )?);
    resource_sets.push(resident.resources_for_indices(
        &QUEUE_CLOSURE_CLEAR_A_RESOURCE_INDICES,
        "IFDS queue closure clear queue A length",
    )?);
    resource_sets.push(resident.resources_for_indices(
        &QUEUE_CLOSURE_CLEAR_B_RESOURCE_INDICES,
        "IFDS queue closure clear queue B length",
    )?);
    resource_sets.push(resident.resources_for_indices(
        &QUEUE_CLOSURE_DELTA_A_TO_B_RESOURCE_INDICES,
        "IFDS queue closure delta A to B",
    )?);
    resource_sets.push(resident.resources_for_indices(
        &QUEUE_CLOSURE_DELTA_B_TO_A_RESOURCE_INDICES,
        "IFDS queue closure delta B to A",
    )?);

    struct StepSpec<'a> {
        program: &'a Program,
        resource_set: usize,
        grid: [u32; 3],
    }

    let reset_grid = [
        prepared
            .stats
            .frontier_words
            .max(prepared.seed_queue_len)
            .div_ceil(QUEUE_CLOSURE_WORKGROUP_SIZE[0])
            .max(1),
        1,
        1,
    ];
    let delta_grid = [
        prepared
            .queue_capacity
            .div_ceil(QUEUE_CLOSURE_WORKGROUP_SIZE[0])
            .max(1),
        1,
        1,
    ];
    let mut specs = Vec::with_capacity(1 + (prepared.closure_iterations as usize * 2));
    specs.push(StepSpec {
        program: &prepared.reset_program,
        resource_set: 0,
        grid: reset_grid,
    });
    for iteration in 0..prepared.closure_iterations {
        let clear_resource_set = if iteration & 1 == 0 { 2 } else { 1 };
        let delta_resource_set = if iteration & 1 == 0 { 3 } else { 4 };
        specs.push(StepSpec {
            program: &prepared.clear_len_program,
            resource_set: clear_resource_set,
            grid: [1, 1, 1],
        });
        specs.push(StepSpec {
            program: &prepared.delta_program,
            resource_set: delta_resource_set,
            grid: delta_grid,
        });
    }

    let steps = specs
        .iter()
        .map(|spec| ResidentDispatchStep {
            program: spec.program,
            resources: &resource_sets[spec.resource_set],
            grid_override: Some(spec.grid),
        })
        .collect::<Vec<_>>();
    let read_ranges = [ResidentReadRange {
        resource: &resource_sets[0][QUEUE_CLOSURE_RESET_ACCUMULATOR_RESOURCE],
        byte_offset: 0,
        byte_len: prepared.baseline_output.len(),
    }];

    let mut accumulator_output = Vec::with_capacity(prepared.baseline_output.len());
    let started = Instant::now();
    ctx.preferred_backend
        .dispatch_resident_sequence_read_ranges_into(
            &steps,
            &read_ranges,
            &mut [&mut accumulator_output],
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    let wall_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;

    Ok(QueueClosureSequenceRun {
        outputs: vec![accumulator_output],
        wall_ns,
    })
}
