//! Shared host-dispatch bookkeeping for queue-based benchmark stages.

use crate::api::metric::elapsed_ns;
use std::time::Instant;

use crate::api::case::{BenchContext, BenchError};
use crate::api::resident::{stated_launch, ResidentInputSet};
use vyre_driver::{ResidentDispatchStep, ResidentReadRange, TimedDispatchResult};
use vyre_foundation::ir::Program;

mod closure;

pub(crate) use closure::{
    build_queue_closure_inputs, build_queue_closure_reset_program,
    dispatch_resident_queue_closure_sequence, queue_closure_repeated_plan, QueueClosureSequenceRun,
    ResidentQueueClosureSpec,
};
// The binding indices are asserted by the contract suites, not read by a case.
#[cfg(test)]
pub(crate) use closure::{
    QUEUE_CLOSURE_ACCUMULATOR_INDEX, QUEUE_CLOSURE_EDGE_KIND_INDEX,
    QUEUE_CLOSURE_EDGE_OFFSETS_INDEX, QUEUE_CLOSURE_EDGE_TARGETS_INDEX, QUEUE_CLOSURE_LEN_A_INDEX,
    QUEUE_CLOSURE_LEN_B_INDEX, QUEUE_CLOSURE_QUEUE_A_INDEX, QUEUE_CLOSURE_QUEUE_B_INDEX,
    QUEUE_CLOSURE_SEED_FRONTIER_INDEX, QUEUE_CLOSURE_SEED_LEN_INDEX,
    QUEUE_CLOSURE_SEED_QUEUE_INDEX,
};

pub(crate) struct QueueStageRun {
    pub(crate) inputs: Vec<Vec<u8>>,
    pub(crate) outputs: Vec<Vec<u8>>,
    pub(crate) timed: TimedDispatchResult,
}

pub(crate) struct QueueSequenceRun {
    pub(crate) outputs: Vec<Vec<u8>>,
    pub(crate) wall_ns: u64,
    pub(crate) dispatch_ns: Option<u64>,
    pub(crate) resident_used: bool,
    pub(crate) bytes_read: u64,
    pub(crate) bytes_written: u64,
}

/// Hash the programs and grids of a staged queue sequence into one value.
///
/// Named for its inputs rather than for the queue-materialize case, because
/// `cases::queue_materialize::queue_materialize_sequence_fingerprint` is the
/// spelling that takes a prepared case and is the only one a case should call.
/// Two functions with one name meant two things could disagree about what a
/// sample hashed with nothing to say which was meant.
pub(crate) fn staged_sequence_fingerprint(
    domain: &[u8],
    programs: [&Program; 3],
    high_traverse_program: Option<&Program>,
    grids: [[u32; 3]; 4],
    extra_values: &[u32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    for program in programs {
        hasher.update(&program.fingerprint());
    }
    if let Some(program) = high_traverse_program {
        hasher.update(&program.fingerprint());
    }
    for value in grids
        .into_iter()
        .flatten()
        .chain(extra_values.iter().copied())
    {
        hasher.update(&value.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

pub(crate) struct HostQueueSequenceSpec<'a> {
    pub(crate) inputs: &'a [Vec<u8>],
    pub(crate) reset_program: &'a Program,
    pub(crate) queue_program: &'a Program,
    pub(crate) traverse_program: &'a Program,
    pub(crate) high_traverse_program: Option<&'a Program>,
    pub(crate) frontier_words: u32,
    pub(crate) traverse_grid: [u32; 3],
    pub(crate) high_traverse_grid: [u32; 3],
    pub(crate) context: &'static str,
}

/// The resident materialize sequence, which binds the shared queue resource
/// layout below rather than restating it per case.
pub(crate) struct ResidentQueueSequenceSpec<'a> {
    pub(crate) reset_program: &'a Program,
    pub(crate) queue_program: &'a Program,
    pub(crate) traverse_program: &'a Program,
    pub(crate) high_traverse_program: Option<&'a Program>,
    pub(crate) frontier_words: u32,
    pub(crate) traverse_grid: [u32; 3],
    pub(crate) high_traverse_grid: [u32; 3],
    pub(crate) baseline_output_len: usize,
    /// Family noun leading every stage label, e.g. `"IFDS"`.
    pub(crate) context: &'static str,
}

pub(crate) fn dispatch_resident_queue_sequence(
    ctx: &BenchContext,
    spec: ResidentQueueSequenceSpec<'_>,
    resident: &ResidentInputSet,
    workgroup: [u32; 3],
) -> Result<QueueSequenceRun, BenchError> {
    let context = spec.context;
    let reset_resources = resident.resources_for_indices(
        &QUEUE_RESET_RESOURCE_INDICES,
        &format!("{context} queue reset"),
    )?;
    let high_reset_resources = resident.resources_for_indices(
        &QUEUE_HIGH_RESET_RESOURCE_INDICES,
        &format!("{context} high queue reset"),
    )?;
    let queue_resources = resident.resources_for_indices(
        &QUEUE_BUILD_RESOURCE_INDICES,
        &format!("{context} queue build"),
    )?;
    let reset_step = ResidentDispatchStep {
        program: spec.reset_program,
        resources: &reset_resources,
        launch: Some(stated_launch(spec.reset_program, QUEUE_RESET_GRID)?),
    };
    let high_reset_step = ResidentDispatchStep {
        program: spec.reset_program,
        resources: &high_reset_resources,
        launch: Some(stated_launch(spec.reset_program, QUEUE_RESET_GRID)?),
    };
    let queue_step = ResidentDispatchStep {
        program: spec.queue_program,
        resources: &queue_resources,
        launch: Some(stated_launch(
            spec.queue_program,
            [spec.frontier_words.div_ceil(workgroup[0]).max(1), 1, 1],
        )?),
    };

    let mut frontier_output = Vec::with_capacity(spec.baseline_output_len);
    let started = Instant::now();
    if let Some(high_program) = spec.high_traverse_program {
        let split_resources = resident.resources_for_indices(
            &QUEUE_SPLIT_LOW_RESOURCE_INDICES,
            &format!("{context} split-low queue traverse"),
        )?;
        let high_resources = resident.resources_for_indices(
            &QUEUE_HIGH_TRAVERSE_RESOURCE_INDICES,
            &format!("{context} high-degree queue traverse"),
        )?;
        let split_step = ResidentDispatchStep {
            program: spec.traverse_program,
            resources: &split_resources,
            launch: Some(stated_launch(spec.traverse_program, spec.traverse_grid)?),
        };
        let high_step = ResidentDispatchStep {
            program: high_program,
            resources: &high_resources,
            launch: Some(stated_launch(high_program, spec.high_traverse_grid)?),
        };
        let read_ranges = [ResidentReadRange {
            resource: &high_resources[5],
            byte_offset: 0,
            byte_len: spec.baseline_output_len,
        }];
        ctx.dispatch_resident_sequence_read_ranges_into(
            &[
                reset_step,
                high_reset_step,
                queue_step,
                split_step,
                high_step,
            ],
            &read_ranges,
            &mut [&mut frontier_output],
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    } else {
        let traverse_resources = resident.resources_for_indices(
            &QUEUE_TRAVERSE_RESOURCE_INDICES,
            &format!("{context} queue traverse"),
        )?;
        let traverse_step = ResidentDispatchStep {
            program: spec.traverse_program,
            resources: &traverse_resources,
            launch: Some(stated_launch(spec.traverse_program, spec.traverse_grid)?),
        };
        let read_ranges = [ResidentReadRange {
            resource: &traverse_resources[5],
            byte_offset: 0,
            byte_len: spec.baseline_output_len,
        }];
        ctx.dispatch_resident_sequence_read_ranges_into(
            &[reset_step, queue_step, traverse_step],
            &read_ranges,
            &mut [&mut frontier_output],
        )
        .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    }
    let wall_ns = elapsed_ns(started);
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

/// Binding order of the split-queue materialize workload.
///
/// The IFDS and CSR materialize cases bind the same resources in the same
/// order, because they run the same `vyre-primitives` queue programs. The
/// layout is one fact and lives here, next to the sequence dispatch that
/// consumes it, rather than once per case.
pub(crate) const QUEUE_FRONTIER_IN_INDEX: usize = 0;
pub(crate) const QUEUE_ACTIVE_QUEUE_INDEX: usize = 1;
pub(crate) const QUEUE_LEN_INDEX: usize = 2;
pub(crate) const QUEUE_EDGE_OFFSETS_INDEX: usize = 3;
pub(crate) const QUEUE_EDGE_TARGETS_INDEX: usize = 4;
pub(crate) const QUEUE_EDGE_KIND_INDEX: usize = 5;
pub(crate) const QUEUE_FRONTIER_OUT_INDEX: usize = 6;
pub(crate) const QUEUE_HIGH_QUEUE_INDEX: usize = 7;
pub(crate) const QUEUE_HIGH_LEN_INDEX: usize = 8;

/// Both queue-length resets are single-lane counter writes.
pub(crate) const QUEUE_RESET_GRID: [u32; 3] = [1, 1, 1];

pub(crate) const QUEUE_RESET_RESOURCE_INDICES: [usize; 1] = [QUEUE_LEN_INDEX];
pub(crate) const QUEUE_HIGH_RESET_RESOURCE_INDICES: [usize; 1] = [QUEUE_HIGH_LEN_INDEX];
pub(crate) const QUEUE_BUILD_RESOURCE_INDICES: [usize; 4] = [
    QUEUE_FRONTIER_IN_INDEX,
    QUEUE_ACTIVE_QUEUE_INDEX,
    QUEUE_LEN_INDEX,
    QUEUE_FRONTIER_OUT_INDEX,
];
pub(crate) const QUEUE_TRAVERSE_RESOURCE_INDICES: [usize; 6] = [
    QUEUE_ACTIVE_QUEUE_INDEX,
    QUEUE_LEN_INDEX,
    QUEUE_EDGE_OFFSETS_INDEX,
    QUEUE_EDGE_TARGETS_INDEX,
    QUEUE_EDGE_KIND_INDEX,
    QUEUE_FRONTIER_OUT_INDEX,
];
pub(crate) const QUEUE_SPLIT_LOW_RESOURCE_INDICES: [usize; 8] = [
    QUEUE_ACTIVE_QUEUE_INDEX,
    QUEUE_LEN_INDEX,
    QUEUE_EDGE_OFFSETS_INDEX,
    QUEUE_EDGE_TARGETS_INDEX,
    QUEUE_EDGE_KIND_INDEX,
    QUEUE_FRONTIER_OUT_INDEX,
    QUEUE_HIGH_QUEUE_INDEX,
    QUEUE_HIGH_LEN_INDEX,
];
pub(crate) const QUEUE_HIGH_TRAVERSE_RESOURCE_INDICES: [usize; 6] = [
    QUEUE_HIGH_QUEUE_INDEX,
    QUEUE_HIGH_LEN_INDEX,
    QUEUE_EDGE_OFFSETS_INDEX,
    QUEUE_EDGE_TARGETS_INDEX,
    QUEUE_EDGE_KIND_INDEX,
    QUEUE_FRONTIER_OUT_INDEX,
];

pub(crate) fn build_queue_inputs(
    frontier_in: &[u32],
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier_out_seed: &[u32],
    active_sources: u64,
    queue_capacity: u32,
    high_degree_queue_capacity: u32,
    context: &str,
) -> Result<Vec<Vec<u8>>, BenchError> {
    if u64::from(queue_capacity) < active_sources {
        return Err(BenchError::EnvironmentInvalid(format!(
            "{context} requires queue_capacity >= active_sources, got capacity={queue_capacity} active_sources={active_sources}. Fix: size the sparse frontier queue from fixture stats."
        )));
    }
    if high_degree_queue_capacity > queue_capacity {
        return Err(BenchError::EnvironmentInvalid(format!(
            "{context} requires high_degree_queue_capacity <= queue_capacity, got high_degree_queue_capacity={high_degree_queue_capacity} queue_capacity={queue_capacity}. Fix: derive high-degree capacity from active sources."
        )));
    }
    let queue_bytes = (queue_capacity as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            BenchError::EnvironmentInvalid(format!(
                "{context} queue_capacity={queue_capacity} overflows host buffer sizing. Fix: split the frontier queue."
            ))
        })?;
    let high_queue_bytes = (high_degree_queue_capacity as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            BenchError::EnvironmentInvalid(format!(
                "{context} high_degree_queue_capacity={high_degree_queue_capacity} overflows host buffer sizing. Fix: split the high-degree queue."
            ))
        })?;
    Ok(vec![
        vyre_primitives::wire::pack_u32_slice(frontier_in),
        vec![0_u8; queue_bytes],
        vyre_primitives::wire::pack_u32_slice(&[0]),
        vyre_primitives::wire::pack_u32_slice(edge_offsets),
        vyre_primitives::wire::pack_u32_slice(edge_targets),
        vyre_primitives::wire::pack_u32_slice(edge_kind_mask),
        vyre_primitives::wire::pack_u32_slice(frontier_out_seed),
        vec![0_u8; high_queue_bytes],
        vyre_primitives::wire::pack_u32_slice(&[0]),
    ])
}

macro_rules! define_queue_input_builder {
    ($visibility:vis $name:ident, $fixture:ty, $context:literal) => {
        $visibility fn $name(
            fixture: &$fixture,
            queue_capacity: u32,
            high_degree_queue_capacity: u32,
        ) -> Result<Vec<Vec<u8>>, $crate::api::case::BenchError> {
            $crate::cases::queue_stage::build_queue_inputs(
                &fixture.frontier_in,
                &fixture.edge_offsets,
                &fixture.edge_targets,
                &fixture.edge_kind_mask,
                &fixture.frontier_out_seed,
                fixture.stats.active_sources,
                queue_capacity,
                high_degree_queue_capacity,
                $context,
            )
        }
    };
}

pub(crate) use define_queue_input_builder;

pub(crate) fn dispatch_queue_stage(
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

pub(crate) fn stage_output<'a>(
    stage: &'a QueueStageRun,
    output_index: usize,
    context: &str,
) -> Result<&'a Vec<u8>, BenchError> {
    stage.outputs.get(output_index).ok_or_else(|| {
        BenchError::ExecutionFailed(format!(
            "{context} did not produce output index {output_index}. Fix: preserve the queue sequence buffer layout."
        ))
    })
}

pub(crate) fn queue_stage_input_bytes(inputs: &[Vec<u8>]) -> u64 {
    inputs.iter().map(Vec::len).sum::<usize>() as u64
}

pub(crate) fn queue_stage_output_bytes(outputs: &[Vec<u8>]) -> u64 {
    outputs.iter().map(Vec::len).sum::<usize>() as u64
}

pub(crate) fn sum_dispatch_ns<const N: usize>(stages: [&TimedDispatchResult; N]) -> Option<u64> {
    let mut total = 0_u64;
    for stage in stages {
        total = total.saturating_add(stage.device_ns?);
    }
    Some(total)
}

pub(crate) fn dispatch_host_queue_sequence(
    ctx: &BenchContext,
    prepared: HostQueueSequenceSpec<'_>,
    workgroup: [u32; 3],
) -> Result<QueueSequenceRun, BenchError> {
    const FRONTIER_IN: usize = 0;
    const ACTIVE_QUEUE: usize = 1;
    const QUEUE_LEN: usize = 2;
    const EDGE_OFFSETS: usize = 3;
    const EDGE_TARGETS: usize = 4;
    const EDGE_KIND: usize = 5;
    const FRONTIER_OUT: usize = 6;
    const HIGH_QUEUE: usize = 7;
    const HIGH_LEN: usize = 8;

    let started = Instant::now();
    let reset = dispatch_queue_stage(
        ctx,
        prepared.reset_program,
        vec![prepared.inputs[QUEUE_LEN].clone()],
        [1, 1, 1],
        prepared.reset_program.workgroup_size(),
    )?;
    let reset_queue_len = stage_output(
        &reset,
        0,
        &format!("{} queue reset queue_len", prepared.context),
    )?
    .clone();

    let queue = dispatch_queue_stage(
        ctx,
        prepared.queue_program,
        vec![
            prepared.inputs[FRONTIER_IN].clone(),
            prepared.inputs[ACTIVE_QUEUE].clone(),
            reset_queue_len,
            prepared.inputs[FRONTIER_OUT].clone(),
        ],
        [prepared.frontier_words.div_ceil(workgroup[0]).max(1), 1, 1],
        workgroup,
    )?;
    let active_queue = stage_output(
        &queue,
        0,
        &format!("{} queue build active_queue", prepared.context),
    )?
    .clone();
    let queue_len = stage_output(
        &queue,
        1,
        &format!("{} queue build queue_len", prepared.context),
    )?
    .clone();
    let cleared_frontier_out = stage_output(
        &queue,
        2,
        &format!("{} queue build frontier_out", prepared.context),
    )?
    .clone();

    let (outputs, high_reset, traverse_timed, split_low, high_traverse) =
        if let Some(high_program) = prepared.high_traverse_program {
            let high_reset = dispatch_queue_stage(
                ctx,
                prepared.reset_program,
                vec![prepared.inputs[HIGH_LEN].clone()],
                [1, 1, 1],
                prepared.reset_program.workgroup_size(),
            )?;
            let reset_high_len = stage_output(
                &high_reset,
                0,
                &format!("{} high queue reset high_len", prepared.context),
            )?
            .clone();
            let split_low = dispatch_queue_stage(
                ctx,
                prepared.traverse_program,
                vec![
                    active_queue,
                    queue_len,
                    prepared.inputs[EDGE_OFFSETS].clone(),
                    prepared.inputs[EDGE_TARGETS].clone(),
                    prepared.inputs[EDGE_KIND].clone(),
                    cleared_frontier_out,
                    prepared.inputs[HIGH_QUEUE].clone(),
                    reset_high_len,
                ],
                prepared.traverse_grid,
                workgroup,
            )?;
            let frontier_after_low = stage_output(
                &split_low,
                0,
                &format!("{} split-low frontier_out", prepared.context),
            )?
            .clone();
            let high_queue = stage_output(
                &split_low,
                1,
                &format!("{} split-low high_queue", prepared.context),
            )?
            .clone();
            let high_len = stage_output(
                &split_low,
                2,
                &format!("{} split-low high_len", prepared.context),
            )?
            .clone();
            let high_traverse = dispatch_queue_stage(
                ctx,
                high_program,
                vec![
                    high_queue,
                    high_len,
                    prepared.inputs[EDGE_OFFSETS].clone(),
                    prepared.inputs[EDGE_TARGETS].clone(),
                    prepared.inputs[EDGE_KIND].clone(),
                    frontier_after_low,
                ],
                prepared.high_traverse_grid,
                high_program.workgroup_size(),
            )?;
            let outputs = high_traverse.outputs.clone();
            (
                outputs,
                Some(high_reset),
                sum_dispatch_ns([&split_low.timed, &high_traverse.timed]),
                Some(split_low),
                Some(high_traverse),
            )
        } else {
            let traverse = dispatch_queue_stage(
                ctx,
                prepared.traverse_program,
                vec![
                    active_queue,
                    queue_len,
                    prepared.inputs[EDGE_OFFSETS].clone(),
                    prepared.inputs[EDGE_TARGETS].clone(),
                    prepared.inputs[EDGE_KIND].clone(),
                    cleared_frontier_out,
                ],
                prepared.traverse_grid,
                workgroup,
            )?;
            let outputs = traverse.outputs.clone();
            (
                outputs,
                None,
                traverse.timed.device_ns,
                Some(traverse),
                None,
            )
        };
    let wall_ns = elapsed_ns(started);
    let bytes_read = queue_stage_input_bytes(&reset.inputs)
        .saturating_add(queue_stage_input_bytes(&queue.inputs))
        .saturating_add(
            high_reset
                .as_ref()
                .map_or(0, |stage| queue_stage_input_bytes(&stage.inputs)),
        )
        .saturating_add(
            split_low
                .as_ref()
                .map_or(0, |stage| queue_stage_input_bytes(&stage.inputs)),
        )
        .saturating_add(
            high_traverse
                .as_ref()
                .map_or(0, |stage| queue_stage_input_bytes(&stage.inputs)),
        );
    let bytes_written = queue_stage_output_bytes(&reset.outputs)
        .saturating_add(queue_stage_output_bytes(&queue.outputs))
        .saturating_add(
            high_reset
                .as_ref()
                .map_or(0, |stage| queue_stage_output_bytes(&stage.outputs)),
        )
        .saturating_add(
            split_low
                .as_ref()
                .map_or(0, |stage| queue_stage_output_bytes(&stage.outputs)),
        )
        .saturating_add(
            high_traverse
                .as_ref()
                .map_or(0, |stage| queue_stage_output_bytes(&stage.outputs)),
        );
    let prefix_dispatch_ns = high_reset.as_ref().map_or_else(
        || sum_dispatch_ns([&reset.timed, &queue.timed]),
        |stage| sum_dispatch_ns([&reset.timed, &stage.timed, &queue.timed]),
    );
    let dispatch_ns = match (prefix_dispatch_ns, traverse_timed) {
        (Some(prefix), Some(traverse)) => Some(prefix.saturating_add(traverse)),
        _ => None,
    };

    Ok(QueueSequenceRun {
        outputs,
        wall_ns,
        dispatch_ns,
        resident_used: false,
        bytes_read,
        bytes_written,
    })
}
