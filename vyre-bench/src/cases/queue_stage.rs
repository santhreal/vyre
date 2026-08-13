//! Shared host-dispatch bookkeeping for queue-based benchmark stages.

use std::time::Instant;

use crate::api::case::{BenchContext, BenchError};
use crate::api::resident::ResidentInputSet;
use vyre_driver::{ResidentDispatchStep, ResidentReadRange, TimedDispatchResult};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

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

pub(crate) fn queue_materialize_sequence_fingerprint(
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_queue_closure_inputs(
    frontier: &[u32],
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    active_sources: u64,
    queue_capacity: u32,
    context: &str,
    materialize_seed_queue: impl FnOnce(usize) -> Result<Vec<u32>, BenchError>,
) -> Result<Vec<Vec<u8>>, BenchError> {
    if u64::from(queue_capacity) < active_sources {
        return Err(BenchError::EnvironmentInvalid(format!(
            "{context} queue closure requires queue_capacity >= active_sources, got capacity={queue_capacity} active_sources={active_sources}. Fix: size ping-pong queues for the seed frontier."
        )));
    }
    let seed_queue_len = u32::try_from(active_sources).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "{context} queue closure active source count {active_sources} exceeds u32 indexing. Fix: split the seed queue."
        ))
    })?;
    let queue_bytes = (queue_capacity as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            BenchError::EnvironmentInvalid(format!(
                "{context} queue closure queue_capacity={queue_capacity} overflows host buffer sizing. Fix: split the frontier queue."
            ))
        })?;
    let seed_frontier = vyre_primitives::wire::pack_u32_slice(frontier);
    let seed_queue = materialize_seed_queue(seed_queue_len as usize)?;

    Ok(vec![
        seed_frontier.clone(),
        vyre_primitives::wire::pack_u32_slice(&seed_queue),
        vyre_primitives::wire::pack_u32_slice(&[seed_queue_len]),
        vec![0_u8; queue_bytes],
        vyre_primitives::wire::pack_u32_slice(&[0]),
        vec![0_u8; queue_bytes],
        vyre_primitives::wire::pack_u32_slice(&[0]),
        vyre_primitives::wire::pack_u32_slice(edge_offsets),
        vyre_primitives::wire::pack_u32_slice(edge_targets),
        vyre_primitives::wire::pack_u32_slice(edge_kind_mask),
        seed_frontier,
    ])
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

pub(crate) struct ResidentQueueSequenceSpec<'a> {
    pub(crate) reset_program: &'a Program,
    pub(crate) queue_program: &'a Program,
    pub(crate) traverse_program: &'a Program,
    pub(crate) high_traverse_program: Option<&'a Program>,
    pub(crate) frontier_words: u32,
    pub(crate) traverse_grid: [u32; 3],
    pub(crate) high_traverse_grid: [u32; 3],
    pub(crate) baseline_output_len: usize,
    pub(crate) reset_grid: [u32; 3],
    pub(crate) reset_indices: &'a [usize],
    pub(crate) high_reset_indices: &'a [usize],
    pub(crate) queue_indices: &'a [usize],
    pub(crate) traverse_indices: &'a [usize],
    pub(crate) split_indices: &'a [usize],
    pub(crate) high_traverse_indices: &'a [usize],
    pub(crate) labels: [&'static str; 6],
}

pub(crate) fn dispatch_resident_queue_sequence(
    ctx: &BenchContext,
    spec: ResidentQueueSequenceSpec<'_>,
    resident: &ResidentInputSet,
    workgroup: [u32; 3],
) -> Result<QueueSequenceRun, BenchError> {
    let [reset_label, high_reset_label, queue_label, traverse_label, split_label, high_label] =
        spec.labels;
    let reset_resources = resident.resources_for_indices(spec.reset_indices, reset_label)?;
    let high_reset_resources =
        resident.resources_for_indices(spec.high_reset_indices, high_reset_label)?;
    let queue_resources = resident.resources_for_indices(spec.queue_indices, queue_label)?;
    let reset_step = ResidentDispatchStep {
        program: spec.reset_program,
        resources: &reset_resources,
        grid_override: Some(spec.reset_grid),
        workgroup_override: None,
    };
    let high_reset_step = ResidentDispatchStep {
        program: spec.reset_program,
        resources: &high_reset_resources,
        grid_override: Some(spec.reset_grid),
        workgroup_override: None,
    };
    let queue_step = ResidentDispatchStep {
        program: spec.queue_program,
        resources: &queue_resources,
        grid_override: Some([spec.frontier_words.div_ceil(workgroup[0]).max(1), 1, 1]),
        workgroup_override: None,
    };

    let mut frontier_output = Vec::with_capacity(spec.baseline_output_len);
    let started = Instant::now();
    if let Some(high_program) = spec.high_traverse_program {
        let split_resources = resident.resources_for_indices(spec.split_indices, split_label)?;
        let high_resources =
            resident.resources_for_indices(spec.high_traverse_indices, high_label)?;
        let split_step = ResidentDispatchStep {
            program: spec.traverse_program,
            resources: &split_resources,
            grid_override: Some(spec.traverse_grid),
            workgroup_override: None,
        };
        let high_step = ResidentDispatchStep {
            program: high_program,
            resources: &high_resources,
            grid_override: Some(spec.high_traverse_grid),
            workgroup_override: None,
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
        let traverse_resources =
            resident.resources_for_indices(spec.traverse_indices, traverse_label)?;
        let traverse_step = ResidentDispatchStep {
            program: spec.traverse_program,
            resources: &traverse_resources,
            grid_override: Some(spec.traverse_grid),
            workgroup_override: None,
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

pub(crate) struct ResidentQueueClosureSpec<'a> {
    pub(crate) reset_program: &'a Program,
    pub(crate) clear_len_program: &'a Program,
    pub(crate) delta_program: &'a Program,
    pub(crate) frontier_words: u32,
    pub(crate) seed_queue_len: u32,
    pub(crate) baseline_output_len: usize,
    pub(crate) closure_iterations: u32,
    pub(crate) delta_grid: [u32; 3],
    pub(crate) workgroup: [u32; 3],
    pub(crate) context: &'static str,
}

pub(crate) struct QueueClosureSequenceRun {
    pub(crate) outputs: Vec<Vec<u8>>,
    pub(crate) wall_ns: u64,
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

/// Binding order of the queue-closure workload, shared the same way.
pub(crate) const QUEUE_CLOSURE_SEED_FRONTIER_INDEX: usize = 0;
pub(crate) const QUEUE_CLOSURE_SEED_QUEUE_INDEX: usize = 1;
pub(crate) const QUEUE_CLOSURE_SEED_LEN_INDEX: usize = 2;
pub(crate) const QUEUE_CLOSURE_QUEUE_A_INDEX: usize = 3;
pub(crate) const QUEUE_CLOSURE_LEN_A_INDEX: usize = 4;
pub(crate) const QUEUE_CLOSURE_QUEUE_B_INDEX: usize = 5;
pub(crate) const QUEUE_CLOSURE_LEN_B_INDEX: usize = 6;
pub(crate) const QUEUE_CLOSURE_EDGE_OFFSETS_INDEX: usize = 7;
pub(crate) const QUEUE_CLOSURE_EDGE_TARGETS_INDEX: usize = 8;
pub(crate) const QUEUE_CLOSURE_EDGE_KIND_INDEX: usize = 9;
pub(crate) const QUEUE_CLOSURE_ACCUMULATOR_INDEX: usize = 10;

macro_rules! define_host_queue_sequence_dispatch {
    ($visibility:vis $name:ident, $prepared:ty, $context:literal) => {
        $visibility fn $name(
            ctx: &$crate::api::case::BenchContext,
            prepared: &$prepared,
            workgroup: [u32; 3],
        ) -> Result<$crate::cases::queue_stage::QueueSequenceRun, $crate::api::case::BenchError> {
            $crate::cases::queue_stage::dispatch_host_queue_sequence(
                ctx,
                $crate::cases::queue_stage::HostQueueSequenceSpec {
                    inputs: &prepared.inputs,
                    reset_program: &prepared.reset_program,
                    queue_program: &prepared.queue_program,
                    traverse_program: &prepared.traverse_program,
                    high_traverse_program: prepared.high_traverse_program.as_ref(),
                    frontier_words: prepared.stats.frontier_words,
                    traverse_grid: prepared.traverse_grid,
                    high_traverse_grid: prepared.high_traverse_grid,
                    context: $context,
                },
                workgroup,
            )
        }
    };
}

pub(crate) use define_host_queue_sequence_dispatch;

macro_rules! define_resident_queue_sequence_dispatch {
    ($visibility:vis $name:ident, $prepared:ty, $context:literal) => {
        $visibility fn $name(
            ctx: &$crate::api::case::BenchContext,
            prepared: &$prepared,
            resident: &$crate::api::resident::ResidentInputSet,
            workgroup: [u32; 3],
        ) -> Result<$crate::cases::queue_stage::QueueSequenceRun, $crate::api::case::BenchError> {
            $crate::cases::queue_stage::dispatch_resident_queue_sequence(
                ctx,
                $crate::cases::queue_stage::ResidentQueueSequenceSpec {
                    reset_program: &prepared.reset_program,
                    queue_program: &prepared.queue_program,
                    traverse_program: &prepared.traverse_program,
                    high_traverse_program: prepared.high_traverse_program.as_ref(),
                    frontier_words: prepared.stats.frontier_words,
                    traverse_grid: prepared.traverse_grid,
                    high_traverse_grid: prepared.high_traverse_grid,
                    baseline_output_len: prepared.baseline_output.len(),
                    reset_grid: $crate::cases::queue_stage::QUEUE_RESET_GRID,
                    reset_indices: &$crate::cases::queue_stage::QUEUE_RESET_RESOURCE_INDICES,
                    high_reset_indices:
                        &$crate::cases::queue_stage::QUEUE_HIGH_RESET_RESOURCE_INDICES,
                    queue_indices: &$crate::cases::queue_stage::QUEUE_BUILD_RESOURCE_INDICES,
                    traverse_indices: &$crate::cases::queue_stage::QUEUE_TRAVERSE_RESOURCE_INDICES,
                    split_indices: &$crate::cases::queue_stage::QUEUE_SPLIT_LOW_RESOURCE_INDICES,
                    high_traverse_indices:
                        &$crate::cases::queue_stage::QUEUE_HIGH_TRAVERSE_RESOURCE_INDICES,
                    labels: [
                        concat!($context, " queue reset"),
                        concat!($context, " high queue reset"),
                        concat!($context, " queue build"),
                        concat!($context, " queue traverse"),
                        concat!($context, " split-low queue traverse"),
                        concat!($context, " high-degree queue traverse"),
                    ],
                },
                resident,
                workgroup,
            )
        }
    };
}

pub(crate) use define_resident_queue_sequence_dispatch;

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

pub(crate) fn build_queue_closure_reset_program(
    frontier_words: u32,
    seed_queue_len: u32,
    queue_capacity: u32,
    workgroup: [u32; 3],
) -> Program {
    let idx = Expr::InvocationId { axis: 0 };
    Program::wrapped(
        vec![
            BufferDecl::storage("frontier_seed", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(frontier_words.max(1)),
            BufferDecl::storage("seed_queue", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(seed_queue_len.max(1)),
            BufferDecl::storage("seed_len", 2, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("active_queue", 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(queue_capacity.max(1)),
            BufferDecl::storage("accumulator", 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(frontier_words.max(1)),
            BufferDecl::storage("queue_a_len", 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
            BufferDecl::storage("queue_b_len", 6, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        workgroup,
        vec![
            Node::if_then(
                Expr::lt(idx.clone(), Expr::u32(frontier_words)),
                vec![Node::store(
                    "accumulator",
                    idx.clone(),
                    Expr::load("frontier_seed", idx.clone()),
                )],
            ),
            Node::if_then(
                Expr::and(
                    Expr::lt(idx.clone(), Expr::u32(queue_capacity)),
                    Expr::and(
                        Expr::lt(idx.clone(), Expr::u32(seed_queue_len)),
                        Expr::lt(idx.clone(), Expr::load("seed_len", Expr::u32(0))),
                    ),
                ),
                vec![Node::store(
                    "active_queue",
                    idx.clone(),
                    Expr::load("seed_queue", idx.clone()),
                )],
            ),
            Node::if_then(
                Expr::eq(idx, Expr::u32(0)),
                vec![
                    Node::store(
                        "queue_a_len",
                        Expr::u32(0),
                        Expr::load("seed_len", Expr::u32(0)),
                    ),
                    Node::store("queue_b_len", Expr::u32(0), Expr::u32(0)),
                ],
            ),
        ],
    )
}
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
    let wall_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
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

/// How a closure of `closure_iterations` half-waves folds into one prefix plus
/// a repeated four-step pair.
///
/// The delta program alternates direction every half-wave, so a pair of
/// half-waves is the shortest repeatable unit. An odd iteration count cannot be
/// expressed as pairs alone: the leading A-to-B half-wave is hoisted into the
/// prefix and the remainder divides evenly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct QueueClosureRepeatedPlan {
    pub(crate) leading_a_to_b_half_wave: bool,
    pub(crate) repeated_pair_count: u32,
}

impl QueueClosureRepeatedPlan {
    /// Half-waves the plan expands to. Must equal the requested iterations.
    pub(crate) const fn total_half_waves(self) -> u32 {
        self.repeated_pair_count
            .saturating_mul(2)
            .saturating_add(self.leading_a_to_b_half_wave as u32)
    }

    /// Dispatches the plan submits: one reset plus two per half-wave.
    pub(crate) const fn dispatch_count(self) -> u32 {
        1_u32.saturating_add(self.total_half_waves().saturating_mul(2))
    }
}

/// Fold an iteration count into its prefix-plus-repeated-pair plan.
pub(crate) const fn queue_closure_repeated_plan(
    closure_iterations: u32,
) -> QueueClosureRepeatedPlan {
    QueueClosureRepeatedPlan {
        leading_a_to_b_half_wave: closure_iterations & 1 == 1,
        repeated_pair_count: closure_iterations / 2,
    }
}

pub(crate) fn dispatch_resident_queue_closure_sequence(
    ctx: &BenchContext,
    prepared: ResidentQueueClosureSpec<'_>,
    resident: &ResidentInputSet,
) -> Result<QueueClosureSequenceRun, BenchError> {
    const RESET_ACCUMULATOR_RESOURCE: usize = 4;
    const RESET_RESOURCE_INDICES: [usize; 7] = [0, 1, 2, 3, 10, 4, 6];
    const CLEAR_A_RESOURCE_INDICES: [usize; 1] = [4];
    const CLEAR_B_RESOURCE_INDICES: [usize; 1] = [6];
    const DELTA_A_TO_B_RESOURCE_INDICES: [usize; 8] = [3, 4, 7, 8, 9, 10, 5, 6];
    const DELTA_B_TO_A_RESOURCE_INDICES: [usize; 8] = [5, 6, 7, 8, 9, 10, 3, 4];

    let resource_sets = [
        resident.resources_for_indices(
            &RESET_RESOURCE_INDICES,
            &format!("{} reset", prepared.context),
        )?,
        resident.resources_for_indices(
            &CLEAR_A_RESOURCE_INDICES,
            &format!("{} clear queue A length", prepared.context),
        )?,
        resident.resources_for_indices(
            &CLEAR_B_RESOURCE_INDICES,
            &format!("{} clear queue B length", prepared.context),
        )?,
        resident.resources_for_indices(
            &DELTA_A_TO_B_RESOURCE_INDICES,
            &format!("{} delta A to B", prepared.context),
        )?,
        resident.resources_for_indices(
            &DELTA_B_TO_A_RESOURCE_INDICES,
            &format!("{} delta B to A", prepared.context),
        )?,
    ];
    let reset_grid = [
        prepared
            .frontier_words
            .max(prepared.seed_queue_len)
            .div_ceil(prepared.workgroup[0])
            .max(1),
        1,
        1,
    ];
    let reset_step = ResidentDispatchStep {
        program: prepared.reset_program,
        resources: &resource_sets[0],
        grid_override: Some(reset_grid),
        workgroup_override: None,
    };
    let read_ranges = [ResidentReadRange {
        resource: &resource_sets[0][RESET_ACCUMULATOR_RESOURCE],
        byte_offset: 0,
        byte_len: prepared.baseline_output_len,
    }];
    let mut accumulator_output = Vec::with_capacity(prepared.baseline_output_len);
    let started = Instant::now();
    let plan = queue_closure_repeated_plan(prepared.closure_iterations);
    let clear_a_step = || ResidentDispatchStep {
        program: prepared.clear_len_program,
        resources: &resource_sets[1],
        grid_override: Some([1, 1, 1]),
        workgroup_override: None,
    };
    let clear_b_step = || ResidentDispatchStep {
        program: prepared.clear_len_program,
        resources: &resource_sets[2],
        grid_override: Some([1, 1, 1]),
        workgroup_override: None,
    };
    let delta_a_to_b_step = || ResidentDispatchStep {
        program: prepared.delta_program,
        resources: &resource_sets[3],
        grid_override: Some(prepared.delta_grid),
        workgroup_override: None,
    };
    let delta_b_to_a_step = || ResidentDispatchStep {
        program: prepared.delta_program,
        resources: &resource_sets[4],
        grid_override: Some(prepared.delta_grid),
        workgroup_override: None,
    };

    if plan.leading_a_to_b_half_wave {
        let prefix_steps = [reset_step, clear_b_step(), delta_a_to_b_step()];
        let repeated_steps = [
            clear_a_step(),
            delta_b_to_a_step(),
            clear_b_step(),
            delta_a_to_b_step(),
        ];
        ctx.dispatch_resident_repeated_sequence_read_ranges_into(
            &prefix_steps,
            &repeated_steps,
            plan.repeated_pair_count,
            &read_ranges,
            &mut [&mut accumulator_output],
        )
    } else {
        let prefix_steps = [reset_step];
        let repeated_steps = [
            clear_b_step(),
            delta_a_to_b_step(),
            clear_a_step(),
            delta_b_to_a_step(),
        ];
        ctx.dispatch_resident_repeated_sequence_read_ranges_into(
            &prefix_steps,
            &repeated_steps,
            plan.repeated_pair_count,
            &read_ranges,
            &mut [&mut accumulator_output],
        )
    }
    .map_err(|error| BenchError::BackendFailed(error.to_string()))?;

    Ok(QueueClosureSequenceRun {
        outputs: vec![accumulator_output],
        wall_ns: started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::mix32;

    /// The repeated-pair plan must expand to exactly the requested half-waves,
    /// in alternating direction, for every iteration count either queue closure
    /// case can produce.
    ///
    /// Both salts are the ones the IFDS and CSR cases each generated against
    /// while they carried their own copy of this plan, so collapsing them onto
    /// one owner did not narrow the input space.
    #[test]
    fn generated_repeated_plan_preserves_every_queue_closure_wave() {
        const CASES: u32 = 10_000;

        for salt in [0xC105_E7E5_u32, 0x6A17_0359] {
            let mut odd_cases = 0_u32;
            let mut repeated_pairs = 0_u64;

            for case in 0..CASES {
                let iterations = mix32(case ^ salt) % 16_385;
                let plan = queue_closure_repeated_plan(iterations);

                assert_eq!(plan.total_half_waves(), iterations, "salt {salt:#x} case {case}");
                assert_eq!(
                    plan.dispatch_count(),
                    1 + iterations.saturating_mul(2),
                    "dispatch count salt {salt:#x} case {case}"
                );
                assert_eq!(
                    plan.leading_a_to_b_half_wave,
                    iterations & 1 == 1,
                    "leading wave parity salt {salt:#x} case {case}"
                );
                assert_eq!(
                    plan.repeated_pair_count,
                    iterations / 2,
                    "pair count salt {salt:#x} case {case}"
                );
                assert_repeated_plan_expands_to_alternating_half_waves(case, iterations, plan);

                odd_cases += u32::from(plan.leading_a_to_b_half_wave);
                repeated_pairs += u64::from(plan.repeated_pair_count);
            }

            assert!(odd_cases > CASES / 3, "salt {salt:#x}");
            assert!(repeated_pairs > u64::from(CASES) * 1_000, "salt {salt:#x}");
        }
    }

    fn assert_repeated_plan_expands_to_alternating_half_waves(
        case: u32,
        iterations: u32,
        plan: QueueClosureRepeatedPlan,
    ) {
        let mut half_wave = 0_u32;
        if plan.leading_a_to_b_half_wave {
            assert_half_wave(case, half_wave, true);
            half_wave += 1;
        }

        for _ in 0..plan.repeated_pair_count {
            if plan.leading_a_to_b_half_wave {
                assert_half_wave(case, half_wave, false);
                half_wave += 1;
                assert_half_wave(case, half_wave, true);
            } else {
                assert_half_wave(case, half_wave, true);
                half_wave += 1;
                assert_half_wave(case, half_wave, false);
            }
            half_wave += 1;
        }

        assert_eq!(half_wave, iterations, "expanded wave count case {case}");
    }

    fn assert_half_wave(case: u32, half_wave: u32, a_to_b: bool) {
        assert_eq!(
            a_to_b,
            half_wave & 1 == 0,
            "half-wave direction case {case} wave {half_wave}"
        );
    }

    /// A zero-iteration closure still submits the reset dispatch and nothing
    /// else; the boundary the generated cases never reach.
    #[test]
    fn a_zero_iteration_closure_submits_only_the_reset() {
        let plan = queue_closure_repeated_plan(0);

        assert_eq!(plan.total_half_waves(), 0);
        assert_eq!(plan.dispatch_count(), 1);
        assert!(!plan.leading_a_to_b_half_wave);
    }
}
