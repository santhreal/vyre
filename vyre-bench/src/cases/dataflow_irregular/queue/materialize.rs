use std::time::Instant;

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PreparedCase, WorkloadClass,
};
use crate::api::metric::BenchMetrics;
use crate::api::resident::{input_bytes_total, ResidentInputSet};
use crate::api::suite::SuiteKind;
use vyre_driver::{ResidentDispatchStep, ResidentReadRange, TimedDispatchResult};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_primitives::graph::csr_frontier_queue::frontier_to_queue_parallel;

use super::super::fixture::{
    build_ifds_skewed_fixture, ifds_queue_inputs, ifds_skewed_cpu_oracle, IfdsSkewedStats,
    NODE_COUNT,
};
use super::super::metrics::{ifds_queue_baseline_metric_points, ifds_queue_metric_points};
use super::{ifds_queue_traverse_plan, ifds_sparse_queue_capacity};

pub(in crate::cases::dataflow_irregular) const QUEUE_MATERIALIZE_SUITES: &[SuiteKind] = &[
    SuiteKind::Smoke,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

pub(in crate::cases::dataflow_irregular) const QUEUE_FRONTIER_IN_INDEX: usize = 0;
pub(in crate::cases::dataflow_irregular) const QUEUE_ACTIVE_QUEUE_INDEX: usize = 1;
pub(in crate::cases::dataflow_irregular) const QUEUE_LEN_INDEX: usize = 2;
pub(in crate::cases::dataflow_irregular) const QUEUE_EDGE_OFFSETS_INDEX: usize = 3;
pub(in crate::cases::dataflow_irregular) const QUEUE_EDGE_TARGETS_INDEX: usize = 4;
pub(in crate::cases::dataflow_irregular) const QUEUE_EDGE_KIND_INDEX: usize = 5;
pub(in crate::cases::dataflow_irregular) const QUEUE_FRONTIER_OUT_INDEX: usize = 6;
pub(in crate::cases::dataflow_irregular) const QUEUE_RESET_RESOURCE_INDICES: [usize; 2] =
    [QUEUE_LEN_INDEX, QUEUE_FRONTIER_OUT_INDEX];
pub(in crate::cases::dataflow_irregular) const QUEUE_BUILD_RESOURCE_INDICES: [usize; 3] = [
    QUEUE_FRONTIER_IN_INDEX,
    QUEUE_ACTIVE_QUEUE_INDEX,
    QUEUE_LEN_INDEX,
];
pub(in crate::cases::dataflow_irregular) const QUEUE_TRAVERSE_RESOURCE_INDICES: [usize; 6] = [
    QUEUE_ACTIVE_QUEUE_INDEX,
    QUEUE_LEN_INDEX,
    QUEUE_EDGE_OFFSETS_INDEX,
    QUEUE_EDGE_TARGETS_INDEX,
    QUEUE_EDGE_KIND_INDEX,
    QUEUE_FRONTIER_OUT_INDEX,
];

pub(in crate::cases::dataflow_irregular) struct DataflowIfdsSkewedQueuePrepared {
    pub(in crate::cases::dataflow_irregular) reset_program: Program,
    pub(in crate::cases::dataflow_irregular) queue_program: Program,
    pub(in crate::cases::dataflow_irregular) traverse_program: Program,
    pub(in crate::cases::dataflow_irregular) traverse_grid: [u32; 3],
    pub(in crate::cases::dataflow_irregular) row_strided_traverse: bool,
    pub(in crate::cases::dataflow_irregular) inputs: Vec<Vec<u8>>,
    pub(in crate::cases::dataflow_irregular) input_bytes_total: u64,
    pub(in crate::cases::dataflow_irregular) baseline_output: Vec<u8>,
    pub(in crate::cases::dataflow_irregular) baseline_wall_ns: u64,
    pub(in crate::cases::dataflow_irregular) stats: IfdsSkewedStats,
    pub(in crate::cases::dataflow_irregular) queue_capacity: u32,
    pub(in crate::cases::dataflow_irregular) resident: Option<ResidentInputSet>,
}

/// Queue-materializing IFDS step for sparse active frontiers.
struct DataflowIfdsSkewedQueueMaterializeStep;

impl BenchCase for DataflowIfdsSkewedQueueMaterializeStep {
    fn id(&self) -> BenchId {
        BenchId("dataflow.ifds.skewed.queue_materialize_step.1m".to_string())
    }

    fn metadata(&self) -> BenchMetadata {
        BenchMetadata {
            id: self.id(),
            name: "Dataflow IFDS Skewed Queue Materialize Step 1M".to_string(),
            description: "One sparse-frontier IFDS propagation step over a million-node skewed exploded-supergraph CSR using GPU-resident queue materialization and queue-driven traversal".to_string(),
            tags: vec![
                "dataflow".to_string(),
                "ifds".to_string(),
                "graph".to_string(),
                "csr".to_string(),
                "frontier-queue".to_string(),
                "bitset".to_string(),
                "skewed-degree".to_string(),
                "irregular".to_string(),
                "resident".to_string(),
                "release".to_string(),
            ],
            layer: BenchLayer::Libs,
            workload: WorkloadClass::Macro,
            determinism: DeterminismClass::Deterministic,
            owner_crate: "vyre-primitives".to_string(),
        }
    }

    fn suites(&self) -> &'static [SuiteKind] {
        QUEUE_MATERIALIZE_SUITES
    }

    fn requirements(&self) -> BenchRequirements {
        BenchRequirements {
            needs_gpu: true,
            needs_network: false,
            min_vram_bytes: Some(96 * 1024 * 1024),
            min_input_bytes: Some(u64::from(NODE_COUNT) * 12),
            feature_set: vec![
                "dataflow".to_string(),
                "ifds".to_string(),
                "skewed-csr".to_string(),
                "frontier-queue".to_string(),
                "resident-sequence".to_string(),
            ],
        }
    }

    fn bytes_touched(&self, prepared: &PreparedCase) -> (u64, u64) {
        prepared
            .downcast_ref::<DataflowIfdsSkewedQueuePrepared>()
            .map(|prepared| {
                (
                    prepared.input_bytes_total,
                    prepared.baseline_output.len() as u64,
                )
            })
            .unwrap_or((0, 0))
    }

    fn prepare(&self, ctx: &mut BenchContext) -> Result<PreparedCase, BenchError> {
        Ok(Box::new(prepare_ifds_skewed_queue_materialize_step(Some(
            ctx,
        ))?))
    }

    fn program<'a>(&self, prepared: &'a PreparedCase) -> Option<&'a Program> {
        prepared
            .downcast_ref::<DataflowIfdsSkewedQueuePrepared>()
            .map(|prepared| &prepared.traverse_program)
    }

    fn run(
        &self,
        ctx: &mut BenchContext,
        prepared: &mut PreparedCase,
    ) -> Result<BenchRun, BenchError> {
        let prepared = prepared
            .downcast_ref::<DataflowIfdsSkewedQueuePrepared>()
            .ok_or_else(|| {
                BenchError::ExecutionFailed(
                    "prepared IFDS queue payload had the wrong type".to_string(),
                )
            })?;
        let workgroup = prepared.queue_program.workgroup_size();
        if workgroup.contains(&0) {
            return Err(BenchError::ExecutionFailed(format!(
                "IFDS queue benchmark received invalid workgroup {:?}. Fix: use positive dispatch dimensions.",
                workgroup
            )));
        }
        if let Some(override_workgroup) = ctx.dispatch_config.workgroup_override {
            if override_workgroup != workgroup {
                return Err(BenchError::ExecutionFailed(format!(
                    "IFDS queue resident sequence uses program workgroup {:?}, but received override {:?}. Fix: run the queue sequence without a workgroup override or rebuild all sequence programs.",
                    workgroup, override_workgroup
                )));
            }
        }

        let sequence = if let Some(resident) = prepared.resident.as_ref() {
            dispatch_resident_queue_sequence(ctx, prepared, resident, workgroup)?
        } else {
            dispatch_host_queue_sequence(ctx, prepared, workgroup)?
        };
        let output_bytes = sequence.outputs.iter().map(Vec::len).sum::<usize>() as u64;

        Ok(BenchRun {
            metrics: BenchMetrics {
                wall_ns: Some(sequence.wall_ns),
                dispatch_ns: sequence.dispatch_ns,
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(output_bytes),
                bytes_read: Some(sequence.bytes_read),
                bytes_written: Some(sequence.bytes_written),
                bytes_touched: Some(sequence.bytes_read.saturating_add(sequence.bytes_written)),
                custom: ifds_queue_metric_points(
                    prepared.stats,
                    prepared.queue_capacity,
                    prepared.baseline_wall_ns,
                    sequence.wall_ns,
                    sequence.resident_used,
                    workgroup[0],
                    true,
                    prepared.row_strided_traverse,
                ),
                ..Default::default()
            },
            baseline_metrics: Some(BenchMetrics {
                wall_ns: Some(prepared.baseline_wall_ns),
                input_bytes: Some(prepared.input_bytes_total),
                output_bytes: Some(prepared.baseline_output.len() as u64),
                custom: ifds_queue_baseline_metric_points(prepared.stats, prepared.queue_capacity),
                ..Default::default()
            }),
            outputs: sequence.outputs,
            baseline_outputs: Some(vec![prepared.baseline_output.clone()]),
        })
    }

    fn verify(&self, _ctx: &mut BenchContext, run: &BenchRun) -> Result<Correctness, BenchError> {
        run.verify_exact_outputs()
    }
}

pub(in crate::cases::dataflow_irregular) fn prepare_ifds_skewed_queue_materialize_step(
    ctx: Option<&BenchContext>,
) -> Result<DataflowIfdsSkewedQueuePrepared, BenchError> {
    let fixture = build_ifds_skewed_fixture(NODE_COUNT)?;
    let queue_capacity = ifds_sparse_queue_capacity(fixture.stats.active_sources)?;
    let reset_program = ifds_queue_reset_program(fixture.stats.frontier_words);
    let queue_program = frontier_to_queue_parallel(
        "frontier_in",
        "active_queue",
        "queue_len",
        fixture.stats.nodes,
        queue_capacity,
    );
    let traverse_plan = ifds_queue_traverse_plan(
        fixture.stats.max_degree,
        fixture.stats.nodes,
        fixture.stats.edges,
        queue_capacity,
    );

    let baseline_start = Instant::now();
    let oracle = ifds_skewed_cpu_oracle(&fixture);
    let baseline_wall_ns = baseline_start
        .elapsed()
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64;
    let mut stats = fixture.stats;
    stats.allowed_edges_from_active = oracle.allowed_edges_from_active;
    stats.filtered_edges_from_active = oracle.filtered_edges_from_active;
    stats.output_words_set = oracle.output_words_set;

    let inputs = ifds_queue_inputs(&fixture, queue_capacity)?;
    let input_bytes_total = input_bytes_total(&inputs);
    let resident = ctx
        .map(|ctx| ResidentInputSet::upload_optional(ctx, &inputs, "dataflow IFDS queue"))
        .transpose()?
        .flatten();

    Ok(DataflowIfdsSkewedQueuePrepared {
        reset_program,
        queue_program,
        traverse_program: traverse_plan.program,
        traverse_grid: traverse_plan.grid,
        row_strided_traverse: traverse_plan.row_strided,
        inputs,
        input_bytes_total,
        baseline_output: vyre_primitives::wire::pack_u32_slice(&oracle.output),
        baseline_wall_ns,
        stats,
        queue_capacity,
        resident,
    })
}

pub(in crate::cases::dataflow_irregular) fn ifds_queue_reset_program(
    frontier_words: u32,
) -> Program {
    let idx = Expr::InvocationId { axis: 0 };
    Program::wrapped(
        vec![
            BufferDecl::storage("queue_len", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
            BufferDecl::storage("frontier_out", 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(frontier_words.max(1)),
        ],
        [256, 1, 1],
        vec![
            Node::if_then(
                Expr::eq(idx.clone(), Expr::u32(0)),
                vec![Node::store("queue_len", Expr::u32(0), Expr::u32(0))],
            ),
            Node::if_then(
                Expr::lt(idx, Expr::u32(frontier_words)),
                vec![Node::store(
                    "frontier_out",
                    Expr::InvocationId { axis: 0 },
                    Expr::u32(0),
                )],
            ),
        ],
    )
}

struct QueueSequenceRun {
    outputs: Vec<Vec<u8>>,
    wall_ns: u64,
    dispatch_ns: Option<u64>,
    resident_used: bool,
    bytes_read: u64,
    bytes_written: u64,
}

fn dispatch_resident_queue_sequence(
    ctx: &BenchContext,
    prepared: &DataflowIfdsSkewedQueuePrepared,
    resident: &ResidentInputSet,
    workgroup: [u32; 3],
) -> Result<QueueSequenceRun, BenchError> {
    let reset_resources =
        resident.resources_for_indices(&QUEUE_RESET_RESOURCE_INDICES, "IFDS queue reset")?;
    let queue_resources =
        resident.resources_for_indices(&QUEUE_BUILD_RESOURCE_INDICES, "IFDS queue build")?;
    let traverse_resources =
        resident.resources_for_indices(&QUEUE_TRAVERSE_RESOURCE_INDICES, "IFDS queue traverse")?;
    let reset_step = ResidentDispatchStep {
        program: &prepared.reset_program,
        resources: &reset_resources,
        grid_override: Some([
            prepared.stats.frontier_words.div_ceil(workgroup[0]).max(1),
            1,
            1,
        ]),
    };
    let queue_step = ResidentDispatchStep {
        program: &prepared.queue_program,
        resources: &queue_resources,
        grid_override: Some([prepared.stats.nodes.div_ceil(workgroup[0]).max(1), 1, 1]),
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

fn dispatch_host_queue_sequence(
    ctx: &BenchContext,
    prepared: &DataflowIfdsSkewedQueuePrepared,
    workgroup: [u32; 3],
) -> Result<QueueSequenceRun, BenchError> {
    let started = Instant::now();
    let reset_inputs = vec![
        prepared.inputs[QUEUE_LEN_INDEX].clone(),
        prepared.inputs[QUEUE_FRONTIER_OUT_INDEX].clone(),
    ];
    let reset = dispatch_queue_stage(
        ctx,
        &prepared.reset_program,
        reset_inputs,
        [
            prepared.stats.frontier_words.div_ceil(workgroup[0]).max(1),
            1,
            1,
        ],
        workgroup,
    )?;
    let reset_queue_len = stage_output(&reset, 0, "IFDS queue reset queue_len")?.clone();
    let reset_frontier_out = stage_output(&reset, 1, "IFDS queue reset frontier_out")?.clone();

    let queue_inputs = vec![
        prepared.inputs[QUEUE_FRONTIER_IN_INDEX].clone(),
        prepared.inputs[QUEUE_ACTIVE_QUEUE_INDEX].clone(),
        reset_queue_len,
    ];
    let queue = dispatch_queue_stage(
        ctx,
        &prepared.queue_program,
        queue_inputs,
        [prepared.stats.nodes.div_ceil(workgroup[0]).max(1), 1, 1],
        workgroup,
    )?;
    let active_queue = stage_output(&queue, 0, "IFDS queue build active_queue")?.clone();
    let queue_len = stage_output(&queue, 1, "IFDS queue build queue_len")?.clone();

    let traverse_inputs = vec![
        active_queue,
        queue_len,
        prepared.inputs[QUEUE_EDGE_OFFSETS_INDEX].clone(),
        prepared.inputs[QUEUE_EDGE_TARGETS_INDEX].clone(),
        prepared.inputs[QUEUE_EDGE_KIND_INDEX].clone(),
        reset_frontier_out,
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
    config.workgroup_override.get_or_insert(workgroup);
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
            "{context} did not produce output index {output_index}. Fix: preserve the queue sequence buffer layout."
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

inventory::submit! {
    &DataflowIfdsSkewedQueueMaterializeStep as &'static dyn BenchCase
}
