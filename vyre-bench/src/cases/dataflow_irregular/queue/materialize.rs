use std::time::Instant;

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchId, BenchLayer, BenchMetadata, BenchRequirements,
    BenchRun, Correctness, DeterminismClass, PreparedCase, WorkloadClass,
};
use crate::api::metric::BenchMetrics;
use crate::api::resident::{input_bytes_total, ResidentInputSet};
use crate::api::suite::SuiteKind;
use vyre_foundation::ir::Program;
use vyre_primitives::graph::csr_frontier_queue::{
    frontier_queue_len_init, frontier_words_to_queue_clear_out_parallel,
};
use vyre_primitives::graph::csr_queue_split::{
    csr_queue_split_low_dispatch_grid, csr_queue_split_low_forward_traverse,
    csr_queue_split_mixed_logical_lanes, CSR_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD,
};
use vyre_primitives::graph::csr_queue_strided::{
    csr_queue_strided_forward_dispatch_grid, csr_queue_strided_forward_traverse,
};

use super::super::fixture::{
    build_ifds_skewed_fixture, ifds_active_high_degree_sources, ifds_queue_inputs,
    ifds_skewed_cpu_oracle, IfdsSkewedStats, IFDS_REACH_MASK, NODE_COUNT,
};
use super::super::metrics::{ifds_queue_baseline_metric_points, ifds_queue_metric_points};
use super::{
    ifds_queue_traverse_logical_lanes, ifds_queue_traverse_plan, ifds_sparse_queue_capacity,
};
use crate::cases::queue_stage::{QUEUE_RESET_GRID};

pub(in crate::cases::dataflow_irregular) const QUEUE_MATERIALIZE_SUITES: &[SuiteKind] = &[
    SuiteKind::Smoke,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];


pub(in crate::cases::dataflow_irregular) struct DataflowIfdsSkewedQueuePrepared {
    pub(in crate::cases::dataflow_irregular) reset_program: Program,
    pub(in crate::cases::dataflow_irregular) queue_program: Program,
    pub(in crate::cases::dataflow_irregular) traverse_program: Program,
    pub(in crate::cases::dataflow_irregular) traverse_grid: [u32; 3],
    pub(in crate::cases::dataflow_irregular) row_strided_traverse: bool,
    pub(in crate::cases::dataflow_irregular) split_high_degree_traverse: bool,
    pub(in crate::cases::dataflow_irregular) high_traverse_program: Option<Program>,
    pub(in crate::cases::dataflow_irregular) high_traverse_grid: [u32; 3],
    pub(in crate::cases::dataflow_irregular) high_degree_queue_capacity: u32,
    pub(in crate::cases::dataflow_irregular) traverse_logical_lanes: u64,
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

    fn workload_fingerprint_bytes(&self, prepared: &PreparedCase) -> Option<[u8; 32]> {
        prepared
            .downcast_ref::<DataflowIfdsSkewedQueuePrepared>()
            .map(ifds_queue_materialize_sequence_fingerprint)
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
                    prepared.high_degree_queue_capacity,
                    prepared.traverse_logical_lanes,
                    prepared.baseline_wall_ns,
                    sequence.wall_ns,
                    sequence.resident_used,
                    workgroup[0],
                    true,
                    prepared.row_strided_traverse,
                    prepared.split_high_degree_traverse,
                    CSR_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD,
                    true,
                    QUEUE_RESET_GRID.into_iter().product(),
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
    let high_degree_queue_capacity =
        ifds_active_high_degree_sources(&fixture, CSR_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD)?;
    let reset_program = frontier_queue_len_init("queue_len");
    let queue_program = frontier_words_to_queue_clear_out_parallel(
        "frontier_in",
        "active_queue",
        "queue_len",
        "frontier_out",
        fixture.stats.nodes,
        queue_capacity,
    );
    let traverse_plan = ifds_queue_materialize_traverse_plan(
        fixture.stats.max_degree,
        fixture.stats.nodes,
        fixture.stats.edges,
        queue_capacity,
        high_degree_queue_capacity,
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

    let inputs = ifds_queue_inputs(&fixture, queue_capacity, high_degree_queue_capacity)?;
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
        split_high_degree_traverse: traverse_plan.split_high_degree,
        high_traverse_program: traverse_plan.high_program,
        high_traverse_grid: traverse_plan.high_grid,
        high_degree_queue_capacity,
        traverse_logical_lanes: traverse_plan.logical_lanes,
        inputs,
        input_bytes_total,
        baseline_output: vyre_primitives::wire::pack_u32_slice(&oracle.output),
        baseline_wall_ns,
        stats,
        queue_capacity,
        resident,
    })
}

pub(in crate::cases::dataflow_irregular) fn ifds_queue_materialize_sequence_fingerprint(
    prepared: &DataflowIfdsSkewedQueuePrepared,
) -> [u8; 32] {
    crate::cases::queue_stage::queue_materialize_sequence_fingerprint(
        b"vyre-bench:dataflow.ifds.skewed.queue_materialize_step.sequence:v2",
        [
            &prepared.reset_program,
            &prepared.queue_program,
            &prepared.traverse_program,
        ],
        prepared.high_traverse_program.as_ref(),
        [
            QUEUE_RESET_GRID,
            prepared.queue_program.workgroup_size(),
            prepared.traverse_grid,
            prepared.high_traverse_grid,
        ],
        &[
            prepared.high_degree_queue_capacity,
            u32::from(prepared.split_high_degree_traverse),
            CSR_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD,
        ],
    )
}

struct IfdsQueueMaterializeTraversePlan {
    program: Program,
    grid: [u32; 3],
    row_strided: bool,
    split_high_degree: bool,
    high_program: Option<Program>,
    high_grid: [u32; 3],
    logical_lanes: u64,
}

fn ifds_queue_materialize_traverse_plan(
    max_degree: u32,
    node_count: u32,
    edge_count: u32,
    queue_capacity: u32,
    high_degree_queue_capacity: u32,
) -> IfdsQueueMaterializeTraversePlan {
    if ifds_queue_should_use_split_high_degree(queue_capacity, high_degree_queue_capacity) {
        let program = csr_queue_split_low_forward_traverse(
            "active_queue",
            "queue_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            "high_queue",
            "high_len",
            node_count,
            edge_count,
            queue_capacity,
            high_degree_queue_capacity,
            CSR_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD,
            IFDS_REACH_MASK,
        );
        let high_program = csr_queue_strided_forward_traverse(
            "high_queue",
            "high_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "frontier_out",
            node_count,
            edge_count,
            high_degree_queue_capacity,
            IFDS_REACH_MASK,
        );
        return IfdsQueueMaterializeTraversePlan {
            program,
            grid: csr_queue_split_low_dispatch_grid(queue_capacity),
            row_strided: true,
            split_high_degree: true,
            high_program: Some(high_program),
            high_grid: csr_queue_strided_forward_dispatch_grid(high_degree_queue_capacity),
            logical_lanes: csr_queue_split_mixed_logical_lanes(
                queue_capacity,
                high_degree_queue_capacity,
            ),
        };
    }

    let plan = ifds_queue_traverse_plan(max_degree, node_count, edge_count, queue_capacity);
    IfdsQueueMaterializeTraversePlan {
        logical_lanes: ifds_queue_traverse_logical_lanes(queue_capacity, plan.row_strided),
        program: plan.program,
        grid: plan.grid,
        row_strided: plan.row_strided,
        split_high_degree: false,
        high_program: None,
        high_grid: [1, 1, 1],
    }
}

pub(in crate::cases::dataflow_irregular) const fn ifds_queue_should_use_split_high_degree(
    queue_capacity: u32,
    high_degree_queue_capacity: u32,
) -> bool {
    high_degree_queue_capacity > 0 && high_degree_queue_capacity < queue_capacity
}

crate::cases::queue_stage::define_resident_queue_sequence_dispatch!(
    dispatch_resident_queue_sequence,
    DataflowIfdsSkewedQueuePrepared,
    "IFDS"
);

crate::cases::queue_stage::define_host_queue_sequence_dispatch!(
    dispatch_host_queue_sequence,
    DataflowIfdsSkewedQueuePrepared,
    "IFDS"
);

inventory::submit! {
    &DataflowIfdsSkewedQueueMaterializeStep as &'static dyn BenchCase
}
