//! Sparse-frontier IFDS propagation from a queue the GPU materializes itself.
//!
//! The payload, workgroup check, dispatch sequence and run assembly are owned by
//! [`crate::cases::queue_materialize`]; the traversal choice by
//! [`crate::cases::queue_traverse_plan`]. What is this case's own: the exploded
//! supergraph fixture, the CPU oracle, and its metric points.

use crate::api::metric::elapsed_ns;
use std::time::Instant;

use crate::api::case::{BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, WorkloadClass};
use crate::api::resident::{input_bytes_total, ResidentInputSet};
use crate::api::suite::SuiteKind;
use crate::cases::harness::{verify_exact, CaseOps, HarnessCase, WorkloadDescription};
use crate::cases::queue_materialize::{
    dispatch_queue_materialize_sequence, queue_materialize_bytes_touched, queue_materialize_run,
    queue_materialize_sequence_fingerprint, queue_materialize_workgroup, QueueMaterializePrepared,
};
use crate::cases::queue_stage::QUEUE_RESET_GRID;
use crate::cases::queue_traverse_plan::queue_traverse_plan;
use vyre_foundation::ir::Program;
use vyre_libs::graph::csr_frontier_queue::{
    frontier_queue_len_init, frontier_words_to_queue_clear_out_parallel,
};
use vyre_libs::graph::csr_queue_split::CSR_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD;

use super::super::fixture::{
    build_ifds_skewed_fixture, ifds_active_high_degree_sources, ifds_queue_inputs,
    ifds_skewed_cpu_oracle, IfdsSkewedStats, IFDS_REACH_MASK, NODE_COUNT,
};
use super::super::metrics::{ifds_queue_baseline_metric_points, ifds_queue_metric_points};
use super::ifds_sparse_queue_capacity;

pub(in crate::cases::dataflow_irregular) const QUEUE_MATERIALIZE_SUITES: &[SuiteKind] = &[
    SuiteKind::Smoke,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

pub(in crate::cases::dataflow_irregular) type DataflowIfdsSkewedQueuePrepared =
    QueueMaterializePrepared<IfdsSkewedStats>;

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "dataflow.ifds.skewed.queue_materialize_step.1m",
    name: "Dataflow IFDS Skewed Queue Materialize Step 1M",
    summary: "One sparse-frontier IFDS propagation step over a million-node skewed exploded-supergraph CSR using GPU-resident queue materialization and queue-driven traversal",
    tags: &[
        "dataflow",
        "ifds",
        "graph",
        "csr",
        "frontier-queue",
        "bitset",
        "skewed-degree",
        "irregular",
        "resident",
        "release",
    ],
    layer: BenchLayer::Libs,
    workload: WorkloadClass::Macro,
    owner_crate: "vyre-primitives",
    suites: QUEUE_MATERIALIZE_SUITES,
    min_vram_bytes: Some(96 * 1024 * 1024),
    min_input_bytes: Some(NODE_COUNT as u64 * 12),
    feature_set: &[
        "dataflow",
        "ifds",
        "skewed-csr",
        "frontier-queue",
        "resident-sequence",
    ],
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<DataflowIfdsSkewedQueuePrepared> = CaseOps {
    build: build_case,
    measure,
    verify: verify_exact,
    program: traverse_program,
    fingerprint: Some(ifds_queue_materialize_sequence_fingerprint),
    bytes_touched: queue_materialize_bytes_touched,
};

static CASE: HarnessCase<DataflowIfdsSkewedQueuePrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn build_case(ctx: &mut BenchContext) -> Result<DataflowIfdsSkewedQueuePrepared, BenchError> {
    prepare_ifds_skewed_queue_materialize_step(Some(ctx))
}

fn traverse_program(prepared: &DataflowIfdsSkewedQueuePrepared) -> Option<&Program> {
    Some(&prepared.traverse_program)
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut DataflowIfdsSkewedQueuePrepared,
) -> Result<BenchRun, BenchError> {
    let workgroup = queue_materialize_workgroup(ctx, prepared, "IFDS queue")?;
    let sequence = dispatch_queue_materialize_sequence(ctx, prepared, workgroup, "IFDS")?;
    let custom = ifds_queue_metric_points(
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
    );
    let baseline_custom =
        ifds_queue_baseline_metric_points(prepared.stats, prepared.queue_capacity);
    Ok(queue_materialize_run(
        prepared,
        sequence,
        custom,
        baseline_custom,
    ))
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
    let traverse_plan = queue_traverse_plan(
        fixture.stats.max_degree,
        fixture.stats.nodes,
        fixture.stats.edges,
        queue_capacity,
        high_degree_queue_capacity,
        IFDS_REACH_MASK,
        CSR_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD,
    );

    let baseline_start = Instant::now();
    let oracle = ifds_skewed_cpu_oracle(&fixture);
    let baseline_wall_ns = elapsed_ns(baseline_start);
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
    queue_materialize_sequence_fingerprint(
        b"vyre-bench:dataflow.ifds.skewed.queue_materialize_step.sequence:v2",
        prepared,
        &[
            prepared.high_degree_queue_capacity,
            u32::from(prepared.split_high_degree_traverse),
            CSR_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD,
        ],
    )
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
