//! Skewed CSR frontier queue materialization, then queue-driven expansion.
//!
//! The payload, workgroup check, dispatch sequence and run assembly are owned by
//! [`crate::cases::queue_materialize`]; the traversal choice by
//! [`crate::cases::queue_traverse_plan`]. What is this case's own: the fixture,
//! the CPU oracle, the split threshold it holds rows to, and its metric points.

use crate::api::metric::elapsed_ns;
use std::time::Instant;

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, DeterminismClass, WorkloadClass,
};
use crate::api::resident::{input_bytes_total, ResidentInputSet};
use crate::cases::harness::{verify_exact, CaseOps, HarnessCase, WorkloadDescription};
use crate::cases::queue_materialize::{
    dispatch_queue_materialize_sequence, queue_materialize_bytes_touched, queue_materialize_run,
    queue_materialize_sequence_fingerprint, queue_materialize_workgroup, QueueMaterializePrepared,
};
use crate::cases::queue_stage::QUEUE_RESET_GRID;
use crate::cases::queue_traverse_plan::queue_traverse_plan;
use vyre_foundation::ir::Program;
use vyre_primitives::graph::csr_frontier_queue::{
    frontier_queue_len_init, frontier_words_to_queue_clear_out_parallel,
};
use vyre_primitives::graph::csr_queue_strided::CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE;

use super::metrics::{skewed_csr_baseline_metric_points, skewed_csr_queue_metric_points};
use super::support::{
    build_skewed_csr_fixture, skewed_csr_active_high_degree_sources, skewed_csr_cpu_oracle,
    skewed_csr_queue_capacity, skewed_csr_queue_inputs, SkewedCsrStats, CSR_ALLOW_MASK,
    CSR_NODE_COUNT, SUITES,
};

/// Degree at which a queued graph row is handed to the high-degree pass.
///
/// Two lane teams' worth of edges is enough for a graph hub, which is a lower
/// bar than the IFDS family sets: graph rows are shorter and more numerous, so
/// splitting earlier keeps more of them off the strided path.
pub(super) const GRAPH_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD: u32 =
    CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE * 2;

pub(super) type GraphCsrSkewedQueuePrepared = QueueMaterializePrepared<SkewedCsrStats>;

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "primitives.graph.csr_skewed_queue_materialize.1m",
    name: "Skewed CSR Queue Materialize 1M",
    summary: "GPU-resident packed-frontier queue materialization plus queue-driven CSR expansion over a million-node skewed graph",
    tags: &[
        "graph",
        "frontier",
        "csr",
        "frontier-queue",
        "skewed-degree",
        "irregular",
        "resident",
        "release",
    ],
    layer: BenchLayer::Foundation,
    workload: WorkloadClass::Macro,
    determinism: DeterminismClass::Deterministic,
    owner_crate: "vyre-primitives",
    suites: SUITES,
    needs_gpu: true,
    needs_network: false,
    min_vram_bytes: Some(96 * 1024 * 1024),
    min_input_bytes: Some(CSR_NODE_COUNT as u64 * 12),
    feature_set: &[
        "graph.csr",
        "graph.frontier.bitset",
        "graph.frontier.queue",
        "graph.skewed-degree",
        "resident-sequence",
    ],
    contract: None,
};

static OPS: CaseOps<GraphCsrSkewedQueuePrepared> = CaseOps {
    build: build_case,
    measure,
    verify: verify_exact,
    program: traverse_program,
    fingerprint: Some(graph_queue_materialize_sequence_fingerprint),
    bytes_touched: queue_materialize_bytes_touched,
};

static CASE: HarnessCase<GraphCsrSkewedQueuePrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn build_case(ctx: &mut BenchContext) -> Result<GraphCsrSkewedQueuePrepared, BenchError> {
    prepare_skewed_csr_queue_materialize_step(Some(ctx))
}

fn traverse_program(prepared: &GraphCsrSkewedQueuePrepared) -> Option<&Program> {
    Some(&prepared.traverse_program)
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut GraphCsrSkewedQueuePrepared,
) -> Result<BenchRun, BenchError> {
    let workgroup = queue_materialize_workgroup(ctx, prepared, "skewed CSR queue")?;
    let sequence = dispatch_queue_materialize_sequence(ctx, prepared, workgroup, "skewed CSR")?;
    let custom = skewed_csr_queue_metric_points(
        prepared.stats,
        prepared.queue_capacity,
        prepared.high_degree_queue_capacity,
        prepared.traverse_logical_lanes,
        prepared.baseline_wall_ns,
        sequence.wall_ns,
        sequence.resident_used,
        workgroup[0],
        prepared.row_strided_traverse,
        prepared.split_high_degree_traverse,
        GRAPH_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD,
        true,
        QUEUE_RESET_GRID.into_iter().product(),
    );
    let baseline_custom = skewed_csr_baseline_metric_points(prepared.stats);
    Ok(queue_materialize_run(
        prepared,
        sequence,
        custom,
        baseline_custom,
    ))
}

pub(super) fn prepare_skewed_csr_queue_materialize_step(
    ctx: Option<&BenchContext>,
) -> Result<GraphCsrSkewedQueuePrepared, BenchError> {
    let fixture = build_skewed_csr_fixture(CSR_NODE_COUNT)?;
    let queue_capacity = skewed_csr_queue_capacity(fixture.stats.active_sources)?;
    let high_degree_queue_capacity =
        skewed_csr_active_high_degree_sources(&fixture, GRAPH_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD)?;
    let reset_program = frontier_queue_len_init("queue_len");
    let queue_program = frontier_words_to_queue_clear_out_parallel(
        "frontier_in",
        "active_queue",
        "queue_len",
        "frontier_out",
        fixture.stats.node_count,
        queue_capacity,
    );
    let traverse_plan = queue_traverse_plan(
        fixture.stats.max_degree,
        fixture.stats.node_count,
        fixture.stats.edge_count,
        queue_capacity,
        high_degree_queue_capacity,
        CSR_ALLOW_MASK,
        GRAPH_QUEUE_SPLIT_HIGH_DEGREE_THRESHOLD,
    );

    let baseline_start = Instant::now();
    let oracle = skewed_csr_cpu_oracle(&fixture);
    let baseline_wall_ns = elapsed_ns(baseline_start);
    let mut stats = fixture.stats;
    stats.allowed_edges_from_active = oracle.allowed_edges_from_active;
    stats.output_words_set = oracle.output_words_set;

    let inputs = skewed_csr_queue_inputs(&fixture, queue_capacity, high_degree_queue_capacity)?;
    let input_bytes_total = input_bytes_total(&inputs);
    let resident = ctx
        .map(|ctx| ResidentInputSet::upload_optional(ctx, &inputs, "skewed CSR graph queue"))
        .transpose()?
        .flatten();

    Ok(GraphCsrSkewedQueuePrepared {
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

pub(super) fn graph_queue_materialize_sequence_fingerprint(
    prepared: &GraphCsrSkewedQueuePrepared,
) -> [u8; 32] {
    queue_materialize_sequence_fingerprint(
        b"vyre-bench:primitives.graph.csr_skewed_queue_materialize.sequence:v2",
        prepared,
        &[
            prepared.high_degree_queue_capacity,
            u32::from(prepared.split_high_degree_traverse),
        ],
    )
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
