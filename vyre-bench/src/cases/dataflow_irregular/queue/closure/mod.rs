//! Sparse-delta IFDS closure seeded from a pre-materialized queue.
//!
//! The payload, the queue sizing, the reset and delta programs, the resident
//! sample and the run assembly are owned by [`crate::cases::queue_closure`]; the
//! CPU reference by [`crate::cases::queue_closure_oracle`]. What is this case's
//! own: the exploded-supergraph fixture, the cross-check against the full bitset
//! closure, and its metric points.

use crate::api::case::{
    BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, DeterminismClass, WorkloadClass,
};
use crate::api::suite::SuiteKind;
use crate::cases::harness::{verify_exact, CaseOps, HarnessCase, WorkloadDescription};
use crate::cases::queue_closure::{
    dispatch_queue_closure, queue_closure_bytes_touched, queue_closure_prepared, queue_closure_run,
    seed_queue_len, timed_closure_oracle, QueueClosureBuild, QueueClosureLabels,
    QueueClosurePrepared,
};
use vyre_foundation::ir::Program;

use super::super::closure::CLOSURE_MAX_ITERS;
use super::super::fixture::{
    build_ifds_skewed_fixture, ifds_queue_closure_inputs, ifds_skewed_closure_oracle,
    ifds_skewed_queue_closure_oracle, IfdsSkewedStats, NODE_COUNT,
};

mod metrics;

use metrics::{queue_closure_baseline_metric_points, queue_closure_metric_points};

const QUEUE_CLOSURE_SUITES: &[SuiteKind] = &[
    SuiteKind::Smoke,
    SuiteKind::Release,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

pub(in crate::cases::dataflow_irregular) type DataflowIfdsSkewedQueueClosurePrepared =
    QueueClosurePrepared<IfdsSkewedStats>;

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "dataflow.ifds.skewed.queue_closure.1m",
    name: "Dataflow IFDS Skewed Queue Closure 1M",
    summary: "Sparse-delta IFDS closure over a million-node skewed exploded-supergraph using a pre-materialized seed queue and GPU-resident ping-pong active queues",
    tags: &[
        "dataflow",
        "ifds",
        "graph",
        "csr",
        "frontier-queue",
        "delta-queue",
        "seed-queue",
        "closure",
        "skewed-degree",
        "irregular",
        "resident",
        "release",
    ],
    layer: BenchLayer::Libs,
    workload: WorkloadClass::Macro,
    determinism: DeterminismClass::Deterministic,
    owner_crate: "vyre-primitives",
    suites: QUEUE_CLOSURE_SUITES,
    needs_gpu: true,
    needs_network: false,
    min_vram_bytes: Some(128 * 1024 * 1024),
    min_input_bytes: Some(NODE_COUNT as u64 * 20),
    feature_set: &[
        "dataflow",
        "ifds",
        "skewed-csr",
        "frontier-queue",
        "delta-queue",
        "seed-queue",
        "resident-sequence",
    ],
    contract: None,
};

static OPS: CaseOps<DataflowIfdsSkewedQueueClosurePrepared> = CaseOps {
    build: build_case,
    measure,
    verify: verify_exact,
    program: delta_program,
    fingerprint: None,
    bytes_touched: queue_closure_bytes_touched,
};

static CASE: HarnessCase<DataflowIfdsSkewedQueueClosurePrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

static LABELS: QueueClosureLabels = QueueClosureLabels {
    label: "IFDS queue closure",
    mixed_workgroup_kernels: "reset, seed, clear, and delta",
    resident_support: "resident sequence",
};

fn build_case(
    ctx: &mut BenchContext,
) -> Result<DataflowIfdsSkewedQueueClosurePrepared, BenchError> {
    prepare_ifds_skewed_queue_closure(Some(ctx))
}

fn delta_program(prepared: &DataflowIfdsSkewedQueueClosurePrepared) -> Option<&Program> {
    Some(&prepared.delta_program)
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut DataflowIfdsSkewedQueueClosurePrepared,
) -> Result<BenchRun, BenchError> {
    let sequence = dispatch_queue_closure(ctx, prepared, &LABELS)?;
    let custom = queue_closure_metric_points(prepared, sequence.wall_ns, true);
    let baseline_custom = queue_closure_baseline_metric_points(prepared);
    Ok(queue_closure_run(
        prepared,
        sequence,
        custom,
        baseline_custom,
    ))
}

pub(in crate::cases::dataflow_irregular) fn prepare_ifds_skewed_queue_closure(
    ctx: Option<&BenchContext>,
) -> Result<DataflowIfdsSkewedQueueClosurePrepared, BenchError> {
    let fixture = build_ifds_skewed_fixture(NODE_COUNT)?;
    let seed_queue_len = seed_queue_len(fixture.stats.active_sources, "IFDS queue closure")?;
    let (oracle, baseline_wall_ns) = timed_closure_oracle(|| {
        ifds_skewed_queue_closure_oracle(&fixture, CLOSURE_MAX_ITERS, fixture.stats.nodes)
    })?;

    let full_oracle = ifds_skewed_closure_oracle(&fixture, CLOSURE_MAX_ITERS);
    if oracle.output != full_oracle.output {
        return Err(BenchError::CorrectnessViolation(
            "IFDS queue-closure oracle disagreed with full bitset closure oracle".to_string(),
        ));
    }

    let mut stats = fixture.stats;
    stats.output_words_set = oracle.output.iter().filter(|word| **word != 0).count() as u64;

    queue_closure_prepared(
        ctx,
        QueueClosureBuild {
            stats,
            node_count: fixture.stats.nodes,
            edge_count: fixture.stats.edges,
            max_degree: fixture.stats.max_degree,
            allow_mask: super::super::fixture::IFDS_REACH_MASK,
            seed_queue_len,
            oracle,
            baseline_wall_ns,
            family: "IFDS",
            resident_label: "dataflow IFDS queue closure",
        },
        |queue_capacity| ifds_queue_closure_inputs(&fixture, queue_capacity),
    )
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
