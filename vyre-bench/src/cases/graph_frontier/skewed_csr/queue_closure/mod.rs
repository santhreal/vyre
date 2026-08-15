//! Sparse-delta reachability closure over a skewed CSR graph.
//!
//! The payload, the queue sizing, the reset and delta programs, the resident
//! sample and the run assembly are owned by [`crate::cases::queue_closure`]; the
//! CPU reference by [`crate::cases::queue_closure_oracle`]. What is this case's
//! own: the fixture, the wave bound it holds the closure to, and its metric
//! points.

use crate::api::case::{BenchCase, BenchContext, BenchError, BenchRun, WorkloadClass};
use crate::cases::harness::{verify_exact, CaseOps, HarnessCase, WorkloadDescription};
use crate::cases::queue_closure::{
    dispatch_queue_closure, queue_closure_bytes_touched, queue_closure_prepared, queue_closure_run,
    seed_queue_len, timed_closure_oracle, QueueClosureBuild, QueueClosureLabels,
    QueueClosurePrepared,
};
use vyre_foundation::ir::Program;

use super::support::{
    build_skewed_csr_fixture, skewed_csr_queue_closure_inputs, skewed_csr_queue_closure_oracle,
    SkewedCsrStats, CSR_ALLOW_MASK, CSR_NODE_COUNT, SUITES,
};

mod metrics;

use metrics::{queue_closure_baseline_metric_points, queue_closure_metric_points};

/// Queue waves the closure is allowed before it is called non-convergent.
///
/// A million-node skewed graph closes in far fewer waves than this; the bound
/// exists so a fixture change that breaks convergence fails loudly.
pub(super) const GRAPH_QUEUE_CLOSURE_MAX_ITERS: u32 = 128;

pub(super) type GraphCsrSkewedQueueClosurePrepared = QueueClosurePrepared<SkewedCsrStats>;

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "primitives.graph.csr_skewed_queue_closure.1m",
    name: "Skewed CSR Queue Closure 1M",
    summary: "Sparse-delta reachability closure over a million-node skewed CSR graph using GPU-resident ping-pong active queues",
    tags: &[
        "graph",
        "frontier",
        "csr",
        "frontier-queue",
        "delta-queue",
        "closure",
        "skewed-degree",
        "irregular",
        "resident",
        "release",
    ],
    workload: WorkloadClass::Macro,
    owner_crate: "vyre-primitives",
    suites: SUITES,
    min_vram_bytes: Some(128 * 1024 * 1024),
    min_input_bytes: Some(CSR_NODE_COUNT as u64 * 16),
    feature_set: &[
        "graph.csr",
        "graph.frontier.queue",
        "graph.delta-queue",
        "graph.skewed-degree",
        "resident-repeated-sequence",
    ],
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<GraphCsrSkewedQueueClosurePrepared> = CaseOps {
    build: build_case,
    measure,
    verify: verify_exact,
    program: delta_program,
    fingerprint: None,
    bytes_touched: queue_closure_bytes_touched,
};

static CASE: HarnessCase<GraphCsrSkewedQueueClosurePrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

static LABELS: QueueClosureLabels = QueueClosureLabels {
    label: "skewed CSR queue closure",
    mixed_workgroup_kernels: "reset, clear, and delta",
    resident_support: "resident repeated-sequence",
};

fn build_case(ctx: &mut BenchContext) -> Result<GraphCsrSkewedQueueClosurePrepared, BenchError> {
    prepare_skewed_csr_queue_closure(Some(ctx))
}

fn delta_program(prepared: &GraphCsrSkewedQueueClosurePrepared) -> Option<&Program> {
    Some(&prepared.delta_program)
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut GraphCsrSkewedQueueClosurePrepared,
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

pub(super) fn prepare_skewed_csr_queue_closure(
    ctx: Option<&BenchContext>,
) -> Result<GraphCsrSkewedQueueClosurePrepared, BenchError> {
    let fixture = build_skewed_csr_fixture(CSR_NODE_COUNT)?;
    let seed_queue_len = seed_queue_len(fixture.stats.active_sources, "skewed CSR queue closure")?;
    let (oracle, baseline_wall_ns) = timed_closure_oracle(|| {
        skewed_csr_queue_closure_oracle(
            &fixture,
            GRAPH_QUEUE_CLOSURE_MAX_ITERS,
            fixture.stats.node_count,
        )
    })?;
    let mut stats = fixture.stats;
    stats.output_words_set = oracle.output.iter().filter(|word| **word != 0).count() as u64;

    queue_closure_prepared(
        ctx,
        QueueClosureBuild {
            stats,
            node_count: fixture.stats.node_count,
            edge_count: fixture.stats.edge_count,
            max_degree: fixture.stats.max_degree,
            allow_mask: CSR_ALLOW_MASK,
            seed_queue_len,
            oracle,
            baseline_wall_ns,
            family: "skewed CSR",
            resident_label: "skewed CSR queue closure",
        },
        |queue_capacity| skewed_csr_queue_closure_inputs(&fixture, queue_capacity),
    )
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
