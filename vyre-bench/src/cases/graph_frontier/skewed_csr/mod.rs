//! Packed-bitset CSR frontier expansion over a skewed-degree graph.
//!
//! The payload, the CPU baseline timing, the dispatch and its transfer
//! accounting, and the run assembly are owned by
//! [`crate::cases::frontier_step`]. What is this case's own: the fixture, the
//! traversal program it expands with, and its metric points.

use crate::api::case::{BenchCase, BenchContext, BenchError, BenchRun, WorkloadClass};
use crate::cases::frontier_step::{
    dispatch_frontier_step, frontier_step, frontier_step_bytes_touched, frontier_step_run,
    FrontierStep, StepGrid,
};
use crate::cases::harness::{verify_exact, CaseOps, HarnessCase, WorkloadDescription};
use crate::cases::reference_sample::timed_reference;
use vyre_foundation::ir::Program;
use vyre_libs::graph::program_graph::ProgramGraphShape;

mod metrics;
mod queue_closure;
mod queue_materialize;
mod support;
#[cfg(test)]
mod tests;

use metrics::{skewed_csr_baseline_metric_points, skewed_csr_metric_points};
use support::{
    build_skewed_csr_fixture, skewed_csr_cpu_oracle, skewed_csr_inputs, SkewedCsrStats,
    CSR_ALLOW_MASK, CSR_NODE_COUNT, SUITES,
};

type GraphCsrSkewedPrepared = FrontierStep<SkewedCsrStats>;

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "primitives.graph.csr_skewed_frontier.1m",
    name: "Skewed CSR Bitset Frontier 1M",
    summary: "Packed-bitset CSR frontier expansion over a million-node skewed-degree graph with edge-kind filtering and atomic output bits",
    tags: &[
        "graph",
        "frontier",
        "csr",
        "bitset",
        "skewed-degree",
        "atomic",
        "irregular",
    ],
    workload: WorkloadClass::Macro,
    owner_crate: "vyre-primitives",
    suites: SUITES,
    min_vram_bytes: Some(96 * 1024 * 1024),
    min_input_bytes: Some(CSR_NODE_COUNT as u64 * 20),
    feature_set: &[
        "graph.csr",
        "graph.frontier.bitset",
        "graph.skewed-degree",
    ],
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<GraphCsrSkewedPrepared> = CaseOps {
    build: build_case,
    measure,
    verify: verify_exact,
    program: traverse_program,
    fingerprint: None,
    bytes_touched: frontier_step_bytes_touched,
};

static CASE: HarnessCase<GraphCsrSkewedPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn build_case(ctx: &mut BenchContext) -> Result<GraphCsrSkewedPrepared, BenchError> {
    prepare_skewed_csr_case(Some(ctx))
}

fn traverse_program(prepared: &GraphCsrSkewedPrepared) -> Option<&Program> {
    Some(&prepared.program)
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut GraphCsrSkewedPrepared,
) -> Result<BenchRun, BenchError> {
    let sample = dispatch_frontier_step(
        ctx,
        prepared,
        StepGrid::PerNode(prepared.stats.node_count),
        "skewed CSR graph",
    )?;
    let custom = skewed_csr_metric_points(
        prepared.stats,
        prepared.baseline_wall_ns,
        sample.wall_ns,
        sample.resident_used,
        sample.workgroup_x,
    );
    let baseline_custom = skewed_csr_baseline_metric_points(prepared.stats);
    Ok(frontier_step_run(prepared, sample, custom, baseline_custom))
}

fn prepare_skewed_csr_case(
    ctx: Option<&BenchContext>,
) -> Result<GraphCsrSkewedPrepared, BenchError> {
    let fixture = build_skewed_csr_fixture(CSR_NODE_COUNT)?;
    let shape = ProgramGraphShape::new(fixture.stats.node_count, fixture.stats.edge_count);
    let program = vyre_libs::graph::csr_forward_traverse::csr_forward_traverse(
        shape,
        "frontier_in",
        "frontier_out",
        CSR_ALLOW_MASK,
    );

    let (oracle, baseline_wall_ns) = timed_reference(|| skewed_csr_cpu_oracle(&fixture));
    let mut stats = fixture.stats;
    stats.allowed_edges_from_active = oracle.allowed_edges_from_active;
    stats.output_words_set = oracle.output_words_set;

    frontier_step(
        ctx,
        "skewed CSR graph frontier",
        program,
        skewed_csr_inputs(&fixture),
        stats,
        &oracle.output,
        baseline_wall_ns,
    )
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
