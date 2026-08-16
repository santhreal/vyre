//! One IFDS propagation step over a skewed exploded supergraph.
//!
//! The payload, the CPU baseline timing, the dispatch and its transfer
//! accounting, and the run assembly are owned by
//! [`crate::cases::frontier_step`]. What is this case's own: the fixture, the
//! edge-kind filter it propagates under, and its metric points.

use crate::api::case::{BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, WorkloadClass};
use crate::api::suite::SuiteKind;
use crate::cases::frontier_step::{
    dispatch_frontier_step, frontier_step, frontier_step_bytes_touched, frontier_step_run,
    FrontierStep, StepGrid,
};
use crate::cases::harness::{verify_exact, CaseOps, HarnessCase, WorkloadDescription};
use crate::cases::reference_sample::timed_reference;
use vyre_foundation::ir::Program;
use vyre_libs::graph::csr_forward_traverse::csr_forward_traverse;
use vyre_libs::graph::program_graph::ProgramGraphShape;

#[cfg(test)]
mod tests;

mod closure;
mod fixture;
mod metrics;
mod queue;
use fixture::{
    build_ifds_skewed_fixture, ifds_skewed_cpu_oracle, ifds_skewed_inputs, IfdsSkewedStats,
    IFDS_REACH_MASK, NODE_COUNT,
};
#[cfg(test)]
use fixture::{
    ifds_skewed_closure_oracle, ifds_skewed_launch_wave_iterations, FRONTIER_WORDS, UGLY_HUB_DEGREE,
};
use metrics::{ifds_skewed_baseline_metric_points, ifds_skewed_metric_points};

const SUITES: &[SuiteKind] = &[
    SuiteKind::Smoke,
    SuiteKind::Release,
    SuiteKind::Gpu,
    SuiteKind::Deep,
    SuiteKind::Honest,
];

type DataflowIfdsSkewedPrepared = FrontierStep<IfdsSkewedStats>;

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "dataflow.ifds.skewed.step.1m",
    name: "Dataflow IFDS Skewed Step 1M",
    summary: "One IFDS propagation step over a million-node skewed exploded-supergraph CSR with packed frontier bits and filtered edge kinds",
    tags: &[
        "dataflow",
        "ifds",
        "graph",
        "csr",
        "bitset",
        "skewed-degree",
        "irregular",
        "release",
    ],
    layer: BenchLayer::Libs,
    workload: WorkloadClass::Macro,
    owner_crate: "vyre-primitives",
    suites: SUITES,
    min_vram_bytes: Some(96 * 1024 * 1024),
    min_input_bytes: Some(NODE_COUNT as u64 * 20),
    feature_set: &["dataflow", "ifds", "skewed-csr"],
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<DataflowIfdsSkewedPrepared> = CaseOps {
    build: build_case,
    measure,
    verify: verify_exact,
    program: traverse_program,
    fingerprint: None,
    bytes_touched: frontier_step_bytes_touched,
};

static CASE: HarnessCase<DataflowIfdsSkewedPrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn build_case(ctx: &mut BenchContext) -> Result<DataflowIfdsSkewedPrepared, BenchError> {
    prepare_ifds_skewed_step(Some(ctx))
}

fn traverse_program(prepared: &DataflowIfdsSkewedPrepared) -> Option<&Program> {
    Some(&prepared.program)
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut DataflowIfdsSkewedPrepared,
) -> Result<BenchRun, BenchError> {
    let sample = dispatch_frontier_step(
        ctx,
        prepared,
        StepGrid::PerNode(prepared.stats.nodes),
        "IFDS skewed",
    )?;
    let custom = ifds_skewed_metric_points(
        prepared.stats,
        prepared.baseline_wall_ns,
        sample.wall_ns,
        sample.resident_used,
        sample.workgroup_x,
    );
    let baseline_custom = ifds_skewed_baseline_metric_points(prepared.stats);
    Ok(frontier_step_run(prepared, sample, custom, baseline_custom))
}

fn prepare_ifds_skewed_step(
    ctx: Option<&BenchContext>,
) -> Result<DataflowIfdsSkewedPrepared, BenchError> {
    let fixture = build_ifds_skewed_fixture(NODE_COUNT)?;
    let shape = ProgramGraphShape::new(fixture.stats.nodes, fixture.stats.edges);
    let program = csr_forward_traverse(shape, "frontier_in", "frontier_out", IFDS_REACH_MASK);

    let (oracle, baseline_wall_ns) = timed_reference(|| ifds_skewed_cpu_oracle(&fixture));
    let mut stats = fixture.stats;
    stats.allowed_edges_from_active = oracle.allowed_edges_from_active;
    stats.filtered_edges_from_active = oracle.filtered_edges_from_active;
    stats.output_words_set = oracle.output_words_set;

    frontier_step(
        ctx,
        "dataflow IFDS skewed",
        program,
        ifds_skewed_inputs(&fixture),
        stats,
        &oracle.output,
        baseline_wall_ns,
    )
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
