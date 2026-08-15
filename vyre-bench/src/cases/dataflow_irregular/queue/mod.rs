//! One IFDS propagation step driven from a pre-materialized active queue.
//!
//! The payload, the CPU baseline timing, the dispatch and its transfer
//! accounting, and the run assembly are owned by
//! [`crate::cases::frontier_step`]; the traversal choice by
//! [`crate::cases::queue_traverse_plan`]. What is this case's own: the fixture,
//! the queue sizing, and its metric points.

use crate::api::case::{BenchCase, BenchContext, BenchError, BenchLayer, BenchRun, WorkloadClass};
use crate::cases::frontier_step::{
    dispatch_frontier_step, frontier_step, frontier_step_run, FrontierStep, StepGrid,
};
use crate::cases::harness::{verify_exact, CaseOps, HarnessCase, WorkloadDescription};
use crate::cases::queue_traverse_plan::{single_queue_traverse, traverse_logical_lanes};
use crate::cases::reference_sample::timed_reference;
use crate::cases::skewed_graph::sparse_queue_capacity;
use vyre_foundation::ir::Program;

use super::fixture::{
    build_ifds_skewed_fixture, ifds_active_queue_inputs, ifds_skewed_cpu_oracle, IfdsSkewedStats,
    IFDS_REACH_MASK, NODE_COUNT,
};
use super::metrics::{ifds_queue_baseline_metric_points, ifds_queue_metric_points};
use super::SUITES;

mod closure;
mod materialize;
#[cfg(test)]
pub(super) use closure::prepare_ifds_skewed_queue_closure;
#[cfg(test)]
pub(super) use materialize::{
    ifds_queue_materialize_sequence_fingerprint, prepare_ifds_skewed_queue_materialize_step,
};

pub(super) const ACTIVE_QUEUE_ACTIVE_QUEUE_INDEX: usize = 0;
pub(super) const ACTIVE_QUEUE_LEN_INDEX: usize = 1;
pub(super) const ACTIVE_QUEUE_EDGE_OFFSETS_INDEX: usize = 2;
pub(super) const ACTIVE_QUEUE_EDGE_TARGETS_INDEX: usize = 3;
pub(super) const ACTIVE_QUEUE_EDGE_KIND_INDEX: usize = 4;
pub(super) const ACTIVE_QUEUE_FRONTIER_OUT_INDEX: usize = 5;

pub(super) struct DataflowIfdsSkewedActiveQueuePrepared {
    pub(super) step: FrontierStep<IfdsSkewedStats>,
    pub(super) traverse_grid: [u32; 3],
    pub(super) row_strided_traverse: bool,
    pub(super) traverse_logical_lanes: u64,
    pub(super) queue_capacity: u32,
}

pub(super) fn ifds_sparse_queue_capacity(active_sources: u64) -> Result<u32, BenchError> {
    sparse_queue_capacity(
        active_sources,
        "IFDS queue benchmark requires at least one active source. Fix: seed the frontier before queue sizing.",
        "IFDS queue",
    )
}

static WORKLOAD: WorkloadDescription = WorkloadDescription {
    id: "dataflow.ifds.skewed.queue_step.1m",
    name: "Dataflow IFDS Skewed Active Queue Step 1M",
    summary: "One IFDS propagation step over a million-node skewed exploded-supergraph CSR from a pre-materialized GPU-resident active frontier queue",
    tags: &[
        "dataflow",
        "ifds",
        "graph",
        "csr",
        "frontier-queue",
        "active-queue",
        "skewed-degree",
        "irregular",
        "resident",
        "release",
    ],
    layer: BenchLayer::Libs,
    workload: WorkloadClass::Macro,
    owner_crate: "vyre-primitives",
    suites: SUITES,
    min_vram_bytes: Some(96 * 1024 * 1024),
    min_input_bytes: Some(NODE_COUNT as u64 * 12),
    feature_set: &[
        "dataflow",
        "ifds",
        "skewed-csr",
        "frontier-queue",
        "active-queue",
    ],
    ..WorkloadDescription::BASE
};

static OPS: CaseOps<DataflowIfdsSkewedActiveQueuePrepared> = CaseOps {
    build: build_case,
    measure,
    verify: verify_exact,
    program: traverse_program,
    fingerprint: None,
    bytes_touched,
};

static CASE: HarnessCase<DataflowIfdsSkewedActiveQueuePrepared> = HarnessCase {
    workload: &WORKLOAD,
    ops: &OPS,
};

fn build_case(ctx: &mut BenchContext) -> Result<DataflowIfdsSkewedActiveQueuePrepared, BenchError> {
    prepare_ifds_skewed_active_queue_step(Some(ctx))
}

fn traverse_program(prepared: &DataflowIfdsSkewedActiveQueuePrepared) -> Option<&Program> {
    Some(&prepared.step.program)
}

fn bytes_touched(prepared: &DataflowIfdsSkewedActiveQueuePrepared) -> (u64, u64) {
    crate::cases::frontier_step::frontier_step_bytes_touched(&prepared.step)
}

fn measure(
    ctx: &mut BenchContext,
    prepared: &mut DataflowIfdsSkewedActiveQueuePrepared,
) -> Result<BenchRun, BenchError> {
    let sample = dispatch_frontier_step(
        ctx,
        &prepared.step,
        StepGrid::Selected(prepared.traverse_grid),
        "IFDS active-queue",
    )?;
    let custom = ifds_queue_metric_points(
        prepared.step.stats,
        prepared.queue_capacity,
        0,
        prepared.traverse_logical_lanes,
        prepared.step.baseline_wall_ns,
        sample.wall_ns,
        sample.resident_used,
        sample.workgroup_x,
        false,
        prepared.row_strided_traverse,
        false,
        0,
        false,
        0,
    );
    let baseline_custom =
        ifds_queue_baseline_metric_points(prepared.step.stats, prepared.queue_capacity);
    Ok(frontier_step_run(
        &prepared.step,
        sample,
        custom,
        baseline_custom,
    ))
}

pub(super) fn prepare_ifds_skewed_active_queue_step(
    ctx: Option<&BenchContext>,
) -> Result<DataflowIfdsSkewedActiveQueuePrepared, BenchError> {
    let fixture = build_ifds_skewed_fixture(NODE_COUNT)?;
    let queue_capacity = ifds_sparse_queue_capacity(fixture.stats.active_sources)?;
    let traverse_plan = single_queue_traverse(
        fixture.stats.max_degree,
        fixture.stats.nodes,
        fixture.stats.edges,
        queue_capacity,
        IFDS_REACH_MASK,
    );

    let (oracle, baseline_wall_ns) = timed_reference(|| ifds_skewed_cpu_oracle(&fixture));
    let mut stats = fixture.stats;
    stats.allowed_edges_from_active = oracle.allowed_edges_from_active;
    stats.filtered_edges_from_active = oracle.filtered_edges_from_active;
    stats.output_words_set = oracle.output_words_set;

    Ok(DataflowIfdsSkewedActiveQueuePrepared {
        step: frontier_step(
            ctx,
            "dataflow IFDS active queue",
            traverse_plan.program,
            ifds_active_queue_inputs(&fixture, queue_capacity)?,
            stats,
            &oracle.output,
            baseline_wall_ns,
        )?,
        traverse_grid: traverse_plan.grid,
        row_strided_traverse: traverse_plan.row_strided,
        traverse_logical_lanes: traverse_logical_lanes(queue_capacity, traverse_plan.row_strided),
        queue_capacity,
    })
}

inventory::submit! {
    &CASE as &'static dyn BenchCase
}
