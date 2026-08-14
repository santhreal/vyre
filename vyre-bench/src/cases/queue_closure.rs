//! What a queue-closure benchmark carries, and how it takes one sample.
//!
//! Two cases drive a reachability closure from a pre-materialized seed queue over
//! GPU-resident ping-pong queues: the skewed CSR frontier family and the IFDS
//! dataflow family. Their fixtures, CPU oracles and reported metric points are
//! their own. The payload, the seed-queue sizing, the reset and delta programs,
//! the resident sample and the assembled run are one fact each, stated here.

use std::time::Instant;

use crate::api::case::{BenchContext, BenchError, BenchRun};
use crate::api::metric::{elapsed_ns, BenchMetrics, MetricPoint};
use crate::api::resident::{input_bytes_total, ResidentInputSet};
use crate::cases::queue_closure_oracle::QueueClosureOracle;
use crate::cases::queue_closure_profile::validate_queue_closure_wave_profile;
use crate::cases::queue_materialize::FrontierWords;
use crate::cases::queue_stage::{
    dispatch_resident_queue_closure_sequence, QueueClosureSequenceRun, ResidentQueueClosureSpec,
};
use crate::cases::queue_traverse_plan::should_use_row_strided;
use vyre_foundation::ir::Program;
use vyre_primitives::graph::csr_frontier_queue::frontier_queue_len_init;
use vyre_primitives::graph::csr_queue_delta::{
    csr_queue_delta_enqueue, csr_queue_delta_strided_dispatch_grid,
    csr_queue_delta_strided_enqueue, CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE,
};

/// Workgroup the seed reset and the delta enqueue both launch at.
///
/// The delta kernel is the widest stage in the sequence, and both families give
/// it the same shape, so the reset that seeds its queues matches it.
pub(crate) const QUEUE_CLOSURE_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Everything a queue-closure case builds during prepare.
///
/// `S` is the family's own stats record; only its metric points read it.
pub(crate) struct QueueClosurePrepared<S> {
    pub(crate) reset_program: Program,
    pub(crate) clear_len_program: Program,
    pub(crate) delta_program: Program,
    pub(crate) delta_grid: [u32; 3],
    pub(crate) row_strided_delta: bool,
    pub(crate) inputs: Vec<Vec<u8>>,
    pub(crate) input_bytes_total: u64,
    pub(crate) baseline_output: Vec<u8>,
    pub(crate) baseline_wall_ns: u64,
    pub(crate) stats: S,
    pub(crate) queue_capacity: u32,
    pub(crate) seed_queue_len: u32,
    pub(crate) closure_iterations: u32,
    pub(crate) closure_changed: u32,
    pub(crate) total_queue_pops: u64,
    pub(crate) max_wave_queue_len: u32,
    pub(crate) wave_queue_lengths: Vec<u32>,
    pub(crate) resident: Option<ResidentInputSet>,
}

/// Lanes the delta kernel gives each queued source.
pub(crate) fn delta_lanes_per_source(row_strided_delta: bool) -> u32 {
    if row_strided_delta {
        CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE
    } else {
        1
    }
}

/// Narrow a fixture's active-source count to the seed queue length.
///
/// `label` names the case, e.g. `"IFDS queue closure"`.
pub(crate) fn seed_queue_len(active_sources: u64, label: &str) -> Result<u32, BenchError> {
    u32::try_from(active_sources).map_err(|_| {
        BenchError::EnvironmentInvalid(format!(
            "{label} active source count {active_sources} exceeds u32 indexing. Fix: split the seed queue."
        ))
    })
}

/// Run a CPU closure oracle and report how long it took.
///
/// The oracle is the case's CPU baseline, so the same call that produces the
/// expected output produces the baseline wall time.
pub(crate) fn timed_closure_oracle(
    oracle: impl FnOnce() -> Result<QueueClosureOracle, BenchError>,
) -> Result<(QueueClosureOracle, u64), BenchError> {
    let start = Instant::now();
    let oracle = oracle()?;
    let wall_ns = elapsed_ns(start);
    Ok((oracle, wall_ns))
}

/// Seed the accumulator and queue A from the fixture's packed frontier.
pub(crate) fn queue_closure_reset_program(
    frontier_words: u32,
    seed_queue_len: u32,
    queue_capacity: u32,
) -> Program {
    crate::cases::queue_stage::build_queue_closure_reset_program(
        frontier_words,
        seed_queue_len,
        queue_capacity,
        QUEUE_CLOSURE_WORKGROUP_SIZE,
    )
}

/// The delta enqueue kernel and the grid it launches on.
pub(crate) struct QueueClosureDeltaPlan {
    pub(crate) program: Program,
    pub(crate) grid: [u32; 3],
    pub(crate) row_strided: bool,
}

/// Choose the delta enqueue kernel for a fixture's degree distribution.
///
/// A fixture with long rows gets the strided kernel, which spreads one queued
/// source across a lane team instead of leaving one lane to walk the row.
pub(crate) fn queue_closure_delta_plan(
    max_degree: u32,
    node_count: u32,
    edge_count: u32,
    queue_capacity: u32,
    allow_mask: u32,
) -> QueueClosureDeltaPlan {
    let row_strided = should_use_row_strided(max_degree);
    if row_strided {
        return QueueClosureDeltaPlan {
            program: csr_queue_delta_strided_enqueue(
                "active_queue",
                "active_len",
                "edge_offsets",
                "edge_targets",
                "edge_kind_mask",
                "accumulator",
                "next_queue",
                "next_len",
                node_count,
                edge_count,
                queue_capacity,
                queue_capacity,
                allow_mask,
            ),
            grid: csr_queue_delta_strided_dispatch_grid(queue_capacity),
            row_strided,
        };
    }

    QueueClosureDeltaPlan {
        program: csr_queue_delta_enqueue(
            "active_queue",
            "active_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "accumulator",
            "next_queue",
            "next_len",
            node_count,
            edge_count,
            queue_capacity,
            queue_capacity,
            allow_mask,
        ),
        grid: [
            queue_capacity
                .div_ceil(QUEUE_CLOSURE_WORKGROUP_SIZE[0])
                .max(1),
            1,
            1,
        ],
        row_strided,
    }
}

/// What a case's fixture and oracle contribute to the shared payload.
pub(crate) struct QueueClosureBuild<S> {
    /// Fixture stats, with `output_words_set` already taken from the oracle.
    pub(crate) stats: S,
    pub(crate) node_count: u32,
    pub(crate) edge_count: u32,
    pub(crate) max_degree: u32,
    pub(crate) allow_mask: u32,
    pub(crate) seed_queue_len: u32,
    pub(crate) oracle: QueueClosureOracle,
    pub(crate) baseline_wall_ns: u64,
    /// Family name the wave-profile validator reports under, e.g. `"IFDS"`.
    pub(crate) family: &'static str,
    /// Label the resident upload reports under.
    pub(crate) resident_label: &'static str,
}

/// Size the queues, build the programs, and upload the inputs.
///
/// `inputs` is called with the queue capacity the oracle's widest wave justifies,
/// because both ping-pong queues and the seed queue are allocated to it.
pub(crate) fn queue_closure_prepared<S: FrontierWords>(
    ctx: Option<&BenchContext>,
    build: QueueClosureBuild<S>,
    inputs: impl FnOnce(u32) -> Result<Vec<Vec<u8>>, BenchError>,
) -> Result<QueueClosurePrepared<S>, BenchError> {
    let QueueClosureBuild {
        stats,
        node_count,
        edge_count,
        max_degree,
        allow_mask,
        seed_queue_len,
        oracle,
        baseline_wall_ns,
        family,
        resident_label,
    } = build;

    let queue_capacity = oracle.max_wave_queue_len.max(seed_queue_len).max(1);
    validate_queue_closure_wave_profile(
        family,
        &oracle.wave_queue_lengths,
        oracle.iterations,
        oracle.total_queue_pops,
        oracle.max_wave_queue_len,
        queue_capacity,
    )?;
    let reset_program =
        queue_closure_reset_program(stats.frontier_words(), seed_queue_len, queue_capacity);
    let clear_len_program = frontier_queue_len_init("queue_len");
    let delta = queue_closure_delta_plan(
        max_degree,
        node_count,
        edge_count,
        queue_capacity,
        allow_mask,
    );

    let inputs = inputs(queue_capacity)?;
    let input_bytes_total = input_bytes_total(&inputs);
    let resident = ctx
        .map(|ctx| ResidentInputSet::upload_optional(ctx, &inputs, resident_label))
        .transpose()?
        .flatten();

    Ok(QueueClosurePrepared {
        reset_program,
        clear_len_program,
        delta_program: delta.program,
        delta_grid: delta.grid,
        row_strided_delta: delta.row_strided,
        inputs,
        input_bytes_total,
        baseline_output: vyre_primitives::wire::pack_u32_slice(&oracle.output),
        baseline_wall_ns,
        stats,
        queue_capacity,
        seed_queue_len,
        closure_iterations: oracle.iterations,
        closure_changed: oracle.changed,
        total_queue_pops: oracle.total_queue_pops,
        max_wave_queue_len: oracle.max_wave_queue_len,
        wave_queue_lengths: oracle.wave_queue_lengths,
        resident,
    })
}

/// How a case names itself when the closure cannot run.
pub(crate) struct QueueClosureLabels {
    /// Case name leading every error, e.g. `"IFDS queue closure"`. Also the
    /// context the resident sequence reports its own failures under.
    pub(crate) label: &'static str,
    /// Kernels a workgroup override would have to satisfy at once, e.g.
    /// `"reset, seed, clear, and delta"`.
    pub(crate) mixed_workgroup_kernels: &'static str,
    /// Backend capability the resident path needs, e.g. `"resident sequence"`.
    pub(crate) resident_support: &'static str,
}

/// Take one resident sample of the closure.
///
/// The closure exists only as a resident repeated sequence: a host path would
/// copy both ping-pong queues across the bus twice per wave and measure the bus
/// instead of the closure.
pub(crate) fn dispatch_queue_closure<S: FrontierWords>(
    ctx: &BenchContext,
    prepared: &QueueClosurePrepared<S>,
    labels: &QueueClosureLabels,
) -> Result<QueueClosureSequenceRun, BenchError> {
    let &QueueClosureLabels {
        label,
        mixed_workgroup_kernels,
        resident_support,
    } = labels;
    if ctx.dispatch_config.workgroup_override.is_some() {
        return Err(BenchError::ExecutionFailed(format!(
            "{label} uses mixed workgroups across {mixed_workgroup_kernels} kernels. Fix: run without a workgroup override."
        )));
    }
    let resident = prepared.resident.as_ref().ok_or_else(|| {
        BenchError::EnvironmentInvalid(format!(
            "{label} requires resident GPU buffers. Fix: run on a backend with {resident_support} support."
        ))
    })?;

    dispatch_resident_queue_closure_sequence(
        ctx,
        ResidentQueueClosureSpec {
            reset_program: &prepared.reset_program,
            clear_len_program: &prepared.clear_len_program,
            delta_program: &prepared.delta_program,
            frontier_words: prepared.stats.frontier_words(),
            seed_queue_len: prepared.seed_queue_len,
            baseline_output_len: prepared.baseline_output.len(),
            closure_iterations: prepared.closure_iterations,
            delta_grid: prepared.delta_grid,
            workgroup: QUEUE_CLOSURE_WORKGROUP_SIZE,
            context: label,
        },
        resident,
    )
}

/// Assemble the run a queue-closure case reports.
///
/// Every wave stays on the device, so the only host transfer is the final
/// accumulator readback and nothing is charged to `bytes_read`.
pub(crate) fn queue_closure_run<S>(
    prepared: &QueueClosurePrepared<S>,
    sequence: QueueClosureSequenceRun,
    custom: Vec<MetricPoint>,
    baseline_custom: Vec<MetricPoint>,
) -> BenchRun {
    let output_bytes = sequence.outputs.iter().map(Vec::len).sum::<usize>() as u64;
    BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(sequence.wall_ns),
            dispatch_ns: None,
            input_bytes: Some(prepared.input_bytes_total),
            output_bytes: Some(output_bytes),
            bytes_read: Some(0),
            bytes_written: Some(output_bytes),
            bytes_touched: Some(output_bytes),
            custom,
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(prepared.baseline_wall_ns),
            input_bytes: Some(prepared.input_bytes_total),
            output_bytes: Some(prepared.baseline_output.len() as u64),
            custom: baseline_custom,
            ..Default::default()
        }),
        outputs: sequence.outputs,
        baseline_outputs: Some(vec![prepared.baseline_output.clone()]),
    }
}

/// Bytes a queue-closure sample reads and writes.
pub(crate) fn queue_closure_bytes_touched<S>(prepared: &QueueClosurePrepared<S>) -> (u64, u64) {
    (
        prepared.input_bytes_total,
        prepared.baseline_output.len() as u64,
    )
}
