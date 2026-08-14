//! What a single-dispatch frontier benchmark carries, and how it takes one sample.
//!
//! Three cases measure one propagation step over a skewed CSR graph: the packed
//! bitset expansion, the IFDS step over the exploded supergraph, and the same
//! step driven from a pre-materialized active queue. Their fixtures, programs and
//! metric points are their own. The payload, the CPU baseline timing, the
//! dispatch and its transfer accounting, and the assembled run are one fact each,
//! stated here.

use std::time::Instant;

use crate::api::case::{BenchContext, BenchError, BenchRun};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::resident::{
    dispatch_program_timed, input_bytes_total, transfer_accounting, ResidentInputSet,
    TransferAccounting,
};
use vyre_foundation::ir::Program;

/// One program over one input set, with the CPU answer it is checked against.
///
/// `S` is the family's own stats record; only its metric points read it.
pub(crate) struct FrontierStep<S> {
    pub(crate) program: Program,
    pub(crate) inputs: Vec<Vec<u8>>,
    pub(crate) input_bytes_total: u64,
    pub(crate) baseline_output: Vec<u8>,
    pub(crate) baseline_wall_ns: u64,
    pub(crate) stats: S,
    pub(crate) resident: Option<ResidentInputSet>,
}

/// Run a CPU baseline and report how long it took.
///
/// The baseline oracle produces both the expected output and the host time the
/// GPU sample is reported against, so one call covers both.
pub(crate) fn timed_baseline<T>(baseline: impl FnOnce() -> T) -> (T, u64) {
    let started = Instant::now();
    let value = baseline();
    let wall_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    (value, wall_ns)
}

/// Account the inputs, upload them when the backend keeps buffers resident, and
/// pack the baseline the outputs are compared against.
pub(crate) fn frontier_step<S>(
    ctx: Option<&BenchContext>,
    resident_label: &'static str,
    program: Program,
    inputs: Vec<Vec<u8>>,
    stats: S,
    baseline: &[u32],
    baseline_wall_ns: u64,
) -> Result<FrontierStep<S>, BenchError> {
    let input_bytes_total = input_bytes_total(&inputs);
    let resident = ctx
        .map(|ctx| ResidentInputSet::upload_optional(ctx, &inputs, resident_label))
        .transpose()?
        .flatten();

    Ok(FrontierStep {
        program,
        inputs,
        input_bytes_total,
        baseline_output: vyre_primitives::wire::pack_u32_slice(baseline),
        baseline_wall_ns,
        stats,
        resident,
    })
}

/// How the dispatch grid for one step is decided.
pub(crate) enum StepGrid {
    /// One lane per node, so the grid follows the resolved workgroup width.
    PerNode(u32),
    /// The traversal already chose a grid, and an override must agree with it.
    Selected([u32; 3]),
}

/// One measured dispatch of a frontier step.
pub(crate) struct FrontierStepSample {
    pub(crate) wall_ns: u64,
    pub(crate) device_ns: Option<u64>,
    pub(crate) outputs: Vec<Vec<u8>>,
    pub(crate) output_bytes: u64,
    pub(crate) accounting: TransferAccounting,
    pub(crate) resident_used: bool,
    /// Resolved workgroup width, which the metric points report.
    pub(crate) workgroup_x: u32,
}

/// Dispatch one frontier step and account its host traffic.
///
/// `label` names the case in every error, e.g. `"IFDS active-queue"`.
pub(crate) fn dispatch_frontier_step<S>(
    ctx: &BenchContext,
    step: &FrontierStep<S>,
    grid: StepGrid,
    label: &str,
) -> Result<FrontierStepSample, BenchError> {
    let mut dispatch_config = ctx.dispatch_config.clone();
    let workgroup = dispatch_config
        .workgroup_override
        .unwrap_or_else(|| step.program.workgroup_size());
    if workgroup.contains(&0) {
        return Err(BenchError::ExecutionFailed(format!(
            "{label} benchmark received invalid workgroup {workgroup:?}. Fix: use positive dispatch dimensions."
        )));
    }
    match grid {
        StepGrid::PerNode(node_count) => {
            dispatch_config
                .grid_override
                .get_or_insert([node_count.div_ceil(workgroup[0]), 1, 1]);
        }
        StepGrid::Selected(selected) => match dispatch_config.grid_override {
            Some(grid_override) if grid_override != selected => {
                return Err(BenchError::ExecutionFailed(format!(
                    "{label} traversal selected grid {selected:?}, but received override {grid_override:?}. Fix: run the queue benchmark without a grid override or use the selected traversal grid."
                )));
            }
            Some(_) => {}
            None => dispatch_config.grid_override = Some(selected),
        },
    }

    let dispatch = dispatch_program_timed(
        ctx,
        &step.program,
        step.resident.as_ref(),
        &step.inputs,
        &dispatch_config,
    )?;
    let resident_used = dispatch.resident_used;
    let timed = dispatch.timed;
    let output_bytes = timed.outputs.iter().map(Vec::len).sum::<usize>() as u64;

    Ok(FrontierStepSample {
        wall_ns: timed.wall_ns,
        device_ns: timed.device_ns,
        outputs: timed.outputs,
        output_bytes,
        accounting: transfer_accounting(step.input_bytes_total, output_bytes, resident_used),
        resident_used,
        workgroup_x: workgroup[0],
    })
}

/// Assemble the run a frontier-step case reports.
pub(crate) fn frontier_step_run<S>(
    step: &FrontierStep<S>,
    sample: FrontierStepSample,
    custom: Vec<MetricPoint>,
    baseline_custom: Vec<MetricPoint>,
) -> BenchRun {
    BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(sample.wall_ns),
            dispatch_ns: sample.device_ns,
            input_bytes: Some(step.input_bytes_total),
            output_bytes: Some(sample.output_bytes),
            bytes_read: Some(sample.accounting.bytes_read),
            bytes_written: Some(sample.accounting.bytes_written),
            bytes_touched: Some(sample.accounting.bytes_touched),
            custom,
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(step.baseline_wall_ns),
            input_bytes: Some(step.input_bytes_total),
            output_bytes: Some(step.baseline_output.len() as u64),
            custom: baseline_custom,
            ..Default::default()
        }),
        outputs: sample.outputs,
        baseline_outputs: Some(vec![step.baseline_output.clone()]),
    }
}

/// Bytes a frontier-step sample reads and writes.
pub(crate) fn frontier_step_bytes_touched<S>(step: &FrontierStep<S>) -> (u64, u64) {
    (step.input_bytes_total, step.baseline_output.len() as u64)
}
