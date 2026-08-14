//! How a scan_ac_irregular case takes one sample.
//!
//! The literal scan and the count-only preflight are two different kernels over
//! the same haystack, and they take a sample in exactly the same way: validate
//! the workgroup, derive the grid from the haystack length, run the resident
//! reset-then-scan sequence when residency is available and a single timed
//! dispatch otherwise, then account host transfer. Only the resource bindings,
//! the readback ranges, and the metric points differ, so those are what a case
//! supplies.

use std::time::Instant;

use vyre_driver::{ResidentDispatchStep, ResidentReadRange};
use vyre_foundation::ir::Program;

use crate::api::case::{BenchContext, BenchError, BenchRun};
use crate::api::metric::{elapsed_ns, BenchMetrics, MetricPoint};
use crate::api::resident::{dispatch_program_timed, transfer_accounting, ResidentInputSet};

use super::metrics::{scan_ac_baseline_metric_points, ScanAcStats};

/// One dispatch of a scan program, whichever path produced it.
pub(super) struct ScanSample {
    pub outputs: Vec<Vec<u8>>,
    pub wall_ns: u64,
    pub dispatch_ns: Option<u64>,
    pub resident_used: bool,
    pub device_reset_sequence: bool,
    /// The x extent the grid was sized against, reported as a metric.
    pub workgroup_x: u32,
}

/// The reset-then-scan resident sequence a scan case runs when residency is up.
pub(super) struct ResetThenScan<'a> {
    pub reset_program: &'a Program,
    pub scan_program: &'a Program,
    pub reset_indices: &'a [usize],
    pub scan_indices: &'a [usize],
    /// Case name in every error raised here, e.g. `"irregular AC scan"`.
    pub label: &'a str,
    /// Case noun inside the fix hint, e.g. `"scan"` or `"count"`.
    pub kind: &'a str,
    /// Context passed to the scan-resource lookup, which predates `label` and is
    /// not derivable from it.
    pub scan_resources_context: &'a str,
    pub haystack_bytes: u32,
}

/// Take one sample of `program`.
///
/// `resident_sequence` is `Some` only when the case has resident resources; it
/// receives the program's own workgroup, because a resident sequence sizes its
/// grid against the program rather than against a caller override.
pub(super) fn take_scan_sample<F>(
    ctx: &BenchContext,
    label: &str,
    program: &Program,
    inputs: &[Vec<u8>],
    haystack_bytes: u32,
    resident_sequence: Option<F>,
) -> Result<ScanSample, BenchError>
where
    F: FnOnce([u32; 3]) -> Result<(Vec<Vec<u8>>, u64), BenchError>,
{
    let mut dispatch_config = ctx.dispatch_config.clone();
    let program_workgroup = program.workgroup_size();
    let workgroup = dispatch_config
        .workgroup_override
        .unwrap_or(program_workgroup);
    if workgroup.contains(&0) {
        return Err(BenchError::ExecutionFailed(format!(
            "{label} received invalid workgroup {workgroup:?}. Fix: use positive dispatch dimensions."
        )));
    }
    dispatch_config.grid_override.get_or_insert([
        haystack_bytes.div_ceil(workgroup[0]).max(1),
        1,
        1,
    ]);

    if let Some(sequence) = resident_sequence {
        let (outputs, wall_ns) = sequence(program_workgroup)?;
        return Ok(ScanSample {
            outputs,
            wall_ns,
            dispatch_ns: None,
            resident_used: true,
            device_reset_sequence: true,
            workgroup_x: workgroup[0],
        });
    }

    let dispatch = dispatch_program_timed(ctx, program, None, inputs, &dispatch_config)?;
    let timed = dispatch.timed;
    Ok(ScanSample {
        outputs: timed.outputs,
        wall_ns: timed.wall_ns,
        dispatch_ns: timed.device_ns,
        resident_used: dispatch.resident_used,
        device_reset_sequence: false,
        workgroup_x: workgroup[0],
    })
}

/// Run the reset step and then the scan step as one resident sequence, reading
/// back `readback` as `(index into the scan resources, byte length)` pairs.
pub(super) fn dispatch_reset_then_scan(
    ctx: &BenchContext,
    resident: &ResidentInputSet,
    workgroup: [u32; 3],
    sequence: ResetThenScan<'_>,
    readback: &[(usize, usize)],
) -> Result<(Vec<Vec<u8>>, u64), BenchError> {
    let ResetThenScan {
        reset_program,
        scan_program,
        reset_indices,
        scan_indices,
        label,
        kind,
        scan_resources_context,
        haystack_bytes,
    } = sequence;

    if let Some(override_workgroup) = ctx.dispatch_config.workgroup_override {
        if override_workgroup != workgroup {
            return Err(BenchError::ExecutionFailed(format!(
                "{label} resident sequence uses program workgroup {workgroup:?}, but received override {override_workgroup:?}. Fix: run the resident {kind} sequence without a workgroup override or rebuild the resident sequence program."
            )));
        }
    }

    let reset_resources =
        resident.resources_for_indices(reset_indices, &format!("{label} reset sequence"))?;
    let scan_resources = resident.resources_for_indices(scan_indices, scan_resources_context)?;
    let steps = [
        ResidentDispatchStep {
            program: reset_program,
            resources: &reset_resources,
            grid_override: Some([1, 1, 1]),
            workgroup_override: None,
        },
        ResidentDispatchStep {
            program: scan_program,
            resources: &scan_resources,
            grid_override: Some([haystack_bytes.div_ceil(workgroup[0]).max(1), 1, 1]),
            workgroup_override: None,
        },
    ];

    let read_ranges: Vec<ResidentReadRange<'_>> = readback
        .iter()
        .map(|&(resource_index, byte_len)| ResidentReadRange {
            resource: &scan_resources[resource_index],
            byte_offset: 0,
            byte_len,
        })
        .collect();
    let mut outputs: Vec<Vec<u8>> = readback
        .iter()
        .map(|&(_, byte_len)| Vec::with_capacity(byte_len))
        .collect();

    let started = Instant::now();
    {
        let mut targets: Vec<&mut Vec<u8>> = outputs.iter_mut().collect();
        ctx.dispatch_resident_sequence_read_ranges_into(&steps, &read_ranges, &mut targets)
            .map_err(|error| BenchError::BackendFailed(error.to_string()))?;
    }
    let wall_ns = elapsed_ns(started);

    Ok((outputs, wall_ns))
}

/// Assemble the `BenchRun` a scan case reports.
///
/// Both cases publish the same metric skeleton around their own `custom` points:
/// the sample's timing and byte accounting, and a baseline sourced from the CPU
/// automaton the case already ran during `prepare`.
pub(super) fn scan_bench_run(
    sample: ScanSample,
    input_bytes_total: u64,
    baseline_wall_ns: u64,
    stats: ScanAcStats,
    custom: Vec<MetricPoint>,
    baseline_outputs: Vec<Vec<u8>>,
) -> BenchRun {
    let output_bytes = sample.outputs.iter().map(Vec::len).sum::<usize>() as u64;
    let accounting = transfer_accounting(input_bytes_total, output_bytes, sample.resident_used);
    BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(sample.wall_ns),
            dispatch_ns: sample.dispatch_ns,
            input_bytes: Some(input_bytes_total),
            output_bytes: Some(output_bytes),
            bytes_read: Some(accounting.bytes_read),
            bytes_written: Some(accounting.bytes_written),
            bytes_touched: Some(accounting.bytes_touched),
            custom,
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(baseline_wall_ns),
            input_bytes: Some(input_bytes_total),
            output_bytes: Some(baseline_outputs.iter().map(Vec::len).sum::<usize>() as u64),
            custom: scan_ac_baseline_metric_points(stats),
            ..Default::default()
        }),
        outputs: sample.outputs,
        baseline_outputs: Some(baseline_outputs),
    }
}
