//! Shared owner for the single-pass three-stream elementwise family.
//!
//! `runtime.adaptive_routing.gpu_resident.1m` and
//! `compound.pipeline.fused_filter.1m` are the same measured shape: upload
//! three generated `u32` streams of equal length, run one resident program that
//! writes one `u32` per lane, and compare against a Rayon oracle applying the
//! same per-lane function. Only the IR program, the stream generator, the
//! per-lane function and the reported metric points differ.

use crate::api::case::{BenchContext, BenchError, BenchRun};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::resident::{
    dispatch_program_timed, input_bytes_total, transfer_accounting, ResidentInputSet,
};
use rayon::prelude::*;
use vyre_foundation::ir::Program;

use super::byte_pack::u32_bytes;

/// What one three-stream elementwise case contributes on top of the shared
/// measured loop.
pub(crate) struct TripletSpec {
    /// The resident program under measurement.
    pub(crate) program: Program,
    /// The three equal-length input streams, in binding order.
    pub(crate) streams: (Vec<u32>, Vec<u32>, Vec<u32>),
    /// The per-lane function the CPU oracle applies.
    pub(crate) lane: fn(u32, u32, u32) -> u32,
    /// Names of the three streams, used in the length-mismatch message.
    pub(crate) stream_names: [&'static str; 3],
    /// Subject naming this case in errors and in the resident upload label.
    pub(crate) subject: &'static str,
    /// Metric points derived from the baseline words, computed once at prepare.
    pub(crate) metrics: fn(&[u32]) -> Vec<MetricPoint>,
}

/// The uploaded state and captured baseline one three-stream case needs.
pub(crate) struct TripletPrepared {
    pub(crate) program: Program,
    pub(crate) inputs: Vec<Vec<u8>>,
    pub(crate) input_bytes_total: u64,
    pub(crate) baseline_output: Vec<u8>,
    pub(crate) baseline_wall_ns: u64,
    pub(crate) resident: Option<ResidentInputSet>,
    /// Case metrics that do not depend on the measured dispatch.
    pub(crate) static_metrics: Vec<MetricPoint>,
}

/// Reject unequal stream lengths before any oracle consumes them.
fn check_stream_lengths(
    first: &[u32],
    second: &[u32],
    third: &[u32],
    stream_names: [&'static str; 3],
    subject: &'static str,
) -> Result<(), BenchError> {
    if first.len() == second.len() && first.len() == third.len() {
        return Ok(());
    }
    let [first_name, second_name, third_name] = stream_names;
    Err(BenchError::ExecutionFailed(format!(
        "{subject} input length mismatch: {first_name}={}, {second_name}={}, {third_name}={}. Fix: generate equal-length streams before building the CPU oracle.",
        first.len(),
        second.len(),
        third.len()
    )))
}

/// Upload the three streams, capture the CPU baseline and record its wall time.
///
/// The streams are length-checked before the oracle runs: a Rayon `zip` over
/// unequal lengths silently truncates, which would make a short baseline
/// compare equal against a short backend read.
pub(crate) fn prepare_triplet(
    ctx: &mut BenchContext,
    spec: TripletSpec,
) -> Result<TripletPrepared, BenchError> {
    let TripletSpec {
        program,
        streams: (first, second, third),
        lane,
        stream_names,
        subject,
        metrics,
    } = spec;

    check_stream_lengths(&first, &second, &third, stream_names, subject)?;

    let inputs = vec![u32_bytes(&first), u32_bytes(&second), u32_bytes(&third)];
    let input_bytes_total = input_bytes_total(&inputs);

    let baseline_start = std::time::Instant::now();
    let baseline_words: Vec<u32> = first
        .par_iter()
        .zip(second.par_iter())
        .zip(third.par_iter())
        .map(|((&a, &b), &c)| lane(a, b, c))
        .collect();
    let baseline_wall_ns = u64::try_from(baseline_start.elapsed().as_nanos()).unwrap_or(u64::MAX);

    let static_metrics = metrics(&baseline_words);
    let baseline_output = u32_bytes(&baseline_words);
    let resident = ResidentInputSet::upload_with_zeroed_outputs_optional(
        ctx,
        &inputs,
        &[baseline_output.len()],
        subject,
    )?;

    Ok(TripletPrepared {
        program,
        inputs,
        input_bytes_total,
        baseline_output,
        baseline_wall_ns,
        resident,
        static_metrics,
    })
}

/// Run one measured dispatch and assemble the sample against the baseline.
pub(crate) fn triplet_measure(
    ctx: &mut BenchContext,
    prepared: &mut TripletPrepared,
) -> Result<BenchRun, BenchError> {
    let dispatch = dispatch_program_timed(
        ctx,
        &prepared.program,
        prepared.resident.as_ref(),
        &prepared.inputs,
        &ctx.dispatch_config,
    )?;
    let timed = dispatch.timed;
    let outputs = timed.outputs;
    let output_bytes = outputs.iter().map(Vec::len).sum::<usize>() as u64;
    let accounting = transfer_accounting(
        prepared.input_bytes_total,
        output_bytes,
        dispatch.resident_used,
    );

    let mut custom = prepared.static_metrics.clone();
    custom.push(MetricPoint {
        name: "resident_buffers".to_string(),
        value: u64::from(dispatch.resident_used),
    });

    Ok(BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(timed.wall_ns),
            dispatch_ns: timed.device_ns,
            input_bytes: Some(prepared.input_bytes_total),
            output_bytes: Some(output_bytes),
            bytes_read: Some(accounting.bytes_read),
            bytes_written: Some(accounting.bytes_written),
            bytes_touched: Some(accounting.bytes_touched),
            custom,
            ..Default::default()
        },
        baseline_metrics: Some(BenchMetrics {
            wall_ns: Some(prepared.baseline_wall_ns),
            input_bytes: Some(prepared.input_bytes_total),
            output_bytes: Some(prepared.baseline_output.len() as u64),
            bytes_touched: Some(
                prepared
                    .input_bytes_total
                    .saturating_add(prepared.baseline_output.len() as u64),
            ),
            ..Default::default()
        }),
        outputs,
        baseline_outputs: Some(vec![prepared.baseline_output.clone()]),
    })
}

/// The IR program the runner may recompile for a three-stream case.
pub(crate) fn triplet_program(prepared: &TripletPrepared) -> Option<&Program> {
    Some(&prepared.program)
}

/// One `u32` in per stream per lane, one `u32` out per lane.
pub(crate) fn triplet_bytes_touched(prepared: &TripletPrepared) -> (u64, u64) {
    let lanes = prepared.input_bytes_total / 12;
    (prepared.input_bytes_total, lanes * 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(streams: (Vec<u32>, Vec<u32>, Vec<u32>)) -> TripletSpec {
        TripletSpec {
            program: Program::wrapped(vec![], [256, 1, 1], vec![]),
            streams,
            lane: |a, b, c| a ^ b ^ c,
            stream_names: ["alpha", "beta", "gamma"],
            subject: "triplet test",
            metrics: |words| vec![MetricPoint {
                name: "lanes".to_string(),
                value: words.len() as u64,
            }],
        }
    }

    /// Unequal stream lengths must be rejected before the oracle runs. A Rayon
    /// `zip` truncates to the shortest stream, which would silently shrink both
    /// the baseline and the comparison it is measured against.
    #[test]
    fn unequal_stream_lengths_are_rejected_before_the_oracle_runs() {
        let error = check_stream_lengths(
            &[1, 2, 3],
            &[4, 5],
            &[6, 7, 8],
            ["alpha", "beta", "gamma"],
            "triplet test",
        )
        .expect_err("mismatched streams must never truncate");

        assert!(error.to_string().contains("input length mismatch"), "{error}");
        assert!(error.to_string().contains("beta=2"), "{error}");
    }

    /// Equal-length streams pass the check.
    #[test]
    fn equal_stream_lengths_are_accepted() {
        check_stream_lengths(
            &[1, 2, 3],
            &[4, 5, 6],
            &[7, 8, 9],
            ["alpha", "beta", "gamma"],
            "triplet test",
        )
        .expect("equal-length streams must be accepted");
    }

    /// Byte accounting reports one output word per lane.
    #[test]
    fn bytes_touched_reports_one_output_word_per_lane() {
        let prepared = TripletPrepared {
            program: Program::wrapped(vec![], [256, 1, 1], vec![]),
            inputs: vec![],
            input_bytes_total: 12 * 1_024,
            baseline_output: vec![],
            baseline_wall_ns: 0,
            resident: None,
            static_metrics: vec![],
        };

        assert_eq!(triplet_bytes_touched(&prepared), (12 * 1_024, 4 * 1_024));
    }

    /// The spec is consumed whole, so a case cannot forget one of its parts.
    #[test]
    fn spec_carries_every_case_specific_part() {
        let spec = spec((vec![1], vec![2], vec![3]));

        assert_eq!((spec.lane)(1, 2, 3), 0);
        assert_eq!((spec.metrics)(&[0, 0])[0].value, 2);
        assert_eq!(spec.stream_names, ["alpha", "beta", "gamma"]);
    }
}
