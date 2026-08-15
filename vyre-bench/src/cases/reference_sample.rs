//! How a CPU reference sample is timed, accounted, and reported.
//!
//! Every case that reports a GPU sample against a host answer needs the same
//! three things: the host time the reference took, the metrics record that time
//! is published in, and the `BenchRun` that pairs the two samples. All three
//! were hand-rolled per case, and the hand-rolled copies did not agree: some
//! cast `as_nanos()` straight to `u64`, which truncates a reference slower than
//! 18 seconds into a small number instead of a large one, and some published a
//! baseline record with no byte accounting at all.

use crate::api::case::{BenchContext, BenchError, BenchRun};
use crate::api::metric::{elapsed_ns, BenchMetrics};
use crate::api::resident::{
    dispatch_program_timed, input_bytes_total, transfer_accounting, ResidentInputSet,
    TransferAccounting,
};
use std::time::Instant;
use vyre_foundation::ir::Program;

/// One IR program, the host inputs it was built for, and the resident resources
/// those inputs were uploaded into.
///
/// Every case whose measured sample is a single program dispatch scored against
/// a host answer carried these four fields under four spellings, followed by the
/// same upload call and the same `input_bytes_total` sum. Naming the group once
/// means the byte total cannot be summed from a different set of buffers than the
/// ones that were uploaded.
pub(crate) struct HostReferencePayload {
    pub(crate) program: Program,
    pub(crate) inputs: Vec<Vec<u8>>,
    pub(crate) input_bytes_total: u64,
    pub(crate) resident: Option<ResidentInputSet>,
}

impl HostReferencePayload {
    /// Upload the inputs plus a zeroed resource per output binding, in the
    /// program's declaration order.
    ///
    /// Raw-IR resident dispatch requires one resident handle per non-shared
    /// binding, the output bindings included, so an inputs-only upload fails
    /// closed at dispatch with a resource-count mismatch.
    pub(crate) fn program_ordered_resident(
        ctx: &BenchContext,
        program: Program,
        inputs: Vec<Vec<u8>>,
        cleanup_label: &'static str,
    ) -> Result<Self, BenchError> {
        let resident = ResidentInputSet::upload_program_ordered_with_zeroed_outputs_optional(
            ctx,
            &program,
            &inputs,
            cleanup_label,
        )?;
        Ok(Self {
            input_bytes_total: input_bytes_total(&inputs),
            program,
            inputs,
            resident,
        })
    }

    /// Dispatch from host buffers every sample, with nothing left resident.
    pub(crate) fn host_buffers(program: Program, inputs: Vec<Vec<u8>>) -> Self {
        Self {
            input_bytes_total: input_bytes_total(&inputs),
            program,
            inputs,
            resident: None,
        }
    }
}

/// A prepared payload whose measured sample is one program dispatch scored
/// against a host reference.
///
/// The case supplies the dispatch state and the host answer; the measured loop,
/// the transfer accounting and the record assembly are not the case's to write.
pub(crate) trait HostReferenced {
    /// The program, inputs and resident resources this sample dispatches.
    fn dispatch(&self) -> &HostReferencePayload;

    /// The host answer the device sample is scored against.
    fn reference(&self) -> Result<Vec<u8>, BenchError>;

    /// Input bytes the host reference itself read.
    ///
    /// The default is the dispatched input total, which is right whenever the
    /// reference consumes the same buffers the device does. A case whose
    /// reference reads only part of them, such as a keystream cipher that never
    /// reads its round-key table, reports the part it read.
    fn reference_input_bytes(&self) -> u64 {
        self.dispatch().input_bytes_total
    }
}

/// Run one measured sample for a `HostReferenced` payload.
///
/// Usable directly as `CaseOps::measure`: instantiate it at the payload type.
pub(crate) fn measure_against_reference<P: HostReferenced>(
    ctx: &mut BenchContext,
    prepared: &mut P,
) -> Result<BenchRun, BenchError> {
    let dispatch = prepared.dispatch();
    let dispatched = dispatch_program_timed(
        ctx,
        &dispatch.program,
        dispatch.resident.as_ref(),
        &dispatch.inputs,
        &ctx.dispatch_config,
    )?;
    let resident_used = dispatched.resident_used;
    let timed = dispatched.timed;

    let (reference, wall_ns) = timed_reference(|| prepared.reference());
    let input_bytes = prepared.dispatch().input_bytes_total;
    let output_bytes = timed.outputs.iter().map(Vec::len).sum::<usize>() as u64;

    Ok(run_against_reference(
        timed,
        input_bytes,
        transfer_accounting(input_bytes, output_bytes, resident_used),
        ReferenceSample {
            outputs: vec![reference?],
            wall_ns,
            input_bytes: prepared.reference_input_bytes(),
        },
    ))
}

/// The IR program a `HostReferenced` payload dispatches.
///
/// Usable directly as `CaseOps::program`.
pub(crate) fn referenced_program<P: HostReferenced>(prepared: &P) -> Option<&Program> {
    Some(&prepared.dispatch().program)
}

/// Bytes one `HostReferenced` sample reads and writes, from the program's own
/// declared buffer sizes.
///
/// Usable directly as `CaseOps::bytes_touched` by a case whose traffic is exactly
/// what its bindings declare.
pub(crate) fn referenced_bytes_touched<P: HostReferenced>(prepared: &P) -> (u64, u64) {
    crate::api::case::static_program_bytes_touched(&prepared.dispatch().program)
}

/// Run a CPU reference and report how long it took.
///
/// The reference produces both the expected output and the host time the GPU
/// sample is reported against, so one call covers both. The closure is the
/// caller's own: nothing here decides what the reference computes, only how long
/// it is measured to take.
pub(crate) fn timed_reference<T>(reference: impl FnOnce() -> T) -> (T, u64) {
    let started = Instant::now();
    let value = reference();
    (value, elapsed_ns(started))
}

/// Metrics for a CPU reference sample that rewrites host buffers.
///
/// The reference reads its whole input set and writes whatever it produced, and
/// never dispatches, so it carries no device time. A case whose reference does
/// not consume its inputs, or reports its own counters, builds its own record
/// instead of calling this.
pub(crate) fn reference_metrics(wall_ns: u64, input_bytes: u64, output_bytes: u64) -> BenchMetrics {
    BenchMetrics {
        wall_ns: Some(wall_ns),
        input_bytes: Some(input_bytes),
        output_bytes: Some(output_bytes),
        bytes_touched: Some(input_bytes.saturating_add(output_bytes)),
        bytes_read: Some(input_bytes),
        bytes_written: Some(output_bytes),
        ..Default::default()
    }
}

/// One host reference sample: what it produced, how long it took, and how many
/// bytes of input it read.
pub(crate) struct ReferenceSample {
    pub(crate) outputs: Vec<Vec<u8>>,
    pub(crate) wall_ns: u64,
    pub(crate) input_bytes: u64,
}

/// The bench record for one GPU sample scored against one host reference.
///
/// Both samples are published with full byte accounting. Hand-rolled copies
/// disagreed on that: several honest cases emitted a baseline record carrying
/// only `wall_ns`, so the reference appeared to touch no memory and its
/// bandwidth could not be compared against the device sample it is scored
/// against. A case that needs extra metric points pushes them onto the returned
/// record rather than rebuilding it.
pub(crate) fn run_against_reference(
    timed: vyre_driver::TimedDispatchResult,
    input_bytes: u64,
    accounting: TransferAccounting,
    reference: ReferenceSample,
) -> BenchRun {
    let output_bytes = timed.outputs.iter().map(Vec::len).sum::<usize>() as u64;
    let reference_output_bytes = reference.outputs.iter().map(Vec::len).sum::<usize>() as u64;
    BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(timed.wall_ns),
            dispatch_ns: timed.device_ns,
            input_bytes: Some(input_bytes),
            output_bytes: Some(output_bytes),
            bytes_read: Some(accounting.bytes_read),
            bytes_written: Some(accounting.bytes_written),
            bytes_touched: Some(accounting.bytes_touched),
            ..Default::default()
        },
        baseline_metrics: Some(reference_metrics(
            reference.wall_ns,
            reference.input_bytes,
            reference_output_bytes,
        )),
        outputs: timed.outputs,
        baseline_outputs: Some(reference.outputs),
    }
}

#[cfg(test)]
mod tests {
    use super::{reference_metrics, run_against_reference, timed_reference, ReferenceSample};
    use crate::api::resident::transfer_accounting;

    /// The closure's value is returned untouched and its side effects happen
    /// exactly once, so timing a reference cannot change what it computed.
    #[test]
    fn reference_value_passes_through_once() {
        let mut calls = 0;
        let (value, wall_ns) = timed_reference(|| {
            calls += 1;
            vec![7_u8, 9]
        });

        assert_eq!(calls, 1);
        assert_eq!(value, vec![7, 9]);
        assert!(wall_ns < 1_000_000_000);
    }

    /// The reference sample touches its inputs and its outputs and nothing else.
    #[test]
    fn reference_metrics_account_both_directions() {
        let metrics = reference_metrics(77, 1_024, 16);

        assert_eq!(metrics.wall_ns, Some(77));
        assert_eq!(metrics.dispatch_ns, None);
        assert_eq!(metrics.input_bytes, Some(1_024));
        assert_eq!(metrics.output_bytes, Some(16));
        assert_eq!(metrics.bytes_read, Some(1_024));
        assert_eq!(metrics.bytes_written, Some(16));
        assert_eq!(metrics.bytes_touched, Some(1_040));
    }

    /// Byte totals saturate rather than wrapping, so an absurd accounting figure
    /// stays absurd instead of becoming a plausible small one.
    #[test]
    fn byte_totals_saturate() {
        let metrics = reference_metrics(1, u64::MAX, 8);

        assert_eq!(metrics.bytes_touched, Some(u64::MAX));
    }
    /// The device sample and the host sample land in their own halves of one
    /// record, each accounted with its own byte totals. Copies of this assembly
    /// disagreed: some published a baseline record carrying only `wall_ns`.
    #[test]
    fn both_samples_are_accounted_separately() {
        let timed = vyre_driver::TimedDispatchResult {
            outputs: vec![vec![0u8; 8]],
            wall_ns: 500,
            device_ns: Some(300),
            enqueue_ns: None,
            wait_ns: None,
        };
        let run = run_against_reference(
            timed,
            64,
            transfer_accounting(64, 8, false),
            ReferenceSample {
                outputs: vec![vec![0u8; 8]],
                wall_ns: 900,
                input_bytes: 32,
            },
        );

        assert_eq!(run.metrics.wall_ns, Some(500));
        assert_eq!(run.metrics.dispatch_ns, Some(300));
        assert_eq!(run.metrics.input_bytes, Some(64));
        assert_eq!(run.metrics.output_bytes, Some(8));
        assert_eq!(run.metrics.bytes_read, Some(64));
        assert_eq!(run.metrics.bytes_written, Some(8));

        let baseline = run.baseline_metrics.expect("Fix: baseline record required");
        assert_eq!(baseline.wall_ns, Some(900));
        assert_eq!(baseline.input_bytes, Some(32));
        assert_eq!(baseline.output_bytes, Some(8));
        assert_eq!(baseline.bytes_touched, Some(40));
        assert_eq!(baseline.dispatch_ns, None);
        assert_eq!(run.baseline_outputs, Some(vec![vec![0u8; 8]]));
    }

    /// A resident sample reports no host read traffic, and that shows in the
    /// device half only: the reference still read its own inputs from the host.
    #[test]
    fn resident_sample_does_not_zero_the_reference_read_total() {
        let timed = vyre_driver::TimedDispatchResult {
            outputs: vec![vec![1u8; 4]],
            wall_ns: 10,
            device_ns: None,
            enqueue_ns: None,
            wait_ns: None,
        };
        let run = run_against_reference(
            timed,
            1_024,
            transfer_accounting(1_024, 4, true),
            ReferenceSample {
                outputs: vec![vec![1u8; 4]],
                wall_ns: 20,
                input_bytes: 1_024,
            },
        );

        assert_eq!(run.metrics.bytes_read, Some(0));
        assert_eq!(run.metrics.dispatch_ns, None);
        let baseline = run.baseline_metrics.expect("Fix: baseline record required");
        assert_eq!(baseline.bytes_read, Some(1_024));
    }
}
