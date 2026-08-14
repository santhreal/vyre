//! The resident work-queue buffer set and reference reporting the megakernel
//! cases share.
//!
//! All three megakernel cases publish a ring into the same four-buffer resident
//! work queue: control, ring, debug log, io queue. Each carried its own copy of
//! that encoding, its own post-dispatch accounting, and its own reference-sample
//! metrics, so a change to the buffer layout had to be made three times.
//!
//! What is deliberately not here is the timed dispatch. The latency case drains
//! a single workgroup and the condition case drives the full grid, and that grid
//! override is precisely what each one measures, so each case keeps its own
//! `grid_override` and its own `dispatch_artifact_timed` call. Nothing in this
//! module reads a clock except [`timed_reference`], which times a closure the
//! caller supplies and never chooses what that closure computes.

use crate::api::case::{BenchContext, BenchError};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::resident::{
    input_bytes_total, transfer_accounting, ResidentDispatch, ResidentInputPool, TransferAccounting,
};
use std::time::Instant;
use vyre_runtime::resident_work_queue;

/// The four resident work-queue buffers a megakernel case dispatches over.
pub(crate) struct QueueBuffers {
    pub(crate) inputs: Vec<Vec<u8>>,
    pub(crate) input_bytes_total: u64,
    pub(crate) resident: Option<ResidentInputPool>,
}

/// Encode the control, debug and io buffers around a case's ring, and upload the
/// set into a rotating resident pool when the backend supports residency.
///
/// The vector order is the program's binding order, so this is the one place the
/// four-buffer layout is decided. `ring_bytes` stays the caller's: the ring is
/// the workload each case is actually built to measure.
pub(crate) fn queue_buffers(
    ctx: &mut BenchContext,
    ring_bytes: Vec<u8>,
    resident_sample_sets: usize,
    cleanup_label: &'static str,
) -> Result<QueueBuffers, BenchError> {
    let control_bytes = resident_work_queue::encode_control(false, 1, 0)
        .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
    let debug_bytes =
        resident_work_queue::encode_empty_debug_log(resident_work_queue::debug::RECORD_CAPACITY)
            .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
    let io_bytes =
        resident_work_queue::io::try_encode_empty_io_queue(resident_work_queue::io::IO_SLOT_COUNT)
            .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
    let inputs = vec![control_bytes, ring_bytes, debug_bytes, io_bytes];
    let input_bytes_total = input_bytes_total(&inputs);
    let resident =
        ResidentInputPool::upload_optional(ctx, &inputs, resident_sample_sets, cleanup_label)?;

    Ok(QueueBuffers {
        inputs,
        input_bytes_total,
        resident,
    })
}

/// One measured megakernel sample, unpacked and accounted.
pub(crate) struct QueueSample {
    pub(crate) outputs: Vec<Vec<u8>>,
    pub(crate) wall_ns: u64,
    pub(crate) dispatch_ns: Option<u64>,
    /// Device time when the backend reported it, otherwise wall time.
    pub(crate) device_ns: u64,
    pub(crate) output_bytes_total: u64,
    pub(crate) accounting: TransferAccounting,
    pub(crate) resident_used: bool,
}

/// Account a dispatch the calling case already measured.
///
/// Every value here is arithmetic over timings and byte counts the dispatch
/// returned. No clock is read, so two cases sharing this cannot change what
/// either of them measured.
pub(crate) fn account(dispatch: ResidentDispatch, input_bytes_total: u64) -> QueueSample {
    let resident_used = dispatch.resident_used;
    let wall_ns = dispatch.timed.wall_ns;
    let dispatch_ns = dispatch.timed.device_ns;
    let outputs = dispatch.timed.outputs;
    let output_bytes_total = outputs.iter().map(Vec::len).sum::<usize>() as u64;

    QueueSample {
        outputs,
        wall_ns,
        dispatch_ns,
        device_ns: dispatch_ns.unwrap_or(wall_ns),
        output_bytes_total,
        accounting: transfer_accounting(input_bytes_total, output_bytes_total, resident_used),
        resident_used,
    }
}

/// Time a CPU reference and report its wall nanoseconds.
///
/// The closure is the caller's reference implementation, which is the arm the
/// dispatched outputs are compared against and is never shared between cases.
pub(crate) fn timed_reference<T>(reference: impl FnOnce() -> T) -> (T, u64) {
    let started = Instant::now();
    let value = reference();
    let wall_ns = started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64;
    (value, wall_ns)
}

/// How many resident input-buffer sets the measured sample rotated through.
///
/// Zero means the backend did not support residency and the sample fell back to
/// a full host round trip, which the transfer accounting already reflects.
pub(crate) fn resident_pool_sets_metric(resident_used: bool, sample_sets: usize) -> MetricPoint {
    MetricPoint {
        name: "megakernel_resident_input_pool_sets".to_string(),
        value: if resident_used { sample_sets as u64 } else { 0 },
    }
}

/// Metrics for the CPU reference sample a megakernel case is reported against.
///
/// The reference rewrites host buffers, so it reads the whole input set and
/// writes whatever it produced; it never dispatches, so it carries no device
/// time.
pub(crate) fn reference_metrics(
    wall_ns: u64,
    input_bytes_total: u64,
    output_bytes: u64,
) -> BenchMetrics {
    BenchMetrics {
        wall_ns: Some(wall_ns),
        input_bytes: Some(input_bytes_total),
        output_bytes: Some(output_bytes),
        bytes_touched: Some(input_bytes_total.saturating_add(output_bytes)),
        bytes_read: Some(input_bytes_total),
        bytes_written: Some(output_bytes),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{account, reference_metrics, resident_pool_sets_metric};
    use crate::api::resident::ResidentDispatch;
    use vyre_driver::TimedDispatchResult;

    fn dispatch(
        wall_ns: u64,
        device_ns: Option<u64>,
        outputs: Vec<Vec<u8>>,
        resident_used: bool,
    ) -> ResidentDispatch {
        ResidentDispatch {
            timed: TimedDispatchResult {
                outputs,
                wall_ns,
                device_ns,
                enqueue_ns: None,
                wait_ns: None,
            },
            resident_used,
        }
    }

    /// A backend that reports no device time falls back to wall time, so a
    /// throughput figure divided by `device_ns` can never divide by a missing
    /// measurement.
    #[test]
    fn device_time_falls_back_to_wall_time() {
        let with_device = account(dispatch(900, Some(300), vec![vec![0; 4]], true), 64);
        let without_device = account(dispatch(900, None, vec![vec![0; 4]], true), 64);

        assert_eq!(with_device.device_ns, 300);
        assert_eq!(without_device.device_ns, 900);
        assert_eq!(without_device.dispatch_ns, None);
    }

    /// Output bytes are summed across every returned buffer, not just the first,
    /// and a resident sample is accounted as output-only host traffic.
    #[test]
    fn accounting_sums_every_output_buffer() {
        let resident = account(
            dispatch(10, Some(5), vec![vec![0; 4], vec![0; 12], vec![]], true),
            1_000,
        );
        let fallback = account(
            dispatch(10, Some(5), vec![vec![0; 4], vec![0; 12], vec![]], false),
            1_000,
        );

        assert_eq!(resident.output_bytes_total, 16);
        assert_eq!(fallback.output_bytes_total, 16);
        assert_eq!(resident.accounting.bytes_read, 0);
        assert_eq!(resident.accounting.bytes_written, 16);
        assert_eq!(resident.accounting.bytes_touched, 16);
        assert_eq!(fallback.accounting.bytes_read, 1_000);
        assert_eq!(fallback.accounting.bytes_written, 16);
        assert_eq!(fallback.accounting.bytes_touched, 1_016);
    }

    /// A fallback sample reports zero pool sets however large the pool was, so a
    /// report cannot claim residency the run did not get.
    #[test]
    fn pool_sets_metric_reports_zero_without_residency() {
        for sample_sets in [0_usize, 1, 8, 64] {
            assert_eq!(
                resident_pool_sets_metric(true, sample_sets).value,
                sample_sets as u64
            );
            assert_eq!(resident_pool_sets_metric(false, sample_sets).value, 0);
        }
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
}
