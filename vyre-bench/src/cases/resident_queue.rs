//! The resident work-queue buffer set the megakernel cases share.
//!
//! All three megakernel cases publish a ring into the same four-buffer resident
//! work queue: control, ring, debug log, io queue. Each carried its own copy of
//! that encoding and its own post-dispatch accounting, so a change to the buffer
//! layout had to be made three times.
//!
//! What is deliberately not here is the timed dispatch. The latency case drains
//! a single workgroup and the condition case drives the full grid, and that grid
//! override is precisely what each one measures, so each case keeps its own
//! `grid_override` and its own `dispatch_artifact_timed` call. Nothing in this
//! module reads a clock at all: `account` is arithmetic over timings the
//! dispatch already returned.

use super::byte_pack::gb_per_second;
use crate::api::case::{BenchContext, BenchError};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::resident::{
    input_bytes_total, transfer_accounting, ResidentDispatch, ResidentInputPool, TransferAccounting,
};
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
    let control_bytes = resident_work_queue::protocol::encode_control(false, 1, 0)
        .map_err(|error| BenchError::ExecutionFailed(error.to_string()))?;
    let debug_bytes = resident_work_queue::protocol::encode_empty_debug_log(
        resident_work_queue::protocol::debug::RECORD_CAPACITY,
    )
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

/// The metrics an accounted sample reports, before the case adds its own.
///
/// Every field here is a byte count or a timing the dispatch already returned,
/// or throughput arithmetic over those two. A case states only what it measures
/// beyond them, through struct update syntax over this value:
///
/// ```text
/// BenchMetrics {
///     atomic_op_count: Some(slots),
///     custom: vec![point],
///     ..accounted_metrics(&sample, prepared.input_bytes_total)
/// }
/// ```
///
/// A case that reports a different set of fields keeps its own block rather
/// than overriding this one, so what a case records stays visible where it is
/// measured.
pub(crate) fn accounted_metrics(sample: &QueueSample, input_bytes_total: u64) -> BenchMetrics {
    BenchMetrics {
        wall_ns: Some(sample.wall_ns),
        dispatch_ns: sample.dispatch_ns,
        input_bytes: Some(input_bytes_total),
        output_bytes: Some(sample.output_bytes_total),
        bytes_touched: Some(sample.accounting.bytes_touched),
        bytes_read: Some(sample.accounting.bytes_read),
        bytes_written: Some(sample.accounting.bytes_written),
        wall_throughput_gb_s: Some(gb_per_second(
            sample.accounting.bytes_touched,
            sample.wall_ns,
        )),
        device_throughput_gb_s: Some(gb_per_second(
            sample.accounting.bytes_touched,
            sample.device_ns,
        )),
        ..Default::default()
    }
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

#[cfg(test)]
mod tests {
    use super::{account, resident_pool_sets_metric};
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
}
