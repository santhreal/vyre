//! Release path for a dispatch enqueue that failed partway through.
//!
//! Every asynchronous CUDA dispatch path holds the same guards between the
//! first enqueue and the moment the pending handle takes ownership: a launch
//! resource lease, transient device allocations, pinned host staging, and for
//! resident dispatch an in-flight resident-handle guard. When an enqueue fails
//! after work is already on the stream, the order in which those are released
//! is a correctness decision, not a formality, so it is made once here instead
//! of in each dispatch path.

use cudarc::driver::sys::CUstream;
use vyre_driver::BackendError;

use crate::backend::allocations::{DispatchAllocations, HostTransferAllocations};
use crate::backend::resident::ResidentUseGuard;
use crate::backend::telemetry::CudaTelemetry;

/// The guards a failed enqueue still owns, borrowed so the caller keeps them
/// for the success path.
pub(crate) struct FailedEnqueueGuards<'a> {
    pub(crate) launch_resources: &'a mut Option<crate::stream::CudaLaunchResourceLease>,
    pub(crate) allocations: &'a mut Option<DispatchAllocations>,
    pub(crate) host_transfers: &'a mut Option<HostTransferAllocations>,
    /// A borrowed dispatch binds no resident buffers, so it passes `None`.
    pub(crate) resident_use: Option<&'a mut Option<ResidentUseGuard>>,
}

/// Release what a failed enqueue still owns and return the error to propagate.
///
/// The stream is synchronized first: copies already queued on it read the
/// staging these guards own, so recycling that memory ahead of the fence hands
/// a live device read back to the next dispatch. When the guards have already
/// been transferred there is nothing on this stream left to fence, and the
/// error propagates unchanged.
///
/// A synchronize that itself fails leaves the queue state unknown, so every
/// guard is leaked deliberately. Leaking a pool block costs memory; recycling
/// one the device may still be reading corrupts the dispatch that gets it next.
///
/// `subject` names the dispatch path and `stage` the enqueue step, both only for
/// the diagnostic; `sync_label` is the label the synchronize reports under.
pub(crate) fn abandon_failed_enqueue(
    error: BackendError,
    telemetry: &CudaTelemetry,
    stream_raw: CUstream,
    sync_label: &'static str,
    subject: &str,
    stage: &str,
    guards: FailedEnqueueGuards<'_>,
) -> BackendError {
    let Some(launch_resources) = guards.launch_resources.take() else {
        return error;
    };
    match crate::stream::synchronize_raw_stream(stream_raw, sync_label) {
        Ok(()) => {
            telemetry.record_sync_point();
            error
        }
        Err(sync_error) => {
            tracing::error!(
                "Fix: failed to synchronize CUDA {subject} stream after {stage} error: {sync_error}. In-flight {subject} resources will not be recycled."
            );
            std::mem::forget(launch_resources);
            if let Some(allocations) = guards.allocations.take() {
                std::mem::forget(allocations);
            }
            if let Some(resident_use) = guards.resident_use.and_then(Option::take) {
                std::mem::forget(resident_use);
            }
            if let Some(host_transfers) = guards.host_transfers.take() {
                std::mem::forget(host_transfers);
            }
            error
        }
    }
}
