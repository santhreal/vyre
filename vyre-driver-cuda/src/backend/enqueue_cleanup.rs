//! What an in-flight dispatch enqueue owns, and how it is handed over or
//! abandoned.
//!
//! Every asynchronous CUDA dispatch path holds the same guards between the
//! first enqueue and the moment the pending handle takes ownership: a launch
//! resource lease, transient device allocations, pinned host staging, and for
//! resident dispatch an in-flight resident-handle guard. When an enqueue fails
//! after work is already on the stream, the order in which those are released
//! is a correctness decision, not a formality, so it is made once here instead
//! of in each dispatch path.
//!
//! Holding them is also one decision. Host, resident and resident-batch
//! dispatch each wrapped the same four values in `Option`, read the raw stream
//! out of the lease, took the four back out for the pending handle, and raised
//! the same five "consumed before" diagnostics with only the path name
//! differing. [`EnqueueGuards`] is that shape: the path name is the one thing a
//! caller supplies, and a guard added here is a guard every dispatch path
//! releases in the proven order.

use cudarc::driver::sys::CUstream;
use vyre_driver::BackendError;

use crate::backend::allocations::{DispatchAllocations, HostTransferAllocations};
use crate::backend::resident::ResidentUseGuard;
use crate::backend::telemetry::CudaTelemetry;
use crate::stream::{CudaEvent, CudaLaunchResourceLease, CudaStream};

/// Diagnostic for a guard that was already handed over.
///
/// Free rather than a method so the borrow bundle can name it while it holds
/// disjoint field borrows of the guards it describes.
fn consumed(subject: &str, what: &str, stage: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!(
            "Fix: CUDA {subject} {what} were consumed before {stage}; rebuild pending dispatch ownership before launching."
        ),
    }
}

/// The guards an enqueue reads while it is recording onto the stream.
pub(crate) struct EnqueueRecording<'a> {
    pub(crate) allocations: &'a DispatchAllocations,
    pub(crate) host_transfers: &'a mut HostTransferAllocations,
    pub(crate) launch_resources: &'a CudaLaunchResourceLease,
}

/// The guards one in-flight enqueue owns, each `None` once it has been handed
/// to the pending dispatch.
///
/// `subject` names the dispatch path in every diagnostic these guards raise.
/// It is the only per-path value: the release order, the handover order, and
/// the failure text are the same decision for all of them.
pub(crate) struct EnqueueGuards {
    subject: &'static str,
    launch_resources: Option<CudaLaunchResourceLease>,
    allocations: Option<DispatchAllocations>,
    host_transfers: Option<HostTransferAllocations>,
    /// A borrowed dispatch binds no resident buffers, so it holds `None` here
    /// for its whole life rather than having taken one already.
    resident_use: Option<ResidentUseGuard>,
}

impl EnqueueGuards {
    pub(crate) fn new(
        subject: &'static str,
        launch_resources: CudaLaunchResourceLease,
        allocations: DispatchAllocations,
        host_transfers: HostTransferAllocations,
        resident_use: Option<ResidentUseGuard>,
    ) -> Self {
        Self {
            subject,
            launch_resources: Some(launch_resources),
            allocations: Some(allocations),
            host_transfers: Some(host_transfers),
            resident_use,
        }
    }

    fn consumed(&self, what: &str, stage: &str) -> BackendError {
        consumed(self.subject, what, stage)
    }

    /// Raw stream every enqueue in this dispatch is recorded on.
    pub(crate) fn stream_raw(&self) -> Result<CUstream, BackendError> {
        self.launch_resources
            .as_ref()
            .ok_or_else(|| self.consumed("launch resources", "enqueue"))?
            .stream_raw()
    }

    /// Timing event pair this dispatch records around its launches, or `None`
    /// when timing was not captured.
    pub(crate) fn timing_events(&self) -> Result<Option<&(CudaEvent, CudaEvent)>, BackendError> {
        self.launch_resources
            .as_ref()
            .ok_or_else(|| self.consumed("launch resources", "timing-event record"))?
            .timing_events()
    }

    /// Borrow the three guards an enqueue reads while it records.
    ///
    /// A dispatch resolves device pointers out of the allocations, pushes the
    /// matching pinned host slot, and records timing events inside one enqueue,
    /// so the three borrows are split here instead of being sequenced by the
    /// caller.
    pub(crate) fn recording(&mut self) -> Result<EnqueueRecording<'_>, BackendError> {
        let subject = self.subject;
        Ok(EnqueueRecording {
            allocations: self
                .allocations
                .as_ref()
                .ok_or_else(|| consumed(subject, "allocations", "enqueue finished"))?,
            host_transfers: self
                .host_transfers
                .as_mut()
                .ok_or_else(|| consumed(subject, "host staging", "enqueue finished"))?,
            launch_resources: self
                .launch_resources
                .as_ref()
                .ok_or_else(|| consumed(subject, "launch resources", "enqueue finished"))?,
        })
    }

    /// Take the lease apart for the pending handle.
    pub(crate) fn take_stream_and_timing(
        &mut self,
    ) -> Result<(CudaStream, Option<(CudaEvent, CudaEvent)>), BackendError> {
        self.launch_resources
            .take()
            .ok_or_else(|| {
                self.consumed("launch resources", "pending dispatch ownership transfer")
            })?
            .into_parts()
    }

    pub(crate) fn take_allocations(&mut self) -> Result<DispatchAllocations, BackendError> {
        self.allocations
            .take()
            .ok_or_else(|| self.consumed("allocations", "pending dispatch ownership transfer"))
    }

    pub(crate) fn take_host_transfers(&mut self) -> Result<HostTransferAllocations, BackendError> {
        self.host_transfers
            .take()
            .ok_or_else(|| self.consumed("host staging", "pending dispatch ownership transfer"))
    }

    /// Take the resident in-flight guard for the pending handle.
    ///
    /// A dispatch that never bound a resident buffer must not call this: the
    /// absent guard and an already-transferred guard are the same `None` here,
    /// and only the second is an error worth a diagnostic.
    pub(crate) fn take_resident_use(&mut self) -> Result<ResidentUseGuard, BackendError> {
        self.resident_use
            .take()
            .ok_or_else(|| self.consumed("use guard", "pending dispatch ownership transfer"))
    }

    /// Release what a failed enqueue still owns and return the error to
    /// propagate.
    ///
    /// The stream is synchronized first: copies already queued on it read the
    /// staging these guards own, so recycling that memory ahead of the fence
    /// hands a live device read back to the next dispatch. When the guards have
    /// already been transferred there is nothing on this stream left to fence,
    /// and the error propagates unchanged.
    ///
    /// A synchronize that itself fails leaves the queue state unknown, so every
    /// guard is leaked deliberately. Leaking a pool block costs memory;
    /// recycling one the device may still be reading corrupts the dispatch that
    /// gets it next.
    ///
    /// `stage` names the enqueue step for the diagnostic; `sync_label` is the
    /// label the synchronize reports under.
    pub(crate) fn abandon(
        &mut self,
        error: BackendError,
        telemetry: &CudaTelemetry,
        stream_raw: CUstream,
        sync_label: &'static str,
        stage: &str,
    ) -> BackendError {
        let Some(launch_resources) = self.launch_resources.take() else {
            return error;
        };
        let subject = self.subject;
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
                if let Some(allocations) = self.allocations.take() {
                    std::mem::forget(allocations);
                }
                if let Some(resident_use) = self.resident_use.take() {
                    std::mem::forget(resident_use);
                }
                if let Some(host_transfers) = self.host_transfers.take() {
                    std::mem::forget(host_transfers);
                }
                error
            }
        }
    }
}
