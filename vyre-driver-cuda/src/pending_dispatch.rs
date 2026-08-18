use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cudarc::driver::CudaContext;
use vyre_driver::sealed;
use vyre_driver::{BackendError, PendingDispatch};

use crate::backend::allocations::DispatchAllocations;
use crate::backend::module_globals::ModuleGlobalsLease;
use crate::backend::pinned_allocations::HostTransferAllocations;
use crate::backend::resident::ResidentUseGuard;
use crate::backend::telemetry::CudaTelemetry;
use crate::stream::{CudaEvent, CudaLaunchResourcePool, CudaStream};

/// CUDA-backed pending dispatch whose result is fenced by a CUDA event.
#[derive(Debug)]
pub(crate) struct CudaPendingDispatch {
    ctx: Arc<CudaContext>,
    pool: Arc<CudaLaunchResourcePool>,
    event: Option<CudaEvent>,
    stream: Option<CudaStream>,
    allocations: Option<DispatchAllocations>,
    resident_use: Option<ResidentUseGuard>,
    host_transfers: Option<HostTransferAllocations>,
    outputs: Vec<Vec<u8>>,
    timing_start: Option<CudaEvent>,
    timing_end: Option<CudaEvent>,
    ready_device_ns: Option<u64>,
    telemetry: Arc<CudaTelemetry>,
    completed: AtomicBool,
    /// The module-scope globals lease held across the kernel's execution.
    ///
    /// Present only on a submission that deferred its release: the gate must stay
    /// held until the completion event has been awaited, because the trap record
    /// and the grid-barrier counter are live for the kernel's whole execution and
    /// not merely until the launch call returned. Ending it at enqueue is what
    /// makes an asynchronous submission block.
    module_globals: Option<ModuleGlobalsLease>,
}

impl CudaPendingDispatch {
    /// Build an already-completed pending dispatch.
    pub(crate) fn new_ready(
        ctx: Arc<CudaContext>,
        pool: Arc<CudaLaunchResourcePool>,
        outputs: Vec<Vec<u8>>,
        telemetry: Arc<CudaTelemetry>,
    ) -> Self {
        Self {
            ctx,
            pool,
            event: None,
            stream: None,
            allocations: None,
            resident_use: None,
            host_transfers: None,
            outputs,
            timing_start: None,
            timing_end: None,
            ready_device_ns: None,
            telemetry,
            completed: AtomicBool::new(true),
            module_globals: None,
        }
    }

    /// Build an already-completed pending dispatch with measured device time.
    pub(crate) fn new_ready_timed(
        ctx: Arc<CudaContext>,
        pool: Arc<CudaLaunchResourcePool>,
        outputs: Vec<Vec<u8>>,
        device_ns: Option<u64>,
        telemetry: Arc<CudaTelemetry>,
    ) -> Self {
        Self {
            ctx,
            pool,
            event: None,
            stream: None,
            allocations: None,
            resident_use: None,
            host_transfers: None,
            outputs,
            timing_start: None,
            timing_end: None,
            ready_device_ns: device_ns,
            telemetry,
            completed: AtomicBool::new(true),
            module_globals: None,
        }
    }

    /// Build a pending resident batch dispatch with no host output slots.
    ///
    /// Resident batch readback uses caller-owned resident handles; the pending
    /// dispatch only fences parameter uploads and kernel launches.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_resident_batch_pending(
        ctx: Arc<CudaContext>,
        pool: Arc<CudaLaunchResourcePool>,
        event: CudaEvent,
        stream: CudaStream,
        allocations: DispatchAllocations,
        resident_use: ResidentUseGuard,
        host_transfers: HostTransferAllocations,
        telemetry: Arc<CudaTelemetry>,
    ) -> Self {
        Self::new(
            ctx,
            pool,
            event,
            stream,
            allocations,
            Some(resident_use),
            Some(host_transfers),
            Vec::new(),
            telemetry,
        )
    }

    /// Build a pending dispatch after all GPU work has been enqueued.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        ctx: Arc<CudaContext>,
        pool: Arc<CudaLaunchResourcePool>,
        event: CudaEvent,
        stream: CudaStream,
        allocations: DispatchAllocations,
        resident_use: Option<ResidentUseGuard>,
        host_transfers: Option<HostTransferAllocations>,
        outputs: Vec<Vec<u8>>,
        telemetry: Arc<CudaTelemetry>,
    ) -> Self {
        Self {
            ctx,
            pool,
            event: Some(event),
            stream: Some(stream),
            allocations: Some(allocations),
            resident_use,
            host_transfers,
            outputs,
            timing_start: None,
            timing_end: None,
            ready_device_ns: None,
            telemetry,
            completed: AtomicBool::new(false),
            module_globals: None,
        }
    }

    /// Build a pending dispatch with timing-enabled start/end events.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_with_timing(
        ctx: Arc<CudaContext>,
        pool: Arc<CudaLaunchResourcePool>,
        event: CudaEvent,
        stream: CudaStream,
        allocations: DispatchAllocations,
        resident_use: Option<ResidentUseGuard>,
        host_transfers: Option<HostTransferAllocations>,
        outputs: Vec<Vec<u8>>,
        timing_start: CudaEvent,
        timing_end: CudaEvent,
        telemetry: Arc<CudaTelemetry>,
    ) -> Self {
        Self {
            ctx,
            pool,
            event: Some(event),
            stream: Some(stream),
            allocations: Some(allocations),
            resident_use,
            host_transfers,
            outputs,
            timing_start: Some(timing_start),
            timing_end: Some(timing_end),
            ready_device_ns: None,
            telemetry,
            completed: AtomicBool::new(false),
            module_globals: None,
        }
    }

    /// Carry a module-globals lease that must outlive the launch.
    ///
    /// The submission path enqueued its launches with
    /// [`ModuleGlobalsLease::launch_then_defer_release`], so the gate is still
    /// held. This handle owns the completion event, which makes it the one place
    /// that can prove the kernel finished, and therefore the one place the lease
    /// may be ended.
    #[must_use]
    pub(crate) fn holding_module_globals(mut self, lease: ModuleGlobalsLease) -> Self {
        self.module_globals = Some(lease);
        self
    }

    fn bind_context(&self) -> Result<(), BackendError> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| BackendError::DispatchFailed {
                code: None,
                message: format!("CUDA context bind failed: {e}"),
            })
    }

    fn synchronize(&mut self) -> Result<(), BackendError> {
        if self.completed.load(Ordering::Acquire) {
            return self.release_module_globals();
        }
        self.bind_context()?;
        let event = self
            .event
            .as_ref()
            .ok_or_else(|| BackendError::DispatchFailed {
                code: None,
                message: "CUDA pending dispatch completion event was already released".to_string(),
            })?;
        event.synchronize()?;
        self.telemetry.record_sync_point();
        self.completed.store(true, Ordering::Release);
        self.release_module_globals()
    }

    /// End the deferred module-globals lease, now that completion is proven.
    ///
    /// Called from every path that observes completion, including the one that
    /// found `completed` already set: [`PendingDispatch::is_ready`] can set that
    /// flag without ending the lease, so an early return that skipped this would
    /// hold the module's gate until the process exited and block every later
    /// launch of the same module.
    ///
    /// The lease is taken out of the handle before it is ended, so a second call
    /// is a no-op and the drop path cannot release it twice.
    fn release_module_globals(&mut self) -> Result<(), BackendError> {
        match self.module_globals.take() {
            Some(lease) => lease.release_after_completion(),
            None => Ok(()),
        }
    }

    fn release_launch_resources(&mut self) {
        if let Some(event) = self.event.take() {
            self.pool.release_event(event);
        }
        if let Some(event) = self.timing_start.take() {
            self.pool.release_timing_event(event);
        }
        if let Some(event) = self.timing_end.take() {
            self.pool.release_timing_event(event);
        }
        if let Some(stream) = self.stream.take() {
            self.pool.release_stream(stream);
        }
    }

    fn force_completion_on_drop(&mut self) -> bool {
        if self.completed.load(Ordering::Acquire) {
            return true;
        }
        if let Err(error) = self.ctx.bind_to_thread() {
            tracing::error!(
                "Fix: failed to bind CUDA context while dropping pending dispatch: {error}. In-flight CUDA resources will not be recycled."
            );
            return false;
        }
        let Some(stream) = self.stream.as_ref() else {
            tracing::error!(
                "Fix: pending CUDA dispatch lost its stream before drop-time synchronization. In-flight CUDA resources will not be recycled."
            );
            return false;
        };
        if let Err(error) = stream.synchronize() {
            tracing::error!(
                "Fix: failed to synchronize CUDA stream while dropping pending dispatch: {error}. In-flight CUDA resources will not be recycled."
            );
            return false;
        }
        self.telemetry.record_sync_point();
        self.completed.store(true, Ordering::Release);
        true
    }

    fn leak_inflight_resources_after_drop_sync_failure(&mut self) {
        tracing::error!(
            "Fix: leaking CUDA pending-dispatch resources because completion could not be proven during drop; await the dispatch result before dropping it."
        );
        std::mem::forget(Arc::clone(&self.ctx));
        if let Some(event) = self.event.take() {
            std::mem::forget(event);
        }
        if let Some(event) = self.timing_start.take() {
            std::mem::forget(event);
        }
        if let Some(event) = self.timing_end.take() {
            std::mem::forget(event);
        }
        if let Some(stream) = self.stream.take() {
            std::mem::forget(stream);
        }
        if let Some(allocations) = self.allocations.take() {
            std::mem::forget(allocations);
        }
        if let Some(resident_use) = self.resident_use.take() {
            std::mem::forget(resident_use);
        }
        if let Some(host_transfers) = self.host_transfers.take() {
            std::mem::forget(host_transfers);
        }
        // The lease is FORGOTTEN, not released. This path could not prove the
        // kernel finished, and freeing the gate under a possibly-live grid lets
        // the next launch of this module zero `_vyre_grid_barrier` underneath it,
        // which is the corruption the gate exists to prevent. Leaving the gate
        // busy blocks later launches of this one module; the stream and its
        // allocations are already lost above, so both outcomes are a hang and
        // this one cannot produce a wrong answer.
        if let Some(lease) = self.module_globals.take() {
            std::mem::forget(lease);
        }
    }

    /// Await completion and return output buffers plus device elapsed time.
    pub(crate) fn await_timed_result(
        mut self,
    ) -> Result<(Vec<Vec<u8>>, Option<u64>), BackendError> {
        self.synchronize()?;
        let device_ns = match self.ready_device_ns.take() {
            Some(device_ns) => Some(device_ns),
            None => match (self.timing_start.as_ref(), self.timing_end.as_ref()) {
                (Some(start), Some(end)) => Some(start.elapsed_time_ns(end)?),
                _ => None,
            },
        };
        self.release_launch_resources();
        self.allocations.take();
        self.resident_use.take();
        let outputs = self.collect_outputs()?;
        self.host_transfers.take();
        Ok((outputs, device_ns))
    }

    fn collect_outputs(&mut self) -> Result<Vec<Vec<u8>>, BackendError> {
        if let Some(transfers) = self.host_transfers.as_ref() {
            let mut outputs = std::mem::take(&mut self.outputs);
            transfers.collect_outputs_into(&mut outputs)?;
            Ok(outputs)
        } else {
            Ok(std::mem::take(&mut self.outputs))
        }
    }

    fn collect_outputs_into(&mut self, outputs: &mut Vec<Vec<u8>>) -> Result<(), BackendError> {
        if let Some(transfers) = self.host_transfers.as_ref() {
            transfers.collect_outputs_into(outputs)?;
        } else {
            vyre_driver::replace_output_buffers_preserving_slots(
                std::mem::take(&mut self.outputs),
                outputs,
            );
        }
        Ok(())
    }
}

impl sealed::Sealed for CudaPendingDispatch {}

impl PendingDispatch for CudaPendingDispatch {
    fn is_ready(&self) -> bool {
        if self.completed.load(Ordering::Acquire) {
            return true;
        }
        if self.bind_context().is_err() {
            return false;
        }
        let Some(event) = self.event.as_ref() else {
            return true;
        };
        let ready = match event.query_ready() {
            Ok(ready) => ready,
            Err(error) => {
                tracing::error!(
                    "Fix: CUDA pending dispatch readiness query failed: {error}. Await the dispatch to surface synchronization failure details."
                );
                false
            }
        };
        if ready {
            self.completed.store(true, Ordering::Release);
        }
        ready
    }

    fn await_result(mut self: Box<Self>) -> Result<Vec<Vec<u8>>, BackendError> {
        self.synchronize()?;
        self.release_launch_resources();
        self.allocations.take();
        self.resident_use.take();
        let outputs = self.collect_outputs()?;
        self.host_transfers.take();
        Ok(outputs)
    }

    fn await_timed_result(
        self: Box<Self>,
    ) -> Result<vyre_driver::TimedDispatchResult, BackendError> {
        let started = std::time::Instant::now();
        let (outputs, device_ns) = CudaPendingDispatch::await_timed_result(*self)?;
        let wall_ns = u64::try_from(started.elapsed().as_nanos()).map_err(|_| {
            BackendError::new(
                "CUDA pending dispatch retirement exceeded the u64 nanosecond timing range",
            )
        })?;
        Ok(vyre_driver::TimedDispatchResult {
            outputs,
            wall_ns,
            device_ns,
            enqueue_ns: None,
            wait_ns: Some(wall_ns),
        })
    }

    fn await_result_into(
        mut self: Box<Self>,
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        self.synchronize()?;
        self.release_launch_resources();
        self.allocations.take();
        self.resident_use.take();
        self.collect_outputs_into(outputs)?;
        self.host_transfers.take();
        Ok(())
    }
}

impl Drop for CudaPendingDispatch {
    fn drop(&mut self) {
        if !self.force_completion_on_drop() {
            self.leak_inflight_resources_after_drop_sync_failure();
            return;
        }
        // Completion is proven, so the trap record and the arrival count are
        // final and the gate can be freed. A drop has nowhere to return an error
        // to, so a trapped kernel whose result was never awaited is reported here
        // instead of being lost.
        if let Err(error) = self.release_module_globals() {
            tracing::error!(
                "Fix: CUDA pending dispatch completed with a module-globals failure that no caller awaited: {error}. Await the dispatch result so the failure reaches the caller."
            );
        }
        self.release_launch_resources();
        self.allocations.take();
        self.resident_use.take();
        self.host_transfers.take();
    }
}
