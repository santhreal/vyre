use vyre_driver::{BackendError, DispatchConfig};
use vyre_foundation::ir::Program;

use crate::backend::dispatch::CudaBackend;
use crate::backend::dispatch_phase_probe as probe;
use crate::backend::module_cache::ModuleCacheKey;
use crate::backend::resident::{
    resident_bindings_from_handles, CudaDispatchBinding, CudaResidentBuffer,
};

impl CudaBackend {
    /// Dispatch with CUDA-resident buffers and return ordered output readbacks.
    pub fn dispatch_resident_timed(
        &self,
        program: &Program,
        handles: &[CudaResidentBuffer],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, BackendError> {
        self.dispatch_bindings_timed(program, &resident_bindings_from_handles(handles)?, config)
    }

    /// Dispatch a mixed binding list and return ordered output readbacks.
    ///
    /// Residency is per binding: a resident entry binds device memory the
    /// caller already owns, a borrowed entry is staged for this dispatch only.
    pub(crate) fn dispatch_bindings_timed(
        &self,
        program: &Program,
        bindings: &[CudaDispatchBinding<'_>],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, BackendError> {
        if crate::instrumentation::cuda_resident_borrowed_fallback_enabled() {
            let started = std::time::Instant::now();
            let enqueue_started = std::time::Instant::now();
            let outputs = self.dispatch_resident_via_borrowed(program, bindings, config)?;
            let enqueue_ns = crate::numeric::CUDA_NUMERIC
                .elapsed_nanos_u64(enqueue_started, "resident-dispatch enqueue latency")?;
            let wait_started = std::time::Instant::now();
            let wait_ns = crate::numeric::CUDA_NUMERIC
                .elapsed_nanos_u64(wait_started, "resident-dispatch wait latency")?;
            let wall_ns = crate::numeric::CUDA_NUMERIC
                .elapsed_nanos_u64(started, "resident-dispatch wall latency")?;
            self.telemetry
                .record_timed_dispatch(wall_ns, None, Some(enqueue_ns), Some(wait_ns));
            return Ok(vyre_driver::TimedDispatchResult::split_timed(
                outputs, wall_ns, None, enqueue_ns, wait_ns,
            ));
        }
        let started = std::time::Instant::now();
        let enqueue_started = std::time::Instant::now();
        let prepared = probe::measure(probe::Phase::Prepare, || {
            self.prepare_resident_dispatch(program, bindings, config)
        })?;
        let (ptx_src, ptx_source_key) = probe::measure(probe::Phase::Ptx, || {
            self.ptx_for_program_cached_with_key(program, config)
        })?;
        let module_key = probe::measure(probe::Phase::ModuleKey, || {
            self.module_cache_key_for_ptx_source_key(ptx_source_key)
        })?;
        probe::record_counts(
            program,
            ptx_src.len(),
            bindings.len(),
            prepared.fixpoint_iterations as usize,
            prepared.launch.grid,
        );
        let resident_dispatch = self.dispatch_resident_async_concrete_with_ptx_key(
            program, bindings, config, &ptx_src, module_key, true, None, true, &prepared,
        )?;
        let enqueue_ns = crate::numeric::CUDA_NUMERIC
            .elapsed_nanos_u64(enqueue_started, "native-resident-dispatch enqueue latency")?;
        let wait_started = std::time::Instant::now();
        let (outputs, device_ns) = resident_dispatch.pending.await_timed_result()?;
        let wait_ns = crate::numeric::CUDA_NUMERIC
            .elapsed_nanos_u64(wait_started, "native-resident-dispatch wait latency")?;
        let wall_ns = crate::numeric::CUDA_NUMERIC
            .elapsed_nanos_u64(started, "native-resident-dispatch wall latency")?;
        self.telemetry
            .record_timed_dispatch(wall_ns, device_ns, Some(enqueue_ns), Some(wait_ns));
        if probe::enabled() {
            let ptx_cache = self.ptx_source_cache_snapshot();
            probe::emit(
                self.telemetry.snapshot().timed_dispatches,
                wall_ns,
                enqueue_ns,
                wait_ns,
                device_ns,
                ptx_cache.hits,
                ptx_cache.misses,
            );
        }
        Ok(vyre_driver::TimedDispatchResult::split_timed(
            outputs, wall_ns, device_ns, enqueue_ns, wait_ns,
        ))
    }

    pub(crate) fn dispatch_resident_outputs_with_ptx_key_into(
        &self,
        program: &Program,
        bindings: &[CudaDispatchBinding<'_>],
        config: &DispatchConfig,
        ptx_src: &str,
        module_key: ModuleCacheKey,
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        if crate::instrumentation::cuda_resident_borrowed_fallback_enabled() {
            return self.dispatch_resident_via_borrowed_into(program, bindings, config, outputs);
        }
        let prepared = self.prepare_resident_dispatch(program, bindings, config)?;
        let dispatch = self.dispatch_resident_async_concrete_with_ptx_key(
            program, bindings, config, ptx_src, module_key, false, None, true, &prepared,
        )?;
        let (dispatch_outputs, _) = dispatch.pending.await_timed_result()?;
        vyre_driver::replace_output_buffers_preserving_slots(dispatch_outputs, outputs);
        Ok(())
    }
}
