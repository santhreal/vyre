use vyre_driver::{BackendError, DispatchConfig};
use vyre_foundation::ir::Program;

use crate::backend::dispatch::CudaBackend;
use crate::backend::resident::{
    resident_bindings_from_handles, CudaDispatchBinding, CudaResidentBuffer,
};

impl CudaBackend {
    /// Dispatch a Program using caller-provided CUDA-resident buffers.
    pub fn dispatch_resident(
        &self,
        program: &Program,
        handles: &[CudaResidentBuffer],
        config: &DispatchConfig,
    ) -> Result<(), BackendError> {
        self.dispatch_bindings(program, &resident_bindings_from_handles(handles)?, config)
    }

    /// Dispatch a Program against a mixed binding list, discarding outputs.
    ///
    /// Residency is per binding: a resident entry binds device memory the
    /// caller already owns, a borrowed entry is staged for this dispatch only.
    pub(crate) fn dispatch_bindings(
        &self,
        program: &Program,
        bindings: &[CudaDispatchBinding<'_>],
        config: &DispatchConfig,
    ) -> Result<(), BackendError> {
        if crate::instrumentation::cuda_resident_borrowed_fallback_enabled() {
            return self
                .dispatch_resident_via_borrowed(program, bindings, config)
                .map(|_| ());
        }
        let prepared = self.prepare_resident_dispatch(program, bindings, config)?;
        let (ptx_src, ptx_source_key) = self.ptx_for_program_cached_with_key(program, config)?;
        let module_key = self.module_cache_key_for_ptx_source_key(ptx_source_key)?;
        self.dispatch_resident_async_concrete_with_ptx_key(
            program, bindings, config, &ptx_src, module_key, false, None, false, &prepared,
        )?;
        Ok(())
    }
}
