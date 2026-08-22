//! Wgpu adapter for the self-hosted optimizer, and the two IR shapes its
//! end-to-end suites assert against.
//!
//! `vyre_pass_engine::optimizer` drives every GPU pass through
//! [`ProgramDispatcher`]. Satisfying that trait from a live `WgpuBackend` is one
//! implementation, not one per suite.

use vyre::ir::Program;
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Adapts a live `WgpuBackend` to the dispatcher the self-hosted optimizer
/// expects.
pub(crate) struct WgpuProgramDispatcher<'a> {
    backend: &'a WgpuBackend,
}

impl<'a> WgpuProgramDispatcher<'a> {
    pub(crate) fn new(backend: &'a WgpuBackend) -> Self {
        Self { backend }
    }
}

impl ProgramDispatcher for WgpuProgramDispatcher<'_> {
    fn dispatch(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut config = DispatchConfig::default();
        config.grid_override = grid_override;
        VyreBackend::dispatch(self.backend, program, inputs, &config)
            .map_err(|err| DispatchError::BackendError(err.to_string()))
    }
}

/// The program shape every optimizer pass suite feeds in, and the reader that
/// peels the `Region` wrapper back off. Owned by `vyre-test-support` so the wgpu
/// and CUDA suites cannot disagree about either.
pub(crate) use vyre_test_support::pass_programs::{first_let_value, wrapped};
