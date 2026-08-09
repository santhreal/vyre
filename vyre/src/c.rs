//! Execution adapters for neutral C preprocessing programs.

pub use vyre_libs::parsing::c::preprocess::gpu_pipeline::*;

/// Connects a driver backend to the neutral C preprocessing callback contract.
pub struct BackendDispatcher<'a>(pub &'a dyn crate::VyreBackend);

impl vyre_libs::parsing::c::preprocess::gpu_pipeline::GpuDispatcher for BackendDispatcher<'_> {
    fn dispatch(
        &self,
        program: &crate::Program,
        inputs: &[Vec<u8>],
    ) -> Result<Vec<Vec<u8>>, String> {
        crate::VyreBackend::dispatch(self.0, program, inputs, &crate::DispatchConfig::default())
            .map_err(|error| format!("backend dispatch: {error}"))
    }

    fn dispatch_borrowed(
        &self,
        program: &crate::Program,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, String> {
        self.0
            .dispatch_borrowed(program, inputs, &crate::DispatchConfig::default())
            .map_err(|error| format!("backend dispatch_borrowed: {error}"))
    }

    fn dispatch_borrowed_into(
        &self,
        program: &crate::Program,
        inputs: &[&[u8]],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), String> {
        self.0
            .dispatch_borrowed_into(program, inputs, &crate::DispatchConfig::default(), outputs)
            .map_err(|error| format!("backend dispatch_borrowed_into: {error}"))
    }
}
