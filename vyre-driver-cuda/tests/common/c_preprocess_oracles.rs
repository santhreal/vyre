//! The two `ProgramOracle` implementations the live C preprocess tests diff
//! against each other: the reference interpreter and the CUDA backend.
//!
//! Both arms of that comparison are contracts, and each was restated in every
//! preprocess test file. A copy that drifts in how it maps a dispatch error, or
//! in whether it reports `requires_output_inputs`, changes what the parity
//! assertion means without touching the assertion, so they live here once.

use vyre::ir::Program;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_libs::parsing::c::preprocess::gpu_pipeline::ProgramOracle;
use vyre_reference::value::Value;

/// Runs the pipeline on the reference interpreter.
pub(crate) struct ReferenceOracle;

impl ProgramOracle for ReferenceOracle {
    fn dispatch(&self, program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
        let values: Vec<Value> = inputs.iter().cloned().map(Value::from).collect();
        let outputs = vyre_reference::reference_eval(program, &values)
            .map_err(|error| format!("reference_eval: {error}"))?;
        Ok(outputs.into_iter().map(|value| value.to_bytes()).collect())
    }

    fn requires_output_inputs(&self) -> bool {
        true
    }
}

/// Runs the pipeline on a live CUDA device, over owned or borrowed inputs.
pub(crate) struct CudaOracle<'a>(pub(crate) &'a CudaBackend);

impl ProgramOracle for CudaOracle<'_> {
    fn dispatch(&self, program: &Program, inputs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
        self.0
            .dispatch(program, inputs, &DispatchConfig::default())
            .map_err(|error| format!("CUDA dispatch: {error}"))
    }

    fn dispatch_borrowed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, String> {
        self.0
            .dispatch_borrowed(program, inputs, &DispatchConfig::default())
            .map_err(|error| format!("CUDA borrowed dispatch: {error}"))
    }
}
