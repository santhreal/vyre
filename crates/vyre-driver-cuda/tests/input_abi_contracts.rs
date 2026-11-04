//! CUDA's answer to the host input ABI contract on a live device.
//!
//! One answer decides which declarations a caller fills:
//! `BufferDecl::consumes_host_input`. The neutral half of that contract, what
//! `BindingPlan` accepts and how it refuses, belongs to `vyre-driver` and is
//! proved there. What is CUDA here is that a live dispatch agrees with the
//! reference interpreter on the same programs.

#[cfg(feature = "device-tests")]
use vyre_foundation::ir::Program;
#[cfg(feature = "device-tests")]
use vyre_test_support::host_input_abi::{
    assert_backend_agrees_with_reference_on_input_counts, assert_long_form_placeholder_is_refused,
    HostInputDispatch,
};

/// The CUDA device entry point is an inherent method, so the bridge to the
/// shared contract is stated here rather than inherited from `VyreBackend`.
#[cfg(feature = "device-tests")]
struct CudaHostInputs(vyre_driver_cuda::CudaBackend);

#[cfg(feature = "device-tests")]
impl HostInputDispatch for CudaHostInputs {
    fn dispatch_host_inputs(&self, program: &Program, inputs: &[Vec<u8>]) -> Result<(), String> {
        self.0
            .dispatch(program, inputs, &vyre_driver::DispatchConfig::default())
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

/// A live CUDA backend, or a loud failure: a probe failure is a configuration
/// failure, never a skip.
#[cfg(feature = "device-tests")]
fn live_backend() -> CudaHostInputs {
    CudaHostInputs(
        vyre_driver_cuda::CudaBackend::acquire()
            .expect("Fix: live CUDA backend is required for input ABI contract coverage"),
    )
}

#[cfg(feature = "device-tests")]
#[test]
fn cuda_refuses_a_long_form_input_list_carrying_an_output_placeholder() {
    assert_long_form_placeholder_is_refused(&live_backend());
}

#[cfg(feature = "device-tests")]
#[test]
fn cuda_and_the_reference_interpreter_accept_the_same_input_counts() {
    assert_backend_agrees_with_reference_on_input_counts(&live_backend());
}
