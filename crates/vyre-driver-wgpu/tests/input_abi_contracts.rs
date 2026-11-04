//! WGPU's answer to the host input ABI contract on a live device.
//!
//! One answer decides which declarations a caller fills:
//! `BufferDecl::consumes_host_input`. The neutral half of that contract, what
//! `BindingPlan` accepts and how it refuses, belongs to `vyre-driver` and is
//! proved there. What is WGPU here is that a live dispatch agrees with the
//! reference interpreter on the same programs.

#[cfg(feature = "device-tests")]
use vyre_test_support::host_input_abi::{
    assert_backend_agrees_with_reference_on_input_counts, assert_long_form_placeholder_is_refused,
};

/// A live WGPU backend, or a loud failure: a probe failure is a configuration
/// failure, never a skip.
#[cfg(feature = "device-tests")]
fn live_backend() -> vyre_driver_wgpu::WgpuBackend {
    vyre_driver_wgpu::WgpuBackend::acquire()
        .expect("Fix: live WGPU backend is required for input ABI contract coverage")
}

#[cfg(feature = "device-tests")]
#[test]
fn wgpu_refuses_a_long_form_input_list_carrying_an_output_placeholder() {
    assert_long_form_placeholder_is_refused(&live_backend());
}

#[cfg(feature = "device-tests")]
#[test]
fn wgpu_and_the_reference_interpreter_accept_the_same_input_counts() {
    assert_backend_agrees_with_reference_on_input_counts(&live_backend());
}
