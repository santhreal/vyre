//! Adversarial closure contract for the SPIR-V backend.
//!
//! The obligations live in `vyre_driver::hostile_input_closure`; this target
//! says which backend owes them and how to reach it. The cases chosen are the
//! ones the backend can decide before it touches Vulkan, so when no Vulkan
//! device is present the target reports that rather than failing: the contract
//! is about the validation path, not about GPU execution.

#![forbid(unsafe_code)]

use vyre_driver::hostile_input_closure::{
    assert_trailing_inputs_rejected, assert_zero_workgroup_rejected,
};
use vyre_driver_spirv::SpirvBackendRegistration;

#[test]
fn adversarial_registry_closure_hostile_inputs() {
    let Ok(backend) = SpirvBackendRegistration::acquire() else {
        println!("no Vulkan compute device present; SPIR-V validation contract not exercised");
        return;
    };
    assert_zero_workgroup_rejected(&backend, "spirv");
    assert_trailing_inputs_rejected(&backend, "spirv");
}
