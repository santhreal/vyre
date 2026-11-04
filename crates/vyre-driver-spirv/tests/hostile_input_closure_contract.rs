//! Adversarial closure contract for the SPIR-V backend.
//!
//! The obligations live in `vyre_driver::hostile_input_closure`; this target
//! says which backend owes them and how to reach it. Reaching the backend means
//! acquiring it, which needs a Vulkan device, so the target is declared
//! `device-tests` and runs on the runner that has one. It used to print and
//! return when acquisition failed, which made it pass on every CPU runner
//! having asserted nothing at all.

#![cfg(feature = "device-tests")]
#![forbid(unsafe_code)]

use vyre_driver::hostile_input_closure::{
    assert_trailing_inputs_rejected, assert_zero_workgroup_rejected,
};
use vyre_driver_spirv::SpirvBackendRegistration;

#[test]
fn adversarial_registry_closure_hostile_inputs() {
    let backend = SpirvBackendRegistration::acquire()
        .expect("a device-tests runner has a Vulkan compute device; a probe failure is a configuration failure");
    assert_zero_workgroup_rejected(&backend, "spirv");
    assert_trailing_inputs_rejected(&backend, "spirv");
}
