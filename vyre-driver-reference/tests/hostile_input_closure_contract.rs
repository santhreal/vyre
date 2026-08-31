//! Adversarial closure contract for the cpu-ref backend.
//!
//! The obligations themselves live in `vyre_driver::hostile_input_closure`,
//! because they are the same obligations for every backend: a hostile byte
//! slice either dispatches or fails with an actionable message, and extra
//! trailing input buffers are rejected rather than ignored. This target says
//! which backend owes them.

#![forbid(unsafe_code)]

use vyre_driver::hostile_input_closure::{
    assert_hostile_bytes_stay_actionable, assert_trailing_inputs_rejected,
};
use vyre_driver_reference::CpuRefBackend;

#[test]
fn adversarial_registry_closure_hostile_inputs() {
    assert_hostile_bytes_stay_actionable(&CpuRefBackend, "cpu-ref");
    assert_trailing_inputs_rejected(&CpuRefBackend, "cpu-ref");
}
