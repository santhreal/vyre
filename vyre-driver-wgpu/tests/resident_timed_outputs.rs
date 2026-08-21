//! Resident timed-dispatch output contracts for the WGPU backend.
//!
//! The asynchronous overlap statement is the shared contract in
//! `tests/support/resident_async_overlap_contract.rs`. The timed readback below
//! is WGPU's own: it pins that a resident timed dispatch returns exactly the
//! public read-write outputs and attributes GPU time to the WGPU timestamp query.

#![cfg(feature = "device-tests")]

use vyre_driver::VyreBackend;

#[path = "../../tests/support/resident_async_overlap_contract.rs"]
mod resident_async_overlap_contract;
use resident_async_overlap_contract::{
    add_program, assert_resident_async_slots_retire_independently, DeviceTiming,
};

fn acquire() -> vyre_driver_wgpu::WgpuBackend {
    vyre_driver_wgpu::WgpuBackend::acquire()
        .expect("Fix: WGPU resident regression tests require a live GPU backend.")
}

#[test]
fn resident_timed_dispatch_returns_public_readwrite_outputs() {
    let backend = acquire();
    let program = add_program();

    let out = backend
        .allocate_resident(4)
        .expect("Fix: WGPU must support resident output allocation.");
    let input = backend
        .allocate_resident(4)
        .expect("Fix: WGPU must support resident input allocation.");
    let result = (|| {
        backend.upload_resident(&out, &[0, 0, 0, 0])?;
        backend.upload_resident(&input, &37u32.to_le_bytes())?;
        let timed = backend.dispatch_resident_timed(
            &program,
            &[out.clone(), input.clone()],
            &vyre_driver::DispatchConfig::default(),
        )?;
        assert_eq!(
            timed.outputs.len(),
            1,
            "resident timed dispatch must return public ReadWrite outputs"
        );
        assert_eq!(timed.outputs[0], 42u32.to_le_bytes());
        assert!(
            timed.device_ns.unwrap_or_default() > 0,
            "Fix: WGPU resident timed dispatch must report GPU timestamp device_ns so release benchmarks do not fall back to readback wall time."
        );
        Ok::<(), vyre_driver::BackendError>(())
    })();
    let free_out = backend.free_resident(out);
    let free_input = backend.free_resident(input);
    result.expect("Fix: resident timed dispatch must execute and read back outputs.");
    free_out.expect("Fix: WGPU resident output cleanup must succeed.");
    free_input.expect("Fix: WGPU resident input cleanup must succeed.");
}

#[test]
fn resident_async_dispatch_retires_two_independent_slots() {
    assert_resident_async_slots_retire_independently(&acquire(), "WGPU", DeviceTiming::Reported);
}
