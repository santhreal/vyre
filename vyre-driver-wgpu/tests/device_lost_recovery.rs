//! Device-loss recovery.
//!
//! See `contracts/release.md`. After a simulated device-lost event
//! the backend must (a) report `device_lost() == true`, (b) recover
//! via `try_recover() -> Ok(())`, (c) accept the next dispatch
//! successfully.

use vyre_driver::VyreBackend;
use vyre_driver_wgpu::WgpuBackend;

#[test]
fn device_lost_recovery_round_trip() {
    let backend = WgpuBackend::acquire().expect("Fix: GPU must be available for recovery test");

    // Simulate device loss through the backend test hook. Recovery must
    // invalidate device-local caches and reacquire the same adapter identity.
    backend
        .force_device_lost()
        .expect("Fix: test hook must invalidate the cached device");

    assert!(
        backend.device_lost(),
        "after force_device_lost the probe must return true"
    );

    backend
        .try_recover()
        .expect("device-loss recovery: try_recover must succeed");

    assert!(
        !backend.device_lost(),
        "after try_recover the device_lost probe must return false"
    );

    // The backend must dispatch successfully after recovery.
    let program = vyre::Program::empty();
    let _ = backend
        .dispatch(&program, &[], &vyre_driver::DispatchConfig::default())
        .expect("Fix: dispatch must succeed after device recovery");
}
