//! Cancellation deadline and post-cancellation device usability contracts.

mod harness;
use harness::long_running_program;

use std::time::{Duration, Instant};
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
#[test]
fn dispatch_cancels_within_deadline() {
    let backend = WgpuBackend::acquire().expect("Fix: GPU required for pre-emption test");
    let program = long_running_program();
    let mut config = DispatchConfig::default();
    config.timeout = Some(Duration::from_millis(100));
    config.label = Some("dispatch-preemption".to_string());

    let start = Instant::now();
    let result = backend.dispatch(&program, &[], &config);
    let elapsed = start.elapsed();

    assert!(
        result.is_err(),
        "dispatch preemption: dispatch must return Err on timeout, got Ok"
    );
    // wgpu/Vulkan does not support mid-kernel preemption, so cancellation
    // can only check at queue boundaries. Allow 2s past the 100ms timeout
    // for the in-flight kernel to drain plus the cancellation observation
    // window. True GPU pre-emption is a separate roadmap item; this test
    // verifies the contract that timeout DOES return Err in bounded
    // wall-clock, not that it kills the kernel mid-execution.
    assert!(
        elapsed < Duration::from_secs(2),
        "dispatch preemption: cancellation must complete within 2s of the deadline; took {:?}",
        elapsed
    );

    // After cancellation the device must accept a fresh dispatch.
    let quick = vyre::Program::empty();
    let _ = backend
        .dispatch(&quick, &[], &DispatchConfig::default())
        .expect("Fix: device must be usable after cancelled dispatch");
}
