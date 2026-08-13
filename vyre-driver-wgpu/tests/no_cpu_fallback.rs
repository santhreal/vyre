//! Failure-oriented tests: no fake CPU fallback on GPU machines.
//!
//! If a real GPU is present, the backend must bind to it. CPU or
//! "Other" adapters must be rejected with an actionable error.

#![allow(clippy::needless_range_loop)]

use vyre_driver_wgpu::WgpuBackend;

#[test]
fn successful_acquisition_means_non_cpu_adapter() {
    let backend = WgpuBackend::acquire().expect(
        "WgpuBackend::acquire failed on a machine that must have a GPU. \
         Fix: inspect driver visibility and adapter probing; this must not silently skip.",
    );
    let info = backend.adapter_info();
    assert!(
        !matches!(info.device_type, wgpu::DeviceType::Cpu | wgpu::DeviceType::Other),
        "Fix: WgpuBackend must never silently fall back to a CPU adapter on a machine with GPU support. \
         Adapter `{}` has type {:?}.",
        info.name,
        info.device_type
    );
}

#[test]
fn default_acquisition_prefers_discrete_gpu_when_enumerable() {
    let backend = WgpuBackend::acquire().expect(
        "WgpuBackend::acquire failed on a machine that must have a GPU. \
         Fix: inspect driver visibility and adapter probing; this must not silently skip.",
    );
    let adapters = vyre_driver_wgpu::runtime::device::enumerate_adapters();
    let has_discrete = adapters
        .iter()
        .any(|adapter| adapter.device_type == wgpu::DeviceType::DiscreteGpu);
    if has_discrete {
        assert_eq!(
            backend.adapter_info().device_type,
            wgpu::DeviceType::DiscreteGpu,
            "Fix: default WgpuBackend acquisition must prefer an enumerable discrete GPU over CPU/integrated adapters."
        );
    }
}

#[test]
fn backend_error_on_missing_gpu_is_actionable() {
    // If there's no compatible GPU, acquire() must fail with an actionable error.
    // If there IS a GPU, this test trivially passes.
    if let Err(e) = WgpuBackend::acquire() {
        let msg = e.to_string();
        assert!(
            msg.contains("Fix:"),
            "Fix: headless backend error must be actionable, got: {msg}"
        );
        assert!(
            msg.contains("adapter") || msg.contains("GPU") || msg.contains("driver"),
            "Fix: headless error must mention adapters, GPU, or driver so the user knows where to look, got: {msg}"
        );
    }
}

#[test]
fn backend_error_lists_probed_adapters() {
    // When acquisition fails, the error should enumerate what was probed
    // so the user can diagnose driver / visibility issues.
    if let Err(e) = WgpuBackend::acquire() {
        let msg = e.to_string();
        assert!(
            msg.contains("Probed adapters") || msg.contains("no compatible GPU adapter"),
            "Fix: backend error should list probed adapters or clearly state none were found, got: {msg}"
        );
    }
}
