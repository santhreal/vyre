//! Repeated backend acquisition keeps the real adapter visible.
//!
//! A `wgpu::Instance` starts the Vulkan loader and dlopens every installed ICD;
//! dropping it unloads them. glibc reserves a fixed static TLS surplus for
//! dlopened libraries and does not return the NVIDIA driver's initial-exec
//! block on unload, so a process that builds one instance per backend runs out
//! after nine cycles: the tenth load of `libGLX_nvidia.so.0` fails with "cannot
//! allocate memory in static TLS block", the loader drops that ICD, and every
//! adapter enumeration afterwards reports the software rasterizer alone. A
//! conformance sweep measured exactly that shape, 9 of 349 operations
//! dispatched on a host holding an idle discrete GPU.
//!
//! The loop below is longer than that ceiling on purpose. It fails against the
//! per-acquisition instance and passes against the process-wide one.

#![cfg(feature = "device-tests")]

use vyre_driver_wgpu::runtime::has_real_gpu_adapter;
use vyre_driver_wgpu::WgpuBackend;

/// Loader cycles to run. Three times the measured ceiling of nine.
const ACQUISITIONS: usize = 27;

#[test]
fn adapter_enumeration_survives_repeated_backend_acquisition() {
    assert!(
        has_real_gpu_adapter(),
        "Fix: this test needs a real GPU adapter before the first acquisition; \
         expose one through a wgpu-supported driver."
    );

    for cycle in 1..=ACQUISITIONS {
        let backend = WgpuBackend::acquire().unwrap_or_else(|error| {
            panic!(
                "Fix: backend acquisition {cycle} of {ACQUISITIONS} failed: {error}. The loader \
                 lost the GPU ICD partway through the run, which is what one instance per \
                 acquisition costs."
            )
        });
        drop(backend);
        assert!(
            has_real_gpu_adapter(),
            "Fix: the real adapter disappeared from enumeration after acquisition {cycle} of \
             {ACQUISITIONS}; the loader unloaded the GPU ICD and could not load it again."
        );
    }
}
