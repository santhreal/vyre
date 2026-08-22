//! WGPU backend resident-buffer API contracts.

#![cfg(feature = "device-tests")]

use vyre_driver::{Resource, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;

fn backend() -> WgpuBackend {
    WgpuBackend::new().expect(
        "Fix: live WGPU backend required for resident-buffer contracts; missing GPU is a configuration bug.",
    )
}

fn alloc_pair(backend: &WgpuBackend, bytes: u64) -> (Resource, Resource) {
    let size = usize::try_from(bytes).expect("test size fits usize");
    let first = backend
        .allocate_resident(size)
        .expect("WGPU backend must allocate first resident buffer");
    let second = backend
        .allocate_resident(size)
        .expect("WGPU backend must allocate second resident buffer");
    (first, second)
}

fn free_pair(backend: &WgpuBackend, first: Resource, second: Resource) {
    backend
        .free_resident(first)
        .expect("first resident buffer must free cleanly");
    backend
        .free_resident(second)
        .expect("second resident buffer must free cleanly");
}

mod lifecycle_range_contracts;
mod ranged_batch_contracts;
mod validation_atomicity_contracts;
