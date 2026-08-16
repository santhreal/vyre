//! WGPU backend resident-buffer API contracts.

use vyre_driver::{Resource, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;

fn backend() -> WgpuBackend {
    WgpuBackend::new().expect(
        "Fix: live WGPU backend required for resident-buffer contracts; missing GPU is a configuration bug.",
    )
}

fn alloc_pair(backend: &WgpuBackend, bytes: u64) -> (Resource, Resource) {
    let first = backend
        .allocate_resident(bytes)
        .expect("WGPU backend must allocate first resident buffer");
    let second = backend
        .allocate_resident(bytes)
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

#[path = "resident_buffer_contracts/lifecycle_range_contracts.rs"]
mod lifecycle_range_contracts;
#[path = "resident_buffer_contracts/ranged_batch_contracts.rs"]
mod ranged_batch_contracts;
#[path = "resident_buffer_contracts/validation_atomicity_contracts.rs"]
mod validation_atomicity_contracts;
