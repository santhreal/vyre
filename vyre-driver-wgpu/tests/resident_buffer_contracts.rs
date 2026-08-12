//! WGPU backend resident-buffer API contracts.

use vyre_driver::{Resource, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;

fn backend() -> WgpuBackend {
    WgpuBackend::new().expect(
        "Fix: live WGPU backend required for resident-buffer contracts; missing GPU is a configuration bug.",
    )
}

#[path = "resident_buffer_contracts/lifecycle_range_contracts.rs"]
mod lifecycle_range_contracts;
#[path = "resident_buffer_contracts/ranged_batch_contracts.rs"]
mod ranged_batch_contracts;
#[path = "resident_buffer_contracts/validation_atomicity_contracts.rs"]
mod validation_atomicity_contracts;
