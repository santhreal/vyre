//! Contracts for `vyre_driver_wgpu::runtime::indirect`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver_wgpu::runtime::indirect::{IndirectArgs, INDIRECT_ARGS_BYTES};

#[test]
fn args_bytes_is_twelve() {
    assert_eq!(INDIRECT_ARGS_BYTES, 12);
}

// Note: tests that actually construct IndirectArgs require a
// real wgpu::Buffer and hence a GPU. The full dispatch path is
// exercised from vyre-wgpu integration tests (`tests/indirect_dispatch.rs`).
