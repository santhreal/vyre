//! WGPU parity for the device-side bitset clear primitive.

#![cfg(feature = "device-tests")]

use crate::harness;
use harness::acquire_live_backend as live_backend;
use harness::bytes_u32;

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_libs::bitset::zero::bitset_zero;

#[test]
fn wgpu_bitset_zero_parity_crosses_workgroup_lanes() {
    let backend = live_backend();
    let words = 600usize;
    let program = bitset_zero("target", words as u32);
    let mut config = DispatchConfig::default();
    config.grid_override = Some([3, 1, 1]);

    // `bitset_zero` declares `target` WriteOnly and backend-allocated, so it
    // takes no host input slot. Buffer pre-state cannot be seeded through the
    // artifact ABI, so this proves the written value and extent across a grid
    // override spanning three workgroups.
    let outputs = backend
        .dispatch(&program, &[], &config)
        .expect("Fix: WGPU bitset_zero dispatch must succeed");

    let mut gpu = bytes_u32(&outputs[0]);
    assert!(
        gpu.len() >= words,
        "Fix: bitset_zero output must cover all {words} declared words, got {}",
        gpu.len()
    );
    gpu.truncate(words);
    assert_eq!(gpu, vec![0u32; words]);
}
