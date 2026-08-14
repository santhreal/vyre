//! Parity test: `vyre_primitives::bitset::popcount::bitset_popcount` on CUDA
//! matches its CPU reference per word.

#![cfg(test)]

mod common;

use common::{bytes_u32, u32_bytes, with_live_backend};
use vyre_driver::DispatchConfig;
use vyre_primitives::bitset::popcount::{bitset_popcount, cpu_ref as popcount_cpu};

fn run_popcount(input: &[u32]) -> Vec<u32> {
    let words = input.len() as u32;
    let program = bitset_popcount("input", "count_words", words);
    let inputs: Vec<Vec<u8>> = vec![u32_bytes(input), vec![0u8; words as usize * 4]];
    let mut config = DispatchConfig::default();
    let workgroup_x = 256u32;
    let grid_x = ((words + workgroup_x - 1) / workgroup_x).max(1);
    config.grid_override = Some([grid_x, 1, 1]);
    let outputs = with_live_backend("bitset popcount batch", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| panic!("Fix: CUDA bitset-popcount dispatch failed: {error}"))
    });
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(words as usize);
    out
}

#[test]
fn cuda_bitset_popcount_basic() {
    let input = vec![0xFFFF_FFFFu32, 0u32, 0b1010_1010_u32, 0xAA55u32];
    let cpu = popcount_cpu(&input);
    let gpu = run_popcount(&input);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![32, 0, 4, 8]);
}

#[test]
fn cuda_bitset_popcount_all_zero() {
    let input = vec![0u32; 16];
    let cpu = popcount_cpu(&input);
    let gpu = run_popcount(&input);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0u32; 16]);
}
