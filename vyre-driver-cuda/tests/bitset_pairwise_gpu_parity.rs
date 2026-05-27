//! Parity tests for vyre-primitives bitset pairwise non-`_into` ops:
//! and, and_not, xor, plus any (reduce) and set_bit (scalar mutate).

#![cfg(test)]

mod common;

use common::{bytes_u32, live_dispatcher, u32_bytes};
use vyre::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_primitives::bitset::and::{bitset_and, cpu_ref as and_cpu};
use vyre_primitives::bitset::and_not::{bitset_and_not, cpu_ref as and_not_cpu};
use vyre_primitives::bitset::any::{bitset_any, cpu_ref as any_cpu};
use vyre_primitives::bitset::set_bit::{bitset_set_bit, cpu_ref as set_bit_cpu};
use vyre_primitives::bitset::xor::{bitset_xor, cpu_ref as xor_cpu};

fn run_pairwise<F>(backend: &CudaBackend, program_builder: F, lhs: &[u32], rhs: &[u32]) -> Vec<u32>
where
    F: FnOnce(&str, &str, &str, u32) -> vyre::Program,
{
    let words = lhs.len() as u32;
    let program = program_builder("lhs", "rhs", "out", words);
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(lhs),
        u32_bytes(rhs),
        vec![0u8; words as usize * 4],
    ];
    let mut config = DispatchConfig::default();
    let workgroup_x = 256u32;
    let grid_x = ((words + workgroup_x - 1) / workgroup_x).max(1);
    config.grid_override = Some([grid_x, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    bytes_u32(&outputs[0])
}

#[test]
fn cuda_bitset_and_parity() {
    let backend = live_dispatcher();
    let lhs = vec![0xFF00FF00u32, 0xAAAA_AAAA, 0u32, 0xFFFF_FFFF];
    let rhs = vec![0x00FFFF00u32, 0x5555_5555, 0xFFFF_FFFF, 0xCAFEBABE];
    let cpu = and_cpu(&lhs, &rhs);
    let gpu = run_pairwise(&backend, bitset_and, &lhs, &rhs);
    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_bitset_and_not_parity() {
    let backend = live_dispatcher();
    let lhs = vec![0xFFFF_FFFFu32, 0xCAFE_BABE, 0u32, 0xAAAA_AAAA];
    let rhs = vec![0x00FF_00FFu32, 0xFFFF_0000, 0xFFFF_FFFF, 0x5555_5555];
    let cpu = and_not_cpu(&lhs, &rhs);
    let gpu = run_pairwise(&backend, bitset_and_not, &lhs, &rhs);
    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_bitset_xor_parity() {
    let backend = live_dispatcher();
    let lhs = vec![0xAAAA_AAAAu32, 0u32, 0xFFFF_FFFF, 0xDEAD_BEEF];
    let rhs = vec![0x5555_5555u32, 0xFFFF_FFFF, 0u32, 0xDEAD_BEEF];
    let cpu = xor_cpu(&lhs, &rhs);
    let gpu = run_pairwise(&backend, bitset_xor, &lhs, &rhs);
    assert_eq!(gpu, cpu);
    // Self-xor should be all zeros for the last lane.
    assert_eq!(gpu[3], 0);
}

fn run_any(backend: &CudaBackend, input: &[u32]) -> u32 {
    let words = input.len() as u32;
    let program = bitset_any("input", "out", words);
    let inputs: Vec<Vec<u8>> = vec![u32_bytes(input), vec![0u8; 4]];
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    bytes_u32(&outputs[0])[0]
}

#[test]
fn cuda_bitset_any_all_zero() {
    let backend = live_dispatcher();
    let input = vec![0u32; 8];
    let cpu = any_cpu(&input);
    let gpu = run_any(&backend, &input);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, 0);
}

#[test]
fn cuda_bitset_any_first_word_set() {
    let backend = live_dispatcher();
    let input = vec![1u32, 0, 0, 0];
    let cpu = any_cpu(&input);
    let gpu = run_any(&backend, &input);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, 1);
}

#[test]
fn cuda_bitset_any_last_word_set() {
    let backend = live_dispatcher();
    let input = vec![0u32, 0, 0, 0x8000_0000];
    let cpu = any_cpu(&input);
    let gpu = run_any(&backend, &input);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, 1);
}

fn run_set_bit(backend: &CudaBackend, target: &[u32], bit_idx: u32) -> Vec<u32> {
    let words = target.len() as u32;
    let program = bitset_set_bit("target", bit_idx, words);
    let inputs: Vec<Vec<u8>> = vec![u32_bytes(target)];
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    bytes_u32(&outputs[0])
}

#[test]
fn cuda_bitset_set_bit_low() {
    let backend = live_dispatcher();
    let target = vec![0u32, 0u32];
    let mut cpu = target.clone();
    set_bit_cpu(&mut cpu, 0);
    let gpu = run_set_bit(&backend, &target, 0);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![1u32, 0u32]);
}

#[test]
fn cuda_bitset_set_bit_second_word() {
    let backend = live_dispatcher();
    let target = vec![0u32, 0u32];
    let mut cpu = target.clone();
    set_bit_cpu(&mut cpu, 33);
    let gpu = run_set_bit(&backend, &target, 33);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0u32, 0b10u32]);
}

#[test]
fn cuda_bitset_set_bit_preserves_other_bits() {
    let backend = live_dispatcher();
    let target = vec![0xFF00u32, 0xAAAA_AAAA];
    let mut cpu = target.clone();
    set_bit_cpu(&mut cpu, 4);
    let gpu = run_set_bit(&backend, &target, 4);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu[0], 0xFF10);
    assert_eq!(gpu[1], 0xAAAA_AAAA);
}
