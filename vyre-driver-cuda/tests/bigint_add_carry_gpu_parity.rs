//! Parity test: `vyre_libs::math::bigint_add_carry` on CUDA matches its
//! CPU reference for per-limb sums and carries, across block boundaries.

#![cfg(test)]

mod harness;

use harness::{bytes_u32, u32_bytes, with_live_backend};
use vyre_driver::DispatchConfig;
use vyre_libs::math::bigint_add_carry::{
    bigint_add_carry, bigint_add_carry_cpu, bigint_add_carry_dispatch_grid, BINDING_A_IN,
    BINDING_B_IN, BINDING_CARRY_PARTIAL_OUT, BINDING_SUM_PARTIAL_OUT,
};

fn run_bigint_add_carry(a: &[u32], b: &[u32]) -> (Vec<u32>, Vec<u32>) {
    assert_eq!(a.len(), b.len());
    let limb_count = a.len() as u32;
    let program = bigint_add_carry(limb_count);
    let mut inputs: Vec<Vec<u8>> = vec![Vec::new(); 4];
    inputs[BINDING_A_IN as usize] = u32_bytes(a);
    inputs[BINDING_B_IN as usize] = u32_bytes(b);
    inputs[BINDING_SUM_PARTIAL_OUT as usize] = vec![0u8; limb_count as usize * 4];
    inputs[BINDING_CARRY_PARTIAL_OUT as usize] = vec![0u8; limb_count as usize * 4];
    let mut config = DispatchConfig::default();
    config.grid_override = Some(bigint_add_carry_dispatch_grid(limb_count));
    let outputs = with_live_backend("bigint add carry", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| panic!("Fix: CUDA bigint add-carry dispatch failed: {error}"))
    });
    // RW outputs are returned in declaration order; A and B are ReadOnly,
    // so outputs[0]=sum_partial, outputs[1]=carry_partial.
    let mut sum = bytes_u32(&outputs[0]);
    let mut carry = bytes_u32(&outputs[1]);
    sum.truncate(limb_count as usize);
    carry.truncate(limb_count as usize);
    (sum, carry)
}

#[test]
fn cuda_bigint_add_carry_no_overflow() {
    let a = vec![1u32, 2, 3, 4];
    let b = vec![10u32, 20, 30, 40];
    let (cpu_sum, cpu_carry) = bigint_add_carry_cpu(&a, &b).expect("ok");
    let (gpu_sum, gpu_carry) = run_bigint_add_carry(&a, &b);
    assert_eq!(gpu_sum, cpu_sum);
    assert_eq!(gpu_carry, cpu_carry);
    assert_eq!(gpu_carry, vec![0u32; 4]);
}

#[test]
fn cuda_bigint_add_carry_with_overflow() {
    // Each limb wraps: 0xFFFF_FFFF + 1 → carry.
    let a = vec![u32::MAX, u32::MAX, 0u32];
    let b = vec![1u32, 1u32, 1u32];
    let (cpu_sum, cpu_carry) = bigint_add_carry_cpu(&a, &b).expect("ok");
    let (gpu_sum, gpu_carry) = run_bigint_add_carry(&a, &b);
    assert_eq!(gpu_sum, cpu_sum);
    assert_eq!(gpu_carry, cpu_carry);
    assert_eq!(gpu_sum, vec![0, 0, 1]);
    assert_eq!(gpu_carry, vec![1, 1, 0]);
}

#[test]
fn cuda_bigint_add_carry_zero_operands() {
    let a = vec![0u32; 5];
    let b = vec![0u32; 5];
    let (cpu_sum, cpu_carry) = bigint_add_carry_cpu(&a, &b).expect("ok");
    let (gpu_sum, gpu_carry) = run_bigint_add_carry(&a, &b);
    assert_eq!(gpu_sum, cpu_sum);
    assert_eq!(gpu_carry, cpu_carry);
    assert_eq!(gpu_sum, vec![0u32; 5]);
}

#[test]
fn cuda_bigint_add_carry_multi_block_overflow_pattern() {
    let limb_count = 1025u32;
    let mut a = Vec::with_capacity(limb_count as usize);
    let mut b = Vec::with_capacity(limb_count as usize);
    for idx in 0..limb_count {
        let (left, right) = match idx {
            0 => (u32::MAX, 1),
            255 => (0, 0),
            256 => (u32::MAX, u32::MAX),
            511 => (0x8000_0000, 0x8000_0000),
            512 => (7, 9),
            1024 => (u32::MAX, 2),
            _ if idx % 5 == 0 => (u32::MAX, idx),
            _ if idx % 7 == 0 => (0x8000_0000, 0x8000_0000),
            _ => (
                idx.wrapping_mul(0x9E37_79B9),
                idx.rotate_left(11).wrapping_mul(0x85EB_CA6B),
            ),
        };
        a.push(left);
        b.push(right);
    }

    let (cpu_sum, cpu_carry) = bigint_add_carry_cpu(&a, &b).expect("ok");
    let (gpu_sum, gpu_carry) = run_bigint_add_carry(&a, &b);

    assert_eq!(bigint_add_carry_dispatch_grid(limb_count), [5, 1, 1]);
    assert_eq!(gpu_sum, cpu_sum);
    assert_eq!(gpu_carry, cpu_carry);
    assert_eq!(gpu_sum[0], 0);
    assert_eq!(gpu_carry[0], 1);
    assert_eq!(gpu_sum[256], u32::MAX - 1);
    assert_eq!(gpu_carry[256], 1);
    assert_eq!(gpu_sum[512], 16);
    assert_eq!(gpu_carry[512], 0);
    assert_eq!(gpu_sum[1024], 1);
    assert_eq!(gpu_carry[1024], 1);
}
