//! Parity test: `vyre_libs::math::submodular_greedy::argmax_of_marginals` on
//! CUDA matches its CPU reference, including the all-picked sentinel.

#![cfg(test)]

mod harness;

use harness::{bytes_u32, u32_bytes, with_live_backend};
use vyre_driver::DispatchConfig;
use vyre_libs::math::submodular_greedy::{argmax_of_marginals, NO_WINNER};
use vyre_reference::composition_witness::argmax_of_marginals_witness as argmax_of_marginals_cpu;

fn run_argmax(gains: &[u32], picked: &[u32]) -> (u32, u32) {
    assert_eq!(gains.len(), picked.len());
    let n = gains.len() as u32;
    let program = argmax_of_marginals("gains", "picked", "winner_idx", "winner_gain", n);
    // Initialize winner_idx to NO_WINNER and winner_gain to 0 so
    // atomic-max merging starts from a meaningful sentinel.
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(gains),
        u32_bytes(picked),
        u32_bytes(&[NO_WINNER]),
        vec![0u8; 4],
    ];
    let mut config = DispatchConfig::default();
    let workgroup_x = 256u32;
    let grid_x = ((n + workgroup_x - 1) / workgroup_x).max(1);
    config.grid_override = Some([grid_x, 1, 1]);
    let outputs = with_live_backend("argmax of marginals", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| {
                panic!("Fix: CUDA argmax-of-marginals dispatch failed: {error}")
            })
    });
    let winner_idx = bytes_u32(&outputs[0])[0];
    let winner_gain = bytes_u32(&outputs[1])[0];
    (winner_idx, winner_gain)
}

#[test]
fn cuda_argmax_of_marginals_picks_highest_unpicked() {
    let gains = vec![10u32, 50, 20, 99, 5];
    let picked = vec![0u32, 0, 0, 0, 0];
    let (cpu_idx, cpu_gain) = argmax_of_marginals_cpu(&gains, &picked);
    let (gpu_idx, gpu_gain) = run_argmax(&gains, &picked);
    assert_eq!((gpu_idx, gpu_gain), (cpu_idx, cpu_gain));
    assert_eq!(gpu_idx, 3);
    assert_eq!(gpu_gain, 99);
}

#[test]
fn cuda_argmax_of_marginals_skips_picked() {
    let gains = vec![10u32, 50, 20, 99, 5];
    // Index 3 is already picked  -  winner shifts to next-highest (50 @ 1).
    let picked = vec![0u32, 0, 0, 1, 0];
    let (cpu_idx, cpu_gain) = argmax_of_marginals_cpu(&gains, &picked);
    let (gpu_idx, gpu_gain) = run_argmax(&gains, &picked);
    assert_eq!((gpu_idx, gpu_gain), (cpu_idx, cpu_gain));
    assert_eq!(gpu_gain, 50);
}

#[test]
fn cuda_argmax_of_marginals_all_picked() {
    let gains = vec![10u32, 50, 20];
    let picked = vec![1u32, 1, 1];
    let (cpu_idx, cpu_gain) = argmax_of_marginals_cpu(&gains, &picked);
    let (gpu_idx, gpu_gain) = run_argmax(&gains, &picked);
    assert_eq!((gpu_idx, gpu_gain), (cpu_idx, cpu_gain));
    assert_eq!(gpu_idx, NO_WINNER);
    assert_eq!(gpu_gain, 0);
}
