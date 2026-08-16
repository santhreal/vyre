//! Parity test: `vyre_primitives::math::interval::interval_merge_program` on CUDA
//! matches its CPU reference for per-lane interval hulls.

#![cfg(test)]

mod harness;

use harness::{bytes_u32, u32_bytes, with_live_backend};
use vyre_driver::DispatchConfig;
use vyre_primitives::math::interval::{cpu_interval_merge, interval_merge_program};

fn run_interval_merge(
    mins_a: &[u32],
    maxs_a: &[u32],
    mins_b: &[u32],
    maxs_b: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let lane_count = mins_a.len() as u32;
    let program = interval_merge_program(
        "mins_a", "maxs_a", "mins_b", "maxs_b", "mins_out", "maxs_out", lane_count,
    );
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(mins_a),
        u32_bytes(maxs_a),
        u32_bytes(mins_b),
        u32_bytes(maxs_b),
        vec![0u8; lane_count as usize * 4],
        vec![0u8; lane_count as usize * 4],
    ];
    let mut config = DispatchConfig::default();
    let workgroup_x = 256u32;
    let grid_x = ((lane_count + workgroup_x - 1) / workgroup_x).max(1);
    config.grid_override = Some([grid_x, 1, 1]);
    let outputs = with_live_backend("interval merge", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| panic!("Fix: CUDA interval merge dispatch failed: {error}"))
    });
    let mut mins = bytes_u32(&outputs[0]);
    let mut maxs = bytes_u32(&outputs[1]);
    mins.truncate(lane_count as usize);
    maxs.truncate(lane_count as usize);
    (mins, maxs)
}

#[test]
fn cuda_interval_merge_basic() {
    let mins_a = vec![10u32, 0, 7];
    let maxs_a = vec![20u32, 3, 9];
    let mins_b = vec![4u32, 2, 8];
    let maxs_b = vec![18u32, 5, 12];
    let (cpu_mins, cpu_maxs) = cpu_interval_merge(&mins_a, &maxs_a, &mins_b, &maxs_b);
    let (gpu_mins, gpu_maxs) = run_interval_merge(&mins_a, &maxs_a, &mins_b, &maxs_b);
    assert_eq!(gpu_mins, cpu_mins);
    assert_eq!(gpu_maxs, cpu_maxs);
    assert_eq!(gpu_mins, vec![4, 0, 7]);
    assert_eq!(gpu_maxs, vec![20, 5, 12]);
}

#[test]
fn cuda_interval_merge_a_dominates() {
    // a fully contains b on every lane.
    let mins_a = vec![0u32, 0, 0];
    let maxs_a = vec![100u32, 100, 100];
    let mins_b = vec![10u32, 20, 30];
    let maxs_b = vec![15u32, 25, 35];
    let (cpu_mins, cpu_maxs) = cpu_interval_merge(&mins_a, &maxs_a, &mins_b, &maxs_b);
    let (gpu_mins, gpu_maxs) = run_interval_merge(&mins_a, &maxs_a, &mins_b, &maxs_b);
    assert_eq!(gpu_mins, cpu_mins);
    assert_eq!(gpu_maxs, cpu_maxs);
    assert_eq!(gpu_mins, vec![0; 3]);
    assert_eq!(gpu_maxs, vec![100; 3]);
}
