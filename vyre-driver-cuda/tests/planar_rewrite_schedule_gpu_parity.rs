//! Parity test: `vyre_libs::parsing::planar_rewrite::planar_rewrite_schedule`
//! on CUDA matches its CPU reference for planar candidate scheduling.

#![cfg(test)]

mod harness;

use harness::{bytes_u32, u32_bytes, with_live_backend};
use vyre_driver::DispatchConfig;
use vyre_libs::parsing::planar_rewrite::planar_rewrite_schedule;
use vyre_reference::composition_witness::planar_rewrite_schedule_witness as reference_planar_rewrite_schedule;

fn run_planar(candidates: &[u32], h: u32, w: u32, k: u32) -> Vec<u32> {
    let cells = (h * w) as usize;
    let program = planar_rewrite_schedule("c", "ch", h, w, k);
    let inputs: Vec<Vec<u8>> = vec![u32_bytes(candidates), vec![0u8; cells * 4]];
    let mut config = DispatchConfig::default();
    let workgroup_x = 256u32;
    let grid_x = ((cells as u32 + workgroup_x - 1) / workgroup_x).max(1);
    config.grid_override = Some([grid_x, 1, 1]);
    let outputs = with_live_backend("planar rewrite schedule", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| panic!("Fix: CUDA planar-rewrite dispatch failed: {error}"))
    });
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(cells);
    out
}

#[test]
fn cuda_planar_rewrite_schedule_no_candidates() {
    let h = 3u32;
    let w = 3u32;
    let k = 1u32;
    let candidates = vec![0u32; (h * w) as usize];
    let cpu = reference_planar_rewrite_schedule(&candidates, h, w, k);
    let gpu = run_planar(&candidates, h, w, k);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0u32; (h * w) as usize]);
}

#[test]
fn cuda_planar_rewrite_schedule_isolated_candidates() {
    let h = 4u32;
    let w = 4u32;
    let k = 1u32;
    // Diagonal candidates spaced by 2  -  none touch each other within k=1.
    let mut candidates = vec![0u32; (h * w) as usize];
    candidates[0] = 1;
    candidates[10] = 1;
    let cpu = reference_planar_rewrite_schedule(&candidates, h, w, k);
    let gpu = run_planar(&candidates, h, w, k);
    assert_eq!(gpu, cpu);
}
