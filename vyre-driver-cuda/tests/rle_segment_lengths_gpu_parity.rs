//! Parity test: `vyre_libs::decode::rle_segment_lengths` on CUDA matches
//! its CPU reference for packed run lengths and values, across block boundaries.

#![cfg(feature = "device-tests")]
#![cfg(test)]

mod harness;

use harness::{bytes_u32, u32_bytes, with_live_backend};
use vyre_driver::DispatchConfig;
use vyre_libs::decode::rle_segment_lengths::{rle_segment_lengths, MAX_SEGMENT_LENGTH};
use vyre_reference::composition_witness::rle_segment_lengths_witness as rle_segment_lengths_cpu;

fn run_rle(segments: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let count = segments.len() as u32;
    let program = rle_segment_lengths(count);
    let inputs: Vec<Vec<u8>> = vec![
        u32_bytes(segments),
        vec![0u8; count as usize * 4],
        vec![0u8; count as usize * 4],
    ];
    let mut config = DispatchConfig::default();
    config.grid_override = Some(vyre_primitives::lane_grid(count, 256));
    let outputs = with_live_backend("RLE segment lengths", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| panic!("Fix: CUDA RLE segment-length dispatch failed: {error}"))
    });
    let mut lengths = bytes_u32(&outputs[0]);
    let mut values = bytes_u32(&outputs[1]);
    lengths.truncate(count as usize);
    values.truncate(count as usize);
    (lengths, values)
}

#[test]
fn cuda_rle_segment_lengths_basic() {
    // pack (length=5, value=0xAA) and (length=10, value=0x55).
    let segments = vec![(5u32 << 8) | 0xAA, (10u32 << 8) | 0x55];
    let (cpu_lengths, cpu_values) = rle_segment_lengths_cpu(&segments);
    let (gpu_lengths, gpu_values) = run_rle(&segments);
    assert_eq!(gpu_lengths, cpu_lengths);
    assert_eq!(gpu_values, cpu_values);
    assert_eq!(gpu_lengths, vec![5, 10]);
    assert_eq!(gpu_values, vec![0xAA, 0x55]);
}

#[test]
fn cuda_rle_segment_lengths_zero_length() {
    let segments = vec![0u32, (1u32 << 8) | 0xFF];
    let (cpu_lengths, cpu_values) = rle_segment_lengths_cpu(&segments);
    let (gpu_lengths, gpu_values) = run_rle(&segments);
    assert_eq!(gpu_lengths, cpu_lengths);
    assert_eq!(gpu_values, cpu_values);
    assert_eq!(gpu_lengths, vec![0, 1]);
    assert_eq!(gpu_values, vec![0, 0xFF]);
}

#[test]
fn cuda_rle_segment_lengths_multi_block_mixed_runs() {
    let count = 1025u32;
    let mut segments = Vec::with_capacity(count as usize);
    for idx in 0..count {
        let length = match idx {
            0 => MAX_SEGMENT_LENGTH,
            255 => 0,
            256 => 1,
            511 => 4096,
            512 => 7,
            1024 => MAX_SEGMENT_LENGTH - 1,
            _ => (idx.wrapping_mul(17) ^ idx.rotate_left(3)) & 0x1FFF,
        };
        let value = (idx.wrapping_mul(37) ^ idx.rotate_right(5)) & 0xFF;
        segments.push((length << 8) | value);
    }

    let (cpu_lengths, cpu_values) = rle_segment_lengths_cpu(&segments);
    let (gpu_lengths, gpu_values) = run_rle(&segments);

    assert_eq!(vyre_primitives::lane_grid(count, 256), [5, 1, 1]);
    assert_eq!(gpu_lengths, cpu_lengths);
    assert_eq!(gpu_values, cpu_values);
    assert_eq!(gpu_lengths[0], MAX_SEGMENT_LENGTH);
    assert_eq!(gpu_lengths[255], 0);
    assert_eq!(gpu_lengths[256], 1);
    assert_eq!(gpu_lengths[1024], MAX_SEGMENT_LENGTH - 1);
    assert_eq!(
        gpu_values[512],
        (512u32.wrapping_mul(37) ^ 512u32.rotate_right(5)) & 0xFF
    );
}
