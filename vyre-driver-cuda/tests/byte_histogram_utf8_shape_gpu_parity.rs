//! Parity test: vyre-primitives byte_histogram_256 + utf8_shape_counts
//! match their reference oracles.

#![cfg(test)]

mod common;

use common::{bytes_u32, live_dispatcher, u32_bytes};
use vyre::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_primitives::text::byte_histogram::{byte_histogram_256, reference_byte_histogram};
use vyre_primitives::text::utf8_shape_counts::{reference_utf8_shape_counts, utf8_shape_counts};

fn bytes_to_u32_per_lane(source: &[u8]) -> Vec<u32> {
    source.iter().map(|&b| b as u32).collect()
}

fn run_histogram(backend: &CudaBackend, source: &[u8]) -> Vec<u32> {
    let n = source.len() as u32;
    let program = byte_histogram_256("source", "histogram", n);
    let inputs: Vec<Vec<u8>> = vec![u32_bytes(&bytes_to_u32_per_lane(source))];
    let mut config = DispatchConfig::default();
    // Histogram kernel: 256 lanes per workgroup (256 buckets).
    config.grid_override = Some([1, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(256);
    out
}

#[test]
fn cuda_byte_histogram_simple() {
    let backend = live_dispatcher();
    let source = b"abacab";
    let cpu = reference_byte_histogram(source);
    let gpu = run_histogram(&backend, source);
    assert_eq!(gpu, cpu.to_vec());
    assert_eq!(gpu[b'a' as usize], 3);
    assert_eq!(gpu[b'b' as usize], 2);
    assert_eq!(gpu[b'c' as usize], 1);
}

#[test]
fn cuda_byte_histogram_utf8_bytes() {
    let backend = live_dispatcher();
    // 'é' = 0xC3 0xA9
    let source = &[b'a', b'b', b'a', 0xC3, 0xA9];
    let cpu = reference_byte_histogram(source);
    let gpu = run_histogram(&backend, source);
    assert_eq!(gpu, cpu.to_vec());
    assert_eq!(gpu[b'a' as usize], 2);
    assert_eq!(gpu[0xC3], 1);
    assert_eq!(gpu[0xA9], 1);
}

#[test]
fn cuda_byte_histogram_empty() {
    let backend = live_dispatcher();
    let source: &[u8] = &[];
    let cpu = reference_byte_histogram(source);
    // Empty input → all zeros.
    let _ = cpu;
    let n = 1u32; // need at least 1-element buffer
    let dummy: Vec<u8> = vec![0u8; 4]; // one u32 = 0
    let program = byte_histogram_256("source", "histogram", 0);
    let inputs: Vec<Vec<u8>> = vec![dummy];
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(256);
    let _ = n;
    for v in out {
        assert_eq!(v, 0);
    }
}

fn run_shape_counts(backend: &CudaBackend, histogram: &[u32; 256]) -> (u32, u32) {
    let program = utf8_shape_counts("histogram", "out");
    let inputs: Vec<Vec<u8>> = vec![u32_bytes(histogram)];
    let mut config = DispatchConfig::default();
    config.grid_override = Some([1, 1, 1]);
    let outputs = backend
        .dispatch(&program, &inputs, &config)
        .expect("dispatch");
    let out = bytes_u32(&outputs[0]);
    (out[0], out[1])
}

#[test]
fn cuda_utf8_shape_counts_two_byte_seq() {
    let backend = live_dispatcher();
    // One 2-byte sequence: 0xC3 0xA9. Expect continuation=1, expected=1.
    let mut histogram = [0u32; 256];
    histogram[0xC3] = 1;
    histogram[0xA9] = 1;
    let cpu = reference_utf8_shape_counts(&histogram);
    let gpu = run_shape_counts(&backend, &histogram);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, (1, 1));
}

#[test]
fn cuda_utf8_shape_counts_two_two_byte_seqs() {
    let backend = live_dispatcher();
    // Two 2-byte sequences: continuation=2, expected=2.
    let mut histogram = [0u32; 256];
    histogram[0xC3] = 1;
    histogram[0xA9] = 1;
    histogram[0xC2] = 1;
    histogram[0x80] = 1;
    let cpu = reference_utf8_shape_counts(&histogram);
    let gpu = run_shape_counts(&backend, &histogram);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, (2, 2));
}

#[test]
fn cuda_utf8_shape_counts_three_byte_seq() {
    let backend = live_dispatcher();
    // 3-byte sequence (0xE0..0xEF): expected += count*2 = 2 continuations.
    let mut histogram = [0u32; 256];
    histogram[0xE2] = 1;
    histogram[0x82] = 1;
    histogram[0xAC] = 1;
    let cpu = reference_utf8_shape_counts(&histogram);
    let gpu = run_shape_counts(&backend, &histogram);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, (2, 2));
}

#[test]
fn cuda_utf8_shape_counts_four_byte_seq() {
    let backend = live_dispatcher();
    // 4-byte sequence (0xF0..0xF4): expected += count*3 = 3 continuations.
    let mut histogram = [0u32; 256];
    histogram[0xF0] = 1;
    histogram[0x9F] = 1;
    histogram[0x98] = 1;
    histogram[0x80] = 1;
    let cpu = reference_utf8_shape_counts(&histogram);
    let gpu = run_shape_counts(&backend, &histogram);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, (3, 3));
}

#[test]
fn cuda_utf8_shape_counts_ascii_only_zero() {
    let backend = live_dispatcher();
    let mut histogram = [0u32; 256];
    for b in 0u8..0x80 {
        histogram[b as usize] = 1;
    }
    let cpu = reference_utf8_shape_counts(&histogram);
    let gpu = run_shape_counts(&backend, &histogram);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, (0, 0));
}
