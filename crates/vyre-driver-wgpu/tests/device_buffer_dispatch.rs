//! Live contracts for `VyreBackend::dispatch_with_device_buffers` on wgpu.
//!
//! The dispatch writes into caller-owned device buffers and performs no
//! host readback; the bytes reach the host only through
//! `download_device_buffer`. These tests prove the outputs land in the
//! device buffer, that repeated dispatches over the same buffers observe
//! fresh uploads, and that an upload larger than the allocation fails.

#![cfg(feature = "device-tests")]

use crate::harness;
use harness::{add_one_input, add_one_program, shared_live_backend};

use vyre_driver::{DispatchConfig, VyreBackend};

const WORDS: u32 = 257;

fn decode_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn device_buffer_dispatch_writes_outputs_into_device_buffer() {
    let backend = shared_live_backend();
    let program = add_one_program(WORDS);
    let byte_len = WORDS as usize * 4;
    let config = DispatchConfig::default();

    let mut input = backend
        .allocate_device_buffer(byte_len)
        .expect("Fix: wgpu must allocate the input device buffer.");
    let mut output = backend
        .allocate_device_buffer(byte_len)
        .expect("Fix: wgpu must allocate the output device buffer.");

    for round in 0..3u32 {
        let host_input: Vec<u8> = (0..WORDS)
            .map(|word| word.wrapping_mul(7).wrapping_add(round * 1_000))
            .flat_map(u32::to_le_bytes)
            .collect();
        backend
            .upload_device_buffer(&mut *input, &host_input)
            .expect("Fix: wgpu must upload into an owned device buffer.");
        backend
            .dispatch_with_device_buffers(&program, &[&*input], &mut [&mut *output], &config)
            .expect("Fix: device-buffer dispatch must succeed on a live adapter.");
        let observed = decode_words(
            &backend
                .download_device_buffer(&*output)
                .expect("Fix: wgpu must download an owned device buffer."),
        );
        let expected: Vec<u32> = decode_words(&host_input)
            .into_iter()
            .map(|word| word.wrapping_add(1))
            .collect();
        assert_eq!(
            observed, expected,
            "round {round}: the output device buffer must hold input + 1 for every word"
        );
    }

    backend
        .free_device_buffer(input)
        .expect("Fix: wgpu must free the input device buffer.");
    backend
        .free_device_buffer(output)
        .expect("Fix: wgpu must free the output device buffer.");
}

#[test]
fn device_buffer_upload_larger_than_allocation_is_rejected() {
    let backend = shared_live_backend();
    let byte_len = WORDS as usize * 4;
    let mut input = backend
        .allocate_device_buffer(byte_len)
        .expect("Fix: wgpu must allocate the input device buffer.");
    let oversized = add_one_input(WORDS + 1);
    let error = backend
        .upload_device_buffer(&mut *input, &oversized)
        .expect_err("an upload larger than the allocation must fail");
    assert!(
        error.to_string().contains("exceeds logical length"),
        "the oversized upload error must state the overflow: {error}"
    );
    backend
        .free_device_buffer(input)
        .expect("Fix: wgpu must free the input device buffer.");
}
