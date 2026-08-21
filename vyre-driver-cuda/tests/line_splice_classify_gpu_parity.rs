//! Parity test: `vyre_libs::parsing::line_splice_classify` on CUDA
//! matches its CPU reference across packed-u32 and raw-u8 source layouts,
//! including word- and workgroup-boundary splices.

#![cfg(feature = "device-tests")]
#![cfg(test)]

mod harness;
#[path = "harness/line_splice_generated_corpus.rs"]
mod line_splice_generated_corpus;

use harness::{bytes_u32, u32_bytes, with_live_backend};
use vyre_driver::DispatchConfig;
use vyre_libs::parsing::line_splice_classify::{
    line_splice_classify, line_splice_classify_dispatch_grid, line_splice_classify_u8,
};
use vyre_reference::composition_witness::line_splice_classify_witness as reference_line_splice_classify;

fn pack_bytes(bytes: &[u8]) -> Vec<u32> {
    let mut padded = bytes.to_vec();
    if padded.is_empty() {
        padded.push(0);
    }
    while padded.len() % 4 != 0 {
        padded.push(0);
    }
    padded
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn run_line_splice(source: &[u8]) -> Vec<u32> {
    let byte_count = source.len() as u32;
    let words = pack_bytes(source);
    let program = line_splice_classify(byte_count);
    let inputs: Vec<Vec<u8>> = vec![u32_bytes(&words), vec![0u8; byte_count.max(1) as usize * 4]];
    let mut config = DispatchConfig::default();
    config.grid_override = Some(line_splice_classify_dispatch_grid(byte_count));
    let outputs = with_live_backend("line splice classify", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| {
                panic!("Fix: CUDA line-splice classify dispatch failed: {error}")
            })
    });
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(byte_count as usize);
    out
}

fn run_line_splice_u8(source: &[u8]) -> Vec<u32> {
    let byte_count = source.len() as u32;
    let program = line_splice_classify_u8(byte_count);
    let inputs: Vec<Vec<u8>> = vec![source.to_vec(), vec![0u8; byte_count.max(1) as usize * 4]];
    let mut config = DispatchConfig::default();
    config.grid_override = Some(line_splice_classify_dispatch_grid(byte_count));
    let outputs = with_live_backend("raw-u8 line splice classify", |backend| {
        backend
            .dispatch(&program, &inputs, &config)
            .unwrap_or_else(|error| {
                panic!("Fix: CUDA raw-u8 line-splice classify dispatch failed: {error}")
            })
    });
    let mut out = bytes_u32(&outputs[0]);
    out.truncate(byte_count as usize);
    out
}

#[test]
fn cuda_line_splice_classify_keeps_plain_text() {
    let source = b"abcd";
    let cpu = reference_line_splice_classify(source);
    let gpu = run_line_splice(source);
    let gpu_u8 = run_line_splice_u8(source);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu_u8, cpu);
    assert_eq!(gpu, vec![1u32, 1, 1, 1]);
    assert_eq!(gpu_u8, vec![1u32, 1, 1, 1]);
}

#[test]
fn cuda_line_splice_classify_drops_backslash_lf() {
    // "ab\\\ncd"  -  backslash + LF should be dropped (kept_mask = 0).
    let source = b"ab\\\ncd";
    let cpu = reference_line_splice_classify(source);
    let gpu = run_line_splice(source);
    let gpu_u8 = run_line_splice_u8(source);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu_u8, cpu);
}

#[test]
fn cuda_line_splice_classify_empty_input() {
    let source = b"";
    let cpu = reference_line_splice_classify(source);
    let gpu = run_line_splice(source);
    let gpu_u8 = run_line_splice_u8(source);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu_u8, cpu);
    assert!(gpu.is_empty());
    assert!(gpu_u8.is_empty());
}

#[test]
fn cuda_line_splice_classify_drops_backslash_cr_lf() {
    let source = b"a\\\r\nb";
    let cpu = reference_line_splice_classify(source);
    let gpu = run_line_splice(source);
    let gpu_u8 = run_line_splice_u8(source);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu_u8, cpu);
    assert_eq!(gpu, vec![1, 0, 0, 0, 1]);
    assert_eq!(gpu_u8, vec![1, 0, 0, 0, 1]);
}

#[test]
fn cuda_line_splice_classify_crosses_packed_word_boundary() {
    let source = b"abc\\\nz";
    let cpu = reference_line_splice_classify(source);
    let gpu = run_line_splice(source);
    let gpu_u8 = run_line_splice_u8(source);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu_u8, cpu);
    assert_eq!(gpu, vec![1, 1, 1, 0, 0, 1]);
    assert_eq!(gpu_u8, vec![1, 1, 1, 0, 0, 1]);
}

#[test]
fn cuda_line_splice_classify_crosses_workgroup_boundary() {
    let mut source = vec![b'x'; 260];
    source[254] = b'\\';
    source[255] = b'\r';
    source[256] = b'\n';

    let cpu = reference_line_splice_classify(&source);
    let gpu = run_line_splice(&source);
    let gpu_u8 = run_line_splice_u8(&source);

    assert_eq!(gpu, cpu);
    assert_eq!(gpu_u8, cpu);
    assert_eq!(&gpu[252..258], &[1, 1, 0, 0, 0, 1]);
    assert_eq!(&gpu_u8[252..258], &[1, 1, 0, 0, 0, 1]);
}

#[test]
fn cuda_line_splice_classify_generated_multi_block_corpus() {
    let mut source = Vec::with_capacity(4101);
    while source.len() < 4101 {
        let line = source.len() / 53;
        match line % 6 {
            0 => source.extend_from_slice(b"#define JOIN(a, b) \\\n  a ## b\n"),
            1 => source.extend_from_slice(b"char slash = '\\\\';\n"),
            2 => source.extend_from_slice(b"int crlf = 1;\\\r\nint next = 2;\n"),
            3 => source.extend_from_slice(b"plain tokens with / at end /\n"),
            4 => source.extend_from_slice(b"continued\\\rmac_style\n"),
            _ => source.extend_from_slice(b"two\\\\\nslashes\n"),
        }
    }
    source.truncate(4101);

    let cpu = reference_line_splice_classify(&source);
    let gpu = run_line_splice(&source);
    let gpu_u8 = run_line_splice_u8(&source);

    assert_eq!(gpu, cpu);
    assert_eq!(gpu_u8, cpu);
    assert!(
        gpu.iter().any(|kept| *kept == 0),
        "Fix: generated CUDA line-splice corpus must exercise deleted bytes."
    );
    assert!(
        gpu.iter().any(|kept| *kept == 1),
        "Fix: generated CUDA line-splice corpus must exercise kept bytes."
    );
}
