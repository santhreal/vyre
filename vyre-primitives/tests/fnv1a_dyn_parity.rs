//! GPU-IR parity for the dynamic-bound FNV-1a32 builders
//! `fnv1a32_program_dyn` (u32 lanes) and `fnv1a32_program_dyn_u8` (packed u8).
//!
//! These differ from the static `fnv1a32_program(input, out, n)` only in the loop
//! bound: `Expr::buf_len(input)` instead of `Expr::u32(n)`, leaving `input`
//! without a static count (so they hash whatever the dispatched buffer declares).
//! They had no parity test (found by the registry-coverage closure gate). Each
//! hashes the LOW BYTE of every input element with the canonical FNV-1a32 walk,
//! so the result must equal `fnv1a::fnv1a32(low_bytes)`. Pins that against
//! `reference_eval`, including the low-byte masking and the empty-input basis.
#![cfg(feature = "hash")]

use vyre_primitives::hash::fnv1a::{fnv1a32, fnv1a32_program_dyn, fnv1a32_program_dyn_u8};
use vyre_primitives::wire::pack_u32_slice as pack_u32;
use vyre_reference::value::Value;

fn hash_out(program: &vyre_foundation::ir::Program, input: Vec<u8>) -> u32 {
    let outputs = vyre_reference::reference_eval(
        program,
        &[Value::from(input), Value::from(0u32.to_le_bytes().to_vec())],
    )
    .expect("fnv1a dyn reference evaluation must succeed");
    // `out` is the sole writable (output) buffer, one u32.
    let bytes = outputs[0].to_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

#[test]
fn dyn_u32_lanes_hash_low_bytes_matching_cpu_ref() {
    // Low bytes are 0x41,0x42,0x43 ('A','B','C'); high bytes must be ignored.
    let words = [0x1234_5641u32, 0x00FF_0042, 0xDEAD_0043];
    let expected = fnv1a32(&[0x41, 0x42, 0x43]);
    let program = fnv1a32_program_dyn("input", "out");
    assert_eq!(
        hash_out(&program, pack_u32(&words)),
        expected,
        "fnv1a32_program_dyn must hash only the low byte of each u32 lane"
    );
}

#[test]
fn dyn_u8_packed_matches_cpu_ref() {
    let bytes = vec![0x41u8, 0x42, 0x43, 0x00, 0xFF, 0x7A];
    let expected = fnv1a32(&bytes);
    let program = fnv1a32_program_dyn_u8("input", "out");
    assert_eq!(
        hash_out(&program, bytes),
        expected,
        "fnv1a32_program_dyn_u8 must hash the packed byte stream"
    );
}

#[test]
fn dyn_empty_input_is_offset_basis() {
    // buf_len == 0 → the walk runs zero updates → the FNV-1a32 offset basis.
    let expected = fnv1a32(&[]);
    assert_eq!(
        hash_out(&fnv1a32_program_dyn("input", "out"), Vec::new()),
        expected,
        "empty input must yield the FNV-1a32 offset basis"
    );
    assert_eq!(
        hash_out(&fnv1a32_program_dyn_u8("input", "out"), Vec::new()),
        expected,
    );
}
