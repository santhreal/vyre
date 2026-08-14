//! IR parity for the u32-lane FNV-1a64 builders `fnv1a64_program_n` (static
//! bound) and `fnv1a64_program` (dynamic `buf_len` bound).
//!
//! The packed-u8 builder `fnv1a64_program_n_u8` already has parity coverage in
//! `proptest_hash_fnv1a.rs` and `adversarial_hash.rs`; the u32-lane pair had
//! none. Each hashes the LOW BYTE of every input element, so the two output
//! words must equal the little-endian halves of `fnv1a::fnv1a64(low_bytes)`.
#![cfg(feature = "hash")]

use vyre_primitives::hash::fnv1a::{fnv1a64, fnv1a64_program, fnv1a64_program_n};
use vyre_primitives::wire::pack_u32_slice as pack_u32;
use vyre_reference::value::Value;

/// Run a u32-lane FNV-1a64 program and recombine its two output words.
fn hash_out(program: &vyre_foundation::ir::Program, input: Vec<u8>) -> u64 {
    let outputs = vyre_reference::reference_eval(
        program,
        &[Value::from(input), Value::from(vec![0u8; 8])],
    )
    .expect("fnv1a64 u32-lane reference evaluation must succeed");
    let bytes = outputs[0].to_bytes();
    assert_eq!(bytes.len(), 8, "fnv1a64 writes exactly two u32 words");
    let lo = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let hi = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    (u64::from(hi) << 32) | u64::from(lo)
}

/// Widen bytes into one u32 lane each, the layout both builders consume.
fn lanes(bytes: &[u8]) -> Vec<u8> {
    pack_u32(&bytes.iter().map(|&b| u32::from(b)).collect::<Vec<_>>())
}

#[test]
fn static_bound_u32_lanes_match_cpu_ref() {
    for message in [b"abc".as_slice(), b"foobar".as_slice()] {
        let n = u32::try_from(message.len()).expect("message length fits u32");
        assert_eq!(
            hash_out(&fnv1a64_program_n("input", "out", n), lanes(message)),
            fnv1a64(message),
            "fnv1a64_program_n must match the CPU reference for {message:?}"
        );
    }
}

#[test]
fn canonical_fnv1a64_vectors_are_pinned() {
    assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_F739_67E8);
    assert_eq!(fnv1a64(b"abc"), 0xE71F_A219_0541_574B);
}

#[test]
fn dynamic_bound_u32_lanes_match_cpu_ref() {
    let message = b"abc";
    assert_eq!(
        hash_out(&fnv1a64_program("input", "out"), lanes(message)),
        fnv1a64(message),
        "fnv1a64_program must hash every declared u32 lane"
    );
}

#[test]
fn high_lane_bits_are_ignored() {
    // Low bytes spell "abc"; the high 24 bits of each lane must not participate.
    let words = [0xFFFF_FF61u32, 0xCAFE_0062, 0x8000_0063];
    assert_eq!(
        hash_out(&fnv1a64_program_n("input", "out", 3), pack_u32(&words)),
        fnv1a64(b"abc"),
        "fnv1a64_program_n must mask each u32 lane to its low byte"
    );
}

#[test]
fn long_inputs_carry_the_64_bit_product() {
    // The 64-bit prime multiply is synthesized from 32-bit parts; long inputs
    // exercise carry propagation between the halves on every update.
    for len in [64usize, 512] {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let bytes: Vec<u8> = (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 0xFF) as u8
            })
            .collect();
        let n = u32::try_from(len).expect("length fits u32");
        assert_eq!(
            hash_out(&fnv1a64_program_n("input", "out", n), lanes(&bytes)),
            fnv1a64(&bytes),
            "fnv1a64_program_n must match the CPU reference over {len} bytes"
        );
    }
}
