//! Tier 3 - Property: differential proptest driving the ACTUAL `decode::rle_segment_lengths` IR
//! through `reference_eval` vs `rle_segment_lengths_cpu`. The op had `reference_eval` = 0 in tests/
//! (its `rle_segment_lengths_contracts.rs` peer checks the CPU packer/oracle, not the kernel).
//!
//! The kernel is one lane per packed RLE header word: `length = packed >> 8`, `value = packed & 0xFF`,
//! written to two separate per-segment output buffers. A swapped length/value, a wrong shift/mask, or
//! a crossed output binding diverges. The sweep runs random packed streams (segment_count 1..=256, one
//! workgroup) with values covering the full 24-bit length / 8-bit value fields, asserting BOTH output
//! buffers (`segment_lengths_out`=binding 1, `segment_values_out`=binding 2) bit-exact vs the oracle.
#![cfg(all(feature = "decode", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::decode::rle_segment_lengths::{rle_segment_lengths, rle_segment_lengths_cpu};

fn decode(v: &Value) -> Vec<u32> {
    v.to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Returns (lengths, values) from the IR.
fn run_ir(segments_in: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let n = segments_in.len() as u32;
    let program = rle_segment_lengths(n);
    let pack = |d: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(d));
    let zeros = vec![0u32; n as usize];
    let outputs =
        vyre_reference::reference_eval(&program, &[pack(segments_in), pack(&zeros), pack(&zeros)])
            .expect("rle_segment_lengths reference evaluation must succeed");
    // RW buffers in binding order: lengths_out(1) then values_out(2).
    (decode(&outputs[0]), decode(&outputs[1]))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn rle_segment_lengths_ir_matches_oracle(
        // Random packed words: full 32-bit range so length (hi 24) and value (lo 8) both vary widely.
        segments_in in prop::collection::vec(any::<u32>(), 1..=256)
    ) {
        let (lengths, values) = run_ir(&segments_in);
        let (want_lengths, want_values) = rle_segment_lengths_cpu(&segments_in);
        prop_assert_eq!(&lengths, &want_lengths, "lengths diverge");
        prop_assert_eq!(&values, &want_values, "values diverge");
    }
}

#[test]
fn rle_segment_lengths_ir_field_boundaries() {
    // Exercise the field seams: max length, max value, both zero, both max, mid mix.
    let segments = vec![
        0x0000_0000,          // length 0, value 0
        0x0000_00FF,          // length 0, value 255
        0xFFFF_FF00,          // length 0xFFFFFF, value 0
        0xFFFF_FFFF,          // length 0xFFFFFF, value 255
        (123u32 << 8) | 0x42, // length 123, value 0x42
        (1u32 << 8),          // length 1, value 0
    ];
    let (lengths, values) = run_ir(&segments);
    let (want_lengths, want_values) = rle_segment_lengths_cpu(&segments);
    assert_eq!(lengths, want_lengths);
    assert_eq!(values, want_values);
    assert_eq!(
        lengths,
        vec![0, 0, 0xFF_FFFF, 0xFF_FFFF, 123, 1],
        "length extraction"
    );
    assert_eq!(values, vec![0, 0xFF, 0, 0xFF, 0x42, 0], "value extraction");
}
