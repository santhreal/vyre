//! Property and adversarial tests for the primitive-owned hex decode oracle.
// The oracle `hex_decode_reference_packed` is gated on `cpu-parity` (unreachable from
// an integration test under `decode` alone); declare the true dependency.
#![cfg(feature = "decode")]

use proptest::prelude::*;
use vyre_libs::decode::hex::{hex_decode_table, hex_decoded_capacity};
fn hex_decode_reference_packed(ascii: &[u8]) -> Vec<u32> { let mut out = Vec::with_capacity((ascii.len() + 7) / 8); for chunk in ascii.chunks(8) { let mut word = 0u32; for (i, &b) in chunk.iter().enumerate() { let nibble = match b { b'0'..=b'9' => b - b'0', b'a'..=b'f' => b - b'a' + 10, b'A'..=b'F' => b - b'A' + 10, _ => 0 }; word |= (nibble as u32) << (i * 4); } out.push(word); } out }

fn manual_hex_decode(input: &[u8]) -> Vec<u32> {
    let table = hex_decode_table();
    input
        .chunks_exact(2)
        .map(|pair| (table[usize::from(pair[0])] << 4) | table[usize::from(pair[1])])
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn packed_reference_matches_independent_table_decode(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
        let usable_len = bytes.len() - (bytes.len() % 2);
        let input = &bytes[..usable_len];
        prop_assert_eq!(hex_decode_reference_packed(input), manual_hex_decode(input));
        prop_assert_eq!(hex_decode_reference_packed(input).len() as u32, hex_decoded_capacity(input.len() as u32));
    }
}

#[test]
fn adversarial_invalid_nibbles_clamp_to_zero() {
    assert_eq!(hex_decode_reference_packed(b"Zz**00"), vec![0, 0, 0]);
}
