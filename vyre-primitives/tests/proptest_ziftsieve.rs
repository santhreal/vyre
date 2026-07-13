//! Property and adversarial tests for primitive-owned LZ4 literal extraction.

// `ziftsieve_reference_extract_literals` is gated on `cpu-parity` (unreachable from an
// integration test under `decode` alone); declare the true dependency.
#![cfg(all(feature = "decode", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::decode::ziftsieve::ziftsieve_reference_extract_literals;

fn literal_only_lz4(bytes: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(bytes.len() + (bytes.len() / 14 + 1) * 3);
    let mut chunks = bytes.chunks(14).peekable();
    while let Some(chunk) = chunks.next() {
        encoded.push((chunk.len() as u8) << 4);
        encoded.extend_from_slice(chunk);
        if chunks.peek().is_some() {
            encoded.extend_from_slice(&[0, 0]);
        }
    }
    encoded
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    #[test]
    fn literal_only_blocks_round_trip(bytes in proptest::collection::vec(any::<u8>(), 0..768)) {
        let encoded = literal_only_lz4(&bytes);
        let result = ziftsieve_reference_extract_literals(&encoded, bytes.len()).unwrap();
        // A complete (uncapped) decode reports the true length and is NOT truncated.
        // Read the Copy field + `&self` predicate BEFORE the Vec comparison, which
        // moves `result.literals`.
        prop_assert_eq!(result.decoded_len, bytes.len());
        prop_assert!(!result.truncated());
        prop_assert_eq!(result.literals, bytes);
    }
}

#[test]
fn reference_honors_output_cap_without_overallocating() {
    let encoded = literal_only_lz4(b"abcdefghijklmnopqrstuvwxyz");
    let got = ziftsieve_reference_extract_literals(&encoded, 7).unwrap();
    // The bytes are capped at max_output=7 (the GPU fixed-output-buffer bound)...
    assert_eq!(got.literals, b"abcdefg");
    // ...but the cap is now OBSERVABLE (Law 10): decoded_len reports the true 26-byte
    // length and `truncated()` is true, so a caller can detect the dropped 19 bytes
    // instead of mistaking the capped output for a complete decode.
    assert_eq!(got.decoded_len, 26);
    assert!(got.truncated());
    assert_eq!(got.decoded_len - got.literals.len(), 19);
}

#[test]
fn reference_rejects_truncated_extended_literal_length() {
    let err = ziftsieve_reference_extract_literals(&[0xF0], 1024).unwrap_err();
    assert!(err.contains("truncated length encoding"));
    assert!(err.contains("Fix:"));
}
