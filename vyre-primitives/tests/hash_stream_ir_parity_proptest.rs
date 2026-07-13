//! Tier 3 - Property: differential proptest driving the ACTUAL byte-stream hash IR of
//! `hash::adler32_program` and `hash::multi_hash_program` through `reference_eval` vs their CPU
//! oracles. Both had `reference_eval` = 0 in the coverage audit; their `sweep_hash_*` peers assert
//! oracle-vs-oracle and the only randomized IR check is the single inventory fixture.
//!
//! - `adler32`: a single-lane rolling `(a, b)` accumulator with the `MOD_ADLER` reduction and the
//!   `b << 16 | a` finalize. A wrong modulus, a swapped a/b, or a missed final combine diverges.
//! - `multi_hash`: one lane fuses THREE independent hashes over the same byte stream into `out[0..3]`.
//!   A crossed-wire between the three lanes or a wrong per-hash constant diverges on one of the three.
//!
//! Both consume one byte per u32 input slot (low 8 bits). The sweep runs random byte strings
//! (0..=256) plus deterministic anchors: `adler32(b"Wikipedia") == 0x11E60398` and the empty string.
#![cfg(all(feature = "hash", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::hash::adler32::{adler32, adler32_program};
use vyre_primitives::hash::multi_hash::{multi_hash_program, multi_hash_reference};

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes.iter().map(|&b| u32::from(b)).collect()
}

fn decode_words(v: &Value) -> Vec<u32> {
    v.to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn run_adler(bytes: &[u8]) -> u32 {
    let program = adler32_program("input", "out", bytes.len() as u32);
    let pack = |d: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(d));
    let outputs =
        vyre_reference::reference_eval(&program, &[pack(&bytes_to_words(bytes)), pack(&[0u32])])
            .expect("adler32 reference evaluation must succeed");
    decode_words(&outputs[0])[0]
}

fn run_multi(bytes: &[u8]) -> (u32, u32, u32) {
    let program = multi_hash_program("input", "out", bytes.len() as u32);
    let pack = |d: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(d));
    let outputs = vyre_reference::reference_eval(
        &program,
        &[pack(&bytes_to_words(bytes)), pack(&[0u32, 0, 0])],
    )
    .expect("multi_hash reference evaluation must succeed");
    let w = decode_words(&outputs[0]);
    (w[0], w[1], w[2])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn adler32_ir_matches_oracle(bytes in prop::collection::vec(any::<u8>(), 0..=256)) {
        prop_assert_eq!(run_adler(&bytes), adler32(&bytes), "len={}", bytes.len());
    }

    #[test]
    fn multi_hash_ir_matches_oracle(bytes in prop::collection::vec(any::<u8>(), 0..=256)) {
        prop_assert_eq!(run_multi(&bytes), multi_hash_reference(&bytes), "len={}", bytes.len());
    }
}

#[test]
fn hash_stream_ir_anchors() {
    // Adler-32 canonical check value.
    assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398, "adler oracle anchor");
    assert_eq!(run_adler(b"Wikipedia"), 0x11E6_0398, "adler IR anchor");
    assert_eq!(run_adler(&[]), adler32(&[]));
    assert_eq!(run_adler(&[0u8]), adler32(&[0u8]));

    // multi_hash on a fixed string agrees end to end.
    for s in [b"abc".as_slice(), b"", b"the quick brown fox jumps over"] {
        assert_eq!(
            run_multi(s),
            multi_hash_reference(s),
            "multi_hash anchor {s:?}"
        );
    }
}
