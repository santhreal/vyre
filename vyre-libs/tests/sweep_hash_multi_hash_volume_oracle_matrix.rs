//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "hash")]

mod wire_words;
use wire_words::hostile_bytes;

use vyre_reference::composition_witness::{
    adler32_witness, crc32_witness, fnv1a32_witness, multi_hash_witness,
};

const CASES: usize = 16384;

fn oracle_multi(bytes: &[u8]) -> (u32, u32, u32) {
    (
        crc32_witness(bytes),
        fnv1a32_witness(bytes),
        adler32_witness(bytes),
    )
}

#[test]
fn sweep_hash_multi_hash_volume_oracle_matrix() {
    for idx in 0..CASES {
        let bytes = hostile_bytes(idx as u32 ^ 0xA11C_0DE1);
        let expected = oracle_multi(&bytes);
        let actual = multi_hash_witness(&bytes);
        assert_eq!(
            actual,
            expected,
            "Fix: multi_hash volume case {idx} len={}",
            bytes.len()
        );
    }
}
