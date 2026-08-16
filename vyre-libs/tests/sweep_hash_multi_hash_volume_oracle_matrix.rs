mod wire_words;
use wire_words::hostile_bytes;

//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "hash")]

use vyre_libs::hash::{adler32, crc32, fnv1a, multi_hash};

const CASES: usize = 16384;


fn oracle_multi(bytes: &[u8]) -> (u32, u32, u32) {
    (
        crc32::crc32(bytes),
        fnv1a::fnv1a32(bytes),
        adler32::adler32(bytes),
    )
}

#[test]
fn sweep_hash_multi_hash_volume_oracle_matrix() {
    for idx in 0..CASES {
        let bytes = hostile_bytes(idx as u32 ^ 0xA11C_0DE1);
        let expected = oracle_multi(&bytes);
        let actual = multi_hash::multi_hash_reference(&bytes);
        assert_eq!(
            actual,
            expected,
            "Fix: multi_hash volume case {idx} len={}",
            bytes.len()
        );
    }
}
