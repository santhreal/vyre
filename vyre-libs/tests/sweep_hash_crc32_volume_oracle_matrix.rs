//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "hash")]

mod wire_words;
use wire_words::hostile_bytes;

use vyre_libs::hash::crc32;

fn oracle_crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

const CASES: usize = 16384;

#[test]
fn sweep_hash_crc32_volume_oracle_matrix() {
    for idx in 0..CASES {
        let bytes = hostile_bytes(idx as u32);
        assert_eq!(
            crc32::crc32(&bytes),
            oracle_crc32(&bytes),
            "Fix: crc32 volume case {idx} len={}",
            bytes.len()
        );
    }
}
