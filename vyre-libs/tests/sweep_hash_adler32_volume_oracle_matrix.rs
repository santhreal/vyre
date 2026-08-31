//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "hash")]

mod wire_words;
use wire_words::hostile_bytes;

use vyre_reference::composition_witness::adler32_witness;

const ADLER_MOD: u32 = 65_521;
const CASES: usize = 16384;

fn oracle_adler32(bytes: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in bytes {
        a = (a + u32::from(byte)) % ADLER_MOD;
        b = (b + a) % ADLER_MOD;
    }
    (b << 16) | a
}

#[test]
fn sweep_hash_adler32_volume_oracle_matrix() {
    for idx in 0..CASES {
        let bytes = hostile_bytes(idx as u32);
        assert_eq!(
            adler32_witness(&bytes),
            oracle_adler32(&bytes),
            "Fix: adler32 volume case {idx} len={}",
            bytes.len()
        );
    }
}
