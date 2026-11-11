//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "hash")]

use crate::hash_oracles;
use hash_oracles::{hostile_bytes, oracle_crc32};

use vyre_reference::composition_witness::crc32_witness;

const CASES: usize = 16384;

#[test]
fn sweep_hash_crc32_volume_oracle_matrix() {
    for idx in 0..CASES {
        let bytes = hostile_bytes(idx as u32);
        assert_eq!(
            crc32_witness(&bytes),
            oracle_crc32(&bytes),
            "Fix: crc32 volume case {idx} len={}",
            bytes.len()
        );
    }
}
