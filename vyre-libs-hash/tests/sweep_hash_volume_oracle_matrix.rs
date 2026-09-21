//! Volume-wave oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
//!
//! CRC-32 over the same corpus is proved by `sweep_hash_crc32_volume_oracle_matrix`.
#![forbid(unsafe_code)]
#![cfg(feature = "hash")]

use crate::hash_oracles;
use hash_oracles::{hostile_bytes, oracle_fnv1a32};

use vyre_reference::composition_witness::fnv1a32_witness;

const CASES: u32 = 16384;

#[test]
fn sweep_fnv1a32_volume_oracle_matrix() {
    for idx in 0..CASES {
        let bytes = hostile_bytes(idx);
        assert_eq!(
            fnv1a32_witness(&bytes),
            oracle_fnv1a32(&bytes),
            "Fix: fnv1a32 volume case {idx} len={}",
            bytes.len()
        );
    }
}
