//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "hash")]

mod wire_words;
use wire_words::oracle_blake3_g;

use vyre_libs::hash::blake3::{cpu_blake3_g, MSG_SCHEDULE};

const CASES: usize = 16384;

#[test]
fn sweep_hash_blake3_g_volume_oracle_matrix() {
    let _ = MSG_SCHEDULE;
    for idx in 0..CASES {
        let mut expected = [0u32; 16];
        let mut actual = [0u32; 16];
        for lane in 0..16 {
            expected[lane] = (idx as u32)
                .wrapping_add(lane as u32)
                .rotate_left((lane % 32) as u32);
            actual[lane] = expected[lane];
        }
        let mx = idx as u32 ^ 0x51ED_BEEF;
        let my = idx.rotate_left(7) as u32;
        oracle_blake3_g(&mut expected, 0, 4, 8, 12, mx, my);
        cpu_blake3_g(&mut actual, 0, 4, 8, 12, mx, my);
        assert_eq!(actual, expected, "Fix: blake3_g volume case {idx}");
    }
}
