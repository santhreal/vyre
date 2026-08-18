//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "math")]

mod wire_words;
use wire_words::{lcg_u32, prefix_scan_cpu_ref as cpu_ref};

use vyre_libs::math::prefix_scan::ScanKind;

fn oracle_exclusive_scan(input: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = 0u32;
    for &x in input {
        out.push(acc);
        acc = acc.wrapping_add(x);
    }
    out
}

const CASES: usize = 16384;

#[test]
fn sweep_math_prefix_scan_exclusive_volume_oracle_matrix() {
    for idx in 0..CASES {
        let len = idx % 256;
        let input = lcg_u32(idx as u32, len);
        assert_eq!(
            cpu_ref(&input, ScanKind::ExclusiveSum),
            oracle_exclusive_scan(&input),
            "Fix: prefix_scan exclusive volume case {idx} len={len}"
        );
    }
}
