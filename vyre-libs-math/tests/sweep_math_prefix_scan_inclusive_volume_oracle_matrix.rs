//! Volume oracle matrix - independent reference vs production cpu_ref.
//! Volume testing.volume - do NOT weaken to shape-only asserts.
#![forbid(unsafe_code)]
#![cfg(feature = "math")]

use crate::scan_oracle;
use vyre_test_support::word_corpora::lcg32_words;
use scan_oracle::prefix_scan_cpu_ref as cpu_ref;

use vyre_libs_math::math::prefix_scan::ScanKind;

fn oracle_inclusive_scan(input: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    let mut acc = 0u32;
    for &x in input {
        acc = acc.wrapping_add(x);
        out.push(acc);
    }
    out
}

const CASES: usize = 16384;

#[test]
fn sweep_math_prefix_scan_inclusive_volume_oracle_matrix() {
    for idx in 0..CASES {
        let len = idx % 256;
        let input = lcg32_words(idx as u32, len);
        assert_eq!(
            cpu_ref(&input, ScanKind::InclusiveSum),
            oracle_inclusive_scan(&input),
            "Fix: prefix_scan inclusive volume case {idx} len={len}"
        );
    }
}
