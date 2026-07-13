//! Tier 3 - Property: differential proptest driving the ACTUAL `text::utf8_shape_counts` IR through
//! `reference_eval` vs an INDEPENDENT in-test reference. The op had `reference_eval` = 0 in tests/, and
//! its shipped oracle (`utf8_shape_counts_from_histogram`) is `pub(crate)` — unreachable from an
//! external test — so this file computes the reference a second, independent way (range sums), which
//! is the correct differential pattern (a second implementation, not a copy of the same function).
//!
//! The kernel loops the byte-value histogram over `0x80..0xF5` and, per lead-byte class, accumulates:
//!   - `continuation`  += count for `0x80..0xC0` (UTF-8 continuation bytes),
//!   - `expected`      += count for `0xC2..0xE0` (2-byte leads, +1 continuation each),
//!                     += 2*count for `0xE0..0xF0` (3-byte leads),
//!                     += 3*count for `0xF0..0xF5` (4-byte leads),
//! all with SATURATING add/mul. The sweep feeds random 256-bin histograms plus histograms biased to
//! `u32::MAX` in the multiplied ranges so the saturating `count*2`/`count*3` and the saturating sums
//! are actually driven into saturation — the exact path a plain wrapping op would get wrong.
#![cfg(all(feature = "text", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::text::utf8_shape_counts::utf8_shape_counts;

/// Independent reference: saturating range sums matching the kernel's lead-byte classes.
fn reference(hist: &[u32; 256]) -> (u32, u32) {
    let continuation = hist[0x80..0xC0]
        .iter()
        .fold(0u32, |a, &c| a.saturating_add(c));
    let mut expected = hist[0xC2..0xE0]
        .iter()
        .fold(0u32, |a, &c| a.saturating_add(c));
    expected = hist[0xE0..0xF0]
        .iter()
        .fold(expected, |a, &c| a.saturating_add(c.saturating_mul(2)));
    expected = hist[0xF0..0xF5]
        .iter()
        .fold(expected, |a, &c| a.saturating_add(c.saturating_mul(3)));
    (continuation, expected)
}

fn run_ir(hist: &[u32; 256]) -> (u32, u32) {
    let program = utf8_shape_counts("histogram", "out");
    let pack = |d: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(d));
    let outputs = vyre_reference::reference_eval(&program, &[pack(hist), pack(&[0u32, 0])])
        .expect("utf8_shape_counts reference evaluation must succeed");
    let w: Vec<u32> = outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    (w[0], w[1])
}

prop_compose! {
    fn arb_hist()(
        raw in prop::collection::vec(
            prop_oneof![0u32..=32, any::<u32>(), Just(u32::MAX), Just(u32::MAX / 2)], 256)
    ) -> [u32; 256] {
        let mut h = [0u32; 256];
        h.copy_from_slice(&raw);
        h
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn utf8_shape_counts_ir_matches_reference(hist in arb_hist()) {
        prop_assert_eq!(run_ir(&hist), reference(&hist));
    }
}

#[test]
fn utf8_shape_counts_ir_saturation_and_ranges() {
    // Empty histogram → both zero.
    let zero = [0u32; 256];
    assert_eq!(run_ir(&zero), (0, 0));

    // One count in each representative class isolates the multipliers.
    let mut h = [0u32; 256];
    h[0x80] = 7; // continuation
    h[0xC2] = 5; // 2-byte lead → +5 expected
    h[0xE0] = 3; // 3-byte lead → +6 expected
    h[0xF0] = 2; // 4-byte lead → +6 expected
    assert_eq!(run_ir(&h), reference(&h));
    assert_eq!(run_ir(&h), (7, 5 + 6 + 6));

    // Saturation: MAX in a x3 range must saturate, not wrap.
    let mut sat = [0u32; 256];
    sat[0xF0] = u32::MAX;
    assert_eq!(
        run_ir(&sat),
        (0, u32::MAX),
        "count*3 must saturate at u32::MAX"
    );
    assert_eq!(run_ir(&sat), reference(&sat));
}
