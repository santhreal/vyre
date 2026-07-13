//! Tier 3 - Property: differential proptest driving the ACTUAL `reduce::range_counts_u32` IR through
//! `reference_eval` vs `cpu_ref`. `grep reference_eval` = 0 in the shipped inline tests; the GPU IR
//! (a `loop_for` summing `histogram[start..end]` with the bounds baked in as COMPILE-TIME constants)
//! is validated only by the single inventory fixture.
//!
//! The sweep varies BOTH the 256-bin histogram contents (including `u32::MAX` bins that force wrapping
//! accumulation, the exact case the op's doc-comment calls out as GPU-vs-`.sum()`-diverging) AND the
//! `[start, end)` window across the full `0..=256` range, rebuilding the Program per case so the
//! constant-folded loop bounds are actually exercised. Empty windows (`start == end`), full-range
//! windows, and single-bin windows are all reachable. Each result is asserted bit-exact vs `cpu_ref`.
#![cfg(all(feature = "reduce", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::reduce::range_counts::{cpu_ref, range_counts_u32};

const BINS: usize = 256;

fn run_ir(histogram: &[u32], start: u32, end: u32) -> u32 {
    let program = range_counts_u32("histogram", "out", start, end);
    let pack = |data: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(data));
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            pack(histogram), // histogram (binding 0, RO, 256 bins)
            pack(&[0u32]),   // out (binding 1, output)
        ],
    )
    .expect("range_counts_u32 reference evaluation must succeed");
    let b = outputs[0].to_bytes();
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

prop_compose! {
    /// A full 256-bin histogram plus an in-range half-open window `[start, end)`.
    fn arb_case()(
        histogram in prop::collection::vec(
            prop_oneof![any::<u32>(), Just(u32::MAX), 0u32..=64], BINS),
        a in 0u32..=BINS as u32,
        b in 0u32..=BINS as u32,
    ) -> (Vec<u32>, u32, u32) {
        let (start, end) = if a <= b { (a, b) } else { (b, a) };
        (histogram, start, end)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2500))]

    #[test]
    fn range_counts_ir_matches_cpu_ref((histogram, start, end) in arb_case()) {
        let got = run_ir(&histogram, start, end);
        let want = cpu_ref(&histogram, start, end);
        prop_assert_eq!(got, want, "start={} end={}", start, end);
    }
}

#[test]
fn range_counts_ir_boundary_windows() {
    let mut hist = vec![0u32; BINS];
    for (i, h) in hist.iter_mut().enumerate() {
        *h = i as u32; // distinct per-bin values so a shifted window changes the sum
    }
    // Empty window, single bin, full range, and a wrap-forcing window.
    let cases = [(0u32, 0u32), (5, 6), (0, 256), (100, 200), (255, 256)];
    for (start, end) in cases {
        let got = run_ir(&hist, start, end);
        let want = cpu_ref(&hist, start, end);
        assert_eq!(got, want, "boundary window [{start},{end})");
    }

    // Overflow wrap: two MAX bins summed → wraps to u32::MAX-1.
    let mut wrap = vec![0u32; BINS];
    wrap[10] = u32::MAX;
    wrap[11] = u32::MAX;
    let got = run_ir(&wrap, 10, 12);
    assert_eq!(got, u32::MAX.wrapping_add(u32::MAX));
    assert_eq!(got, cpu_ref(&wrap, 10, 12));
}
