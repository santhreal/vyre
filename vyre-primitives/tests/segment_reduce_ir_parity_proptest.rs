//! Tier 3 - Property: differential proptest driving the ACTUAL CSR per-segment reduction IR of
//! `reduce::segment_reduce_sum` through `reference_eval` vs `cpu_ref`.
//!
//! MOTIVATION — real IR gap. The shipped file's tests are all CPU-oracle self-consistency
//! (`two_segments`, `wraps_on_overflow`, `try_cpu_ref_into_*`) or program SHAPE
//! (`emitted_program_has_expected_buffers`, `zero_segments_traps`); `grep reference_eval` = 0. The
//! GPU IR — a per-lane `loop_for` whose bounds are DATA-DEPENDENT (`start = offsets[seg]`,
//! `end = offsets[seg+1]`) accumulating `input[start..end]` — is validated ONLY by the single
//! inventory fixture (`num_segments = 2`, one case). A wrong per-segment offset gather, an off-by-one
//! on the `end` bound, an empty-segment mishandle, or a u32 wrap divergence passes every existing test.
//!
//! This sweep runs the real Program over randomized CSR layouts: `num_segments` 1..=64, per-segment
//! lengths 0..=8 (so EMPTY segments and long segments coexist), random `input` including values large
//! enough to force wrapping accumulation. Each `output[seg]` is asserted bit-exact vs `cpu_ref`. A
//! deterministic case pins overflow wrap and an all-empty-segments layout.
#![cfg(all(feature = "reduce", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::reduce::segment_reduce::{cpu_ref, segment_reduce_sum};

/// Build CSR offsets from per-segment lengths; returns (offsets, total).
fn offsets_from_lengths(lengths: &[u32]) -> (Vec<u32>, u32) {
    let mut offsets = Vec::with_capacity(lengths.len() + 1);
    let mut acc = 0u32;
    offsets.push(0);
    for &l in lengths {
        acc += l;
        offsets.push(acc);
    }
    (offsets, acc)
}

fn run_ir(input: &[u32], offsets: &[u32], num_segments: u32) -> Vec<u32> {
    let program = segment_reduce_sum("input", "segment_offsets", "output", num_segments);
    let pack = |data: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(data));
    // Guard: an all-empty layout has total==0; the IR never loads `input`, but hand it one dummy
    // element so the RO buffer is non-degenerate.
    let input_arg: &[u32] = if input.is_empty() { &[0u32] } else { input };
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            pack(input_arg),                          // input (binding 0, RO)
            pack(offsets),                            // segment_offsets (binding 1, RO)
            pack(&vec![0u32; num_segments as usize]), // output (binding 2, RW)
        ],
    )
    .expect("segment_reduce_sum reference evaluation must succeed");
    // Sole RW buffer is `output` (binding 2) → results[0].
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

prop_compose! {
    fn arb_case()(num_segments in 1u32..=64)
        (lengths in prop::collection::vec(0u32..=8, num_segments as usize),
         // Values biased to include large magnitudes so accumulation wraps u32.
         seed in prop::collection::vec(
            prop_oneof![any::<u32>(), Just(u32::MAX), 0u32..=16], 512))
        -> (Vec<u32>, Vec<u32>) {
        (lengths, seed)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2500))]

    #[test]
    fn segment_reduce_ir_matches_cpu_ref((lengths, seed) in arb_case()) {
        let num_segments = lengths.len() as u32;
        let (offsets, total) = offsets_from_lengths(&lengths);
        // Fill `input` of length `total` by cycling the random seed pool (total <= 64*8 = 512).
        let input: Vec<u32> = (0..total as usize).map(|i| seed[i % seed.len()]).collect();

        let got = run_ir(&input, &offsets, num_segments);
        let want = cpu_ref(&input, &offsets);
        prop_assert_eq!(
            &got, &want,
            "lengths={:?} offsets={:?} total={}", lengths, offsets, total
        );
    }
}

#[test]
fn segment_reduce_ir_overflow_and_empty_segments() {
    // Overflow wrap: MAX + 1 = 0 in segment 0; segment 1 empty; segment 2 sums to 5.
    let input = [u32::MAX, 1, 2, 3];
    let offsets = [0u32, 2, 2, 4]; // seg0=[MAX,1]->0(wrap), seg1=[]->0, seg2=[2,3]->5
    let got = run_ir(&input, &offsets, 3);
    let want = cpu_ref(&input, &offsets);
    assert_eq!(got, vec![0, 0, 5], "IR overflow/empty result");
    assert_eq!(got, want, "IR must match oracle on overflow/empty layout");

    // All-empty layout: every segment zero.
    let offsets_empty = [0u32, 0, 0, 0];
    let got_empty = run_ir(&[], &offsets_empty, 3);
    assert_eq!(got_empty, vec![0, 0, 0]);
}
