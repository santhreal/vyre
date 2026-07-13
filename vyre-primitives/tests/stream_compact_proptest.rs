//! Tier 3 - Property: proptest over random liveness masks for `math::stream_compact`, driving the
//! GPU IR through `reference_eval` vs `cpu_ref`. The shipped file has only inline CPU-oracle unit
//! tests + a program-shape test; the actual data-derived SCATTER
//! (`compacted[offsets[i]] = payloads[i]` for live lanes, with `offsets` = exclusive prefix sum of
//! `flags`) is never run through a faithful executor.
//!
//! For each of 3000 random instances the generator builds `flags` (0/1 per lane), random `payloads`,
//! and the EXACT exclusive prefix-sum `offsets` the op contracts on, then runs the IR and asserts:
//! (a) `compacted[0..live_count]` equals `cpu_ref`'s dense survivor list (every position `0..live`
//! is written by exactly one live lane, so the comparison is payload-value-agnostic), and
//! (b) `live_count` == the survivor count. A wrong offset gather, an off-by-one on the final-lane
//! `live_count` write, or a missed live lane diverges. Complements the inline oracle tests with the
//! GPU-IR scatter path over randomized masks incl. all-dead, all-live, and single-lane.
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::math::stream_compact::{cpu_ref, stream_compact};
use vyre_reference::value::Value;

/// Exclusive prefix sum of a 0/1 flag buffer — the `offsets` the op requires.
fn exclusive_prefix(flags: &[u32]) -> Vec<u32> {
    let mut acc = 0u32;
    flags
        .iter()
        .map(|&f| {
            let out = acc;
            acc += u32::from(f != 0);
            out
        })
        .collect()
}

/// Returns (compacted, live_count) from the IR.
fn run_ir(payloads: &[u32], flags: &[u32], offsets: &[u32]) -> (Vec<u32>, u32) {
    let count = payloads.len() as u32;
    let program = stream_compact("payloads", "flags", "offsets", "compacted", "live", count);
    let pack = |data: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(data));
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            pack(payloads),                        // payloads (0, RO)
            pack(flags),                           // flags (1, RO)
            pack(offsets),                         // offsets (2, RO)
            pack(&vec![0u32; count as usize]),     // compacted (3, RW)
            pack(&[0u32]),                         // live_count (4, RW)
        ],
    )
    .expect("stream_compact reference evaluation must succeed");
    // RW buffers in binding order: compacted(3) then live_count(4).
    let compacted: Vec<u32> = outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let live = {
        let b = outputs[1].to_bytes();
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    };
    (compacted, live)
}

prop_compose! {
    fn arb_case()(count in 1usize..=64)
        (flags in prop::collection::vec(prop_oneof![Just(0u32), Just(1u32)], count),
         payloads in prop::collection::vec(any::<u32>(), count))
        -> (Vec<u32>, Vec<u32>) {
        (payloads, flags)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    #[test]
    fn stream_compact_ir_matches_cpu_ref((payloads, flags) in arb_case()) {
        let offsets = exclusive_prefix(&flags);
        let (compacted, live) = run_ir(&payloads, &flags, &offsets);
        let (want_compacted, want_live) = cpu_ref(&payloads, &flags);

        prop_assert_eq!(live, want_live, "live_count mismatch: flags={:?}", flags);
        prop_assert_eq!(
            &compacted[..want_live as usize], &want_compacted[..],
            "compacted survivors diverge: payloads={:?} flags={:?} offsets={:?}",
            payloads, flags, offsets
        );
    }
}
