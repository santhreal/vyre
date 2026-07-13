//! Tier 3 - Property: proptest over random brace-token sequences for `matching::bracket_match`,
//! driving the ACTUAL GPU IR through `reference_eval` vs `cpu_ref`. The shipped file value-checks
//! only the CPU oracle's self-consistency (`generated_uncapped_cases_match_stack_reference_contract`
//! asserts pair symmetry, NOT the IR) plus a SINGLE balanced inventory fixture — so neither GPU IR
//! path is validated against the oracle over real inputs.
//!
//! `bracket_match` chooses ONE of two structurally-different kernels by `max_depth` vs `n`:
//! - `max_depth >= n` → the PARALLEL per-lane matcher (each open scans forward for its close, each
//!   close scans backward), which must reproduce full bounded-stack matching (no overflow possible).
//! - `max_depth < n`  → the single-lane BOUNDED-STACK walk, where overflow opens are deliberately
//!   dropped.
//! This sweep draws `max_depth in 1..=(n+2)` so BOTH kernels are exercised, over sequences that are
//! balanced, nested, unbalanced (extra opens AND extra closes), and depth-overflowing — the exact
//! cases a single hand fixture cannot reach. Each result is asserted BIT-EXACT vs `cpu_ref`
//! (`match_pairs`: bidirectional links, `MATCH_NONE` for unmatched). Any divergence is a real
//! IR/oracle defect.
#![cfg(all(feature = "matching", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::matching::bracket_match::{
    bracket_match, cpu_ref, CLOSE_BRACE, MATCH_NONE, OPEN_BRACE, OTHER,
};
use vyre_reference::value::Value;

/// Run the IR and return the `match_pairs` output (results[1]: the RW buffers are `stack`(1) then
/// `match_pairs`(2), in binding order).
fn run_ir(kinds: &[u32], max_depth: u32) -> Vec<u32> {
    let n = kinds.len() as u32;
    let program = bracket_match("kinds", "stack", "match_pairs", n, max_depth);
    let pack = |data: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(data));
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            pack(kinds),                                  // kinds (binding 0, RO)
            pack(&vec![0u32; max_depth.max(1) as usize]), // stack (binding 1, RW)
            pack(&vec![MATCH_NONE; kinds.len()]),         // match_pairs (binding 2, output)
        ],
    )
    .expect("bracket_match reference evaluation must succeed");
    // results[0] = stack (RW), results[1] = match_pairs (output).
    outputs[1]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

prop_compose! {
    /// A random brace-token sequence (length 1..=64, each token OTHER/OPEN/CLOSE) and a `max_depth`
    /// drawn to straddle `n` so both the parallel (`>= n`) and bounded-stack (`< n`) kernels fire.
    fn arb_case()(len in 1usize..=64)
        (kinds in prop::collection::vec(
            prop_oneof![Just(OTHER), Just(OPEN_BRACE), Just(CLOSE_BRACE)], len),
         max_depth in 1u32..=(len as u32 + 2))
        -> (Vec<u32>, u32) {
        (kinds, max_depth)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    #[test]
    fn bracket_match_ir_matches_cpu_ref_across_both_kernels(
        (kinds, max_depth) in arb_case()
    ) {
        let got = run_ir(&kinds, max_depth);
        let want = cpu_ref(&kinds, max_depth);
        prop_assert_eq!(
            &got, &want,
            "kinds={:?} max_depth={} (n={}, kernel={}): IR {:?} != cpu_ref {:?}",
            kinds, max_depth, kinds.len(),
            if max_depth >= kinds.len() as u32 { "parallel" } else { "bounded-stack" },
            got, want
        );
        // Structural invariant on the IR output itself: every non-MATCH_NONE link is symmetric and
        // in range (a real GPU write of a stale/OOB index would break this even if it matched a
        // buggy oracle).
        for (i, &p) in got.iter().enumerate() {
            if p != MATCH_NONE {
                prop_assert!((p as usize) < got.len(), "link {} at {} out of range", p, i);
                prop_assert_eq!(got[p as usize], i as u32, "asymmetric link at {}", i);
            }
        }
    }
}
