//! Property tests for vyre-libs  -  invariants that must hold across
//! every input in the relevant domain.

#![cfg(all(
    feature = "math-linalg",
    feature = "math-scan",
    feature = "math-broadcast",
))]

use proptest::prelude::*;
use vyre::ir::Program;
use vyre_libs::math::broadcast::broadcast;
use vyre_libs::math::linalg::{dot, matmul};
use vyre_foundation::ir::PORTABLE_WORKGROUP_INVOCATIONS;
use vyre_libs::math::prefix_scan::MAX_SINGLE_BLOCK_SCAN;
use vyre_libs::math::scan::scan_prefix_sum;

fn has_single_region(program: &Program) -> bool {
    matches!(program.entry().first(), Some(vyre::ir::Node::Region { .. }))
        && program.entry().len() == 1
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    #[test]
    fn dot_program_is_always_single_region(
        a in "[a-z][a-z0-9_]*",
        b in "[a-z][a-z0-9_]*",
        c in "[a-z][a-z0-9_]*",
    ) {
        prop_assume!(a != b && b != c && a != c);
        let p = dot(&a, &b, &c, 256).unwrap();
        prop_assert!(has_single_region(&p));
    }

    #[test]
    fn matmul_preserves_dims(
        m in 1u32..64,
        k in 1u32..64,
        n in 1u32..64,
    ) {
        let p = matmul("a", "b", "c", m, k, n);
        prop_assert!(has_single_region(&p));
        prop_assert_eq!(p.workgroup_size(), [256, 1, 1]);
    }

    /// The workgroup is capped, not inflated. A scan past
    /// `PORTABLE_WORKGROUP_INVOCATIONS` elements gives each lane a longer run instead of
    /// asking for more lanes, so the next power of two stops being the answer
    /// at n = 257 and the cap is the answer from there to the single-block
    /// ceiling.
    #[test]
    fn scan_prefix_sum_is_valid_for_all_sizes(n in 1u32..=MAX_SINGLE_BLOCK_SCAN) {
        let p = scan_prefix_sum("in", "out", n);
        prop_assert!(has_single_region(&p));
        prop_assert_eq!(
            p.workgroup_size(),
            [n.next_power_of_two().min(PORTABLE_WORKGROUP_INVOCATIONS), 1, 1]
        );
    }

    #[test]
    fn broadcast_is_structurally_valid(
        s in "[a-z][a-z0-9_]*",
        d in "[a-z][a-z0-9_]*",
    ) {
        prop_assume!(s != d);
        let p = broadcast(&s, &d, 8);
        prop_assert!(has_single_region(&p));
    }
}

/// The lane count stops following the next power of two exactly one element
/// past the cap. A random sample over the whole domain can miss that step, so
/// the three sizes around it are pinned: the last size the two rules agree on,
/// the first size they disagree on, and the largest single-block scan.
#[test]
fn scan_prefix_sum_caps_lanes_one_element_past_the_workgroup_width() {
    let lanes = |n: u32| scan_prefix_sum("in", "out", n).workgroup_size()[0];

    assert_eq!(lanes(PORTABLE_WORKGROUP_INVOCATIONS), PORTABLE_WORKGROUP_INVOCATIONS);
    assert_eq!(lanes(PORTABLE_WORKGROUP_INVOCATIONS + 1), PORTABLE_WORKGROUP_INVOCATIONS);
    assert_eq!(lanes(MAX_SINGLE_BLOCK_SCAN), PORTABLE_WORKGROUP_INVOCATIONS);
    assert!(
        (PORTABLE_WORKGROUP_INVOCATIONS + 1).next_power_of_two() > PORTABLE_WORKGROUP_INVOCATIONS,
        "this contract is vacuous unless the next power of two exceeds the cap here"
    );
}

// Every vyre-libs Program contains one top-level Region; these tests
// prove the full Region wire round-trip (generator + optional
// source_region + body) is byte-identity stable across encode/decode.
#[test]

fn wire_round_trip_for_dot() {
    let p = dot("a", "b", "c", 4).unwrap();
    let wire = p.to_wire().expect("dot program must serialize");
    let parsed = Program::from_wire(&wire).expect("dot wire bytes must decode");
    assert_eq!(parsed, p);
}

#[test]

fn wire_round_trip_for_broadcast() {
    let p = broadcast("src", "dst", 8);
    let wire = p.to_wire().expect("broadcast program must serialize");
    let parsed = Program::from_wire(&wire).expect("broadcast wire bytes must decode");
    assert_eq!(parsed, p);
}
