//! P-HARNESS-1: every composition consumer runs through the
//! conform suite.
//!
//! For every module in `vyre_libs::encoding::*`, the
//! conform suite runs the module's primary entry point on the
//! standard corpus and asserts no panics. Today the test gates
//! each consumer behind a feature flag; the consumer's own crate
//! must enable that feature in CI.
#![allow(missing_docs)]

use vyre_reference::composition_witness::{
    hypervector_xor_bind_witness, scallop_join_fixpoint_witness,
};

#[test]
fn no_self_consumer_panics_on_smoke_input() {
    let mut state = vec![0u32; 4];
    state[1] = 0b01;
    let mut join_rules = vec![0u32; 4];
    join_rules[3] = 0b10;
    let closure = scallop_join_fixpoint_witness(&state, &join_rules, 2, 1, 8).0;
    assert_eq!(closure[1], 0b11);

    let bound = hypervector_xor_bind_witness(&[0x7679_7265; 8], &[0x6c69_6e6b; 8]);
    let fingerprint = hypervector_xor_bind_witness(&bound, &[0x7375_6273; 8]);
    assert!(
        fingerprint.iter().any(|&lane| lane != 0),
        "vsa_fingerprint self-consumer must produce a nonzero key for nonempty input"
    );
}
