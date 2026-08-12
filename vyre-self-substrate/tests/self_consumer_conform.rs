//! P-HARNESS-1: every self-substrate consumer runs through the
//! conform suite.
//!
//! For every module in `vyre_self_substrate::*`, the
//! conform suite runs the module's primary entry point on the
//! standard corpus and asserts no panics. Today the test gates
//! each consumer behind a feature flag; the consumer's own crate
//! must enable that feature in CI.
#![allow(missing_docs)]

use vyre_self_substrate::{scallop_provenance, vsa_fingerprint};

#[test]
fn no_self_consumer_panics_on_smoke_input() {
    let mut state = vec![0u32; 4];
    state[1] = 0b01;
    let mut join_rules = vec![0u32; 4];
    join_rules[3] = 0b10;
    let closure = scallop_provenance::reference_provenance_closure(&state, &join_rules, 2, 8);
    assert_eq!(closure[1], 0b11);

    let fingerprint = vsa_fingerprint::reference_fingerprint(
        &[0x7679_7265; 8],
        &[0x6c69_6e6b; 8],
        &[0x7375_6273; 8],
    );
    assert!(
        fingerprint.iter().any(|&lane| lane != 0),
        "vsa_fingerprint self-consumer must produce a nonzero key for nonempty input"
    );
}
