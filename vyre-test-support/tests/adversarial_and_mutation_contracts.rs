//! Integration tests for adversarial generator contracts and mutation proofs.

use vyre_test_support::adversarial_generators::{
    assert_adversarial_suite_validity, generate_adversarial_suite,
};
use vyre_test_support::mutation_testing::{
    assert_mutations_are_detected, representative_mutations,
};

#[test]
fn adversarial_suite_contracts_hold() {
    let suite = generate_adversarial_suite();
    assert!(suite.len() >= 5);
    assert_adversarial_suite_validity();
}

#[test]
fn mutation_proof_suite_detects_all_invariants() {
    let mutations = representative_mutations();
    assert!(mutations.len() >= 3);
    assert_mutations_are_detected();
}
