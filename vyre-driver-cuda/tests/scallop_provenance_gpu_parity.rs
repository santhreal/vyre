//! Parity test: GPU scallop provenance closure matches Reference oracle.

#![cfg(feature = "device-tests")]
#![cfg(test)]

mod harness;

use harness::with_cuda_optimizer_dispatcher;
use vyre_libs::encoding::scallop_provenance::provenance_closure_via;
use vyre_reference::composition_witness::scallop_join_fixpoint_witness;

#[test]
fn cuda_scallop_provenance_closure_via_matches_reference_chain() {
    // 4x4 state: clause-bitset on direct (out, src). Diagonal seeded
    // so each region claims clause i at (i, i).
    let state = vec![
        0b0001u32, 0, 0, 0, 0, 0b0010, 0, 0, 0, 0, 0b0100, 0, 0, 0, 0, 0b1000,
    ];
    // join_rules: 0 contains 1, 1 contains 2, 2 contains 3
    let join_rules = vec![0u32, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0];
    let gpu = with_cuda_optimizer_dispatcher("scallop provenance closure", |dispatcher| {
        provenance_closure_via(dispatcher, &state, &join_rules, 4, 8).expect("dispatch")
    });
    let (reference, _iters) = scallop_join_fixpoint_witness(&state, &join_rules, 4, 1, 8);
    assert_eq!(gpu, reference, "scallop provenance closure divergence");
}
