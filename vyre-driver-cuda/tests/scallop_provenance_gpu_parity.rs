//! Parity test: GPU scallop provenance closure matches Reference oracle.

#![cfg(test)]

mod common;

use common::{live_dispatcher, CudaOptimizerDispatcher};
use vyre_self_substrate::scallop_provenance::{
    provenance_closure_via, reference_provenance_closure,
};

#[test]
fn cuda_scallop_provenance_closure_via_matches_reference_chain() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    // 4x4 state: clause-bitset on direct (out, src). Diagonal seeded
    // so each region claims clause i at (i, i).
    let state = vec![
        0b0001u32, 0, 0, 0, 0, 0b0010, 0, 0, 0, 0, 0b0100, 0, 0, 0, 0, 0b1000,
    ];
    // join_rules: 0 contains 1, 1 contains 2, 2 contains 3
    let join_rules = vec![0u32, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0];
    let gpu = provenance_closure_via(&dispatcher, &state, &join_rules, 4, 8).expect("dispatch");
    let reference = reference_provenance_closure(&state, &join_rules, 4, 8);
    assert_eq!(gpu, reference, "scallop provenance closure divergence");
}
