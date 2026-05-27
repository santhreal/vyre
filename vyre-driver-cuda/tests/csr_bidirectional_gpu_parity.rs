//! Parity test: GPU bidirectional CSR step matches the reference oracle.

#![cfg(test)]

mod common;

use common::{live_dispatcher, CudaOptimizerDispatcher};
use vyre_self_substrate::csr_bidirectional::{
    bidirectional_closure_via, bidirectional_step_via, reference_bidirectional_closure,
    reference_bidirectional_step,
};

fn linear_chain() -> (u32, Vec<u32>, Vec<u32>, Vec<u32>) {
    // 0 -> 1 -> 2 -> 3
    (4, vec![0, 1, 2, 3, 3], vec![1, 2, 3], vec![1, 1, 1])
}

#[test]
fn cuda_bidirectional_step_via_matches_reference_chain() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let (n, off, tgt, msk) = linear_chain();
    // Seed = {1}. Expected: forward {2} ∪ backward {0} = {0, 2}.
    let gpu = bidirectional_step_via(&dispatcher, n, &off, &tgt, &msk, &[0b0010], 0xFFFF_FFFF)
        .expect("dispatch");
    let reference = reference_bidirectional_step(n, &off, &tgt, &msk, &[0b0010], 0xFFFF_FFFF);
    assert_eq!(gpu, reference);
}

#[test]
fn cuda_bidirectional_step_via_respects_allow_mask() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    // 2-node graph: 0 -[k=2]-> 1
    let n = 2;
    let off = vec![0u32, 1, 1];
    let tgt = vec![1u32];
    let msk = vec![0b0010];
    let seed = vec![0b01u32];
    // allow_mask = 0b0001 → no edges match → empty step output.
    let gpu =
        bidirectional_step_via(&dispatcher, n, &off, &tgt, &msk, &seed, 0b0001).expect("dispatch");
    let reference = reference_bidirectional_step(n, &off, &tgt, &msk, &seed, 0b0001);
    assert_eq!(gpu, reference);
    // allow_mask = 0b0010 → forward 0→1 fires.
    let gpu =
        bidirectional_step_via(&dispatcher, n, &off, &tgt, &msk, &seed, 0b0010).expect("dispatch");
    let reference = reference_bidirectional_step(n, &off, &tgt, &msk, &seed, 0b0010);
    assert_eq!(gpu, reference);
}

#[test]
fn cuda_bidirectional_closure_via_matches_reference_chain() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let (n, off, tgt, msk) = linear_chain();
    let gpu = bidirectional_closure_via(
        &dispatcher,
        n,
        &off,
        &tgt,
        &msk,
        &[0b0001u32],
        0xFFFF_FFFF,
        n,
    )
    .expect("dispatch");
    let reference = reference_bidirectional_closure(n, &off, &tgt, &msk, &[0b0001], 0xFFFF_FFFF, n);
    assert_eq!(gpu, reference);
}
