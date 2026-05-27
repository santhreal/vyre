//! Parity test: GPU iterated forward closure matches CPU iterated closure.

#![cfg(test)]

mod common;

use common::{live_dispatcher, CudaOptimizerDispatcher};
use vyre_self_substrate::csr_forward_or_changed::{
    forward_closure_via_change_flag_gpu, reference_forward_closure_via_change_flag,
};

#[test]
fn cuda_forward_closure_chain_matches_cpu() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    // Chain 0 -> 1 -> 2 -> 3.
    let off = vec![0u32, 1, 2, 3, 3];
    let tgt = vec![1u32, 2, 3];
    let msk = vec![1u32, 1, 1];
    let seed = vec![0b0001u32];
    let cpu =
        reference_forward_closure_via_change_flag(4, &off, &tgt, &msk, &seed, 0xFFFF_FFFF, 16);
    let gpu = forward_closure_via_change_flag_gpu(
        &dispatcher,
        4,
        &off,
        &tgt,
        &msk,
        &seed,
        0xFFFF_FFFF,
        16,
    )
    .expect("dispatch");
    assert_eq!(gpu, cpu, "chain closure divergence");
    assert_eq!(gpu, vec![0b1111u32]);
}

#[test]
fn cuda_forward_closure_disconnected() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    // 0 -> 1, 2 -> 3, no cross.
    let off = vec![0u32, 1, 1, 2, 2];
    let tgt = vec![1u32, 3];
    let msk = vec![1u32, 1];
    let seed = vec![0b0001u32]; // only {0}
    let cpu =
        reference_forward_closure_via_change_flag(4, &off, &tgt, &msk, &seed, 0xFFFF_FFFF, 16);
    let gpu = forward_closure_via_change_flag_gpu(
        &dispatcher,
        4,
        &off,
        &tgt,
        &msk,
        &seed,
        0xFFFF_FFFF,
        16,
    )
    .expect("dispatch");
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0b0011u32]);
}

#[test]
fn cuda_forward_closure_self_loop_terminates() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let off = vec![0u32, 1, 1];
    let tgt = vec![0u32];
    let msk = vec![1u32];
    let seed = vec![0b01u32];
    let cpu =
        reference_forward_closure_via_change_flag(2, &off, &tgt, &msk, &seed, 0xFFFF_FFFF, 50);
    let gpu = forward_closure_via_change_flag_gpu(
        &dispatcher,
        2,
        &off,
        &tgt,
        &msk,
        &seed,
        0xFFFF_FFFF,
        50,
    )
    .expect("dispatch");
    assert_eq!(gpu, cpu);
}

#[test]
fn cuda_forward_closure_allow_mask_filters() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let off = vec![0u32, 1, 1];
    let tgt = vec![1u32];
    let msk = vec![0b0010u32];
    let seed = vec![0b01u32];
    let allow = 0b0001;
    let cpu = reference_forward_closure_via_change_flag(2, &off, &tgt, &msk, &seed, allow, 16);
    let gpu =
        forward_closure_via_change_flag_gpu(&dispatcher, 2, &off, &tgt, &msk, &seed, allow, 16)
            .expect("dispatch");
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0b01u32]);
}

#[test]
fn cuda_forward_closure_diamond() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    // Diamond 0 -> {1, 2} -> 3.
    let off = vec![0u32, 2, 3, 4, 4];
    let tgt = vec![1u32, 2, 3, 3];
    let msk = vec![1u32, 1, 1, 1];
    let seed = vec![0b0001u32];
    let cpu =
        reference_forward_closure_via_change_flag(4, &off, &tgt, &msk, &seed, 0xFFFF_FFFF, 16);
    let gpu = forward_closure_via_change_flag_gpu(
        &dispatcher,
        4,
        &off,
        &tgt,
        &msk,
        &seed,
        0xFFFF_FFFF,
        16,
    )
    .expect("dispatch");
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0b1111u32]);
}
