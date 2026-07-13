//! End-to-end parity for `math::conv1d_latency_smoothing::smooth_latency_trace_via`, the Gaussian 1D
//! convolution latency smoother (through the shared faithful [`common::ReferenceEvalDispatcher`]).
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the `conv1d_program` IR is not run through a faithful dispatch boundary by any `vyre-primitives/tests/*`
//! file. This is the FIRST-EVER execution of the conv1d smoothing kernel through a boundary that models the
//! real backend.
//!
//! Contract (audited CLEAN): the via computes the Gaussian kernel with `gaussian_weights(radius, sigma)`
//! and dispatches `conv1d_program`, which binds latency RO(0) + out RW(1) + weights RO(2) + params RO(3)
//! = 4 IC (out zero-filled), decoding outputs[0] = the smoothed trace.
//!
//! BIT-EXACT (no tolerance): the via and the reference use the SAME `gaussian_weights`, and the kernel is
//! raw u32 `wrapping_mul`/`wrapping_add` with clamped boundaries, the exact semantics of the importable
//! `cpu_conv1d`. So `smooth_latency_trace_via(latency, radius, sigma)` must equal
//! `cpu_conv1d(latency, gaussian_weights(radius, sigma), 1)` bit-for-bit; this pins that `conv1d_program`
//! reproduces `cpu_conv1d` (mul/accumulate order AND the clamp-to-edge boundary handling).
#![cfg(feature = "cpu-parity")]

use vyre_primitives::math::conv1d::{cpu_conv1d, gaussian_weights};
use vyre_self_substrate::math::conv1d_latency_smoothing::smooth_latency_trace_via;

mod common;
use common::ReferenceEvalDispatcher;

const FIXED_ONE: u32 = 1 << 16;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn smooth_latency_trace_via_matches_cpu_conv1d_bit_exact() {
    let d = ReferenceEvalDispatcher;
    let mut state = 0xC0_11_00_01u32;
    // Representative radius / sigma combinations covering the shipped smoothing kernels.
    let configs = [
        (1u32, 0.5f32),
        (1, 1.0),
        (2, 0.75),
        (2, 1.5),
        (3, 1.0),
        (3, 2.0),
        (4, 1.25),
    ];
    let mut nontrivial = 0u32; // cases where smoothing actually changed the trace
    for case in 0..350u32 {
        let (radius, sigma) = configs[case as usize % configs.len()];
        let n = (radius as usize * 2) + 1 + (case % 24) as usize; // >= kernel diameter
                                                                  // Latency values in 16.16 across [0, 4.0) keep the raw-u32 accumulation well-scaled.
        let latency: Vec<u32> = (0..n)
            .map(|_| xorshift(&mut state) % (4 * FIXED_ONE))
            .collect();

        let got = smooth_latency_trace_via(&d, &latency, radius, sigma)
            .expect("smooth_latency_trace_via must dispatch the conv1d smoother");
        let weights = gaussian_weights(radius, sigma);
        let want = cpu_conv1d(&latency, &weights, 1);
        assert_eq!(
            got, want,
            "case {case}: GPU conv1d must match cpu_conv1d bit-for-bit; radius={radius} sigma={sigma} \
             n={n} latency={latency:?}"
        );

        if got != latency {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 250,
        "sweep must exercise traces the smoother actually changes, got {nontrivial}"
    );
}

#[test]
fn smooth_latency_trace_via_hand_checked_empty_and_edge_clamp() {
    let d = ReferenceEvalDispatcher;

    // Empty trace → empty output.
    let got = smooth_latency_trace_via(&d, &[], 1, 1.0).unwrap();
    assert!(got.is_empty(), "empty latency trace smooths to empty");

    // A short constant trace: convolving a constant with a normalized-ish kernel is dominated by the
    // kernel-weight sum; whatever the exact value, the GPU must equal cpu_conv1d exactly (edge clamp).
    let latency = vec![FIXED_ONE, FIXED_ONE, FIXED_ONE, FIXED_ONE, FIXED_ONE];
    let got = smooth_latency_trace_via(&d, &latency, 2, 1.0).unwrap();
    let want = cpu_conv1d(&latency, &gaussian_weights(2, 1.0), 1);
    assert_eq!(
        got, want,
        "constant trace under edge-clamped conv matches cpu_conv1d"
    );
    // A constant input stays constant across all interior AND clamped-boundary positions.
    assert!(
        got.iter().all(|&v| v == got[0]),
        "clamped-boundary conv of a constant trace is itself constant, got {got:?}"
    );
}
