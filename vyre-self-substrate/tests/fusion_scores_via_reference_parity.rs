//! End-to-end parity for `scheduling::spectral_schedule::fusion_scores_fixed_via`, the Chebyshev
//! spectral-fusion-score filter `f(L̂)·v`: through the shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the `chebyshev_filter` IR is not run through a faithful dispatch boundary by any
//! `vyre-primitives/tests/*` file (the in-file dispatcher hand-computes it). This is the FIRST-EVER
//! execution of the barrier-cooperative Chebyshev matrix-polynomial kernel through a boundary that models
//! the real backend (workgroup barriers between the T_{k-1}/T_k recurrence steps, which reference_eval
//! honors as uniform-control-flow barriers).
//!
//! Contract (audited CLEAN): `chebyshev_filter` binds laplacian RO(0) + signal RO(1) + coeffs RO(2) +
//! output RW(3) + scratch RW(4, 2n words) = 5 IC; the via zero-fills output/scratch and decodes
//! outputs[0] = the length-n score vector.
//!
//! BIT-EXACT (no tolerance): the IR performs RAW u32 INTEGER arithmetic, plain `mul`/`add`/`sub`/`2·`,
//! NO fixed-point shift, and `chebyshev_filter_cpu` computes the IDENTICAL formula in f32. With small
//! integer inputs the recurrence `T_next = 2·(L̂·T_curr) − T_prev` never u32-underflows (every
//! `2·(L̂·T_curr)` dominates `T_prev`) and every intermediate stays < 2^24, where f32 represents integers
//! EXACTLY. So the u32 GPU output equals the f32 reference cast to integer, bit-for-bit.
#![cfg(feature = "cpu-parity")]

use vyre_primitives::graph::chebyshev_filter::chebyshev_filter_cpu;
use vyre_self_substrate::spectral_schedule::fusion_scores_fixed_via;

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// A small positive integer in `1..=hi`.
fn small(state: &mut u32, hi: u32) -> u32 {
    1 + xorshift(state) % hi
}

#[test]
fn fusion_scores_via_matches_chebyshev_cpu_bit_exact() {
    let d = ReferenceEvalDispatcher;
    let mut state = 0x5EC7_0001u32;
    let k_steps = 2u32; // 3 coefficients, the shipped fusion-score order
    let mut nonzero_out = 0u32;
    for case in 0..300u32 {
        let n = 2 + (case % 4); // 2..5
                                // Small positive integers keep the raw-integer recurrence from u32-underflowing and well
                                // under 2^24, so the f32 reference is an EXACT integer oracle.
        let laplacian: Vec<u32> = (0..n * n).map(|_| small(&mut state, 3)).collect();
        let signal: Vec<u32> = (0..n).map(|_| small(&mut state, 3)).collect();
        let coeffs: Vec<u32> = (0..=k_steps).map(|_| small(&mut state, 3)).collect();

        let got = fusion_scores_fixed_via(&d, &laplacian, &signal, &coeffs, n, k_steps)
            .expect("fusion_scores_fixed_via must dispatch the chebyshev filter");

        let lap_f: Vec<f32> = laplacian.iter().map(|&v| v as f32).collect();
        let sig_f: Vec<f32> = signal.iter().map(|&v| v as f32).collect();
        let coeff_f: Vec<f32> = coeffs.iter().map(|&v| v as f32).collect();
        let want_f = chebyshev_filter_cpu(&lap_f, &sig_f, &coeff_f, n, k_steps);
        let want: Vec<u32> = want_f.iter().map(|&v| v.round() as u32).collect();

        assert_eq!(
            got, want,
            "case {case}: GPU chebyshev filter must match the CPU reference bit-for-bit; \
             n={n} laplacian={laplacian:?} signal={signal:?} coeffs={coeffs:?}"
        );
        if got.iter().any(|&v| v != 0) {
            nonzero_out += 1;
        }
    }
    assert!(
        nonzero_out > 250,
        "sweep must produce nonzero fusion scores, got {nonzero_out}"
    );
}

#[test]
fn fusion_scores_via_hand_checked_identity_filter() {
    let d = ReferenceEvalDispatcher;
    // Identity L̂ (I), signal [2, 3], coeffs [1, 0, 0] → only the c0·signal term survives: out == signal.
    let laplacian = vec![1u32, 0, 0, 1];
    let signal = [2u32, 3];
    let coeffs = [1u32, 0, 0];
    let got = fusion_scores_fixed_via(&d, &laplacian, &signal, &coeffs, 2, 2).unwrap();
    assert_eq!(
        got,
        vec![2, 3],
        "c0=1 with c1=c2=0 passes the signal through unchanged"
    );

    // coeffs [0, 1, 0] with identity L̂ → out == L̂·signal == signal (T_1 = L̂·signal = signal).
    let coeffs = [0u32, 1, 0];
    let got = fusion_scores_fixed_via(&d, &laplacian, &signal, &coeffs, 2, 2).unwrap();
    assert_eq!(
        got,
        vec![2, 3],
        "c1=1 term = L̂·signal = signal under the identity operator"
    );

    // A non-trivial operator L̂ = [[1,1],[0,1]], signal [1, 1], coeffs [0, 1, 0]:
    // T_1 = L̂·signal = [1+1, 0+1] = [2, 1]; out = c1·T_1 = [2, 1].
    let laplacian = vec![1u32, 1, 0, 1];
    let signal = [1u32, 1];
    let coeffs = [0u32, 1, 0];
    let got = fusion_scores_fixed_via(&d, &laplacian, &signal, &coeffs, 2, 2).unwrap();
    let want_f = chebyshev_filter_cpu(&[1.0, 1.0, 0.0, 1.0], &[1.0, 1.0], &[0.0, 1.0, 0.0], 2, 2);
    let want: Vec<u32> = want_f.iter().map(|&v| v.round() as u32).collect();
    assert_eq!(got, want, "non-trivial operator matches the CPU reference");
    assert_eq!(
        got,
        vec![2, 1],
        "L̂·signal for the upper-triangular operator"
    );
}
