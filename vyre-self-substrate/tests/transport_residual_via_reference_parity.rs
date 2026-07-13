//! End-to-end parity for `math::qsvt_matrix_function_fusion::transport_residual_fixed_via`, the QSVT
//! `f(M)·v` Chebyshev matrix-function filter (negative-truncator transport residual), through the shared
//! faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! this is the SIGNED-coefficient companion to `fusion_scores_via_reference_parity`. Both dispatch the SAME
//! `chebyshev_filter` IR, but the transport residual uses NEGATIVE truncator coefficients, the case that
//! made this conversion look hard until the IR was confirmed to be RAW u32 INTEGER arithmetic.
//!
//! Contract (audited CLEAN): `chebyshev_filter("dispatch_cost_scaled","weights","coeffs",…)` binds
//! dispatch_cost_scaled RO(0) + weights RO(1) + coeffs RO(2) + output RW(3) + scratch RW(4, 2n) = 5 IC;
//! the via zero-fills output/scratch and decodes outputs[0] = the length-n residual.
//!
//! BIT-EXACT via TWO'S-COMPLEMENT (no tolerance): u32 `wrapping` mul/add/sub is EXACTLY signed i32
//! arithmetic on the low 32 bits. Feeding signed small integers as two's-complement u32 makes the u32 IR
//! compute the true signed result mod 2^32; reinterpreting the u32 output as i32 recovers it. The identical
//! `chebyshev_filter_cpu` formula in f32 (exact for |value| < 2^24) is the oracle, so
//! `got[i] as i32 == want_f32[i].round() as i32` bit-for-bit. INCLUDING the negative-coefficient recurrence
//! `T_next = 2·(M·T_curr) − T_prev` that distinguishes transport from the positive-only fusion filter.
#![cfg(feature = "cpu-parity")]

use vyre_primitives::graph::chebyshev_filter::chebyshev_filter_cpu;
use vyre_self_substrate::math::qsvt_matrix_function_fusion::transport_residual_fixed_via;

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// A small SIGNED integer in `-lo..=hi`, returned as a two's-complement u32.
fn signed(state: &mut u32, span: i32) -> u32 {
    let v = (xorshift(state) % (2 * span as u32 + 1)) as i32 - span;
    v as u32
}

/// Reinterpret a two's-complement u32 as the signed integer it encodes.
fn as_i32(v: u32) -> i32 {
    v as i32
}

#[test]
fn transport_residual_via_matches_chebyshev_cpu_signed_bit_exact() {
    let d = ReferenceEvalDispatcher;
    let mut state = 0x7A_11_00_01u32;
    let chebyshev_order = 2u32; // 3 coefficients
    let mut nonzero = 0u32;
    let mut has_negative = 0u32;
    for case in 0..300u32 {
        let n = 2 + (case % 4); // 2..5
                                // Small signed integers (two's-complement u32); magnitudes small enough that every intermediate
                                // stays within i32 and < 2^24 so the f32 reference is an EXACT signed oracle.
        let dispatch_cost: Vec<u32> = (0..n * n).map(|_| signed(&mut state, 3)).collect();
        let weights: Vec<u32> = (0..n).map(|_| signed(&mut state, 3)).collect();
        // Include negative coefficients (the transport-residual regime the truncator uses).
        let coeffs: Vec<u32> = (0..=chebyshev_order)
            .map(|_| signed(&mut state, 3))
            .collect();

        let got =
            transport_residual_fixed_via(&d, &dispatch_cost, &weights, &coeffs, n, chebyshev_order)
                .expect("transport_residual_fixed_via must dispatch the chebyshev filter");

        let lap_f: Vec<f32> = dispatch_cost.iter().map(|&v| as_i32(v) as f32).collect();
        let sig_f: Vec<f32> = weights.iter().map(|&v| as_i32(v) as f32).collect();
        let coeff_f: Vec<f32> = coeffs.iter().map(|&v| as_i32(v) as f32).collect();
        let want_f = chebyshev_filter_cpu(&lap_f, &sig_f, &coeff_f, n, chebyshev_order);

        let got_signed: Vec<i32> = got.iter().map(|&v| as_i32(v)).collect();
        let want_signed: Vec<i32> = want_f.iter().map(|&v| v.round() as i32).collect();
        assert_eq!(
            got_signed, want_signed,
            "case {case}: GPU transport residual (two's-complement) must match the signed chebyshev CPU; \
             n={n} dispatch_cost={dispatch_cost:?} weights={weights:?} coeffs={coeffs:?}"
        );

        if got_signed.iter().any(|&v| v != 0) {
            nonzero += 1;
        }
        if got_signed.iter().any(|&v| v < 0) || coeffs.iter().any(|&c| as_i32(c) < 0) {
            has_negative += 1;
        }
    }
    assert!(
        nonzero > 200,
        "sweep must produce nonzero residuals, got {nonzero}"
    );
    assert!(
        has_negative > 200,
        "sweep must exercise the SIGNED (negative) regime that distinguishes transport from fusion, got {has_negative}"
    );
}

#[test]
fn transport_residual_via_hand_checked_negative_coefficient() {
    let d = ReferenceEvalDispatcher;
    // Identity operator M = I, weights [1, 2], coeffs [-1, 0, 0] → out = c0·weights = [-1, -2].
    let dispatch_cost = vec![1u32, 0, 0, 1];
    let weights = [1u32, 2];
    let coeffs = [(-1i32) as u32, 0, 0];
    let got = transport_residual_fixed_via(&d, &dispatch_cost, &weights, &coeffs, 2, 2).unwrap();
    let got_signed: Vec<i32> = got.iter().map(|&v| as_i32(v)).collect();
    assert_eq!(
        got_signed,
        vec![-1, -2],
        "a -1 leading coefficient negates the signal"
    );

    let want_f = chebyshev_filter_cpu(&[1.0, 0.0, 0.0, 1.0], &[1.0, 2.0], &[-1.0, 0.0, 0.0], 2, 2);
    let want_signed: Vec<i32> = want_f.iter().map(|&v| v.round() as i32).collect();
    assert_eq!(
        got_signed, want_signed,
        "negative-coefficient result matches the signed CPU reference"
    );
}
