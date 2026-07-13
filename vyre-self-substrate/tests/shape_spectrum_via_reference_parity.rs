//! End-to-end parity for `scheduling::spectral_schedule::shape_spectrum_fixed_via`, the
//! Marchenko-Pastur outlier-eigenvalue edge clip, through the shared faithful
//! [`common::ReferenceEvalDispatcher`].
//!
//! Closes a mock-dispatcher-coherence gap (see BACKLOG `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the `mp_edge_clip` IR is not run through a faithful dispatch boundary by any `vyre-primitives/tests/*`
//! file. This is the FIRST-EVER execution of the MP edge-clip kernel through a boundary that models the
//! real backend.
//!
//! Contract (audited CLEAN): `mp_edge_clip` (a `u32_vector_scalar_map_program` with `Expr::min`) binds
//! eigenvalues RO(0) + mp_edge scalar RO(1) + out RW(2) = 3 IC; the via zero-fills `out` and decodes
//! outputs[0] = the clipped vector.
//!
//! BIT-EXACT (no tolerance): the kernel is pure u32 elementwise `out[i] = min(eigenvalues[i], mp_edge)`.
//! Because `min` is monotone and order-preserving, the u32 result equals the documented f64 reference
//! `mp_edge_clip_cpu` (`v.min(edge)`) applied to the same magnitudes and re-encoded, so this suite
//! asserts BOTH the direct u32 min oracle AND agreement with the importable `mp_edge_clip_cpu`.
#![cfg(feature = "cpu-parity")]

use vyre_primitives::math::spectral_shape::mp_edge_clip_cpu;
use vyre_self_substrate::spectral_schedule::shape_spectrum_fixed_via;

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
fn shape_spectrum_via_matches_u32_min_clip_bit_exact() {
    let d = ReferenceEvalDispatcher;
    let mut state = 0x5A9E_0001u32;
    let mut clipped_some = 0u32; // cases where at least one eigenvalue was actually clipped
    let mut passed_some = 0u32; // cases where at least one eigenvalue was below the edge (unchanged)
    for case in 0..400u32 {
        let n = 1 + (case % 20) as usize;
        // Eigenvalues in 16.16 across [0, 8.0); the edge somewhere in (0, 8.0) so both clip + pass occur.
        let eigenvalues: Vec<u32> = (0..n)
            .map(|_| xorshift(&mut state) % (8 * FIXED_ONE))
            .collect();
        let mp_edge = 1 + xorshift(&mut state) % (8 * FIXED_ONE);

        let got = shape_spectrum_fixed_via(&d, &eigenvalues, mp_edge)
            .expect("shape_spectrum_fixed_via must dispatch the MP edge clip");

        // Direct u32 oracle = the exact kernel semantics.
        let want: Vec<u32> = eigenvalues.iter().map(|&e| e.min(mp_edge)).collect();
        assert_eq!(
            got, want,
            "case {case}: GPU MP clip must equal the u32 elementwise min; edge={mp_edge} eig={eigenvalues:?}"
        );

        // Cross-check against the importable f64 reference on the same 16.16 magnitudes.
        let eig_f: Vec<f64> = eigenvalues
            .iter()
            .map(|&v| f64::from(v) / f64::from(FIXED_ONE))
            .collect();
        let edge_f = f64::from(mp_edge) / f64::from(FIXED_ONE);
        let ref_f = mp_edge_clip_cpu(&eig_f, edge_f);
        let ref_fixed: Vec<u32> = ref_f
            .iter()
            .map(|&v| (v * f64::from(FIXED_ONE)).round() as u32)
            .collect();
        assert_eq!(
            got, ref_fixed,
            "case {case}: GPU MP clip must match mp_edge_clip_cpu re-encoded to 16.16"
        );

        if eigenvalues.iter().any(|&e| e > mp_edge) {
            clipped_some += 1;
        }
        if eigenvalues.iter().any(|&e| e <= mp_edge) {
            passed_some += 1;
        }
    }
    assert!(
        clipped_some > 250,
        "sweep must exercise real clipping (eigenvalue above the edge), got {clipped_some}"
    );
    assert!(
        passed_some > 250,
        "sweep must exercise pass-through (eigenvalue below the edge), got {passed_some}"
    );
}

#[test]
fn shape_spectrum_via_hand_checked_clip_boundary() {
    let d = ReferenceEvalDispatcher;
    // Edge = 2.0 (16.16). Eigenvalues 1.0, 2.0, 3.5 → clip to 1.0, 2.0, 2.0.
    let edge = 2 * FIXED_ONE;
    let eig = [FIXED_ONE, 2 * FIXED_ONE, 7 * FIXED_ONE / 2];
    let got = shape_spectrum_fixed_via(&d, &eig, edge).unwrap();
    assert_eq!(
        got,
        vec![FIXED_ONE, 2 * FIXED_ONE, 2 * FIXED_ONE],
        "below-edge passes, at-edge stays, above-edge clips to the edge"
    );

    // All below the edge → identity.
    let got = shape_spectrum_fixed_via(&d, &[FIXED_ONE / 2, FIXED_ONE], edge).unwrap();
    assert_eq!(
        got,
        vec![FIXED_ONE / 2, FIXED_ONE],
        "all-below-edge is an identity clip"
    );

    // All above the edge → every value collapses to the edge.
    let got = shape_spectrum_fixed_via(&d, &[3 * FIXED_ONE, 5 * FIXED_ONE], edge).unwrap();
    assert_eq!(
        got,
        vec![edge, edge],
        "all-above-edge collapses to the edge"
    );
}
