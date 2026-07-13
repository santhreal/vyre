//! Tier 3 - Parity: drives the ACTUAL 2-level AMG V-cycle IR (`math::amg_v_cycle`, a single-lane
//! serial 16.16 FIXED-POINT V-cycle: presmooth -> residual -> restrict -> 4x coarse-Jacobi -> prolong
//! -> postsmooth) through `reference_eval` and compares the decoded fine solution against the shipped
//! f64 oracle `amg_v_cycle::cpu_ref`. The op had NO `reference_eval` test.
//!
//! Because the IR is 16.16 fixed-point and the oracle is f64, parity is APPROXIMATE (like the kfac
//! parity test's f32-vs-f64 bound). The system is chosen so fixed-point error is small and bounded:
//! all matrix DIAGONALS are 4 (a power of two, so the Jacobi `/diag` is a near-exact shift), omega is
//! 0.5 (exact in 16.16, `*0.5` is a shift), and every off-diagonal / transfer / rhs entry is a small
//! integer or half-integer that is exactly representable. The tolerance is set from the observed
//! fixed-point-vs-f64 gap over the full V-cycle, tight enough that a real kernel defect (a wrong
//! matvec index, a dropped phase, a sign error in the residual, a mis-scaled restrict/prolong) exceeds
//! it by orders of magnitude.
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use vyre_reference::value::Value;

use vyre_primitives::math::amg_v_cycle::{amg_v_cycle, cpu_ref};

/// Encode an f64 as signed 16.16 fixed-point (two's complement u32).
fn enc(v: f64) -> u32 {
    (v * 65536.0).round() as i32 as u32
}

/// Decode a signed 16.16 fixed-point u32 back to f64.
fn dec(u: u32) -> f64 {
    (u as i32) as f64 / 65536.0
}

fn enc_vec(v: &[f64]) -> Vec<u32> {
    v.iter().copied().map(enc).collect()
}

/// Run the IR and return the decoded fine solution `x` (binding 2, first RW buffer).
#[allow(clippy::too_many_arguments)]
fn run_ir(
    a: &[f64],
    b: &[f64],
    x0: &[f64],
    r_mat: &[f64],
    p_mat: &[f64],
    a_c: &[f64],
    omega: f64,
    n_fine: u32,
    n_coarse: u32,
) -> Vec<f64> {
    let program = amg_v_cycle(
        "a",
        "b",
        "x",
        "r_mat",
        "p_mat",
        "a_c",
        "omega",
        "scratch_fine",
        "scratch_coarse_b",
        "scratch_coarse_x",
        n_fine,
        n_coarse,
    );
    let pack = |data: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(data));
    let nf = n_fine as usize;
    let nc = n_coarse as usize;
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            pack(&enc_vec(a)),     // a (0, RO)
            pack(&enc_vec(b)),     // b (1, RO)
            pack(&enc_vec(x0)),    // x (2, RW) <- output
            pack(&enc_vec(r_mat)), // r_mat (3, RO)
            pack(&enc_vec(p_mat)), // p_mat (4, RO)
            pack(&enc_vec(a_c)),   // a_c (5, RO)
            pack(&[enc(omega)]),   // omega (6, RO)
            pack(&vec![0u32; nf]), // scratch_fine (7, RW)
            pack(&vec![0u32; nc]), // scratch_coarse_b (8, RW)
            pack(&vec![0u32; nc]), // scratch_coarse_x (9, RW)
            pack(&vec![0u32; nc]), // temp_coarse (10, RW)
        ],
    )
    .expect("amg_v_cycle reference evaluation must succeed");
    // RW buffers in binding order: x(2) first.
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| dec(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
        .collect()
}

/// A well-conditioned 4->2 V-cycle whose entries are all exactly 16.16-representable.
fn fixture() -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    f64,
) {
    // Fine A: 4x4 tridiagonal(-1, 4, -1), SPD + diagonally dominant, diagonal = 4 (power of two).
    let a = vec![
        4.0, -1.0, 0.0, 0.0, //
        -1.0, 4.0, -1.0, 0.0, //
        0.0, -1.0, 4.0, -1.0, //
        0.0, 0.0, -1.0, 4.0,
    ];
    let b = vec![4.0, 4.0, 4.0, 4.0];
    let x0 = vec![0.0, 0.0, 0.0, 0.0];
    // Restriction R (2x4): aggregate adjacent pairs with weight 0.5 (exact in 16.16).
    let r_mat = vec![
        0.5, 0.5, 0.0, 0.0, //
        0.0, 0.0, 0.5, 0.5,
    ];
    // Prolongation P (4x2): piecewise-constant interpolation (transpose-like, integer entries).
    let p_mat = vec![
        1.0, 0.0, //
        1.0, 0.0, //
        0.0, 1.0, //
        0.0, 1.0,
    ];
    // Coarse A_c (2x2): diagonal 4 (power of two) again.
    let a_c = vec![
        4.0, -1.0, //
        -1.0, 4.0,
    ];
    let omega = 0.5;
    (a, b, x0, r_mat, p_mat, a_c, omega)
}

#[test]
fn amg_v_cycle_ir_matches_f64_oracle() {
    let (a, b, x0, r_mat, p_mat, a_c, omega) = fixture();
    let n_fine = 4u32;
    let n_coarse = 2u32;

    let got = run_ir(&a, &b, &x0, &r_mat, &p_mat, &a_c, omega, n_fine, n_coarse);
    let want = cpu_ref(&a, &b, &x0, &r_mat, &p_mat, &a_c, omega, n_fine, n_coarse);

    assert_eq!(got.len(), want.len(), "solution length");
    assert_eq!(got.len(), n_fine as usize);

    // Fixed-point (16.16) vs f64 over a full V-cycle with power-of-two diagonals: the per-op error is
    // ~2^-16 and there are O(n_fine^2) accumulating ops, so a few * 1e-3 is the expected envelope. A
    // real kernel defect diverges by O(1). Assert BOTH element-wise closeness and a non-trivial
    // solution (not all-zero) so the test cannot pass vacuously.
    let max_abs = want.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    assert!(
        max_abs > 0.1,
        "oracle solution must be non-trivial, got {want:?}"
    );

    let max_diff = got
        .iter()
        .zip(want.iter())
        .fold(0.0f64, |m, (&g, &w)| m.max((g - w).abs()));
    // The observed 16.16-vs-f64 gap over this V-cycle is ~1e-4 (power-of-two diagonals keep the Jacobi
    // divide near-exact). Bound at 2e-3 — ~20x headroom over the numeric noise, yet a real kernel
    // defect (wrong matvec index, dropped phase, residual sign flip, mis-scaled transfer) is O(1) and
    // blows past it by 2-3 orders of magnitude.
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let diff = (g - w).abs();
        assert!(
            diff < 2e-3,
            "row {i}: IR={g} oracle={w} diff={diff} max_diff={max_diff} (got={got:?} want={want:?})"
        );
    }
}
