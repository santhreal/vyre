//! Correctness of the f32 serial Jacobi symmetric eigensolver IR (`math::symmetric_eigen_jacobi`).
//!
//! The kernel is the numerical core of the real tensor-train SVD. It is verified by the
//! basis/order-INVARIANT eigenpair contract rather than element-wise vs an f64 reference: for the
//! reported eigenvalues `λ` and eigenvector matrix `V` (columns), every pair must satisfy
//! `A·v_k ≈ λ_k·v_k` (small residual against the ORIGINAL matrix, the kernel overwrites its `a`
//! buffer with the rotated diagonal form) and `V` must be orthonormal (`VᵀV ≈ I`). These hold for
//! any valid eigenbasis, so near-degenerate spectra (which admit different-but-valid bases) do not
//! make the test flaky. A stub or a broken rotation fails the residual by orders of magnitude.
#![cfg(feature = "math")]

use vyre_primitives::math::symmetric_eigen_jacobi::symmetric_eigen_jacobi;
use vyre_primitives::wire::{decode_f32_le_bytes_all as unpack_f32, pack_f32_slice as pack_f32};
use vyre_reference::value::Value;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Uniform f32 in [-2, 2) from the PRNG.
fn rand_f32(state: &mut u32) -> f32 {
    let bits = xorshift(state);
    ((bits >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
}

/// Run the eigensolver IR on a symmetric `n x n` matrix (row-major f32) and return `(eigenvalues,
/// eigenvectors)`; eigenvectors are row-major with column `k` the eigenvector for eigenvalue `k`.
fn run(a: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let program = symmetric_eigen_jacobi("a", "evec", "eval", n as u32);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32(a)),
            Value::from(pack_f32(&vec![0.0f32; n * n])),
            Value::from(pack_f32(&vec![0.0f32; n])),
        ],
    )
    .expect("symmetric_eigen_jacobi reference evaluation must succeed");
    let eigvals = unpack_f32(
        &outputs[vyre_reference::output_index(&program, "eval").expect("eval output")].to_bytes(),
    );
    let eigvecs = unpack_f32(
        &outputs[vyre_reference::output_index(&program, "evec").expect("evec output")].to_bytes(),
    );
    (eigvals, eigvecs)
}

/// Assert the eigenpair contract: `A·v_k ≈ λ_k·v_k` for every k, and `VᵀV ≈ I`.
fn assert_eigenpairs(a: &[f32], eigvals: &[f32], eigvecs: &[f32], n: usize, ctx: &str) {
    // Scale the residual tolerance by the matrix magnitude (Frobenius-ish).
    let mut a_mag = 0.0f64;
    for &x in a {
        a_mag += f64::from(x) * f64::from(x);
    }
    let a_mag = a_mag.sqrt().max(1.0);
    let resid_tol = 2.0e-2 * a_mag;

    for k in 0..n {
        let lambda = f64::from(eigvals[k]);
        // v_k = column k of V.
        let vk: Vec<f64> = (0..n).map(|i| f64::from(eigvecs[i * n + k])).collect();
        // norm of v_k should be ~1 (orthonormal columns); skip a numerically-zero column.
        let vk_norm: f64 = vk.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (vk_norm - 1.0).abs() <= 1.0e-2,
            "{ctx}: eigenvector {k} not unit-norm (|v|={vk_norm})"
        );
        for row in 0..n {
            let mut av = 0.0f64;
            for col in 0..n {
                av += f64::from(a[row * n + col]) * vk[col];
            }
            let residual = (av - lambda * vk[row]).abs();
            assert!(
                residual <= resid_tol,
                "{ctx}: (A·v - λv)[{row}] = {residual} exceeds tol {resid_tol} for eigenpair {k} \
                 (λ={lambda}); a_mag={a_mag}"
            );
        }
    }

    // Orthonormality: VᵀV ≈ I.
    for a_col in 0..n {
        for b_col in 0..n {
            let mut dot = 0.0f64;
            for i in 0..n {
                dot += f64::from(eigvecs[i * n + a_col]) * f64::from(eigvecs[i * n + b_col]);
            }
            let expected = if a_col == b_col { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() <= 1.0e-2,
                "{ctx}: (VᵀV)[{a_col},{b_col}] = {dot}, expected {expected}"
            );
        }
    }
}

#[test]
fn jacobi_diagonalizes_random_symmetric_matrices() {
    let mut state = 0x0D15_EA5Eu32;
    let mut nondegenerate = 0u32;
    for case in 0..120u32 {
        let n = 2 + (xorshift(&mut state) % 4) as usize; // 2..=5
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            for j in i..n {
                let v = rand_f32(&mut state);
                a[i * n + j] = v;
                a[j * n + i] = v; // symmetric
            }
        }
        let (eigvals, eigvecs) = run(&a, n);
        assert_eq!(eigvals.len(), n, "case {case}: eigenvalue count");
        assert_eq!(eigvecs.len(), n * n, "case {case}: eigenvector count");
        assert_eigenpairs(&a, &eigvals, &eigvecs, n, &format!("case {case} (n={n})"));

        // Count cases where the spectrum is genuinely spread (not near-scalar), so the rotation path
        // is exercised rather than trivially-already-diagonal inputs.
        let max_ev = eigvals.iter().cloned().fold(f32::MIN, f32::max);
        let min_ev = eigvals.iter().cloned().fold(f32::MAX, f32::min);
        if (max_ev - min_ev) > 0.5 {
            nondegenerate += 1;
        }
    }
    assert!(
        nondegenerate > 100,
        "only {nondegenerate}/120 matrices had a spread spectrum, the off-diagonal rotation path is \
         under-exercised"
    );
}

#[test]
fn jacobi_recovers_known_2x2_spectrum() {
    // A = [[0, 1], [1, 0]] has eigenvalues {+1, -1} with eigenvectors (1, ±1)/√2.
    let a = vec![0.0f32, 1.0, 1.0, 0.0];
    let (eigvals, eigvecs) = run(&a, 2);
    assert_eigenpairs(&a, &eigvals, &eigvecs, 2, "swap-2x2");
    let mut sorted = eigvals.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert!(
        (sorted[0] - (-1.0)).abs() <= 1e-3 && (sorted[1] - 1.0).abs() <= 1e-3,
        "eigenvalues must be {{-1, +1}}, got {eigvals:?}"
    );
}

#[test]
fn jacobi_handles_already_diagonal_input() {
    // Diagonal A: eigenvalues are the diagonal, eigenvectors are the identity; no rotation needed.
    let a = vec![
        3.0f32, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0,
    ];
    let (eigvals, eigvecs) = run(&a, 4);
    assert_eigenpairs(&a, &eigvals, &eigvecs, 4, "diagonal-4x4");
    // Diagonal is untouched (no off-diagonal exceeds the threshold), so eigenvalues == diagonal in
    // order and V == I.
    assert_eq!(eigvals, vec![3.0, 5.0, 0.0, 7.0]);
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((eigvecs[i * 4 + j] - expected).abs() <= 1e-6);
        }
    }
}
