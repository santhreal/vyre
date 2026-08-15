//! Numerical contract of the eigendecomposition inside `math::tensor_train_decompose_step`.
//!
//! WHY: the step's truncated SVD is a Gram-matrix eigendecomposition, and that eigensolve has
//! exactly one Program-emitting owner in this crate,
//! `math::symmetric_eigen_jacobi::jacobi_eigen_body`, which the step composes through
//! `jacobi_eigen_region`. Nothing structural stops a future edit from splicing a second,
//! hand-rolled spelling back in: the existing reconstruction test
//! (`tensor_train_decompose_step_parity`) checks only `M ≈ U·remainder`, which any SVD-shaped
//! routine satisfies, so a re-hand-rolled eigensolve that drops the shared body's sign
//! canonicalization or mis-accumulates one rotation reconstructs the matrix just fine and
//! changes every reported number.
//!
//! This test pins the eigensolve itself, through the step's own scratch buffers, on invariants
//! that hold for ANY valid eigenbasis (so near-degenerate spectra are not flaky) plus the one
//! convention that is a deliberate choice rather than a mathematical fact:
//!
//! 1. eigenpair residual `G·v_k ≈ λ_k·v_k` against the Gram matrix `G = MᵀM` recomputed here,
//! 2. orthonormality `VᵀV ≈ I`,
//! 3. the sign canonicalization the shared body applies: the first component of each
//!    eigenvector column above `EIGENVECTOR_SIGN_EPSILON` is positive,
//! 4. singular values `σ_k² ≈ λ_k` taken in descending order, and left singular vectors
//!    `UᵀU ≈ I`.
//!
//! What it does not catch: a change to the shared body that is a valid eigendecomposition with
//! the same sign convention (a different rotation ORDER, say) is invisible here by design, and
//! so is any drift in the f64 CPU oracle, which is a separate independent implementation.
#![cfg(feature = "math")]

use vyre_primitives::math::eigenvector_column_sign::EIGENVECTOR_SIGN_EPSILON;
use vyre_primitives::math::tensor_train_decompose::tensor_train_decompose_step;
use vyre_primitives::wire::{decode_f32_le_bytes_all as unpack_f32, pack_f32_slice as pack_f32};
use vyre_reference::value::Value;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// Uniform f32 in `[-2, 2)`.
fn rand_f32(state: &mut u32) -> f32 {
    let bits = xorshift(state);
    ((bits >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
}

/// Everything the step writes for an `m x n` unfolding truncated to `r_next` ranks.
struct StepOutputs {
    /// `m x r_next` core, column `k` is the k-th left singular vector.
    u: Vec<f32>,
    /// `r_next x n` remainder, row `k` is `σ_k · v_kᵀ`.
    rem: Vec<f32>,
    /// `n x n` Gram matrix after rotation; its diagonal holds the eigenvalues.
    rotated: Vec<f32>,
    /// `n x n` eigenvector matrix, column `k` is the eigenvector for `rotated[k,k]`.
    evec: Vec<f32>,
}

fn run(matrix: &[f32], m: u32, n: u32, r_next: u32) -> StepOutputs {
    let program = tensor_train_decompose_step("input", "u", "rem", 1, m, n, r_next);
    let gram = (n * n) as usize;
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32(matrix)),
            Value::from(pack_f32(&vec![0.0f32; (m * r_next) as usize])),
            Value::from(pack_f32(&vec![0.0f32; (r_next * n) as usize])),
            Value::from(pack_f32(&vec![0.0f32; gram])),
            Value::from(pack_f32(&vec![0.0f32; gram])),
            Value::from(pack_f32(&vec![0.0f32; n as usize])),
        ],
    )
    .expect("tensor_train_decompose_step reference evaluation must succeed");
    let read = |name: &str| {
        unpack_f32(
            &outputs[vyre_reference::output_index(&program, name)
                .unwrap_or_else(|| panic!("{name} must be a program output"))]
            .to_bytes(),
        )
    };
    StepOutputs {
        u: read("u"),
        rem: read("rem"),
        rotated: read("tt_ata"),
        evec: read("tt_evec"),
    }
}

/// `G = MᵀM`, the symmetric matrix the step's eigensolve is applied to, in f64.
fn gram(matrix: &[f32], m: usize, n: usize) -> Vec<f64> {
    let mut g = vec![0.0f64; n * n];
    for row in 0..m {
        for col_a in 0..n {
            let a = f64::from(matrix[row * n + col_a]);
            for col_b in 0..n {
                g[col_a * n + col_b] += a * f64::from(matrix[row * n + col_b]);
            }
        }
    }
    g
}

/// Assert `G·v_k ≈ λ_k·v_k` and `VᵀV ≈ I` for the step's Gram eigendecomposition, and that the
/// shared body's sign canonicalization survived.
fn assert_eigen_contract(g: &[f64], out: &StepOutputs, n: usize, ctx: &str) {
    let g_mag = g.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
    let resid_tol = 2.0e-2 * g_mag;

    for k in 0..n {
        let lambda = f64::from(out.rotated[k * n + k]);
        let vk: Vec<f64> = (0..n).map(|i| f64::from(out.evec[i * n + k])).collect();

        let norm = vk.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (norm - 1.0).abs() <= 1.0e-2,
            "{ctx}: eigenvector {k} is not unit-norm (|v|={norm})"
        );

        for row in 0..n {
            let gv: f64 = (0..n).map(|col| g[row * n + col] * vk[col]).sum();
            let residual = (gv - lambda * vk[row]).abs();
            assert!(
                residual <= resid_tol,
                "{ctx}: (G·v - λv)[{row}] = {residual} exceeds {resid_tol} for eigenpair {k} \
                 (λ={lambda}); the Gram eigensolve is wrong"
            );
        }

        // The sign of an eigenvector is mathematically free, so the shared body fixes it: the
        // first component above the epsilon is positive. A re-hand-rolled eigensolve that skips
        // that pass reconstructs the matrix identically and flips this.
        let deciding = (0..n).find(|&i| out.evec[i * n + k].abs() > EIGENVECTOR_SIGN_EPSILON);
        if let Some(i) = deciding {
            assert!(
                out.evec[i * n + k] > 0.0,
                "{ctx}: eigenvector column {k} is not sign-canonical; first component above \
                 {EIGENVECTOR_SIGN_EPSILON} is evec[{i}][{k}]={}",
                out.evec[i * n + k]
            );
        }
    }

    for col_a in 0..n {
        for col_b in 0..n {
            let dot: f64 = (0..n)
                .map(|i| f64::from(out.evec[i * n + col_a]) * f64::from(out.evec[i * n + col_b]))
                .sum();
            let expected = if col_a == col_b { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() <= 1.0e-2,
                "{ctx}: (VᵀV)[{col_a},{col_b}] = {dot}, expected {expected}"
            );
        }
    }
}

/// Assert the truncation contract: `σ_k² ≈ λ_k` over the descending spectrum, `σ` descending,
/// and left singular vectors orthonormal wherever `σ_k` is above the step's own cutoff.
fn assert_truncation_contract(
    g: &[f64],
    out: &StepOutputs,
    m: usize,
    n: usize,
    r_next: usize,
    ctx: &str,
) {
    let g_mag = g.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);

    let mut spectrum: Vec<f64> = (0..n).map(|k| f64::from(out.rotated[k * n + k])).collect();
    spectrum.sort_by(|left, right| right.partial_cmp(left).expect("eigenvalues are finite"));

    let sigmas: Vec<f64> = (0..r_next)
        .map(|rank| {
            (0..n)
                .map(|col| {
                    let x = f64::from(out.rem[rank * n + col]);
                    x * x
                })
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    for rank in 0..r_next {
        let expected = spectrum[rank].max(0.0).sqrt();
        assert!(
            (sigmas[rank] - expected).abs() <= 2.0e-2 * g_mag.sqrt(),
            "{ctx}: |remainder row {rank}| = {} but the {rank}-th largest eigenvalue gives \
             σ = {expected}; the rank selection or the σ scaling is wrong",
            sigmas[rank]
        );
        if rank > 0 {
            assert!(
                sigmas[rank] <= sigmas[rank - 1] + 1.0e-3 * g_mag.sqrt(),
                "{ctx}: singular values are not descending: σ[{}]={}, σ[{rank}]={}",
                rank - 1,
                sigmas[rank - 1],
                sigmas[rank]
            );
        }
    }

    // `u_out` columns are M·v/σ, which is orthonormal for every rank the step actually emitted
    // (it writes zeros when σ is at or below 1e-6).
    for col_a in 0..r_next {
        if sigmas[col_a] <= 1.0e-3 {
            continue;
        }
        for col_b in 0..r_next {
            if sigmas[col_b] <= 1.0e-3 {
                continue;
            }
            let dot: f64 = (0..m)
                .map(|row| {
                    f64::from(out.u[row * r_next + col_a]) * f64::from(out.u[row * r_next + col_b])
                })
                .sum();
            let expected = if col_a == col_b { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() <= 2.0e-2,
                "{ctx}: (UᵀU)[{col_a},{col_b}] = {dot}, expected {expected}; the core columns \
                 are not left singular vectors"
            );
        }
    }
}

#[test]
fn gram_eigendecomposition_satisfies_the_eigenpair_contract() {
    let mut state = 0x51E7_C0DEu32;
    let mut spread = 0u32;
    for case in 0..80u32 {
        let n = 2 + xorshift(&mut state) % 4; // 2..=5 columns
        let m = n + xorshift(&mut state) % (n + 1); // n..=2n rows, so G is generically full rank
        let matrix: Vec<f32> = (0..(m * n)).map(|_| rand_f32(&mut state)).collect();
        let out = run(&matrix, m, n, n);
        let g = gram(&matrix, m as usize, n as usize);
        let ctx = format!("case {case} (m={m}, n={n})");
        assert_eigen_contract(&g, &out, n as usize, &ctx);
        assert_truncation_contract(&g, &out, m as usize, n as usize, n as usize, &ctx);

        let diag: Vec<f64> = (0..n as usize)
            .map(|k| f64::from(out.rotated[k * n as usize + k]))
            .collect();
        let hi = diag.iter().cloned().fold(f64::MIN, f64::max);
        let lo = diag.iter().cloned().fold(f64::MAX, f64::min);
        if hi - lo > 0.5 {
            spread += 1;
        }
    }
    assert!(
        spread > 70,
        "only {spread}/80 Gram matrices had a spread spectrum, so the rotation path is \
         under-exercised"
    );
}

#[test]
fn truncated_step_keeps_the_eigen_contract_on_the_retained_ranks() {
    // Truncation must not disturb the eigendecomposition it selects from: the scratch spectrum
    // and eigenvectors still have to satisfy the contract when only some ranks are emitted.
    let mut state = 0x0BAD_F00Du32;
    for case in 0..40u32 {
        let n = 3 + xorshift(&mut state) % 3; // 3..=5 columns
        let m = n + 1 + xorshift(&mut state) % n;
        let r_next = 1 + xorshift(&mut state) % (n - 1); // strictly truncating
        let matrix: Vec<f32> = (0..(m * n)).map(|_| rand_f32(&mut state)).collect();
        let out = run(&matrix, m, n, r_next);
        let g = gram(&matrix, m as usize, n as usize);
        let ctx = format!("case {case} (m={m}, n={n}, r_next={r_next})");
        assert_eigen_contract(&g, &out, n as usize, &ctx);
        assert_truncation_contract(&g, &out, m as usize, n as usize, r_next as usize, &ctx);
    }
}

#[test]
fn hand_checked_gram_spectrum_is_reproduced_exactly() {
    // M = [[1,2],[3,4],[5,6],[7,8]] gives G = MᵀM = [[84,100],[100,120]], whose eigenvalues are
    // (204 ± √41296)/2 = {203.60709, 0.3929119}. This is the registered fixture's matrix, so it
    // also pins the inventory oracle's derivation.
    let matrix = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let out = run(&matrix, 4, 2, 2);
    let g = gram(&matrix, 4, 2);
    assert_eigen_contract(&g, &out, 2, "hand-checked 4x2");
    assert_truncation_contract(&g, &out, 4, 2, 2, "hand-checked 4x2");

    let mut spectrum = [f64::from(out.rotated[0]), f64::from(out.rotated[3])];
    spectrum.sort_by(|left, right| right.partial_cmp(left).expect("finite"));
    assert!(
        (spectrum[0] - 203.607_09).abs() < 1.0e-2,
        "dominant Gram eigenvalue must be 203.60709, got {}",
        spectrum[0]
    );
    assert!(
        (spectrum[1] - 0.392_911_9).abs() < 1.0e-3,
        "trailing Gram eigenvalue must be 0.3929119, got {}",
        spectrum[1]
    );

    // Dominant eigenvector, sign-canonicalized: v1 = (0.64142, 0.76719).
    let dominant_col = usize::from(f64::from(out.rotated[3]) > f64::from(out.rotated[0]));
    let v = [
        f64::from(out.evec[dominant_col]),
        f64::from(out.evec[2 + dominant_col]),
    ];
    assert!(
        (v[0] - 0.641_423).abs() < 1.0e-4 && (v[1] - 0.767_187).abs() < 1.0e-4,
        "dominant eigenvector must be the sign-canonical (0.641423, 0.767187), got {v:?}"
    );
}
