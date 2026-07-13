//! Correctness of the real f32 per-mode TT-SVD step (`math::tensor_train_decompose_step`).
//!
//! Before this was a STUB (leading-column copy + identity remainder, see BACKLOG
//! `BUG-tensor-train-decompose-step-is-stub-not-svd`). It now performs a genuine truncated SVD of the
//! `m x n` unfolding `M`: `core = U`, `remainder = S·Vᵀ`, so `M ≈ U·(S·Vᵀ)`. The decomposition is
//! basis-dependent, so we verify the RECONSTRUCTION `M ≈ U·remainder` (invariant to eigenbasis /
//! sign / ordering), not the factors element-wise:
//!   * full rank (`r_next == n`): reconstruction is near-exact, a stub fails this by orders of
//!     magnitude;
//!   * rank-1 input truncated to `r_next == 1`: the dominant component is kept exactly;
//!   * a hand-checked diagonal case.
#![cfg(feature = "math")]

use vyre_primitives::math::tensor_train_decompose::tensor_train_decompose_step;
use vyre_primitives::wire::{decode_f32_le_bytes_all as unpack_f32, pack_f32_slice as pack_f32};
use vyre_reference::value::Value;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn rand_f32(state: &mut u32) -> f32 {
    let bits = xorshift(state);
    ((bits >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
}

/// Run the step on an `m x n` matrix `M` (r_prev=1, nk=m, rem=n) and return `(u [m x r_next],
/// remainder [r_next x n])`.
fn run(matrix: &[f32], m: u32, n: u32, r_next: u32) -> (Vec<f32>, Vec<f32>) {
    let program = tensor_train_decompose_step("input", "u", "rem", 1, m, n, r_next);
    let u_count = (m * r_next) as usize;
    let rem_count = (r_next * n) as usize;
    let gram = (n * n) as usize;
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack_f32(matrix)),                    // input_matrix (RO)
            Value::from(pack_f32(&vec![0.0f32; u_count])),    // u_out
            Value::from(pack_f32(&vec![0.0f32; rem_count])),  // rem_out
            Value::from(pack_f32(&vec![0.0f32; gram])),       // tt_ata
            Value::from(pack_f32(&vec![0.0f32; gram])),       // tt_evec
            Value::from(pack_f32(&vec![0.0f32; n as usize])), // tt_eval
        ],
    )
    .expect("tensor_train_decompose_step reference evaluation must succeed");
    let u = unpack_f32(
        &outputs[vyre_reference::output_index(&program, "u").expect("u output")].to_bytes(),
    );
    let rem = unpack_f32(
        &outputs[vyre_reference::output_index(&program, "rem").expect("rem output")].to_bytes(),
    );
    (u, rem)
}

/// Reconstruct `M̂ = U · remainder` (`m x n`).
fn reconstruct(u: &[f32], rem: &[f32], m: usize, n: usize, r_next: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f64;
            for k in 0..r_next {
                acc += f64::from(u[row * r_next + k]) * f64::from(rem[k * n + col]);
            }
            out[row * n + col] = acc as f32;
        }
    }
    out
}

fn rel_frobenius_error(m_hat: &[f32], m: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (h, o) in m_hat.iter().zip(m.iter()) {
        let d = f64::from(*h) - f64::from(*o);
        num += d * d;
        den += f64::from(*o) * f64::from(*o);
    }
    (num.sqrt()) / den.sqrt().max(1e-9)
}

#[test]
fn full_rank_step_reconstructs_the_matrix() {
    let mut state = 0x7EE7_5EEDu32;
    let mut nontrivial = 0u32;
    for case in 0..80u32 {
        let n = 2 + xorshift(&mut state) % 4; // 2..=5 columns
        let m = n + xorshift(&mut state) % (n + 1); // n..=2n rows (m >= n so full column rank possible)
        let matrix: Vec<f32> = (0..(m * n)).map(|_| rand_f32(&mut state)).collect();
        // Full column rank: r_next = n.
        let (u, rem) = run(&matrix, m, n, n);
        let m_hat = reconstruct(&u, &rem, m as usize, n as usize, n as usize);
        let err = rel_frobenius_error(&m_hat, &matrix);
        assert!(
            err < 2.0e-2,
            "case {case} (m={m}, n={n}): full-rank reconstruction relative error {err} too large \
             (a stub would be O(1)); M={matrix:?} U={u:?} rem={rem:?}"
        );
        let mag: f64 = matrix.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
        if mag.sqrt() > 1.0 {
            nontrivial += 1;
        }
    }
    assert!(
        nontrivial > 70,
        "only {nontrivial}/80 matrices had meaningful magnitude, strengthen the generator"
    );
}

#[test]
fn rank1_input_truncated_to_rank1_is_kept_exactly() {
    // M = a · bᵀ (outer product) is exactly rank 1; truncating to r_next = 1 must reconstruct it.
    let a = [1.0f32, 2.0, 3.0, 4.0]; // m = 4
    let b = [2.0f32, -1.0, 0.5]; // n = 3
    let (m, n) = (4u32, 3u32);
    let mut matrix = vec![0.0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            matrix[i * n as usize + j] = a[i] * b[j];
        }
    }
    let (u, rem) = run(&matrix, m, n, 1);
    let m_hat = reconstruct(&u, &rem, m as usize, n as usize, 1);
    let err = rel_frobenius_error(&m_hat, &matrix);
    assert!(
        err < 1.0e-2,
        "rank-1 outer product truncated to rank 1 must reconstruct exactly, relative error {err}"
    );
}

#[test]
fn diagonal_step_recovers_scaled_axes() {
    // A 3x3 diagonal M = diag(4, 1, 9); full-rank SVD reconstructs it. The singular values are the
    // magnitudes {9, 4, 1}; reconstruction must be near-exact.
    let matrix = vec![4.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 9.0];
    let (u, rem) = run(&matrix, 3, 3, 3);
    let m_hat = reconstruct(&u, &rem, 3, 3, 3);
    let err = rel_frobenius_error(&m_hat, &matrix);
    assert!(
        err < 1.0e-2,
        "diagonal reconstruction relative error {err}; U={u:?} rem={rem:?}"
    );
}
