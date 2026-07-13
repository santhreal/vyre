//! End-to-end parity for `math::tensor_train_compression::compress_cost_tensor_f32_via`.
//!
//! Closes the tensor-train mock-dispatcher gap (see BACKLOG
//! `BUG-tensor-train-decompose-step-is-stub-not-svd` + `SWEEP-self-substrate-mock-dispatcher-coherence`):
//! the consumer's old `TtDecomposeDispatcher` mock IGNORED the IR and returned hand-picked bytes,
//! and the underlying `tensor_train_decompose_step` was a STUB. Now the step is a real f32 truncated
//! SVD and this runs the WHOLE per-mode compression chain through the shared `ReferenceEvalDispatcher`
//! (real reference-eval of the kernel), then reconstructs the tensor from the TT cores and asserts it
//! matches the input within an f32 tolerance, the basis-invariant correctness contract for a
//! decomposition (a stub reconstructs to garbage).
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::math::tensor_train_compression::compress_cost_tensor_f32_via;

mod common;
use common::ReferenceEvalDispatcher;

/// Contract a TT-core chain to the scalar tensor value at multi-index `idx`.
///
/// `cores[k]` is `[ranks[k] x dims[k] x ranks[k+1]]` row-major, so
/// `cores[k][a, i_k, b] = cores[k][(a*dims[k] + i_k)*ranks[k+1] + b]`. Contraction carries a vector
/// over the current rank bond: `v_{k+1}[b] = Σ_a v_k[a] · core[k][a, i_k, b]`, `v_0 = [1]`.
fn tt_value(cores: &[Vec<f32>], dims: &[u32], ranks: &[u32], idx: &[usize]) -> f64 {
    let mut v = vec![1.0f64];
    for k in 0..dims.len() {
        let rk = ranks[k] as usize;
        let rk1 = ranks[k + 1] as usize;
        let nk = dims[k] as usize;
        let mut nv = vec![0.0f64; rk1];
        for a in 0..rk {
            for b in 0..rk1 {
                let c = f64::from(cores[k][(a * nk + idx[k]) * rk1 + b]);
                nv[b] += v[a] * c;
            }
        }
        v = nv;
    }
    v[0]
}

/// Reconstruct the full tensor from its TT cores (row-major over `dims`).
fn tt_reconstruct(cores: &[Vec<f32>], dims: &[u32], ranks: &[u32]) -> Vec<f64> {
    let total: usize = dims.iter().map(|&d| d as usize).product();
    let mut out = vec![0.0f64; total];
    let mut idx = vec![0usize; dims.len()];
    for flat in 0..total {
        // Decode the row-major multi-index.
        let mut rem = flat;
        for k in (0..dims.len()).rev() {
            let nk = dims[k] as usize;
            idx[k] = rem % nk;
            rem /= nk;
        }
        out[flat] = tt_value(cores, dims, ranks, &idx);
    }
    out
}

fn rel_error(recon: &[f64], original: &[f32]) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (r, o) in recon.iter().zip(original.iter()) {
        let d = r - f64::from(*o);
        num += d * d;
        den += f64::from(*o) * f64::from(*o);
    }
    num.sqrt() / den.sqrt().max(1e-9)
}

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[test]
fn two_mode_full_rank_compression_reconstructs_the_matrix() {
    // d=2 (single decompose step): a 3x2 matrix with full column rank r=2 must reconstruct exactly.
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x7A17_C0DEu32;
    for case in 0..40u32 {
        let dims = [3u32, 2];
        let ranks = [1u32, 2, 1]; // full column rank (r = n1 = 2, m = 3 >= 2)
        let tensor: Vec<f32> = (0..6)
            .map(|_| ((xorshift(&mut state) >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0)
            .collect();
        let compressed = compress_cost_tensor_f32_via(&dispatcher, &tensor, &dims, &ranks)
            .expect("compress_cost_tensor_f32_via must dispatch the TT-SVD chain");
        assert_eq!(compressed.cores.len(), 2, "case {case}: TT has d cores");
        let recon = tt_reconstruct(&compressed.cores, &dims, &ranks);
        let err = rel_error(&recon, &tensor);
        assert!(
            err < 3.0e-2,
            "case {case}: full-rank 3x2 TT reconstruction error {err} too large (stub would be O(1)); \
             tensor={tensor:?} cores={:?}",
            compressed.cores
        );
    }
}

#[test]
fn three_mode_rank1_tensor_compresses_and_reconstructs() {
    // d=3 (TWO chained decompose steps + a final core): an exact rank-1 outer product
    // T(i,j,k) = a[i]·b[j]·c[k] is TT-rank-1, so ranks [1,1,1,1] reconstruct it near-exactly. This
    // exercises the multi-step remainder carry, not just a single step.
    let dispatcher = ReferenceEvalDispatcher;
    let a = [1.5f32, -0.5];
    let b = [2.0f32, 1.0, -1.0];
    let c = [0.5f32, 3.0];
    let dims = [2u32, 3, 2];
    let ranks = [1u32, 1, 1, 1];
    let mut tensor = vec![0.0f32; 12];
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..2 {
                tensor[(i * 3 + j) * 2 + k] = a[i] * b[j] * c[k];
            }
        }
    }
    let compressed = compress_cost_tensor_f32_via(&dispatcher, &tensor, &dims, &ranks)
        .expect("compress_cost_tensor_f32_via must dispatch the 2-step chain");
    assert_eq!(compressed.cores.len(), 3, "TT has 3 cores");
    let recon = tt_reconstruct(&compressed.cores, &dims, &ranks);
    let err = rel_error(&recon, &tensor);
    assert!(
        err < 5.0e-2,
        "rank-1 3-mode tensor must reconstruct near-exactly, relative error {err}; recon={recon:?} \
         tensor={tensor:?}"
    );
}
