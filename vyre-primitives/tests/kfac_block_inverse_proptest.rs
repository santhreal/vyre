//! Tier 3 - Property: proptest over random well-conditioned block-diagonal matrices for
//! `math::kfac_block_inverse`. The shipped file has a SINGLE 2x2 parity case (`test_parity_2x2`);
//! this drives the GPU IR through `reference_eval` against `cpu_ref` over thousands of random
//! instances with varying block size `n` and `num_blocks`, exercising the parallel per-block
//! Gauss-Jordan elimination + the flat `b*n*n + i*n + j` indexing that a single hand case cannot.
//!
//! Matrices are STRICTLY DIAGONALLY DOMINANT (diagonal in `[2, 4]`, off-diagonals in `[-0.5, 0.5]`,
//! so `|a_ii| > Σ_{j≠i} |a_ij|`): guaranteed non-singular AND well-conditioned, which keeps the
//! no-pivoting Gauss-Jordan stable and bounds the f32 divergence between the two identical
//! implementations. The GPU IR and `cpu_ref` run the SAME elimination order, so agreement is tight;
//! a real defect (wrong block offset, mis-parallelized elimination, a dropped row op) diverges far
//! beyond the tolerance. Complements `test_parity_2x2` with randomized (n, num_blocks, values)
//! breadth.
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::math::kfac_block_inverse::{cpu_ref, kfac_block_inverse};
use vyre_reference::value::Value;

/// Well-conditioned tolerance: both sides run identical f32 Gauss-Jordan, and strict diagonal
/// dominance keeps the condition number small, so the interpreter-vs-Rust f32 gap stays well under
/// this. A real elimination/indexing bug diverges by O(1), far above it.
const TOL: f32 = 1.0e-3;

fn run_ir(blocks_in: &[f32], num_blocks: u32, n: u32) -> Vec<f32> {
    let program = kfac_block_inverse("bo", "bi", "s", num_blocks, n);
    let cells = (num_blocks * n * n) as usize;
    let pack = |data: &[f32]| Value::from(vyre_primitives::wire::pack_f32_slice(data));
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            pack(&vec![0.0f32; cells]), // bo (binding 0, ReadWrite), the inverse output
            pack(blocks_in),            // bi (binding 1, ReadOnly)
            pack(&vec![0.0f32; cells]), // s  (binding 2, ReadWrite scratch)
        ],
    )
    .expect("kfac_block_inverse reference evaluation must succeed");
    // results[0] is the first ReadWrite buffer, `bo`.
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

prop_compose! {
    /// A random strictly-diagonally-dominant block-diagonal matrix set: `num_blocks` blocks, each
    /// `n x n`, flattened row-major per block. Diagonal in [2,4], off-diagonal in [-0.5,0.5].
    fn arb_blocks()(num_blocks in 1u32..=6, n in 1u32..=4)
        (num_blocks in Just(num_blocks),
         n in Just(n),
         // one f32 per cell to fill in; diagonal cells get replaced by a strong positive value.
         raw in prop::collection::vec(-0.5f32..0.5f32, (num_blocks * n * n) as usize),
         diag in prop::collection::vec(2.0f32..4.0f32, (num_blocks * n) as usize))
        -> (u32, u32, Vec<f32>) {
        let nn = (n * n) as usize;
        let mut blocks = raw;
        for b in 0..num_blocks as usize {
            for i in 0..n as usize {
                let idx = b * nn + i * n as usize + i;
                blocks[idx] = diag[b * n as usize + i];
            }
        }
        (num_blocks, n, blocks)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn kfac_ir_matches_cpu_over_well_conditioned_blocks(
        (num_blocks, n, blocks) in arb_blocks()
    ) {
        let got = run_ir(&blocks, num_blocks, n);
        let want = cpu_ref(&blocks, num_blocks, n);
        prop_assert_eq!(got.len(), want.len(), "output length must match cpu_ref");
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            prop_assert!(
                (g - w).abs() <= TOL,
                "cell {} diverged: IR {} vs cpu_ref {} (num_blocks={}, n={}, blocks={:?})",
                idx, g, w, num_blocks, n, blocks
            );
        }
    }

    /// The inverse must actually invert: block · inverse ≈ identity (an independent cross-check that
    /// does not rely on `cpu_ref`, so a shared oracle+IR bug cannot hide here).
    #[test]
    fn kfac_ir_product_with_input_is_identity(
        (num_blocks, n, blocks) in arb_blocks()
    ) {
        let inv = run_ir(&blocks, num_blocks, n);
        let nn = (n * n) as usize;
        let ns = n as usize;
        for b in 0..num_blocks as usize {
            let base = b * nn;
            for i in 0..ns {
                for j in 0..ns {
                    // (A · A^{-1})[i,j] = Σ_k A[i,k] · inv[k,j]
                    let mut acc = 0.0f32;
                    for k in 0..ns {
                        acc += blocks[base + i * ns + k] * inv[base + k * ns + j];
                    }
                    let expected = if i == j { 1.0 } else { 0.0 };
                    prop_assert!(
                        (acc - expected).abs() <= 5.0e-3,
                        "block {} product[{},{}] = {} != {} (n={})",
                        b, i, j, acc, expected, n
                    );
                }
            }
        }
    }
}
