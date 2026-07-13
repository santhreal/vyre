//! End-to-end parity for `math::kfac_autotune_step::kfac_autotune_step_via`.
//!
//! Closes the ninth mock-dispatcher-coherence family (see BACKLOG
//! `SWEEP-self-substrate-mock-dispatcher-coherence`): the consumer's in-file `KfacDispatcher` mock
//! IGNORES the `_program` IR and hand-computes the inverse via `cpu_ref`, so it proves buffer
//! packing/grid plumbing but never executes the `kfac_block_inverse` kernel (a Gauss-Jordan
//! elimination in vyre IR). This runs the WHOLE dispatch path through the shared
//! `ReferenceEvalDispatcher` (real reference-eval of the kernel IR) and asserts the BASIS-INVARIANT
//! inverse contract `A · A⁻¹ ≈ I` per block, the correctness contract for a matrix inverse (a stub
//! reconstructs to garbage), plus exact known 2×2 inverses.
//!
//! kfac is ALSO the family that surfaced the shared dispatcher-coherence defect: its output buffer
//! `blocks_out` is plain-ReadWrite at binding 0. BEFORE the read-only `blocks_in` at binding 1, so
//! the old RO-sequential `ReferenceEvalDispatcher` mapped `blocks_in` to the wrong (zero `blocks_out`)
//! slot. The faithful dispatcher consumes one input per input-consuming buffer in buffer order,
//! matching the real backend, so kfac's already-correct 3-input consumer now runs.
#![cfg(feature = "cpu-parity")]

use vyre_self_substrate::math::kfac_autotune_step::kfac_autotune_step_via;

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// A uniform sample in `[-1, 1)` from the PRNG.
fn unit(state: &mut u32) -> f32 {
    (xorshift(state) >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
}

/// Build an `n×n` diagonally-dominant (hence invertible, and safe for pivot-free Gauss-Jordan)
/// matrix, row-major.
fn diagonally_dominant(state: &mut u32, n: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        let mut off = 0.0f32;
        for j in 0..n {
            if i != j {
                let v = unit(state);
                a[i * n + j] = v;
                off += v.abs();
            }
        }
        // Diagonal strictly dominates the row's off-diagonal magnitude sum → nonsingular.
        let sign = if xorshift(state) & 1 == 0 { 1.0 } else { -1.0 };
        a[i * n + i] = sign * (off + 1.0 + unit(state).abs());
    }
    a
}

/// Largest absolute deviation of `A · A_inv` from the identity, over one `n×n` block.
fn identity_residual(a: &[f32], a_inv: &[f32], n: usize) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for k in 0..n {
                acc += f64::from(a[i * n + k]) * f64::from(a_inv[k * n + j]);
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((acc - expected).abs());
        }
    }
    worst
}

#[test]
fn inverse_satisfies_a_times_ainv_is_identity_over_generated_blocks() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x51F0_1234u32;
    let mut nontrivial = 0u32;
    for case in 0..300u32 {
        let n = 2 + (case % 4) as usize; // n = 2..5
        let num_blocks = 1 + (case % 3); // 1..3 independent blocks
        let mut blocks = Vec::new();
        let mut per_block = Vec::new();
        for _ in 0..num_blocks {
            let a = diagonally_dominant(&mut state, n);
            per_block.push(a.clone());
            blocks.extend_from_slice(&a);
        }

        let inv = kfac_autotune_step_via(&dispatcher, &blocks, num_blocks, n as u32)
            .expect("kfac_autotune_step_via must dispatch the block-inverse kernel");
        assert_eq!(
            inv.len(),
            num_blocks as usize * n * n,
            "case {case}: inverse has num_blocks*n*n entries"
        );

        for (b, a) in per_block.iter().enumerate() {
            let block_inv = &inv[b * n * n..(b + 1) * n * n];
            let residual = identity_residual(a, block_inv, n);
            assert!(
                residual < 1.0e-3,
                "case {case} block {b} (n={n}): A·A⁻¹ deviates from I by {residual} (a stub is O(1)); \
                 A={a:?} A_inv={block_inv:?}"
            );
            // A diagonally-dominant block with real off-diagonals is a non-trivial inverse
            // (not itself diagonal), so this exercises full Gauss-Jordan elimination.
            if a.iter()
                .enumerate()
                .any(|(idx, &v)| idx % (n + 1) != 0 && v.abs() > 0.1)
            {
                nontrivial += 1;
            }
        }
    }
    assert!(
        nontrivial > 250,
        "expected >250 non-diagonal (full-elimination) blocks, got {nontrivial}"
    );
}

#[test]
fn inverts_known_diagonal_and_dense_two_by_two_blocks() {
    let dispatcher = ReferenceEvalDispatcher;

    // Two blocks in ONE dispatch: identity and diagonal[2,4] → [I | diag(0.5,0.25)].
    let two_diag = vec![
        1.0, 0.0, 0.0, 1.0, // block 0 = I → I
        2.0, 0.0, 0.0, 4.0, // block 1 = diag(2,4) → diag(0.5,0.25)
    ];
    let inv = kfac_autotune_step_via(&dispatcher, &two_diag, 2, 2).unwrap();
    assert_eq!(inv[0..4], [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(inv[4..8], [0.5, 0.0, 0.0, 0.25]);

    // Dense symmetric block [[4,3],[3,2]], det = -1 → inverse [[-2,3],[3,-4]].
    let dense = vec![4.0, 3.0, 3.0, 2.0];
    let inv = kfac_autotune_step_via(&dispatcher, &dense, 1, 2).unwrap();
    let residual = identity_residual(&dense, &inv, 2);
    assert!(
        residual < 1.0e-5,
        "dense 2×2 A·A⁻¹ deviates from I by {residual}; inv={inv:?}"
    );
    for (got, want) in inv.iter().zip([-2.0f32, 3.0, 3.0, -4.0].iter()) {
        assert!(
            (got - want).abs() < 1.0e-4,
            "dense 2×2 inverse {inv:?} != [-2,3,3,-4]"
        );
    }
}
