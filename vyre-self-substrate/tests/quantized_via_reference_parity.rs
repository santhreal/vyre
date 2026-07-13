//! End-to-end parity for the `math::quantized_dispatch` scaled INT4 `_via` entry points through the
//! shared faithful [`common::ReferenceEvalDispatcher`].
//!
//! Closes the quantized_dispatch mock-dispatcher-coherence family (see BACKLOG
//! `BUG-quantized-dispatch-family-over-feeds-backend-allocated-output`). Every scaled consumer's
//! result buffer is `BufferDecl::output` (backend-allocated, consumes NO dispatch input), but each
//! consumer used to pass a zero slot for it → a SYSTEMATIC over-feed that would fail the real
//! backend's strict input-count validation. `top1` additionally treated the kernel's SINGLE
//! interleaved `out` buffer (`[score, index-as-f32]` per batch) as two separate output buffers.
//!
//! Running each `_via` through the faithful dispatcher (which models the real backend's input+output
//! contract exactly, one input per input-consuming buffer, strict count; writable buffers returned
//! in binding order) executes the actual quantized kernel IR for the first time and proves the fixed
//! consumers match their `_cpu` oracles. The pre-fix over-feed surfaces here as a hard dispatch error.
#![cfg(feature = "cpu-parity")]

use vyre_primitives::math::quantized::{
    i4x8_batched_matmul_f32_scaled_cpu, i4x8_batched_matmul_top1_f32_scaled_cpu,
    i4x8_batched_matvec_f32_scaled_cpu, i4x8_dot_f32_scaled_cpu, i4x8_matvec_f32_scaled_cpu,
    pack_i4x8_cpu,
};
use vyre_self_substrate::math::quantized_dispatch::{
    i4x8_batched_matmul_f32_scaled_via, i4x8_batched_matmul_top1_f32_scaled_via,
    i4x8_batched_matvec_f32_scaled_via, i4x8_dot_f32_scaled_via, i4x8_matvec_f32_scaled_via,
};

mod common;
use common::ReferenceEvalDispatcher;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

/// A signed INT4 nibble value in `[-8, 7]`.
fn i4(state: &mut u32) -> i32 {
    (xorshift(state) % 16) as i32 - 8
}

/// A small f32 scale in `[0.0625, ~2.0]` (a clean power-of-two-ish scale, no denormals).
fn scale(state: &mut u32) -> f32 {
    (1 + xorshift(state) % 32) as f32 * 0.0625
}

/// An activation value in `[-4, 4)` step 0.25.
fn act(state: &mut u32) -> f32 {
    (xorshift(state) % 32) as f32 * 0.25 - 4.0
}

/// Pack `rows` rows of `cols` INT4 values (cols must be a multiple of 8) row-major.
fn pack_rows(state: &mut u32, rows: usize, cols: usize) -> Vec<u32> {
    let mut lanes = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        lanes.push(i4(state));
    }
    pack_i4x8_cpu(&lanes)
}

fn approx(got: f32, want: f32, ctx: &str) {
    let tol = 1.0e-3 + 1.0e-3 * want.abs();
    assert!(
        (got - want).abs() <= tol,
        "{ctx}: reference-eval f32 {got} deviates from cpu {want} beyond {tol}"
    );
}

#[test]
fn matvec_via_matches_cpu_over_generated_systems() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x4A17_0001u32;
    for case in 0..200u32 {
        let rows = 1 + (case as usize % 6);
        let cols = 8 * (1 + (case as usize % 3)); // 8, 16, or 24
        let weights = pack_rows(&mut state, rows, cols);
        let x: Vec<f32> = (0..cols).map(|_| act(&mut state)).collect();
        let row_scales: Vec<f32> = (0..rows).map(|_| scale(&mut state)).collect();

        let got = i4x8_matvec_f32_scaled_via(
            &dispatcher,
            &weights,
            &x,
            &row_scales,
            rows as u32,
            cols as u32,
        )
        .expect("matvec_via must dispatch the INT4 matvec kernel");
        let want = i4x8_matvec_f32_scaled_cpu(&weights, &x, &row_scales, rows as u32, cols as u32);
        assert_eq!(got.len(), want.len(), "case {case}: matvec output length");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            approx(*g, *w, &format!("case {case} matvec row {i}"));
        }
    }
}

#[test]
fn batched_matvec_via_matches_cpu_over_generated_systems() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x4A17_0002u32;
    for case in 0..200u32 {
        let rows = 1 + (case as usize % 5);
        let cols = 8 * (1 + (case as usize % 2)); // 8 or 16
        let batch = 1 + (case as usize % 4);
        let weights = pack_rows(&mut state, rows, cols);
        let x_batches: Vec<f32> = (0..batch * cols).map(|_| act(&mut state)).collect();
        let row_scales: Vec<f32> = (0..rows).map(|_| scale(&mut state)).collect();

        let got = i4x8_batched_matvec_f32_scaled_via(
            &dispatcher,
            &weights,
            &x_batches,
            &row_scales,
            batch as u32,
            rows as u32,
            cols as u32,
        )
        .expect("batched_matvec_via must dispatch the INT4 batched matvec kernel");
        let want = i4x8_batched_matvec_f32_scaled_cpu(
            &weights,
            &x_batches,
            &row_scales,
            batch as u32,
            rows as u32,
            cols as u32,
        );
        assert_eq!(got.len(), want.len(), "case {case}: batched matvec length");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            approx(*g, *w, &format!("case {case} batched_matvec {i}"));
        }
    }
}

#[test]
fn dot_via_matches_cpu_over_generated_vectors() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x4A17_0003u32;
    for case in 0..200u32 {
        let lanes = 8 * (1 + (case as usize % 4)); // 8..32
        let lhs = pack_rows(&mut state, 1, lanes);
        let rhs = pack_rows(&mut state, 1, lanes);
        let lhs_scale = scale(&mut state);
        let rhs_scale = scale(&mut state);

        let got =
            i4x8_dot_f32_scaled_via(&dispatcher, &lhs, &rhs, lhs_scale, rhs_scale, lanes as u32)
                .expect("dot_via must dispatch the INT4 dot kernel");
        let want = i4x8_dot_f32_scaled_cpu(&lhs, &rhs, lhs_scale, rhs_scale, lanes as u32);
        approx(got, want, &format!("case {case} dot"));
    }
}

#[test]
fn batched_matmul_via_matches_cpu_over_generated_systems() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x4A17_0004u32;
    for case in 0..200u32 {
        let rows = 1 + (case as usize % 5);
        let cols = 8 * (1 + (case as usize % 2));
        let batch = 1 + (case as usize % 4);
        let weights = pack_rows(&mut state, rows, cols);
        let activations = pack_rows(&mut state, batch, cols);
        let row_scales: Vec<f32> = (0..rows).map(|_| scale(&mut state)).collect();
        let batch_scales: Vec<f32> = (0..batch).map(|_| scale(&mut state)).collect();

        let got = i4x8_batched_matmul_f32_scaled_via(
            &dispatcher,
            &weights,
            &activations,
            &row_scales,
            &batch_scales,
            batch as u32,
            rows as u32,
            cols as u32,
        )
        .expect("batched_matmul_via must dispatch the INT4 batched matmul kernel");
        let want = i4x8_batched_matmul_f32_scaled_cpu(
            &weights,
            &activations,
            &row_scales,
            &batch_scales,
            batch as u32,
            rows as u32,
            cols as u32,
        );
        assert_eq!(got.len(), want.len(), "case {case}: batched matmul length");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            approx(*g, *w, &format!("case {case} batched_matmul {i}"));
        }
    }
}

#[test]
fn top1_via_matches_cpu_scores_and_indices_over_generated_systems() {
    let dispatcher = ReferenceEvalDispatcher;
    let mut state = 0x4A17_0005u32;
    for case in 0..200u32 {
        let rows = 2 + (case as usize % 5); // >=2 so argmax is non-trivial
        let cols = 8 * (1 + (case as usize % 2));
        let batch = 1 + (case as usize % 4);
        let weights = pack_rows(&mut state, rows, cols);
        let activations = pack_rows(&mut state, batch, cols);
        let row_scales: Vec<f32> = (0..rows).map(|_| scale(&mut state)).collect();
        let batch_scales: Vec<f32> = (0..batch).map(|_| scale(&mut state)).collect();

        let (scores, indices) = i4x8_batched_matmul_top1_f32_scaled_via(
            &dispatcher,
            &weights,
            &activations,
            &row_scales,
            &batch_scales,
            batch as u32,
            rows as u32,
            cols as u32,
        )
        .expect("top1_via must dispatch the INT4 top-1 kernel and de-interleave its output");
        let (want_scores, want_indices) = i4x8_batched_matmul_top1_f32_scaled_cpu(
            &weights,
            &activations,
            &row_scales,
            &batch_scales,
            batch as u32,
            rows as u32,
            cols as u32,
        );
        assert_eq!(
            scores.len(),
            batch,
            "case {case}: one top-1 score per batch"
        );
        assert_eq!(
            indices.len(),
            batch,
            "case {case}: one top-1 index per batch"
        );
        // The IR and cpu do the identical integer dot + scale, so they select the SAME argmax row.
        assert_eq!(
            indices, want_indices,
            "case {case}: top-1 argmax indices must match cpu"
        );
        for (b, (g, w)) in scores.iter().zip(want_scores.iter()).enumerate() {
            approx(*g, *w, &format!("case {case} top1 score batch {b}"));
        }
    }
}
