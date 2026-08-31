//! RMT-based deterministic spectrum projection primitive.
//!
//! Random matrix theory predicts the bulk spectrum of large random
//! matrices (Marchenko-Pastur, Wigner). Recent work (Pennington 2017,
//! Martin 2021 weight-watcher, Edelman 2024) uses RMT to PREDICT
//! training dynamics and SHAPE the weight spectrum. This file ships
//! the **Marchenko-Pastur edge-clipping** primitive  -  given the
//! eigenvalue/singular-value distribution of a matrix, clip values
//! outside the predicted bulk to the bulk-edge.
//!
//! Composes with chebyshev_filter for the spectrum projection
//! without computing the eigendecomposition.
//!
//! # Why this primitive is dual-use
//!
//! | Consumer | Use |
//! |---|---|
//! | future `vyre-libs::ml::implicit_reg` | implicit regularization without hyperparameters |
//! | future `vyre-libs::ml::training_dynamics` | training-dynamics-aware optimizers |
//! | `vyre-foundation::transform` spectral scheduling | clip outlier eigenvalues in vyre's dispatch graph |

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Expr, Program};

/// Op id.
pub const OP_ID: &str = "vyre-libs::math::mp_edge_clip";

/// Marchenko-Pastur upper edge: `(1 + √(p/n))²` where `p, n` are
/// matrix dimensions and `σ²` = entry variance. The caller passes a
/// scaled upper bound `mp_edge` (16.16 fp).
///
/// Emit: clip each eigenvalue to `min(mp_edge, eigenvalue)`.
#[must_use]
pub fn mp_edge_clip(eigenvalues: &str, mp_edge: &str, out: &str, n: u32) -> Program {
    if n == 0 {
        return trap_program(
            OP_ID,
            Some((out, DataType::U32)),
            format!("Fix: mp_edge_clip requires n > 0, got {n}."),
        );
    }

    crate::math::u32_binary_map::u32_vector_scalar_map_program(
        OP_ID,
        eigenvalues,
        mp_edge,
        out,
        n,
        Expr::min,
    )
}

/// Compute the Marchenko-Pastur upper edge for an `m × n` matrix with
/// entry variance `sigma_sq`.
#[must_use]
pub fn mp_upper_edge(m: u32, n: u32, sigma_sq: f64) -> f64 {
    if m == 0 || n == 0 {
        return f64::NAN;
    }
    let q = (m.min(n) as f64) / (m.max(n) as f64);
    let factor = (1.0 + q.sqrt()).powi(2);
    sigma_sq * factor
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || {
            mp_edge_clip("a", "b", "out", 4)
        },
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[1, 5, 10, 3]),
                vyre_primitives::wire::pack_u32_slice(&[4]),
            ]]
        }),
        Some(|| {
            vec![vec![vec![
                0x01, 0x00, 0x00, 0x00, // 1
                0x04, 0x00, 0x00, 0x00, // 4
                0x04, 0x00, 0x00, 0x00, // 4
                0x03, 0x00, 0x00, 0x00, // 3
            ]]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_reference::composition_witness::mp_edge_clip_witness as mp_edge_clip_cpu;

    fn try_mp_edge_clip_cpu_into(
        eigenvalues: &[f64],
        edge: f64,
        out: &mut Vec<f64>,
    ) -> Result<(), String> {
        let res = mp_edge_clip_cpu(eigenvalues, edge);
        out.clear();
        out.extend_from_slice(&res);
        Ok(())
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn cpu_mp_edge_square_matrix() {
        // m = n = 100, σ² = 1 → MP edge = 4 (since q = 1, factor = (1+1)² = 4)
        let edge = mp_upper_edge(100, 100, 1.0);
        assert!(approx_eq(edge, 4.0));
    }

    #[test]
    fn cpu_mp_edge_tall_matrix() {
        // m = 100, n = 25, σ² = 1, q = 0.25 → factor = (1+0.5)² = 2.25
        let edge = mp_upper_edge(100, 25, 1.0);
        assert!(approx_eq(edge, 2.25));
    }

    #[test]
    fn cpu_clip_below_edge_unchanged() {
        let eig = vec![1.0, 2.0, 3.0];
        let out = mp_edge_clip_cpu(&eig, 4.0);
        assert_eq!(out, eig);
    }

    #[test]
    fn cpu_clip_above_edge_clamped() {
        let eig = vec![1.0, 5.0, 10.0];
        let out = mp_edge_clip_cpu(&eig, 4.0);
        assert!(approx_eq(out[0], 1.0));
        assert!(approx_eq(out[1], 4.0));
        assert!(approx_eq(out[2], 4.0));
    }

    #[test]
    fn cpu_clip_into_reuses_output_and_truncates_stale_tail() {
        let mut out = Vec::with_capacity(4);
        out.extend_from_slice(&[99.0, 98.0, 97.0, 96.0]);
        let ptr = out.as_ptr();
        let capacity = out.capacity();

        try_mp_edge_clip_cpu_into(&[1.0, 5.0, 10.0], 4.0, &mut out)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - MP edge clip CPU oracle should reuse caller-owned output");

        assert_eq!(out, vec![1.0, 4.0, 4.0]);
        assert_eq!(out.as_ptr(), ptr);
        assert_eq!(out.capacity(), capacity);

        try_mp_edge_clip_cpu_into(&[8.0], 4.0, &mut out)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - MP edge clip CPU oracle should truncate stale output");

        assert_eq!(out, vec![4.0]);
        assert_eq!(out.as_ptr(), ptr);
        assert_eq!(out.capacity(), capacity);
    }

    #[test]
    fn generated_cpu_clip_matches_scalar_reference() {
        let mut out = Vec::new();
        for case in 0..2048usize {
            let len = case % 129;
            let edge = ((case % 31) as f64 - 15.0) / 3.0;
            let eigenvalues: Vec<f64> = (0..len)
                .map(|idx| ((idx * 17 + case) % 97) as f64 / 5.0 - 9.0)
                .collect();

            try_mp_edge_clip_cpu_into(&eigenvalues, edge, &mut out)
                .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - generated MP edge clip CPU oracle should evaluate");

            assert_eq!(out.len(), eigenvalues.len(), "case {case}: output length");
            for idx in 0..out.len() {
                assert!(
                    approx_eq(out[idx], eigenvalues[idx].min(edge)),
                    "case {case} idx {idx}: expected {}, got {}",
                    eigenvalues[idx].min(edge),
                    out[idx]
                );
            }
        }
    }

    #[test]
    fn ir_program_buffer_layout() {
        let p = mp_edge_clip("e", "edge", "out", 16);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        assert_eq!(p.buffers[0].count(), 16);
        assert_eq!(p.buffers[1].count(), 1);
        assert_eq!(p.buffers[2].count(), 16);
    }

    #[test]
    fn zero_n_traps() {
        let p = mp_edge_clip("e", "edge", "out", 0);
        assert!(p.stats().trap());
    }
}
