//! Self-substrate wrappers for numerical optimizer and scientific kernels.
//!
//! These functions give scheduler and optimizer code named self-consumers for
//! math primitives without reimplementing the primitive algorithms here.

use crate::math::{
    dp_accountant::gaussian_rdp_step,
    preconditioner::{newton_schulz_poly5_f32, newton_schulz_y_step},
    randomized_svd::randomized_projection_step,
};
use vyre_foundation::ir::Program;

/// Build a randomized projection dispatch for low-rank optimizer telemetry.
#[must_use]
pub fn dispatch_randomized_projection(
    a: &str,
    omega: &str,
    y: &str,
    m: u32,
    n: u32,
    l: u32,
) -> Program {
    randomized_projection_step(a, omega, y, m, n, l)
}

/// Build a Newton-Schulz Y-update dispatch.
#[must_use]
pub fn dispatch_newton_schulz_y_step(y_curr: &str, yzy: &str, y_next: &str, n: u32) -> Program {
    newton_schulz_y_step(y_curr, yzy, y_next, n)
}

/// Build the fused f32 Newton-Schulz quintic polynomial dispatch.
#[must_use]
pub fn dispatch_newton_schulz_poly5_f32(mat: &str, output: &str, rows: u32, cols: u32) -> Program {
    newton_schulz_poly5_f32(mat, output, rows, cols)
}

/// Build a quantized Sinkhorn fixed-point dispatch.
///
/// A re-export rather than a facade: a facade would forward the same binding
/// record and the same extents and add nothing, and the only thing it could add
/// is a restatement of the primitive's parameter list that is free to drift from
/// it. The composition names the primitive; it does not repeat it.
pub use crate::math::sinkhorn_iterate::sinkhorn_iterate as dispatch_sinkhorn_iterate;

/// Build a Gaussian RDP per-step dispatch.
#[must_use]
pub fn dispatch_gaussian_rdp_step(
    alpha: &str,
    sigma_squared: &str,
    out: &str,
    count: u32,
) -> Program {
    gaussian_rdp_step(alpha, sigma_squared, out, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::sinkhorn_iterate::{SinkhornBuffers, SinkhornExtents};
    use vyre_foundation::ir::Node;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6 * (1.0 + a.abs() + b.abs())
    }

    fn program_generator(program: &Program) -> &str {
        let Some(Node::Region { generator, .. }) = program.entry.first() else {
            panic!("Fix: numerical kernel Program must start with a Region.");
        };
        generator.as_str()
    }

    /// The binding names the Sinkhorn dispatch tests build against.
    const SINKHORN_FIXTURE: SinkhornBuffers<'static> = SinkhornBuffers::CANONICAL;

    fn reference_randomized_projection(
        a: &[f64],
        omega: &[f64],
        m: u32,
        n: u32,
        l: u32,
    ) -> Vec<f64> {
        let mut y = Vec::new();
        reference_randomized_projection_into(a, omega, m, n, l, &mut y);
        y
    }

    fn reference_randomized_projection_into(
        a: &[f64],
        omega: &[f64],
        m: u32,
        n: u32,
        l: u32,
        y: &mut Vec<f64>,
    ) {
        vyre_reference::composition_witness::dense_matrix_multiply_witness_into(
            a, omega, m as usize, n as usize, l as usize, y,
        );
    }

    fn reference_modified_gram_schmidt(y: &[f64], m: u32, l: u32) -> Vec<f64> {
        let mut q = Vec::new();
        reference_modified_gram_schmidt_into(y, m, l, &mut q);
        q
    }

    fn reference_modified_gram_schmidt_into(y: &[f64], m: u32, l: u32, q: &mut Vec<f64>) {
        vyre_reference::composition_witness::modified_gram_schmidt_witness_into(y, m, l, q);
    }

    fn reference_fractional_derivative(f: &[f64], alpha: f64, step: f64) -> Vec<f64> {
        let mut kernel = Vec::new();
        let mut out = Vec::new();
        reference_fractional_derivative_into(f, alpha, step, &mut kernel, &mut out);
        out
    }

    fn reference_fractional_derivative_into(
        f: &[f64],
        alpha: f64,
        step: f64,
        kernel: &mut Vec<f64>,
        out: &mut Vec<f64>,
    ) {
        vyre_reference::composition_witness::try_fractional_derivative_witness_into(
            f, alpha, step, kernel, out,
        )
        .unwrap_or_else(|error| panic!("{error}"));
    }

    type NewtonSchulzScratch = vyre_reference::composition_witness::NewtonSchulzScratchWitness;

    fn reference_newton_schulz_y_step(y_curr: &[f64], yzy: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();
        reference_newton_schulz_y_step_into(y_curr, yzy, &mut out);
        out
    }

    fn reference_newton_schulz_y_step_into(y_curr: &[f64], yzy: &[f64], out: &mut Vec<f64>) {
        vyre_reference::composition_witness::newton_schulz_y_step_witness_into(y_curr, yzy, out);
    }

    fn reference_newton_schulz_inverse_sqrt(m: &[f64], n: usize, iters: u32) -> Vec<f64> {
        let mut out = Vec::new();
        let mut scratch = NewtonSchulzScratch::new();
        reference_newton_schulz_inverse_sqrt_into(m, n, iters, &mut out, &mut scratch);
        out
    }

    fn reference_newton_schulz_inverse_sqrt_into(
        m: &[f64],
        n: usize,
        iters: u32,
        out: &mut Vec<f64>,
        scratch: &mut NewtonSchulzScratch,
    ) {
        vyre_reference::composition_witness::newton_schulz_inverse_sqrt_witness_into(
            m, n, iters, out, scratch,
        );
    }
    use vyre_reference::composition_witness::{
        sinkhorn_iterate_witness as reference_sinkhorn_quantized, try_sinkhorn_iterate_witness_into,
    };

    use crate::math::sinkhorn_iterate::{
        sinkhorn_iterate_f64 as reference_sinkhorn_f64,
        sinkhorn_iterate_f64_into as reference_sinkhorn_f64_into,
    };

    fn reference_sinkhorn_row_residual(k: &[f64], u: &[f64], v: &[f64], a: &[f64]) -> f64 {
        vyre_reference::composition_witness::sinkhorn_row_residual_witness(k, u, v, a)
    }

    fn reference_sinkhorn_col_residual(k: &[f64], u: &[f64], v: &[f64], b: &[f64]) -> f64 {
        vyre_reference::composition_witness::sinkhorn_col_residual_witness(k, u, v, b)
    }

    fn reference_gaussian_rdp_step(alpha: &[f64], sigma_squared: &[f64]) -> Vec<f64> {
        vyre_reference::composition_witness::gaussian_rdp_step_witness(alpha, sigma_squared)
    }

    #[test]
    fn program_builders_emit_expected_numerical_primitives() {
        assert_eq!(
            program_generator(&dispatch_randomized_projection("a", "omega", "y", 2, 2, 2)),
            "vyre-libs::math::randomized_projection_step"
        );
        assert_eq!(
            program_generator(&dispatch_newton_schulz_y_step("y", "yzy", "next", 2)),
            "vyre-libs::math::newton_schulz_y_step"
        );
        assert_eq!(
            program_generator(&dispatch_newton_schulz_poly5_f32("mat", "out", 2, 2)),
            "vyre-libs::math::newton_schulz_poly5_f32"
        );
        assert_eq!(
            program_generator(&dispatch_sinkhorn_iterate(
                SINKHORN_FIXTURE,
                SinkhornExtents {
                    m: 2,
                    n: 2,
                    max_iterations: 3,
                },
            )),
            "vyre-libs::math::sinkhorn_iterate"
        );
        assert_eq!(
            program_generator(&dispatch_gaussian_rdp_step("alpha", "sigma", "out", 4)),
            "vyre-libs::math::gaussian_rdp_step"
        );
    }

    #[test]
    fn fractional_wrappers_preserve_kernel_and_fixed_point_contracts() {
        let kernel = vyre_reference::composition_witness::grunwald_letnikov_kernel_witness(1.0, 3);
        assert!(approx_eq(kernel[0], 1.0));
        assert!(approx_eq(kernel[1], -1.0));
        assert!(approx_eq(kernel[2], 0.0));

        let mut kernel_into = Vec::with_capacity(3);
        vyre_reference::composition_witness::try_grunwald_letnikov_kernel_witness_into(
            1.0,
            3,
            &mut kernel_into,
        )
        .unwrap();
        assert_eq!(kernel, kernel_into);

        let fixed = vyre_reference::composition_witness::kernel_to_fixed_16_16_witness(
            &[1.0, -0.5],
            1.0,
            1.0,
        );
        assert_eq!(fixed[0], 65536);
        assert_eq!(fixed[1] as i32, -32768);

        let mut fixed_into = Vec::with_capacity(2);
        vyre_reference::composition_witness::kernel_to_fixed_16_16_witness_into(
            &[1.0, -0.5],
            1.0,
            1.0,
            &mut fixed_into,
        );
        assert_eq!(fixed, fixed_into);

        let derivative = reference_fractional_derivative(&[0.0, 1.0, 2.0], 1.0, 1.0);
        assert_eq!(derivative, vec![0.0, 1.0, 1.0]);

        let mut derivative_kernel = Vec::new();
        let mut derivative_into = Vec::new();
        reference_fractional_derivative_into(
            &[0.0, 1.0, 2.0],
            1.0,
            1.0,
            &mut derivative_kernel,
            &mut derivative_into,
        );
        assert_eq!(derivative_into, derivative);
    }

    #[test]
    fn randomized_projection_and_qr_references_match_contracts() {
        let a = [1.0, 0.0, 0.0, 1.0];
        let omega = [1.0, 0.0, 0.0, 1.0];
        let projection = reference_randomized_projection(&a, &omega, 2, 2, 2);
        assert_eq!(projection, a);

        let mut projection_into = Vec::with_capacity(4);
        reference_randomized_projection_into(&a, &omega, 2, 2, 2, &mut projection_into);
        assert_eq!(projection_into, projection);

        let q = reference_modified_gram_schmidt(&[1.0, 0.0, 0.0, 1.0], 2, 2);
        assert!(approx_eq(q[0], 1.0));
        assert!(approx_eq(q[3], 1.0));

        let mut q_into = Vec::with_capacity(4);
        reference_modified_gram_schmidt_into(&[1.0, 0.0, 0.0, 1.0], 2, 2, &mut q_into);
        assert_eq!(q_into, q);
    }

    #[test]
    fn newton_schulz_references_match_optimizer_contracts() {
        let y_step = reference_newton_schulz_y_step(&[0.5], &[0.25]);
        assert!(approx_eq(y_step[0], 0.625));

        let mut y_step_into = Vec::with_capacity(1);
        reference_newton_schulz_y_step_into(&[0.5], &[0.25], &mut y_step_into);
        assert_eq!(y_step_into, y_step);

        let inverse = reference_newton_schulz_inverse_sqrt(&[1.0, 0.0, 0.0, 1.0], 2, 12);
        assert!(approx_eq(inverse[0], 1.0));
        assert!(approx_eq(inverse[3], 1.0));

        let mut inverse_into = Vec::with_capacity(4);
        let mut scratch = NewtonSchulzScratch::new();
        reference_newton_schulz_inverse_sqrt_into(
            &[1.0, 0.0, 0.0, 1.0],
            2,
            12,
            &mut inverse_into,
            &mut scratch,
        );
        assert_eq!(inverse_into.len(), inverse.len());
    }

    #[test]
    fn sinkhorn_and_privacy_references_match_contracts() {
        let (u, v, _) = reference_sinkhorn_quantized(
            &[65536],
            &[65536],
            &[65536],
            &[65536],
            &[65536],
            &[65536],
            1,
            1,
            5,
        );
        assert_eq!(u, vec![65536]);
        assert_eq!(v, vec![65536]);

        let mut u_into = Vec::new();
        let mut v_into = Vec::new();
        let mut u_old = Vec::new();
        try_sinkhorn_iterate_witness_into(
            &[65536],
            &[65536],
            &[65536],
            &[65536],
            &[65536],
            &[65536],
            1,
            1,
            5,
            &mut u_into,
            &mut v_into,
            &mut u_old,
        )
        .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(u_into, u);
        assert_eq!(v_into, v);

        let (uf, vf, _) = reference_sinkhorn_f64(&[1.0], &[1.0], &[1.0], 1e-12, 10);
        assert!(reference_sinkhorn_row_residual(&[1.0], &uf, &vf, &[1.0]) < 1e-9);
        assert!(reference_sinkhorn_col_residual(&[1.0], &uf, &vf, &[1.0]) < 1e-9);

        let mut uf_into = Vec::new();
        let mut vf_into = Vec::new();
        let mut uf_old = Vec::new();
        reference_sinkhorn_f64_into(
            &[1.0],
            &[1.0],
            &[1.0],
            1e-12,
            10,
            &mut uf_into,
            &mut vf_into,
            &mut uf_old,
        );
        assert!(reference_sinkhorn_row_residual(&[1.0], &uf_into, &vf_into, &[1.0]) < 1e-9);

        let rdp = reference_gaussian_rdp_step(&[2.0], &[1.0]);
        assert!(approx_eq(rdp[0], 1.0));
        assert!(approx_eq(
            vyre_reference::composition_witness::privacy_epsilon_from_rdp_witness(
                0.0,
                2.0,
                std::f64::consts::E.recip()
            ),
            1.0
        ));
    }
}
