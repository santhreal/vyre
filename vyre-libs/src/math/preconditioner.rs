//! Information-geometric preconditioner primitives  -  Newton-Schulz
//! matrix inverse-square-root (Shampoo / Sophia core kernel).
//!
//! KFAC (Martens 2015), Shampoo (Gupta 2018), Sophia (Liu 2024)
//! preconditioned optimizers all need `M^{-1/2}` for some block-
//! diagonal Fisher-style matrix `M`. Newton-Schulz iteration
//! computes it without an SVD:
//!
//! ```text
//!   Y_0 = M / ||M||           (normalize so spectrum lies in [0, 1])
//!   Y_{k+1} = (1/2) Y_k (3·I - Y_k² Y_k^{-1?})  (one variant)
//! ```
//!
//! This file ships the **Newton-Schulz iteration step** primitive that
//! takes the current iterate and one matmul output and emits the next
//! iterate. The matrix-matrix multiplies inside the iteration are
//! [`crate::math::semiring_gemm`] composed by the caller.
//!
//! # Why this primitive is dual-use
//!
//! | Consumer | Use |
//! |---|---|
//! | future `vyre-libs::optim::shampoo` | Shampoo / Sophia preconditioned SGD |
//! | future `vyre-libs::optim::kfac` | K-FAC natural gradient |
//! | future `vyre-libs::math::matrix_function` | general matrix-function family (sqrt, inv-sqrt, log, exp via QSVT  -  composes with `qsvt`) |
//!
//! Self-consumer is currently weak; revisit once optimizer-aware
//! dispatch scheduling lands (megakernel auto-scheduler may use
//! preconditioned SGD on its own ILP relaxation).
//!
//! # Newton-Schulz variant
//!
//! For the inverse square root `Y = M^{-1/2}`, the standard variant is
//! the **coupled iteration**:
//!
//! ```text
//!   Y_{k+1} = (1/2) Y_k (3·I - Z_k Y_k)
//!   Z_{k+1} = (1/2) (3·I - Z_k Y_k) Z_k
//!   Z_0 = M / ||M||,  Y_0 = I / sqrt(||M||)
//! ```
//!
//! converging to `Z_k → M / sqrt(M·M) = sqrt(M)/||M||·M = sqrt(M^2)/||M||`
//! and `Y_k → 1/sqrt(M)·sqrt(||M||)`. After `k` iterations, the caller
//! rescales by the saved norm to recover `M^{-1/2}`.
//!
//! This file ships the **Y update step**: given `(Y_k, Z_k Y_k)`,
//! emit `Y_{k+1}`. The matrix product `Z_k Y_k` is the caller's job
//! (one `semiring_gemm` dispatch per step).

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-libs::math::newton_schulz_y_step";
/// Primitive op id for the fused f32 Newton-Schulz scalar polynomial.
pub const POLY5_F32_OP_ID: &str = "vyre-libs::math::newton_schulz_poly5_f32";

/// Emit `y_next = (1/2) y_curr (3·I - zy)` per cell.
///
/// Inputs:
/// - `y_curr`: `n × n` u32 buffer (current Y_k iterate, 16.16 fp).
/// - `zy`: `n × n` u32 buffer = `Z_k · Y_k` (caller-precomputed via
///   one `semiring_gemm`).
///
/// Output:
/// - `y_next`: `n × n` u32 buffer.
///
/// Computation per cell `(i, j)`:
///   `y_next[i,j] = 0.5 · (3 · y_curr[i,j] - Σ_k y_curr[i,k] · zy[k,j])`
///
/// Wait  -  `Y · (3·I - Z·Y)` involves another matmul. Decomposing:
///   `Y · (3·I - Z·Y) = 3·Y - Y·Z·Y`
///
/// The caller pre-computes `YZY = Y·Z·Y` via TWO `semiring_gemm`s
/// (Y·Z then result·Y) and supplies the full `n × n` buffer. This
/// primitive then does the cheap elementwise `0.5 (3·Y - YZY)`.
#[must_use]
pub fn newton_schulz_y_step(y_curr: &str, yzy: &str, y_next: &str, n: u32) -> Program {
    if n == 0 {
        return trap_program(
            OP_ID,
            Some((y_next, DataType::U32)),
            format!("Fix: newton_schulz_y_step requires n > 0, got {n}."),
        );
    }

    let cells = n * n;
    let t = Expr::LogicalIndex { axis: 0 };

    // value = (3 * y_curr[t] - yzy[t]) / 2
    let three_y = Expr::mul(Expr::u32(3), Expr::load(y_curr, t.clone()));
    let diff = Expr::sub(three_y, Expr::load(yzy, t.clone()));
    let half = Expr::shr(diff, Expr::u32(1));

    let body = vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(cells)),
        vec![Node::store(y_next, t, half)],
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(y_curr, 0, BufferAccess::ReadOnly, DataType::U32).with_count(cells),
            BufferDecl::storage(yzy, 1, BufferAccess::ReadOnly, DataType::U32).with_count(cells),
            BufferDecl::storage(y_next, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(cells),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

/// Emit the fused f32 Newton-Schulz quintic polynomial for each matrix cell.
///
/// One step is `p(x) = x (3.4445 + x² (-4.7750 + 2.0315 x²))`, stated as two
/// explicit [`Expr::fma`] nodes rather than five multiplies and two adds. The
/// shape is load-bearing twice over. It is four operations instead of seven per
/// step, so the kernel does less work per cell. And it leaves no
/// multiply-followed-by-add for a backend to contract: `a*b + c` written as two
/// operations may be fused into one rounding by the device and is not by the
/// reference, which over five chained steps amplified into a 46 ULP disagreement
/// on the composition `givens_rotate_pair -> newton_schulz_poly5_f32` while each
/// individual step stayed inside the elementary window. An `Fma` node states the
/// single rounding, and the reference and every emitter answer it with one fused
/// instruction.
#[must_use]
pub fn newton_schulz_poly5_f32(mat: &str, output: &str, rows: u32, cols: u32) -> Program {
    let total = rows * cols;
    let i = Expr::var("i");
    let mut iter_body = vec![Node::let_bind("x0", Expr::load(mat, i.clone()))];
    for step in 0..5 {
        let x = Expr::var(format!("x{step}"));
        let x2 = format!("x{step}_2");
        let inner = format!("x{step}_inner");
        let outer = format!("x{step}_outer");
        let next = format!("x{}", step + 1);
        iter_body.push(Node::let_bind(&x2, Expr::mul(x.clone(), x.clone())));
        iter_body.push(Node::let_bind(
            &inner,
            Expr::fma(Expr::f32(2.0315), Expr::var(&x2), Expr::f32(-4.7750)),
        ));
        iter_body.push(Node::let_bind(
            &outer,
            Expr::fma(Expr::var(&x2), Expr::var(&inner), Expr::f32(3.4445)),
        ));
        iter_body.push(Node::let_bind(&next, Expr::mul(x, Expr::var(&outer))));
    }
    iter_body.push(Node::Store {
        buffer: output.into(),
        index: i.clone(),
        value: Expr::var("x5"),
    });

    let body = vec![
        Node::let_bind("i", Expr::LogicalIndex { axis: 0 }),
        Node::if_then(Expr::lt(i.clone(), Expr::u32(total)), iter_body),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(mat, 0, BufferAccess::ReadOnly, DataType::F32).with_count(total),
            BufferDecl::output(output, 1, DataType::F32).with_count(total),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(POLY5_F32_OP_ID, body)],
    )
}

fn fixture_f32(values: &[f32]) -> Vec<u8> {
    vyre_primitives::wire::pack_f32_slice(values)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        POLY5_F32_OP_ID,
        || newton_schulz_poly5_f32("mat", "output", 2, 2),
        Some(|| vec![vec![
            fixture_f32(&[0.25, 0.5, 0.75, 1.0]),
        ]]),
        Some(|| {
            vec![vec![vec![
                0x30, 0xeb, 0x36, 0x3f, // 0.71452618
                0xc6, 0xf3, 0x43, 0x3f, // 0.76543844
                0x07, 0xfa, 0x85, 0x3f, // 1.0466927
                0xaa, 0x49, 0x32, 0x3f, // 0.69643652
            ]]]
        }),
    )
}

/// Helper: f64 matrix-matrix multiply (for the CPU reference test
/// driver below). Not an op  -  testing convenience.
#[cfg(test)]
fn matmul_dense(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    vyre_reference::composition_witness::dense_matrix_multiply_witness(a, b, n, n, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    type NewtonSchulzScratch = vyre_reference::composition_witness::NewtonSchulzScratchWitness;

    fn try_newton_schulz_y_step_cpu_into(
        y: &[f64],
        yzy: &[f64],
        out: &mut Vec<f64>,
    ) -> Result<(), String> {
        vyre_reference::composition_witness::newton_schulz_y_step_witness_into(y, yzy, out);
        Ok(())
    }

    fn newton_schulz_y_step_cpu_into(y: &[f64], yzy: &[f64], out: &mut Vec<f64>) {
        vyre_reference::composition_witness::newton_schulz_y_step_witness_into(y, yzy, out);
    }

    fn newton_schulz_y_step_cpu(y: &[f64], yzy: &[f64]) -> Vec<f64> {
        vyre_reference::composition_witness::newton_schulz_y_step_witness(y, yzy)
    }

    fn matmul_dense(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a.get(i * n + k).copied().unwrap_or(0.0)
                        * b.get(k * n + j).copied().unwrap_or(0.0);
                }
                out[i * n + j] = sum;
            }
        }
        out
    }

    fn newton_schulz_inverse_sqrt_cpu_into(
        m: &[f64],
        n: u32,
        iters: u32,
        out: &mut Vec<f64>,
        scratch: &mut NewtonSchulzScratch,
    ) {
        vyre_reference::composition_witness::newton_schulz_inverse_sqrt_witness_into(
            m, n as usize, iters, out, scratch,
        );
    }

    fn newton_schulz_inverse_sqrt_cpu(m: &[f64], n: u32, iters: u32) -> Vec<f64> {
        vyre_reference::composition_witness::newton_schulz_inverse_sqrt_witness(
            m, n as usize, iters,
        )
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-3 * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn cpu_one_step_decreases_error_for_diagonal() {
        // Start with Y = 0, after one step with yzy = 0: y_next = 0.5 * (3*0 - 0) = 0.
        // Doesn't move. Use Y = 0.5, yzy = 0.25: y_next = 0.5 * (1.5 - 0.25) = 0.625.
        let y = vec![0.5];
        let yzy = vec![0.25];
        let yn = newton_schulz_y_step_cpu(&y, &yzy);
        assert!(approx_eq(yn[0], 0.625));
    }

    #[test]
    fn cpu_y_step_into_reuses_output_storage() {
        let y = vec![0.5, 0.25];
        let yzy = vec![0.25, 0.125];
        let expected = newton_schulz_y_step_cpu(&y, &yzy);
        let mut out = Vec::with_capacity(expected.len());
        out.extend_from_slice(&[99.0, 98.0]);

        newton_schulz_y_step_cpu_into(&y, &yzy, &mut out);
        let ptr = out.as_ptr();
        let capacity = out.capacity();
        newton_schulz_y_step_cpu_into(&y, &yzy, &mut out);

        assert_eq!(out, expected);
        assert_eq!(out.as_ptr(), ptr);
        assert_eq!(out.capacity(), capacity);

        try_newton_schulz_y_step_cpu_into(&[1.0], &[0.5], &mut out)
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - Newton-Schulz Y-step should truncate stale output");
        assert_eq!(out, vec![1.25]);
        assert_eq!(out.as_ptr(), ptr);
        assert_eq!(out.capacity(), capacity);
    }

    #[test]
    fn cpu_y_step_truncates_mismatched_inputs() {
        let out = newton_schulz_y_step_cpu(&[1.0, 2.0], &[0.5]);
        assert_eq!(out, vec![1.25]);
    }

    #[test]
    fn cpu_inverse_sqrt_recovers_identity_inverse() {
        // M = I → M^{-1/2} = I.
        let m = vec![1.0, 0.0, 0.0, 1.0];
        let result = newton_schulz_inverse_sqrt_cpu(&m, 2, 12);
        // Expect ~ identity.
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[1], 0.0));
        assert!(approx_eq(result[2], 0.0));
        assert!(approx_eq(result[3], 1.0));
    }

    #[test]
    fn cpu_inverse_sqrt_pads_short_matrix() {
        let out = newton_schulz_inverse_sqrt_cpu(&[1.0], 2, 1);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn cpu_inverse_sqrt_recovers_diagonal_two() {
        // M = diag(4, 4) → M^{-1/2} = diag(0.5, 0.5).
        let m = vec![4.0, 0.0, 0.0, 4.0];
        let result = newton_schulz_inverse_sqrt_cpu(&m, 2, 20);
        assert!(approx_eq(result[0], 0.5));
        assert!(approx_eq(result[1], 0.0));
        assert!(approx_eq(result[2], 0.0));
        assert!(approx_eq(result[3], 0.5));
    }

    #[test]
    fn cpu_inverse_sqrt_into_reuses_workspace() {
        let m = vec![4.0, 0.0, 0.0, 4.0];
        let expected = newton_schulz_inverse_sqrt_cpu(&m, 2, 8);
        let mut out = Vec::with_capacity(4);
        let mut scratch = NewtonSchulzScratch::new();

        newton_schulz_inverse_sqrt_cpu_into(&m, 2, 8, &mut out, &mut scratch);
        let out_ptr = out.as_ptr();
        let y_ptr = scratch.y.as_ptr();
        let z_ptr = scratch.z.as_ptr();
        let zy_ptr = scratch.zy.as_ptr();
        let three_i_ptr = scratch.three_i_minus_zy.as_ptr();
        newton_schulz_inverse_sqrt_cpu_into(&m, 2, 8, &mut out, &mut scratch);

        assert_eq!(out.as_ptr(), out_ptr);
        assert_eq!(scratch.y.as_ptr(), y_ptr);
        assert_eq!(scratch.z.as_ptr(), z_ptr);
        assert_eq!(scratch.zy.as_ptr(), zy_ptr);
        assert_eq!(scratch.three_i_minus_zy.as_ptr(), three_i_ptr);
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!(approx_eq(*got, *want));
        }
    }

    #[test]
    fn cpu_inverse_sqrt_property_y_squared_times_m_is_identity() {
        // For any PSD M, after enough iterations: Y² · M ≈ I.
        let m = vec![2.0, 0.5, 0.5, 1.5];
        let y = newton_schulz_inverse_sqrt_cpu(&m, 2, 30);
        let y_sq = matmul_dense(&y, &y, 2);
        let prod = matmul_dense(&y_sq, &m, 2);
        // prod ≈ identity
        assert!(approx_eq(prod[0], 1.0));
        assert!(approx_eq(prod[3], 1.0));
        assert!(prod[1].abs() < 1e-3);
        assert!(prod[2].abs() < 1e-3);
    }

    #[test]
    fn ir_program_buffer_layout() {
        let p = newton_schulz_y_step("y", "yzy", "yn", 4);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["y", "yzy", "yn"]);
        for buf in p.buffers.iter() {
            assert_eq!(buf.count(), 16); // n*n = 4*4
        }
    }

    #[test]
    fn zero_n_traps() {
        let p = newton_schulz_y_step("y", "yzy", "yn", 0);
        assert!(p.stats().trap());
    }
}
