//! p-adic numerical analysis primitives (#54, research scaffold).
//!
//! p-adic numbers (Krasner 1986) give stable arithmetic for problems
//! ill-conditioned in real numbers. Recent ML work (Robin 2024) uses
//! p-adic embeddings for stable training of deep networks. Hensel
//! lifting is the algorithmic core.
//!
//! This file ships the **single Hensel lift step** primitive  -  given
//! an approximate root x of `f(x) ≡ 0 (mod p^k)` and the formal
//! derivative `f'(x)`, return a refined root accurate `mod p^{2k}`.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{BufferAccess, DataType, Expr, Program};

/// Op id.
pub const OP_ID: &str = "vyre-libs::math::hensel_lift_step";

/// Hensel iteration: `x_next = x - f(x) · (f'(x))^{-1}` modulo `p^{2k}`.
/// Inputs are pre-evaluated `f_x` and `inv_f_prime` from the caller.
#[must_use]
pub fn hensel_lift_step(x: &str, f_x: &str, inv_f_prime: &str, out: &str, n: u32) -> Program {
    if n == 0 {
        return trap_program(
            OP_ID,
            Some((out, DataType::U32)),
            "Fix: hensel_lift_step requires n > 0, got 0.".to_string(),
        );
    }

    ElementwiseComposer::new(OP_ID, n)
        .with_workgroup_size([256, 1, 1])
        .add_input_storage(x, BufferAccess::ReadOnly, DataType::U32, n)
        .add_input_storage(f_x, BufferAccess::ReadOnly, DataType::U32, n)
        .add_input_storage(inv_f_prime, BufferAccess::ReadOnly, DataType::U32, n)
        .add_output_storage(out, BufferAccess::ReadWrite, DataType::U32, n)
        .build_pointwise(out, |i| {
            Expr::sub(
                Expr::load(x, i.clone()),
                crate::math::fixed::fixed_mul_16_16_expr(
                    Expr::load(f_x, i.clone()),
                    Expr::load(inv_f_prime, i),
                ),
            )
        })
}

/// CPU reference (f64)  -  Hensel iteration single step.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn hensel_lift_step_cpu(x: f64, f_x: f64, inv_f_prime: f64) -> f64 {
    x - f_x * inv_f_prime
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || {
            hensel_lift_step("x", "f_x", "inv_f_prime", "out", 4)
        },
        Some(|| {
            vec![vec![
                vyre_primitives::wire::pack_u32_slice(&[10, 20, 30, 40]),
                vyre_primitives::wire::pack_u32_slice(&[2, 4, 6, 8]),
                vyre_primitives::wire::pack_u32_slice(&[1u32 << 16; 4]),
                vyre_primitives::wire::pack_u32_slice(&[0; 4]),
            ]]
        }),
        Some(|| {
            vec![vec![vyre_primitives::wire::pack_u32_slice(&[8, 16, 24, 32])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn cpu_zero_residual_holds_root() {
        // If f(x) = 0 already, lift step doesn't move x.
        let x_next = hensel_lift_step_cpu(2.5, 0.0, 1.0);
        assert!(approx_eq(x_next, 2.5));
    }

    #[test]
    fn cpu_quadratic_root_converges() {
        // f(x) = x² - 2, find sqrt(2) ≈ 1.414...
        // Newton/Hensel: x_{k+1} = x_k - (x_k² - 2) / (2 x_k)
        let mut x = 1.5;
        for _ in 0..10 {
            let f_x = x * x - 2.0;
            let inv_f_prime = 1.0 / (2.0 * x);
            x = hensel_lift_step_cpu(x, f_x, inv_f_prime);
        }
        assert!(approx_eq(x, 2.0_f64.sqrt()));
    }

    #[test]
    fn ir_program_buffer_layout() {
        let p = hensel_lift_step("x", "fx", "ip", "out", 16);
        assert_eq!(p.workgroup_size, [256, 1, 1]);
        assert_eq!(p.buffers[0].count(), 16);
        assert_eq!(p.buffers[3].count(), 16);
    }

    #[test]
    fn zero_n_traps() {
        let p = hensel_lift_step("x", "fx", "ip", "out", 0);
        assert!(p.stats().trap());
    }
}
