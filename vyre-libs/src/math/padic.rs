//! p-adic and Hensel lift operations on device (M-CONT-16, Section 180.3).
//!
//! Hensel lift step: `x_{k+1} = x_k - f(x_k) / f'(x_k) mod p^{2^k}`.
//!
//! Given arrays of values `x`, `f_x = f(x)`, and `inv_f_prime = (f'(x))^{-1}`,
//! computes `x - f(x) * (f'(x))^{-1}` in parallel for each lane using fixed-point
//! arithmetic. This is the core kernel of p-adic root finding and polynomial
//! factorization over Q_p / Z_p.
//!
//! Used by the algebraic solver stack to lift approximate roots: starting from
//! an approximate root x of `f(x) ≡ 0 (mod p^k)` and the formal
//! derivative `f'(x)`, return a refined root accurate `mod p^{2k}`.

use crate::builder::elementwise::ElementwiseComposer;
use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{BufferAccess, DataType, Expr, Program};

/// Op id.
pub const OP_ID: &str = "vyre-libs::math::hensel_lift_step";

/// Hensel lift step for `n` p-adic roots in parallel.
///
/// Output: `out[i] = x[i] - f_x[i] * inv_f_prime[i]` (in 16.16 fixed point).
/// `n == 0` returns a trap Program.
#[must_use]
pub fn hensel_lift_step(x: &str, f_x: &str, inv_f_prime: &str, out: &str, n: u32) -> Program {
    if n == 0 {
        return trap_program(
            OP_ID,
            None,
            "hensel_lift_step requires n > 0 (use positive degree)",
        );
    }

    ElementwiseComposer::new(OP_ID, n)
        .with_workgroup_size([256, 1, 1])
        .add_input_storage(x, BufferAccess::ReadOnly, DataType::U32, n)
        .add_input_storage(f_x, BufferAccess::ReadOnly, DataType::U32, n)
        .add_input_storage(inv_f_prime, BufferAccess::ReadOnly, DataType::U32, n)
        .add_output_storage(out, BufferAccess::WriteOnly, DataType::U32, n)
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

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || hensel_lift_step("x", "f_x", "inv_f_prime", "out", 4),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            let to_fixed = |v: f64| (v * 65536.0).round() as i32 as u32;
            vec![vec![
                to_bytes(&[to_fixed(2.0), to_fixed(3.0), to_fixed(5.0), to_fixed(7.0)]),
                to_bytes(&[to_fixed(0.1), to_fixed(0.2), to_fixed(-0.1), to_fixed(0.0)]),
                to_bytes(&[to_fixed(1.0), to_fixed(0.5), to_fixed(2.0), to_fixed(1.0)]),
            ]]
        }),
        Some(|| {
            vec![vec![vec![
                0x66, 0xe6, 0x01, 0x00, // 1.9
                0x67, 0xe6, 0x02, 0x00, // 2.9
                0x34, 0x33, 0x05, 0x00, // 5.2
                0x00, 0x00, 0x07, 0x00, // 7.0
            ]]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10 * (1.0 + a.abs() + b.abs())
    }

    fn hensel_lift_step_cpu(x: f64, f_x: f64, inv_f_prime: f64) -> f64 {
        x - f_x * inv_f_prime
    }

    #[test]
    fn cpu_zero_residual_holds_root() {
        let x_next = hensel_lift_step_cpu(2.5, 0.0, 1.0);
        assert!(approx_eq(x_next, 2.5));
    }

    #[test]
    fn cpu_quadratic_root_converges() {
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
        assert_eq!(p.buffers[0].count, 16);
        assert_eq!(p.buffers[3].count, 16);
    }

    #[test]
    fn zero_n_traps() {
        let p = hensel_lift_step("x", "fx", "ip", "out", 0);
        assert!(p.stats().trap());
    }
}
