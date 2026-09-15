// FMA simplification rules.
//
// Covers: constant folding, identity multiplier, zero multiplier,
// zero addend elimination.

use crate::ir::eval::fold_fma_literal;
use crate::ir::Expr;

/// Fma simplifications  -  constant folding and edge-case elimination.
pub(super) fn simplify_fma(a: &Expr, b: &Expr, c: &Expr) -> Option<Expr> {
    // Full constant fold: fma(a, b, c) → a*b+c
    if let Some(folded) = fold_fma_literal(a, b, c) {
        return Some(folded);
    }
    // fma(1, b, c) → b + c   (identity multiplier)
    if matches!(a, Expr::LitF32(v) if lit_f32_eq(*v, 1.0)) {
        return Some(Expr::add(b.clone(), c.clone()));
    }
    // fma(a, 1, c) → a + c
    if matches!(b, Expr::LitF32(v) if lit_f32_eq(*v, 1.0)) {
        return Some(Expr::add(a.clone(), c.clone()));
    }
    // A zero product is an additive identity only when its sign is negative.
    // Folding a positive-zero product to `c` changes `c = -0.0` into `+0.0`.
    // Both factors must be literals so the product sign and finiteness are known.
    if let (Expr::LitF32(zero), Expr::LitF32(other)) = (a, b) {
        if zero.is_finite()
            && other.is_finite()
            && *zero == 0.0
            && zero.is_sign_negative() != other.is_sign_negative()
        {
            return Some(c.clone());
        }
    }
    if let (Expr::LitF32(other), Expr::LitF32(zero)) = (a, b) {
        if other.is_finite()
            && zero.is_finite()
            && *zero == 0.0
            && other.is_sign_negative() != zero.is_sign_negative()
        {
            return Some(c.clone());
        }
    }
    None
}

#[inline]
fn lit_f32_eq(value: f32, expected: f32) -> bool {
    value.to_bits() == expected.to_bits()
}
