//! Constant-division simplifications that remove a division outright.
//!
//! Granlund-Montgomery turns one division into a multiply-high plus a shift.
//! The rewrites here are cheaper still: each one either deletes a division or
//! collapses two constant divisions into one, before the general lowering runs.

use super::bounds::u32_upper_bound;
use crate::ir::{BinOp, Expr};

/// Cancel the common factor between a constant multiplier inside the dividend
/// and a constant divisor: `(x * c) / d` becomes `(x * (c/g)) / (d/g)` for
/// `g = gcd(c, d)`, collapsing to a bare multiply when `g == d` and to `x`
/// itself when the two constants are equal.
///
/// Over the integers `x*c/d` and `x*(c/g)/(d/g)` are the same rational, so the
/// rewrite is exact whenever the original product is the product 32-bit
/// arithmetic actually computes. That is the whole precondition, and it is
/// discharged by proving `x * c < 2^32` from [`u32_upper_bound`]. Without the
/// proof the rewrite is wrong rather than merely unprofitable: at `x = 2^31`,
/// `c = 2`, `d = 2` the original wraps to `0` while `x` does not.
///
/// The reduced product `x * (c/g)` never exceeds the original, so it cannot
/// wrap either.
pub(super) fn cancel_constant_factor(dividend: &Expr, divisor: u32) -> Option<Expr> {
    if divisor <= 1 {
        return None;
    }
    let (operand, multiplier) = constant_multiply(dividend)?;
    let common = gcd(multiplier, divisor);
    if common <= 1 {
        return None;
    }
    // Proof obligation: the product the unoptimized program computes must be
    // the mathematical product, not a wrapped one.
    u32_upper_bound(operand)?.checked_mul(multiplier)?;

    let reduced_multiplier = multiplier / common;
    let reduced_divisor = divisor / common;
    let product = if reduced_multiplier == 1 {
        operand.clone()
    } else {
        Expr::mul(operand.clone(), Expr::u32(reduced_multiplier))
    };
    if reduced_divisor == 1 {
        Some(product)
    } else {
        Some(Expr::div(product, Expr::u32(reduced_divisor)))
    }
}

/// Collapse `(x / a) / b` into `x / (a*b)`, turning two constant divisions
/// into one.
///
/// Exact for every unsigned `x`: `floor(floor(x/a)/b) == floor(x/(a*b))`. No
/// range proof is needed because neither side ever exceeds `x`. A zero divisor
/// is left alone so the reference div-by-zero value survives, and an `a*b`
/// that does not fit in `u32` is declined rather than wrapped.
pub(super) fn fuse_constant_divisors(dividend: &Expr, divisor: u32) -> Option<Expr> {
    let (inner, inner_divisor) = nonzero_constant_operand(dividend, BinOp::Div)?;
    if divisor == 0 {
        return None;
    }
    let fused = inner_divisor.checked_mul(divisor)?;
    Some(Expr::div(inner.clone(), Expr::u32(fused)))
}

/// Collapse `(x % a) % b` into `x % b` when `b` divides `a`.
///
/// Exact for every unsigned `x`: writing `x = q*a + r`, `b | a` makes
/// `x mod b == r mod b`, which is the inner remainder reduced by `b`.
pub(super) fn narrow_nested_modulus(dividend: &Expr, modulus: u32) -> Option<Expr> {
    let (inner, inner_modulus) = nonzero_constant_operand(dividend, BinOp::Mod)?;
    if modulus == 0 || inner_modulus % modulus != 0 {
        return None;
    }
    Some(Expr::rem(inner.clone(), Expr::u32(modulus)))
}

/// Match `x <op> C` for a non-zero literal `C`, returning the other operand.
///
/// A zero right operand is declined everywhere here: it is the reference
/// division and remainder case, whose value no rewrite may fabricate.
fn nonzero_constant_operand(expr: &Expr, wanted: BinOp) -> Option<(&Expr, u32)> {
    let Expr::BinOp { op, left, right } = expr else {
        return None;
    };
    if *op != wanted {
        return None;
    }
    match right.as_ref() {
        Expr::LitU32(0) => None,
        Expr::LitU32(constant) => Some((left.as_ref(), *constant)),
        _ => None,
    }
}

/// Split `x * C` (in either operand order) into the variable side and `C`.
fn constant_multiply(expr: &Expr) -> Option<(&Expr, u32)> {
    let Expr::BinOp {
        op: BinOp::Mul,
        left,
        right,
    } = expr
    else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (operand, Expr::LitU32(constant)) | (Expr::LitU32(constant), operand) => {
            Some((operand, *constant))
        }
        _ => None,
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}
