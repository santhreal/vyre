//! Lemire's constant-divisor divisibility test.
//!
//! `x % d == 0` does not need the remainder. Lowering the remainder the
//! general way costs a multiply-high, a shift, a fixup on some divisors, a
//! multiply, a subtract, and a compare, and it reads `x` twice. The
//! divisibility question answers in three operations that read `x` once.
//!
//! Write `d = 2^k * q` with `q` odd, let `qinv` be the inverse of `q` modulo
//! `2^32`, and let `limit = floor((2^32 - 1) / d)`. Then for every `u32` `x`:
//!
//! ```text
//! x % d == 0   <=>   rotate_right(x * qinv, k) <= limit
//! ```
//!
//! Proof. Multiplication by the odd unit `qinv` is a bijection on `Z/2^32`,
//! and rotation is a bijection, so the whole left-hand map is a bijection.
//! Take a multiple `x = d*m`, necessarily `0 <= m <= limit`. Then
//! `x * qinv = 2^k * q * m * qinv = 2^k * m (mod 2^32)`, and `m <= limit`
//! forces `2^k * m < 2^32`, so that product does not wrap and its low `k` bits
//! are zero. Rotating it right by `k` is therefore a plain shift and yields
//! `m <= limit`. The multiples of `d` number `limit + 1`, exactly the size of
//! `[0, limit]`, so a bijection carrying the first into the second carries it
//! onto the second, and every non-multiple lands above `limit`.
//!
//! The rewrite must run before the remainder is lowered, because the general
//! lowering destroys the `x % d` shape. It is applied as its own expression
//! sweep at the head of the pass rather than as a table entry, since the
//! expression rewriter assembles children before their parent and the `Mod`
//! child would already be gone by the time the comparison is visited.

use crate::ir::{BinOp, Expr};

/// Rewrite a constant-divisor divisibility comparison into Lemire's test.
///
/// Recognizes `x % d == 0` and `x % d != 0` in either operand order.
pub(super) fn rewrite_divisibility_test(expr: &Expr) -> Option<Expr> {
    let Expr::BinOp { op, left, right } = expr else {
        return None;
    };
    let divisible = match op {
        BinOp::Eq => true,
        BinOp::Ne => false,
        _ => return None,
    };
    let (operand, divisor) = zero_compared_remainder(left, right)?;
    // `d == 0` keeps the reference remainder semantics, `d == 1` divides
    // everything and belongs to const-fold, and a power of two already
    // answers in two operations as `x & (d-1) == 0`.
    if divisor <= 1 || divisor.is_power_of_two() {
        return None;
    }

    let trailing = divisor.trailing_zeros();
    let odd_part = divisor >> trailing;
    let scaled = Expr::mul(operand.clone(), Expr::u32(odd_inverse(odd_part)));
    let rotated = if trailing == 0 {
        scaled
    } else {
        Expr::rotate_right(scaled, Expr::u32(trailing))
    };
    let limit = Expr::u32(u32::MAX / divisor);
    Some(if divisible {
        Expr::le(rotated, limit)
    } else {
        Expr::gt(rotated, limit)
    })
}

/// Match `x % C` against a literal zero on the other side of the comparison.
fn zero_compared_remainder<'a>(left: &'a Expr, right: &'a Expr) -> Option<(&'a Expr, u32)> {
    match (left, right) {
        (candidate, Expr::LitU32(0)) | (Expr::LitU32(0), candidate) => {
            let Expr::BinOp {
                op: BinOp::Mod,
                left: operand,
                right: divisor,
            } = candidate
            else {
                return None;
            };
            match divisor.as_ref() {
                Expr::LitU32(divisor) => Some((operand.as_ref(), *divisor)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Multiplicative inverse of an odd `u32` modulo `2^32`.
///
/// Newton-Raphson on `f(y) = 1/y - value`. The seed is correct to three bits
/// because every odd `q` satisfies `q*q == 1 (mod 8)`, and each step doubles
/// the correct bit count, so four steps cover 48 bits and therefore all 32.
fn odd_inverse(value: u32) -> u32 {
    debug_assert!(value % 2 == 1, "modular inverse requires an odd modulus");
    let mut inverse = value;
    for _ in 0..4 {
        inverse = inverse.wrapping_mul(2u32.wrapping_sub(value.wrapping_mul(inverse)));
    }
    debug_assert_eq!(value.wrapping_mul(inverse), 1);
    inverse
}
