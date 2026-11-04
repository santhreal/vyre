//! Provable upper bounds for unsigned expressions.
//!
//! Several strength-reduction rewrites are algebraically exact over the
//! integers but only agree with 32-bit wrapping arithmetic while an
//! intermediate product stays below `2^32`. [`u32_upper_bound`] supplies the
//! proof: it returns a value the expression provably never exceeds, or `None`
//! when no bound follows from the expression's own structure.
//!
//! Soundness rule for every arm: an operation whose result can wrap reports
//! `None` rather than a bound the wrap would violate. `Sub` is absent for
//! exactly that reason: unsigned subtraction wraps to values near `u32::MAX`,
//! so a bounded left operand says nothing about the difference. `Cast`,
//! `BufLen`, loads, and identifier-shaped expressions are unbounded here
//! because their range comes from outside the expression tree.
//!
//! Shift arms follow the IR's own semantics, where the shift amount is taken
//! modulo 32 (see `ir_eval::fold_u32_binary`), so only literal amounts below
//! 32 produce a bound.

use crate::ir::{BinOp, Expr};

/// Depth past which the search reports "no bound" instead of walking further.
///
/// Bounded expressions in real kernels are shallow (a mask, a modulo, a
/// literal, one or two arithmetic layers on top). The cap keeps a rewrite
/// admission check O(1) on a deeply nested tree.
const MAX_DEPTH: u32 = 8;

/// Largest value `expr` can evaluate to under unsigned 32-bit semantics, when
/// that is provable from the expression alone.
#[must_use]
pub(super) fn u32_upper_bound(expr: &Expr) -> Option<u32> {
    bound(expr, MAX_DEPTH)
}

fn bound(expr: &Expr, depth: u32) -> Option<u32> {
    let next = depth.checked_sub(1)?;
    match expr {
        Expr::LitU32(value) => Some(*value),
        // Either arm may be taken, so the bound is the larger of the two.
        Expr::Select {
            true_val,
            false_val,
            ..
        } => Some(bound(true_val, next)?.max(bound(false_val, next)?)),
        Expr::BinOp { op, left, right } => binop_bound(*op, left, right, next),
        _ => None,
    }
}

fn binop_bound(op: BinOp, left: &Expr, right: &Expr, depth: u32) -> Option<u32> {
    match op {
        // `checked_*` returning None is exactly the wrapping case, where no
        // bound below u32::MAX holds.
        BinOp::Add => bound(left, depth)?.checked_add(bound(right, depth)?),
        BinOp::Mul => bound(left, depth)?.checked_mul(bound(right, depth)?),
        // `x & y` never exceeds either operand, so one bounded side suffices.
        BinOp::BitAnd | BinOp::Min => match (bound(left, depth), bound(right, depth)) {
            (Some(l), Some(r)) => Some(l.min(r)),
            (Some(only), None) | (None, Some(only)) => Some(only),
            (None, None) => None,
        },
        // `x % d <= d - 1` for every unsigned x. A zero divisor is the
        // reference-value case and carries no bound.
        BinOp::Mod => match right {
            Expr::LitU32(divisor) if *divisor > 0 => Some(divisor - 1),
            _ => None,
        },
        // Division only shrinks, so an unbounded dividend still yields
        // u32::MAX / d.
        BinOp::Div => match right {
            Expr::LitU32(divisor) if *divisor > 0 => {
                Some(bound(left, depth).unwrap_or(u32::MAX) / divisor)
            }
            _ => None,
        },
        BinOp::Shr => match right {
            Expr::LitU32(shift) if *shift < 32 => {
                Some(bound(left, depth).unwrap_or(u32::MAX) >> shift)
            }
            _ => None,
        },
        BinOp::Shl => match right {
            Expr::LitU32(shift) if *shift < 32 => bound(left, depth)?.checked_mul(1u32 << shift),
            _ => None,
        },
        _ => None,
    }
}
