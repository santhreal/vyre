use crate::ir::{BinOp, Expr, Program, UnOp};
use crate::optimizer::rewrite::rewrite_program;
use crate::optimizer::{vyre_pass, PassAnalysis, PassResult};

mod arithmetic;
mod bounds;
mod divisibility;
mod duplication;
mod exact_div;

use arithmetic::{
    div_operand_copies, granlund_montgomery_div, horner_polynomial_int, power_of_two_shift,
    reciprocal_constant_fold, shift_add_decompose,
};
use divisibility::rewrite_divisibility_test;
use duplication::may_duplicate;
use exact_div::{cancel_constant_factor, fuse_constant_divisors, narrow_nested_modulus};

/// Replace multiplication by powers of two with shifts.
#[derive(Debug, Default)]
#[vyre_pass(
    name = "strength_reduce",
    requires = ["const_fold"],
    invalidates = ["const_fold", "reaching_def_propagate", "value_numbering"],
    phase = "scalar_algebra",
    boundary_class = "abi_preserving",
    cost_model_family = "scalar"
)]
pub struct StrengthReduce;

impl StrengthReduce {
    /// O(1) gate: strength reduction only fires on expressions, which only
    /// live inside Let / Assign / Store / If-cond / Loop bound / Trap nodes.
    /// A program that is just Return / Barrier / `IndirectDispatch` / `AsyncWait`
    /// has no expression tree to rewrite. The bitset mask covers every
    /// expression-bearing node kind so SKIP fires on truly expression-free
    /// programs without false negatives.
    #[must_use]
    fn analyze_impl(program: &Program) -> PassAnalysis {
        super::expression_bearing_analysis(program)
    }

    /// Rewrite multiply-by-power-of-two expressions into left shifts.
    ///
    /// AUDIT_2026-04-24 F-SR-01 (closed): `rewrite_program` already
    /// preserves `non_composable_with_self` via `with_rewritten_entry`
    /// (see builder.rs line ~134). No explicit call needed here.
    #[must_use]
    pub fn transform(program: Program) -> PassResult {
        // Recognition runs first and on its own sweep. The expression
        // rewriter assembles children before their parent, so by the time
        // `reduce_expr` reaches `x % d == 0` or `(x * c) / d`, the inner
        // remainder or product has already been lowered and the cheaper
        // shape is unreachable.
        let (program, recognized) = rewrite_program(program, recognize_source_shape);
        let (program, reduced) = rewrite_program(program, reduce_expr);
        PassResult {
            program,
            changed: recognized || reduced,
        }
    }
}

/// Rewrites that must see the operands as written.
///
/// Each one keys on a constant that the lowering table consumes: the
/// remainder in a divisibility test, the multiplier inside a dividend, the
/// inner divisor of a division chain. Lowering a child destroys that
/// constant, and a bottom-up rewriter lowers children first, so these have to
/// run as a separate earlier sweep rather than as entries in the table.
fn recognize_source_shape(expr: &Expr) -> Option<Expr> {
    if let Some(test) = rewrite_divisibility_test(expr) {
        return Some(test);
    }
    let Expr::BinOp { op, left, right } = expr else {
        return None;
    };
    match (op, right.as_ref()) {
        (BinOp::Div, Expr::LitU32(divisor)) => cancel_constant_factor(left, *divisor)
            .or_else(|| fuse_constant_divisors(left, *divisor)),
        (BinOp::Mod, Expr::LitU32(modulus)) => narrow_nested_modulus(left, *modulus),
        _ => None,
    }
}

enum ReducePick {
    Left,
    Right,
    Expr(Expr),
}

#[expect(
    clippy::too_many_lines,
    reason = "strength-reduction table keeps algebraic rewrite precedence auditable"
)]
fn reduce_expr(expr: &Expr) -> Option<Expr> {
    if let Some(reduced) = horner_polynomial_int(expr) {
        return Some(reduced);
    }

    let Expr::BinOp { op, left, right } = expr else {
        return None;
    };
    let (l, r) = (left.as_ref(), right.as_ref());
    let other_if = |pred: fn(&Expr) -> bool| {
        if pred(r) {
            Some(ReducePick::Left)
        } else if pred(l) {
            Some(ReducePick::Right)
        } else {
            None
        }
    };

    let pick = match op {
        // Integer Mul-by-2^k → Shl by k.
        BinOp::Mul => {
            if matches!(r, Expr::LitU32(0)) || matches!(l, Expr::LitU32(0)) {
                return Some(Expr::u32(0));
            }
            if let Some(id) = other_if(|e| matches!(e, Expr::LitU32(1))) {
                Some(id)
            } else if let Some(shift) = power_of_two_shift(right) {
                Some(ReducePick::Expr(Expr::shl(l.clone(), Expr::u32(shift))))
            } else if let Some(shift) = power_of_two_shift(left) {
                Some(ReducePick::Expr(Expr::shl(r.clone(), Expr::u32(shift))))
            } else if let Some(decomposed) =
                shift_add_decompose(l, r).or_else(|| shift_add_decompose(r, l))
            {
                Some(ReducePick::Expr(decomposed))
            } else if let Some(neg_c) = neg_i32_const(r) {
                Some(ReducePick::Expr(Expr::negate(Expr::mul(
                    l.clone(),
                    Expr::i32(neg_c),
                ))))
            } else if let Some(neg_c) = neg_i32_const(l) {
                Some(ReducePick::Expr(Expr::negate(Expr::mul(
                    r.clone(),
                    Expr::i32(neg_c),
                ))))
            } else if matches!(r, Expr::LitF32(v) if lit_f32_eq(*v, 2.0)) {
                let x = l.clone();
                Some(ReducePick::Expr(Expr::add(x.clone(), x)))
            } else if matches!(l, Expr::LitF32(v) if lit_f32_eq(*v, 2.0)) {
                let x = r.clone();
                Some(ReducePick::Expr(Expr::add(x.clone(), x)))
            } else {
                other_if(|e| matches!(e, Expr::LitF32(v) if lit_f32_eq(*v, 1.0)))
            }
        }
        // Unsigned Div-by-2^k → Shr by k.
        BinOp::Div => {
            if let Some(folded) = reciprocal_constant_fold(l, r) {
                return Some(folded);
            }
            if matches!(l, Expr::LitF32(v) if lit_f32_eq(*v, 1.0)) && !matches!(r, Expr::LitF32(_))
            {
                return Some(Expr::reciprocal(r.clone()));
            }
            match r {
                Expr::LitU32(1) => Some(ReducePick::Left),
                Expr::LitU32(value) if value.is_power_of_two() => Some(ReducePick::Expr(
                    Expr::shr(l.clone(), Expr::u32(value.trailing_zeros())),
                )),
                Expr::LitU32(d) if *d > 1 && !d.is_power_of_two() => {
                    granlund_montgomery_div(l, *d).map(ReducePick::Expr)
                }
                Expr::LitF32(v) if lit_f32_eq(*v, 1.0) => Some(ReducePick::Left),
                Expr::LitF32(v) => exact_f32_reciprocal(*v).map(|reciprocal| {
                    ReducePick::Expr(Expr::mul(l.clone(), Expr::f32(reciprocal)))
                }),
                _ => None,
            }
        }
        // Unsigned Mod-by-constant.
        BinOp::Mod => {
            let Expr::LitU32(value) = r else {
                return None;
            };
            if *value == 1 {
                return Some(Expr::u32(0));
            }
            if value.is_power_of_two() {
                return Some(Expr::bitand(l.clone(), Expr::u32(value - 1)));
            }
            if !may_duplicate(left, 1 + div_operand_copies(*value)) {
                return None;
            }
            granlund_montgomery_div(l, *value).map(|quotient| {
                ReducePick::Expr(Expr::sub(l.clone(), Expr::mul(quotient, Expr::u32(*value))))
            })
        }
        // Float: x + (-0.0) → x; Integer: x + 0 → x; Negation fusion.
        BinOp::Add => other_if(|e| {
            matches!(e, Expr::LitU32(0)) || matches!(e, Expr::LitF32(v) if lit_f32_eq(*v, -0.0))
        })
        .or_else(|| {
            if let Expr::UnOp {
                op: UnOp::Negate,
                operand: y,
            } = r
            {
                Some(ReducePick::Expr(Expr::sub(l.clone(), y.as_ref().clone())))
            } else if let Expr::UnOp {
                op: UnOp::Negate,
                operand: x,
            } = l
            {
                Some(ReducePick::Expr(Expr::sub(r.clone(), x.as_ref().clone())))
            } else {
                None
            }
        }),
        // Float: x - (+0.0) → x; Integer: x - 0 → x; x - (-y) → x + y.
        BinOp::Sub => {
            if matches!(r, Expr::LitU32(0)) || matches!(r, Expr::LitF32(v) if lit_f32_eq(*v, 0.0)) {
                Some(ReducePick::Left)
            } else if let Expr::UnOp {
                op: UnOp::Negate,
                operand: y,
            } = r
            {
                Some(ReducePick::Expr(Expr::add(l.clone(), y.as_ref().clone())))
            } else {
                None
            }
        }
        BinOp::Shl | BinOp::Shr => {
            super::shift_fusion::reduce_shift(*op, l, r).map(ReducePick::Expr)
        }

        // BitAnd complement annihilator: x & ~x → 0, ~x & x → 0
        BinOp::BitAnd if is_bitnot_of(r, l) || is_bitnot_of(l, r) => {
            Some(ReducePick::Expr(Expr::u32(0)))
        }
        // BitOr / BitXor complement → all-ones
        BinOp::BitOr | BinOp::BitXor if is_bitnot_of(r, l) || is_bitnot_of(l, r) => {
            Some(ReducePick::Expr(Expr::u32(u32::MAX)))
        }

        // Rotate-by-zero or 32 → identity for u32
        BinOp::RotateLeft | BinOp::RotateRight if matches!(r, Expr::LitU32(0 | 32)) => {
            Some(ReducePick::Left)
        }

        // AbsDiff self-identity
        BinOp::AbsDiff if l == r => Some(ReducePick::Expr(Expr::u32(0))),

        BinOp::Min if matches!(r, Expr::LitU32(0)) || matches!(l, Expr::LitU32(0)) => {
            Some(ReducePick::Expr(Expr::u32(0)))
        }
        BinOp::Max => {
            if matches!(r, Expr::LitU32(u32::MAX)) || matches!(l, Expr::LitU32(u32::MAX)) {
                Some(ReducePick::Expr(Expr::u32(u32::MAX)))
            } else {
                other_if(|e| matches!(e, Expr::LitU32(0)))
            }
        }
        BinOp::Min => other_if(|e| matches!(e, Expr::LitU32(u32::MAX))),

        BinOp::Lt if matches!(r, Expr::LitU32(0)) => Some(ReducePick::Expr(Expr::bool(false))),
        BinOp::Ge if matches!(r, Expr::LitU32(0)) => Some(ReducePick::Expr(Expr::bool(true))),
        BinOp::Gt if matches!(l, Expr::LitU32(0)) => Some(ReducePick::Expr(Expr::bool(false))),
        BinOp::Le if matches!(l, Expr::LitU32(0)) => Some(ReducePick::Expr(Expr::bool(true))),

        _ => None,
    };
    match pick? {
        ReducePick::Left => Some(l.clone()),
        ReducePick::Right => Some(r.clone()),
        ReducePick::Expr(expr) => Some(expr),
    }
}

#[inline]
fn neg_i32_const(expr: &Expr) -> Option<i32> {
    match expr {
        Expr::LitI32(c) if *c < -1 && *c != i32::MIN => Some(-*c),
        _ => None,
    }
}

#[inline]
fn is_bitnot_of(candidate: &Expr, target: &Expr) -> bool {
    matches!(
        candidate,
        Expr::UnOp {
            op: UnOp::BitNot,
            operand,
        } if operand.as_ref() == target
    )
}

#[inline]
fn lit_f32_eq(value: f32, expected: f32) -> bool {
    value.to_bits() == expected.to_bits()
}

#[inline]
fn f32_nonzero(value: f32) -> bool {
    value.to_bits() & 0x7FFF_FFFF != 0
}

/// The reciprocal of `value` when `1.0 / value` carries no rounding error,
/// which is what makes `x / value` and `x * (1.0 / value)` the same program.
///
/// Exactness needs the significand to be one, so the divisor is a power of two.
/// That alone is not enough at the ends of the range: a divisor below `2^-127`
/// reciprocates to infinity, which turns every quotient into an infinity or a
/// NaN, and rejecting a non-finite or zero divisor does not catch it. The
/// reciprocal is required to be normal, which also declines the two exponents
/// whose exact reciprocal is subnormal rather than reasoning about underflow.
///
/// A divisor whose reciprocal rounds belongs to the emitter, which is where a
/// declared ULP budget authorizes trading a division for a multiply. The
/// optimizer states what the program computes, so it only takes the rewrite
/// that computes the same thing.
#[inline]
fn exact_f32_reciprocal(value: f32) -> Option<f32> {
    if !value.is_finite() || !f32_nonzero(value) {
        return None;
    }
    let bits = value.to_bits();
    let significand = bits & 0x007F_FFFF;
    let power_of_two = if bits & 0x7F80_0000 == 0 {
        significand.is_power_of_two()
    } else {
        significand == 0
    };
    if !power_of_two {
        return None;
    }
    let reciprocal = 1.0_f32 / value;
    reciprocal.is_normal().then_some(reciprocal)
}

#[cfg(test)]
#[path = "../../../../../tests/internal/optimizer/passes/algebraic/strength_reduce/mod.rs"]
mod tests;
