//! The one answer to "apply this operator at this width to these scalars".
//!
//! Both scalar consumers route here: the literal constant folder
//! ([`crate::ir_eval`]) unwraps two `Expr` literals and calls [`apply_binary`]
//! or [`apply_unary`], and the reference interpreter
//! ([`crate::ir_inner::model::node_kind`]) unwraps two [`Value`]s and calls the
//! same pair. Neither carries an arithmetic row of its own.
//!
//! WHY one owner: the two used to carry parallel tables and the tables drifted
//! per width, so an optimized program and its own reference run computed
//! different answers for the same expression. The folder folded `AbsDiff`,
//! `And`, `Or`, `RotateLeft` and `RotateRight` on i32, `Mod` on f32, `BitXor`
//! on bool and every transcendental on integer literals; the interpreter
//! rejected all of them. The interpreter shifted i32 by a total masked count
//! while the folder declined a negative count. A single table cannot disagree
//! with itself.
//!
//! WHY this support set: it follows [`crate::validate::typecheck`]. An operator
//! the validator refuses at a width is refused here too, so no optimizer pass
//! can hand back a value for an expression validation rejects. `AbsDiff` on i32
//! is the sharp case (V086: `i32::MIN.abs_diff(i32::MAX)` has no signed
//! result), and the folder used to answer it with a `u32` literal, silently
//! retyping the expression.

use crate::fp_parity::canonical_f32;
use crate::ir_inner::model::node_kind::{EvalError, Value};
use crate::ir_inner::model::spec_types::{BinOp, UnOp};

fn unsupported_binary(op: BinOp, width: &str) -> EvalError {
    EvalError::new(format!(
        "unsupported {width} binary operation {op:?}. Fix: add interpreter semantics before registering this operation."
    ))
}

fn unsupported_unary(op: &UnOp, width: &str) -> EvalError {
    EvalError::new(format!(
        "unsupported {width} unary operation {op:?}. Fix: add interpreter semantics before registering this operation."
    ))
}

fn undefined_i32_division(kind: &str, left: i32, right: i32) -> EvalError {
    EvalError::new(format!(
        "i32 {kind} `{left} / {right}` has undefined target-text semantics. Fix: guard the signed divisor/overflow case before interpretation, or use unsigned operands when zero-divisor semantics must be total."
    ))
}

/// Apply `op` to two scalars of the same width.
///
/// # Errors
///
/// Returns [`EvalError`] when the operands have different widths, when `op` has
/// no defined meaning at that width, or when the operands hit a case the target
/// backends leave undefined (signed division by zero and `i32::MIN / -1`).
#[expect(
    clippy::too_many_lines,
    reason = "scalar operator semantics are one exhaustive per-width table; splitting it hides which widths a row covers"
)]
pub(crate) fn apply_binary(op: BinOp, left: Value, right: Value) -> Result<Value, EvalError> {
    match (left, right) {
        (Value::U32(left), Value::U32(right)) => match op {
            BinOp::Add | BinOp::WrappingAdd => Ok(Value::U32(left.wrapping_add(right))),
            BinOp::Sub | BinOp::WrappingSub => Ok(Value::U32(left.wrapping_sub(right))),
            BinOp::Mul => Ok(Value::U32(left.wrapping_mul(right))),
            // Unsigned division by zero is the one defined divide case: every
            // emitter guards it with `Select(divisor == 0 ? MAX : x / y)`.
            BinOp::Div => Ok(Value::U32(left.checked_div(right).unwrap_or(u32::MAX))),
            BinOp::Mod => Ok(Value::U32(left.checked_rem(right).unwrap_or(0))),
            BinOp::BitAnd => Ok(Value::U32(left & right)),
            BinOp::BitOr => Ok(Value::U32(left | right)),
            BinOp::BitXor => Ok(Value::U32(left ^ right)),
            BinOp::Shl => Ok(Value::U32(left.wrapping_shl(right & 31))),
            BinOp::Shr => Ok(Value::U32(left.wrapping_shr(right & 31))),
            BinOp::Eq => Ok(Value::Bool(left == right)),
            BinOp::Ne => Ok(Value::Bool(left != right)),
            BinOp::Lt => Ok(Value::Bool(left < right)),
            BinOp::Gt => Ok(Value::Bool(left > right)),
            BinOp::Le => Ok(Value::Bool(left <= right)),
            BinOp::Ge => Ok(Value::Bool(left >= right)),
            BinOp::And => Ok(Value::Bool(left != 0 && right != 0)),
            BinOp::Or => Ok(Value::Bool(left != 0 || right != 0)),
            BinOp::Min => Ok(Value::U32(left.min(right))),
            BinOp::Max => Ok(Value::U32(left.max(right))),
            BinOp::AbsDiff => Ok(Value::U32(left.abs_diff(right))),
            BinOp::SaturatingAdd => Ok(Value::U32(left.saturating_add(right))),
            BinOp::SaturatingSub => Ok(Value::U32(left.saturating_sub(right))),
            BinOp::SaturatingMul => Ok(Value::U32(left.saturating_mul(right))),
            BinOp::RotateLeft => Ok(Value::U32(left.rotate_left(right & 31))),
            BinOp::RotateRight => Ok(Value::U32(left.rotate_right(right & 31))),
            BinOp::MulHigh => Ok(Value::U32(
                ((u64::from(left).wrapping_mul(u64::from(right))) >> 32) as u32,
            )),
            _ => Err(unsupported_binary(op, "u32")),
        },
        (Value::U64(left), Value::U64(right)) => match op {
            BinOp::Add | BinOp::WrappingAdd => Ok(Value::U64(left.wrapping_add(right))),
            BinOp::Sub | BinOp::WrappingSub => Ok(Value::U64(left.wrapping_sub(right))),
            BinOp::Mul => Ok(Value::U64(left.wrapping_mul(right))),
            BinOp::Div => Ok(Value::U64(left.checked_div(right).unwrap_or(u64::MAX))),
            BinOp::Mod => Ok(Value::U64(left.checked_rem(right).unwrap_or(0))),
            BinOp::BitAnd => Ok(Value::U64(left & right)),
            BinOp::BitOr => Ok(Value::U64(left | right)),
            BinOp::BitXor => Ok(Value::U64(left ^ right)),
            BinOp::Shl => Ok(Value::U64(left.wrapping_shl((right & 63) as u32))),
            BinOp::Shr => Ok(Value::U64(left.wrapping_shr((right & 63) as u32))),
            BinOp::Eq => Ok(Value::Bool(left == right)),
            BinOp::Ne => Ok(Value::Bool(left != right)),
            BinOp::Lt => Ok(Value::Bool(left < right)),
            BinOp::Gt => Ok(Value::Bool(left > right)),
            BinOp::Le => Ok(Value::Bool(left <= right)),
            BinOp::Ge => Ok(Value::Bool(left >= right)),
            BinOp::And => Ok(Value::Bool(left != 0 && right != 0)),
            BinOp::Or => Ok(Value::Bool(left != 0 || right != 0)),
            BinOp::Min => Ok(Value::U64(left.min(right))),
            BinOp::Max => Ok(Value::U64(left.max(right))),
            BinOp::AbsDiff => Ok(Value::U64(left.abs_diff(right))),
            BinOp::SaturatingAdd => Ok(Value::U64(left.saturating_add(right))),
            BinOp::SaturatingSub => Ok(Value::U64(left.saturating_sub(right))),
            BinOp::SaturatingMul => Ok(Value::U64(left.saturating_mul(right))),
            BinOp::MulHigh => Ok(Value::U64(
                ((u128::from(left) * u128::from(right)) >> 64) as u64,
            )),
            _ => Err(unsupported_binary(op, "u64")),
        },
        (Value::I32(left), Value::I32(right)) => match op {
            BinOp::Add | BinOp::WrappingAdd => Ok(Value::I32(left.wrapping_add(right))),
            BinOp::Sub | BinOp::WrappingSub => Ok(Value::I32(left.wrapping_sub(right))),
            BinOp::Mul => Ok(Value::I32(left.wrapping_mul(right))),
            // Signed division by zero and `i32::MIN / -1` are undefined on the
            // target backends: the emitter lowers signed division to a raw
            // SDiv with no divisor guard, so answering with a value would make
            // the optimized program produce one the unoptimized program never
            // produces.
            BinOp::Div => {
                if right == 0 || (left == i32::MIN && right == -1) {
                    Err(undefined_i32_division("division", left, right))
                } else {
                    Ok(Value::I32(left.wrapping_div(right)))
                }
            }
            BinOp::Mod => {
                if right == 0 || (left == i32::MIN && right == -1) {
                    Err(undefined_i32_division("remainder", left, right))
                } else {
                    Ok(Value::I32(left.wrapping_rem(right)))
                }
            }
            BinOp::BitAnd => Ok(Value::I32(left & right)),
            BinOp::BitOr => Ok(Value::I32(left | right)),
            BinOp::BitXor => Ok(Value::I32(left ^ right)),
            // The shift count is the operand's bit pattern taken modulo the
            // width, matching the `& 31u` every emitter writes, so a negative
            // count is a large count rather than a rejection.
            BinOp::Shl => Ok(Value::I32(left.wrapping_shl(shift_count_i32(right)))),
            BinOp::Shr => Ok(Value::I32(left.wrapping_shr(shift_count_i32(right)))),
            BinOp::Eq => Ok(Value::Bool(left == right)),
            BinOp::Ne => Ok(Value::Bool(left != right)),
            BinOp::Lt => Ok(Value::Bool(left < right)),
            BinOp::Gt => Ok(Value::Bool(left > right)),
            BinOp::Le => Ok(Value::Bool(left <= right)),
            BinOp::Ge => Ok(Value::Bool(left >= right)),
            BinOp::Min => Ok(Value::I32(left.min(right))),
            BinOp::Max => Ok(Value::I32(left.max(right))),
            BinOp::SaturatingAdd => Ok(Value::I32(left.saturating_add(right))),
            BinOp::SaturatingSub => Ok(Value::I32(left.saturating_sub(right))),
            BinOp::SaturatingMul => Ok(Value::I32(left.saturating_mul(right))),
            // AbsDiff, And, Or, RotateLeft and RotateRight are absent on
            // purpose: typecheck V086 rejects signed AbsDiff, V095 restricts
            // And/Or to u32 and bool, and V094 restricts rotates to u32.
            _ => Err(unsupported_binary(op, "i32")),
        },
        (Value::F32(left), Value::F32(right)) => {
            let left = canonical_f32(left);
            let right = canonical_f32(right);
            match op {
                BinOp::Add => Ok(Value::F32(canonical_f32(left + right))),
                BinOp::Sub => Ok(Value::F32(canonical_f32(left - right))),
                BinOp::Mul => Ok(Value::F32(canonical_f32(left * right))),
                // IEEE-754 division is total: a zero divisor yields a signed
                // infinity, or a NaN for `0.0 / 0.0`, on every backend.
                BinOp::Div => Ok(Value::F32(canonical_f32(left / right))),
                BinOp::Eq => Ok(Value::Bool(left == right)),
                BinOp::Ne => Ok(Value::Bool(left != right)),
                BinOp::Lt => Ok(Value::Bool(left < right)),
                BinOp::Gt => Ok(Value::Bool(left > right)),
                BinOp::Le => Ok(Value::Bool(left <= right)),
                BinOp::Ge => Ok(Value::Bool(left >= right)),
                BinOp::Min => Ok(Value::F32(canonical_f32(left.min(right)))),
                BinOp::Max => Ok(Value::F32(canonical_f32(left.max(right)))),
                // Mod is integer-only: typecheck V089 restricts it to u32/i32.
                _ => Err(unsupported_binary(op, "f32")),
            }
        }
        (Value::Bool(left), Value::Bool(right)) => match op {
            BinOp::And => Ok(Value::Bool(left && right)),
            BinOp::Or => Ok(Value::Bool(left || right)),
            BinOp::Eq => Ok(Value::Bool(left == right)),
            BinOp::Ne => Ok(Value::Bool(left != right)),
            // BitXor on bool is rejected by typecheck V091/V092, which allow
            // only u32 and i32 bitwise operands.
            _ => Err(unsupported_binary(op, "bool")),
        },
        _ => Err(EvalError::new(
            "type mismatch in binary operation. Fix: validate operand types before interpretation.",
        )),
    }
}

/// Shift count for a signed operand: the bit pattern taken modulo 32.
fn shift_count_i32(count: i32) -> u32 {
    u32::from_ne_bytes(count.to_ne_bytes()) & 31
}

/// `(shift, mask)` for the bit-unpack unary ops, identical to the emit
/// lowering and to the CPU oracle: the trailing mask makes the operand's
/// signedness irrelevant, so u32 and i32 words extract the same bits into a
/// u32 result.
fn unpack_shift_mask(op: &UnOp) -> Option<(u32, u32)> {
    match op {
        UnOp::Unpack4Low => Some((0, 0x0F)),
        UnOp::Unpack4High => Some((4, 0x0F)),
        UnOp::Unpack8Low => Some((0, 0xFF)),
        UnOp::Unpack8High => Some((24, 0xFF)),
        _ => None,
    }
}

/// Apply `op` to one scalar.
///
/// # Errors
///
/// Returns [`EvalError`] when `op` has no defined meaning at the operand's
/// width.
pub(crate) fn apply_unary(op: &UnOp, operand: Value) -> Result<Value, EvalError> {
    if let Some((shift, mask)) = unpack_shift_mask(op) {
        let bits = match operand {
            Value::U32(value) => value,
            Value::I32(value) => u32::from_ne_bytes(value.to_ne_bytes()),
            Value::U64(_) => return Err(unsupported_unary(op, "u64")),
            Value::F32(_) => return Err(unsupported_unary(op, "f32")),
            Value::Bool(_) => return Err(unsupported_unary(op, "bool")),
        };
        return Ok(Value::U32((bits >> shift) & mask));
    }
    match operand {
        Value::U32(value) => match op {
            UnOp::Negate => Ok(Value::U32(value.wrapping_neg())),
            UnOp::BitNot => Ok(Value::U32(!value)),
            UnOp::LogicalNot => Ok(Value::Bool(value == 0)),
            UnOp::Popcount => Ok(Value::U32(value.count_ones())),
            UnOp::Clz => Ok(Value::U32(value.leading_zeros())),
            UnOp::Ctz => Ok(Value::U32(value.trailing_zeros())),
            UnOp::ReverseBits => Ok(Value::U32(value.reverse_bits())),
            // The transcendental, rounding and classification ops are f32-only
            // per typecheck V102; folding them over an integer literal used to
            // retype the expression to f32 behind the validator's back.
            _ => Err(unsupported_unary(op, "u32")),
        },
        Value::U64(value) => match op {
            UnOp::Negate => Ok(Value::U64(value.wrapping_neg())),
            UnOp::BitNot => Ok(Value::U64(!value)),
            UnOp::LogicalNot => Ok(Value::Bool(value == 0)),
            UnOp::Popcount => Ok(Value::U64(u64::from(value.count_ones()))),
            UnOp::Clz => Ok(Value::U64(u64::from(value.leading_zeros()))),
            UnOp::Ctz => Ok(Value::U64(u64::from(value.trailing_zeros()))),
            UnOp::ReverseBits => Ok(Value::U64(value.reverse_bits())),
            _ => Err(unsupported_unary(op, "u64")),
        },
        Value::I32(value) => match op {
            UnOp::Negate => Ok(Value::I32(value.wrapping_neg())),
            UnOp::BitNot => Ok(Value::I32(!value)),
            UnOp::Popcount => Ok(Value::I32(value.count_ones() as i32)),
            UnOp::Clz => Ok(Value::I32(value.leading_zeros() as i32)),
            UnOp::Ctz => Ok(Value::I32(value.trailing_zeros() as i32)),
            UnOp::ReverseBits => Ok(Value::I32(value.reverse_bits())),
            // LogicalNot is u32/bool only per typecheck V100.
            _ => Err(unsupported_unary(op, "i32")),
        },
        Value::F32(value) => {
            let value = canonical_f32(value);
            let wrap = |result: f32| Ok(Value::F32(canonical_f32(result)));
            match op {
                UnOp::Negate => wrap(-value),
                UnOp::Abs => wrap(value.abs()),
                UnOp::Sign => wrap(if value == 0.0 { 0.0 } else { value.signum() }),
                UnOp::Sqrt => wrap(libm::sqrtf(value)),
                UnOp::InverseSqrt => wrap(1.0 / libm::sqrtf(value)),
                UnOp::Reciprocal => wrap(1.0 / value),
                UnOp::Exp => wrap(libm::expf(value)),
                UnOp::Exp2 => wrap(libm::exp2f(value)),
                UnOp::Log => wrap(libm::logf(value)),
                UnOp::Log2 => wrap(libm::log2f(value)),
                UnOp::Sin => wrap(libm::sinf(value)),
                UnOp::Cos => wrap(libm::cosf(value)),
                UnOp::Tan => wrap(libm::tanf(value)),
                UnOp::Asin => wrap(libm::asinf(value)),
                UnOp::Acos => wrap(libm::acosf(value)),
                UnOp::Atan => wrap(libm::atanf(value)),
                UnOp::Sinh => wrap(libm::sinhf(value)),
                UnOp::Cosh => wrap(libm::coshf(value)),
                UnOp::Tanh => wrap(libm::tanhf(value)),
                UnOp::Floor => wrap(value.floor()),
                UnOp::Ceil => wrap(value.ceil()),
                UnOp::Round => wrap(value.round()),
                UnOp::Trunc => wrap(value.trunc()),
                UnOp::IsNan => Ok(Value::Bool(value.is_nan())),
                UnOp::IsInf => Ok(Value::Bool(value.is_infinite())),
                UnOp::IsFinite => Ok(Value::Bool(value.is_finite())),
                // LogicalNot is u32/bool only per typecheck V100, and the
                // `unop_logical_not_on_f32_is_rejected` contract pins it.
                _ => Err(unsupported_unary(op, "f32")),
            }
        }
        Value::Bool(value) => match op {
            UnOp::LogicalNot => Ok(Value::Bool(!value)),
            // BitNot needs an integer width (V101) and the IEEE-754
            // classifications need f32 (V102).
            _ => Err(unsupported_unary(op, "bool")),
        },
    }
}
