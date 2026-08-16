//! Per-operator operand checks.
//!
//! What a binary or unary operator will accept in each position, reported as
//! `ValidationError` rows rather than a first-failure abort so one pass over a
//! program names every bad operand in it.

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::program::BufferDecl;
use crate::ir_inner::model::op_signature::{BinOp, DataType};
use crate::validate::{err, Binding, ValidationError};
use crate::validate::{ValidationLocation, ValidationPhase};
use rustc_hash::FxHashMap;

use super::{expr_type, ScopeTypes};

#[inline]
#[expect(
    clippy::too_many_lines,
    reason = "binary operator validation is kept as one exhaustive BinOp policy table so type-safety edits review the complete operator surface"
)]
pub(crate) fn validate_binop_operands(
    op: BinOp,
    left: &Expr,
    right: &Expr,
    buffers: &FxHashMap<&str, &BufferDecl>,
    scope: &FxHashMap<crate::ir::Ident, Binding>,
    errors: &mut Vec<ValidationError>,
) {
    let left_ty = expr_type(left, &mut ScopeTypes::new(buffers, scope));
    let right_ty = expr_type(right, &mut ScopeTypes::new(buffers, scope));

    match op {
        // Arithmetic: U32, I32, and F32 are all valid in target-text.
        // Bool is NOT  -  `(a && b) + 1` must be rejected at validation time.
        // Operand types must also match: `u32 + f32` is silently ambiguous
        // today and must be rejected (VAL-003).
        //
        // Which operators those are is `BinOp::takes_numeric_operands`, the
        // one owner. Listing them here as well is how the list and the result
        // classifier in `expr_type` drifted apart on `AbsDiff`.
        _ if op.takes_numeric_operands() => {
            if matches!(op, BinOp::Div) && expr_is_static_zero(right) {
                errors.push(err("V044", ValidationPhase::Type, ValidationLocation::Program, "binary operation `Div` has a statically-zero divisor"
                        .to_string(), "guard the divisor, use Select to substitute a non-zero value, or reject the input before building IR."
                        .to_string()));
            }
            if let (Some(l), Some(r)) = (&left_ty, &right_ty) {
                if matches!(l, DataType::U64 | DataType::I64)
                    || matches!(r, DataType::U64 | DataType::I64)
                {
                    errors.push(err("V084", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `{op:?}` received left=`{l}`, right=`{r}`. 64-bit integer arithmetic is outside vyre-foundation's cross-backend arithmetic contract"
                    ), "express the operation as a U32 pair with explicit carry/borrow, or use a backend-specific op whose schema declares native 64-bit arithmetic.".to_string()));
                }

                if matches!(
                    op,
                    BinOp::SaturatingAdd | BinOp::SaturatingSub | BinOp::SaturatingMul
                ) && (l != &DataType::U32 || r != &DataType::U32)
                {
                    errors.push(err("V085", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "Saturating arithmetic `{op:?}` received left=`{l}`, right=`{r}`; legal set is only U32 in the current lowering"
                        )
                            .to_string(), "cast both operands to U32, or clamp explicitly for I32/F32.".to_string()
                            .to_string()));
                }

                if matches!(op, BinOp::AbsDiff) && (l == &DataType::I32 || r == &DataType::I32) {
                    errors.push(err("V086", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "AbsDiff has left=`{l}`, right=`{r}` and can overflow (i32::MIN - i32::MAX invokes target-text signed-integer UB)"
                        )
                            .to_string(), "cast operands to U32 before AbsDiff, or rewrite as an explicit branch.".to_string()
                            .to_string()));
                }
            }
            for (side, ty) in [("left", &left_ty), ("right", &right_ty)] {
                if let Some(ty) = ty {
                    if matches!(ty, DataType::Bool) {
                        errors.push(err("V087", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "binary operation `{op:?}` {side} operand has type `{ty}`, but numeric arithmetic expects one of `u32`, `i32`, or `f32`"
                        ), "cast the operand to U32 or I32 before arithmetic, or rewrite to avoid mixing logical and arithmetic operators.".to_string()));
                    }
                }
            }
            // VAL-003: reject mixed numeric types. target-text has no implicit
            // promotion; `a: u32 + b: f32` must be a cast at the call site,
            // not a silent validator pass.
            if let (Some(l), Some(r)) = (&left_ty, &right_ty) {
                let both_numeric = matches!(l, DataType::U32 | DataType::I32 | DataType::F32)
                    && matches!(r, DataType::U32 | DataType::I32 | DataType::F32);
                if both_numeric && l != r {
                    errors.push(err("V088", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `{op:?}` operands have mismatched numeric types: left=`{l}`, right=`{r}` (legal set: U32, I32, F32)"
                    ), "cast one operand so both sides share a type (target-text has no implicit promotion).".to_string()));
                }
            }
        }
        // Modulo: target emitters support total unsigned modulo and signed
        // modulo with explicit zero/overflow guards, so both operands must be
        // integer operands of the same width.
        BinOp::Mod => {
            if expr_is_static_zero(right) {
                errors.push(err("V044", ValidationPhase::Type, ValidationLocation::Program, "binary operation `Mod` has a statically-zero divisor"
                        .to_string(), "guard the divisor, use Select to substitute a non-zero value, or reject the input before building IR."
                        .to_string()));
            }
            for (side, ty) in [("left", left_ty.as_ref()), ("right", right_ty.as_ref())] {
                if let Some(ty) = ty {
                    if !matches!(ty, DataType::U32 | DataType::I32) {
                        errors.push(err("V089", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "binary operation `Mod` {side} operand must be `u32` or `i32`, got `{ty}`. Legal set for Mod is integer-only"
                        ), "cast both operands to the same integer type before modulo.".to_string()));
                    }
                }
            }
            if let (Some(left), Some(right)) = (&left_ty, &right_ty) {
                if matches!(left, DataType::U32 | DataType::I32)
                    && matches!(right, DataType::U32 | DataType::I32)
                    && left != right
                {
                    errors.push(err("V090", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `Mod` operands have mismatched integer types: left=`{left}`, right=`{right}`"
                    ), "cast one operand so both sides share the same integer type.".to_string()));
                }
            }
        }
        // Bitwise: target-text `&` / `|` / `^` require integer operands of the same type.
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
            if let (Some(l), Some(r)) = (&left_ty, &right_ty) {
                if !matches!(l, DataType::U32 | DataType::I32) {
                    errors.push(err("V091", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `{op:?}` left operand has type `{l}`; legal integer set is `u32` or `i32`"
                    ), "cast the left operand to U32 or I32.".to_string()));
                }
                if !matches!(r, DataType::U32 | DataType::I32) {
                    errors.push(err("V092", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `{op:?}` right operand has type `{r}`; legal integer set is `u32` or `i32`"
                    ), "cast the right operand to U32 or I32.".to_string()));
                }
                if l != r {
                    errors.push(err("V093", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `{op:?}` operands have mismatched integer types: left=`{l}`, right=`{r}`. Integer operands must match and belong to `u32` or `i32`"
                    ), "cast both operands to the same integer type.".to_string()));
                }
            }
        }
        // Shifts and rotates: target-text masks the right operand with `& 31u`,
        // so both sides must be u32. Rotates share the same typing  -
        // left is the bit-pattern, right is the rotation count in bits.
        BinOp::Shl | BinOp::Shr | BinOp::RotateLeft | BinOp::RotateRight => {
            for (side, ty) in [("left", left_ty), ("right", right_ty)] {
                if let Some(ty) = ty {
                    if !matches!(ty, DataType::U32) {
                        errors.push(err("V094", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "binary operation `{op:?}` {side} operand has type `{ty}`; shift/rotate operands must be `u32`"
                        ), "cast the operand to U32 before shifting/rotating.".to_string()));
                    }
                }
            }
        }
        // Logical And/Or: target-text lowers via `!= 0u`, so only u32 and bool are valid.
        BinOp::And | BinOp::Or => {
            for (side, ty) in [("left", left_ty), ("right", right_ty)] {
                if let Some(ty) = ty {
                    if !matches!(ty, DataType::U32 | DataType::Bool) {
                        errors.push(err("V095", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "binary operation `{op:?}` {side} operand has type `{ty}`; logical And/Or operands must be `u32` or `bool`"
                        ), "cast the operand to U32 or Bool.".to_string()));
                    }
                }
            }
        }
        // Comparisons: target-text requires both operands to have the same type.
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
            if let (Some(l), Some(r)) = (&left_ty, &right_ty) {
                if l != r {
                    errors.push(err("V096", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary comparison `{op:?}` operands have mismatched types: left=`{l}`, right=`{r}`. Comparisons require matching types"
                    ), "cast both operands to the same type before comparing.".to_string()));
                }
            }
        }
        BinOp::Shuffle | BinOp::Ballot | BinOp::WaveReduce | BinOp::WaveBroadcast => {
            errors.push(err("V097", ValidationPhase::Type, ValidationLocation::Program, format!(
                "binary operation `{op:?}` requires backend subgroup semantics (`supports_subgroup_ops() == true`) before foundation validation can guarantee safety"
            ), format!(
                "validate with ValidationOptions::with_backend(backend) where `backend.supports_subgroup_ops() == true`, or remove `{op:?}` before lowering."
            )));
        }
        _ => {}
    }
}

#[inline]
fn expr_is_static_zero(expr: &Expr) -> bool {
    match expr {
        Expr::LitU32(0) | Expr::LitI32(0) => true,
        Expr::LitF32(value) => *value == 0.0,
        Expr::Cast { value, .. } => expr_is_static_zero(value),
        _ => false,
    }
}

#[inline]
pub(crate) fn validate_unop_operand(
    op: &crate::ir_inner::model::op_signature::UnOp,
    expr: &Expr,
    buffers: &FxHashMap<&str, &BufferDecl>,
    scope: &FxHashMap<crate::ir::Ident, Binding>,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(ty) = expr_type(expr, &mut ScopeTypes::new(buffers, scope)) {
        match op {
            crate::ir_inner::model::op_signature::UnOp::Negate => {
                if matches!(ty, DataType::I32) {
                    errors.push(err("V098", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `Negate` operand has type `{ty}`, but legal total Negate types are `u32` and `f32`; raw i32 negation has the i32::MIN overflow case"
                    ), "use `0 - x` for wrapping i32 negation, cast to U32 before Negate, or guard with Select(i32::MIN, 0, -x).".to_string()));
                } else if !matches!(ty, DataType::U32 | DataType::F32) {
                    errors.push(err("V099", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `{op:?}` operand has type `{ty}`, but legal set is U32, I32, or F32"
                    ), "cast or rewrite the operand to U32/I32/F32.".to_string()));
                }
            }
            crate::ir_inner::model::op_signature::UnOp::LogicalNot => {
                if !matches!(ty, DataType::U32 | DataType::Bool) {
                    errors.push(err("V100", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `LogicalNot` operand has type `{ty}`; legal set is `u32` or `bool`"
                    ), "cast or rewrite the operand to produce U32 or Bool.".to_string()));
                }
            }
            crate::ir_inner::model::op_signature::UnOp::BitNot
            | crate::ir_inner::model::op_signature::UnOp::Popcount
            | crate::ir_inner::model::op_signature::UnOp::Clz
            | crate::ir_inner::model::op_signature::UnOp::Ctz
            | crate::ir_inner::model::op_signature::UnOp::ReverseBits => {
                // VAL-004: U64 operands are valid for every bitwise-unary
                // op. The reference interpreter handles Value::U64 for
                // BitNot/Popcount/Clz/Ctz/ReverseBits and target-text ≥ the 64-bit
                // extension emits the right intrinsics. Previously the
                // validator rejected U64 and forced an avoidable down-cast.
                if !matches!(ty, DataType::U32 | DataType::I32 | DataType::U64) {
                    errors.push(err("V101", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `{op:?}` operand has type `{ty}`; legal integer set is `u32`, `i32`, or `u64`"
                    ), "cast or rewrite the operand to produce U32, I32, or U64.".to_string()));
                }
            }
            crate::ir_inner::model::op_signature::UnOp::Sin
            | crate::ir_inner::model::op_signature::UnOp::Cos
            | crate::ir_inner::model::op_signature::UnOp::Exp
            | crate::ir_inner::model::op_signature::UnOp::Log
            | crate::ir_inner::model::op_signature::UnOp::Log2
            | crate::ir_inner::model::op_signature::UnOp::Exp2
            | crate::ir_inner::model::op_signature::UnOp::Tan
            | crate::ir_inner::model::op_signature::UnOp::Acos
            | crate::ir_inner::model::op_signature::UnOp::Asin
            | crate::ir_inner::model::op_signature::UnOp::Atan
            | crate::ir_inner::model::op_signature::UnOp::Tanh
            | crate::ir_inner::model::op_signature::UnOp::Sinh
            | crate::ir_inner::model::op_signature::UnOp::Cosh
            | crate::ir_inner::model::op_signature::UnOp::Abs
            | crate::ir_inner::model::op_signature::UnOp::Sqrt
            | crate::ir_inner::model::op_signature::UnOp::InverseSqrt
            | crate::ir_inner::model::op_signature::UnOp::Reciprocal
            | crate::ir_inner::model::op_signature::UnOp::Floor
            | crate::ir_inner::model::op_signature::UnOp::Ceil
            | crate::ir_inner::model::op_signature::UnOp::Round
            | crate::ir_inner::model::op_signature::UnOp::Trunc
            | crate::ir_inner::model::op_signature::UnOp::Sign
            | crate::ir_inner::model::op_signature::UnOp::IsNan
            | crate::ir_inner::model::op_signature::UnOp::IsInf
            | crate::ir_inner::model::op_signature::UnOp::IsFinite => {
                if ty != DataType::F32 {
                    errors.push(err("V102", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `{op:?}` operand has type `{ty}`; legal set for math ops is `f32`"
                    ), "cast or rewrite the operand to produce F32.".to_string()));
                }
            }
            crate::ir_inner::model::op_signature::UnOp::Unpack4Low
            | crate::ir_inner::model::op_signature::UnOp::Unpack4High
            | crate::ir_inner::model::op_signature::UnOp::Unpack8Low
            | crate::ir_inner::model::op_signature::UnOp::Unpack8High => {
                // VAL-004: nibble/byte unpack ops extract a masked, shifted lane
                // from a 32-bit integer word, emit lowers them to
                // `(v >> shift) & mask` and the reference interpreter mirrors it,
                // so operand and result are 32-bit integers. These previously
                // fell through to the `_` catch-all and were rejected as "not
                // recognized" even though that message LISTS them as valid and
                // every backend lowers them: a validator rejecting ops it emits.
                if !matches!(ty, DataType::U32 | DataType::I32) {
                    errors.push(err("V103", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `{op:?}` operand has type `{ty}`; unpack ops require a 32-bit integer (`u32` or `i32`) word"
                    ), "cast or rewrite the operand to produce U32 or I32.".to_string()));
                }
            }
            _ => {
                errors.push(err("V104", ValidationPhase::Type, ValidationLocation::Program, format!(
                    "unary operation `{op:?}` is not recognized"
                ), "use a known UnOp variant from this enum (`Negate`, `LogicalNot`, `BitNot`, `Popcount`, `Clz`, `Ctz`, `ReverseBits`, `Sin`, `Cos`, `Exp`, `Log`, `Log2`, `Exp2`, `Tan`, `Acos`, `Asin`, `Atan`, `Tanh`, `Sinh`, `Cosh`, `Abs`, `Sqrt`, `InverseSqrt`, `Reciprocal`, `Floor`, `Ceil`, `Round`, `Trunc`, `Sign`, `IsNan`, `IsInf`, `IsFinite`, `Unpack4Low`, `Unpack4High`, `Unpack8Low`, `Unpack8High`).".to_string()));
            }
        }
    }
}
