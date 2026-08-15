//! Backend-neutral literal evaluation for the vyre IR.
//!
//! This module recognizes that an expression tree is literal-only and hands
//! each operator to [`crate::scalar_ops`], the one owner of scalar operator
//! semantics. It carries no arithmetic of its own: a rule written here rather
//! than there is a rule the reference interpreter does not know about, which
//! is how the folder and the interpreter came to disagree per width.
//!
//! Backends may use the recursive folder before emission to avoid
//! target-language constant-evaluation traps.

use std::borrow::Cow;

use crate::ir::{BinOp, DataType, Expr, UnOp};
use crate::ir_inner::model::node_kind::Value;
use crate::scalar_ops::{apply_binary, apply_unary, canonical_f32};

/// Recursively fold a literal-only expression tree.
#[must_use]
pub fn fold_literal_tree(expr: &Expr) -> Option<Cow<'_, Expr>> {
    match expr {
        Expr::BinOp { op, left, right } => {
            let folded_left = fold_literal_tree(left);
            let folded_right = fold_literal_tree(right);
            let left = folded_left.as_deref().unwrap_or(left.as_ref());
            let right = folded_right.as_deref().unwrap_or(right.as_ref());
            fold_binary_literal(op, left, right).map(Cow::Owned)
        }
        Expr::Fma { a, b, c } => {
            let folded_a = fold_literal_tree(a);
            let folded_b = fold_literal_tree(b);
            let folded_c = fold_literal_tree(c);
            let a = folded_a.as_deref().unwrap_or(a.as_ref());
            let b = folded_b.as_deref().unwrap_or(b.as_ref());
            let c = folded_c.as_deref().unwrap_or(c.as_ref());
            fold_fma_literal(a, b, c).map(Cow::Owned)
        }
        Expr::UnOp { op, operand } => {
            let folded_operand = fold_literal_tree(operand);
            let operand = folded_operand.as_deref().unwrap_or(operand.as_ref());
            fold_unary_literal(op, operand).map(Cow::Owned)
        }
        Expr::Cast { target, value } => {
            let folded_value = fold_literal_tree(value);
            let value = folded_value.as_deref().unwrap_or(value.as_ref());
            fold_cast_literal(target, value).map(Cow::Owned)
        }
        _ => None,
    }
}

/// Fold one binary operator applied to literal operands.
///
/// `None` means "do not rewrite": either an operand is not a literal, or
/// the private `scalar_ops` module has no defined answer at that width, in which
/// case the expression must survive to validation rather than acquire a value the
/// unoptimized program never produces.
#[must_use]
pub fn fold_binary_literal(op: &BinOp, left: &Expr, right: &Expr) -> Option<Expr> {
    let left = literal_scalar(left)?;
    let right = literal_scalar(right)?;
    scalar_literal(apply_binary(*op, left, right).ok()?)
}

/// Fold one unary operator applied to a literal operand.
///
/// `None` carries the same meaning as in [`fold_binary_literal`].
#[must_use]
pub fn fold_unary_literal(op: &UnOp, operand: &Expr) -> Option<Expr> {
    let operand = literal_scalar(operand)?;
    scalar_literal(apply_unary(op, operand).ok()?)
}

/// Fold a cast applied to a literal operand.
#[must_use]
pub fn fold_cast_literal(target: &DataType, value: &Expr) -> Option<Expr> {
    match (target, value) {
        (DataType::U32, Expr::LitU32(v)) => Some(Expr::LitU32(*v)),
        (DataType::U32, Expr::LitI32(v)) => Some(Expr::LitU32(*v as u32)),
        (DataType::U32, Expr::LitF32(v)) if v.is_finite() => Some(Expr::LitU32(*v as u32)),
        (DataType::U32, Expr::LitBool(v)) => Some(Expr::LitU32(u32::from(*v))),
        (DataType::I32, Expr::LitU32(v)) => Some(Expr::LitI32(*v as i32)),
        (DataType::I32, Expr::LitI32(v)) => Some(Expr::LitI32(*v)),
        (DataType::I32, Expr::LitF32(v)) if v.is_finite() => Some(Expr::LitI32(*v as i32)),
        (DataType::I32, Expr::LitBool(v)) => Some(Expr::LitI32(i32::from(*v))),
        (DataType::F32, Expr::LitU32(v)) => Some(Expr::LitF32(*v as f32)),
        (DataType::F32, Expr::LitI32(v)) => Some(Expr::LitF32(*v as f32)),
        (DataType::F32, Expr::LitF32(v)) => Some(Expr::LitF32(*v)),
        (DataType::F32, Expr::LitBool(v)) => Some(Expr::LitF32(if *v { 1.0 } else { 0.0 })),
        (DataType::Bool, Expr::LitU32(v)) => Some(Expr::LitBool(*v != 0)),
        (DataType::Bool, Expr::LitI32(v)) => Some(Expr::LitBool(*v != 0)),
        (DataType::Bool, Expr::LitF32(v)) => Some(Expr::LitBool(*v != 0.0)),
        (DataType::Bool, Expr::LitBool(v)) => Some(Expr::LitBool(*v)),
        _ => None,
    }
}

/// Fold an FMA with literal operands.
#[must_use]
pub fn fold_fma_literal(a: &Expr, b: &Expr, c: &Expr) -> Option<Expr> {
    match (a, b, c) {
        (Expr::LitF32(a), Expr::LitF32(b), Expr::LitF32(c))
            if !(a.is_nan() || b.is_nan() || c.is_nan()) =>
        {
            Some(Expr::LitF32(canonical_f32(a.mul_add(*b, *c))))
        }
        _ => None,
    }
}

/// Read a literal expression as a scalar; `None` for every other expression.
fn literal_scalar(expr: &Expr) -> Option<Value> {
    match expr {
        Expr::LitU32(value) => Some(Value::U32(*value)),
        Expr::LitI32(value) => Some(Value::I32(*value)),
        Expr::LitF32(value) => Some(Value::F32(*value)),
        Expr::LitBool(value) => Some(Value::Bool(*value)),
        _ => None,
    }
}

/// Write a scalar back as a literal expression.
///
/// `Value::U64` has no `Expr` literal, so a 64-bit result cannot be folded
/// into the expression IR and the expression is left alone.
fn scalar_literal(value: Value) -> Option<Expr> {
    match value {
        Value::U32(value) => Some(Expr::LitU32(value)),
        Value::I32(value) => Some(Expr::LitI32(value)),
        Value::F32(value) => Some(Expr::LitF32(value)),
        Value::Bool(value) => Some(Expr::LitBool(value)),
        Value::U64(_) => None,
    }
}
