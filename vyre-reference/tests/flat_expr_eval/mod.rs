//! Evaluating a single `Expr` through the flat reference evaluator.
//!
//! Three test targets sweep the flat evaluator: the adversarial proptest, the
//! adversarial gap suite, and the subnormal flushing contract. Each one used to
//! carry its own copy of the wrapper program, the zero invocation, and the
//! literal-to-`Expr` builders, so a change to how a bare expression is evaluated
//! had to be made three times and the copies had already drifted in their panic
//! text. The harness lives here; the expectations each target compares against
//! stay in that target.
//!
//! `canonical_f32` here is deliberately a restatement rather than a call to
//! `vyre_reference::ieee754::canonical_f32`. These sweeps compare the reference
//! interpreter against an independent statement of IEEE-754 canonicalization, so
//! calling the interpreter's own helper would make the oracle check itself.
//! `subnormal_contract.rs` owns the direct contract on the public helper.
#![allow(dead_code)]

use vyre_foundation::ir::{BinOp, Expr, Program, UnOp};
use vyre_reference::execution::expr as eval_expr;
use vyre_reference::value::Value;
use vyre_reference::workgroup::{Invocation, InvocationIds, Memory};

/// A program with no buffers and a single workgroup.
pub(crate) fn empty_program() -> Program {
    Program::wrapped(Vec::new(), [1, 1, 1], Vec::new())
}

/// The invocation at thread zero of `program`.
pub(crate) fn zero_invocation(program: &Program) -> Invocation<'_> {
    Invocation::new(InvocationIds::ZERO, program.entry())
}

/// Evaluate `expr` with no buffers bound.
pub(crate) fn eval_expr_value(expr: &Expr) -> Value {
    let program = empty_program();
    eval_expr::eval(
        expr,
        &mut zero_invocation(&program),
        &mut Memory::empty(),
        &program,
    )
    .expect("Fix: flat reference evaluator must evaluate generated expression")
}

pub(crate) fn eval_binop_u32(op: BinOp, a: u32, b: u32) -> Value {
    eval_expr_value(&Expr::BinOp {
        op,
        left: Box::new(Expr::u32(a)),
        right: Box::new(Expr::u32(b)),
    })
}

pub(crate) fn eval_binop_i32(op: BinOp, a: i32, b: i32) -> Value {
    eval_expr_value(&Expr::BinOp {
        op,
        left: Box::new(Expr::i32(a)),
        right: Box::new(Expr::i32(b)),
    })
}

pub(crate) fn eval_binop_f32(op: BinOp, a: f32, b: f32) -> Value {
    eval_expr_value(&Expr::BinOp {
        op,
        left: Box::new(Expr::f32(a)),
        right: Box::new(Expr::f32(b)),
    })
}

pub(crate) fn eval_unop_u32(op: UnOp, a: u32) -> Value {
    eval_expr_value(&Expr::UnOp {
        op,
        operand: Box::new(Expr::u32(a)),
    })
}

pub(crate) fn eval_unop_i32(op: UnOp, a: i32) -> Value {
    eval_expr_value(&Expr::UnOp {
        op,
        operand: Box::new(Expr::i32(a)),
    })
}

pub(crate) fn eval_unop_f32(op: UnOp, a: f32) -> Value {
    eval_expr_value(&Expr::UnOp {
        op,
        operand: Box::new(Expr::f32(a)),
    })
}

/// The f32 bit pattern behind a float `Value`.
pub(crate) fn float_bits(value: Value) -> u32 {
    match value {
        Value::Float(v) => (v as f32).to_bits(),
        other => panic!("expected float value, got {other:?}"),
    }
}

/// IEEE-754 canonicalization stated independently of the interpreter: any NaN
/// collapses to the canonical quiet NaN, any subnormal flushes to its signed
/// zero.
pub(crate) fn canonical_f32(value: f32) -> f32 {
    if value.is_nan() {
        f32::from_bits(0x7FC0_0000)
    } else if value.is_subnormal() {
        f32::from_bits(value.to_bits() & 0x8000_0000)
    } else {
        value
    }
}

/// The `Value` a canonicalized `value` must compare equal to.
pub(crate) fn expected_f32(value: f32) -> Value {
    Value::Float(f64::from(canonical_f32(value)))
}
