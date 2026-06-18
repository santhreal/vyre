//! Test: BinOp literal folding semantics.
use super::*;
use naga::{Expression, Literal};

/// BinOp::WrappingSub of two u32 literals must fold to wrapping_sub, not
/// saturating_sub. 0u32 WrappingSub 1u32 must produce 0xFFFF_FFFFu32 — the
/// two's-complement wrap-around that the GPU would compute at runtime. The
/// previous saturating_sub produced Literal::U32(0), which is a silently
/// wrong constant (any downstream op using it would compute with the wrong
/// value without any error).
#[test]
fn fold_literal_wrapping_sub_u32_underflow_wraps() {
    let desc = KernelDescriptor {
        id: "wrapping_sub_fold".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![
                // result 0: literal 0u32 (left operand)
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                // result 1: literal 1u32 (right operand)
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(1),
                },
                // result 2: 0u32 WrappingSub 1u32 — must fold to 0xFFFF_FFFF,
                // NOT 0 (saturating). Operands are [left_result_id,
                // right_result_id].
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::WrappingSub),
                    operands: vec![0, 1],
                    result: Some(2),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
        },
    };
    let module = emit(&desc).expect("WrappingSub of u32 literals must emit without error");

    // The fold must have produced Literal::U32(0xFFFF_FFFF) in the
    // expression arena. A saturating_sub would leave Literal::U32(0) here
    // instead, which is a silently wrong constant.
    let entry = &module.entry_points[0];
    let has_wrapping_result = entry
        .function
        .expressions
        .iter()
        .any(|(_, expr)| matches!(expr, Expression::Literal(Literal::U32(0xFFFF_FFFF))));
    assert!(
        has_wrapping_result,
        "fold_literal_binop WrappingSub(0u32, 1u32) must produce Literal::U32(0xFFFF_FFFF); \
         got saturating result (Literal::U32(0)) instead — GPU u32 subtraction wraps, not saturates"
    );
}

/// BinOp::Sub of two u32 literals with underflow must also wrap, because
/// WGSL u32 arithmetic is two's-complement at runtime; fold semantics must
/// match runtime semantics.
#[test]
fn fold_literal_plain_sub_u32_underflow_wraps() {
    let desc = KernelDescriptor {
        id: "plain_sub_fold".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Sub),
                    operands: vec![0, 1],
                    result: Some(2),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
        },
    };
    let module = emit(&desc).expect("Sub of u32 literals must emit without error");
    let entry = &module.entry_points[0];
    let has_wrapping_result = entry
        .function
        .expressions
        .iter()
        .any(|(_, expr)| matches!(expr, Expression::Literal(Literal::U32(0xFFFF_FFFF))));
    assert!(
        has_wrapping_result,
        "fold_literal_binop Sub(0u32, 1u32) must produce Literal::U32(0xFFFF_FFFF); \
         WGSL u32 subtraction wraps at runtime"
    );
}
