use crate::common::self_optimizer::{b_load_branch_program, binop, if_cond, run_pipeline};
use vyre::ir::UnOp;
use vyre::ir::{BinOp, Expr};

#[test]
fn cuda_const_prop_simplifies_bool_false_comparisons_to_logical_not() {
    for (label, cond) in [
        (
            "b == false",
            binop(BinOp::Eq, Expr::var("b"), Expr::bool(false)),
        ),
        (
            "false == b",
            binop(BinOp::Eq, Expr::bool(false), Expr::var("b")),
        ),
        (
            "b != true",
            binop(BinOp::Ne, Expr::var("b"), Expr::bool(true)),
        ),
        (
            "true != b",
            binop(BinOp::Ne, Expr::bool(true), Expr::var("b")),
        ),
    ] {
        // `if_cond` panics when the branch was folded away, which is itself a
        // failure for these cases: `b` is a runtime value, so the If must
        // survive with a simplified condition.
        let cond = if_cond(&run_pipeline(b_load_branch_program(cond, 1, 2)));
        assert!(
            matches!(
                cond,
                Expr::UnOp {
                    op: UnOp::LogicalNot,
                    ..
                }
            ),
            "{label} must simplify to LogicalNot(b); got cond={cond:?}"
        );
    }
}
