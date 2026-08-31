//! End-to-end test: constant propagation in the GPU pipeline.
//!
//! After const-fold + let-dedupe, the const-prop CPU rewrite turns
//! `Var(name)` into `LitU32(value)` whenever `name` was let-bound to
//! a literal in an enclosing scope. Subsequent DCE drops the now-
//! unused let bindings.

#![cfg(all(test, feature = "device-tests"))]

mod harness;
#[path = "harness/self_optimizer_const_prop_bool.rs"]
mod self_optimizer_const_prop_bool;

use harness::self_optimizer::{
    assert_cond_not_headed_by, assert_lit_i32, assert_lit_u32, assert_var, b_load_branch_program,
    binds_any_let, binds_let, binop, folded_store_value, is_lit_u32, run_pipeline, store_value,
    taken_branch_marker, unop,
};
use vyre::ir::UnOp;
use vyre::ir::{BinOp, Expr, Node, Program};

#[test]
fn cuda_const_prop_replaces_var_with_literal() {
    // let a = 42
    // store buf 0 (Var a)
    //   ⇒ const-prop should rewrite the store to `store buf 0 42`
    //   ⇒ DCE drops the now-dead `let a = 42`
    let out = run_pipeline(Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![
            Node::let_bind("a", Expr::u32(42)),
            Node::store("buf", Expr::u32(0), Expr::var("a")),
        ],
    ));
    assert_lit_u32(&store_value(&out), 42);
    // The `let a = 42` should be dropped by DCE since its only use
    // was rewritten to a literal.
    assert!(
        !binds_let(&out, "a"),
        "DCE should drop `let a` after const-prop replaced its only use"
    );
}

#[test]
fn cuda_const_prop_cascades_through_dedupe() {
    // let a = 5
    // let b = 5     ← CSE rewrites RHS to Var(a)
    // store buf 0 (Var b)
    //   After CSE+let-dedupe: `let b = Var(a)`.
    //   After const-prop: `let b = 5` (since Var(a) → 5), then
    //   store turns into `store buf 0 5`. DCE drops both lets.
    let out = run_pipeline(Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![
            Node::let_bind("a", Expr::u32(5)),
            Node::let_bind("b", Expr::u32(5)),
            Node::store("buf", Expr::u32(0), Expr::var("b")),
        ],
    ));
    assert_lit_u32(&store_value(&out), 5);
    // Both lets should be dead after the cascading rewrite.
    assert!(
        !binds_any_let(&out),
        "DCE should drop both lets after const-prop cascades"
    );
}

#[test]
fn cuda_const_prop_folds_i32_arithmetic() {
    // store buf 0 (LitI32(7) - LitI32(10))   →  store buf 0 LitI32(-3)
    assert_lit_i32(
        &folded_store_value(Expr::sub(Expr::i32(7), Expr::i32(10))),
        -3,
    );
}

#[test]
fn cuda_const_prop_folds_i32_via_var() {
    // let n = LitI32(-5)
    // store buf 0 (Var(n) * LitI32(3))   →  store buf 0 LitI32(-15)
    let out = run_pipeline(Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![
            Node::let_bind("n", Expr::i32(-5)),
            Node::store("buf", Expr::u32(0), Expr::mul(Expr::var("n"), Expr::i32(3))),
        ],
    ));
    assert_lit_i32(&store_value(&out), -15);
}

#[test]
fn cuda_select_const_true_collapses_to_arm() {
    // store buf 0 (Select(true, 1, 99)) → store buf 0 1
    assert_lit_u32(
        &folded_store_value(Expr::Select {
            cond: Box::new(Expr::bool(true)),
            true_val: Box::new(Expr::u32(1)),
            false_val: Box::new(Expr::u32(99)),
        }),
        1,
    );
}

#[test]
fn cuda_select_const_zero_keeps_false_arm() {
    // store buf 0 (Select(0u32, 1, 7)) → store buf 0 7
    assert_lit_u32(
        &folded_store_value(Expr::Select {
            cond: Box::new(Expr::u32(0)),
            true_val: Box::new(Expr::u32(1)),
            false_val: Box::new(Expr::u32(7)),
        }),
        7,
    );
}

#[test]
fn cuda_const_prop_folds_u32_min_max_absdiff() {
    // Min(20, 7) → 7; Max(20, 7) → 20; AbsDiff(20, 7) → 13.
    for (op, expected) in [
        (BinOp::Min, 7u32),
        (BinOp::Max, 20u32),
        (BinOp::AbsDiff, 13u32),
    ] {
        let value = folded_store_value(binop(op, Expr::u32(20), Expr::u32(7)));
        assert!(
            is_lit_u32(&value, expected),
            "expected LitU32({expected}) after {op:?} fold; got {value:?}"
        );
    }
}

#[test]
fn cuda_const_prop_folds_saturating_arithmetic() {
    // SaturatingAdd(MAX-3, 10) clamps to MAX rather than wrapping.
    assert_lit_u32(
        &folded_store_value(binop(
            BinOp::SaturatingAdd,
            Expr::u32(u32::MAX - 3),
            Expr::u32(10),
        )),
        u32::MAX,
    );
    // SaturatingSub(5, 8) clamps at 0 rather than underflowing.
    assert_lit_u32(
        &folded_store_value(binop(BinOp::SaturatingSub, Expr::u32(5), Expr::u32(8))),
        0,
    );
}

#[test]
fn cuda_const_prop_folds_unop_literals() {
    // BitNot(0xF0F0F0F0) → 0x0F0F0F0F
    assert_lit_u32(
        &folded_store_value(unop(UnOp::BitNot, Expr::u32(0xF0F0_F0F0))),
        0x0F0F_0F0F,
    );

    // Popcount(0xFF) → 8
    assert_lit_u32(
        &folded_store_value(unop(UnOp::Popcount, Expr::u32(0xFF))),
        8,
    );

    // LogicalNot(true) → false, so the branch picks the else arm.
    let out = run_pipeline(Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![
            Node::let_bind("b", Expr::bool(true)),
            Node::if_then_else(
                unop(UnOp::LogicalNot, Expr::var("b")),
                vec![Node::store("buf", Expr::u32(0), Expr::u32(1))],
                vec![Node::store("buf", Expr::u32(0), Expr::u32(99))],
            ),
        ],
    ));
    let value = store_value(&out);
    assert!(
        is_lit_u32(&value, 99),
        "LogicalNot(true) → false should pick the else arm; got {value:?}"
    );
}

#[test]
fn cuda_const_prop_folds_bool_binops() {
    // (true && false) → false; gates the else branch.
    let marker = taken_branch_marker(binop(BinOp::And, Expr::bool(true), Expr::bool(false)), 1, 7);
    assert!(
        is_lit_u32(&marker, 7),
        "(true && false) → false should pick else; got {marker:?}"
    );
}

#[test]
fn cuda_const_prop_simplifies_bool_eq_with_literal() {
    // (b == true) collapses to Var(b); the If survives because `b` is a
    // runtime value. Without folding the cond stays a BinOp::Eq.
    let out = run_pipeline(b_load_branch_program(
        binop(BinOp::Eq, Expr::var("b"), Expr::bool(true)),
        1,
        2,
    ));
    assert_cond_not_headed_by(&out, BinOp::Eq);
}

#[test]
fn cuda_const_prop_preserves_non_literal_var() {
    // let a = Load(buf, 0)   ← NOT a literal; const-prop must skip
    // store buf 0 (Var a)
    //   The store keeps its `Var(a)` reference.
    let out = run_pipeline(Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![
            Node::let_bind("a", Expr::load("buf", Expr::u32(0))),
            Node::store("buf", Expr::u32(0), Expr::var("a")),
        ],
    ));
    assert_var(&store_value(&out), "a");
}
