use super::*;

#[test]
fn cuda_sub_add_cancel_right_via_cse() {
    // store buf 0 ((Var(x) + Var(y)) - Var(y))  →  store buf 0 Var(x)
    assert_var(
        &folded_xy_store_value(Expr::sub(
            Expr::add(Expr::var("x"), Expr::var("y")),
            Expr::var("y"),
        )),
        "x",
    );
}

#[test]
fn cuda_add_sub_cancel_via_cse() {
    // store buf 0 ((Var(x) - Var(y)) + Var(y))  →  store buf 0 Var(x)
    assert_var(
        &folded_xy_store_value(Expr::add(
            Expr::sub(Expr::var("x"), Expr::var("y")),
            Expr::var("y"),
        )),
        "x",
    );
}

#[test]
fn cuda_div_by_one_collapses_to_left() {
    // store buf 0 (var("x") / 1) → store buf 0 var("x")
    assert_var(
        &folded_x_store_value(Expr::div(Expr::var("x"), Expr::u32(1))),
        "x",
    );
}

#[test]
fn cuda_mod_by_one_collapses_to_zero() {
    // store buf 0 (var("x") % 1) → store buf 0 0
    assert_lit_u32(
        &folded_x_store_value(Expr::rem(Expr::var("x"), Expr::u32(1))),
        0,
    );
}

#[test]
fn cuda_double_abs_does_not_collapse_to_inner() {
    // Abs is idempotent (Abs(Abs(x)) == Abs(x)), NOT involutive
    // (Abs(Abs(x)) ≠ x in general). Adversarial test: catches a previous bug
    // where the UnOp double-application matcher fired for any same-op pair,
    // incorrectly collapsing Abs(Abs(x)) → x.
    //
    // Either the outer Abs is preserved (correct shape) OR it collapsed to the
    // inner Abs (also correct, since Abs is idempotent). Either way the result
    // must NOT be raw Var(x).
    let value = folded_x_store_value(unop(UnOp::Abs, unop(UnOp::Abs, Expr::var("x"))));
    assert!(
        !matches!(&value, Expr::Var(n) if n.as_str() == "x"),
        "Abs(Abs(x)) must not collapse to Var(x); got {value:?}"
    );
}
