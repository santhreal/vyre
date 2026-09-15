use super::*;

#[test]
fn cuda_eq_self_collapses_to_true_via_cse() {
    // store buf 0 (var("x") == var("x"))  →  store buf 0 LitBool(true)
    assert_lit_bool(
        &folded_x_store_value(Expr::eq(Expr::var("x"), Expr::var("x"))),
        true,
    );
}

#[test]
fn cuda_bool_and_self_collapses_via_cse() {
    // (b && b) → b. Both operands are Var(b) and CSE proves equality, so the
    // rewritten cond no longer has BinOp::And on top. The If itself survives
    // because `b` is a runtime value.
    let out = run_pipeline(b_load_branch_program(
        binop(BinOp::And, Expr::var("b"), Expr::var("b")),
        1,
        2,
    ));
    assert_cond_not_headed_by(&out, BinOp::And);
}

#[test]
fn cuda_bool_and_with_false_collapses_to_false() {
    // (Var(b) && false) → false, which is statically decidable, so the If is
    // dropped and the else arm's marker is the only store left.
    let out = run_pipeline(b_load_branch_program(
        binop(BinOp::And, Expr::var("b"), Expr::bool(false)),
        1,
        99,
    ));
    assert_branch_folded_to(&out, 99);
}

#[test]
fn cuda_bool_or_with_true_collapses_to_true() {
    // (Var(b) || true) → true, so the If is dropped and the then arm's marker
    // is the only store left.
    let out = run_pipeline(b_load_branch_program(
        binop(BinOp::Or, Expr::var("b"), Expr::bool(true)),
        1,
        99,
    ));
    assert_branch_folded_to(&out, 1);
}

#[test]
fn cuda_gt_self_collapses_to_false_via_cse() {
    // (var("x") > var("x")) → LitBool(false). Adversarial: catches the previous
    // miswiring where `is_cmp_gt` was bound to the wrong op tag, which would
    // have collapsed `Gt(x,x)` to `true`.
    assert_lit_bool(
        &folded_x_store_value(binop(BinOp::Gt, Expr::var("x"), Expr::var("x"))),
        false,
    );
}

#[test]
fn cuda_le_self_collapses_to_true_via_cse() {
    // (var("x") <= var("x")) → LitBool(true)
    assert_lit_bool(
        &folded_x_store_value(binop(BinOp::Le, Expr::var("x"), Expr::var("x"))),
        true,
    );
}

#[test]
fn cuda_ge_self_collapses_to_true_via_cse() {
    // (var("x") >= var("x")) → LitBool(true)
    assert_lit_bool(
        &folded_x_store_value(binop(BinOp::Ge, Expr::var("x"), Expr::var("x"))),
        true,
    );
}

#[test]
fn cuda_lt_self_collapses_to_false_via_cse() {
    // (var("x") < var("x")) → LitBool(false)
    assert_lit_bool(
        &folded_x_store_value(Expr::lt(Expr::var("x"), Expr::var("x"))),
        false,
    );
}
