use super::*;

#[test]
fn cuda_double_bitnot_collapses() {
    // store buf 0 (~~ var("x"))  →  store buf 0 var("x")
    // BitNot is involutive, unlike Abs, so the double application does collapse
    // all the way to the operand.
    assert_var(
        &folded_x_store_value(unop(UnOp::BitNot, unop(UnOp::BitNot, Expr::var("x")))),
        "x",
    );
}

#[test]
fn cuda_bitand_max_collapses_to_left() {
    // store buf 0 (var("x") & u32::MAX)  →  store buf 0 var("x")
    assert_var(
        &folded_x_store_value(Expr::bitand(Expr::var("x"), Expr::u32(u32::MAX))),
        "x",
    );
}

#[test]
fn cuda_shl_zero_collapses_to_left() {
    // store buf 0 (var("x") << 0)  →  store buf 0 var("x")
    assert_var(
        &folded_x_store_value(Expr::shl(Expr::var("x"), Expr::u32(0))),
        "x",
    );
}

#[test]
fn cuda_shr_zero_collapses_to_left() {
    // store buf 0 (var("x") >> 0)  →  store buf 0 var("x")
    assert_var(
        &folded_x_store_value(Expr::shr(Expr::var("x"), Expr::u32(0))),
        "x",
    );
}

#[test]
fn cuda_zero_shl_collapses_to_zero() {
    // store buf 0 (0u32 << var("x"))  →  store buf 0 0
    assert_lit_u32(
        &folded_x_store_value(Expr::shl(Expr::u32(0), Expr::var("x"))),
        0,
    );
}
