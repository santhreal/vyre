#![cfg(feature = "device-tests")]

use super::*;

#[test]
fn cuda_sub_zero_collapses_to_left() {
    // store buf 0 (var("x") - 0)  →  store buf 0 var("x")
    assert_var(
        &folded_x_store_value(Expr::sub(Expr::var("x"), Expr::u32(0))),
        "x",
    );
}

#[test]
fn cuda_bitand_zero_collapses_to_zero() {
    // store buf 0 (var("x") & 0)  →  store buf 0 0
    assert_lit_u32(
        &folded_x_store_value(Expr::bitand(Expr::var("x"), Expr::u32(0))),
        0,
    );
}

#[test]
fn cuda_bitor_zero_collapses_to_left() {
    // store buf 0 (var("x") | 0)  →  store buf 0 var("x")
    assert_var(
        &folded_x_store_value(Expr::bitor(Expr::var("x"), Expr::u32(0))),
        "x",
    );
}

#[test]
fn cuda_sub_add_cancel_left_via_cse() {
    // store buf 0 ((Var(x) + Var(y)) - Var(x))  →  store buf 0 Var(y)
    // Both x and y are bound to non-literal Loads so they survive const-prop
    // and remain Var refs.
    assert_var(
        &folded_xy_store_value(Expr::sub(
            Expr::add(Expr::var("x"), Expr::var("y")),
            Expr::var("x"),
        )),
        "y",
    );
}
