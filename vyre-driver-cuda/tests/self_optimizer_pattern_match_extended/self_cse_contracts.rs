#![cfg(feature = "device-tests")]

use super::*;

#[test]
fn cuda_xor_self_collapses_via_cse() {
    // store buf 0 (var("x") ^ var("x")) → store buf 0 0
    // Requires CSE-aware pattern_match: canonical[arg1] == canonical[arg2].
    assert_lit_u32(
        &folded_x_store_value(Expr::bitxor(Expr::var("x"), Expr::var("x"))),
        0,
    );
}

#[test]
fn cuda_sub_self_collapses_via_cse() {
    // store buf 0 (var("x") - var("x")) → store buf 0 0
    assert_lit_u32(
        &folded_x_store_value(Expr::sub(Expr::var("x"), Expr::var("x"))),
        0,
    );
}

#[test]
fn cuda_bitand_self_collapses_via_cse() {
    // store buf 0 (var("x") & var("x")) → store buf 0 var("x")
    assert_var(
        &folded_x_store_value(Expr::bitand(Expr::var("x"), Expr::var("x"))),
        "x",
    );
}
