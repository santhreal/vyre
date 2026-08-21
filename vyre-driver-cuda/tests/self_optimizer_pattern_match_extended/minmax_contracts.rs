use super::*;

#[test]
fn cuda_min_with_zero_collapses_to_zero() {
    // Min(x, 0u) → 0u (u32 minimum is 0).
    assert_lit_u32(
        &folded_x_store_value(binop(BinOp::Min, Expr::var("x"), Expr::u32(0))),
        0,
    );
}

#[test]
fn cuda_max_with_max_collapses_to_max() {
    // Max(x, MAX) → MAX (u32 maximum saturates).
    assert_lit_u32(
        &folded_x_store_value(binop(BinOp::Max, Expr::var("x"), Expr::u32(u32::MAX))),
        u32::MAX,
    );
}

#[test]
fn cuda_min_with_max_collapses_to_left() {
    // Min(x, MAX) → x (clamp to MAX is a no-op).
    assert_var(
        &folded_x_store_value(binop(BinOp::Min, Expr::var("x"), Expr::u32(u32::MAX))),
        "x",
    );
}

#[test]
fn cuda_max_with_zero_collapses_to_left() {
    // Max(x, 0u) → x (clamp from below by 0 is a no-op for u32).
    assert_var(
        &folded_x_store_value(binop(BinOp::Max, Expr::var("x"), Expr::u32(0))),
        "x",
    );
}

#[test]
fn cuda_min_self_collapses_via_cse() {
    // store buf 0 (min(x, x)) → store buf 0 var("x")
    assert_var(
        &folded_x_store_value(binop(BinOp::Min, Expr::var("x"), Expr::var("x"))),
        "x",
    );
}

#[test]
fn cuda_max_self_collapses_via_cse() {
    // store buf 0 (max(x, x)) → store buf 0 var("x")
    assert_var(
        &folded_x_store_value(binop(BinOp::Max, Expr::var("x"), Expr::var("x"))),
        "x",
    );
}

#[test]
fn cuda_absdiff_self_collapses_to_zero() {
    // store buf 0 (absdiff(x, x)) → store buf 0 0
    assert_lit_u32(
        &folded_x_store_value(binop(BinOp::AbsDiff, Expr::var("x"), Expr::var("x"))),
        0,
    );
}

#[test]
fn cuda_bitxor_zero_collapses_to_left() {
    // store buf 0 (var("x") ^ 0)  →  store buf 0 var("x")
    assert_var(
        &folded_x_store_value(Expr::bitxor(Expr::var("x"), Expr::u32(0))),
        "x",
    );
}
