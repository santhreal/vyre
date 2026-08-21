//! Const-fold coverage for the shift, division, remainder, saturating and
//! comparison ops, including the safe-zero guards that must refuse to fold.

#![cfg(feature = "device-tests")]
#![cfg(test)]

mod harness;

use harness::self_optimizer::{
    assert_lit_u32, assert_unfolded_u32_binop, binop, folded_store_value, is_bool_word,
};
use vyre::ir::{BinOp, Expr};

#[test]
fn cuda_const_fold_shl() {
    // 5u32 << 3u32 = 40u32
    assert_lit_u32(
        &folded_store_value(Expr::shl(Expr::u32(5), Expr::u32(3))),
        40,
    );
}

#[test]
fn cuda_const_fold_shr() {
    // 80u32 >> 2u32 = 20u32
    assert_lit_u32(
        &folded_store_value(Expr::shr(Expr::u32(80), Expr::u32(2))),
        20,
    );
}

#[test]
fn cuda_const_fold_div_nonzero() {
    // 100u32 / 4u32 = 25u32
    assert_lit_u32(
        &folded_store_value(Expr::div(Expr::u32(100), Expr::u32(4))),
        25,
    );
}

#[test]
fn cuda_const_fold_div_by_zero_skipped() {
    // 100u32 / 0u32 must NOT fold: dividing by zero would crash the compiler at
    // emit time, so the original Div Expr has to survive intact.
    assert_unfolded_u32_binop(
        &folded_store_value(Expr::div(Expr::u32(100), Expr::u32(0))),
        100,
        0,
    );
}

#[test]
fn cuda_const_fold_rem_nonzero() {
    // 17u32 % 5u32 = 2u32
    assert_lit_u32(
        &folded_store_value(Expr::rem(Expr::u32(17), Expr::u32(5))),
        2,
    );
}

#[test]
fn cuda_const_fold_rem_by_zero_skipped() {
    // 17u32 % 0u32 must NOT fold, for the same reason as Div by zero.
    assert_unfolded_u32_binop(
        &folded_store_value(Expr::rem(Expr::u32(17), Expr::u32(0))),
        17,
        0,
    );
}

#[test]
fn cuda_const_fold_chained_shifts() {
    // (1u32 << 4u32) >> 2u32 = 16 >> 2 = 4
    assert_lit_u32(
        &folded_store_value(Expr::shr(
            Expr::shl(Expr::u32(1), Expr::u32(4)),
            Expr::u32(2),
        )),
        4,
    );
}

#[test]
fn cuda_const_fold_saturating_mul_gpu() {
    // Non-overflowing: 7 * 9 = 63
    assert_lit_u32(
        &folded_store_value(binop(BinOp::SaturatingMul, Expr::u32(7), Expr::u32(9))),
        63,
    );

    // Overflowing: 0xFFFFFFFE * 2 exceeds u32, so it must clamp to MAX rather
    // than wrap.
    assert_lit_u32(
        &folded_store_value(binop(
            BinOp::SaturatingMul,
            Expr::u32(0xFFFF_FFFE),
            Expr::u32(2),
        )),
        u32::MAX,
    );

    // Zero left operand: the result is 0 regardless of the right operand.
    assert_lit_u32(
        &folded_store_value(binop(BinOp::SaturatingMul, Expr::u32(0), Expr::u32(99))),
        0,
    );
}

#[test]
fn cuda_const_fold_eq_lt_gt_le_ge_ne_gpu() {
    // GPU const-fold evaluates every comparison op on literal operands. The
    // kernel writes 0/1 into the value buffer and the decoder reconstructs
    // either LitU32(0|1) or LitBool.
    for (op, expected) in [
        (BinOp::Eq, 0u32),
        (BinOp::Ne, 1),
        (BinOp::Lt, 1),
        (BinOp::Gt, 0),
        (BinOp::Le, 1),
        (BinOp::Ge, 0),
    ] {
        let value = folded_store_value(binop(op, Expr::u32(3), Expr::u32(7)));
        assert!(
            is_bool_word(&value, expected),
            "{op:?}(3, 7) expected {expected}; got {value:?}"
        );
    }
}
