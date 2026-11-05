//! Integration test crate for the containing Vyre package.

use super::*;

#[test]
fn float_mul_by_two_becomes_add() {
    // x * 2.0 → x + x
    let result = reduce_expr(&Expr::mul(Expr::var("x"), Expr::f32(2.0)));
    assert!(result.is_some());
    let reduced = result.unwrap();
    assert!(matches!(&reduced, Expr::BinOp { op: BinOp::Add, .. }));
}

#[test]
fn float_mul_by_one_becomes_identity() {
    // x * 1.0 → x
    let result = reduce_expr(&Expr::mul(Expr::var("x"), Expr::f32(1.0)));
    assert_eq!(result, Some(Expr::var("x")));
}

#[test]
fn float_mul_by_zero_does_not_hide_runtime_nan() {
    let result = reduce_expr(&Expr::mul(Expr::var("x"), Expr::f32(0.0)));
    assert_eq!(result, None);
}

#[test]
fn float_div_by_two_becomes_mul_half() {
    // x / 2.0 → x * 0.5
    let result = reduce_expr(&Expr::div(Expr::var("x"), Expr::f32(2.0)));
    assert!(result.is_some());
    let reduced = result.unwrap();
    assert!(matches!(&reduced, Expr::BinOp { op: BinOp::Mul, .. }));
}

/// Signed zero decides which additive fold is sound, and the two operators take
/// opposite zeros. `-0.0 + 0.0` is `+0.0`, so eliminating `x + 0.0` rewrites the
/// sign of a negative-zero input; `-0.0 - -0.0` is `+0.0`, so eliminating
/// `x - -0.0` does the same. Rust's `-0.0 == 0.0` hides the difference, which is
/// how both folds shipped unsound, so each case is pinned separately here.
///
/// These cases cover the literal on both sides for addition and the right-hand
/// literal for subtraction, which is the whole fold surface. They say nothing
/// about NaN or infinity, which pass through unfolded either way.
#[test]
fn float_additive_identity_folds_only_the_zero_that_preserves_the_sign() {
    let x = Expr::var("x");

    assert_eq!(
        reduce_expr(&Expr::add(x.clone(), Expr::f32(-0.0))),
        Some(x.clone()),
        "x + -0.0 is x for every input, including -0.0",
    );
    assert_eq!(
        reduce_expr(&Expr::add(Expr::f32(-0.0), x.clone())),
        Some(x.clone()),
        "-0.0 + x is x for every input, including -0.0",
    );
    assert_eq!(
        reduce_expr(&Expr::add(x.clone(), Expr::f32(0.0))),
        None,
        "x + 0.0 must survive: it maps -0.0 to +0.0",
    );
    assert_eq!(
        reduce_expr(&Expr::add(Expr::f32(0.0), x.clone())),
        None,
        "0.0 + x must survive: it maps -0.0 to +0.0",
    );

    assert_eq!(
        reduce_expr(&Expr::sub(x.clone(), Expr::f32(0.0))),
        Some(x.clone()),
        "x - 0.0 is x for every input, including -0.0",
    );
    assert_eq!(
        reduce_expr(&Expr::sub(x, Expr::f32(-0.0))),
        None,
        "x - -0.0 must survive: it maps -0.0 to +0.0",
    );
}

#[test]
fn int_div_by_power_of_two_becomes_shr() {
    // x / 8 → x >> 3
    let result = reduce_expr(&Expr::div(Expr::var("x"), Expr::u32(8)));
    assert!(result.is_some());
    let reduced = result.unwrap();
    assert!(matches!(&reduced, Expr::BinOp { op: BinOp::Shr, .. }));
}

#[test]
fn int_div_by_constant_becomes_mulhi() {
    // x / 3 → mulhi(x, magic) >> shift (Granlund-Montgomery)
    let result = reduce_expr(&Expr::div(Expr::var("x"), Expr::u32(3)));
    assert!(result.is_some(), "x/3 must be strength-reduced");
    let reduced = result.unwrap();
    // The top-level should be a Shr wrapping a MulHigh.
    match &reduced {
        Expr::BinOp {
            op: BinOp::Shr,
            left,
            ..
        } => {
            assert!(
                matches!(
                    left.as_ref(),
                    Expr::BinOp {
                        op: BinOp::MulHigh,
                        ..
                    }
                ),
                "inner must be MulHigh: {left:?}"
            );
        }
        other => panic!("x/3 must reduce to Shr(MulHigh(...)), got {other:?}"),
    }
}

#[test]
fn int_div_by_seven_uses_fixup() {
    // x / 7 needs the fixup path: (t + ((x - t) >> 1)) >> s
    let result = reduce_expr(&Expr::div(Expr::var("x"), Expr::u32(7)));
    assert!(result.is_some(), "x/7 must be strength-reduced");
    // Top level should be Shr wrapping an Add (the fixup accumulation)
    let reduced = result.unwrap();
    match &reduced {
        Expr::BinOp {
            op: BinOp::Shr,
            left,
            ..
        } => {
            assert!(
                matches!(left.as_ref(), Expr::BinOp { op: BinOp::Add, .. }),
                "fixup must produce Add at top: {left:?}"
            );
        }
        other => panic!("x/7 must reduce to Shr(Add(...)), got {other:?}"),
    }
}

#[test]
fn int_mod_by_power_of_two_becomes_bitand() {
    // x % 16 → x & 15
    let result = reduce_expr(&Expr::BinOp {
        op: BinOp::Mod,
        left: Box::new(Expr::var("x")),
        right: Box::new(Expr::u32(16)),
    });
    assert!(result.is_some());
    let reduced = result.unwrap();
    assert!(matches!(
        &reduced,
        Expr::BinOp {
            op: BinOp::BitAnd,
            ..
        }
    ));
}

/// WHY: `x / C` becomes `x * (1/C)` only when the reciprocal is exact, so the
/// pair computes one quotient bit for bit. The rewrite used to fire for every
/// finite non-zero constant, which rounded `1/C` and shifted the result of any
/// division by a constant that is not a power of two.
///
/// The class is the exponent range, not one divisor: every power of two whose
/// reciprocal is normal reduces, down to the subnormal divisor `2^-127` whose
/// reciprocal is still the largest normal power of two. `1.0` is excluded
/// because dividing by it is the identity and drops the multiply entirely.
#[test]
fn float_div_reduces_only_where_the_reciprocal_is_exact() {
    let exact = (-127_i32..=126).map(|exponent| {
        let divisor = if exponent < -126 {
            f32::from_bits(1_u32 << (149 + exponent))
        } else {
            f32::from_bits(((exponent + 127) as u32) << 23)
        };
        (exponent, divisor)
    });
    for (exponent, divisor) in exact {
        assert!(
            (1.0_f32 / divisor).is_normal(),
            "2^{exponent} is only in this class while its reciprocal is normal"
        );
        if exponent != 0 {
            assert_eq!(
                reduce_expr(&Expr::div(Expr::var("x"), Expr::f32(divisor))),
                Some(Expr::mul(Expr::var("x"), Expr::f32(1.0 / divisor))),
                "2^{exponent} has an exact reciprocal and must reduce"
            );
        }
        assert_eq!(
            reduce_expr(&Expr::div(Expr::var("x"), Expr::f32(-divisor))),
            Some(Expr::mul(Expr::var("x"), Expr::f32(-1.0 / divisor))),
            "the sign does not change whether 2^{exponent} reciprocates exactly"
        );
    }
    assert_eq!(
        reduce_expr(&Expr::div(Expr::var("x"), Expr::f32(1.0))),
        Some(Expr::var("x")),
        "dividing by one is the identity, not a multiply by one"
    );
}

/// WHY: a rounded reciprocal is a different program. Each divisor here is
/// finite and non-zero, which is all the previous guard asked for, and each
/// one's reciprocal is inexact or catastrophic.
#[test]
fn float_div_by_an_inexact_reciprocal_stays_a_division() {
    // A significand other than one: the reciprocal rounds.
    for divisor in [3.0_f32, 10.0, 0.1, 1.5, 7.0, -3.0, 1e6] {
        assert_eq!(
            reduce_expr(&Expr::div(Expr::var("x"), Expr::f32(divisor))),
            None,
            "{divisor} has no exact reciprocal and must not reduce"
        );
    }
    // A power of two whose reciprocal overflows: `x * inf`, not `x / C`. The
    // step from -127 to -128 is the whole boundary, so both sides are pinned:
    // -127 reduces in the case above and -128 does not.
    for exponent in [-128_i32, -140, -149] {
        let divisor = f32::from_bits(1_u32 << (149 + exponent));
        assert!(divisor > 0.0 && divisor.is_finite());
        assert!(
            (1.0_f32 / divisor).is_infinite(),
            "2^{exponent} must be one of the divisors the guard exists for"
        );
        assert_eq!(
            reduce_expr(&Expr::div(Expr::var("x"), Expr::f32(divisor))),
            None,
            "2^{exponent} reciprocates out of range and must not reduce"
        );
    }
    // A power of two whose exact reciprocal is subnormal, which the rule
    // declines rather than reasoning about underflow in the product.
    let largest = f32::from_bits(254_u32 << 23);
    assert!(!(1.0_f32 / largest).is_normal() && (1.0_f32 / largest) > 0.0);
    assert_eq!(
        reduce_expr(&Expr::div(Expr::var("x"), Expr::f32(largest))),
        None,
        "2^127 reciprocates to a subnormal and must not reduce"
    );
}

#[test]
fn float_one_div_variable_becomes_reciprocal_unop() {
    let result = reduce_expr(&Expr::div(Expr::f32(1.0), Expr::var("x")));
    assert_eq!(result, Some(Expr::reciprocal(Expr::var("x"))));
}

#[test]
fn float_div_by_nan_does_not_reduce() {
    let result = reduce_expr(&Expr::div(Expr::var("x"), Expr::f32(f32::NAN)));
    assert!(result.is_none(), "NaN divisor must not fold");
}

#[test]
fn float_div_by_zero_does_not_reduce() {
    let result = reduce_expr(&Expr::div(Expr::var("x"), Expr::f32(0.0)));
    assert!(result.is_none(), "zero divisor must not fold");
}

// ── Shift-add decomposition tests ────────────────────────────────
