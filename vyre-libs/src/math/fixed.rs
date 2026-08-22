//! Signed 16.16 fixed-point arithmetic over `Expr`.
//!
//! 16.16 fixed-point is a signed format: an operand is a two's-complement i32
//! carried in a u32, so a negative value is stored wrapped. `Expr::mul`,
//! `Expr::mulhi` and `Expr::div` are unsigned, so a kernel that reaches for them
//! directly corrupts every negative intermediate. Both helpers here are strict
//! correctness supersets of the unsigned forms: for non-negative operands they
//! emit bit-identical IR.

use vyre_foundation::ir::Expr;

/// Return `(left * right) >> 16` for signed 16.16 fixed-point lanes without
/// losing the high half of the product to 32-bit overflow.
pub(crate) fn fixed_mul_16_16_expr(left: Expr, right: Expr) -> Expr {
    // Extracting the 16.16 product as `(low >> 16) | (high << 16)` requires the
    // SIGNED 64-bit high word. `Expr::mulhi` is UNSIGNED, so reconstruct the
    // signed high word with the standard correction:
    //   signed_high = unsigned_high - (left < 0 ? right : 0) - (right < 0 ? left : 0)
    // An all-unsigned `mulhi` treats a negative operand as ~2^32 and produces a
    // garbage giant product, which is what made the fixed-point AMG V-cycle
    // diverge from its f64 reference the moment a residual `b - A*x` went
    // negative. For non-negative operands (|v| < 2^31, every legitimate 16.16
    // magnitude) both corrections are zero.
    let low = Expr::mul(left.clone(), right.clone());
    let unsigned_high = Expr::mulhi(left.clone(), right.clone());
    // `0 - (x >> 31)` is an all-ones mask when `x`'s sign bit is set, else zero (logical u32 shift).
    let left_sign_mask = Expr::sub(Expr::u32(0), Expr::shr(left.clone(), Expr::u32(31)));
    let right_sign_mask = Expr::sub(Expr::u32(0), Expr::shr(right.clone(), Expr::u32(31)));
    let correction_left = Expr::bitand(left_sign_mask, right);
    let correction_right = Expr::bitand(right_sign_mask, left);
    let signed_high = Expr::sub(Expr::sub(unsigned_high, correction_left), correction_right);
    Expr::bitor(
        Expr::shr(low, Expr::u32(16)),
        Expr::shl(signed_high, Expr::u32(16)),
    )
}

/// Signed integer division of a two's-complement `numerator` by a
/// known-positive `denominator`, truncating toward zero.
///
/// `Expr::div` is unsigned, so dividing a wrapped-negative 16.16 numerator (a
/// Jacobi residual `b - A*x` that went negative) by a small positive integer
/// yields garbage. This computes `sign*(|numerator| / denominator)` with the
/// branchless mask-abs idiom: `mask = numerator >> 31` broadcast to all-ones on
/// a negative value, `abs = (n ^ m) - m`, `q = abs / d` (a genuine unsigned
/// divide of a non-negative magnitude), then reapply the sign as `(q ^ m) - m`.
/// For a non-negative numerator `mask == 0` and this reduces to plain
/// `Expr::div`. The denominator MUST be positive; a negative denominator is not
/// handled.
pub(crate) fn fixed_sdiv_by_positive_expr(numerator: Expr, denominator: Expr) -> Expr {
    // `numerator >> 31` is 0 or 1 (logical u32 shift); `0 - that` broadcasts to the all-ones sign mask.
    let sign_mask = Expr::sub(Expr::u32(0), Expr::shr(numerator.clone(), Expr::u32(31)));
    // abs(numerator) = (numerator ^ sign_mask) - sign_mask (two's-complement branchless absolute value).
    let magnitude = Expr::sub(
        Expr::bitxor(numerator, sign_mask.clone()),
        sign_mask.clone(),
    );
    let quotient = Expr::div(magnitude, denominator);
    // Reapply the original sign: (quotient ^ sign_mask) - sign_mask.
    Expr::sub(Expr::bitxor(quotient, sign_mask.clone()), sign_mask)
}
