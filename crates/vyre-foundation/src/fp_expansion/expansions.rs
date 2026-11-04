//! The five strict-IEEE f32 expansions.
//!
//! Each builds an `Expr` from correctly-rounded f32 add, subtract, multiply,
//! minimum, maximum and comparison, from exact integer operations on the
//! exponent and mantissa fields, and from bit-preserving casts between the two.
//! Division, fused multiply-add, an f32 literal and every approximate native
//! instruction are absent by construction, which is what makes an expanded
//! program produce identical bits on a device and in the reference interpreter
//! once [`FloatLoweringMode::StrictIeee`](crate::fp_parity::FloatLoweringMode)
//! blocks contraction.

use super::constants::{
    ABS_MASK, COS_COEFFICIENTS, EXPONENT_BIAS, EXPONENT_BIAS_F32_BITS, EXPONENT_SHIFT,
    EXP_CLAMP_HI_BITS, EXP_CLAMP_LO_BITS, EXP_COEFFICIENTS, EXP_FINITE_EXPONENT_FIELD,
    EXP_OVERFLOW_ABOVE_BITS, EXP_SATURATED_EXPONENT_FIELD, EXP_UNDERFLOW_BELOW_BITS, HALF_BITS,
    INFINITY_BITS, INV_PIO2_BITS, LN2_HI_BITS, LN2_LO_BITS, LOG2_E_BITS, LOG_COEFFICIENTS,
    MANTISSA_MASK, MIN_NORMAL_BITS, NAN_RESULT_BITS, NEG_INFINITY_BITS, NEG_ONE_BITS, ONE_BITS,
    PIO2_1_BITS, PIO2_2_BITS, PIO2_3_BITS, ROUND_SHIFT_BITS, RSQRT_NEWTON_STEPS, RSQRT_SEED_MAGIC,
    SIGN_MASK, SIN_COEFFICIENTS, SIN_COS_DOMAIN_BITS, SIN_COS_QUADRANT_BIAS_BITS,
    SIN_COS_QUOTIENT_LIMIT_BITS, SIN_COS_QUOTIENT_LIMIT_NEG_BITS, SIN_COS_SMALL_ARGUMENT_BITS,
    SQRT_2_BITS, THREE_HALVES_BITS, TWO_BITS, ZERO_BITS,
};
use crate::ir::{DataType, Expr};

/// An f32 constant, stated as the exact bits it must be.
fn constant(bits: u32) -> Expr {
    Expr::bitcast_u32_to_f32(Expr::u32(bits))
}

/// `coefficients[0] + v*(coefficients[1] + v*(...))`, ascending order.
fn horner(value: &Expr, coefficients: &[u32]) -> Expr {
    let mut descending = coefficients.iter().rev();
    let mut accumulated = match descending.next() {
        Some(bits) => constant(*bits),
        None => return constant(ZERO_BITS),
    };
    for bits in descending {
        accumulated = Expr::add(constant(*bits), Expr::mul(value.clone(), accumulated));
    }
    accumulated
}

/// The argument with subnormals flushed to a zero of their own sign, paired
/// with the absolute bit pattern of the flushed value.
///
/// Both sides of the parity claim flush a subnormal f32: the reference through
/// [`canonical_f32`](crate::fp_parity::canonical_f32) after every arithmetic
/// operation, and a device because its f32 pipeline does. An expansion that
/// read the exponent field of a subnormal would therefore compute from a value
/// the surrounding arithmetic cannot represent. The flush is integer work on
/// the bit pattern, so it is exact on both sides.
///
/// The absolute pattern is the domain test every expansion uses: `0` is a zero,
/// below [`MIN_NORMAL_BITS`] is impossible after the flush,
/// [`INFINITY_BITS`] is an infinity, and anything above it is a NaN.
fn flushed_argument(argument: &Expr) -> (Expr, Expr) {
    let raw = Expr::bitcast_f32_to_u32(argument.clone());
    let magnitude = Expr::bitand(
        Expr::bitcast_f32_to_u32(argument.clone()),
        Expr::u32(ABS_MASK),
    );
    let flushed = Expr::select(
        Expr::lt(magnitude, Expr::u32(MIN_NORMAL_BITS)),
        Expr::bitand(
            Expr::bitcast_f32_to_u32(argument.clone()),
            Expr::u32(SIGN_MASK),
        ),
        raw,
    );
    (
        Expr::bitcast_u32_to_f32(flushed.clone()),
        Expr::bitand(flushed, Expr::u32(ABS_MASK)),
    )
}

/// `round_to_nearest_integer(value * scale)` as an f32, without a round
/// instruction: adding and subtracting `1.5 * 2^23` discards every fractional
/// bit under the ambient rounding mode.
fn scaled_round(value: &Expr, scale_bits: u32) -> Expr {
    Expr::sub(
        Expr::add(
            Expr::mul(value.clone(), constant(scale_bits)),
            constant(ROUND_SHIFT_BITS),
        ),
        constant(ROUND_SHIFT_BITS),
    )
}

/// `e^x`.
///
/// `x = k*ln(2) + r` with `|r| <= ln(2)/2`, evaluated as `exp(r) * 2^k`. The
/// two-limb `ln(2)` makes the reduction exact, and `2^k` is assembled directly
/// in the exponent field, so the only rounding is the polynomial's.
pub(super) fn exponential(argument: &Expr) -> Expr {
    let (value, magnitude) = flushed_argument(argument);
    let clamped = Expr::max(
        Expr::min(value.clone(), constant(EXP_CLAMP_HI_BITS)),
        constant(EXP_CLAMP_LO_BITS),
    );
    let quotient = scaled_round(&clamped, LOG2_E_BITS);
    let reduced = Expr::sub(
        Expr::sub(clamped, Expr::mul(quotient.clone(), constant(LN2_HI_BITS))),
        Expr::mul(quotient.clone(), constant(LN2_LO_BITS)),
    );
    let kernel = horner(&reduced, &EXP_COEFFICIENTS);

    // The exponent field the result wants. A field of 255 denotes infinity, so
    // the scale is built one power of two low and the product doubled, which
    // keeps every representable result reachable without ever forming an
    // infinity mid-expression.
    let exponent_field = Expr::cast(
        DataType::U32,
        Expr::add(quotient, constant(EXPONENT_BIAS_F32_BITS)),
    );
    let scale = Expr::bitcast_u32_to_f32(Expr::shl(
        Expr::min(exponent_field.clone(), Expr::u32(EXP_FINITE_EXPONENT_FIELD)),
        Expr::u32(EXPONENT_SHIFT),
    ));
    let scaled = Expr::mul(kernel, scale);
    let result = Expr::select(
        Expr::eq(exponent_field, Expr::u32(EXP_SATURATED_EXPONENT_FIELD)),
        Expr::mul(scaled.clone(), constant(TWO_BITS)),
        scaled,
    );

    let result = Expr::select(
        Expr::gt(value.clone(), constant(EXP_OVERFLOW_ABOVE_BITS)),
        constant(INFINITY_BITS),
        result,
    );
    let result = Expr::select(
        Expr::lt(value, constant(EXP_UNDERFLOW_BELOW_BITS)),
        constant(ZERO_BITS),
        result,
    );
    Expr::select(
        Expr::gt(magnitude, Expr::u32(INFINITY_BITS)),
        constant(NAN_RESULT_BITS),
        result,
    )
}

/// `ln(x)`.
///
/// `x = m * 2^e` with `m` in `[sqrt(2)/2, sqrt(2))`, so `ln(x) = ln(m) + e*ln(2)`.
/// The mantissa is extracted with integer masks and the exponent contribution is
/// exact: [`LN2_HI_BITS`] has eight zero low mantissa bits, so `e * LN2_HI` is
/// exact for every exponent an f32 can hold.
pub(super) fn logarithm(argument: &Expr) -> Expr {
    let (value, magnitude) = flushed_argument(argument);
    let raw = Expr::bitcast_f32_to_u32(value.clone());
    let mantissa_bits = Expr::bitor(
        Expr::bitand(raw.clone(), Expr::u32(MANTISSA_MASK)),
        Expr::u32(ONE_BITS),
    );
    let mantissa = Expr::bitcast_u32_to_f32(mantissa_bits.clone());

    // A mantissa above sqrt(2) is halved and the exponent raised, which centres
    // the polynomial's argument on zero and halves its range.
    let halve = Expr::gt(mantissa.clone(), constant(SQRT_2_BITS));
    let mantissa = Expr::select(
        halve.clone(),
        Expr::bitcast_u32_to_f32(Expr::sub(mantissa_bits, Expr::u32(MIN_NORMAL_BITS))),
        mantissa,
    );
    let exponent_field = Expr::add(
        Expr::shr(raw, Expr::u32(EXPONENT_SHIFT)),
        Expr::select(halve, Expr::u32(1), Expr::u32(0)),
    );
    let exponent = Expr::sub(
        Expr::cast(DataType::F32, exponent_field),
        constant(EXPONENT_BIAS_F32_BITS),
    );

    let offset = Expr::sub(mantissa, constant(ONE_BITS));
    let kernel = Expr::add(
        offset.clone(),
        Expr::mul(
            Expr::mul(offset.clone(), offset.clone()),
            horner(&offset, &LOG_COEFFICIENTS),
        ),
    );
    let result = Expr::add(
        Expr::add(kernel, Expr::mul(exponent.clone(), constant(LN2_LO_BITS))),
        Expr::mul(exponent, constant(LN2_HI_BITS)),
    );

    let result = Expr::select(
        Expr::eq(value.clone(), constant(ZERO_BITS)),
        constant(NEG_INFINITY_BITS),
        result,
    );
    let result = Expr::select(
        Expr::eq(magnitude.clone(), Expr::u32(INFINITY_BITS)),
        constant(INFINITY_BITS),
        result,
    );
    Expr::select(
        Expr::or(
            Expr::lt(value, constant(ZERO_BITS)),
            Expr::gt(magnitude, Expr::u32(INFINITY_BITS)),
        ),
        constant(NAN_RESULT_BITS),
        result,
    )
}

/// `sqrt(x)`.
///
/// The exponent is reduced to an even value so the iteration only ever sees a
/// mantissa in `[1, 4)`, which removes every overflow and subnormal hazard and
/// is why an exhaustive sweep of `2^24` inputs covers the whole function. Three
/// Newton refinements of a reciprocal-square-root seed and one residual step
/// reach the result without a division.
pub(super) fn square_root(argument: &Expr) -> Expr {
    let (value, magnitude) = flushed_argument(argument);
    let raw = Expr::bitcast_f32_to_u32(value.clone());
    let exponent_field = Expr::shr(raw.clone(), Expr::u32(EXPONENT_SHIFT));
    let odd = Expr::bitand(
        Expr::add(exponent_field.clone(), Expr::u32(1)),
        Expr::u32(1),
    );
    let mantissa_bits = Expr::bitor(
        Expr::bitand(raw, Expr::u32(MANTISSA_MASK)),
        Expr::shl(
            Expr::add(Expr::u32(EXPONENT_BIAS), odd.clone()),
            Expr::u32(EXPONENT_SHIFT),
        ),
    );
    let half_exponent_field = Expr::shr(
        Expr::sub(Expr::add(exponent_field, Expr::u32(EXPONENT_BIAS)), odd),
        Expr::u32(1),
    );
    let mantissa = Expr::bitcast_u32_to_f32(mantissa_bits.clone());

    let mut reciprocal = Expr::bitcast_u32_to_f32(Expr::sub(
        Expr::u32(RSQRT_SEED_MAGIC),
        Expr::shr(mantissa_bits, Expr::u32(1)),
    ));
    let half_mantissa = Expr::mul(constant(HALF_BITS), mantissa.clone());
    for _ in 0..RSQRT_NEWTON_STEPS {
        // Grouped so the small factor multiplies the halved mantissa first: the
        // alternative grouping forms `y*y` alone, which is subnormal for a
        // mantissa near 4.
        reciprocal = Expr::mul(
            reciprocal.clone(),
            Expr::sub(
                constant(THREE_HALVES_BITS),
                Expr::mul(
                    Expr::mul(half_mantissa.clone(), reciprocal.clone()),
                    reciprocal,
                ),
            ),
        );
    }
    let root = Expr::mul(mantissa.clone(), reciprocal.clone());
    let root = Expr::add(
        root.clone(),
        Expr::mul(
            Expr::sub(mantissa, Expr::mul(root.clone(), root)),
            Expr::mul(constant(HALF_BITS), reciprocal),
        ),
    );
    let result = Expr::mul(
        root,
        Expr::bitcast_u32_to_f32(Expr::shl(half_exponent_field, Expr::u32(EXPONENT_SHIFT))),
    );

    let result = Expr::select(
        Expr::eq(magnitude.clone(), Expr::u32(0)),
        value.clone(),
        result,
    );
    let result = Expr::select(
        Expr::eq(magnitude.clone(), Expr::u32(INFINITY_BITS)),
        constant(INFINITY_BITS),
        result,
    );
    Expr::select(
        Expr::or(
            Expr::lt(value, constant(ZERO_BITS)),
            Expr::gt(magnitude, Expr::u32(INFINITY_BITS)),
        ),
        constant(NAN_RESULT_BITS),
        result,
    )
}

/// The shared circular argument reduction: `x = k*(pi/2) + r`.
struct Circular {
    /// `r`, in `[-pi/4, pi/4]`.
    reduced: Expr,
    /// `k mod 4`, the quadrant.
    quadrant: Expr,
    /// The argument with subnormals flushed.
    value: Expr,
    /// Absolute bit pattern of [`Self::value`].
    magnitude: Expr,
}

fn circular_reduction(argument: &Expr) -> Circular {
    let (value, magnitude) = flushed_argument(argument);

    // Clamping the quotient keeps `r` finite and the float-to-integer cast in
    // range for every f32 argument. Inside the stated domain the clamp never
    // fires; outside it the result is replaced by a NaN.
    let quotient = Expr::max(
        Expr::min(
            scaled_round(&value, INV_PIO2_BITS),
            constant(SIN_COS_QUOTIENT_LIMIT_BITS),
        ),
        constant(SIN_COS_QUOTIENT_LIMIT_NEG_BITS),
    );
    let reduced = Expr::sub(
        Expr::sub(
            Expr::sub(
                value.clone(),
                Expr::mul(quotient.clone(), constant(PIO2_1_BITS)),
            ),
            Expr::mul(quotient.clone(), constant(PIO2_2_BITS)),
        ),
        Expr::mul(quotient.clone(), constant(PIO2_3_BITS)),
    );
    let quadrant = Expr::bitand(
        Expr::cast(
            DataType::U32,
            Expr::add(quotient, constant(SIN_COS_QUADRANT_BIAS_BITS)),
        ),
        Expr::u32(3),
    );
    Circular {
        reduced,
        quadrant,
        value,
        magnitude,
    }
}

/// True when the quadrant is odd, which is when the two kernels exchange roles.
fn quadrant_is_odd(quadrant: &Expr) -> Expr {
    Expr::eq(Expr::bitand(quadrant.clone(), Expr::u32(1)), Expr::u32(1))
}

/// `-1.0` when bit 1 of `quadrant + rotation` is set, `1.0` otherwise.
///
/// Multiplying by an exact `+/-1.0` is bit-exact for every f32, including a
/// zero, an infinity and a NaN, so the sign of the quadrant costs one multiply
/// and no accuracy.
fn quadrant_sign(quadrant: &Expr, rotation: u32) -> Expr {
    Expr::select(
        Expr::eq(
            Expr::bitand(
                Expr::add(quadrant.clone(), Expr::u32(rotation)),
                Expr::u32(2),
            ),
            Expr::u32(2),
        ),
        constant(NEG_ONE_BITS),
        constant(ONE_BITS),
    )
}

/// `sin(r)` for `|r| <= pi/4`.
fn sine_kernel(reduced: &Expr, square: &Expr) -> Expr {
    Expr::add(
        reduced.clone(),
        Expr::mul(
            Expr::mul(reduced.clone(), square.clone()),
            horner(square, &SIN_COEFFICIENTS),
        ),
    )
}

/// `cos(r)` for `|r| <= pi/4`.
fn cosine_kernel(square: &Expr) -> Expr {
    Expr::add(
        constant(ONE_BITS),
        Expr::mul(square.clone(), horner(square, &COS_COEFFICIENTS)),
    )
}

/// `sin(x)` for `|x| <= 401`, and [`NAN_RESULT_BITS`] outside that domain.
pub(super) fn sine(argument: &Expr) -> Expr {
    let circular = circular_reduction(argument);
    let square = Expr::mul(circular.reduced.clone(), circular.reduced.clone());
    let result = Expr::mul(
        quadrant_sign(&circular.quadrant, 0),
        Expr::select(
            quadrant_is_odd(&circular.quadrant),
            cosine_kernel(&square),
            sine_kernel(&circular.reduced, &square),
        ),
    );
    // Below 2^-12 the reduction cannot improve on the argument itself, and
    // returning it keeps a small magnitude exact.
    let result = Expr::select(
        Expr::lt(
            circular.magnitude.clone(),
            Expr::u32(SIN_COS_SMALL_ARGUMENT_BITS),
        ),
        circular.value,
        result,
    );
    Expr::select(
        Expr::gt(circular.magnitude, Expr::u32(SIN_COS_DOMAIN_BITS)),
        constant(NAN_RESULT_BITS),
        result,
    )
}

/// `cos(x)` for `|x| <= 401`, and [`NAN_RESULT_BITS`] outside that domain.
pub(super) fn cosine(argument: &Expr) -> Expr {
    let circular = circular_reduction(argument);
    let square = Expr::mul(circular.reduced.clone(), circular.reduced.clone());
    let result = Expr::mul(
        quadrant_sign(&circular.quadrant, 1),
        Expr::select(
            quadrant_is_odd(&circular.quadrant),
            sine_kernel(&circular.reduced, &square),
            cosine_kernel(&square),
        ),
    );
    let result = Expr::select(
        Expr::lt(
            circular.magnitude.clone(),
            Expr::u32(SIN_COS_SMALL_ARGUMENT_BITS),
        ),
        constant(ONE_BITS),
        result,
    );
    Expr::select(
        Expr::gt(circular.magnitude, Expr::u32(SIN_COS_DOMAIN_BITS)),
        constant(NAN_RESULT_BITS),
        result,
    )
}
