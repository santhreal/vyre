//! Every numeric constant the strict f32 expansions use, as an f32 bit pattern.
//!
//! A decimal literal in Rust source is rounded by the Rust parser and a decimal
//! literal in emitted shader text is rounded again by the shader compiler. Two
//! roundings of the same digits are not required to agree, and a coefficient
//! that differs in its last bit moves the polynomial's error. Every value below
//! is therefore stated as the exact 32 bits it must be, materialised through
//! `UnOp::BitcastU32ToF32`, so an f32 literal never appears in an expanded
//! program at all.
//!
//! The polynomial coefficients are minimax fits validated against the
//! correctly-rounded f32 result: measured worst case is 1 ulp for `exp`, `log`
//! and `sqrt` and 2 ulp for `sin` and `cos`, inside
//! [`REFERENCE_TRANSCENDENTAL_ULP_BUDGET`](crate::fp_parity::REFERENCE_TRANSCENDENTAL_ULP_BUDGET).

/// The one quiet NaN every expansion produces for an out-of-domain argument.
///
/// IEEE-754 leaves the payload of a generated NaN unspecified, so a device that
/// picks a different payload than the reference would break bit identity on an
/// input neither side considers exceptional. The expansions never generate a
/// NaN arithmetically: an out-of-domain argument selects these bits.
pub const NAN_RESULT_BITS: u32 = 0x7FC0_0000;

/// Sign bit of an f32.
pub(super) const SIGN_MASK: u32 = 0x8000_0000;

/// Every bit of an f32 except the sign.
pub(super) const ABS_MASK: u32 = 0x7FFF_FFFF;

/// Bit pattern of the smallest positive normal f32, and the threshold a
/// magnitude must reach to survive the subnormal flush.
pub const MIN_NORMAL_BITS: u32 = 0x0080_0000;

/// Bit pattern of positive infinity, and the largest absolute pattern that is
/// still a number.
pub(super) const INFINITY_BITS: u32 = 0x7F80_0000;

/// Bit pattern of negative infinity.
pub(super) const NEG_INFINITY_BITS: u32 = 0xFF80_0000;

/// The 23 mantissa bits of an f32.
pub(super) const MANTISSA_MASK: u32 = 0x007F_FFFF;

/// Distance from the low bit of an f32 to the low bit of its exponent field.
pub(super) const EXPONENT_SHIFT: u32 = 23;

/// Exponent-field bias of an f32.
pub(super) const EXPONENT_BIAS: u32 = 127;

/// `127.0`, the exponent bias as an f32.
pub(super) const EXPONENT_BIAS_F32_BITS: u32 = 0x42FE_0000;

/// `+0.0`.
pub(super) const ZERO_BITS: u32 = 0x0000_0000;

/// `0.5`.
pub(super) const HALF_BITS: u32 = 0x3F00_0000;

/// `1.0`.
pub(super) const ONE_BITS: u32 = 0x3F80_0000;

/// `1.5`.
pub(super) const THREE_HALVES_BITS: u32 = 0x3FC0_0000;

/// `2.0`.
pub(super) const TWO_BITS: u32 = 0x4000_0000;

/// `-1.0`.
pub(super) const NEG_ONE_BITS: u32 = 0xBF80_0000;

/// `1.5 * 2^23`. Adding then subtracting this rounds an f32 to the nearest
/// integer under the current rounding mode without a round instruction.
pub(super) const ROUND_SHIFT_BITS: u32 = 0x4B40_0000;

/// `log2(e)`, for the `exp` argument reduction.
pub(super) const LOG2_E_BITS: u32 = 0x3FB8_AA3B;

/// High limb of `ln(2)`. The low 8 mantissa bits are zero, so a product with an
/// integer of magnitude below 2^8 is exact.
pub(super) const LN2_HI_BITS: u32 = 0x3F31_7200;

/// Low limb of `ln(2)`, carrying the bits `LN2_HI_BITS` drops.
pub(super) const LN2_LO_BITS: u32 = 0x35BF_BE8E;

/// `sqrt(2)`, the mantissa split point for `log`.
pub(super) const SQRT_2_BITS: u32 = 0x3FB5_04F3;

/// `2/pi`, for the circular argument reduction.
pub(super) const INV_PIO2_BITS: u32 = 0x3F22_F983;

/// High limb of `pi/2`. The low 8 mantissa bits are zero, so a product with an
/// integer of magnitude at most 255 is exact and the leading subtraction of the
/// reduction is exact by Sterbenz.
pub(super) const PIO2_1_BITS: u32 = 0x3FC9_0F00;

/// Middle limb of `pi/2`.
pub(super) const PIO2_2_BITS: u32 = 0x37DA_A200;

/// Low limb of `pi/2`.
pub(super) const PIO2_3_BITS: u32 = 0x2E85_A300;

/// Ascending coefficients of the `exp` kernel on `|r| <= ln(2)/2`.
pub(super) const EXP_COEFFICIENTS: [u32; 7] = [
    0x3F80_0000,
    0x3F80_0000,
    0x3EFF_FFFE,
    0x3E2A_AA21,
    0x3D2A_AC4E,
    0x3C09_28D3,
    0x3AB5_3455,
];

/// Ascending coefficients of the `log` kernel in `t = m - 1` on
/// `m` in `[sqrt(2)/2, sqrt(2))`.
pub(super) const LOG_COEFFICIENTS: [u32; 10] = [
    0xBF00_0000,
    0x3EAA_AAA8,
    0xBE80_0001,
    0x3E4C_CFDA,
    0xBE2A_AD7B,
    0x3E11_D0BC,
    0xBDFE_6E87,
    0x3DF0_A754,
    0xBDEB_9E60,
    0x3D8C_4EC6,
];

/// Ascending coefficients of the `sin` kernel in `u = r*r` on `|r| <= pi/4`.
pub(super) const SIN_COEFFICIENTS: [u32; 4] = [0xBE2A_AAAB, 0x3C08_8887, 0xB950_09BD, 0x3636_DF0B];

/// Ascending coefficients of the `cos` kernel in `u = r*r` on `|r| <= pi/4`.
pub(super) const COS_COEFFICIENTS: [u32; 5] = [
    0xBF00_0000,
    0x3D2A_AAAB,
    0xBAB6_0B5D,
    0x37D0_093E,
    0xB492_3A78,
];

/// `88.72284`. `exp` of any larger argument overflows f32.
pub const EXP_OVERFLOW_ABOVE_BITS: u32 = 0x42B1_7218;

/// `-87.33654`. `exp` of any smaller argument is below the smallest positive
/// normal f32.
///
/// The result flushes to `+0.0` there rather than producing a subnormal. A
/// subnormal result cannot be bit-identical across a device that flushes
/// subnormals to zero and a reference that does not, and flushing is the same
/// decision [`canonical_f32`](crate::fp_parity::canonical_f32) already applies
/// to every f32 the parity contract compares.
pub const EXP_UNDERFLOW_BELOW_BITS: u32 = 0xC2AE_AC4F;

/// `88.75`. Clamping the argument keeps `k` inside the exponent field so the
/// float-to-integer cast is in range; the overflow select replaces the result.
pub(super) const EXP_CLAMP_HI_BITS: u32 = 0x42B1_8000;

/// `-87.5`, the low counterpart of [`EXP_CLAMP_HI_BITS`].
pub(super) const EXP_CLAMP_LO_BITS: u32 = 0xC2AF_0000;

/// Largest exponent field that denotes a finite f32.
pub(super) const EXP_FINITE_EXPONENT_FIELD: u32 = 254;

/// The exponent field one past finite, reached only by an argument whose result
/// is the largest representable magnitudes; the scale is halved and the product
/// doubled to stay in range.
pub(super) const EXP_SATURATED_EXPONENT_FIELD: u32 = 255;

/// `2^-12`. Below this magnitude `sin(x)` rounds to `x` and `cos(x)` to `1.0`,
/// which also keeps a small argument exact instead of routing it through a
/// reduction that cannot improve on it.
pub const SIN_COS_SMALL_ARGUMENT_BITS: u32 = 0x3980_0000;

/// `401.0`, the stated domain of the strict circular expansions.
///
/// `401.0 * 2/pi` rounds to 255, and 255 is the largest integer whose product
/// with `PIO2_1_BITS` is exact. An f32-only argument reduction cannot reach
/// further: extending it needs the multi-word integer arithmetic of a
/// Payne-Hanek reduction. Beyond this magnitude the expansions produce
/// [`NAN_RESULT_BITS`] rather than the silently clamped value of a saturated
/// reduction.
pub const SIN_COS_DOMAIN_BITS: u32 = 0x43C8_8000;

/// `255.0`, the reduction quotient bound implied by [`SIN_COS_DOMAIN_BITS`].
pub(super) const SIN_COS_QUOTIENT_LIMIT_BITS: u32 = 0x437F_0000;

/// `-255.0`, the low counterpart of [`SIN_COS_QUOTIENT_LIMIT_BITS`].
pub(super) const SIN_COS_QUOTIENT_LIMIT_NEG_BITS: u32 = 0xC37F_0000;

/// `256.0`. Added before the float-to-integer cast so the quadrant index is
/// non-negative; a multiple of four leaves the low two bits unchanged.
pub(super) const SIN_COS_QUADRANT_BIAS_BITS: u32 = 0x4380_0000;

/// The reciprocal-square-root seed magic. Subtracting a halved bit pattern from
/// it approximates `1/sqrt(m)` to about 3.4% over the reduced range.
pub(super) const RSQRT_SEED_MAGIC: u32 = 0x5F37_59DF;

/// Newton-Raphson refinements applied to the reciprocal-square-root seed.
///
/// Three take the relative error from 3.4% to below 2^-22, which the final
/// residual step then carries to the 1 ulp measured over the exhaustive sweep
/// of the reduced range.
pub(super) const RSQRT_NEWTON_STEPS: usize = 3;
