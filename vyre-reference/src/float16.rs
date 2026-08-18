//! Canonical IEEE binary16 and bfloat16 conversion helpers.

use crate::execution::typed_ops::canonical_f32;

pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    canonical_f32(f32::from_bits(u32::from(bits) << 16))
}

pub(crate) fn f32_to_bf16(value: f32) -> u16 {
    let bits = canonical_f32(value).to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0 {
        return 0x7fc0;
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let result = match (exponent, fraction) {
        (0, 0) => sign,
        (0, _) => {
            let leading = 31 - fraction.leading_zeros();
            let shift = 10 - leading;
            let normalized_fraction = (fraction << shift) & 0x03ff;
            // A nonzero 10-bit fraction shifts by 1..=10, so the binary32 bias
            // of the subnormal exponent (-14 - shift) stays inside 103..=112 and
            // the whole expression is unsigned.
            let biased_exponent = 127 - 14 - shift;
            sign | (biased_exponent << 23) | (normalized_fraction << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, _) => 0x7fc0_0000,
        _ => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    canonical_f32(f32::from_bits(result))
}

pub(crate) fn f32_to_f16(value: f32) -> u16 {
    let bits = canonical_f32(value).to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0xff {
        return if fraction == 0 { sign | 0x7c00 } else { 0x7e00 };
    }
    if exponent == 0 {
        return sign;
    }
    // Every bound is a comparison on the stored binary32 exponent, so the whole
    // conversion stays unsigned: 142 is unbiased +15, 113 is unbiased -14, and
    // 102 is unbiased -25. Subtracting 112 rebiases to binary16.
    if exponent > 142 {
        return sign | 0x7c00;
    }
    if exponent >= 113 {
        let mut half_exponent = exponent - 112;
        let rounded_fraction = round_shift_right(fraction, 13);
        if rounded_fraction == 0x400 {
            half_exponent += 1;
            if half_exponent >= 0x1f {
                return sign | 0x7c00;
            }
            return sign | (half_exponent << 10) as u16;
        }
        return sign | ((half_exponent << 10) | rounded_fraction) as u16;
    }
    if exponent < 102 {
        return sign;
    }
    let significand = 0x0080_0000 | fraction;
    let shift = 126 - exponent;
    let rounded = round_shift_right(significand, shift);
    if rounded >= 0x400 {
        sign | 0x0400
    } else {
        sign | rounded as u16
    }
}

fn round_shift_right(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    let quotient = value >> shift;
    let mask = (1_u32 << shift) - 1;
    let remainder = value & mask;
    let halfway = 1_u32 << (shift - 1);
    quotient + u32::from(remainder > halfway || (remainder == halfway && quotient & 1 == 1))
}

// Inline: covers the crate-private `bf16_to_f32` and `f16_to_f32`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;

    /// Locks canonical edge values and round-to-nearest-even ties for both 16-bit formats.
    #[test]
    fn canonical_half_conversion_edges() {
        for (bits, value) in [
            (0x0000, 0.0),
            (0x8000, -0.0),
            (0x3c00, 1.0),
            (0xc000, -2.0),
            (0x7c00, f32::INFINITY),
            (0xfc00, f32::NEG_INFINITY),
            (0x0001, 5.960_464_5e-8),
            (0x03ff, 6.097_555e-5),
        ] {
            assert_eq!(f16_to_f32(bits).to_bits(), value.to_bits());
            assert_eq!(f32_to_f16(value), bits);
        }
        assert!(f16_to_f32(0x7e01).is_nan());
        assert_eq!(f32_to_f16(f32::NAN), 0x7e00);
        assert_eq!(f32_to_f16(1.000_488_3), 0x3c00);
        assert_eq!(f32_to_f16(1.001_464_8), 0x3c02);

        for (bits, value) in [
            (0x0000, 0.0),
            (0x8000, -0.0),
            (0x3f80, 1.0),
            (0xc000, -2.0),
            (0x7f80, f32::INFINITY),
            (0xff80, f32::NEG_INFINITY),
        ] {
            assert_eq!(bf16_to_f32(bits).to_bits(), value.to_bits());
            assert_eq!(f32_to_bf16(value), bits);
        }
        assert!(bf16_to_f32(0x7fc1).is_nan());
        assert_eq!(f32_to_bf16(f32::NAN), 0x7fc0);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
        assert_eq!(f32_to_bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    }

    /// WHY: both directions of the binary16 path do their own exponent
    /// arithmetic on a biased field, and an off-by-one in a bound reaches only a
    /// handful of encodings. The domain is 65536 values, so cover all of them
    /// rather than the few edges a table lists: every non-NaN half must survive
    /// a decode followed by an encode, and every NaN must decode to a NaN.
    #[test]
    fn every_half_encoding_survives_a_round_trip() {
        for bits in 0..=u16::MAX {
            let value = f16_to_f32(bits);
            let is_nan_encoding = bits & 0x7c00 == 0x7c00 && bits & 0x03ff != 0;
            if is_nan_encoding {
                assert!(value.is_nan(), "half {bits:#06x} decoded to {value}");
                assert_eq!(f32_to_f16(value), 0x7e00);
                continue;
            }
            assert_eq!(
                f32_to_f16(value),
                bits,
                "half {bits:#06x} decoded to {value} and did not re-encode"
            );
        }
    }
}
