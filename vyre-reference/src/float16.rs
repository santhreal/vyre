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
            let unbiased = -14_i32 - i32::try_from(shift).expect("bounded half shift");
            sign | (u32::try_from(unbiased + 127).expect("half exponent fits") << 23)
                | (normalized_fraction << 13)
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
    let exponent = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0xff {
        return if fraction == 0 { sign | 0x7c00 } else { 0x7e00 };
    }
    if exponent == 0 {
        return sign;
    }
    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        let mut half_exponent = u16::try_from(unbiased + 15).expect("normal half exponent fits");
        let rounded_fraction = round_shift_right(fraction, 13);
        if rounded_fraction == 0x400 {
            half_exponent += 1;
            if half_exponent >= 0x1f {
                return sign | 0x7c00;
            }
            return sign | (half_exponent << 10);
        }
        return sign | (half_exponent << 10) | rounded_fraction as u16;
    }
    if unbiased < -25 {
        return sign;
    }
    let significand = 0x0080_0000 | fraction;
    let shift = u32::try_from(-unbiased - 1).expect("subnormal half shift fits");
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
        assert_eq!(f32_to_f16(1.000_488_281_25), 0x3c00);
        assert_eq!(f32_to_f16(1.001_464_843_75), 0x3c02);

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
}
