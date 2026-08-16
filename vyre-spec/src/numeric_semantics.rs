//! Versioned numeric semantics table and conversion algorithms for all scalar and quantized types.
//!
//! This module is the specification authority for encoding, rounding, overflow,
//! saturation, signed-zero, subnormal, infinity, and NaN behavior across every
//! canonical and quantized datatype in vyre IR.

use crate::data_type::DataType;

/// Version of the numeric semantics specification.
pub const NUMERIC_SEMANTICS_SCHEMA_VERSION: u32 = 1;

/// Classification of the underlying numeric representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub enum NumericFormat {
    /// Unsigned integer format with standard two's complement binary encoding.
    UnsignedInteger,
    /// Signed integer format with two's complement binary encoding.
    SignedInteger,
    /// IEEE 754 standard binary floating-point representation.
    IeeeFloat,
    /// IEEE-like floating-point representation with non-standard exponent/mantissa bias.
    IeeeLikeFloat,
    /// Quantized Normal-Float (NF4) quantile representation for normal distributions.
    NormalFloat,
    /// Boolean truth value.
    Boolean,
    /// Compound or structured tensor/sparse/handle layout.
    Structured,
}

/// Rounding mode applied on conversion or arithmetic operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (IEEE 754 default).
    RoundToNearestEven,
    /// Truncation towards zero.
    TruncateTowardsZero,
    /// Nearest quantile index in a discrete codebook.
    NearestQuantile,
    /// Exact representation without rounding.
    Exact,
}

/// Behavior when a numeric value exceeds the representable range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub enum OverflowBehavior {
    /// Modular wrapping in two's complement.
    WrapTwoComplement,
    /// Saturation to the minimum / maximum representable finite value.
    SaturateToFiniteRange,
    /// Overflow produces signed IEEE infinity.
    SignedInfinity,
    /// Overflow produces canonical NaN (e.g. F8E4M3 max exceeded under certain modes).
    NanOrSaturate,
    /// Not applicable / structured.
    NotApplicable,
}

/// Saturation policy for numeric conversions and arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub enum SaturationBehavior {
    /// No saturation; wraps or errors.
    None,
    /// Clamps input to the representable finite range [min, max].
    ClampsToFiniteRange,
    /// Clamps to normalized unit interval [-1.0, 1.0].
    ClampsToUnitInterval,
}

/// Handling of signed zero (+0.0 and -0.0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub enum SignedZeroBehavior {
    /// Distinct +0.0 and -0.0 preserved.
    Preserved,
    /// Subnormals flushed to signed zero preserving sign bit.
    PreservedWithCanonicalFlushing,
    /// Only a single unsigned zero exists.
    UnsignedZero,
    /// Not applicable.
    NotApplicable,
}

/// Handling of subnormal (denormal) floating-point values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub enum SubnormalBehavior {
    /// Subnormals are preserved according to IEEE 754 rules.
    PreservedIEEE,
    /// Subnormals are flushed to signed zero in canonical reference evaluation.
    FlushedToSignedZero,
    /// Subnormals are not supported by the encoding.
    Unsupported,
}

/// Handling of IEEE infinity values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub enum InfinityBehavior {
    /// Supports +Inf and -Inf representations.
    SignedInfinity,
    /// Format has no infinity; saturates to maximum finite value.
    SaturatedToMaxFinite,
    /// Unsupported.
    Unsupported,
}

/// Handling of NaN (Not a Number) values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Deserialize, serde::Serialize)]
pub enum NanBehavior {
    /// Canonical quiet NaN bit pattern produced on invalid operations.
    CanonicalQuietNan,
    /// Dedicated NaN bit pattern supported (e.g. 0x7F / 0xFF in F8E4M3).
    DedicatedNanBitPattern,
    /// NaN is not representable; maps to zero or maximum finite.
    UnsupportedOrZero,
    /// Not applicable.
    NotApplicable,
}

/// Full specification of numeric semantics for one datatype.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct NumericSemantics {
    /// The datatype this specification describes.
    pub datatype: DataType,
    /// Bit width per scalar element, if fixed.
    pub bit_width: Option<usize>,
    /// Classification of numeric representation.
    pub format: NumericFormat,
    /// Exponent bit count for floating formats.
    pub exponent_bits: Option<u32>,
    /// Mantissa / significand bit count for floating formats.
    pub mantissa_bits: Option<u32>,
    /// Exponent bias for floating formats.
    pub exponent_bias: Option<i32>,
    /// Rounding mode.
    pub rounding: RoundingMode,
    /// Overflow behavior.
    pub overflow: OverflowBehavior,
    /// Saturation behavior.
    pub saturation: SaturationBehavior,
    /// Signed zero behavior.
    pub signed_zero: SignedZeroBehavior,
    /// Subnormal behavior.
    pub subnormal: SubnormalBehavior,
    /// Infinity behavior.
    pub infinity: InfinityBehavior,
    /// NaN behavior.
    pub nan: NanBehavior,
    /// Minimum finite representable value as f64.
    pub min_finite: f64,
    /// Maximum finite representable value as f64.
    pub max_finite: f64,
}

/// Canonical 16-element NF4 quantile table for normal distribution quantization.
pub const NF4_QUANTILE_TABLE: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_06,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// Canonical 16-element FP4 (E2M1) decode table.
/// Format: 1 sign bit, 2 exponent bits (bias=1), 1 mantissa bit.
pub const FP4_DECODE_TABLE: [f32; 16] = [
    0.0,  // 0000: +0.0
    0.5,  // 0001: +0.5 (subnormal: 2^0 * 0.5)
    1.0,  // 0010: +1.0 (normal: 2^(1-1) * 1.0)
    1.5,  // 0011: +1.5 (normal: 2^(1-1) * 1.5)
    2.0,  // 0100: +2.0 (normal: 2^(2-1) * 1.0)
    3.0,  // 0101: +3.0 (normal: 2^(2-1) * 1.5)
    4.0,  // 0110: +4.0 (normal: 2^(3-1) * 1.0)
    6.0,  // 0111: +6.0 (normal: 2^(3-1) * 1.5)
    -0.0, // 1000: -0.0
    -0.5, // 1001: -0.5
    -1.0, // 1010: -1.0
    -1.5, // 1011: -1.5
    -2.0, // 1100: -2.0
    -3.0, // 1101: -3.0
    -4.0, // 1110: -4.0
    -6.0, // 1111: -6.0
];

/// Canonical 16-element I4 decode table.
/// Range: [-8, 7]
pub const I4_DECODE_TABLE: [i32; 16] = [
    0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
];

/// Decode one 8-bit F8E4M3 float value to f32.
/// E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa.
/// Max finite = 448.0 (0x7E / 0xFE).
/// NaN = 0x7F / 0xFF.
#[must_use]
pub fn f8e4m3_to_f32(byte: u8) -> f32 {
    let sign = (byte & 0x80) != 0;
    let exp = (byte >> 3) & 0x0F;
    let mant = byte & 0x07;

    let sign_mult = if sign { -1.0 } else { 1.0 };

    if exp == 0x0F && mant == 0x07 {
        return f32::NAN;
    }

    if exp == 0 {
        if mant == 0 {
            if sign { -0.0 } else { 0.0 }
        } else {
            sign_mult * (2.0f32.powi(-6)) * (mant as f32 / 8.0)
        }
    } else {
        let e = exp as i32 - 7;
        sign_mult * (2.0f32.powi(e)) * (1.0 + mant as f32 / 8.0)
    }
}

/// Encode one f32 value to 8-bit F8E4M3.
#[must_use]
pub fn f32_to_f8e4m3(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7F;
    }
    let sign_bit = if val.is_sign_negative() { 0x80u8 } else { 0x00u8 };
    let abs_val = val.abs();

    if abs_val == 0.0 {
        return sign_bit;
    }
    if abs_val > 448.0 {
        return sign_bit | 0x7E; // Saturate to max finite
    }

    // Smallest subnormal is 2^-6 * (1/8) = 2^-9 = 1/512 ~ 0.001953125
    if abs_val < (2.0f32.powi(-9) * 0.5) {
        return sign_bit;
    }

    let mut best_byte = 0u8;
    let mut best_diff = f32::INFINITY;

    for byte in 0..=0x7E_u8 {
        let decoded = f8e4m3_to_f32(byte);
        let diff = (decoded - abs_val).abs();
        if diff < best_diff {
            best_diff = diff;
            best_byte = byte;
        }
    }

    sign_bit | best_byte
}

/// Decode one 8-bit F8E5M2 float value to f32.
/// E5M2: 1 sign, 5 exponent (bias 15), 2 mantissa.
/// Max finite = 57344.0.
/// Inf = 0x7C / 0xFC.
/// NaN = 0x7D..0x7F / 0xFD..0xFF.
#[must_use]
pub fn f8e5m2_to_f32(byte: u8) -> f32 {
    let sign = (byte & 0x80) != 0;
    let exp = (byte >> 2) & 0x1F;
    let mant = byte & 0x03;

    let sign_mult = if sign { -1.0 } else { 1.0 };

    if exp == 0x1F {
        if mant == 0 {
            return if sign { f32::NEG_INFINITY } else { f32::INFINITY };
        }
        return f32::NAN;
    }

    if exp == 0 {
        if mant == 0 {
            if sign { -0.0 } else { 0.0 }
        } else {
            sign_mult * (2.0f32.powi(-14)) * (mant as f32 / 4.0)
        }
    } else {
        let e = exp as i32 - 15;
        sign_mult * (2.0f32.powi(e)) * (1.0 + mant as f32 / 4.0)
    }
}

/// Generate the 256-element F8E4M3 decode table.
#[must_use]
pub fn f8e4m3_decode_table() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        table[i] = f8e4m3_to_f32(i as u8);
        i += 1;
    }
    table
}

/// Generate the 256-element F8E5M2 decode table.
#[must_use]
pub fn f8e5m2_decode_table() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        table[i] = f8e5m2_to_f32(i as u8);
        i += 1;
    }
    table
}

/// Encode one f32 value to 8-bit F8E5M2.
#[must_use]
pub fn f32_to_f8e5m2(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7E;
    }
    let sign_bit = if val.is_sign_negative() { 0x80u8 } else { 0x00u8 };
    if val.is_infinite() {
        return sign_bit | 0x7C;
    }
    let abs_val = val.abs();

    if abs_val == 0.0 {
        return sign_bit;
    }
    if abs_val > 57344.0 {
        return sign_bit | 0x7C; // Infinity
    }

    let mut best_byte = 0u8;
    let mut best_diff = f32::INFINITY;

    for byte in 0..=0x7C_u8 {
        let decoded = f8e5m2_to_f32(byte);
        let diff = (decoded - abs_val).abs();
        if diff < best_diff {
            best_diff = diff;
            best_byte = byte;
        }
    }

    sign_bit | best_byte
}

/// Decode one 4-bit FP4 nibble (0..15) to f32.
#[must_use]
pub fn fp4_to_f32(nibble: u8) -> f32 {
    FP4_DECODE_TABLE[(nibble & 0x0F) as usize]
}

/// Encode one f32 value to 4-bit FP4 nibble (0..15).
#[must_use]
pub fn f32_to_fp4(val: f32) -> u8 {
    if val.is_nan() {
        return 0x07; // Saturated max finite
    }
    let mut best_nibble = 0u8;
    let mut best_diff = f32::INFINITY;

    for i in 0..16u8 {
        let decoded = FP4_DECODE_TABLE[i as usize];
        let diff = (decoded - val).abs();
        if diff < best_diff {
            best_diff = diff;
            best_nibble = i;
        }
    }

    best_nibble
}

/// Decode one 4-bit NF4 nibble (0..15) to normalized f32 in [-1.0, 1.0].
#[must_use]
pub fn nf4_to_f32(nibble: u8) -> f32 {
    NF4_QUANTILE_TABLE[(nibble & 0x0F) as usize]
}

/// Encode one f32 value to 4-bit NF4 nibble (0..15) via nearest quantile.
#[must_use]
pub fn f32_to_nf4(val: f32) -> u8 {
    let clamped = if val.is_nan() { 0.0 } else { val.clamp(-1.0, 1.0) };
    let mut best_idx = 0u8;
    let mut best_diff = f32::INFINITY;

    for i in 0..16u8 {
        let q = NF4_QUANTILE_TABLE[i as usize];
        let diff = (q - clamped).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = i;
        }
    }

    best_idx
}

/// Decode one 4-bit signed I4 nibble to i32 in [-8, 7].
#[must_use]
pub fn i4_to_i32(nibble: u8) -> i32 {
    let raw = nibble & 0x0F;
    if raw >= 8 {
        raw as i32 - 16
    } else {
        raw as i32
    }
}

/// Encode one i32 value to 4-bit signed I4 nibble (0..15), saturated to [-8, 7].
#[must_use]
pub fn i32_to_i4(val: i32) -> u8 {
    let clamped = val.clamp(-8, 7);
    (clamped as u8) & 0x0F
}

/// Dequantize a packed buffer of quantized values using per-group or per-tensor scales.
///
/// # Errors
/// Returns `Err` if buffer size is insufficient or group size is invalid.
pub fn dequantize_grouped_f32(
    storage_bytes: &[u8],
    storage_type: &DataType,
    scales: &[f32],
    zero_points: Option<&[f32]>,
    group_size: usize,
    element_count: usize,
) -> Result<Vec<f32>, String> {
    if group_size == 0 {
        return Err("group_size must be greater than zero".to_string());
    }

    let mut result = Vec::with_capacity(element_count);

    for i in 0..element_count {
        let group_idx = i / group_size;
        let scale = *scales.get(group_idx).unwrap_or(&1.0);
        let zp = zero_points.and_then(|z| z.get(group_idx).copied()).unwrap_or(0.0);

        let unscaled = match storage_type {
            DataType::I4 => {
                let byte_idx = i / 2;
                let is_high = (i % 2) != 0;
                let byte = *storage_bytes.get(byte_idx).ok_or("storage_bytes truncated")?;
                let nibble = if is_high { (byte >> 4) & 0x0F } else { byte & 0x0F };
                i4_to_i32(nibble) as f32
            }
            DataType::FP4 => {
                let byte_idx = i / 2;
                let is_high = (i % 2) != 0;
                let byte = *storage_bytes.get(byte_idx).ok_or("storage_bytes truncated")?;
                let nibble = if is_high { (byte >> 4) & 0x0F } else { byte & 0x0F };
                fp4_to_f32(nibble)
            }
            DataType::NF4 => {
                let byte_idx = i / 2;
                let is_high = (i % 2) != 0;
                let byte = *storage_bytes.get(byte_idx).ok_or("storage_bytes truncated")?;
                let nibble = if is_high { (byte >> 4) & 0x0F } else { byte & 0x0F };
                nf4_to_f32(nibble)
            }
            DataType::F8E4M3 => {
                let byte = *storage_bytes.get(i).ok_or("storage_bytes truncated")?;
                f8e4m3_to_f32(byte)
            }
            DataType::F8E5M2 => {
                let byte = *storage_bytes.get(i).ok_or("storage_bytes truncated")?;
                f8e5m2_to_f32(byte)
            }
            DataType::I8 => {
                let byte = *storage_bytes.get(i).ok_or("storage_bytes truncated")?;
                (byte as i8) as f32
            }
            DataType::U8 => {
                let byte = *storage_bytes.get(i).ok_or("storage_bytes truncated")?;
                byte as f32
            }
            _ => return Err(format!("unsupported storage datatype `{storage_type}`")),
        };

        result.push((unscaled - zp) * scale);
    }

    Ok(result)
}

/// Return the authoritative numeric semantics specification for `dtype`.
#[must_use]
pub fn numeric_semantics_for(dtype: &DataType) -> NumericSemantics {
    match dtype {
        DataType::U8 => NumericSemantics {
            datatype: DataType::U8,
            bit_width: Some(8),
            format: NumericFormat::UnsignedInteger,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::WrapTwoComplement,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::UnsignedZero,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::Unsupported,
            nan: NanBehavior::NotApplicable,
            min_finite: 0.0,
            max_finite: 255.0,
        },
        DataType::U16 => NumericSemantics {
            datatype: DataType::U16,
            bit_width: Some(16),
            format: NumericFormat::UnsignedInteger,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::WrapTwoComplement,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::UnsignedZero,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::Unsupported,
            nan: NanBehavior::NotApplicable,
            min_finite: 0.0,
            max_finite: 65535.0,
        },
        DataType::U32 => NumericSemantics {
            datatype: DataType::U32,
            bit_width: Some(32),
            format: NumericFormat::UnsignedInteger,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::WrapTwoComplement,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::UnsignedZero,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::Unsupported,
            nan: NanBehavior::NotApplicable,
            min_finite: 0.0,
            max_finite: u32::MAX as f64,
        },
        DataType::U64 => NumericSemantics {
            datatype: DataType::U64,
            bit_width: Some(64),
            format: NumericFormat::UnsignedInteger,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::WrapTwoComplement,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::UnsignedZero,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::Unsupported,
            nan: NanBehavior::NotApplicable,
            min_finite: 0.0,
            max_finite: u64::MAX as f64,
        },
        DataType::I8 => NumericSemantics {
            datatype: DataType::I8,
            bit_width: Some(8),
            format: NumericFormat::SignedInteger,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::WrapTwoComplement,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::UnsignedZero,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::Unsupported,
            nan: NanBehavior::NotApplicable,
            min_finite: -128.0,
            max_finite: 127.0,
        },
        DataType::I16 => NumericSemantics {
            datatype: DataType::I16,
            bit_width: Some(16),
            format: NumericFormat::SignedInteger,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::WrapTwoComplement,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::UnsignedZero,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::Unsupported,
            nan: NanBehavior::NotApplicable,
            min_finite: -32768.0,
            max_finite: 32767.0,
        },
        DataType::I32 => NumericSemantics {
            datatype: DataType::I32,
            bit_width: Some(32),
            format: NumericFormat::SignedInteger,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::WrapTwoComplement,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::UnsignedZero,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::Unsupported,
            nan: NanBehavior::NotApplicable,
            min_finite: i32::MIN as f64,
            max_finite: i32::MAX as f64,
        },
        DataType::I64 => NumericSemantics {
            datatype: DataType::I64,
            bit_width: Some(64),
            format: NumericFormat::SignedInteger,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::WrapTwoComplement,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::UnsignedZero,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::Unsupported,
            nan: NanBehavior::NotApplicable,
            min_finite: i64::MIN as f64,
            max_finite: i64::MAX as f64,
        },
        DataType::I4 => NumericSemantics {
            datatype: DataType::I4,
            bit_width: Some(4),
            format: NumericFormat::SignedInteger,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::WrapTwoComplement,
            saturation: SaturationBehavior::ClampsToFiniteRange,
            signed_zero: SignedZeroBehavior::UnsignedZero,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::Unsupported,
            nan: NanBehavior::NotApplicable,
            min_finite: -8.0,
            max_finite: 7.0,
        },
        DataType::FP4 => NumericSemantics {
            datatype: DataType::FP4,
            bit_width: Some(4),
            format: NumericFormat::IeeeLikeFloat,
            exponent_bits: Some(2),
            mantissa_bits: Some(1),
            exponent_bias: Some(1),
            rounding: RoundingMode::RoundToNearestEven,
            overflow: OverflowBehavior::SaturateToFiniteRange,
            saturation: SaturationBehavior::ClampsToFiniteRange,
            signed_zero: SignedZeroBehavior::Preserved,
            subnormal: SubnormalBehavior::PreservedIEEE,
            infinity: InfinityBehavior::SaturatedToMaxFinite,
            nan: NanBehavior::UnsupportedOrZero,
            min_finite: -6.0,
            max_finite: 6.0,
        },
        DataType::NF4 => NumericSemantics {
            datatype: DataType::NF4,
            bit_width: Some(4),
            format: NumericFormat::NormalFloat,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::NearestQuantile,
            overflow: OverflowBehavior::SaturateToFiniteRange,
            saturation: SaturationBehavior::ClampsToUnitInterval,
            signed_zero: SignedZeroBehavior::UnsignedZero,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::SaturatedToMaxFinite,
            nan: NanBehavior::UnsupportedOrZero,
            min_finite: -1.0,
            max_finite: 1.0,
        },
        DataType::F8E4M3 => NumericSemantics {
            datatype: DataType::F8E4M3,
            bit_width: Some(8),
            format: NumericFormat::IeeeLikeFloat,
            exponent_bits: Some(4),
            mantissa_bits: Some(3),
            exponent_bias: Some(7),
            rounding: RoundingMode::RoundToNearestEven,
            overflow: OverflowBehavior::NanOrSaturate,
            saturation: SaturationBehavior::ClampsToFiniteRange,
            signed_zero: SignedZeroBehavior::Preserved,
            subnormal: SubnormalBehavior::PreservedIEEE,
            infinity: InfinityBehavior::SaturatedToMaxFinite,
            nan: NanBehavior::DedicatedNanBitPattern,
            min_finite: -448.0,
            max_finite: 448.0,
        },
        DataType::F8E5M2 => NumericSemantics {
            datatype: DataType::F8E5M2,
            bit_width: Some(8),
            format: NumericFormat::IeeeLikeFloat,
            exponent_bits: Some(5),
            mantissa_bits: Some(2),
            exponent_bias: Some(15),
            rounding: RoundingMode::RoundToNearestEven,
            overflow: OverflowBehavior::SignedInfinity,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::Preserved,
            subnormal: SubnormalBehavior::PreservedIEEE,
            infinity: InfinityBehavior::SignedInfinity,
            nan: NanBehavior::CanonicalQuietNan,
            min_finite: -57344.0,
            max_finite: 57344.0,
        },
        DataType::F16 => NumericSemantics {
            datatype: DataType::F16,
            bit_width: Some(16),
            format: NumericFormat::IeeeFloat,
            exponent_bits: Some(5),
            mantissa_bits: Some(10),
            exponent_bias: Some(15),
            rounding: RoundingMode::RoundToNearestEven,
            overflow: OverflowBehavior::SignedInfinity,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::PreservedWithCanonicalFlushing,
            subnormal: SubnormalBehavior::FlushedToSignedZero,
            infinity: InfinityBehavior::SignedInfinity,
            nan: NanBehavior::CanonicalQuietNan,
            min_finite: -65504.0,
            max_finite: 65504.0,
        },
        DataType::BF16 => NumericSemantics {
            datatype: DataType::BF16,
            bit_width: Some(16),
            format: NumericFormat::IeeeFloat,
            exponent_bits: Some(8),
            mantissa_bits: Some(7),
            exponent_bias: Some(127),
            rounding: RoundingMode::RoundToNearestEven,
            overflow: OverflowBehavior::SignedInfinity,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::PreservedWithCanonicalFlushing,
            subnormal: SubnormalBehavior::FlushedToSignedZero,
            infinity: InfinityBehavior::SignedInfinity,
            nan: NanBehavior::CanonicalQuietNan,
            min_finite: -3.3895314e38,
            max_finite: 3.3895314e38,
        },
        DataType::F32 => NumericSemantics {
            datatype: DataType::F32,
            bit_width: Some(32),
            format: NumericFormat::IeeeFloat,
            exponent_bits: Some(8),
            mantissa_bits: Some(23),
            exponent_bias: Some(127),
            rounding: RoundingMode::RoundToNearestEven,
            overflow: OverflowBehavior::SignedInfinity,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::PreservedWithCanonicalFlushing,
            subnormal: SubnormalBehavior::FlushedToSignedZero,
            infinity: InfinityBehavior::SignedInfinity,
            nan: NanBehavior::CanonicalQuietNan,
            min_finite: -f32::MAX as f64,
            max_finite: f32::MAX as f64,
        },
        DataType::F64 => NumericSemantics {
            datatype: DataType::F64,
            bit_width: Some(64),
            format: NumericFormat::IeeeFloat,
            exponent_bits: Some(11),
            mantissa_bits: Some(52),
            exponent_bias: Some(1023),
            rounding: RoundingMode::RoundToNearestEven,
            overflow: OverflowBehavior::SignedInfinity,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::Preserved,
            subnormal: SubnormalBehavior::PreservedIEEE,
            infinity: InfinityBehavior::SignedInfinity,
            nan: NanBehavior::CanonicalQuietNan,
            min_finite: f64::MIN,
            max_finite: f64::MAX,
        },
        DataType::Bool => NumericSemantics {
            datatype: DataType::Bool,
            bit_width: Some(1),
            format: NumericFormat::Boolean,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::NotApplicable,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::NotApplicable,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::Unsupported,
            nan: NanBehavior::NotApplicable,
            min_finite: 0.0,
            max_finite: 1.0,
        },
        other => NumericSemantics {
            datatype: other.clone(),
            bit_width: other.bit_width(),
            format: NumericFormat::Structured,
            exponent_bits: None,
            mantissa_bits: None,
            exponent_bias: None,
            rounding: RoundingMode::Exact,
            overflow: OverflowBehavior::NotApplicable,
            saturation: SaturationBehavior::None,
            signed_zero: SignedZeroBehavior::NotApplicable,
            subnormal: SubnormalBehavior::Unsupported,
            infinity: InfinityBehavior::Unsupported,
            nan: NanBehavior::NotApplicable,
            min_finite: 0.0,
            max_finite: 0.0,
        },
    }
}
