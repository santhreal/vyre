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
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Canonical 16-element I4 decode table (`[-8, 7]`).
pub const I4_DECODE_TABLE: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1];

fn nearest_codebook_index(table: &[f32; 16], target: f32) -> u8 {
    let mut best_idx = 0u8;
    let mut best_diff = f32::INFINITY;
    for (i, &entry) in table.iter().enumerate() {
        let diff = (entry - target).abs();
        if diff < best_diff {
            best_diff = diff;
            best_idx = i as u8;
        }
    }
    best_idx
}

fn nearest_f8_byte(decode: fn(u8) -> f32, max_code: u8, abs_val: f32) -> u8 {
    let mut best_byte = 0u8;
    let mut best_diff = f32::INFINITY;
    for byte in 0..=max_code {
        let diff = (decode(byte) - abs_val).abs();
        if diff < best_diff {
            best_diff = diff;
            best_byte = byte;
        }
    }
    best_byte
}

/// Decode one 8-bit F8E4M3 float value to f32.
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
            if sign {
                -0.0
            } else {
                0.0
            }
        } else {
            sign_mult * (2.0f32.powi(-6)) * (mant as f32 / 8.0)
        }
    } else {
        sign_mult * (2.0f32.powi(exp as i32 - 7)) * (1.0 + mant as f32 / 8.0)
    }
}

/// Encode one f32 value to 8-bit F8E4M3.
#[must_use]
pub fn f32_to_f8e4m3(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7F;
    }
    let sign_bit = if val.is_sign_negative() {
        0x80u8
    } else {
        0x00u8
    };
    let abs_val = val.abs();
    if abs_val == 0.0 || abs_val < (2.0f32.powi(-9) * 0.5) {
        return sign_bit;
    }
    if abs_val > 448.0 {
        return sign_bit | 0x7E;
    }
    sign_bit | nearest_f8_byte(f8e4m3_to_f32, 0x7E, abs_val)
}

/// Decode one 8-bit F8E5M2 float value to f32.
#[must_use]
pub fn f8e5m2_to_f32(byte: u8) -> f32 {
    let sign = (byte & 0x80) != 0;
    let exp = (byte >> 2) & 0x1F;
    let mant = byte & 0x03;
    let sign_mult = if sign { -1.0 } else { 1.0 };

    if exp == 0x1F {
        return if mant == 0 {
            if sign {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    if exp == 0 {
        if mant == 0 {
            if sign {
                -0.0
            } else {
                0.0
            }
        } else {
            sign_mult * (2.0f32.powi(-14)) * (mant as f32 / 4.0)
        }
    } else {
        sign_mult * (2.0f32.powi(exp as i32 - 15)) * (1.0 + mant as f32 / 4.0)
    }
}

/// Generate the 256-element F8E4M3 decode table.
#[must_use]
pub fn f8e4m3_decode_table() -> [f32; 256] {
    core::array::from_fn(|i| f8e4m3_to_f32(i as u8))
}

/// Generate the 256-element F8E5M2 decode table.
#[must_use]
pub fn f8e5m2_decode_table() -> [f32; 256] {
    core::array::from_fn(|i| f8e5m2_to_f32(i as u8))
}

/// Encode one f32 value to 8-bit F8E5M2.
#[must_use]
pub fn f32_to_f8e5m2(val: f32) -> u8 {
    if val.is_nan() {
        return 0x7E;
    }
    let sign_bit = if val.is_sign_negative() {
        0x80u8
    } else {
        0x00u8
    };
    if val.is_infinite() {
        return sign_bit | 0x7C;
    }
    let abs_val = val.abs();
    if abs_val == 0.0 {
        return sign_bit;
    }
    if abs_val > 57344.0 {
        return sign_bit | 0x7C;
    }
    sign_bit | nearest_f8_byte(f8e5m2_to_f32, 0x7C, abs_val)
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
        return 0x07;
    }
    nearest_codebook_index(&FP4_DECODE_TABLE, val)
}

/// Decode one 4-bit NF4 nibble (0..15) to normalized f32 in [-1.0, 1.0].
#[must_use]
pub fn nf4_to_f32(nibble: u8) -> f32 {
    NF4_QUANTILE_TABLE[(nibble & 0x0F) as usize]
}

/// Encode one f32 value to 4-bit NF4 nibble (0..15) via nearest quantile.
#[must_use]
pub fn f32_to_nf4(val: f32) -> u8 {
    let clamped = if val.is_nan() {
        0.0
    } else {
        val.clamp(-1.0, 1.0)
    };
    nearest_codebook_index(&NF4_QUANTILE_TABLE, clamped)
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
    (val.clamp(-8, 7) as u8) & 0x0F
}

fn unpack_nibble(storage_bytes: &[u8], elem_index: usize) -> Result<u8, &'static str> {
    let byte = *storage_bytes
        .get(elem_index / 2)
        .ok_or("storage_bytes truncated")?;
    Ok(if elem_index % 2 != 0 {
        (byte >> 4) & 0x0F
    } else {
        byte & 0x0F
    })
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
        return Err("group_size must be greater than zero".into());
    }

    let mut result = Vec::with_capacity(element_count);
    for i in 0..element_count {
        let group_idx = i / group_size;
        let scale = *scales.get(group_idx).unwrap_or(&1.0);
        let zp = zero_points
            .and_then(|z| z.get(group_idx).copied())
            .unwrap_or(0.0);

        let unscaled = match storage_type {
            DataType::I4 => i4_to_i32(unpack_nibble(storage_bytes, i)?) as f32,
            DataType::FP4 => fp4_to_f32(unpack_nibble(storage_bytes, i)?),
            DataType::NF4 => nf4_to_f32(unpack_nibble(storage_bytes, i)?),
            DataType::F8E4M3 => {
                f8e4m3_to_f32(*storage_bytes.get(i).ok_or("storage_bytes truncated")?)
            }
            DataType::F8E5M2 => {
                f8e5m2_to_f32(*storage_bytes.get(i).ok_or("storage_bytes truncated")?)
            }
            DataType::I8 => (*storage_bytes.get(i).ok_or("storage_bytes truncated")? as i8) as f32,
            DataType::U8 => *storage_bytes.get(i).ok_or("storage_bytes truncated")? as f32,
            _ => return Err(format!("unsupported storage datatype `{storage_type}`")),
        };
        result.push((unscaled - zp) * scale);
    }
    Ok(result)
}

fn int_semantics(
    datatype: DataType,
    bits: usize,
    format: NumericFormat,
    saturation: SaturationBehavior,
    min_finite: f64,
    max_finite: f64,
) -> NumericSemantics {
    NumericSemantics {
        datatype,
        bit_width: Some(bits),
        format,
        exponent_bits: None,
        mantissa_bits: None,
        exponent_bias: None,
        rounding: RoundingMode::Exact,
        overflow: OverflowBehavior::WrapTwoComplement,
        saturation,
        signed_zero: SignedZeroBehavior::UnsignedZero,
        subnormal: SubnormalBehavior::Unsupported,
        infinity: InfinityBehavior::Unsupported,
        nan: NanBehavior::NotApplicable,
        min_finite,
        max_finite,
    }
}

#[allow(clippy::too_many_arguments)]
fn float_semantics(
    datatype: DataType,
    bits: usize,
    format: NumericFormat,
    exp_mant_bias: Option<(u32, u32, i32)>,
    rounding: RoundingMode,
    overflow: OverflowBehavior,
    saturation: SaturationBehavior,
    signed_zero: SignedZeroBehavior,
    subnormal: SubnormalBehavior,
    infinity: InfinityBehavior,
    nan: NanBehavior,
    max_finite: f64,
) -> NumericSemantics {
    NumericSemantics {
        datatype,
        bit_width: Some(bits),
        format,
        exponent_bits: exp_mant_bias.map(|(e, _, _)| e),
        mantissa_bits: exp_mant_bias.map(|(_, m, _)| m),
        exponent_bias: exp_mant_bias.map(|(_, _, b)| b),
        rounding,
        overflow,
        saturation,
        signed_zero,
        subnormal,
        infinity,
        nan,
        min_finite: -max_finite,
        max_finite,
    }
}

fn ieee_flushed_semantics(
    datatype: DataType,
    bits: usize,
    exp: u32,
    mant: u32,
    bias: i32,
    max_finite: f64,
) -> NumericSemantics {
    float_semantics(
        datatype,
        bits,
        NumericFormat::IeeeFloat,
        Some((exp, mant, bias)),
        RoundingMode::RoundToNearestEven,
        OverflowBehavior::SignedInfinity,
        SaturationBehavior::None,
        SignedZeroBehavior::PreservedWithCanonicalFlushing,
        SubnormalBehavior::FlushedToSignedZero,
        InfinityBehavior::SignedInfinity,
        NanBehavior::CanonicalQuietNan,
        max_finite,
    )
}

/// Return the authoritative numeric semantics specification for `dtype`.
#[must_use]
pub fn numeric_semantics_for(dtype: &DataType) -> NumericSemantics {
    match dtype {
        DataType::U8 => int_semantics(
            DataType::U8,
            8,
            NumericFormat::UnsignedInteger,
            SaturationBehavior::None,
            0.0,
            255.0,
        ),
        DataType::U16 => int_semantics(
            DataType::U16,
            16,
            NumericFormat::UnsignedInteger,
            SaturationBehavior::None,
            0.0,
            65535.0,
        ),
        DataType::U32 => int_semantics(
            DataType::U32,
            32,
            NumericFormat::UnsignedInteger,
            SaturationBehavior::None,
            0.0,
            u32::MAX as f64,
        ),
        DataType::U64 => int_semantics(
            DataType::U64,
            64,
            NumericFormat::UnsignedInteger,
            SaturationBehavior::None,
            0.0,
            u64::MAX as f64,
        ),
        DataType::I8 => int_semantics(
            DataType::I8,
            8,
            NumericFormat::SignedInteger,
            SaturationBehavior::None,
            -128.0,
            127.0,
        ),
        DataType::I16 => int_semantics(
            DataType::I16,
            16,
            NumericFormat::SignedInteger,
            SaturationBehavior::None,
            -32768.0,
            32767.0,
        ),
        DataType::I32 => int_semantics(
            DataType::I32,
            32,
            NumericFormat::SignedInteger,
            SaturationBehavior::None,
            i32::MIN as f64,
            i32::MAX as f64,
        ),
        DataType::I64 => int_semantics(
            DataType::I64,
            64,
            NumericFormat::SignedInteger,
            SaturationBehavior::None,
            i64::MIN as f64,
            i64::MAX as f64,
        ),
        DataType::I4 => int_semantics(
            DataType::I4,
            4,
            NumericFormat::SignedInteger,
            SaturationBehavior::ClampsToFiniteRange,
            -8.0,
            7.0,
        ),
        DataType::FP4 => float_semantics(
            DataType::FP4,
            4,
            NumericFormat::IeeeLikeFloat,
            Some((2, 1, 1)),
            RoundingMode::RoundToNearestEven,
            OverflowBehavior::SaturateToFiniteRange,
            SaturationBehavior::ClampsToFiniteRange,
            SignedZeroBehavior::Preserved,
            SubnormalBehavior::PreservedIEEE,
            InfinityBehavior::SaturatedToMaxFinite,
            NanBehavior::UnsupportedOrZero,
            6.0,
        ),
        DataType::NF4 => float_semantics(
            DataType::NF4,
            4,
            NumericFormat::NormalFloat,
            None,
            RoundingMode::NearestQuantile,
            OverflowBehavior::SaturateToFiniteRange,
            SaturationBehavior::ClampsToUnitInterval,
            SignedZeroBehavior::UnsignedZero,
            SubnormalBehavior::Unsupported,
            InfinityBehavior::SaturatedToMaxFinite,
            NanBehavior::UnsupportedOrZero,
            1.0,
        ),
        DataType::F8E4M3 => float_semantics(
            DataType::F8E4M3,
            8,
            NumericFormat::IeeeLikeFloat,
            Some((4, 3, 7)),
            RoundingMode::RoundToNearestEven,
            OverflowBehavior::NanOrSaturate,
            SaturationBehavior::ClampsToFiniteRange,
            SignedZeroBehavior::Preserved,
            SubnormalBehavior::PreservedIEEE,
            InfinityBehavior::SaturatedToMaxFinite,
            NanBehavior::DedicatedNanBitPattern,
            448.0,
        ),
        DataType::F8E5M2 => float_semantics(
            DataType::F8E5M2,
            8,
            NumericFormat::IeeeLikeFloat,
            Some((5, 2, 15)),
            RoundingMode::RoundToNearestEven,
            OverflowBehavior::SignedInfinity,
            SaturationBehavior::None,
            SignedZeroBehavior::Preserved,
            SubnormalBehavior::PreservedIEEE,
            InfinityBehavior::SignedInfinity,
            NanBehavior::CanonicalQuietNan,
            57344.0,
        ),
        DataType::F16 => ieee_flushed_semantics(DataType::F16, 16, 5, 10, 15, 65504.0),
        DataType::BF16 => ieee_flushed_semantics(DataType::BF16, 16, 8, 7, 127, 3.3895314e38),
        DataType::F32 => ieee_flushed_semantics(DataType::F32, 32, 8, 23, 127, f32::MAX as f64),
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
