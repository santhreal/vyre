//! Cast helpers for the expression evaluator.

use crate::ops::{read_u32_prefix, read_u64_prefix};
use crate::value::Value;
use vyre::ir::DataType;
use vyre::Error;

pub(crate) fn spec_output_value(ty: DataType, bytes: &[u8]) -> Value {
    match ty {
        DataType::U32 => Value::U32(read_u32_prefix(bytes)),
        DataType::I32 => Value::I32(read_u32_prefix(bytes) as i32),
        DataType::Bool => Value::Bool(read_u32_prefix(bytes) != 0),
        DataType::U64 => Value::U64(read_u64_prefix(bytes)),
        DataType::F32 => Value::Float(f64::from(crate::execution::typed_ops::canonical_f32(
            f32::from_bits(read_u32_prefix(bytes)),
        ))),
        DataType::Vec2U32 => Value::from(read_fixed_prefix(bytes, 8)),
        DataType::Vec4U32 => Value::from(read_fixed_prefix(bytes, 16)),
        DataType::Bytes => Value::from(bytes),
        _ => Value::from(bytes),
    }
}

pub(crate) fn cast_value(target: &DataType, value: &Value) -> Result<Value, vyre::Error> {
    match target {
        DataType::U32 => match value {
            Value::I32(v) => Ok(Value::U32(*v as u32)),
            Value::Float(v) => Ok(Value::U32((*v) as u32)),
            _ => value
                .try_as_u32()
                .map(Value::U32)
                .ok_or_else(|| invalid_cast(target, value)),
        },
        DataType::I32 => match value {
            Value::I32(value) => Ok(Value::I32(*value)),
            Value::Float(v) => Ok(Value::I32(*v as i32)),
            _ => value
                .try_as_u32()
                .map(|value| Value::I32(value as i32))
                .ok_or_else(|| invalid_cast(target, value)),
        },
        DataType::U64 => value
            .try_as_u64()
            .map(Value::U64)
            .ok_or_else(|| invalid_cast(target, value)),
        DataType::F32 => match value {
            // Integer → float is a value conversion, not a bit-cast,
            // matching backend `f32(u32_value)` semantics. Without this
            // arm the evaluator dropped through to the bytes-copy
            // fallback and produced f32 bits equal to the raw u32,
            // which aliases u32(5) to 7e-45 instead of 5.0.
            Value::U32(v) => Ok(Value::Float(f64::from(*v as f32))),
            Value::I32(v) => Ok(Value::Float(f64::from(*v as f32))),
            Value::U64(v) => Ok(Value::Float(f64::from(*v as f32))),
            Value::Float(v) => Ok(Value::Float(*v)),
            Value::Bool(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
            _ => value
                .try_as_u32()
                .map(|v| Value::Float(f64::from(v as f32)))
                .ok_or_else(|| invalid_cast(target, value)),
        },
        DataType::Bool => Ok(Value::Bool(value.truthy())),
        DataType::Bytes => Ok(Value::from(value.to_bytes())),
        DataType::Vec2U32 => Ok(Value::from(widen_to_words(value, 2))),
        DataType::Vec4U32 => Ok(Value::from(widen_to_words(value, 4))),
        _ => Ok(Value::from(value.to_bytes())),
    }
}

fn read_fixed_prefix(bytes: &[u8], width: usize) -> Vec<u8> {
    let mut fixed = vec![0u8; width];
    let len = bytes.len().min(width);
    fixed[..len].copy_from_slice(&bytes[..len]);
    fixed
}

fn invalid_cast(target: &DataType, value: &Value) -> Error {
    Error::interp(format!(
        "cast to {target:?} cannot represent {value:?} losslessly. Fix: cast from an in-range scalar value."
    ))
}

fn widen_to_words(value: &Value, words: usize) -> Vec<u8> {
    let target_bytes = words * 4;
    let mut bytes = vec![0u8; target_bytes];
    value.write_bytes_width_into(&mut bytes);
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cast_to_vec2_pads_scalar_bytes_to_fixed_width() {
        let value = cast_value(&DataType::Vec2U32, &Value::U32(0x0403_0201))
            .expect("Fix: scalar to vec2 cast must be representable.");

        assert_eq!(
            value.to_bytes(),
            vec![1, 2, 3, 4, 0, 0, 0, 0],
            "Fix: vector casts must preserve source bytes and zero-fill the declared lane width."
        );
    }

    #[test]
    fn cast_to_vec4_truncates_oversized_byte_payload_to_fixed_width() {
        let value = cast_value(
            &DataType::Vec4U32,
            &Value::from((0u8..24).collect::<Vec<_>>()),
        )
        .expect("Fix: byte payload to vec4 cast must be representable.");

        assert_eq!(
            value.to_bytes(),
            (0u8..16).collect::<Vec<_>>(),
            "Fix: vector casts must truncate oversized payloads at the declared lane width."
        );
    }

    /// VRH-002: spec_output_value must apply canonical_f32 for F32 call outputs
    /// so subnormals and NaN payloads round-trip identically to from_element_bytes.
    ///
    /// Before the fix, 0x0000_0001 (positive subnormal) yielded Value::Float(1.4e-45);
    /// after the fix it yields Value::Float(0.0) — matching from_element_bytes.
    #[test]
    fn spec_output_value_canonicalizes_f32_subnormal_to_zero() {
        // Positive subnormal: smallest positive subnormal f32 = 0x0000_0001.
        let subnormal_bytes = 0x0000_0001_u32.to_le_bytes();
        let result = spec_output_value(DataType::F32, &subnormal_bytes);
        // canonical_f32 maps positive subnormal → +0.0 (preserves sign bit only).
        assert_eq!(
            result,
            Value::Float(0.0_f64),
            "Fix: spec_output_value must canonicalize positive subnormal f32 to +0.0, \
             identical to from_element_bytes — before VRH-002 fix this returned 1.4e-45"
        );
    }

    /// VRH-002: negative subnormal canonicalizes to -0.0.
    #[test]
    fn spec_output_value_canonicalizes_f32_negative_subnormal_to_negative_zero() {
        // Negative subnormal: 0x8000_0001.
        let neg_subnormal_bytes = 0x8000_0001_u32.to_le_bytes();
        let result = spec_output_value(DataType::F32, &neg_subnormal_bytes);
        // canonical_f32 maps negative subnormal → -0.0 (sign bit preserved, mantissa cleared).
        assert_eq!(
            result,
            Value::Float(f64::from(f32::from_bits(0x8000_0000))),
            "Fix: spec_output_value must canonicalize negative subnormal f32 to -0.0"
        );
    }

    /// VRH-002: NaN payload canonicalizes to the canonical quiet NaN (0x7FC0_0000).
    #[test]
    fn spec_output_value_canonicalizes_f32_nan_payload_to_canonical_nan() {
        // Payload NaN: signaling NaN with a custom payload bit.
        let payload_nan_bytes = 0x7FA0_0001_u32.to_le_bytes();
        let result = spec_output_value(DataType::F32, &payload_nan_bytes);
        // canonical_f32 maps any NaN to 0x7FC0_0000.
        let expected_bits = f32::from_bits(0x7FC0_0000);
        assert_eq!(
            result,
            Value::Float(f64::from(expected_bits)),
            "Fix: spec_output_value must canonicalize NaN f32 payloads to 0x7FC0_0000, \
             identical to from_element_bytes — before VRH-002 fix this returned the raw NaN payload"
        );
    }

    /// VRH-002: normal f32 values pass through unchanged.
    #[test]
    fn spec_output_value_normal_f32_passes_through_unchanged() {
        // 1.5f32 = 0x3FC0_0000 — a normal value, not subnormal or NaN.
        let normal_bytes = 0x3FC0_0000_u32.to_le_bytes();
        let result = spec_output_value(DataType::F32, &normal_bytes);
        assert_eq!(
            result,
            Value::Float(f64::from(1.5_f32)),
            "Fix: spec_output_value must not alter normal f32 values"
        );
    }
}
