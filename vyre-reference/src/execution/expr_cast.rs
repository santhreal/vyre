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
    // A float source converts numerically ONLY to U32/I32 (saturating), Bool
    // (truthy), or F32 (identity). There is NO defined float -> {narrow int,
    // 64-bit int, vector, bytes} conversion: the foundation validator
    // (`validate::cast::cast_is_valid`) rejects these, and the naga and PTX
    // emitters fail closed on them. Without this guard the bytes/widen fallbacks
    // below would return a meaningless byte payload (the float's raw bits) for
    // such a cast, silently diverging from every backend. Fail closed so the
    // reference SPEC agrees with the emitters (Law 10 coherence).
    if matches!(value, Value::Float(_))
        && !matches!(
            target,
            DataType::U32 | DataType::I32 | DataType::Bool | DataType::F32
        )
    {
        return Err(Error::interp(format!(
            "cast from f32 to {target:?} has no defined conversion: a float source \
             converts only to u32/i32 (saturating), bool (truthy), or f32. Fix: cast \
             the f32 to u32 or i32 first, then narrow or widen the integer."
        )));
    }
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
        // 64-bit integer widening. `I64` and `U64` share the `Value::U64`
        // bit-pattern representation (the model has no distinct `I64`). The
        // high bits extend per the SOURCE's signedness so the reference matches
        // the backends (PTX `cvt.s64.s32`, naga sign-replicate) and Rust `as`:
        // a signed `i32` SIGN-extends (`-1i32 -> 0xFFFF_FFFF_FFFF_FFFF`), an
        // unsigned/bool source zero-extends. `try_as_u64`'s `u64::try_from`
        // would instead REJECT negative `i32` — diverging from every backend —
        // so the signed case is handled explicitly here before delegating the
        // rest. Adding `I64` here also removes it from the `_ => to_bytes()`
        // catch-all, which silently produced a 4-byte payload with no extension.
        DataType::U64 | DataType::I64 => match value {
            Value::I32(v) => Ok(Value::U64(*v as i64 as u64)),
            _ => value
                .try_as_u64()
                .map(Value::U64)
                .ok_or_else(|| invalid_cast(target, value)),
        },
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

    /// Signed 32-bit widened to a 64-bit integer must SIGN-extend so the
    /// reference matches the backends (PTX `cvt.s64.s32`, naga sign-replicate)
    /// and Rust `i32 as i64`. Before the fix `try_as_u64`'s `u64::try_from`
    /// REJECTED negatives, so a negative `i32 -> u64` cast errored in the oracle
    /// while every backend produced a value — a silent divergence. And `I64`
    /// had no arm at all (fell to the `to_bytes()` catch-all, 4 bytes, no
    /// extension).
    #[test]
    fn cast_i32_to_u64_and_i64_sign_extends_negative() {
        // -1i32 -> 0xFFFF_FFFF_FFFF_FFFF in both U64 and I64 (shared bits).
        assert_eq!(
            cast_value(&DataType::U64, &Value::I32(-1)).expect("i32->u64 must succeed"),
            Value::U64(0xFFFF_FFFF_FFFF_FFFF),
            "negative i32 -> u64 must sign-extend, not reject or zero-extend"
        );
        assert_eq!(
            cast_value(&DataType::I64, &Value::I32(-1)).expect("i32->i64 must succeed"),
            Value::U64(0xFFFF_FFFF_FFFF_FFFF),
            "negative i32 -> i64 must sign-extend (shares the U64 bit pattern)"
        );
        // A different negative proves true sign extension, not a constant.
        assert_eq!(
            cast_value(&DataType::I64, &Value::I32(-2)).expect("i32->i64 must succeed"),
            Value::U64(0xFFFF_FFFF_FFFF_FFFE),
            "i32 -2 -> i64 must be 0xFFFF_FFFF_FFFF_FFFE"
        );
    }

    /// The non-negative twin: u32 and non-negative i32 ZERO-extend into the
    /// 64-bit value (high bits clear).
    #[test]
    fn cast_to_u64_zero_extends_non_negative() {
        assert_eq!(
            cast_value(&DataType::U64, &Value::U32(7)).expect("u32->u64 must succeed"),
            Value::U64(7),
            "u32 -> u64 must zero-extend"
        );
        assert_eq!(
            cast_value(&DataType::I64, &Value::U32(0xDEAD_BEEF)).expect("u32->i64 must succeed"),
            Value::U64(0x0000_0000_DEAD_BEEF),
            "u32 -> i64 must zero-extend (high word clear)"
        );
        assert_eq!(
            cast_value(&DataType::U64, &Value::I32(5)).expect("i32->u64 must succeed"),
            Value::U64(5),
            "non-negative i32 -> u64 must equal the value (sign bit clear)"
        );
    }

    /// A float source has no defined conversion to a narrow int, a 64-bit int, a
    /// vector, or bytes — the validator rejects these and the naga/PTX emitters
    /// fail closed, so the reference SPEC must too (rather than returning the
    /// float's raw bytes via the catch-all). Only u32/i32 (saturating), bool, and
    /// f32 are valid float targets.
    #[test]
    fn cast_f32_to_undefined_target_fails_closed() {
        for target in [
            DataType::U8,
            DataType::U16,
            DataType::I8,
            DataType::I16,
            DataType::U64,
            DataType::I64,
            DataType::Vec2U32,
            DataType::Vec4U32,
            DataType::Bytes,
        ] {
            let err = cast_value(&target, &Value::Float(3.5))
                .expect_err(&format!("f32 -> {target:?} must fail closed in the oracle"));
            assert!(
                format!("{err:?}").contains("no defined conversion"),
                "f32 -> {target:?} must fail closed with the float-cast message, got: {err:?}"
            );
        }
    }

    /// The positive twin: the float targets every layer permits still convert,
    /// with the saturating (truncate-toward-zero) / truthy / identity semantics.
    #[test]
    fn cast_f32_to_permitted_targets_converts() {
        assert_eq!(
            cast_value(&DataType::U32, &Value::Float(3.9)).expect("f32->u32 must succeed"),
            Value::U32(3),
            "f32 3.9 -> u32 truncates toward zero"
        );
        assert_eq!(
            cast_value(&DataType::I32, &Value::Float(-2.9)).expect("f32->i32 must succeed"),
            Value::I32(-2),
            "f32 -2.9 -> i32 truncates toward zero"
        );
        assert_eq!(
            cast_value(&DataType::Bool, &Value::Float(0.0)).expect("f32->bool must succeed"),
            Value::Bool(false),
            "f32 0.0 -> bool is false"
        );
        assert_eq!(
            cast_value(&DataType::Bool, &Value::Float(1.5)).expect("f32->bool must succeed"),
            Value::Bool(true),
            "f32 1.5 -> bool is true"
        );
        assert_eq!(
            cast_value(&DataType::F32, &Value::Float(2.5)).expect("f32->f32 must succeed"),
            Value::Float(2.5),
            "f32 -> f32 is identity"
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
