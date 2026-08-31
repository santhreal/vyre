//! Exhaustive and boundary contract tests for the versioned numeric semantics table.

#[path = "spec_variants/mod.rs"]
mod spec_variants;

use spec_variants::SCALAR_LEAF_TYPES;
use vyre_spec::*;

#[test]
fn schema_version_is_current() {
    assert_eq!(NUMERIC_SEMANTICS_SCHEMA_VERSION, 1);
}

#[test]
fn all_scalar_types_have_valid_numeric_semantics() {
    for dtype in SCALAR_LEAF_TYPES.iter().filter(|d| {
        !matches!(
            d,
            DataType::Vec2U32 | DataType::Vec4U32 | DataType::Bytes | DataType::Tensor
        )
    }) {
        let sem = numeric_semantics_for(dtype);
        assert_eq!(&sem.datatype, dtype);
        assert!(sem.min_finite <= sem.max_finite);
        assert!(sem.bit_width.is_some());
    }
}

#[test]
fn i4_exhaustive_16_elements_roundtrip() {
    for nibble in 0..16u8 {
        let decoded = i4_to_i32(nibble);
        assert!(
            (-8..=7).contains(&decoded),
            "I4 decoded value {decoded} out of range [-8, 7]"
        );
        let encoded = i32_to_i4(decoded);
        assert_eq!(
            encoded, nibble,
            "I4 roundtrip failed for nibble {nibble:#04x}"
        );
    }
}

#[test]
fn fp4_exhaustive_16_elements_table_and_decode() {
    for nibble in 0..16u8 {
        let decoded = fp4_to_f32(nibble);
        assert_eq!(
            decoded, FP4_DECODE_TABLE[nibble as usize],
            "FP4 decode table mismatch for nibble {nibble}"
        );
        assert!(
            decoded.abs() <= 6.0,
            "FP4 value {decoded} exceeds max finite 6.0"
        );
    }

    // Check specific known FP4 values
    assert_eq!(fp4_to_f32(0x00).to_bits(), 0.0f32.to_bits());
    assert_eq!(fp4_to_f32(0x08).to_bits(), (-0.0f32).to_bits());
    assert_eq!(fp4_to_f32(0x01), 0.5);
    assert_eq!(fp4_to_f32(0x09), -0.5);
    assert_eq!(fp4_to_f32(0x02), 1.0);
    assert_eq!(fp4_to_f32(0x0A), -1.0);
    assert_eq!(fp4_to_f32(0x07), 6.0);
    assert_eq!(fp4_to_f32(0x0F), -6.0);
}

#[test]
fn nf4_exhaustive_16_elements_monotonic_and_quantile_bounds() {
    assert_eq!(NF4_QUANTILE_TABLE.len(), 16);

    // Exact index 7 must be zero
    assert_eq!(NF4_QUANTILE_TABLE[7], 0.0);
    assert_eq!(NF4_QUANTILE_TABLE[0], -1.0);
    assert_eq!(NF4_QUANTILE_TABLE[15], 1.0);

    // Verify strict monotonicity
    for i in 0..15 {
        assert!(
            NF4_QUANTILE_TABLE[i] < NF4_QUANTILE_TABLE[i + 1],
            "NF4 quantiles must be strictly increasing: index {i} ({}) >= index {} ({})",
            NF4_QUANTILE_TABLE[i],
            i + 1,
            NF4_QUANTILE_TABLE[i + 1]
        );
    }

    // Verify all 16 roundtrip through decode/encode
    for nibble in 0..16u8 {
        let decoded = nf4_to_f32(nibble);
        let encoded = f32_to_nf4(decoded);
        assert_eq!(
            encoded, nibble,
            "NF4 roundtrip failed for quantile index {nibble}"
        );
    }
}

#[test]
fn f8e4m3_exhaustive_256_elements_properties() {
    for byte in 0..=255u8 {
        let decoded = f8e4m3_to_f32(byte);
        if byte == 0x7F || byte == 0xFF {
            assert!(
                decoded.is_nan(),
                "F8E4M3 byte {byte:#04x} must decode to NaN"
            );
        } else if byte == 0x00 {
            assert_eq!(decoded.to_bits(), 0.0f32.to_bits());
        } else if byte == 0x80 {
            assert_eq!(decoded.to_bits(), (-0.0f32).to_bits());
        } else {
            assert!(
                decoded.abs() <= 448.0,
                "F8E4M3 byte {byte:#04x} decoded to {decoded}, exceeding max 448.0"
            );
        }
    }
}

#[test]
fn f8e5m2_exhaustive_256_elements_properties() {
    for byte in 0..=255u8 {
        let decoded = f8e5m2_to_f32(byte);
        if byte == 0x7C {
            assert_eq!(decoded, f32::INFINITY);
        } else if byte == 0xFC {
            assert_eq!(decoded, f32::NEG_INFINITY);
        } else if (0x7D..=0x7F).contains(&byte) || (0xFD..=0xFF).contains(&byte) {
            assert!(
                decoded.is_nan(),
                "F8E5M2 byte {byte:#04x} must decode to NaN"
            );
        } else if byte == 0x00 {
            assert_eq!(decoded.to_bits(), 0.0f32.to_bits());
        } else if byte == 0x80 {
            assert_eq!(decoded.to_bits(), (-0.0f32).to_bits());
        } else {
            assert!(
                decoded.abs() <= 57344.0,
                "F8E5M2 byte {byte:#04x} decoded to {decoded}, exceeding max 57344.0"
            );
        }
    }
}

#[test]
fn grouped_quantization_dequantize_matches_exact_math() {
    // 4 elements packed in 2 bytes of INT4: [1, -2, 3, -4]
    // byte 0: low nibble = 1 (0x01), high nibble = -2 (0xFE -> 0x0E) -> 0xE1
    // byte 1: low nibble = 3 (0x03), high nibble = -4 (0xFC -> 0x0C) -> 0xC3
    let storage = vec![0xE1, 0xC3];
    let scales = vec![0.5f32, 2.0f32];
    let zero_points = vec![0.0f32, 1.0f32];

    let dequantized = dequantize_grouped_f32(
        &storage,
        &DataType::I4,
        &scales,
        Some(&zero_points),
        2, // group size 2
        4, // 4 elements
    )
    .expect("dequantize must succeed");

    assert_eq!(dequantized.len(), 4);
    // Group 0 (elements 0 and 1, scale 0.5, zp 0.0):
    // element 0 = (1 - 0.0) * 0.5 = 0.5
    // element 1 = (-2 - 0.0) * 0.5 = -1.0
    assert_eq!(dequantized[0], 0.5);
    assert_eq!(dequantized[1], -1.0);

    // Group 1 (elements 2 and 3, scale 2.0, zp 1.0):
    // element 2 = (3 - 1.0) * 2.0 = 4.0
    // element 3 = (-4 - 1.0) * 2.0 = -10.0
    assert_eq!(dequantized[2], 4.0);
    assert_eq!(dequantized[3], -10.0);
}
