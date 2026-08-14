//! Qwen gated RMSNorm execution and refusal contracts.

#![forbid(unsafe_code)]

mod common;
use common::{f32_bytes, f32_words as decode_f32};

use vyre::ir::DataType;
use vyre_libs::nn::norm::{gated_rms_norm, gated_rms_norm_with_weight_dtype, GatedRmsNormError};
use vyre_reference::value::Value;

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn execute(
    input: &[f32],
    weight: &[f32],
    gate: &[f32],
    rows: u32,
    hidden: u32,
    eps: f32,
) -> Vec<f32> {
    let program = gated_rms_norm(
        "input",
        "weight",
        "gate",
        "output",
        rows,
        hidden,
        eps,
        DataType::F32,
    )
    .expect("Fix: valid gated RMSNorm fixture must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(f32_bytes(input)),
            Value::from(f32_bytes(weight)),
            Value::from(f32_bytes(gate)),
            Value::from(vec![0; input.len() * 4]),
        ],
    )
    .expect("Fix: gated RMSNorm must execute in the reference evaluator");
    assert_eq!(outputs.len(), 1);
    decode_f32(&outputs[0].to_bytes())
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: expected {expected:?}, got {actual:?}, tolerance {tolerance}"
        );
    }
}

/// Locks the Transformers ordering: normalize a row, round, scale, then apply the float32 SiLU gate.
#[test]
fn exact_two_row_fixture_matches_authoritative_operation_order() {
    let actual = execute(
        &[1.0, -2.0, 3.0, -4.0, 0.25, -0.5, 2.0, 1.0],
        &[0.5, 1.5, -1.0, 2.0],
        &[-3.0, 0.0, 2.0, 8.0, -1.0, 1.0, -8.0, 3.0],
        2,
        4,
        1e-6,
    );
    assert_close(
        &actual,
        &[
            -0.025_976_218,
            0.0,
            -1.929_729_6,
            -23.361_658,
            -0.029_170_781,
            -0.475_766_45,
            0.004_655_848_3,
            4.959_414_5,
        ],
        3e-6,
    );
}

/// Prevents zero-variance rows from producing nonzero values or division-by-zero artifacts.
#[test]
fn zero_rows_produce_exact_signed_zero_independent_of_gate() {
    let actual = execute(
        &[0.0; 8],
        &[1.0, -2.0, 3.0, -4.0],
        &[-100.0, -1.0, 0.0, 100.0, 1.0, 2.0, 3.0, 4.0],
        2,
        4,
        1e-6,
    );
    assert_eq!(actual, vec![0.0; 8]);
}

/// Proves epsilon participates inside inverse RMS rather than after normalization.
#[test]
fn epsilon_changes_tiny_variance_rows_at_the_expected_boundary() {
    let input = [1e-8, -1e-8];
    let weight = [1.0, 1.0];
    let gate = [1.0, 1.0];
    let small = execute(&input, &weight, &gate, 1, 2, 1e-12);
    let large = execute(&input, &weight, &gate, 1, 2, 1e-4);
    assert!(small[0] > 0.007 && small[0] < 0.008);
    assert!(large[0] > 7e-7 && large[0] < 8e-7);
    assert!(small[0] > large[0] * 9_000.0);
}

/// Locks fail-propagation semantics: one NaN in a row poisons that row's normalized values, not other rows.
#[test]
fn nan_variance_is_confined_to_its_source_row() {
    let actual = execute(
        &[f32::NAN, 1.0, 3.0, 4.0],
        &[1.0, 1.0],
        &[1.0, 1.0, 1.0, 1.0],
        2,
        2,
        1e-6,
    );
    assert!(actual[0].is_nan());
    assert!(actual[1].is_nan());
    assert_close(&actual[2..], &[0.620_323_8, 0.827_098_4], 3e-6);
}

/// Prevents zero dimensions, index overflow, and integer dtypes from materializing misleading Programs.
#[test]
fn invalid_shapes_and_dtypes_fail_before_program_construction() {
    assert_eq!(
        gated_rms_norm("x", "w", "g", "y", 0, 4, 1e-6, DataType::F32)
            .expect_err("Fix: empty rows must fail"),
        GatedRmsNormError::EmptyShape { rows: 0, hidden: 4 }
    );
    assert_eq!(
        gated_rms_norm("x", "w", "g", "y", u32::MAX, 2, 1e-6, DataType::F32)
            .expect_err("Fix: flattened overflow must fail"),
        GatedRmsNormError::ElementCountOverflow
    );
    assert_eq!(
        gated_rms_norm("x", "w", "g", "y", 1, 4, 1e-6, DataType::U32)
            .expect_err("Fix: integer source must fail"),
        GatedRmsNormError::UnsupportedDtype {
            dtype: DataType::U32
        }
    );
}

/// Ensures F16 and BF16 select two-byte source and output contracts rather than silently materializing F32 buffers.
#[test]
fn low_precision_programs_execute_with_exact_source_dtype_rounding() {
    for (dtype, input, weight, gate, expected) in [
        (
            DataType::F16,
            [0x3c00, 0xc000],
            [0x3c00, 0x3800],
            [0x3c00, 0xbc00],
            [0x3765, 0x3171],
        ),
        (
            DataType::BF16,
            [0x3f80, 0xc000],
            [0x3f80, 0x3f00],
            [0x3f80, 0xbf80],
            [0x3eed, 0x3e2e],
        ),
    ] {
        let program = gated_rms_norm("x", "w", "g", "y", 1, 2, 1e-6, dtype.clone())
            .expect("Fix: low-precision contract must build");
        assert_eq!(program.buffers().len(), 4);
        assert!(program
            .buffers()
            .iter()
            .all(|buffer| buffer.element == dtype));
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(u16_bytes(&input)),
                Value::from(u16_bytes(&weight)),
                Value::from(u16_bytes(&gate)),
                Value::from(vec![0; 4]),
            ],
        )
        .expect("Fix: low-precision gated RMSNorm must execute");
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].to_bytes(), u16_bytes(&expected));
    }
}
/// Locks Qwen's BF16 activation plus F32 learned-scale checkpoint contract to exact output words.
#[test]
fn bf16_activations_with_f32_weights_execute_exactly() {
    let program = gated_rms_norm_with_weight_dtype(
        "x",
        "w",
        "g",
        "y",
        1,
        2,
        1e-6,
        DataType::BF16,
        DataType::F32,
    )
    .expect("Fix: mixed-weight gated RMSNorm must build");
    assert_eq!(program.buffers()[0].element, DataType::BF16);
    assert_eq!(program.buffers()[1].element, DataType::F32);
    assert_eq!(program.buffers()[2].element, DataType::BF16);
    assert_eq!(program.buffers()[3].element, DataType::BF16);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(u16_bytes(&[0x3f80, 0xc000])),
            Value::from(f32_bytes(&[1.0, 0.5])),
            Value::from(u16_bytes(&[0x3f80, 0xbf80])),
            Value::from(vec![0; 4]),
        ],
    )
    .expect("Fix: mixed-weight gated RMSNorm must execute");
    assert_eq!(outputs[0].to_bytes(), u16_bytes(&[0x3eed, 0x3e2e]));
}

/// Prevents an integer learned scale from entering mixed-precision normalization.
#[test]
fn mixed_weight_dtype_must_remain_floating() {
    assert_eq!(
        gated_rms_norm_with_weight_dtype(
            "x",
            "w",
            "g",
            "y",
            1,
            2,
            1e-6,
            DataType::BF16,
            DataType::U32,
        )
        .expect_err("Fix: integer learned scale must fail"),
        GatedRmsNormError::UnsupportedDtype {
            dtype: DataType::U32
        }
    );
}
