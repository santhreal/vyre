//! Last-dimension L2 normalization execution contracts.

#![forbid(unsafe_code)]

mod wire_words;
use wire_words::f32_bytes;

use vyre::ir::DataType;
use vyre_libs::nn::norm::{last_dim_l2_norm, LastDimL2NormError};
use vyre_reference::value::Value;

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn execute_f32(input: &[f32], rows: u32, width: u32, eps: f32) -> Vec<f32> {
    let program = last_dim_l2_norm("input", "output", rows, width, eps, DataType::F32)
        .expect("Fix: valid L2 fixture must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(f32_bytes(input)),
            Value::from(vec![0; input.len() * 4]),
        ],
    )
    .expect("Fix: L2 normalization must execute");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("Fix: exact f32 word")))
        .collect()
}

/// Locks row isolation and the Qwen/FLA sum-of-squares plus epsilon formula.
#[test]
fn two_rows_match_exact_last_dimension_oracle() {
    let actual = execute_f32(&[3.0, 4.0, 5.0, 12.0], 2, 2, 1e-6);
    for (index, (actual, expected)) in actual
        .iter()
        .zip([0.6_f32, 0.8, 5.0 / 13.0, 12.0 / 13.0])
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 2e-6,
            "element {index}: expected {expected}, got {actual}"
        );
    }
}

/// Prevents zero and flushed F32 subnormal rows from producing NaN or nonzero output.
#[test]
fn zero_and_subnormal_rows_normalize_to_exact_zero() {
    let tiny = f32::from_bits(1);
    assert_eq!(
        execute_f32(&[0.0, 0.0, tiny, -tiny], 2, 2, 1e-6),
        vec![0.0; 4]
    );
}

/// Proves epsilon remains inside the inverse-square-root denominator.
#[test]
fn epsilon_controls_tiny_vector_magnitude() {
    let small = execute_f32(&[1e-4], 1, 1, 1e-8)[0];
    let large = execute_f32(&[1e-4], 1, 1, 1.0)[0];
    assert!(small > 0.70 && small < 0.71);
    assert!(large > 9.9e-5 && large < 1.01e-4);
}

/// Locks the authoritative F32 overflow behavior for unscaled sum-of-squares.
#[test]
fn overflowing_sum_of_squares_yields_zero_via_inverse_infinity() {
    assert_eq!(execute_f32(&[3e20, 4e20], 1, 2, 1e-6), vec![0.0, 0.0]);
}

/// Exercises scalar, grouped-head, and Qwen-size rows without crossing row boundaries.
#[test]
fn representative_head_widths_have_unit_l2_magnitude() {
    for width in [1_u32, 8, 64, 256] {
        let input = (0..width)
            .map(|index| (index as f32 + 1.0) / width as f32)
            .collect::<Vec<_>>();
        let output = execute_f32(&input, 1, width, 1e-6);
        let magnitude = output.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!(
            (magnitude - 1.0).abs() <= 2e-5,
            "width {width}: expected unit magnitude, got {magnitude}"
        );
    }
}

/// Proves BF16 input and output use exact round-to-nearest-even bytes around F32 accumulation.
#[test]
fn bf16_execution_matches_exact_output_words() {
    let program = last_dim_l2_norm("input", "output", 1, 2, 1e-6, DataType::BF16)
        .expect("Fix: BF16 L2 normalization must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(u16_bytes(&[0x4040, 0x4080])),
            Value::from(vec![0; 4]),
        ],
    )
    .expect("Fix: BF16 L2 normalization must execute");
    assert_eq!(outputs[0].to_bytes(), u16_bytes(&[0x3f1a, 0x3f4d]));
}

/// Prevents empty, overflowing, and integer tensors from becoming invalid executable artifacts.
#[test]
fn invalid_l2_shapes_and_dtypes_fail_closed() {
    assert_eq!(
        last_dim_l2_norm("x", "y", 1, 0, 1e-6, DataType::F32)
            .expect_err("Fix: zero width must fail"),
        LastDimL2NormError::EmptyShape { rows: 1, width: 0 }
    );
    assert_eq!(
        last_dim_l2_norm("x", "y", u32::MAX, 2, 1e-6, DataType::F32)
            .expect_err("Fix: flattened overflow must fail"),
        LastDimL2NormError::ElementCountOverflow
    );
    assert_eq!(
        last_dim_l2_norm("x", "y", 1, 2, 1e-6, DataType::I32)
            .expect_err("Fix: integer source must fail"),
        LastDimL2NormError::UnsupportedDtype {
            dtype: DataType::I32
        }
    );
}
