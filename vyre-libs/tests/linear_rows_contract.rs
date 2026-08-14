//! Row-batched canonical linear projection contracts.

#![forbid(unsafe_code)]

mod common;
use common::{f32_bytes as bytes, f32_words_of as decode};

use vyre::ir::DataType;
use vyre_libs::nn::linear::{linear_rows, linear_rows_no_bias_out_in_typed};
use vyre_reference::value::Value;

/// Locks row-major affine projection and bias application for more than one token row.
#[test]
fn two_rows_match_exact_hand_computed_projection() {
    let program = linear_rows("x", "weight", "bias", "output", 2, 3, 2)
        .expect("Fix: valid row projection must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bytes(&[1.0, 2.0, 3.0, -1.0, 0.0, 2.0])),
            Value::from(bytes(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0])),
            Value::from(bytes(&[0.5, -0.5])),
            Value::from(vec![0; 4 * 4]),
        ],
    )
    .expect("Fix: row projection must execute");
    assert_eq!(decode(&outputs[0]), vec![4.5, 4.5, 1.5, 1.5]);
}

/// Prevents one prompt row from leaking into another through flattened indexing.
#[test]
fn zero_first_row_does_not_change_nonzero_second_row() {
    let program = linear_rows("x", "weight", "bias", "output", 2, 2, 1)
        .expect("Fix: valid row isolation fixture must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bytes(&[0.0, 0.0, 4.0, -1.0])),
            Value::from(bytes(&[2.0, 3.0])),
            Value::from(bytes(&[0.0])),
            Value::from(vec![0; 2 * 4]),
        ],
    )
    .expect("Fix: isolated row projection must execute");
    assert_eq!(decode(&outputs[0]), vec![0.0, 5.0]);
}

/// Locks checkpoint-native `[out_dim, in_dim]` indexing against accidental transposition.
#[test]
fn output_major_checkpoint_weights_project_exactly() {
    let program = linear_rows_no_bias_out_in_typed("x", "weight", "output", 1, 2, 2, DataType::F32)
        .expect("Fix: valid checkpoint projection must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bytes(&[2.0, 3.0])),
            Value::from(bytes(&[1.0, 10.0, 100.0, 1000.0])),
            Value::from(vec![0; 2 * 4]),
        ],
    )
    .expect("Fix: checkpoint projection must execute");
    assert_eq!(decode(&outputs[0]), vec![32.0, 3200.0]);
}

/// Ensures empty row and feature dimensions fail before creating buffer declarations.
#[test]
fn zero_dimensions_fail_closed() {
    for dimensions in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
        let error = linear_rows("x", "w", "b", "o", dimensions.0, dimensions.1, dimensions.2)
            .expect_err("Fix: zero linear_rows dimension must fail");
        assert!(error.contains("nonzero"));
    }
}

/// Locks every flattened-count overflow boundary independently.
#[test]
fn flattened_count_overflow_fails_before_building() {
    assert!(linear_rows("x", "w", "b", "o", u32::MAX, 2, 1)
        .expect_err("Fix: input row overflow must fail")
        .contains("rows*in_dim"));
    assert!(linear_rows("x", "w", "b", "o", u32::MAX, 1, 2)
        .expect_err("Fix: output row overflow must fail")
        .contains("rows*out_dim"));
    assert!(linear_rows("x", "w", "b", "o", 1, u32::MAX, 2)
        .expect_err("Fix: weight overflow must fail")
        .contains("in_dim*out_dim"));
}
