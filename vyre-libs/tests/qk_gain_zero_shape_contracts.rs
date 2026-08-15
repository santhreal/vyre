//! Empty-shape contracts for the QK-gain output allocation.

#![cfg(feature = "nn-attention")]

use vyre_libs::nn::attention::qk_gain;
use vyre_reference::value::Value;

fn evaluate_empty_shape(num_heads: u32, seq_len: u32, head_dim: u32) -> Vec<u8> {
    let program = qk_gain("q_in", "q_out", "gain", num_heads, seq_len, head_dim);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(Vec::<u8>::new()),
            Value::from(Vec::<u8>::new()),
            Value::from(vec![0_u8; num_heads as usize * size_of::<f32>()]),
        ],
    )
    .expect("an empty QK-gain shape must declare an explicit zero-byte output");

    outputs[0].to_bytes()
}

/// Zero heads must produce an explicit empty output instead of an unknown-size backend allocation.
#[test]
fn zero_heads_produce_zero_output_bytes() {
    assert_eq!(evaluate_empty_shape(0, 7, 11), Vec::<u8>::new());
}

/// Zero head width must remain executable even when the head and sequence dimensions are nonzero.
#[test]
fn zero_head_dimension_produces_zero_output_bytes() {
    assert_eq!(evaluate_empty_shape(3, 5, 0), Vec::<u8>::new());
}

/// An all-zero tensor shape must not invent a sentinel element to satisfy readback sizing.
#[test]
fn all_zero_dimensions_produce_zero_output_bytes() {
    assert_eq!(evaluate_empty_shape(0, 0, 0), Vec::<u8>::new());
}
