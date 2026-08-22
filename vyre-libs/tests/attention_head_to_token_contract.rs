//! Head-major to token-major attention layout contracts.

#![forbid(unsafe_code)]

mod wire_words;
use wire_words::f32_bytes as bytes;

use vyre::ir::DataType;
use vyre_libs::nn::attention::{attention_head_to_token, AttentionPermuteSpec};
use vyre_reference::value::Value;

fn execute(input: &[f32], batch: u32, heads: u32, sequence: u32, dim: u32) -> Vec<f32> {
    let program = attention_head_to_token(AttentionPermuteSpec {
        input: "input",
        output: "output",
        batch,
        heads,
        sequence,
        head_dim: dim,
        dtype: DataType::F32,
    })
    .expect("Fix: valid attention layout fixture must build");
    let outputs = vyre_reference::reference_eval(&program, &[Value::from(bytes(input))])
        .expect("Fix: attention layout conversion must execute");
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("Fix: exact f32 word")))
        .collect()
}

/// Locks the projection-row order for multiple heads and prompt tokens.
#[test]
fn prompt_heads_are_interleaved_inside_each_token_row() {
    assert_eq!(
        execute(&[0.0, 1.0, 2.0, 10.0, 11.0, 12.0], 1, 2, 3, 1),
        vec![0.0, 10.0, 1.0, 11.0, 2.0, 12.0]
    );
}

/// Proves batch, token, head, and feature axes remain isolated simultaneously.
#[test]
fn batches_and_multidimensional_heads_permute_exactly() {
    let input = (0..16).map(|value| value as f32).collect::<Vec<_>>();
    assert_eq!(
        execute(&input, 2, 2, 2, 2),
        vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 10.0, 11.0, 14.0, 15.0,]
    );
}

/// Locks decode's one-token layout as a value-preserving identity per batch.
#[test]
fn one_token_decode_preserves_concatenated_head_order() {
    assert_eq!(
        execute(&[1.0, 2.0, 3.0, 4.0], 1, 2, 1, 2),
        vec![1.0, 2.0, 3.0, 4.0]
    );
}

/// Ensures empty shapes and hostile flattened counts fail before buffer creation.
#[test]
fn invalid_layout_dimensions_fail_closed() {
    let spec = |batch, heads| AttentionPermuteSpec {
        input: "i",
        output: "o",
        batch,
        heads,
        sequence: 1,
        head_dim: 1,
        dtype: DataType::F32,
    };
    assert!(attention_head_to_token(spec(0, 1))
        .expect_err("Fix: zero batch must fail")
        .contains("nonzero"));
    assert!(attention_head_to_token(spec(u32::MAX, 2))
        .expect_err("Fix: flattened count overflow must fail")
        .contains("overflows"));
}
