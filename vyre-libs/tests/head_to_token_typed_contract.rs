//! Typed head-major to token-major layout contracts.

#![forbid(unsafe_code)]

use vyre::ir::{DataType, Program};
use vyre_libs::nn::attention::{
    attention_head_to_token, attention_token_to_head, AttentionPermuteSpec,
};
use vyre_reference::value::Value;

fn spec<'a>(
    input: &'a str,
    output: &'a str,
    batch: u32,
    heads: u32,
    sequence: u32,
    head_dim: u32,
    dtype: DataType,
) -> AttentionPermuteSpec<'a> {
    AttentionPermuteSpec {
        input,
        output,
        batch,
        heads,
        sequence,
        head_dim,
        dtype,
    }
}

fn execute_words(program: &Program, input: &[u16]) -> Vec<u16> {
    let input_bytes = input
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    let outputs = vyre_reference::reference_eval(program, &[Value::from(input_bytes)])
        .expect("Fix: typed layout conversion must execute");
    outputs[0]
        .to_bytes()
        .chunks_exact(size_of::<u16>())
        .map(|word| u16::from_le_bytes(word.try_into().expect("Fix: exact 16-bit word")))
        .collect()
}

/// Prevents BF16 payloads from being widened or numerically rounded during a pure layout change.
#[test]
fn bf16_head_major_words_are_permuted_bit_exactly() {
    let program = attention_head_to_token(spec("input", "output", 1, 2, 2, 2, DataType::BF16))
        .expect("Fix: BF16 is a supported activation dtype");
    let input = [
        0x3f80, 0x4000, 0x4040, 0x4080, 0xbf80, 0xc000, 0xc040, 0x3f81,
    ];
    assert_eq!(
        execute_words(&program, &input),
        vec![0x3f80, 0x4000, 0xbf80, 0xc000, 0x4040, 0x4080, 0xc040, 0x3f81]
    );
}

/// Locks the same byte-preserving permutation for F16 checkpoint activations.
#[test]
fn f16_head_major_words_are_permuted_bit_exactly() {
    let program = attention_head_to_token(spec("input", "output", 1, 2, 2, 1, DataType::F16))
        .expect("Fix: F16 is a supported activation dtype");
    assert_eq!(
        execute_words(&program, &[0x3c00, 0x4000, 0xbc00, 0x3c01]),
        vec![0x3c00, 0xbc00, 0x4000, 0x3c01]
    );
}

/// Proves both typed permutations are true inverses across batches, heads, tokens, and dimensions.
#[test]
fn bf16_token_and_head_layouts_round_trip_exact_words() {
    let to_heads = attention_token_to_head(spec("input", "heads", 2, 2, 3, 2, DataType::BF16))
        .expect("Fix: valid BF16 token layout must build");
    let to_tokens = attention_head_to_token(spec("heads", "output", 2, 2, 3, 2, DataType::BF16))
        .expect("Fix: valid BF16 head layout must build");
    let input = (0_u16..24).map(|word| word ^ 0xa55a).collect::<Vec<_>>();
    let heads = execute_words(&to_heads, &input);
    assert_eq!(execute_words(&to_tokens, &heads), input);
}

/// Prevents integer or opaque payload types from entering a floating-point attention graph.
#[test]
fn non_float_layout_dtype_fails_closed() {
    let error = attention_head_to_token(spec("input", "output", 1, 1, 1, 1, DataType::U32))
        .expect_err("Fix: integer attention activations must be rejected");
    assert!(
        error.contains("floating dtype"),
        "unexpected error: {error}"
    );
}

/// Prevents zero and overflowing geometry from producing undersized typed buffers.
#[test]
fn typed_layout_geometry_fails_closed() {
    assert!(
        attention_head_to_token(spec("input", "output", 0, 1, 1, 1, DataType::BF16))
            .expect_err("Fix: zero batch must fail")
            .contains("nonzero")
    );
    assert!(
        attention_head_to_token(spec("input", "output", u32::MAX, 2, 1, 1, DataType::BF16))
            .expect_err("Fix: element-count overflow must fail")
            .contains("overflows")
    );
}
