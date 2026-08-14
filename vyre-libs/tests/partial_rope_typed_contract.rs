//! Mixed-precision partial rotary-position contracts.

#![forbid(unsafe_code)]

mod common;
use common::{bf16_bytes, bf16_word, f32_bytes};

use vyre::ir::DataType;
use vyre_libs::nn::attention::partial_rope_at_offset_typed;
use vyre_reference::value::Value;

fn decode_words(value: &Value) -> Vec<u16> {
    value
        .to_bytes()
        .chunks_exact(size_of::<u16>())
        .map(|word| u16::from_le_bytes(word.try_into().expect("Fix: exact BF16 word")))
        .collect()
}

/// Proves cached BF16 decode uses its absolute F32 table row and preserves the unrotated suffix exactly.
#[test]
fn bf16_decode_offset_rotates_prefix_and_preserves_suffix_words() {
    let program = partial_rope_at_offset_typed(
        "input",
        "cos",
        "sin",
        "output",
        1,
        1,
        4,
        2,
        1,
        2,
        DataType::BF16,
    )
    .expect("Fix: BF16 partial RoPE must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bf16_bytes(&[1.0, 2.0, 3.0, 4.0])),
            Value::from(f32_bytes(&[1.0, 0.0])),
            Value::from(f32_bytes(&[0.0, 1.0])),
            Value::from(vec![0; 4 * size_of::<u16>()]),
        ],
    )
    .expect("Fix: BF16 offset RoPE must execute");
    assert_eq!(
        decode_words(&outputs[0]),
        vec![
            bf16_word(-2.0),
            bf16_word(1.0),
            bf16_word(3.0),
            bf16_word(4.0)
        ]
    );
}

/// Locks F32 rotation math followed by one BF16 conversion instead of repeated reduced-precision arithmetic.
#[test]
fn bf16_rotation_rounds_once_after_f32_math() {
    let program = partial_rope_at_offset_typed(
        "input",
        "cos",
        "sin",
        "output",
        1,
        1,
        2,
        2,
        0,
        1,
        DataType::BF16,
    )
    .expect("Fix: BF16 full RoPE must build");
    let x0 = f32::from_bits(u32::from(bf16_word(1.1)) << 16);
    let x1 = f32::from_bits(u32::from(bf16_word(-0.7)) << 16);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bf16_bytes(&[x0, x1])),
            Value::from(f32_bytes(&[0.5])),
            Value::from(f32_bytes(&[0.25])),
            Value::from(vec![0; 2 * size_of::<u16>()]),
        ],
    )
    .expect("Fix: mixed-precision RoPE must execute");
    assert_eq!(
        decode_words(&outputs[0]),
        vec![
            bf16_word(x0 * 0.5 - x1 * 0.25),
            bf16_word(x0 * 0.25 + x1 * 0.5),
        ]
    );
}

/// Prevents integer activation storage from entering the rotary attention path.
#[test]
fn integer_rotary_activation_dtype_fails_closed() {
    let error = partial_rope_at_offset_typed(
        "input",
        "cos",
        "sin",
        "output",
        1,
        1,
        2,
        2,
        0,
        1,
        DataType::U32,
    )
    .expect_err("Fix: integer rotary activations must fail");
    assert!(
        error.contains("F16, BF16, or F32"),
        "unexpected error: {error}"
    );
}

/// Prevents an overflowing cache-position range from becoming an executable BF16 graph.
#[test]
fn bf16_position_range_overflow_traps_before_buffer_access() {
    let program = partial_rope_at_offset_typed(
        "input",
        "cos",
        "sin",
        "output",
        1,
        2,
        2,
        2,
        u32::MAX,
        u32::MAX,
        DataType::BF16,
    )
    .expect("Fix: shape errors are represented by invalid programs");
    let error = vyre_reference::reference_eval(&program, &[])
        .expect_err("Fix: overflowing rotary position range must trap");
    assert!(error.to_string().contains("position range"), "{error}");
}

/// Locks production Qwen buffer dtypes: BF16 activations with F32 trigonometric tables.
#[test]
fn qwen35_typed_rotary_contract_uses_mixed_buffer_dtypes() {
    let program = partial_rope_at_offset_typed(
        "q",
        "cos",
        "sin",
        "q.rotated",
        24,
        1,
        256,
        64,
        17,
        32,
        DataType::BF16,
    )
    .expect("Fix: production Qwen BF16 RoPE must build");
    assert_eq!(program.buffers()[0].element, DataType::BF16);
    assert_eq!(program.buffers()[1].element, DataType::F32);
    assert_eq!(program.buffers()[2].element, DataType::F32);
    assert_eq!(program.buffers()[3].element, DataType::BF16);
    assert_eq!(program.buffers()[0].count(), 24 * 256);
    assert_eq!(program.buffers()[1].count(), 32 * 32);
}
