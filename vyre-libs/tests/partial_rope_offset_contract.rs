//! Cache-position and pass-through contracts for partial RoPE.

#![forbid(unsafe_code)]

mod wire_words;
use wire_words::{f32_bytes as bytes, f32_words_of as decode};

use vyre_libs::nn::attention::partial_rope_at_offset;
use vyre_reference::value::Value;

/// Proves cached decode reads its absolute table position and leaves pass-through dimensions byte-exact.
#[test]
fn decode_offset_rotates_only_the_configured_prefix() {
    let program = partial_rope_at_offset("input", "cos", "sin", "output", 1, 1, 4, 2, 1, 2);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bytes(&[1.0, 2.0, 3.0, 4.0])),
            Value::from(bytes(&[1.0, 0.0])),
            Value::from(bytes(&[0.0, 1.0])),
        ],
    )
    .expect("Fix: offset RoPE must execute");
    assert_eq!(decode(&outputs[0]), vec![-2.0, 1.0, 3.0, 4.0]);
}

/// Locks full rotary coverage while preserving the same absolute cache position semantics.
#[test]
fn full_rotary_dimensions_use_offset_table_rows() {
    let program = partial_rope_at_offset("input", "cos", "sin", "output", 1, 1, 4, 4, 1, 2);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bytes(&[1.0, 2.0, 3.0, 4.0])),
            Value::from(bytes(&[1.0, 1.0, 0.0, 0.0])),
            Value::from(bytes(&[0.0, 0.0, 1.0, 1.0])),
        ],
    )
    .expect("Fix: full offset RoPE must execute");
    assert_eq!(decode(&outputs[0]), vec![-2.0, 1.0, -4.0, 3.0]);
}

/// Prevents a cache range beyond the available tables from becoming executable through OOB zero-fill.
#[test]
fn offset_range_beyond_tables_fails_validation() {
    let program = partial_rope_at_offset("input", "cos", "sin", "output", 1, 2, 4, 2, 2, 3);
    let error = vyre_reference::reference_eval(&program, &[])
        .expect_err("Fix: table range overflow must remain invalid");
    assert!(error.to_string().contains("position range"), "{error}");
}

/// Proves Qwen3.5 production head and partial-rotary dimensions construct exact buffer sizes.
#[test]
fn qwen35_production_dimensions_materialize_exact_contracts() {
    let program = partial_rope_at_offset("q", "cos", "sin", "q.rotated", 24, 1, 256, 64, 17, 32);
    assert_eq!(program.buffers()[0].count(), 24 * 256);
    assert_eq!(program.buffers()[1].count(), 32 * 32);
    assert_eq!(program.buffers()[2].count(), 32 * 32);
    assert_eq!(program.buffers()[3].count(), 24 * 256);
}
