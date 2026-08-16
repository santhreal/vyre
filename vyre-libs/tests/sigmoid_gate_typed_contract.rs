//! Mixed-precision sigmoid output-gate contracts.

#![forbid(unsafe_code)]

mod wire_words;
use wire_words::bf16_word;

use vyre::ir::DataType;
use vyre_libs::nn::activation::sigmoid_gate_typed;
use vyre_reference::value::Value;

fn execute_bf16(gate: &[f32], branch: &[f32]) -> Vec<u16> {
    let encode = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| bf16_word(*value).to_le_bytes())
            .collect::<Vec<_>>()
    };
    let program = sigmoid_gate_typed(
        "gate",
        "branch",
        "output",
        gate.len() as u32,
        DataType::BF16,
    )
    .expect("Fix: BF16 sigmoid gate must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(encode(gate)),
            Value::from(encode(branch)),
            Value::from(vec![0; gate.len() * size_of::<u16>()]),
        ],
    )
    .expect("Fix: BF16 sigmoid gate must execute");
    outputs[0]
        .to_bytes()
        .chunks_exact(size_of::<u16>())
        .map(|word| u16::from_le_bytes(word.try_into().expect("Fix: exact BF16 word")))
        .collect()
}

/// Locks the neutral Qwen gate at one half with exact BF16 output conversion.
#[test]
fn bf16_zero_logit_halves_positive_and_negative_branches() {
    assert_eq!(
        execute_bf16(&[0.0, 0.0], &[8.0, -6.0]),
        vec![bf16_word(4.0), bf16_word(-3.0)]
    );
}

/// Proves nonlinear math accumulates in F32 and rounds once at the BF16 output boundary.
#[test]
fn bf16_sigmoid_uses_f32_math_before_output_rounding() {
    let gate = 1.0_f32;
    let branch = 2.0_f32;
    let expected = branch / (1.0 + (-gate).exp());
    assert_eq!(execute_bf16(&[gate], &[branch]), vec![bf16_word(expected)]);
}

/// Prevents saturated BF16 logits from producing NaN, infinity, or a sign error.
#[test]
fn bf16_extreme_logits_reach_closed_and_open_boundaries() {
    assert_eq!(
        execute_bf16(&[-100.0, 100.0], &[7.0, -7.0]),
        vec![bf16_word(0.0), bf16_word(-7.0)]
    );
}

/// Prevents an empty gate from constructing a graph with an unbound output.
#[test]
fn typed_empty_gate_fails_closed() {
    assert!(
        sigmoid_gate_typed("gate", "branch", "output", 0, DataType::BF16)
            .expect_err("Fix: empty typed sigmoid gate must fail")
            .contains("n > 0")
    );
}

/// Prevents integer storage from bypassing the floating attention activation contract.
#[test]
fn typed_integer_gate_fails_closed() {
    let error = sigmoid_gate_typed("gate", "branch", "output", 1, DataType::U32)
        .expect_err("Fix: integer gate storage must fail");
    assert!(
        error.contains("F16, BF16, or F32"),
        "unexpected error: {error}"
    );
}
