//! Mixed-precision causal grouped-query attention contracts.

#![forbid(unsafe_code)]

use vyre::ir::DataType;
use vyre_libs::nn::attention::gqa_attention_causal_typed;
use vyre_reference::value::Value;

fn bf16_word(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

fn bf16_value(value: f32) -> f32 {
    f32::from_bits(u32::from(bf16_word(value)) << 16)
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16_word(*value).to_le_bytes())
        .collect()
}

fn decode_words(value: &Value) -> Vec<u16> {
    value
        .to_bytes()
        .chunks_exact(size_of::<u16>())
        .map(|word| u16::from_le_bytes(word.try_into().expect("Fix: exact BF16 word")))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn execute_bf16(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch: u32,
    query_heads: u32,
    kv_heads: u32,
    query_len: u32,
    kv_len: u32,
    dim: u32,
    offset: u32,
) -> Vec<u16> {
    let program = gqa_attention_causal_typed(
        "q",
        "k",
        "v",
        "output",
        batch,
        query_heads,
        kv_heads,
        query_len,
        kv_len,
        dim,
        offset,
        DataType::BF16,
    )
    .expect("Fix: valid BF16 causal GQA must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bf16_bytes(q)),
            Value::from(bf16_bytes(k)),
            Value::from(bf16_bytes(v)),
            Value::from(vec![0; q.len() * size_of::<u16>()]),
        ],
    )
    .expect("Fix: BF16 causal GQA must execute");
    decode_words(&outputs[0])
}

/// Proves one visible cache row broadcasts its exact BF16 value across every query head.
#[test]
fn bf16_single_token_broadcasts_exact_value_words() {
    assert_eq!(
        execute_bf16(
            &[1.0, 0.0, 0.0, 1.0],
            &[1.0, 0.0],
            &[10.0, -20.0],
            1,
            2,
            1,
            1,
            1,
            2,
            0,
        ),
        vec![
            bf16_word(10.0),
            bf16_word(-20.0),
            bf16_word(10.0),
            bf16_word(-20.0)
        ]
    );
}

/// Prevents BF16 prefill token zero from receiving probability mass from a future value.
#[test]
fn bf16_prefill_applies_exact_triangular_visibility() {
    assert_eq!(
        execute_bf16(&[0.0, 0.0], &[0.0, 0.0], &[2.0, 10.0], 1, 1, 1, 2, 2, 1, 0),
        vec![bf16_word(2.0), bf16_word(6.0)]
    );
}

/// Proves BF16 cached decode attends every prior row through the absolute cache offset.
#[test]
fn bf16_decode_offset_attends_complete_visible_cache() {
    assert_eq!(
        execute_bf16(&[0.0], &[0.0; 3], &[1.0, 2.0, 9.0], 1, 1, 1, 1, 3, 1, 2),
        vec![bf16_word(4.0)]
    );
}

/// Locks F32 dot, softmax, and value accumulation followed by one BF16 output conversion.
#[test]
fn bf16_attention_matches_f32_accumulation_oracle() {
    let q = [1.0, 2.0, 3.0, 4.0];
    let k = [0.5, -0.5, 0.25, -0.25, -0.25, 0.5, -0.5, 0.25];
    let v = [1.0, 1.0, 1.0, 1.0, 9.0, 9.0, 9.0, 9.0];
    let quantized_q = q.map(bf16_value);
    let quantized_k = k.map(bf16_value);
    let score = |key: &[f32]| {
        quantized_q
            .iter()
            .zip(key)
            .map(|(query, key)| query * key)
            .sum::<f32>()
            / 2.0
    };
    let score0 = score(&quantized_k[..4]);
    let score1 = score(&quantized_k[4..]);
    let maximum = score0.max(score1);
    let weight0 = (score0 - maximum).exp();
    let weight1 = (score1 - maximum).exp();
    let expected = (weight0 * 1.0 + weight1 * 9.0) / (weight0 + weight1);
    assert_eq!(
        execute_bf16(&q, &k, &v, 1, 1, 1, 1, 2, 4, 1),
        vec![bf16_word(expected); 4]
    );
}

/// Prevents flattened BF16 addressing from mixing independent batches or grouped query heads.
#[test]
fn bf16_batch_and_group_addressing_is_exact() {
    assert_eq!(
        execute_bf16(&[0.0; 4], &[0.0; 2], &[3.0, 7.0], 2, 2, 1, 1, 1, 1, 0,),
        vec![
            bf16_word(3.0),
            bf16_word(3.0),
            bf16_word(7.0),
            bf16_word(7.0)
        ]
    );
}

/// Prevents integer tensor storage from entering floating-point attention.
#[test]
fn integer_causal_gqa_dtype_fails_closed() {
    let error =
        gqa_attention_causal_typed("q", "k", "v", "output", 1, 1, 1, 1, 1, 1, 0, DataType::U32)
            .expect_err("Fix: integer causal GQA storage must fail");
    assert!(
        error.contains("F16, BF16, or F32"),
        "unexpected error: {error}"
    );
}

/// Locks production Qwen BF16 cache geometry and buffer element types.
#[test]
fn qwen35_bf16_decode_dimensions_build_exact_contracts() {
    let program = gqa_attention_causal_typed(
        "q",
        "k",
        "v",
        "output",
        1,
        24,
        4,
        1,
        18,
        256,
        17,
        DataType::BF16,
    )
    .expect("Fix: Qwen production BF16 decode dimensions must build");
    assert_eq!(program.buffers()[0].element, DataType::BF16);
    assert_eq!(program.buffers()[1].element, DataType::BF16);
    assert_eq!(program.buffers()[2].element, DataType::BF16);
    assert_eq!(program.buffers()[3].element, DataType::BF16);
    assert_eq!(program.buffers()[0].count(), 24 * 256);
    assert_eq!(program.buffers()[1].count(), 4 * 18 * 256);
    assert_eq!(program.buffers()[3].count(), 24 * 256);
}
