//! Causal and cached grouped-query attention contracts.

#![forbid(unsafe_code)]

use vyre_libs::nn::attention::gqa_attention_causal;
use vyre_reference::value::Value;

fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn decode(value: &Value) -> Vec<f32> {
    value
        .to_bytes()
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().expect("Fix: exact f32 word")))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn execute(
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
) -> Vec<f32> {
    let program = gqa_attention_causal(
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
    )
    .expect("Fix: valid causal GQA fixture must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bytes(q)),
            Value::from(bytes(k)),
            Value::from(bytes(v)),
            Value::from(vec![0; q.len() * 4]),
        ],
    )
    .expect("Fix: causal GQA must execute");
    decode(&outputs[0])
}

/// Proves prompt token zero cannot receive probability mass from a future value.
#[test]
fn prefill_applies_exact_triangular_causal_limit() {
    assert_eq!(
        execute(&[0.0, 0.0], &[0.0, 0.0], &[2.0, 10.0], 1, 1, 1, 2, 2, 1, 0),
        vec![2.0, 6.0]
    );
}

/// Proves cached decode can attend all prior cache rows through its absolute offset.
#[test]
fn decode_offset_attends_complete_visible_cache() {
    assert_eq!(
        execute(&[0.0], &[0.0; 3], &[1.0, 2.0, 9.0], 1, 1, 1, 1, 3, 1, 2),
        vec![4.0]
    );
}

/// Ensures query heads in one group broadcast the same KV head without sharing output rows.
#[test]
fn explicit_query_to_kv_grouping_preserves_each_query_row() {
    assert_eq!(
        execute(&[0.0, 0.0], &[0.0], &[3.0], 1, 2, 1, 1, 1, 1, 0),
        vec![3.0, 3.0]
    );
}

/// Prevents flattened addressing from mixing independent batches.
#[test]
fn batch_addressing_isolation_is_exact() {
    assert_eq!(
        execute(&[0.0, 0.0], &[0.0, 0.0], &[3.0, 7.0], 2, 1, 1, 1, 1, 1, 0),
        vec![3.0, 7.0]
    );
}

/// Locks construction refusal for invalid groups, cache ranges, and overflowed production shapes.
#[test]
fn invalid_causal_gqa_contracts_fail_closed() {
    assert!(
        gqa_attention_causal("q", "k", "v", "o", 1, 3, 2, 1, 1, 1, 0)
            .expect_err("Fix: invalid head ratio must fail")
            .contains("multiple")
    );
    assert!(
        gqa_attention_causal("q", "k", "v", "o", 1, 1, 1, 2, 2, 1, 1)
            .expect_err("Fix: cache range overflow must fail")
            .contains("exceeds")
    );
    assert!(
        gqa_attention_causal("q", "k", "v", "o", u32::MAX, 24, 4, 1, 1, 256, 0,)
            .expect_err("Fix: flattened production overflow must fail")
            .contains("overflows")
    );
}

/// Proves exact Qwen3.5 head, KV-group, head-width, and decode-cache dimensions materialize.
#[test]
fn qwen35_production_decode_dimensions_build_exact_buffers() {
    let program = gqa_attention_causal("q", "k", "v", "o", 1, 24, 4, 1, 18, 256, 17)
        .expect("Fix: Qwen production decode dimensions must build");
    assert_eq!(program.buffers()[0].count(), 24 * 256);
    assert_eq!(program.buffers()[1].count(), 4 * 18 * 256);
    assert_eq!(program.buffers()[2].count(), 4 * 18 * 256);
    assert_eq!(program.buffers()[3].count(), 24 * 256);
}
