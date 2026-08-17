//! Numerical reference tests and adversarial cases for paged attention programs.

#![forbid(unsafe_code)]

use vyre_foundation::ir::DataType;
use vyre_libs::nn::attention::{paged_attention, PagedAttentionSpec};
use vyre_reference::reference_eval;
use vyre_reference::value::Value;

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    bytes
}

fn u32_to_bytes(data: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_ne_bytes());
    }
    bytes
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn paged_gqa_attention_broadcasts_kv_heads() {
    // 1 sequence, 4 Q heads, 2 KV heads (group_size = 2), 1 query token, 2 context tokens
    // 1 physical block of 2 tokens, head_dim = 2
    let spec = PagedAttentionSpec {
        q: "q",
        k_cache: "k_cache",
        v_cache: "v_cache",
        block_table: "block_table",
        output: "output",
        sequences: 1,
        q_heads: 4,
        kv_heads: 2,
        query_tokens: 1,
        context_tokens: 2,
        blocks: 1,
        block_tokens: 2,
        blocks_per_sequence: 1,
        head_dim: 2,
        cache_offset: 1,
        causal: false,
        dtype: DataType::F32,
        scale: Some(1.0),
    };
    let program = paged_attention(&spec).expect("paged gqa program");

    // q: 4 heads * 2 dim = 8 elements
    let q_data = vec![
        1.0f32, 0.0, // Q head 0 -> maps to KV head 0
        0.0, 1.0, // Q head 1 -> maps to KV head 0
        1.0, 0.0, // Q head 2 -> maps to KV head 1
        0.0, 1.0, // Q head 3 -> maps to KV head 1
    ];

    // KV Cache: [1 block, 2 kv_heads, 2 tokens, 2 dim] = 8 elements
    // KV Head 0: tok0=[1.0, 0.0], tok1=[0.0, 1.0]
    // KV Head 1: tok0=[0.5, 0.5], tok1=[0.5, -0.5]
    let k_data = vec![
        1.0f32, 0.0, 0.0, 1.0, // KV Head 0
        0.5, 0.5, 0.5, -0.5, // KV Head 1
    ];
    let v_data = vec![
        10.0f32, 20.0, 30.0, 40.0, // KV Head 0
        100.0, 200.0, 300.0, 400.0, // KV Head 1
    ];

    let table_data = vec![0u32];
    let out_init = vec![0.0f32; 8];

    let inputs = vec![
        Value::from(f32_to_bytes(&q_data)),
        Value::from(f32_to_bytes(&k_data)),
        Value::from(f32_to_bytes(&v_data)),
        Value::from(u32_to_bytes(&table_data)),
        Value::from(f32_to_bytes(&out_init)),
    ];

    let outputs = reference_eval(&program, &inputs).expect("eval");
    let result = bytes_to_f32(&outputs[0].to_bytes());
    assert_eq!(result.len(), 8);

    // Q Head 0 dot with KV Head 0: tok 0 dot=1, tok 1 dot=0 -> weights exp(1)/(e+1), 1/(e+1)
    let e = 1.0f32.exp();
    let sum = e + 1.0;
    let expected_q0_0 = (e * 10.0 + 1.0 * 30.0) / sum;
    let expected_q0_1 = (e * 20.0 + 1.0 * 40.0) / sum;

    assert!((result[0] - expected_q0_0).abs() < 1e-4);
    assert!((result[1] - expected_q0_1).abs() < 1e-4);
}

#[test]
fn paged_attention_partial_page_boundary() {
    // Block size = 4, but context_tokens = 3 (partial page)
    let spec = PagedAttentionSpec {
        q: "q",
        k_cache: "k_cache",
        v_cache: "v_cache",
        block_table: "block_table",
        output: "output",
        sequences: 1,
        q_heads: 1,
        kv_heads: 1,
        query_tokens: 1,
        context_tokens: 3, // only read 3 tokens
        blocks: 1,
        block_tokens: 4,
        blocks_per_sequence: 1,
        head_dim: 2,
        cache_offset: 2,
        causal: false,
        dtype: DataType::F32,
        scale: Some(1.0),
    };
    let program = paged_attention(&spec).expect("program");

    let q_data = vec![1.0f32, 1.0];
    let k_data = vec![
        1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 999.0, 999.0, // token 3 is uninitialized/garbage
    ];
    let v_data = vec![10.0f32, 10.0, 20.0, 20.0, 30.0, 30.0, 9999.0, 9999.0];
    let table_data = vec![0u32];
    let out_init = vec![0.0f32; 2];

    let inputs = vec![
        Value::from(f32_to_bytes(&q_data)),
        Value::from(f32_to_bytes(&k_data)),
        Value::from(f32_to_bytes(&v_data)),
        Value::from(u32_to_bytes(&table_data)),
        Value::from(f32_to_bytes(&out_init)),
    ];

    let outputs = reference_eval(&program, &inputs).expect("eval");
    let result = bytes_to_f32(&outputs[0].to_bytes());

    // Dot products:
    // tok 0: 1.0
    // tok 1: 1.0
    // tok 2: 2.0
    // tok 3 must NOT be accessed
    let e1 = 1.0f32.exp();
    let e2 = 2.0f32.exp();
    let sum = e1 + e1 + e2;
    let expected = (e1 * 10.0 + e1 * 20.0 + e2 * 30.0) / sum;

    assert!((result[0] - expected).abs() < 1e-4);
    assert!((result[1] - expected).abs() < 1e-4);
}
