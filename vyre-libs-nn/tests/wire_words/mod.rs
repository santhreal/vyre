//! Wire helpers for tests.
#![allow(dead_code, unused_imports, unused_variables)]

use vyre_primitives::wire::decode_u16_le_bytes_all;
use vyre_reference::value::Value;
use vyre_test_support::test_parity_oracles::f32_bytes;

pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    pub(crate) fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            self.next_u32() % n
        }
    }
}

pub(crate) use vyre_primitives::wire::decode_f32_le_bytes_all as f32_words;
pub(crate) use vyre_primitives::wire::decode_u16_le_bytes_all as u16_words;
pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;
pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as words_from_bytes;
pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;
pub(crate) use vyre_primitives::wire::pack_u32_slice as bytes_to_u32;

pub(crate) fn u16_bytes(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    vyre_primitives::wire::pack_u16_slice_into(values, &mut out);
    out
}

/// F32 words from an oracle output value.
pub(crate) fn f32_words_of(value: &Value) -> Vec<f32> {
    f32_words(&value.to_bytes())
}

/// U16 words from an oracle output value, the carrier for a BF16 or F16 lane.
pub(crate) fn u16_words_of(value: &Value) -> Vec<u16> {
    decode_u16_le_bytes_all(&value.to_bytes())
}

/// Round `value` to BF16, breaking ties toward even, the rounding the typed
/// kernels do when they narrow an F32 lane.
pub(crate) fn bf16_word(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// BF16 wire bytes for `values`.
pub(crate) fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    u16_bytes(&values.iter().copied().map(bf16_word).collect::<Vec<_>>())
}

pub(crate) fn lcg_u32(count: usize, seed: u64) -> Vec<u32> {
    let mut rng = Lcg::new(seed);
    (0..count).map(|_| rng.next_u32()).collect()
}

pub(crate) fn ramp(count: usize, start: u32, step: u32) -> Vec<u32> {
    (0..count)
        .map(|i| start.wrapping_add((i as u32).wrapping_mul(step)))
        .collect()
}

pub(crate) fn alternating(count: usize, a: u32, b: u32) -> Vec<u32> {
    (0..count).map(|i| if i % 2 == 0 { a } else { b }).collect()
}

/// Helper for building standard test KvCacheAppendSpec.
pub(crate) fn kv_cache_append_test_spec<'a>(
    batch: u32,
    heads: u32,
    capacity: u32,
    chunk_len: u32,
    head_dim: u32,
    offset: u32,
    dtype: vyre::ir::DataType,
) -> vyre_libs_nn::nn::attention::KvCacheAppendSpec<'a> {
    vyre_libs_nn::nn::attention::KvCacheAppendSpec {
        prior: "prior",
        chunk: "chunk",
        next: "next",
        batch,
        heads,
        capacity,
        chunk_len,
        head_dim,
        offset,
        dtype,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_causal_gqa(
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
    let program = vyre_libs_nn::nn::attention::gqa_attention_causal(
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
            Value::from(f32_bytes(q)),
            Value::from(f32_bytes(k)),
            Value::from(f32_bytes(v)),
        ],
    )
    .expect("Fix: causal GQA must execute");
    f32_words_of(&outputs[0])
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_causal_gqa_typed(
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
    dtype: vyre::ir::DataType,
) -> Vec<u16> {
    let program = vyre_libs_nn::nn::attention::gqa_attention_causal_typed(
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
        dtype,
    )
    .expect("Fix: valid BF16 causal GQA must build");
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(bf16_bytes(q)),
            Value::from(bf16_bytes(k)),
            Value::from(bf16_bytes(v)),
        ],
    )
    .expect("Fix: BF16 causal GQA must execute");
    u16_words_of(&outputs[0])
}

pub(crate) fn default_gated_delta_spec(
    sequence: u32,
    key_heads: u32,
    value_heads: u32,
    key_dim: u32,
    value_dim: u32,
    dtype: vyre::ir::DataType,
) -> vyre_libs_nn::nn::attention::GatedDeltaSpec<'static> {
    vyre_libs_nn::nn::attention::GatedDeltaSpec {
        query: "query",
        key: "key",
        value: "value",
        decay_log: "decay",
        beta_logits: "beta",
        state_input: "state.in",
        output: "output",
        state_output: "state.out",
        batch: 1,
        sequence,
        key_heads,
        value_heads,
        key_dim,
        value_dim,
        eps: 0.0,
        dtype,
    }
}
