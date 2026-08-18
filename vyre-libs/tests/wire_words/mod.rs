//! Wire helpers every `vyre-libs` contract test packs its oracle buffers with.
//!
//! `vyre_primitives::wire` already owns the little-endian packers and decoders,
//! so a test that writes its own `flat_map(to_le_bytes)` loop is a second copy
//! of a shipped primitive. The BF16 rounding has no production owner because
//! only the typed contracts need it, so it is owned here.
#![allow(unused_imports, unused_macros)]

use vyre_primitives::wire::decode_u16_le_bytes_all;
use vyre_reference::value::Value;

/// Minimal linear congruential generator for tests.
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

pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;

pub(crate) use vyre_primitives::wire::pack_u32_slice as bytes_from_words;

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as words_from_bytes;

pub(crate) use vyre_primitives::wire::pack_f32_slice as f32_bytes;

pub(crate) use vyre_primitives::wire::decode_f32_le_bytes_all as f32_words;

pub(crate) fn u16_bytes(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    vyre_primitives::wire::pack_u16_slice_into(values, &mut out);
    out
}

pub(crate) use vyre_primitives::wire::decode_u16_le_bytes_all as u16_words;

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

/// Reference implementation of the Blake3 quarter-round G mixing function.
pub(crate) fn oracle_blake3_g(
    state: &mut [u32; 16],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    mx: u32,
    my: u32,
) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

/// Pseudo-random u32 ramp test vector generator.
pub(crate) fn ramp(len: usize, start: u32) -> Vec<u32> {
    (0..len)
        .map(|idx| start.wrapping_add((idx as u32).wrapping_mul(0x9E37_79B9)))
        .collect()
}

/// Generate pseudo-random u32 sequence from seed.
pub(crate) fn lcg_u32(seed: u32, len: usize) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|idx| {
            state = state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .wrapping_add(idx as u32);
            state
        })
        .collect()
}

/// Generate alternating sequence.
pub(crate) fn alternating(len: usize, even: u32, odd: u32) -> Vec<u32> {
    (0..len)
        .map(|idx| if idx % 2 == 0 { even } else { odd })
        .collect()
}

/// Generate hostile pseudo-random byte vector.
pub(crate) fn hostile_bytes(seed: u32) -> Vec<u8> {
    let len = 1 + (seed as usize % 512);
    let mut v = Vec::with_capacity(len);
    let mut s = seed as u64 ^ 0xDEAD_BEEF_CAFE_BABE;
    for _ in 0..len {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        v.push(s as u8);
    }
    v
}

/// Advance a 64-bit splitmix state.
pub(crate) fn mix64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
#[cfg(feature = "graph")]
pub(crate) fn toposort(
    node_count: u32,
    edges: &[(u32, u32)],
) -> Result<Vec<u32>, vyre_libs::graph::toposort::ToposortError> {
    for (edge_idx, &(from, to)) in edges.iter().enumerate() {
        if from >= node_count {
            return Err(vyre_libs::graph::toposort::ToposortError::UnknownNode {
                edge: edge_idx,
                node: from,
            });
        }
        if to >= node_count {
            return Err(vyre_libs::graph::toposort::ToposortError::UnknownNode {
                edge: edge_idx,
                node: to,
            });
        }
    }
    vyre_reference::composition_witness::toposort_witness(node_count, edges).map_err(|err| {
        if let Some(rest) = err.strip_prefix("Cycle detected involving node ") {
            if let Ok(node) = rest.parse::<u32>() {
                return vyre_libs::graph::toposort::ToposortError::Cycle { node };
            }
        }
        vyre_libs::graph::toposort::ToposortError::InconsistentState { message: err }
    })
}

pub(crate) fn prefix_scan_cpu_ref(
    input: &[u32],
    kind: vyre_libs::math::prefix_scan::ScanKind,
) -> Vec<u32> {
    match kind {
        vyre_libs::math::prefix_scan::ScanKind::InclusiveSum => {
            vyre_reference::composition_witness::inclusive_prefix_sum_witness(input)
        }
        vyre_libs::math::prefix_scan::ScanKind::ExclusiveSum => {
            vyre_reference::composition_witness::exclusive_prefix_sum_witness(input)
        }
    }
}

#[cfg(feature = "pattern")]
pub(crate) fn reference_dedup_regions(
    regions: Vec<vyre_libs::pattern::RegionTriple>,
) -> Vec<vyre_libs::pattern::RegionTriple> {
    let input: Vec<(u32, u32, u32)> = regions.iter().map(|r| (r.pid, r.start, r.end)).collect();
    let deduped = vyre_reference::composition_witness::dedup_regions_witness(input);
    deduped
        .into_iter()
        .map(|(pid, start, end)| vyre_libs::pattern::RegionTriple::new(pid, start, end))
        .collect()
}

#[cfg(feature = "pattern")]
pub(crate) fn reference_dedup_regions_in_place(
    regions: &mut Vec<vyre_libs::pattern::RegionTriple>,
) {
    let deduped = reference_dedup_regions(std::mem::take(regions));
    *regions = deduped;
}

/// Advance a 32-bit xorshift state and return next u32.
pub(crate) fn next_u32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

#[cfg(feature = "nn-attention")]
/// Helper for building standard test KvCacheAppendSpec.
pub(crate) fn kv_cache_append_test_spec<'a>(
    batch: u32,
    heads: u32,
    capacity: u32,
    chunk_len: u32,
    head_dim: u32,
    offset: u32,
    dtype: vyre::ir::DataType,
) -> vyre_libs::nn::attention::KvCacheAppendSpec<'a> {
    vyre_libs::nn::attention::KvCacheAppendSpec {
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
#[cfg(feature = "graph")]
pub(crate) fn queue_forward_oracle(
    active_queue: &[u32],
    queue_len: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    node_count: u32,
    allow_mask: u32,
) -> Vec<u32> {
    let mut out = vec![0u32; vyre_libs::bitset::bitset_words(node_count) as usize];
    let take = (queue_len as usize).min(active_queue.len());
    for &src in &active_queue[..take] {
        if src >= node_count {
            continue;
        }
        let start = edge_offsets[src as usize] as usize;
        let end = edge_offsets[src as usize + 1] as usize;
        for edge in start..end {
            if edge_kind_mask[edge] & allow_mask != 0 {
                let dst = edge_targets[edge];
                out[dst as usize / 32] |= 1u32 << (dst % 32);
            }
        }
    }
    out
}

#[cfg(all(feature = "math-kernels", feature = "graph"))]
pub(crate) fn matroid_intersection_eval(
    exchange_adj: &[u32],
    sources: &[u32],
    sinks: &[u32],
    set_x: &[u32],
    n: u32,
    max_augmentations: u32,
    min_dispatch: u32,
) -> Vec<u32> {
    use vyre_libs::math::matroid_intersection_full::matroid_intersection_full;
    use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
    let program = matroid_intersection_full(
        "exchange_adj",
        "sources",
        "sinks",
        "set_x",
        "parent",
        "frontier",
        "next_frontier",
        "visited",
        "any_change",
        "path_out",
        "path_len",
        n,
        max_augmentations,
    );
    let zeros_n = vec![0u32; n as usize];
    let zero1 = vec![0u32];
    let outputs = vyre_reference::reference_eval_with_dispatch(
        &program,
        &[
            Value::from(pack(exchange_adj)),
            Value::from(pack(sources)),
            Value::from(pack(sinks)),
            Value::from(pack(set_x)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zero1)),
            Value::from(pack(&zeros_n)),
            Value::from(pack(&zero1)),
            Value::from(pack(&zero1)),
        ],
        min_dispatch,
    )
    .expect("matroid_intersection_full reference evaluation must succeed");
    let index = vyre_reference::output_index(&program, "set_x")
        .expect("matroid_intersection_full must declare output set_x");
    unpack(&outputs[index].to_bytes())[..n as usize].to_vec()
}

#[cfg(feature = "nn-attention")]
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
    let program = vyre_libs::nn::attention::gqa_attention_causal(
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
            Value::from(vec![0; q.len() * 4]),
        ],
    )
    .expect("Fix: causal GQA must execute");
    f32_words_of(&outputs[0])
}

#[cfg(feature = "nn-attention")]
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
    let program = vyre_libs::nn::attention::gqa_attention_causal_typed(
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
            Value::from(vec![0; q.len() * std::mem::size_of::<u16>()]),
        ],
    )
    .expect("Fix: BF16 causal GQA must execute");
    u16_words_of(&outputs[0])
}

#[cfg(feature = "nn-attention")]
pub(crate) fn default_gated_delta_spec(
    sequence: u32,
    key_heads: u32,
    value_heads: u32,
    key_dim: u32,
    value_dim: u32,
    dtype: vyre::ir::DataType,
) -> vyre_libs::nn::attention::GatedDeltaSpec<'static> {
    vyre_libs::nn::attention::GatedDeltaSpec {
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

pub(crate) fn signed_fixed_16_16(state: &mut u32) -> u32 {
    let raw = next_u32(state) & 0x0001_FFFF;
    if raw & 0x0001_0000 != 0 {
        0xFFFF_0000 | (raw & 0x0000_FFFF)
    } else {
        raw & 0x0000_FFFF
    }
}
#[cfg(feature = "go-parser")]
pub(crate) mod go;
