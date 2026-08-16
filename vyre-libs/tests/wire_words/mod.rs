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

//! Wire helpers every `vyre-libs` contract test packs its oracle buffers with.
//!
//! `vyre_primitives::wire` already owns the little-endian packers and decoders,
//! so a test that writes its own `flat_map(to_le_bytes)` loop is a second copy
//! of a shipped primitive. The BF16 rounding has no production owner because
//! only the typed contracts need it, so it is owned here.
#![allow(unused_imports, unused_macros)]

use vyre_primitives::wire::decode_u16_le_bytes_all;
use vyre_reference::value::Value;

pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_bytes;

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;

pub(crate) use vyre_primitives::wire::pack_u32_slice as bytes_from_words;

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as words_from_bytes;

pub(crate) use vyre_primitives::wire::pack_f32_slice as f32_bytes;

pub(crate) use vyre_primitives::wire::decode_f32_le_bytes_all as f32_words;

pub(crate) use vyre_primitives::wire::pack_u16_slice as u16_bytes;

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

/// Advance a 32-bit xorshift state and return next u32.
pub(crate) fn next_u32(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
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
#[cfg(feature = "go-parser")]
pub(crate) mod go;
