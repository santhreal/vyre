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

#[cfg(feature = "go-parser")]
pub(crate) mod go;
