//! Shared visual inventory helpers.

/// Convert packed `u32` pixels/words into little-endian bytes for harness IO.
///
/// Routes through the canonical `vyre-primitives::wire::pack_u32_slice`
/// LEGO primitive (with `bytemuck::cast_slice` fast path on LE hosts).
/// Local visual helpers used to re-implement the same `flat_map(to_le_bytes)`
/// loop in every byte-pack call site; the dedup makes the visual harness
/// pay the same allocation/throughput cost as every other GPU dispatch.
pub(crate) use vyre_primitives::wire::pack_u32_slice as u32_words_to_le_bytes;
