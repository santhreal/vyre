//! Wire helpers for tests.

pub(crate) use vyre_primitives::wire::decode_u32_le_bytes_all as words_from_bytes;
pub(crate) use vyre_primitives::wire::pack_u32_slice as bytes_from_words;
