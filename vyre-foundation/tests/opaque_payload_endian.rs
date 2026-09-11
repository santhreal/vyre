//! Test: opaque payload endian.
//! Implementation lives in two chunks under `contract_cases/`:
//! `opaque_payload_endian__u16_little_endian_bytes_match_canonical_pattern.rs`
//! and its child
//! `opaque_payload_endian__canonical_f32_zero_nonzero_with_sign_bit_set_passes_through.rs`.

#[path = "contract_cases/opaque_payload_endian__u16_little_endian_bytes_match_canonical_pattern.rs"]
mod opaque_payload_endian_u16_little_endian_bytes_match_canonical_pattern;
