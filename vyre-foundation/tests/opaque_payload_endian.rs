//! Test: opaque payload endian.
include!(
    "contract_cases/opaque_payload_endian__u16_little_endian_bytes_match_canonical_pattern.rs"
);
include!("contract_cases/opaque_payload_endian__canonical_f32_zero_nonzero_with_sign_bit_set_passes_through.rs");
