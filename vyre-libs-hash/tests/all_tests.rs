//! One binary for every integration test in this crate.

#[path = "hash_oracles/mod.rs"]
pub mod hash_oracles;

#[path = "adversarial_hash.rs"]
pub mod adversarial_hash;

#[allow(deprecated)]
#[path = "blake3_wrong_size.rs"]
pub mod blake3_wrong_size;

#[path = "fnv1a64_builder_parity.rs"]
pub mod fnv1a64_builder_parity;

#[path = "fnv1a_dyn_parity.rs"]
pub mod fnv1a_dyn_parity;

#[path = "hash_crc32_ir_parity_proptest.rs"]
pub mod hash_crc32_ir_parity_proptest;

#[path = "hash_stream_ir_parity_proptest.rs"]
pub mod hash_stream_ir_parity_proptest;

#[path = "hypervector_ir_parity_proptest.rs"]
pub mod hypervector_ir_parity_proptest;

#[path = "proptest_hash_fnv1a.rs"]
pub mod proptest_hash_fnv1a;

#[path = "sweep_hash_adler32_volume_oracle_matrix.rs"]
pub mod sweep_hash_adler32_volume_oracle_matrix;

#[path = "sweep_hash_blake3_g_volume_oracle_matrix.rs"]
pub mod sweep_hash_blake3_g_volume_oracle_matrix;

#[path = "sweep_hash_blake3_round_volume_oracle_matrix.rs"]
pub mod sweep_hash_blake3_round_volume_oracle_matrix;

#[path = "sweep_hash_crc32_reference_matrix.rs"]
pub mod sweep_hash_crc32_reference_matrix;

#[path = "sweep_hash_crc32_volume_oracle_matrix.rs"]
pub mod sweep_hash_crc32_volume_oracle_matrix;

#[path = "sweep_hash_crc_oracle_matrix.rs"]
pub mod sweep_hash_crc_oracle_matrix;

#[path = "sweep_hash_fnv1a_oracle_matrix.rs"]
pub mod sweep_hash_fnv1a_oracle_matrix;

#[path = "sweep_hash_multi_hash_volume_oracle_matrix.rs"]
pub mod sweep_hash_multi_hash_volume_oracle_matrix;

#[path = "sweep_hash_volume_oracle_matrix.rs"]
pub mod sweep_hash_volume_oracle_matrix;
