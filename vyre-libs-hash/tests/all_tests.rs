//! One binary for every integration test in this crate.

#[path = "hash_oracles/mod.rs"]
pub mod hash_oracles;

#[path = "fnv1a64_builder_parity.rs"]
pub mod fnv1a64_builder_parity;

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
