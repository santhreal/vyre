//! One binary for every hash integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[cfg(feature = "hash")]
#[allow(deprecated)]
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/fnv1a64_builder_parity.rs`.
#[cfg(feature = "hash")]
#[path = "fnv1a64_builder_parity.rs"]
pub mod fnv1a64_builder_parity;

/// Integration tests from `tests/hash_registration_witnesses.rs`.
#[cfg(feature = "hash")]
#[path = "hash_registration_witnesses.rs"]
pub mod hash_registration_witnesses;

/// Integration tests from `tests/sweep_hash_adler32_volume_oracle_matrix.rs`.
#[cfg(feature = "hash")]
#[path = "sweep_hash_adler32_volume_oracle_matrix.rs"]
pub mod sweep_hash_adler32_volume_oracle_matrix;

/// Integration tests from `tests/sweep_hash_blake3_g_volume_oracle_matrix.rs`.
#[cfg(feature = "hash")]
#[path = "sweep_hash_blake3_g_volume_oracle_matrix.rs"]
pub mod sweep_hash_blake3_g_volume_oracle_matrix;

/// Integration tests from `tests/sweep_hash_blake3_round_volume_oracle_matrix.rs`.
#[cfg(feature = "hash")]
#[path = "sweep_hash_blake3_round_volume_oracle_matrix.rs"]
pub mod sweep_hash_blake3_round_volume_oracle_matrix;

/// Integration tests from `tests/sweep_hash_crc32_reference_matrix.rs`.
#[cfg(feature = "hash")]
#[allow(deprecated)]
#[path = "sweep_hash_crc32_reference_matrix.rs"]
pub mod sweep_hash_crc32_reference_matrix;

/// Integration tests from `tests/sweep_hash_crc32_volume_oracle_matrix.rs`.
#[cfg(feature = "hash")]
#[path = "sweep_hash_crc32_volume_oracle_matrix.rs"]
pub mod sweep_hash_crc32_volume_oracle_matrix;

/// Integration tests from `tests/sweep_hash_crc_oracle_matrix.rs`.
#[cfg(feature = "hash")]
#[path = "sweep_hash_crc_oracle_matrix.rs"]
pub mod sweep_hash_crc_oracle_matrix;

/// Integration tests from `tests/sweep_hash_fnv1a_oracle_matrix.rs`.
#[path = "sweep_hash_fnv1a_oracle_matrix.rs"]
pub mod sweep_hash_fnv1a_oracle_matrix;

/// Integration tests from `tests/sweep_hash_multi_hash_volume_oracle_matrix.rs`.
#[cfg(feature = "hash")]
#[path = "sweep_hash_multi_hash_volume_oracle_matrix.rs"]
pub mod sweep_hash_multi_hash_volume_oracle_matrix;

/// Integration tests from `tests/sweep_hash_volume_oracle_matrix.rs`.
#[cfg(feature = "hash")]
#[path = "sweep_hash_volume_oracle_matrix.rs"]
pub mod sweep_hash_volume_oracle_matrix;
