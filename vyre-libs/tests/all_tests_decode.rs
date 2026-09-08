//! One binary for every decode integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/inflate_program.rs`.
#[path = "inflate_program.rs"]
pub mod inflate_program;

/// Integration tests from `tests/sweep_decode_base64_volume_oracle_matrix.rs`.
#[cfg(feature = "decode")]
#[path = "sweep_decode_base64_volume_oracle_matrix.rs"]
pub mod sweep_decode_base64_volume_oracle_matrix;

/// Integration tests from `tests/sweep_decode_hex_oracle_matrix.rs`.
#[cfg(feature = "decode")]
#[path = "sweep_decode_hex_oracle_matrix.rs"]
pub mod sweep_decode_hex_oracle_matrix;

/// Integration tests from `tests/sweep_decode_hex_primitives_volume_oracle_matrix.rs`.
#[cfg(feature = "decode")]
#[path = "sweep_decode_hex_primitives_volume_oracle_matrix.rs"]
pub mod sweep_decode_hex_primitives_volume_oracle_matrix;
