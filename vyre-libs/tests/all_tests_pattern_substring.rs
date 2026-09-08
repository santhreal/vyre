//! One binary for every pattern-substring integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/presence_oracle/mod.rs`.
#[cfg(feature = "pattern-substring")]
#[path = "presence_oracle/mod.rs"]
pub mod presence_oracle;

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[cfg(feature = "pattern-substring")]
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/literal_set_presence_and_positions_reference.rs`.
#[cfg(feature = "pattern-substring")]
#[path = "literal_set_presence_and_positions_reference.rs"]
pub mod literal_set_presence_and_positions_reference;

/// Integration tests from `tests/literal_set_presence_reference.rs`.
#[cfg(feature = "pattern-substring")]
#[path = "literal_set_presence_reference.rs"]
pub mod literal_set_presence_reference;

/// Integration tests from `tests/scan_hit_buffer_layout_contracts.rs`.
#[path = "scan_hit_buffer_layout_contracts.rs"]
pub mod scan_hit_buffer_layout_contracts;
