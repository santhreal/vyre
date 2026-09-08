//! One binary for every parsing integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/harness/mod.rs`.
#[cfg(feature = "parsing")]
#[allow(deprecated)]
#[path = "harness/mod.rs"]
pub mod harness;

/// Shared fixture module from `tests/wire_words/mod.rs`.
#[cfg(feature = "parsing")]
#[allow(deprecated)]
#[path = "wire_words/mod.rs"]
pub mod wire_words;

/// Integration tests from `tests/ast_shunting_yard.rs`.
#[cfg(feature = "parsing")]
#[allow(deprecated)]
#[path = "ast_shunting_yard.rs"]
pub mod ast_shunting_yard;

/// Integration tests from `tests/lr_tables_contracts.rs`.
#[path = "lr_tables_contracts.rs"]
pub mod lr_tables_contracts;

/// Integration tests from `tests/parsing_walker_clone_family.rs`.
#[cfg(feature = "parsing")]
#[path = "parsing_walker_clone_family.rs"]
pub mod parsing_walker_clone_family;
