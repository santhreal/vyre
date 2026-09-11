//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/consumer_coupling.rs`.
#[path = "consumer_coupling.rs"]
pub mod consumer_coupling;

/// Integration tests from `tests/gpu_skip_guards.rs`.
#[path = "gpu_skip_guards.rs"]
pub mod gpu_skip_guards;

/// Integration tests from `tests/module_forks.rs`.
#[path = "module_forks.rs"]
pub mod module_forks;

/// Integration tests from `tests/production_cpu_fallbacks.rs`.
#[path = "production_cpu_fallbacks.rs"]
pub mod production_cpu_fallbacks;

/// Integration tests from `tests/raw_ir_in_libs.rs`.
#[path = "raw_ir_in_libs.rs"]
pub mod raw_ir_in_libs;
