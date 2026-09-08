//! One binary for every device integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.
//!
//! Run this target with `--test-threads=1`. Every case in it opens a vendor
//! device context, and the vendor userspace driver spins inside its own
//! thread-local destructor when several test threads in one process each build
//! and tear one down.

/// Shared fixture module from `tests/float_lowering/mod.rs`.
#[path = "float_lowering/mod.rs"]
pub mod float_lowering;

/// Integration tests from `tests/float_lowering_device_decisions.rs`.
#[path = "float_lowering_device_decisions.rs"]
pub mod float_lowering_device_decisions;
