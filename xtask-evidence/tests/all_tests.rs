//! One binary for every default-feature integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/release_evidence_dispatch.rs`.
#[path = "release_evidence_dispatch.rs"]
pub mod release_evidence_dispatch;

/// Integration tests from `tests/source_fingerprint_producers.rs`.
#[path = "source_fingerprint_producers.rs"]
pub mod source_fingerprint_producers;
