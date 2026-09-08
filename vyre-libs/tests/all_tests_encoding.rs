//! One binary for every encoding integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/bounded_compile_policy.rs`.
#[allow(missing_docs)]
#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

/// Shared fixture module from `tests/dense_matvec_cases/mod.rs`.
#[path = "dense_matvec_cases/mod.rs"]
pub mod dense_matvec_cases;

/// Integration tests from `tests/bitset_dense_matvec_pipeline_generated.rs`.
#[path = "bitset_dense_matvec_pipeline_generated.rs"]
pub mod bitset_dense_matvec_pipeline_generated;

/// Integration tests from `tests/bitset_mask_algebra_via_reference_parity.rs`.
#[path = "bitset_mask_algebra_via_reference_parity.rs"]
pub mod bitset_mask_algebra_via_reference_parity;

/// Integration tests from `tests/bitset_summary_via_reference_parity.rs`.
#[path = "bitset_summary_via_reference_parity.rs"]
pub mod bitset_summary_via_reference_parity;

/// Integration tests from `tests/matching_diagnostic_via_reference_parity.rs`.
#[path = "matching_diagnostic_via_reference_parity.rs"]
pub mod matching_diagnostic_via_reference_parity;

/// Integration tests from `tests/matroid_exact_subset_via_reference_parity.rs`.
#[path = "matroid_exact_subset_via_reference_parity.rs"]
pub mod matroid_exact_subset_via_reference_parity;

/// Integration tests from `tests/provenance_closure.rs`.
#[allow(missing_docs)]
#[path = "provenance_closure.rs"]
pub mod provenance_closure;

/// Integration tests from `tests/scallop_provenance_via_reference_parity.rs`.
#[path = "scallop_provenance_via_reference_parity.rs"]
pub mod scallop_provenance_via_reference_parity;

/// Integration tests from `tests/self_consumer_conform.rs`.
#[allow(missing_docs)]
#[path = "self_consumer_conform.rs"]
pub mod self_consumer_conform;

/// Integration tests from `tests/vsa_fingerprint_via_reference_parity.rs`.
#[path = "vsa_fingerprint_via_reference_parity.rs"]
pub mod vsa_fingerprint_via_reference_parity;
