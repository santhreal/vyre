//! One binary for every integration test in this crate.

#[path = "bitset_dense_matvec_pipeline_generated.rs"]
pub mod bitset_dense_matvec_pipeline_generated;

#[path = "bitset_mask_algebra_via_reference_parity.rs"]
pub mod bitset_mask_algebra_via_reference_parity;

#[path = "bitset_summary_via_reference_parity.rs"]
pub mod bitset_summary_via_reference_parity;

#[path = "dense_matvec_cases/mod.rs"]
pub mod dense_matvec_cases;

#[path = "matching_diagnostic_via_reference_parity.rs"]
pub mod matching_diagnostic_via_reference_parity;

#[path = "matroid_exact_subset_via_reference_parity.rs"]
pub mod matroid_exact_subset_via_reference_parity;

#[path = "provenance_closure.rs"]
pub mod provenance_closure;

#[path = "scallop_provenance_via_reference_parity.rs"]
pub mod scallop_provenance_via_reference_parity;

#[path = "self_consumer_conform.rs"]
pub mod self_consumer_conform;

#[path = "vsa_fingerprint_via_reference_parity.rs"]
pub mod vsa_fingerprint_via_reference_parity;
