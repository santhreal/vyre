//! One binary for every integration test in this crate.

#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

#[path = "fusion_scores_via_reference_parity.rs"]
pub mod fusion_scores_via_reference_parity;

#[path = "planar_rewrite_via_reference_parity.rs"]
pub mod planar_rewrite_via_reference_parity;

#[path = "shape_spectrum_via_reference_parity.rs"]
pub mod shape_spectrum_via_reference_parity;

#[path = "submodular_retention_via_reference_parity.rs"]
pub mod submodular_retention_via_reference_parity;
