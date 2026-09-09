//! One binary for every integration test in this crate.

#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

#[path = "categorical_laws_proptest.rs"]
pub mod categorical_laws_proptest;

#[path = "do_calculus_surgery_via_reference_parity.rs"]
pub mod do_calculus_surgery_via_reference_parity;

#[path = "functor_apply_via_reference_parity.rs"]
pub mod functor_apply_via_reference_parity;

#[path = "predict_impact_via_reference_parity.rs"]
pub mod predict_impact_via_reference_parity;

#[path = "string_diagram_via_reference_parity.rs"]
pub mod string_diagram_via_reference_parity;

