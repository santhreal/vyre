//! One binary for every reasoning integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/bounded_compile_policy.rs`.
#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

/// Integration tests from `tests/categorical_laws_proptest.rs`.
#[path = "categorical_laws_proptest.rs"]
pub mod categorical_laws_proptest;

/// Integration tests from `tests/do_calculus_surgery_via_reference_parity.rs`.
#[path = "do_calculus_surgery_via_reference_parity.rs"]
pub mod do_calculus_surgery_via_reference_parity;

/// Integration tests from `tests/functor_apply_via_reference_parity.rs`.
#[path = "functor_apply_via_reference_parity.rs"]
pub mod functor_apply_via_reference_parity;

/// Integration tests from `tests/predict_impact_via_reference_parity.rs`.
#[path = "predict_impact_via_reference_parity.rs"]
pub mod predict_impact_via_reference_parity;

/// Integration tests from `tests/string_diagram_via_reference_parity.rs`.
#[path = "string_diagram_via_reference_parity.rs"]
pub mod string_diagram_via_reference_parity;
