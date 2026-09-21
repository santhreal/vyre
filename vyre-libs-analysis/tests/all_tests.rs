//! One binary for every integration test in this crate.

#[path = "cost_model_predict_runtime_via_reference_parity.rs"]
pub mod cost_model_predict_runtime_via_reference_parity;

#[path = "primitive_vs_consumer.rs"]
pub mod primitive_vs_consumer;

#[path = "semiring_gemm_via_reference_parity.rs"]
pub mod semiring_gemm_via_reference_parity;
