//! One binary for every solvers integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/bounded_compile_policy.rs`.
#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

/// Integration tests from `tests/bellman_shortest_path_via_reference_parity.rs`.
#[path = "bellman_shortest_path_via_reference_parity.rs"]
pub mod bellman_shortest_path_via_reference_parity;

/// Integration tests from `tests/fmm_compress_pairwise_via_reference_parity.rs`.
#[path = "fmm_compress_pairwise_via_reference_parity.rs"]
pub mod fmm_compress_pairwise_via_reference_parity;

/// Integration tests from `tests/fmm_polyhedral_via_reference_parity.rs`.
#[path = "fmm_polyhedral_via_reference_parity.rs"]
pub mod fmm_polyhedral_via_reference_parity;

/// Integration tests from `tests/kfac_via_reference_parity.rs`.
#[path = "kfac_via_reference_parity.rs"]
pub mod kfac_via_reference_parity;

/// Integration tests from `tests/multigrid_matroid_via_reference_parity.rs`.
#[path = "multigrid_matroid_via_reference_parity.rs"]
pub mod multigrid_matroid_via_reference_parity;

/// Integration tests from `tests/mz_project_via_reference_parity.rs`.
#[path = "mz_project_via_reference_parity.rs"]
pub mod mz_project_via_reference_parity;

/// Integration tests from `tests/natural_config_gradient_via_reference_parity.rs`.
#[path = "natural_config_gradient_via_reference_parity.rs"]
pub mod natural_config_gradient_via_reference_parity;

/// Integration tests from `tests/natural_gradient_via_reference_parity.rs`.
#[path = "natural_gradient_via_reference_parity.rs"]
pub mod natural_gradient_via_reference_parity;

/// Integration tests from `tests/quantized_dispatch_variant_coverage_gate.rs`.
#[path = "quantized_dispatch_variant_coverage_gate.rs"]
pub mod quantized_dispatch_variant_coverage_gate;

/// Integration tests from `tests/quantized_via_reference_parity.rs`.
#[path = "quantized_via_reference_parity.rs"]
pub mod quantized_via_reference_parity;

/// Integration tests from `tests/sheaf_heterophilic_via_reference_parity.rs`.
#[path = "sheaf_heterophilic_via_reference_parity.rs"]
pub mod sheaf_heterophilic_via_reference_parity;

/// Integration tests from `tests/sheaf_spectrum_via_reference_parity.rs`.
#[path = "sheaf_spectrum_via_reference_parity.rs"]
pub mod sheaf_spectrum_via_reference_parity;

/// Integration tests from `tests/sinkhorn_via_reference_parity.rs`.
#[path = "sinkhorn_via_reference_parity.rs"]
pub mod sinkhorn_via_reference_parity;

/// Integration tests from `tests/smooth_latency_trace_via_reference_parity.rs`.
#[path = "smooth_latency_trace_via_reference_parity.rs"]
pub mod smooth_latency_trace_via_reference_parity;

/// Integration tests from `tests/smooth_matroid_flow_via_reference_parity.rs`.
#[path = "smooth_matroid_flow_via_reference_parity.rs"]
pub mod smooth_matroid_flow_via_reference_parity;

/// Integration tests from `tests/softmax_pick_config_via_reference_parity.rs`.
#[path = "softmax_pick_config_via_reference_parity.rs"]
pub mod softmax_pick_config_via_reference_parity;

/// Integration tests from `tests/solvers_dispatch_softmax_contract.rs`.
#[path = "solvers_dispatch_softmax_contract.rs"]
pub mod solvers_dispatch_softmax_contract;

/// Integration tests from `tests/tensor_train_chain_fusion_via_reference_parity.rs`.
#[path = "tensor_train_chain_fusion_via_reference_parity.rs"]
pub mod tensor_train_chain_fusion_via_reference_parity;

/// Integration tests from `tests/tensor_train_compress_via_reference_parity.rs`.
#[path = "tensor_train_compress_via_reference_parity.rs"]
pub mod tensor_train_compress_via_reference_parity;

/// Integration tests from `tests/transport_residual_via_reference_parity.rs`.
#[path = "transport_residual_via_reference_parity.rs"]
pub mod transport_residual_via_reference_parity;

/// Integration tests from `tests/vietoris_rips_via_reference_parity.rs`.
#[path = "vietoris_rips_via_reference_parity.rs"]
pub mod vietoris_rips_via_reference_parity;
