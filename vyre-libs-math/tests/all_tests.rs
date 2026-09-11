//! One binary for every integration test in this crate.

#[path = "scan_oracle/mod.rs"]
pub mod scan_oracle;

#[path = "float_sample/mod.rs"]
pub mod float_sample;

#[path = "signed_coverage/mod.rs"]
pub mod signed_coverage;

#[path = "adversarial.rs"]
pub mod adversarial;

#[path = "adversarial_math.rs"]
pub mod adversarial_math;

#[allow(deprecated)]
#[path = "algebra_lattice_semiring_contracts.rs"]
pub mod algebra_lattice_semiring_contracts;

#[path = "argmax_of_marginals_ir_parity_proptest.rs"]
pub mod argmax_of_marginals_ir_parity_proptest;

#[path = "bellman_oob_edge_parity.rs"]
pub mod bellman_oob_edge_parity;

#[path = "bigint_add_carry_ir_parity_proptest.rs"]
pub mod bigint_add_carry_ir_parity_proptest;

#[path = "clifford_geometric_product_program_parity.rs"]
pub mod clifford_geometric_product_program_parity;

#[path = "dp_clip_signed_newton_parity.rs"]
pub mod dp_clip_signed_newton_parity;

#[path = "fmm_program_parity.rs"]
pub mod fmm_program_parity;

#[path = "homotopy_euler_signed_parity.rs"]
pub mod homotopy_euler_signed_parity;

#[path = "iht_threshold_ir_parity_proptest.rs"]
pub mod iht_threshold_ir_parity_proptest;

#[path = "jacobi_serial_body_matches_per_lane.rs"]
pub mod jacobi_serial_body_matches_per_lane;

#[path = "jacobi_workgroup_cooperative_contracts.rs"]
pub mod jacobi_workgroup_cooperative_contracts;

#[path = "kfac_block_inverse_proptest.rs"]
pub mod kfac_block_inverse_proptest;

#[path = "padic_hensel_signed_parity.rs"]
pub mod padic_hensel_signed_parity;

#[path = "prefix_scan_contract.rs"]
pub mod prefix_scan_contract;

#[path = "quantized_packing_contracts.rs"]
pub mod quantized_packing_contracts;

#[path = "randomized_svd_signed_parity.rs"]
pub mod randomized_svd_signed_parity;

#[path = "scallop_join_ir_parity.rs"]
pub mod scallop_join_ir_parity;

#[path = "scan_prefix_sum_size_contract.rs"]
pub mod scan_prefix_sum_size_contract;

#[path = "score_denoise_signed_parity.rs"]
pub mod score_denoise_signed_parity;

#[path = "semiring_gemm_wide_parity.rs"]
pub mod semiring_gemm_wide_parity;

#[path = "sheaf_laplacian_eigenvalue_dispatch_parity.rs"]
pub mod sheaf_laplacian_eigenvalue_dispatch_parity;

#[path = "sinkhorn_scale_ir_parity_proptest.rs"]
pub mod sinkhorn_scale_ir_parity_proptest;

#[path = "sos_gram_construct_proptest.rs"]
pub mod sos_gram_construct_proptest;

#[path = "sos_gram_oob_parity.rs"]
pub mod sos_gram_oob_parity;

#[path = "stream_compact_proptest.rs"]
pub mod stream_compact_proptest;

#[path = "sweep_math_prefix_scan_exclusive_volume_oracle_matrix.rs"]
pub mod sweep_math_prefix_scan_exclusive_volume_oracle_matrix;

#[path = "sweep_math_prefix_scan_inclusive_volume_oracle_matrix.rs"]
pub mod sweep_math_prefix_scan_inclusive_volume_oracle_matrix;

#[path = "symmetric_eigen_jacobi_parity.rs"]
pub mod symmetric_eigen_jacobi_parity;

#[path = "symmetric_eigen_jacobi_registration.rs"]
pub mod symmetric_eigen_jacobi_registration;

#[path = "tensor_scc_value_parity.rs"]
pub mod tensor_scc_value_parity;

#[path = "tensor_train_contract_signed_parity.rs"]
pub mod tensor_train_contract_signed_parity;

#[path = "tensor_train_decompose_eigen_contract.rs"]
pub mod tensor_train_decompose_eigen_contract;

#[path = "tensor_train_decompose_step_parity.rs"]
pub mod tensor_train_decompose_step_parity;

#[path = "tfn_scalar_mix_signed_parity.rs"]
pub mod tfn_scalar_mix_signed_parity;

#[path = "tiled_matmul_composition.rs"]
pub mod tiled_matmul_composition;
