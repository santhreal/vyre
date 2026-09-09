//! One binary for every integration test in this crate.

#[path = "adversarial_math.rs"]
pub mod adversarial_math;

#[path = "name_collision.rs"]
pub mod name_collision;

#[path = "operation_registry.rs"]
pub mod operation_registry;

#[path = "prefix_scan_contract.rs"]
pub mod prefix_scan_contract;

#[path = "quantized_packing_contracts.rs"]
pub mod quantized_packing_contracts;

#[path = "scan_prefix_sum_size_contract.rs"]
pub mod scan_prefix_sum_size_contract;

#[path = "sweep_math_prefix_scan_exclusive_volume_oracle_matrix.rs"]
pub mod sweep_math_prefix_scan_exclusive_volume_oracle_matrix;

#[path = "sweep_math_prefix_scan_inclusive_volume_oracle_matrix.rs"]
pub mod sweep_math_prefix_scan_inclusive_volume_oracle_matrix;

#[path = "wire_words/mod.rs"]
pub mod wire_words;

