//! One binary for every integration test in this crate.

#[macro_use]
#[path = "gate_fixtures/mod.rs"]
pub mod gate_fixtures;

#[path = "adversarial_reduce_gather.rs"]
pub mod adversarial_reduce_gather;

#[path = "adversarial_reduce_histogram.rs"]
pub mod adversarial_reduce_histogram;

#[path = "adversarial_reduce_radix_sort.rs"]
pub mod adversarial_reduce_radix_sort;

#[path = "adversarial_reduce_scatter.rs"]
pub mod adversarial_reduce_scatter;

#[path = "adversarial_reduce_segment_reduce.rs"]
pub mod adversarial_reduce_segment_reduce;


#[path = "grid_stride_tree_buffer_contract.rs"]
pub mod grid_stride_tree_buffer_contract;

#[path = "grid_stride_tree_sum_covers_every_element.rs"]
pub mod grid_stride_tree_sum_covers_every_element;

#[path = "histogram_atomic_scatter_parity.rs"]
pub mod histogram_atomic_scatter_parity;

#[path = "indexed_move_gather_oob_parity.rs"]
pub mod indexed_move_gather_oob_parity;

#[path = "launch_seam_registry_closure.rs"]
pub mod launch_seam_registry_closure;

#[path = "multi_block_prefix_scan_carry_parity.rs"]
pub mod multi_block_prefix_scan_carry_parity;

#[path = "range_counts_ir_parity_proptest.rs"]
pub mod range_counts_ir_parity_proptest;

#[path = "reduce_atomic_ir_parity_proptest.rs"]
pub mod reduce_atomic_ir_parity_proptest;

#[path = "reduction_metrics_via_reference_parity.rs"]
pub mod reduction_metrics_via_reference_parity;

#[path = "reduction_route_parity.rs"]
pub mod reduction_route_parity;

#[path = "segment_reduce_ir_parity_proptest.rs"]
pub mod segment_reduce_ir_parity_proptest;

#[path = "sweep_radix_sort_oracle_matrix.rs"]
pub mod sweep_radix_sort_oracle_matrix;

#[path = "sweep_reduce_oracle_matrix.rs"]
pub mod sweep_reduce_oracle_matrix;

#[path = "sweep_segment_reduce_oracle_matrix.rs"]
pub mod sweep_segment_reduce_oracle_matrix;

#[path = "workgroup_any_ir_parity_proptest.rs"]
pub mod workgroup_any_ir_parity_proptest;
