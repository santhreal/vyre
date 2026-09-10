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

#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

#[path = "reduction_metrics_via_reference_parity.rs"]
pub mod reduction_metrics_via_reference_parity;

#[path = "sweep_radix_sort_oracle_matrix.rs"]
pub mod sweep_radix_sort_oracle_matrix;

#[path = "sweep_reduce_oracle_matrix.rs"]
pub mod sweep_reduce_oracle_matrix;

#[path = "sweep_segment_reduce_oracle_matrix.rs"]
pub mod sweep_segment_reduce_oracle_matrix;
