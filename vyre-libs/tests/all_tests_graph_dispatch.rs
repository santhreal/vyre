//! One binary for every graph-dispatch integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Shared fixture module from `tests/bounded_compile_policy.rs`.
#[path = "bounded_compile_policy.rs"]
pub mod bounded_compile_policy;

/// Shared fixture module from `tests/csr_sweep/mod.rs`.
#[path = "csr_sweep/mod.rs"]
pub mod csr_sweep;

/// Integration tests from `tests/graph_single_source_contracts.rs`.
#[path = "graph_single_source_contracts.rs"]
pub mod graph_single_source_contracts;

/// Integration tests from `tests/match_motif_via_reference_parity.rs`.
#[path = "match_motif_via_reference_parity.rs"]
pub mod match_motif_via_reference_parity;

/// Integration tests from `tests/reconstruct_path_via_reference_parity.rs`.
#[path = "reconstruct_path_via_reference_parity.rs"]
pub mod reconstruct_path_via_reference_parity;

/// Integration tests from `tests/sweep_graph_cpu_oracle_matrix.rs`.
#[path = "sweep_graph_cpu_oracle_matrix.rs"]
pub mod sweep_graph_cpu_oracle_matrix;

/// Integration tests from `tests/sweep_graph_persistent_bfs_volume_oracle_matrix.rs`.
#[cfg(feature = "graph-dispatch")]
#[path = "sweep_graph_persistent_bfs_volume_oracle_matrix.rs"]
pub mod sweep_graph_persistent_bfs_volume_oracle_matrix;

/// Integration tests from `tests/union_find_alias_via_reference_parity.rs`.
#[path = "union_find_alias_via_reference_parity.rs"]
pub mod union_find_alias_via_reference_parity;
