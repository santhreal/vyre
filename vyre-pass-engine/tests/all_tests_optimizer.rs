//! One binary for every optimizer integration test in this crate.
//!
//! Cargo links one executable per integration-test target. Each file below ran
//! as its own target and now runs as a module of this one, which links one
//! binary for the whole set. A test that cannot share a process stays its own
//! target and states why in `xtask/test-harness-isolation.toml`.

/// Integration tests from `tests/cse_arena_kernel_index_bounds.rs`.
#[path = "cse_arena_kernel_index_bounds.rs"]
pub mod cse_arena_kernel_index_bounds;

/// Integration tests from `tests/cross_scope_cse_still_fires.rs`.
#[path = "cross_scope_cse_still_fires.rs"]
pub mod cross_scope_cse_still_fires;

/// Integration tests from `tests/dce_dispatch_binding_contract.rs`.
#[path = "dce_dispatch_binding_contract.rs"]
pub mod dce_dispatch_binding_contract;

/// Integration tests from `tests/dce_program_back_edge_contract.rs`.
#[path = "dce_program_back_edge_contract.rs"]
pub mod dce_program_back_edge_contract;

/// Integration tests from `tests/encoded_rewrite_walk_contract.rs`.
#[path = "encoded_rewrite_walk_contract.rs"]
pub mod encoded_rewrite_walk_contract;
