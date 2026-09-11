//! One binary for every integration test in this crate.

#[path = "persistent_fixpoint_grid_contracts/mod.rs"]
pub mod persistent_fixpoint_grid_contracts;

#[path = "bitset_fixpoint_warm_start_parity.rs"]
pub mod bitset_fixpoint_warm_start_parity;

#[path = "persistent_fixpoint_loop_contracts.rs"]
pub mod persistent_fixpoint_loop_contracts;
