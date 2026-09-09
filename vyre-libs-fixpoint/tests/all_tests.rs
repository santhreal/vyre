//! One binary for every integration test in this crate.

#[path = "adversarial_graph_reachability_fixpoint/mod.rs"]
pub mod adversarial_graph_reachability_fixpoint;

#[path = "loop_back_edge_audit.rs"]
pub mod loop_back_edge_audit;

#[path = "loop_unroll_trip1_idempotence.rs"]
pub mod loop_unroll_trip1_idempotence;

#[path = "persistent_fixpoint_grid_contracts/mod.rs"]
pub mod persistent_fixpoint_grid_contracts;

#[path = "scallop_join_grid_contract.rs"]
pub mod scallop_join_grid_contract;

#[path = "scallop_join_ir_parity.rs"]
pub mod scallop_join_ir_parity;

#[path = "wire_words/mod.rs"]
pub mod wire_words;

