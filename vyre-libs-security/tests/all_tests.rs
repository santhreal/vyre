//! One binary for every integration test in this crate.

#[path = "arg_of_slot_precision.rs"]
pub mod arg_of_slot_precision;

#[path = "node_kind_eq_ir_parity_proptest.rs"]
pub mod node_kind_eq_ir_parity_proptest;

#[path = "resolve_family_ir_parity_proptest.rs"]
pub mod resolve_family_ir_parity_proptest;

#[path = "security_flow_skeleton_family_guard.rs"]
pub mod security_flow_skeleton_family_guard;

#[path = "sweep_predicate_node_kind_oracle_matrix.rs"]
pub mod sweep_predicate_node_kind_oracle_matrix;
