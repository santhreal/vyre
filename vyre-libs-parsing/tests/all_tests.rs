//! One binary for every integration test in this crate.

#[path = "harness/mod.rs"]
pub mod harness;

#[path = "wire_words/mod.rs"]
pub mod wire_words;

#[path = "ast_shunting_yard.rs"]
pub mod ast_shunting_yard;

#[path = "line_splice_classify_roundtrip.rs"]
pub mod line_splice_classify_roundtrip;

#[path = "lr_tables_contracts.rs"]
pub mod lr_tables_contracts;

#[path = "parsing_walker_clone_family.rs"]
pub mod parsing_walker_clone_family;

#[path = "planar_rewrite_ir_parity_proptest.rs"]
pub mod planar_rewrite_ir_parity_proptest;

#[path = "proptest_dispatch_pack_roundtrip.rs"]
pub mod proptest_dispatch_pack_roundtrip;

#[path = "ssa_dominance_phi_overflow_parity.rs"]
pub mod ssa_dominance_phi_overflow_parity;
