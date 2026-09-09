//! One binary for every integration test in this crate.

#[path = "ast_shunting_yard.rs"]
pub mod ast_shunting_yard;

#[path = "harness/mod.rs"]
pub mod harness;

#[path = "line_splice_classify_roundtrip.rs"]
pub mod line_splice_classify_roundtrip;

#[path = "lr_tables_contracts.rs"]
pub mod lr_tables_contracts;

#[path = "parsing_walker_clone_family.rs"]
pub mod parsing_walker_clone_family;

#[path = "wire_words/mod.rs"]
pub mod wire_words;
