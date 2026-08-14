//! ProgramStats cache invariants  -  50 random programs verify every field.
//! Implementation lives in two chunks under `contract_cases/`:
//! `program_stats_proptest__extension_kind.rs` and its child
//! `program_stats_proptest__arb_node.rs`.
#![allow(dead_code)]

#[path = "contract_cases/program_stats_proptest__extension_kind.rs"]
mod program_stats_proptest_extension_kind;
