//! ProgramStats cache invariants  -  50 random programs verify every field.
//! Implementation lives in two `include!`-d chunks under `contract_cases/`.
#![allow(dead_code)]
include!("contract_cases/program_stats_proptest__extension_kind.rs");
include!("contract_cases/program_stats_proptest__arb_node.rs");
