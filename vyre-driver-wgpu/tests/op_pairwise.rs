//! Pairwise op-composition proptest. Implementation lives in two
//! `include!`-d chunks under `contract_cases/`.
#![allow(deprecated)]
include!("contract_cases/op_pairwise__all_entries_vec.rs");
include!("contract_cases/op_pairwise__entry_cases.rs");
