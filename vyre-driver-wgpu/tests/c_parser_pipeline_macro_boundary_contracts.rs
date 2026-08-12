//! Generated wrapper test crate for c parser pipeline macro boundary contracts.
//!
//! Implementation lives in `contract_cases/` chunks.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]

use vyre_primitives::wire::{
    decode_u32_le_bytes_all as decode_u32_words, pack_u32_slice as u32_bytes,
};

include!("contract_cases/c_parser_pipeline_macro_boundary_contracts__hash_token.rs");
include!("contract_cases/c_parser_pipeline_macro_boundary_contracts__conditional_mask_is_deterministic_across_many_runs.rs");
