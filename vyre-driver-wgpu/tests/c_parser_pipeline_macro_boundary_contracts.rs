//! Contract tests for c parser pipeline macro boundary contracts.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod c_ast_gpu_parity_support;

use vyre_primitives::wire::{
    decode_u32_le_bytes_all as decode_u32_words, pack_u32_slice as u32_bytes,
};

#[path = "c_parser_pipeline_macro_boundary_contracts/conditional_mask_is_deterministic_across_many_runs.rs"]
mod conditional_mask_is_deterministic_across_many_runs;
#[path = "c_parser_pipeline_macro_boundary_contracts/hash_token.rs"]
mod hash_token;
