//! Generated wrapper test crate for c parser pipeline macro boundary contracts.
//!
//! Implementation lives in `__split/` chunks.
#![cfg(feature = "c-parser")]
#![allow(deprecated)]

use vyre_primitives::wire::{
    decode_u32_le_bytes_all as decode_u32_words, pack_u32_slice as u32_bytes,
};

include!("__split/c_parser_pipeline_macro_boundary_contracts_chunk1.rs");
include!("__split/c_parser_pipeline_macro_boundary_contracts_chunk2.rs");
