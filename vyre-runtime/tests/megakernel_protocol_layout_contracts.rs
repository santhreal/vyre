//! Megakernel protocol layout contracts  -  exact byte/word placement
//! and non-overlap. Implementation lives in two chunks under
//! `contract_cases/`: `megakernel_protocol_layout_contracts__write_word.rs`
//! and its child
//! `megakernel_protocol_layout_contracts__slot_word_layout_args_start_at_word_4.rs`.
#![allow(clippy::assertions_on_constants)]

#[path = "contract_cases/megakernel_protocol_layout_contracts__write_word.rs"]
mod megakernel_protocol_layout_contracts_write_word;
