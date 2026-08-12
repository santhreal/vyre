//! Megakernel protocol layout contracts  -  exact byte/word placement
//! and non-overlap. Implementation lives in two `include!`-d chunks
//! under `contract_cases/`.
#![allow(clippy::assertions_on_constants)]

include!("contract_cases/megakernel_protocol_layout_contracts__write_word.rs");
include!(
    "contract_cases/megakernel_protocol_layout_contracts__slot_word_layout_args_start_at_word_4.rs"
);
