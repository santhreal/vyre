//! Megakernel core contract tests  -  assert that the megakernel scheduler
//! and dispatch pipeline preserve key invariants across edge cases.
//!
//! Implementation lives in two chunks under `contract_cases/`:
//! `megakernel_core_contracts__decode_packed_slot_words.rs` and its child
//! `megakernel_core_contracts__read_metrics_returns_nonzero_only.rs`.

#[path = "contract_cases/megakernel_core_contracts__decode_packed_slot_words.rs"]
mod megakernel_core_contracts_decode_packed_slot_words;
