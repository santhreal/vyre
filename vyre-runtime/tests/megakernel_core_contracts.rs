//! Megakernel core contract tests  -  assert that the megakernel scheduler
//! and dispatch pipeline preserve key invariants across edge cases.
//!
//! Implementation lives in two `include!`-d chunks under `contract_cases/`.

include!("contract_cases/megakernel_core_contracts__decode_packed_slot_words.rs");
include!("contract_cases/megakernel_core_contracts__read_metrics_returns_nonzero_only.rs");
