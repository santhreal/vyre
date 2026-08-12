//! Cross-backend parity matrix: registered backends, wire shapes, and buffer comparison.
//!
//! Implementation lives in two `include!`-d chunks under `contract_cases/`.
#![forbid(unsafe_code)]

include!("contract_cases/parity_matrix__program.rs");
include!("contract_cases/parity_matrix__synthetic_entries.rs");
