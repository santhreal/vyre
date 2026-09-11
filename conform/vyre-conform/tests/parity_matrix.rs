//! Cross-backend parity matrix: registered backends, wire shapes, and buffer comparison.
//!
//! Implementation lives in two chunks under `contract_cases/`:
//! `parity_matrix__program.rs` and its child `parity_matrix__synthetic_entries.rs`.
#![forbid(unsafe_code)]

#[path = "contract_cases/parity_matrix__program.rs"]
mod parity_matrix_program;
