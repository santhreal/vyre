//! Precomputed LR(1) action/goto tables for GPU parser pipelines.
//!
//! **Build-time migration note:** These tables are currently hardcoded as
//! `&'static [u32]` slices derived from a manual SLR(1) construction of the
//! C expression grammar. When the grammar grows beyond expressions, move
//! table generation into a `build.rs` script that emits literal arrays, then
//! keep this module as the stable runtime API.
//!
//! `vyre_spec::c11_expr_token` owns the terminal ids these tables index by. A
//! caller that builds a token stream names that module.

mod action;
mod c11_expr;
mod table;

pub use action::Action;
pub use c11_expr::{ACTION_TABLE, C11_EXPR, GOTO_TABLE, NT_E, NT_F, NT_T, PRODUCTIONS};
pub use table::{LrTables, Production};
