//! DFA / Aho-Corasick sub-dialect: pre-built transition tables + scanner.
mod aho_corasick;
mod cooperative_dfa;

pub use aho_corasick::{aho_corasick, aho_corasick_bounded, aho_corasick_program_from_dfa_wire};
pub use cooperative_dfa::{cooperative_dfa_scan, cooperative_dfa_scan_body_with_store};
