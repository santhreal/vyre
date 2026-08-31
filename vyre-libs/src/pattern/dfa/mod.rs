//! DFA / Aho-Corasick sub-dialect: pre-built transition tables + scanner.
pub mod aho_corasick;
pub(crate) mod cooperative_dfa;

pub use aho_corasick::aho_corasick;
pub use cooperative_dfa::{cooperative_dfa_scan, cooperative_dfa_scan_body_with_store};
