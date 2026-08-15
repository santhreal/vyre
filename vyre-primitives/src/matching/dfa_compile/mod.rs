//! CPU-side DFA compiler for multi-pattern scanning.
//!
//! `dfa_compile` produces a transition table for Aho-Corasick-style
//! byte scanning. The table is pure data (`Vec<u32>`) so downstream
//! crates can upload it to a GPU buffer or consume it from CPU tests
//! without depending on the higher-level matching dialect.
//!
//! The table layout is deliberately simple so kernels can step the
//! DFA in one load per byte:
//!
//! ```text
//! transitions[state * 256 + byte] = next_state
//! accept   [state]                  = nonzero if `state` matches a pattern
//! ```
//!
//! Patterns are compiled with failure links collapsed, so scanners
//! never have to walk failure pointers while processing input.

use std::{error::Error, fmt};

mod compile;
#[cfg(test)]
mod tests;
mod wire;

pub use compile::{
    dfa_compile, dfa_compile_case_insensitive, dfa_compile_case_insensitive_with_budget,
    dfa_compile_with_budget, DEFAULT_DFA_BUDGET_BYTES,
};
pub use wire::DfaWireError;

/// Compiled DFA ready to be uploaded to a GPU buffer.
#[derive(Debug, Clone)]
pub struct CompiledDfa {
    /// `transitions[state * 256 + byte] = next_state`. Length =
    /// `state_count * 256`.
    pub transitions: Vec<u32>,
    /// `accept[state] = pattern_id + 1` when `state` accepts, else 0.
    /// Length = `state_count`.
    pub accept: Vec<u32>,
    /// Number of states in the automaton (>= 1; state 0 is root).
    pub state_count: u32,
    /// Longest pattern length in bytes. Scanners can limit each
    /// per-position replay to this suffix window without changing
    /// Aho-Corasick semantics.
    pub max_pattern_len: u32,
    /// `output_offsets[state]` = start index in `output_records` for
    /// `state`. Length = `state_count + 1`. The last element is the
    /// total length of `output_records`.
    pub output_offsets: Vec<u32>,
    /// Flat array of pattern ids. Each state `s` owns the slice
    /// `output_records[output_offsets[s]..output_offsets[s+1]]`.
    /// These are all patterns that match at `s` (including via
    /// failure links), not just the single `accept[state]` id.
    pub output_records: Vec<u32>,
}

/// Structured failure from [`dfa_compile_with_budget`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DfaCompileError {
    /// Built DFA would exceed the caller's transition-table budget.
    TooLarge {
        /// Number of bytes the naive table would require.
        requested_bytes: usize,
        /// Caller-supplied budget.
        budget_bytes: usize,
        /// State count at the point of budget exhaustion.
        state_count: u32,
    },
    /// Trie grew past the permitted state cap during construction.
    TrieStateCapExceeded {
        /// State cap derived from the caller-supplied budget.
        state_cap: usize,
    },
}

impl fmt::Display for DfaCompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge {
                requested_bytes,
                budget_bytes,
                ..
            } => write!(
                formatter,
                "DFA transition table is too large: {requested_bytes} bytes (cap = {budget_bytes}). Fix: reduce the pattern set, raise the budget, or shard patterns into multiple DFAs."
            ),
            Self::TrieStateCapExceeded { state_cap } => write!(
                formatter,
                "DFA trie exceeded state cap during construction: requested > {state_cap} states. Fix: reduce the pattern set or raise the budget (cap derived from budget_bytes / 1024)."
            ),
        }
    }
}

impl Error for DfaCompileError {}

impl CompiledDfa {
    /// Empty DFA with a single rejecting root state.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            transitions: vec![0; 256],
            accept: vec![0],
            state_count: 1,
            max_pattern_len: 0,
            output_offsets: vec![0, 0],
            output_records: Vec::new(),
        }
    }
}
