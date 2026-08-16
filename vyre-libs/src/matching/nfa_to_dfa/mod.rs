//! NFA → DFA subset construction.
//!
//! Lowers a state-major NFA bit-table (the shape `compile_regex_set` /
//! `nfa_scan_with_plan` emit) into the dense `state * 256 + byte → next_state`
//! [`CompiledDfa`] the [`crate::matching::dfa_compile`] family also produces.
//!
//! # Why this lives here
//!
//! Two GPU scan kernels exist in vyre-libs today:
//!
//! * `classic_ac_bounded_ranges_program` - consumes [`CompiledDfa`], does ONE
//!   transition-table load per input byte (`transitions[state * 256 + byte]`).
//!   O(1) per byte regardless of state count.
//! * `nfa_scan_with_plan` - consumes the state-major NFA bit-table, walks a
//!   bit-vector state with ~LANES² subgroup_shuffle steps per byte. Necessary
//!   when an NFA cannot be subset-constructed under budget (state explosion),
//!   but expensive per byte.
//!
//! Regex sets (`compile_regex_set`) emit the second shape. For pattern sets
//! whose subset construction stays under a reasonable state cap, lowering to
//! the dense DFA lets the regex run through the dense kernel instead - same
//! throughput as a literal AC scan. This primitive is the bridge.
//!
//! # Algorithm
//!
//! Textbook subset construction. A DFA state is the set of NFA states the
//! automaton could be in. Start = ε-closure({entry NFA state}). For each DFA
//! state D, for each byte b: collect all NFA targets of (s, b) for s ∈ D,
//! take ε-closure, deduplicate against existing DFA states. Termination is
//! bounded by the caller-supplied state cap.
//!
//! Accept metadata: a DFA state accepts if any NFA state in its set is an
//! accept. `output_records[state]` enumerates every pattern_id whose accept
//! state is in the set, preserving multi-match semantics.

mod dedup;
mod error;
mod state_set;
mod subset;

#[cfg(test)]
mod tests;

pub use dedup::{
    dfa_fingerprint, dfa_wire_bytes, DfaDedupBatch, DfaDedupResult, DfaDedupStats, DfaDedupTable,
};
pub use error::NfaToDfaError;
pub use subset::{nfa_to_dfa, NfaTables};

/// Canonical op id.
pub(crate) const OP_ID: &str = "vyre-libs::matching::nfa_to_dfa";
