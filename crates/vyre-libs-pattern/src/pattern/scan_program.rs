//! Neutral NFA scan composition artifact.
//!
//! This module owns program and immutable table construction only. Dispatch,
//! resident allocation, readback, and timing adapters live above `vyre-libs`.

use vyre_foundation::ir::Program;

use super::nfa;

/// Typed program plus immutable inputs required by the NFA scan composition.
#[derive(Debug, Clone)]
pub struct ScanProgram {
    /// Substrate-neutral scan program.
    pub program: Program,
    /// Lane-major transition table consumed by `nfa_transition`.
    pub transition_table: Vec<u32>,
    /// Lane-major epsilon table consumed by `nfa_epsilon`.
    pub epsilon_table: Vec<u32>,
    /// Typed NFA plan describing state and input bounds.
    pub plan: nfa::NfaPlan,
}

/// Build a neutral NFA program artifact and its immutable table inputs.
#[must_use]
pub fn build(patterns: &[&str], input_buf: &str, hit_buf: &str, input_len: u32) -> ScanProgram {
    let plan = nfa::compile(patterns).for_input_len(input_len);
    let program = nfa::nfa_scan(patterns, input_buf, hit_buf, input_len);
    let transition_table = nfa::build_transition_table(patterns);
    let epsilon_table = nfa::build_epsilon_table(patterns);
    ScanProgram {
        program,
        transition_table,
        epsilon_table,
        plan,
    }
}
