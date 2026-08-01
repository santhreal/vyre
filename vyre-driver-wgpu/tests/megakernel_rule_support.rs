//! Shared deterministic rule fixtures for megakernel integration tests.

use vyre_runtime::megakernel::BatchRuleProgram;

/// Build a two-state unanchored DFA that accepts every occurrence of `byte`.
pub(crate) fn byte_finder_rule(rule_idx: u32, byte: u8) -> BatchRuleProgram {
    let mut transitions = vec![0u32; 2 * 256];
    transitions[byte as usize] = 1;
    transitions[256 + byte as usize] = 1;
    BatchRuleProgram::new(rule_idx, transitions, vec![0u32, 1u32], 2)
        .expect("valid two-state byte finder DFA")
}
