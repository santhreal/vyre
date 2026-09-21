//! NFA transition and epsilon table packing.

use super::{alloc::reserve_vec, try_compile, NfaCompileError};
use crate::nfa::subgroup_nfa::LANES_PER_SUBGROUP;

/// Build the `nfa_transition` state-major bit-table matching the
/// [`subgroup_nfa::nfa_step`] contract:
/// `[num_states × 256 × LANES_PER_SUBGROUP]` u32s. Entry
/// `trans[src * 256 * LANES + byte * LANES + dst_lane]` is the
/// destination bitset held by `dst_lane` when state `src` sees `byte`.
/// The source state is the outermost index, so every lane's word for
/// one `(src, byte)` pair is adjacent and one subgroup load coalesces.
///
/// [`subgroup_nfa::nfa_step`]: crate::nfa::subgroup_nfa::nfa_step
///
/// # Panics
///
/// Aborts when the plan or table allocation cannot be represented safely. This
/// is LOUD on purpose (Law 10): returning an empty `Vec` here would build an
/// NFA with no transitions, a scanner that silently matches NOTHING, reporting
/// every input as clean. Callers that must recover use
/// [`try_build_transition_table`].
#[must_use]
pub fn build_transition_table(patterns: &[&str]) -> Vec<u32> {
    match try_build_transition_table(patterns) {
        Ok(table) => table,
        Err(error) => {
            // An empty table builds an NFA with no transitions, a scanner that
            // silently matches NOTHING, reporting every input as clean. That is
            // a total recall-loss silent fallback (Law 10). Fail closed.
            panic!(
                "vyre-libs NFA transition-table build failed: {error}. \
                 returning an empty table would build a scanner that silently matches nothing; \
                 use try_build_transition_table and reduce the pattern set below the state cap."
            )
        }
    }
}

/// Fallible counterpart of [`build_transition_table`].
///
/// # Errors
///
/// Returns [`NfaCompileError`] when the plan or table allocation cannot be
/// represented safely.
pub fn try_build_transition_table(patterns: &[&str]) -> Result<Vec<u32>, NfaCompileError> {
    let plan = try_compile(patterns)?;
    let num_states = plan.num_states as usize;
    let table_words = table_word_count(num_states, 256, "transition")?;
    let mut table = zeroed_u32_table(table_words, "transition table word")?;
    let mut state_cursor: usize = 1;
    for p in patterns {
        let mut src = 0_usize;
        for b in p.bytes() {
            let dst = state_cursor;
            let dst_lane = dst / 32;
            let dst_bit = 1_u32 << (dst % 32);
            let idx = src * 256 * LANES_PER_SUBGROUP + (b as usize) * LANES_PER_SUBGROUP + dst_lane;
            table[idx] |= dst_bit;
            src = dst;
            state_cursor += 1;
        }
    }
    Ok(table)
}

/// Build the `nfa_epsilon` state-major table
/// `[num_states × LANES_PER_SUBGROUP]`. Literal-only → all zero.
///
/// # Panics
///
/// Aborts when the plan or table allocation cannot be represented safely
/// loudly, never an empty `Vec` (Law 10). Callers that must recover use
/// [`try_build_epsilon_table`].
#[must_use]
pub fn build_epsilon_table(patterns: &[&str]) -> Vec<u32> {
    match try_build_epsilon_table(patterns) {
        Ok(table) => table,
        Err(error) => {
            // An empty epsilon table drops the ε-closures that wire each
            // pattern's start state (a silent recall fallback (Law 10). Fail closed).
            panic!(
                "vyre-libs NFA epsilon-table build failed: {error}. \
                 returning an empty table would silently drop pattern ε-wiring; \
                 use try_build_epsilon_table and reduce the pattern set below the state cap."
            )
        }
    }
}

/// Fallible counterpart of [`build_epsilon_table`].
///
/// # Errors
///
/// Returns [`NfaCompileError`] when the plan or table allocation cannot be
/// represented safely.
pub fn try_build_epsilon_table(patterns: &[&str]) -> Result<Vec<u32>, NfaCompileError> {
    let n = try_compile(patterns)?.num_states as usize;
    let table_words = n
        .checked_mul(LANES_PER_SUBGROUP)
        .ok_or(NfaCompileError::TableWordCountOverflow { table: "epsilon" })?;
    zeroed_u32_table(table_words, "epsilon table word")
}

fn table_word_count(
    states: usize,
    byte_columns: usize,
    table: &'static str,
) -> Result<usize, NfaCompileError> {
    states
        .checked_mul(byte_columns)
        .and_then(|words| words.checked_mul(LANES_PER_SUBGROUP))
        .ok_or(NfaCompileError::TableWordCountOverflow { table })
}

fn zeroed_u32_table(words: usize, field: &'static str) -> Result<Vec<u32>, NfaCompileError> {
    let mut table = Vec::new();
    reserve_vec(&mut table, words, field)?;
    table.resize(words, 0);
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::{
        build_epsilon_table, build_transition_table, try_build_epsilon_table,
        try_build_transition_table,
    };
    use crate::nfa::subgroup_nfa::LANES_PER_SUBGROUP;

    #[test]
    fn compatibility_table_builders_match_fallible_builders_for_empty_plan() {
        assert_eq!(
            build_transition_table(&[]),
            try_build_transition_table(&[]).expect("Fix: empty transition table must reserve")
        );
        assert_eq!(
            build_epsilon_table(&[]),
            try_build_epsilon_table(&[]).expect("Fix: empty epsilon table must reserve")
        );
    }

    #[test]
    fn empty_transition_tables_preserve_gpu_lane_shape() {
        assert_eq!(build_transition_table(&[]).len(), 256 * LANES_PER_SUBGROUP);
        assert_eq!(build_epsilon_table(&[]).len(), LANES_PER_SUBGROUP);
    }
}
