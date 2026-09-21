//! Pattern-set compilation into the transition table.

use super::{builder::dfa_compile_inner_capped, CompiledDfa, DfaCompileError};

/// Default transition-table budget: 16 MiB.
///
/// Covers roughly 16k states x 256 transitions x 4 bytes/word. Most
/// real pattern sets stay well under this; callers that need more can
/// use [`dfa_compile_with_budget`].
pub const DEFAULT_DFA_BUDGET_BYTES: usize = 16 * 1024 * 1024;

/// Compile a list of byte patterns into a CPU-built DFA under the
/// default [`DEFAULT_DFA_BUDGET_BYTES`] budget.
///
/// # Panics
///
/// Panics when the transition table would exceed the default budget. Returning
/// an empty DFA in that case would silently drop EVERY match (the empty
/// automaton rejects all input), an invisible recall loss in any scanner built
/// on it. The pattern set is operator-supplied (a rule catalog, never attacker
/// haystack), so an oversized set is a configuration error that must fail
/// closed and loud. Callers that need to handle oversized sets programmatically
/// must use [`dfa_compile_with_budget`] and shard oversized pattern sets,
/// capturing the structured [`DfaCompileError`] instead of panicking.
#[must_use]
pub fn dfa_compile(patterns: &[&[u8]]) -> CompiledDfa {
    match dfa_compile_with_budget(patterns, DEFAULT_DFA_BUDGET_BYTES) {
        Ok(dfa) => dfa,
        Err(error) => panic!(
            "dfa_compile: compiling {} pattern(s) exceeded the default {DEFAULT_DFA_BUDGET_BYTES}-byte DFA budget ({error}). \
             Returning the empty rejecting automaton would silently drop every match; \
             use dfa_compile_with_budget and shard oversized pattern sets to handle this as a structured error.",
            patterns.len()
        ),
    }
}

/// Compile a list of byte patterns with an explicit transition-table
/// budget. Use this when the caller wants to handle oversized DFAs
/// programmatically instead of panicking.
///
/// # Errors
///
/// Returns [`DfaCompileError::TooLarge`] when the DFA would exceed
/// `budget_bytes`. The error carries the requested size and the
/// budget for diagnostic messages.
pub fn dfa_compile_with_budget(
    patterns: &[&[u8]],
    budget_bytes: usize,
) -> Result<CompiledDfa, DfaCompileError> {
    dfa_compile_with_budget_ci(patterns, budget_bytes, false)
}

/// ASCII-CASE-INSENSITIVE counterpart of [`dfa_compile`]: `A`/`a` … `Z`/`z` are
/// matched interchangeably. The case fold is baked into the TRANSITION TABLE, not
/// the haystack, patterns are canonicalized to lowercase at trie construction,
/// and the flattened transition for a raw byte `b` resolves through
/// `fold(b)`, so `transitions[state][b'A'] == transitions[state][b'a']`. A scanner
/// therefore matches mixed-case input with ZERO per-byte folding work and no
/// second resident haystack copy (it kills the consumer-side `to_ascii_lowercase`
/// pass entirely). Non-ASCII and non-letter bytes are unchanged.
///
/// NOTE for downstream matchers: a case-insensitive DFA also needs its candidate
/// PREFILTER masks (end-byte / suffix2 / suffix3) folded to admit both cases of
/// each pattern byte, the masks are checked against the RAW haystack byte, which
/// this DFA does not fold. Build those masks with the case-insensitive variant.
///
/// # Panics
/// See [`dfa_compile`].
#[must_use]
pub fn dfa_compile_case_insensitive(patterns: &[&[u8]]) -> CompiledDfa {
    match dfa_compile_case_insensitive_with_budget(patterns, DEFAULT_DFA_BUDGET_BYTES) {
        Ok(dfa) => dfa,
        Err(error) => panic!(
            "dfa_compile_case_insensitive: compiling {} pattern(s) exceeded the default {DEFAULT_DFA_BUDGET_BYTES}-byte DFA budget ({error}). \
             Returning the empty rejecting automaton would silently drop every match; \
             use dfa_compile_case_insensitive_with_budget and shard oversized pattern sets to handle this as a structured error.",
            patterns.len()
        ),
    }
}

/// ASCII-case-insensitive counterpart of [`dfa_compile_with_budget`].
///
/// # Errors
/// See [`dfa_compile_with_budget`].
pub fn dfa_compile_case_insensitive_with_budget(
    patterns: &[&[u8]],
    budget_bytes: usize,
) -> Result<CompiledDfa, DfaCompileError> {
    dfa_compile_with_budget_ci(patterns, budget_bytes, true)
}

fn dfa_compile_with_budget_ci(
    patterns: &[&[u8]],
    budget_bytes: usize,
    case_insensitive: bool,
) -> Result<CompiledDfa, DfaCompileError> {
    let state_cap = budget_bytes / (256 * core::mem::size_of::<u32>());
    let inner = dfa_compile_inner_capped(patterns, state_cap, case_insensitive)?;
    let requested_bytes = (inner.state_count as usize)
        .saturating_mul(256)
        .saturating_mul(core::mem::size_of::<u32>());
    if requested_bytes > budget_bytes {
        return Err(DfaCompileError::TooLarge {
            requested_bytes,
            budget_bytes,
            state_count: inner.state_count,
        });
    }
    Ok(inner)
}
