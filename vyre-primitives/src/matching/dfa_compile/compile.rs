//! Pattern-set compilation into the transition table.

use super::{CompiledDfa, DfaCompileError};

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

/// Canonicalize an ASCII byte for a case-insensitive DFA: `A`..=`Z` map to their
/// lowercase; every other byte (including non-ASCII) is unchanged. Identity when
/// `case_insensitive` is false. One owner for the fold so the insert path and the
/// transition-flatten path cannot disagree on the byte class.
#[inline]
fn fold_ascii_byte(b: usize, case_insensitive: bool) -> usize {
    if case_insensitive && (0x41..=0x5A).contains(&b) {
        b | 0x20
    } else {
        b
    }
}

/// Compile a DFA with an explicit state cap.
///
/// # Panics
/// Panics when `pattern_idx` exceeds `u32::MAX - 1`, which the `pid + 1` wire encoding
/// cannot represent. The caller bounds the pattern count first.
fn dfa_compile_inner_capped(
    patterns: &[&[u8]],
    state_cap: usize,
    case_insensitive: bool,
) -> Result<CompiledDfa, DfaCompileError> {
    const NO_TRANSITION: u32 = u32::MAX;

    let upper_bound = patterns
        .iter()
        .fold(0usize, |acc, p| acc.saturating_add(p.len()))
        .saturating_add(1);
    let max_pattern_len = patterns
        .iter()
        .map(|pattern| pattern.len())
        .max()
        .unwrap_or(0)
        .min(u32::MAX as usize) as u32;
    let trie_capacity = state_cap.min(upper_bound).max(1);

    let mut trie: Vec<[u32; 256]> = Vec::with_capacity(trie_capacity);
    let mut accept: Vec<u32> = Vec::with_capacity(trie_capacity);
    let mut local_accepts: Vec<Vec<u32>> = Vec::with_capacity(trie_capacity);

    trie.push([NO_TRANSITION; 256]);
    accept.push(0);
    local_accepts.push(Vec::new());

    for (pattern_idx, pat) in patterns.iter().enumerate() {
        let mut cur = 0usize;
        for &b in *pat {
            // Case-insensitive: fold pattern bytes to lowercase so the trie is
            // built over the canonical alphabet; uppercase input is redirected
            // onto the same path in the transition-flatten step below.
            let b = fold_ascii_byte(b as usize, case_insensitive);
            let next = trie[cur][b];
            if next != NO_TRANSITION {
                cur = next as usize;
            } else {
                if trie.len() >= state_cap {
                    return Err(DfaCompileError::TrieStateCapExceeded { state_cap });
                }
                let new_id = trie.len() as u32;
                trie.push([NO_TRANSITION; 256]);
                accept.push(0);
                local_accepts.push(Vec::new());
                trie[cur][b] = new_id;
                cur = new_id as usize;
            }
        }
        local_accepts[cur].push(pattern_idx as u32);
        // The accept fast-path field stores the FIRST (lowest) pattern id that reaches
        // a given trie node, encoded as pid+1. Using the first-inserted pattern preserves
        // the stable, predictable semantics documented at CompiledDfa.accept: the
        // lowest pattern id is canonical. If we overwrote on each iteration, the last
        // pattern would win, silently misreporting earlier patterns on the fast path
        // (output_records is unaffected and always carries all pids).
        if accept[cur] == 0 {
            accept[cur] = (pattern_idx as u32)
                .checked_add(1)
                .expect("pattern_idx must be <= u32::MAX - 1 to fit the pid+1 encoding");
        }
    }

    let state_count = trie.len();
    let mut fail = vec![0u32; state_count];
    let mut queue = Vec::new();
    for b in 0..256usize {
        let child = trie[0][b];
        if child != NO_TRANSITION {
            fail[child as usize] = 0;
            queue.push(child as usize);
        }
    }
    let mut head = 0usize;
    while head < queue.len() {
        let state = queue[head];
        head += 1;
        for b in 0..256usize {
            let child = trie[state][b];
            if child != NO_TRANSITION {
                let mut f = fail[state] as usize;
                while f != 0 && trie[f][b] == NO_TRANSITION {
                    f = fail[f] as usize;
                }
                let f_child = trie[f][b];
                if f_child != NO_TRANSITION && f_child != child {
                    fail[child as usize] = f_child;
                }
                if accept[child as usize] == 0 {
                    let f_accept = accept[fail[child as usize] as usize];
                    if f_accept != 0 {
                        accept[child as usize] = f_accept;
                    }
                }
                queue.push(child as usize);
            }
        }
    }

    let mut bfs_order = Vec::with_capacity(state_count);
    let mut bfs_queue = Vec::with_capacity(state_count);
    bfs_queue.push(0usize);
    let mut bfs_head = 0usize;
    while bfs_head < bfs_queue.len() {
        let state = bfs_queue[bfs_head];
        bfs_head += 1;
        bfs_order.push(state);

        for b in 0..256usize {
            let child = trie[state][b];
            if child != NO_TRANSITION {
                bfs_queue.push(child as usize);
            }
        }
    }

    let mut output_counts = vec![0usize; state_count];
    for &state in &bfs_order {
        let f = fail[state] as usize;
        let inherited = if f != 0 && f != state {
            output_counts[f]
        } else {
            0
        };
        let adds_local = local_accepts[state]
            .iter()
            .filter(|&&pattern| !fail_chain_accepts_pattern(state, pattern, &fail, &local_accepts))
            .count();
        output_counts[state] = inherited + adds_local;
    }

    let mut output_offsets = vec![0u32; state_count + 1];
    for state in 0..state_count {
        output_offsets[state + 1] =
            output_offsets[state].saturating_add(output_counts[state] as u32);
    }
    let mut output_records = vec![0u32; output_offsets[state_count] as usize];
    for &state in &bfs_order {
        let mut write = output_offsets[state] as usize;
        let f = fail[state] as usize;
        if f != 0 && f != state {
            let start = output_offsets[f] as usize;
            let end = output_offsets[f + 1] as usize;
            let len = end - start;
            output_records.copy_within(start..end, write);
            write += len;
        }
        for &pattern in &local_accepts[state] {
            let start = output_offsets[state] as usize;
            if !output_records[start..write].contains(&pattern) {
                output_records[write] = pattern;
                write += 1;
            }
        }
        debug_assert_eq!(write, output_offsets[state + 1] as usize);
    }

    let mut transitions = vec![0u32; state_count * 256];
    let mut accept_out = vec![0u32; state_count];
    for state in 0..state_count {
        accept_out[state] = accept[state];
        for b in 0..256usize {
            // Resolve the goto for the FOLDED byte class and store it under the
            // raw byte column `b`, so a case-insensitive DFA answers `b'A'` with
            // the same next state as `b'a'` (identity when case-sensitive). The
            // trie only carries folded edges, so the fail-chain walk uses the
            // folded index throughout.
            let fb = fold_ascii_byte(b, case_insensitive);
            let mut s = state;
            loop {
                let child = trie[s][fb];
                if child != NO_TRANSITION {
                    transitions[state * 256 + b] = child;
                    break;
                }
                if s == 0 {
                    transitions[state * 256 + b] = 0;
                    break;
                }
                s = fail[s] as usize;
            }
        }
    }

    Ok(CompiledDfa {
        transitions,
        accept: accept_out,
        state_count: state_count as u32,
        max_pattern_len,
        output_offsets,
        output_records,
    })
}

fn fail_chain_accepts_pattern(
    state: usize,
    pattern: u32,
    fail: &[u32],
    local_accepts: &[Vec<u32>],
) -> bool {
    let mut f = fail[state] as usize;
    while f != 0 && f != state {
        if local_accepts[f].contains(&pattern) {
            return true;
        }
        let next = fail[f] as usize;
        if next == f {
            return false;
        }
        f = next;
    }
    false
}
